# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:anno_task_scmamba.py
# @Software:PyCharm
# @Created Time:2024/5/23 4:58 PM
import os,torch
import scanpy as sc
import pickle
import gc
import seaborn as sns
import pandas as pd
import numpy as np
from torch import nn
import copy
import warnings
from regformer.utils.utils import load_config
from pathlib import Path
import wandb,json
import time
import regformer as scmb
from regformer.utils.utils import seed_all,model_config,load_ckpt,define_wandb_metrcis
import matplotlib.pyplot as plt
from regformer.data.dataset import Load_Data,SeqDataset
from torch.utils.data import DataLoader
from regformer.data.dataloader import Get_DataLoader
from regformer.model.mambaLM import MambaModel
from regformer.data.gene_tokenizer import GeneVocab
import datetime
import shutil

class AnnoTaskScMamba(object):
    def __init__(self,config_file,device,pad_token="<pad>",unk_token='<unk>'): #初始化方法
        self.device =device  #设置设备(CPU/GPU)
        self.args=load_config(config_file)  #加载配置文件
        self.check_parameters() #检查参数有效性
        save_dir = os.path.join(self.args.save_dir, self.args.task, self.args.data_name, self.args.model_name,
                                self.args.run_name) #创建保存目录
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"save to {self.save_dir}")
        # save the whole script to the dir
        os.system(f"cp {__file__} {self.save_dir}")
        self.logger = scmb.logger  
        scmb.utils.add_file_handler(self.logger, self.save_dir / "run.log")  #设置日志记录
        seed_all(self.args.seed)
        #初始化特殊token和掩码值
        if self.args.input_emb_style == "category":
            self.mask_value = self.args.n_bins + 1
            self.pad_value = self.args.n_bins  # for padding gene expr values
            self.n_input_bins = self.args.n_bins + 2
        else:
            self.mask_value = -1
            self.pad_value = -2
            self.n_input_bins = self.args.n_bins
        self.pad_token,self.unk_token=pad_token,unk_token
        self.pad_token="<pad>"
        self.unk_token = '<unk>' if self.args.append_cls else '<cls>'

    def check_parameters(self):
        assert self.args.input_style in ["normed_raw", "log1p", "binned"]
        assert self.args.output_style in ["normed_raw", "log1p", "binned"]
        assert self.args.input_emb_style in ["category", "continuous", "scaling"]
        if self.args.input_style == "binned":
            if self.args.input_emb_style == "scaling":
                raise ValueError("input_emb_style `scaling` is not supported for binned input.")
        elif self.args.input_style == "log1p" or self.args.input_style == "normed_raw":
            if self.args.input_emb_style == "category":
                raise ValueError(
                    "input_emb_style `category` is not supported for log1p or normed_raw input."
                )

    def load_data_and_model(self):  #加载模型配置、词汇表和预训练模型
        #load config  参数设置
        model_configs,vocab_file,model_file=model_config(self.args)
        vocab = GeneVocab.from_file(vocab_file)
        self.mask_token = '<mask>' if '<mask>' in vocab.vocab.itos_ else '<eoc>'  #掩码标记

        special_tokens = [self.pad_token, self.mask_token, self.unk_token]  #特殊标记设置
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        vocab.set_default_index(vocab["<pad>"])
        shutil.copy(vocab_file, self.save_dir / "vocab.json")
        self.vocab=vocab

        #load data  加载训练、验证和测试数据
        train_data_pt, valid_data_pt, test_data_pt, data_configs = self.load_data(vocab,pad_token=self.pad_token,
                                                                               pad_value=self.pad_value,mask_value=self.mask_value)
        train_loader = Get_DataLoader(train_data_pt, args=self.args, shuffle=False, intra_domain_shuffle=True,drop_last=False)
        valid_loader = Get_DataLoader(valid_data_pt, args=self.args, shuffle=False,intra_domain_shuffle=False, drop_last=False)
        test_loader = DataLoader(dataset=SeqDataset(test_data_pt),batch_size=self.args.batch_size * 4,shuffle=False,
            drop_last=False,pin_memory=True,)

        #load model and ckpt 加载预训练模型 
        model = self.load_model(model_configs,vocab,data_configs)
        if self.args.load_model is not None:
            model=load_ckpt(model,model_file,self.args,self.logger)
        model=model.to(self.device)
        return model,train_loader,valid_loader,test_loader,data_configs

    def load_data(self,vocab,pad_token="<pad>",pad_value=-2,mask_value=-1):  #使用Load_Data函数加载预处理好的单细胞数据
        train_data_pt, valid_data_pt, test_data_pt, num_batch_types, celltypes, id2type, num_types, adata_test_raw = \
            Load_Data(data_path=self.args.data_path, args=self.args, logger=self.logger, vocab=vocab, is_master=True,
                      mask_value=mask_value, pad_value=pad_value, pad_token=pad_token)
        data_configs={'num_batch_types':num_batch_types,'celltypes':celltypes,'id2type':id2type,
                     'num_types':num_types,'adata_test_raw':adata_test_raw,'test_labels':test_data_pt['celltype_labels']}
        self.cls_count = torch.bincount(train_data_pt['celltype_labels'])
        return train_data_pt,valid_data_pt,test_data_pt,data_configs  #返回训练、验证、测试数据及配置信息

    def load_model(self,model_configs,vocab,data_configs):  #根据配置构建Mamba模型
        args=self.args
        ntokens = len(vocab)
        num_batch_types=data_configs['num_batch_types']  #模型参数包括：词汇表大小、嵌入维度、层数、细胞类型数量等
        model = MambaModel(ntoken=ntokens,d_model=model_configs['embsize'],nlayers=model_configs['nlayers'],nlayers_cls=3,n_cls=data_configs['num_types'] if args.CLS else 1,
            device=self.device,vocab=vocab,dropout=args.dropout,pad_token=self.pad_token,pad_value=self.pad_value,num_batch_labels=num_batch_types,
            n_input_bins=self.n_input_bins,input_emb_style=args.input_emb_style,cell_emb_style=args.cell_emb_style,
            pre_norm=args.pre_norm,do_pretrain=False,if_bimamba=args.bimamba_type != "none",
            bimamba_type=args.bimamba_type,if_devide_out=False,init_layer_scale=None)
        return model

    def set_wandb(self):  #wandb设置
        model_ckpt_name = self.args.load_model.split('/')[-1] if self.args.load_model != "none" else 'from_scratch'
        ## wandb setting
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        wandb_name = f'{self.args.task}_{self.args.data_name}_{self.args.model_name}_(ckpt-{model_ckpt_name})_{self.args.run_name}' \
                     f'{"_CCE" if self.args.CCE else ""}{"_CLS" if self.args.CLS else ""}' \
                     f'_{self.args.cell_emb_style}_{"wZero_" if self.args.include_zero_gene else ""}{now}'
        wandb_tags = ['Finetune', self.args.task, self.args.data_name,self.args.cell_emb_style,
                      "w Zero" if self.args.include_zero_gene else "w/o Zero",
                      "CLS" if self.args.CLS else "w/o CLS", "CCE" if self.args.CCE else "w/o CCE"]
        self.run = wandb.init(
            config=self.args.__dict__,
            job_type=self.args.task,
            project="scLLM-CA",
            name=wandb_name,
            tags=wandb_tags,
            reinit=True,
            settings=wandb.Settings(start_method="fork"),
        )
        print(self.args.__dict__)
    
    def evaluate(self,model,loader,epoch,return_raw=False):  #在验证集上评估模型，主要功能为计算损失和错误率
        model.eval()  #调整为评估模式
        total_loss = 0.0   #初始化各种属性
        total_error = 0.0
        total_num = 0
        predictions = []  #初始化预测列表
        with torch.no_grad():  #禁用梯度计算 evaluate与train的主要差别
            for batch_data in loader:
                input_gene_ids = batch_data["gene_ids"].to(self.device)
                input_values = batch_data["values"].to(self.device)
                target_values = batch_data["target_values"].to(self.device)
                batch_labels = batch_data["batch_labels"].to(self.device)
                celltype_labels = batch_data["celltype_labels"].to(self.device)
                if self.args.graph_sort and self.args.layer_emb:
                    sorted_layer_idx = batch_data['sorted_layer_idx'].to(self.device)
                else:
                    sorted_layer_idx = None

                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.pad_token])
                with torch.cuda.amp.autocast(enabled=self.args.amp): #下面计算使用混合精度训练
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        CLS=self.args.CLS,
                        CCE=self.args.CCE,
                        sorted_layer_idx=sorted_layer_idx
                    )
                    output_values = output_dict["cls_output"]
                    loss = self.criterion_cls(output_values, celltype_labels)
                total_loss += loss.item() * len(input_gene_ids)  #计算损失和错误率
                accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
                total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
                total_num += len(input_gene_ids)
                preds = output_values.argmax(1).cpu().numpy()
                predictions.append(preds)  #可返回原始预测结果
        mse = total_loss / total_num  
        mre = total_error / total_num
        wandb.log(
            {
                "valid/mse": mse,
                "valid/err": mre,
                "epoch": epoch,
            },
        )

        if return_raw:
            return np.concatenate(predictions, axis=0)

        return mse, mre

    def test(self,model,loader,celltypes_labels):  #在测试集上评估模型 使用到上面的evalute的函数
        predictions = self.evaluate(
            model,
            loader=loader,
            epoch='final',
            return_raw=True,
        )
        # compute accuracy, precision, recall, f1  计算准确率、精确率、召回率和F1分数
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(celltypes_labels, predictions)
        precision = precision_score(celltypes_labels, predictions, average="macro")
        recall = recall_score(celltypes_labels, predictions, average="macro")
        macro_f1 = f1_score(celltypes_labels, predictions, average="macro")
        self.logger.info(
            f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
            f"Macro F1: {macro_f1:.3f}")
        results = {
            "test/accuracy": accuracy,
            "test/precision": precision,
            "test/recall": recall,
            "test/macro_f1": macro_f1,
        }
        return predictions, celltypes_labels, results #返回预测值，label,计算结果

    def train(self,model,loader,epoch):  #训练循环
        model.train()  #转为训练模式
        total_loss,total_cce,total_cls,total_error=0.0, 0.0, 0.0, 0.0  #初始化值
        start_time = time.time()  #计算时间
        num_batches = len(loader)
        for batch, batch_data in enumerate(loader):   #处理输入数据
            input_gene_ids = batch_data["gene_ids"].to(self.device)
            input_values = batch_data["values"].to(self.device)
            target_values = batch_data["target_values"].to(self.device)
            batch_labels = batch_data["batch_labels"].to(self.device)
            celltype_labels = batch_data["celltype_labels"].to(self.device)
            if self.args.graph_sort and self.args.layer_emb:
                sorted_layer_idx = batch_data['sorted_layer_idx'].to(self.device)
            else:
                sorted_layer_idx = None
            src_key_padding_mask = input_gene_ids.eq(self.vocab[self.pad_token])  #掩码
            with torch.cuda.amp.autocast(enabled=self.args.amp):  #混合精度计算
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=self.args.CLS,
                    CCE=self.args.CCE,
                    sorted_layer_idx=sorted_layer_idx
                )
                masked_positions = input_values.eq(self.mask_value)  # the postions to predict
                loss = 0.0  #计算损失(分类损失和可选的其他损失)
                metrics_to_log = {}
                if self.args.CLS:  #细胞注释
                    loss_cls = self.criterion_cls(output_dict["cls_output"], celltype_labels)
                    loss = loss + loss_cls
                    metrics_to_log.update({"train/cls": loss_cls.item()})

                    error_rate = 1 - (
                        (output_dict["cls_output"].argmax(1) == celltype_labels)
                        .sum()
                        .item()
                    ) / celltype_labels.size(0)
                if self.args.CCE:  #对比学习
                    loss_cce = 10 * output_dict["loss_cce"]
                    loss = loss + loss_cce
                    metrics_to_log.update({"train/cce": loss_cce.item()})
                metrics_to_log.update({"train/loss": loss.item()})
                model.zero_grad()  #梯度清零
                self.scaler.scale(loss).backward()  # 对 loss 进行放大（scaling），然后进行反向传播。
                self.scaler.unscale_(self.optimizer)  #将放大的梯度还原回原始尺度。
                with warnings.catch_warnings(record=True) as w:
                    warnings.filterwarnings("always")
                    torch.nn.utils.clip_grad_norm_(  #梯度裁剪
                        model.parameters(),1.0,error_if_nonfinite=False if self.scaler.is_enabled() else True,)
                self.scaler.step(self.optimizer)  #更新参数（如果梯度正常）
                self.scaler.update()  #更新 scaler 的 scale 因子
                wandb.log(metrics_to_log)  #记录训练指标
                total_loss += loss.item()
                total_cls += loss_cls.item() if self.args.CLS else 0.0
                total_cce += loss_cce.item() if self.args.CCE else 0.0
                total_error += error_rate
                if batch % self.args.log_interval == 0 and batch > 0:  #定期输出当前的训练状态信息（如损失、学习率、每 batch 耗时等），方便监控训练过程和调试模型。
                    lr = self.scheduler.get_last_lr()[0]
                    ms_per_batch = (time.time() - start_time) * 1000 / self.args.log_interval
                    cur_loss = total_loss / self.args.log_interval
                    cur_cls = total_cls / self.args.log_interval if self.args.CLS else 0.0
                    cur_cce = total_cce / self.args.log_interval if self.args.CCE else 0.0
                    cur_error = total_error / self.args.log_interval
                    self.logger.info(
                        f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                        f"lr {lr:05.6f} | ms/batch {ms_per_batch:5.2f} | "
                        f"loss {cur_loss:5.2f} | "
                        + (f"cls {cur_cls:5.2f} | " if self.args.CLS else "")
                        + (f"err {cur_error:5.2f} | " if self.args.CLS else "")
                        + (f"cce {cur_cce:5.2f} |" if self.args.CCE else "")
                    )
                    total_loss, total_cce, total_cls, total_error = 0.0, 0.0, 0.0, 0.0
                    start_time = time.time()

    def load_criterion_and_opt(self,model):  #初始化损失函数（criterion）和优化器（optimizer）
        if self.args.cls_weight:  
            cls_weight = self.cls_count.sum() / self.cls_count.float()
            cls_weight = torch.where(torch.isinf(cls_weight), torch.tensor(0.0), cls_weight)
            cls_weight = cls_weight / cls_weight.sum()
            self.criterion_cls = nn.CrossEntropyLoss(weight=cls_weight.to(self.device))
        else:
            self.criterion_cls = nn.CrossEntropyLoss()  #分类损失（CrossEntropyLoss）
        self.optimizer = torch.optim.Adam(  #优化器（Adam）用于更新模型参数，最小化损失函数。
            model.parameters(), lr=self.args.lr, eps=1e-4 if self.args.amp else 1e-8
        )
        schedule_interval=max(1, int(self.args.epochs * 0.1))
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, schedule_interval, gamma=self.args.schedule_ratio
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.amp)



    def run_CA(self):  #主运行流程
        self.set_wandb()  #初始化wandb日志
        model,train_loader,valid_loader,test_loader,data_configs=self.load_data_and_model()  #加载数据和模型
        self.load_criterion_and_opt(model)
        wandb.watch(model)
        best_val_loss = float("inf")
        best_avg_bio = 0.0
        best_model = copy.deepcopy(model)
        define_wandb_metrcis()
        for epoch in range(1,self.args.epochs + 1):  #训练循环
            epoch_start_time = time.time()
            if self.args.do_train:
                self.train(model,loader=train_loader,epoch=epoch)
                val_loss,val_err=self.evaluate(model,loader=valid_loader,epoch=epoch)
                elapsed = time.time() - epoch_start_time
                self.logger.info("-" * 89)
                self.logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                    f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}")
                self.logger.info("-" * 89)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(model)
                    best_model_epoch = epoch
                    self.logger.info(f"Best model with score {best_val_loss:5.4f}")
                self.scheduler.step()
        predictions, labels, results=self.test(best_model,test_loader,data_configs['test_labels']) #评估和保存最佳模型
        self.result_organizing(predictions,labels,results,data_configs)

        torch.save(best_model.state_dict(), self.save_dir / "best_model.pt")
        # %%
        artifact = wandb.Artifact(f"best_model", type="model") #记录最终结果
        glob_str = os.path.join(self.save_dir, "best_model.pt")
        artifact.add_file(glob_str)
        self.run.log_artifact(artifact)

        self.run.finish()
        wandb.finish()
        gc.collect()

    def result_organizing(self, predictions, labels, results,data_configs):  #组织结果并可视化
        adata_test_raw=data_configs['adata_test_raw']
        id2type=data_configs['id2type']
        celltypes=data_configs['celltypes']

        adata_test_raw.obs["predictions"] = [id2type[p] for p in predictions]
        # plot
        palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"] + plt.rcParams["axes.prop_cycle"].by_key()[
            "color"] + \
                   plt.rcParams["axes.prop_cycle"].by_key()["color"]
        palette_ = {c: palette_[i] for i, c in enumerate(celltypes)}
        with plt.rc_context({"figure.figsize": (8, 4), "figure.dpi": (300)}): #生成UMAP图展示预测结果
            sc.pl.umap(
                adata_test_raw,
                color=["celltype", "predictions"],
                palette=palette_,
                show=False,
            )
            plt.savefig(self.save_dir / "results.png", dpi=300)

        save_dict = {
            "predictions": predictions,
            "labels": labels,
            "results": results,
            "id_maps": id2type
        }
        with open(self.save_dir / "results.pkl", "wb") as f:
            pickle.dump(save_dict, f)

        results["test/cell_umap"] = wandb.Image(
            str(self.save_dir / "results.png"),
            caption=f"predictions macro f1 {results['test/macro_f1']:.3f}",
        )
        from sklearn.metrics import confusion_matrix  #创建混淆矩阵
        celltypes = list(celltypes)
        for i in set([id2type[p] for p in predictions]):
            if i not in celltypes:
                celltypes.remove(i)
        cm = confusion_matrix(labels, predictions)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = pd.DataFrame(cm, index=celltypes[:cm.shape[0]], columns=celltypes[:cm.shape[1]])
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
        plt.savefig(self.save_dir / "confusion_matrix.png", dpi=300)

        results["test/confusion_matrix"] = wandb.Image( 
            str(self.save_dir / "confusion_matrix.png"),
            caption=f"confusion matrix",
        )
        wandb.log(results) #保存结果文件

if __name__ == "__main__":  #主程序入口判断 作为主程序运行（而不是被导入为模块）
    debug=False
    if debug:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["CUDA_VISIBLE_DEVICES"] = '2'
        local_rank=0
    else:
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ["RANK"])
    device = torch.device("cuda", local_rank)
    # config_file = sys.argv[1]
    config_file = r'../config/scmamba_CA.toml'
    task = AnnoTaskScMamba(config_file,device=device)
    task.run_CA()




