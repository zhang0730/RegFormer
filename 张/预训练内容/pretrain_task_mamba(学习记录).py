#主要功能
#1.预训练任务：实现了一个基于Mamba架构的单细胞RNA测序数据预训练模型
#2.多任务学习：同时处理基因表达值预测(MLM)、基因排序预测(拓扑排序)和基因表达值补全(MVC)
#3.分布式训练：支持多GPU分布式训练
#4.模型评估：包含训练和验证流程，支持模型保存和日志记录

# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:pretrain_task_mamba.py
# @Software:PyCharm
# @Created Time:2024/6/1 3:48 PM
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
import torch.distributed as dist
import regformer as scmb
from regformer.utils.utils import seed_all,model_config,load_ckpt,define_wandb_metrcis
import matplotlib.pyplot as plt
from regformer.data.dataset import Load_Data,SeqDataset
from torch.utils.data import DataLoader
from regformer.data.dataloader import Get_DataLoader
from regformer.model.mambaLM import MambaModel
from regformer.utils.utils import get_reduced
from regformer.data.gene_tokenizer import GeneVocab
from regformer.model.loss import masked_mse_loss,masked_relative_error
import datetime
import shutil
warnings.filterwarnings('ignore')


class PretrainTaskScMamba(object):
    def __init__(self,config_file,pad_token="<pad>",unk_token='<unk>'): # 初始化 (__init__)
        self.args=load_config(config_file) #加载配置文件
        if self.args.distributed: #设置分布式训练环境(如启用)
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            dist.init_process_group(backend='nccl')
            self.world_size = torch.distributed.get_world_size()
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            seed_all(self.args.seed + torch.distributed.get_rank())
        else:
            os.environ["WANDB_MODE"] = "offline"
            self.world_size=1
            self.rank=0
            self.local_rank=0
            self.device = torch.device("cuda", self.local_rank) #初始化设备(CUDA)
        self.is_master = self.rank == 0

        save_dir = os.path.join(self.args.save_dir, self.args.task, self.args.data_name, self.args.model_name,
                                self.args.run_name) # 创建保存目录

        if self.is_master:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"save to {self.save_dir}")
            # save the whole script to the dir
            os.system(f"cp {__file__} {self.save_dir}")
            self.logger = scmb.logger #创建日志系统
            scmb.utils.add_file_handler(self.logger, self.save_dir / "run.log")
        else:
            self.logger=None
        seed_all(self.args.seed)
        # 设置特殊token和掩码值
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

    def load_data_and_model(self): # 加载词汇表和模型配置
        #load config
        model_configs,vocab_file,model_file=model_config(self.args)
        vocab = GeneVocab.from_file(vocab_file) # 这个作者提供了
        self.mask_token = '<mask>' if '<mask>' in vocab.vocab.itos_ else '<eoc>'

        special_tokens = [self.pad_token, self.mask_token, self.unk_token]
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        vocab.set_default_index(vocab["<pad>"])
        if self.is_master:
            shutil.copy(vocab_file, self.save_dir / "vocab.json")
        self.vocab=vocab

        #load data 
        train_dataset,valid_dataset = self.load_data(vocab,pad_token=self.pad_token,
                                                     pad_value=self.pad_value,mask_value=self.mask_value) #初始化训练和验证数据集
        train_loader = Get_DataLoader(dataset=train_dataset,args=self.args,shuffle=False,
            drop_last=False,pin_memory=True) # 创建数据加载器
        valid_loader = Get_DataLoader(dataset=valid_dataset,args=self.args,shuffle=False,
            drop_last=False,pin_memory=True)
        if self.is_master:
            self.logger.info(
                f"Total set number of cells: {len(train_dataset) + len(valid_dataset)}."
                f"\n\t Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
            if not self.args.include_zero_gene:
                self.logger.info(f"\n\t Only using non-zero genes:"
                            f"\n\tUniform the length into max_seq_len: {self.args.max_seq_len}")
            else:
                self.logger.info(
                    f"\n\t Using all the genes, the length of whole gene, uniform the length into max_seq_len: {self.args.max_seq_len}")

        #load model and ckpt 初始化Mamba模型并加载预训练权重(如存在)
        model = self.load_model(model_configs,vocab)
        if self.args.load_model is not None:
            model=load_ckpt(model,model_file,self.args,self.logger)
        model=model.to(self.device)
        return model,train_loader,valid_loader
    
    def load_data(self,vocab,pad_token="<pad>",pad_value=-2,mask_value=-1):
        '''
        Loading LMDB dataset
        Args:
            vocab:
            pad_token:
            pad_value:
            mask_value:
        从指定路径加载LMDB格式的单细胞数据
        Returns:

        '''
        data_path=os.path.join(self.args.data_path, self.args.task, self.args.data_name)
        train_dataset, valid_dataset = Load_Data(data_path=data_path, args=self.args,
                                                 vocab=vocab, mask_ratio=self.args.mask_ratio, append_cls=self.args.append_cls,
                                                 include_zero_gene=self.args.include_zero_gene, max_seq_len=self.args.max_seq_len,
                                                 mask_token=self.mask_token, unk_token=self.unk_token)
        # data_configs={'num_batch_types':num_batch_types,'celltypes':celltypes,'id2type':id2type,
        #              'num_types':num_types,'adata_test_raw':adata_test_raw,'test_labels':test_data_pt['celltype_labels']}
        # self.cls_count = torch.bincount(train_data_pt['celltype_labels'])
        return train_dataset,valid_dataset # 返回训练和验证数据集
    
    def load_model(self,model_configs,vocab): #根据配置创建MambaModel实例
        args=self.args
        ntokens = len(vocab)
        model = MambaModel( #设置模型参数如层数、dropout、是否使用双向Mamba等
            ntoken=ntokens,d_model=model_configs['embsize'],nlayers=model_configs['nlayers'],
            vocab=vocab,dropout=args.dropout,pad_token=self.pad_token,pad_value=self.pad_value,do_mvc=args.MVC,
            do_dab=False,domain_spec_batchnorm=self.args.DSBN,
            n_input_bins=self.n_input_bins,input_emb_style=args.input_emb_style,
            cell_emb_style=args.cell_emb_style,mvc_decoder_style=args.mvc_decoder_style,pre_norm=args.pre_norm,
            do_pretrain=True,topo_graph=args.graph_sort,if_bimamba=args.bimamba_type != "none",bimamba_type=args.bimamba_type,
            if_devide_out=False,init_layer_scale=None,token_emb_freeze=args.token_emb_freeze)
        return model #返回模型实例

    def set_wandb(self): #初始化Weights & Biases日志记录
        ## wandb setting
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        wandb_name = f'{self.args.data_name}_{self.args.model_name}{str(self.args.bimamba_type) if self.args.bimamba_type != "none" else ""}_{"lye" if self.args.layer_emb else ""}_{"CLM" if self.args.generative_pretraining else "MLM"}_{self.args.run_name}_G{self.world_size}_{now}'
        wandb_tags = [self.args.task, f'G{self.world_size}',
                      self.args.data_name, self.args.model_name, 'gts' if self.args.graph_sort else 'wogts',
                      f'layer_mask{self.args.mask_ratio}' if self.args.layer_mask else f'random_mask{self.args.mask_ratio}',
                      'layer_positional_emb' if self.args.layer_emb else 'w/o lyemb']
        run = wandb.init(
            config=self.args.__dict__,
            job_type=self.args.task,
            project="scLLM-Pretrain",
            name=wandb_name,
            tags=wandb_tags,
            reinit=True,
            settings=wandb.Settings(start_method="fork"),
        )
        print(self.args)

    def load_criterion_and_opt(self,model):
        self.criterion = masked_mse_loss # 设置损失函数(掩蔽MSE损失)
        if self.args.graph_sort:
            self.lm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = torch.optim.Adam( #初始化优化器(Adam)
            model.parameters(), lr=self.args.lr, eps=1e-4 if self.args.amp else 1e-8
        )
        schedule_interval=max(1, int(self.args.epochs * 0.1)) # 确保至少每一轮(epoch)进行一次学习率更新，但更倾向于每隔总轮数的10%进行一次调整。
        self.scheduler = torch.optim.lr_scheduler.StepLR( # StepLR 作为学习率调度器 进行学习率衰减
            self.optimizer, schedule_interval, gamma=self.args.schedule_ratio
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.amp) # 自动混合精度训练

    def train(self,model,loader,epoch): # 训练循环
        model.train() # 初始化一切
        total_loss, total_mse, total_MVC, total_topo = 0.0, 0.0, 0.0, 0.0
        total_error = 0.0
        log_interval = self.args.log_interval
        start_time = time.time()
        num_batches = len(loader)
        for batch_idx, batch_data in enumerate(loader):
            model.zero_grad()
            input_gene_ids = batch_data["gene_ids"].to(self.device) #从 batch_data 中取出数据并移动到指定设备（GPU/CPU） 处理输入数据(基因ID和表达值)
            src_key_padding_mask = input_gene_ids.eq(self.vocab[self.pad_token]) # 基因 ID（token IDs）
            input_values = batch_data["masked_values"].to(self.device)  # input_values: 已经被掩码处理的表达值masked-> -1
            target_values = batch_data["target_values"].to(self.device) #  真实的目标表达值（用于监督训练）
            if self.args.graph_sort: #如果启用了 graph_sort（图排序任务），则还加载了与拓扑结构相关的输入和标签：
                target_sorted_gene = batch_data['sorted_gene_ids'].to(self.device)
                input_sorted_gene = batch_data['masked_sorted_gene_ids'].to(self.device)
                sorted_layer_idx = batch_data['sorted_layer_idx'].to(self.device)
                topo_padding_mask = input_sorted_gene.eq(self.vocab[self.pad_token]).to(self.device)
            else:
                input_sorted_gene = None
                topo_padding_mask = None
                target_sorted_gene = None
            with torch.cuda.amp.autocast(enabled=self.args.amp): # 启用自动混合精度（AMP）训练，提升训练速度并节省显存。
                output_dict = model( # 将输入传入模型，得到输出字典 output_dict，其中包含多个输出头：
                    src=input_gene_ids,
                    values=input_values,
                    batch_labels=None,
                    MVC=self.args.MVC,
                    src_key_padding_mask=src_key_padding_mask,
                    input_sorted_gene=input_sorted_gene if not self.args.generative_pretraining else target_sorted_gene,
                    topo_padding_mask=topo_padding_mask,
                    sorted_layer_idx=sorted_layer_idx if (self.args.graph_sort and self.args.layer_emb) else None
                ) #
                ## masked_value_prediction  计算主任务（MLM：Masked Language Modeling）的损失，使用的是 masked_mse_loss（均方误差）
                masked_positions = input_values.eq(self.mask_value) # masked_positions 标记哪些位置是被掩码的，只对这些位置计算损失。
                loss = loss_mse = self.criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                ) # 模型输出的mlm_output: 掩码表达值预测结果 target_values 
                metrics_to_log = {"train/mlm": loss_mse.item()} # 记录该 loss 到 metrics_to_log 字典中。
                if self.args.graph_sort: #如果启用了 图排序
                    if self.args.generative_pretraining:
                        logit = output_dict['lm_logit'][:, :-1, :].clone()
                        logit = logit.view(-1, output_dict['lm_logit'].size(-1))
                        label = target_sorted_gene[:, 1:].clone()
                        label = label.view(-1).long()
                        padded_positions = label.eq(self.vocab[self.pad_token])
                        label[padded_positions] = -100
                    else:  # randolym masked token prediction
                        logit = output_dict['lm_logit'].view(-1, output_dict['lm_logit'].size(-1))
                        label = target_sorted_gene.view(-1).long()
                        topo_needed_to_pred = input_sorted_gene.eq(self.vocab[self.mask_token]).to(self.device)
                        masked_pos = torch.logical_or(topo_padding_mask, ~topo_needed_to_pred).view(-1)
                        label[masked_pos] = -100
                    topo_sorting_loss = self.lm_criterion(logit, label) #计算交叉熵损失（CrossEntropyLoss）
                    weight = loss_mse.item() / topo_sorting_loss.item() # 动态加权策略：根据两个任务的 loss 大小动态调整权重。
                    loss = loss + weight * topo_sorting_loss
                    metrics_to_log.update({"train/topo_loss": topo_sorting_loss.item()})
                if self.args.MVC: # 可选任务：MVC（Masked Value Classification）
                    loss_MVC = self.criterion(
                        output_dict["mvc_output"], target_values, masked_positions
                    )
                    weight = loss_mse.item() / loss_MVC.item() # 同样使用动态加权策略融合进总 loss。
                    loss = loss + weight * loss_MVC
                    metrics_to_log.update({"train/mvc": loss_MVC.item()})
            self.scaler.scale(loss).backward() # 反向传播 & 梯度裁剪
            self.scaler.unscale_(self.optimizer) #使用 GradScaler 支持混合精度下的梯度缩放。
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_( # 
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if self.scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    if self.is_master:
                        self.logger.warning(
                            f"Found infinite gradient. This may be caused by the gradient "
                            f"scaler. The current scale is {self.scaler.get_scale()}. This warning "
                            "can be ignored if no longer occurs after autoscaling of the scaler."
                        )
            self.scaler.step(self.optimizer) #更新参数。
            self.scaler.update() #更新 scaler 缩放因子。
            if self.is_master: # 如果是主进程（分布式训练下），记录到 WandB。
                wandb.log(metrics_to_log)
            with torch.no_grad():
                mre = masked_relative_error(
                    output_dict["mlm_output"], target_values, masked_positions
                ) # 使用 masked_relative_error 计算相对误差（非损失，但作为评估指标）。
            total_loss += loss.item() # 累加各项指标，用于后续日志输出。
            total_mse += loss_mse.item()
            total_MVC += loss_MVC.item() if self.args.MVC else 0.0
            total_topo += topo_sorting_loss.item() if self.args.graph_sort else 0.0
            total_error += mre.item()
            if batch_idx % log_interval == 0 and batch_idx > 0: #每隔 log_interval 批次，输出当前训练状态。
                lr = self.scheduler.get_last_lr()[0] #包括：学习率、平均 loss、mse、mre 等。
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                cur_gepc = total_gepc / log_interval if self.args.MVC else 0.0
                cur_topo = total_topo / log_interval if self.args.graph_sort else 0.0
                cur_error = total_error / log_interval
                # ppl = math.exp(cur_loss)
                if self.args.distributed:
                    cur_loss = get_reduced(cur_loss, self.local_rank, 0, self.world_size)
                    cur_mse = get_reduced(cur_mse, self.local_rank, 0, self.world_size)
                    cur_gepc = get_reduced(cur_gepc, self.local_rank, 0, self.world_size)
                    cur_error = get_reduced(cur_error, self.local_rank, 0, self.world_size)
                    cur_topo = get_reduced(cur_topo, self.local_rank, 0, self.world_size)
                if self.is_master: #清空累加器，准备下一阶段统计。
                    self.logger.info(
                        f"| epoch {epoch:3d} | {batch_idx:3d}/{num_batches:3d} batches | "
                        f"lr {lr:05.6f} | ms/batch {ms_per_batch:5.2f} | "
                        f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                        + (f"gepc {cur_gepc:5.2f} |" if self.args.MVC else "")
                        + (f"topo {cur_topo:5.2f} |" if self.args.graph_sort else "")
                    )
                total_loss = 0
                total_mse = 0
                total_gepc = 0
                total_error = 0
                total_topo = 0
                start_time = time.time()

    def evaluate(self,model: nn.Module, loader: DataLoader,epoch) -> float: # 验证循环
        """
        Evaluate the model on the evaluation data.
        """
        model.eval() # 将模型设置为评估模式（关闭 dropout、batch norm 等训练专用行为）。
        if self.args.distributed: # 如果是分布式训练，则调用 dist.barrier() 保证所有进程同步。
            dist.barrier()
        total_loss = 0.0 # 初始化统计变量
        total_error = 0.0
        total_num = 0
        total_topo = 0
        total_mvc = 0
        with torch.no_grad(): # 使用 torch.no_grad() 关闭梯度计算，节省内存。
            for batch_data in loader: # 数据遍历与前向推理 对每个 batch 执行前向传播，获取输出字典 output_dict。
                input_gene_ids = batch_data["gene_ids"].to(self.device)
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.pad_token])
                input_values = batch_data["masked_values"].to(self.device)  # masked-> -1
                target_values = batch_data["target_values"].to(self.device)
                if self.args.graph_sort:
                    target_sorted_gene = batch_data['sorted_gene_ids'].to(self.device)
                    input_sorted_gene = batch_data['masked_sorted_gene_ids'].to(self.device)
                    sorted_layer_idx = batch_data['sorted_layer_idx'].to(self.device)
                    topo_padding_mask = input_sorted_gene.eq(self.vocab[self.pad_token]).to(self.device)
                else:
                    input_sorted_gene = None
                    topo_padding_mask = None
                    sorted_layer_idx = None
                    target_sorted_gene = None
                with torch.cuda.amp.autocast(enabled=self.args.amp): # 同样支持混合精度 (autocast)。
                    output_dict = model(
                        src=input_gene_ids,
                        values=input_values,
                        MVC=self.args.MVC,
                        batch_labels=None,
                        src_key_padding_mask=src_key_padding_mask,
                        input_sorted_gene=input_sorted_gene if not self.args.generative_pretraining else target_sorted_gene,
                        topo_padding_mask=topo_padding_mask,
                        sorted_layer_idx=sorted_layer_idx if (self.args.graph_sort and self.args.layer_emb) else None
                    )
                    # causal loss
                    output_values = output_dict["mlm_output"]
                    masked_positions = input_values.eq(self.mask_value)
                    loss = self.criterion(output_values, target_values, masked_positions) #损失计算（多任务）
                    if self.args.graph_sort: #  可选任务：图排序（Topological Sorting）
                        if self.args.generative_pretraining:
                            logit = output_dict['lm_logit'][:, :-1, :].clone()
                            logit = logit.view(-1, output_dict['lm_logit'].size(-1))
                            label = target_sorted_gene[:, 1:].clone()
                            label = label.view(-1).long()
                            padded_positions = label.eq(self.vocab[self.pad_token])
                            label[padded_positions] = -100
                        else:  # randolym masked token prediction 
                            logit = output_dict['lm_logit'].view(-1, output_dict['lm_logit'].size(-1))
                            label = target_sorted_gene.view(-1).long()
                            topo_needed_to_pred = input_sorted_gene.eq(self.vocab[self.mask_token]).to(self.device)
                            masked_pos = torch.logical_or(topo_padding_mask, ~topo_needed_to_pred).view(-1)
                            label[masked_pos] = -100
                        topo_sorting_loss = self.lm_criterion(logit, label) #计算交叉熵损失用于图结构排序任务。
                        weight = loss.item() / topo_sorting_loss.item() # 使用动态加权策略平衡不同任务的重要性。
                        loss = loss + weight * topo_sorting_loss
                        # loss = loss + topo_sorting_loss
                    if self.args.MVC: # 可选任务：MVC（Masked Value Classification）
                        loss_gepc = self.criterion(
                            output_dict["mvc_output"], target_values, masked_positions
                        )
                        weight = loss.item() / loss_gepc.item()
                        loss = loss + weight * loss_gepc
                total_loss += loss.item() * len(input_gene_ids) #累加损失和误差，并乘以样本数量以便后续取平均。
                total_topo += topo_sorting_loss.item() * len(input_gene_ids) if self.args.graph_sort else 0
                total_mvc += loss_gepc.item() * len(input_gene_ids) if self.args.MVC else 0
                total_error += masked_relative_error(
                    output_values, target_values, masked_positions
                ).item() * len(input_gene_ids)
                # total_dab += loss_dab.item() * len(input_gene_ids)
                total_num += len(input_gene_ids)
        if self.args.distributed: # 分布式训练处理（多 GPU）
            mse = get_reduced(total_loss / total_num, self.local_rank, 0, self.world_size) # 使用 get_reduced来聚合各个 GPU 上的结果。
            mre = get_reduced(total_error / total_num, self.local_rank, 0, self.world_size) # 最终得到平均损失（mse）、平均误差（mre）等。
            topo_mse = get_reduced(total_topo / total_num, self.local_rank, 0, self.world_size)
            mvc_mse = get_reduced(total_mvc / total_num, self.local_rank, 0, self.world_size)
        else:
            mse = total_loss / total_num
            mre = total_error / total_num
            topo_mse = total_topo / total_num
            mvc_mse = total_mvc / total_num
        total_mse = mse + topo_mse + mvc_mse
        if self.is_master: # 日志记录（WandB）
            wandb.log(
                {
                    "valid/mse": mse,
                    "valid/mre": mre,
                    "valid/topo": topo_mse,
                    "valid/mvc": mvc_mse,
                    # / total_num,
                    "epoch": epoch,
                })

        return mse, mre, total_mse
        #返回三个值： mse: 主任务（MLM）的均方误差   mre: 掩码相对误差  total_mse: 所有任务损失之和（用于早停判断）

    def run_pretrain(self,): # 主训练流程
        if self.is_master: # 初始化 WandB（仅主进程）
            self.set_wandb()
            define_wandb_metrcis()
        model, train_loader, valid_loader=self.load_data_and_model() # 加载模型和数据
        self.load_criterion_and_opt(model) # 加载损失函数、优化器、学习率调度器等
        if self.is_master: # 注册模型到 WandB（仅主进程）
            wandb.watch(model)
        best_val_loss = float("inf") # 初始化最小验证损失为无穷大。
        best_model = copy.deepcopy(model) # best_model 用于保存验证损失最低时的模型状态。 复制一份当前模型作为初始最佳模型。
        args=self.args

        for epoch in range(1, args.epochs + 1): #协调训练和验证循环 开始训练循环（按 epoch）
            if args.distributed:
                dist.barrier()
            epoch_start_time = time.time() # 统计每一轮训练耗时。
            if args.do_train: # 执行训练（如果启用）调用 train() 方法进行训练
                self.train(
                    model,
                    loader=train_loader,
                    epoch=epoch
                )
                val_loss, val_mre, val_total_mse = self.evaluate( # 验证模型表现 调用 evaluate() 方法评估模型在验证集上的性能。
                    model,
                    loader=valid_loader,
                    epoch=epoch
                )
                elapsed = time.time() - epoch_start_time
                if self.is_master: # 打印训练结果（仅主进程）
                    self.logger.info("-" * 89)
                    self.logger.info(
                        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                        f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
                    )
                    self.logger.info("-" * 89)
                if val_total_mse < best_val_loss: #更新最佳模型（如果当前模型更好）
                    best_val_loss = val_total_mse
                    best_model = copy.deepcopy(model)
                    best_model_epoch = epoch
                    if self.is_master:
                        self.logger.info(f"Best model with total mse score {best_val_loss:5.4f}")
                if epoch == 1 or epoch % args.save_eval_interval == 0 or epoch == args.epochs: # 定期保存模型（仅主进程）
                    if self.is_master:
                        self.logger.info(f"Saving model to {self.save_dir}")
                        if args.distributed:
                            torch.save(best_model.module.state_dict(), self.save_dir / f"model_e{best_model_epoch}.pt")
                        else:
                            torch.save(best_model.state_dict(), self.save_dir / f"model_e{best_model_epoch}.pt")
            if self.args.distributed:
                dist.barrier()
            self.scheduler.step() #学习率调度器更新（每个 epoch 结束后）更新学习率，通常是根据 StepLR 的规则进行衰减。
        if self.is_master: # 保存最佳模型
            if args.distributed:
                torch.save(best_model.module.state_dict(), self.save_dir / "best_model.pt")
            else:
                torch.save(best_model.state_dict(), self.save_dir / "best_model.pt")
            artifact = wandb.Artifact(f"best_model", type="model")
            glob_str = os.path.join(self.save_dir, "best_model.pt")
            artifact.add_file(glob_str)
            self.run.log_artifact(artifact)

            self.run.finish() # 清理资源
            wandb.finish()
            gc.collect()
    #负责协调数据加载、模型初始化、训练与验证循环、模型保存以及日志记录等关键步骤。我们可以把它理解为整个训练过程的“总指挥”。

if __name__ == "__main__": # Python脚本的主程序入口
    debug=False
    if debug:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    # config_file = sys.argv[1]
    config_file = r'../config/scmamba_PT.toml' ## 重点 这是可以改的 通过配置文件(scmamba_PT.toml)指定参数 改一下就可以干自己的
    task = PretrainTaskScMamba(config_file)
    task.run_pretrain()
# 当这个Python脚本被直接运行（而不是被导入为模块）时，if __name__ == "__main__": 下的代码块会自动执行。 
#自动执行：当脚本被直接运行时，if __name__ == "__main__": 下的代码会自动触发，完成从配置加载到训练的全流程。
#手动控制：如需定制化，可以通过修改 debug 或配置文件路径，或者将脚本作为模块导入后选择性调用功能。