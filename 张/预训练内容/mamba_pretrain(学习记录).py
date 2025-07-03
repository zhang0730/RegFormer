# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:mamba_pretrain.py
# @Software:PyCharm
# @Created Time:2024/1/15 3:38 PM
# %%

#模型预训练：使用单细胞RNA测序数据进行预训练，学习基因表达模式
#核心是 MambaModel，这是一种改进的 Mamba 架构(一种新型的状态空间模型，可替代 Transformer)
#模型构架：基因嵌入层  多层 Mamba 结构  掩码基因表达值预测(MLM)  基于图的基因排序功能  细胞嵌入生成器
#数据输入：支持两种数据格式：LMDB(内存映射数据库)和 H5AD(单细胞分析常用格式) 后者是真的吗
#使用 GeneVocab 类进行基因标记化  处理基因表达值的掩码，用于自监督学习  可选启用基于图的基因排序

import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import List, Dict, Optional
import warnings
import torch
import scanpy as sc
from anndata import AnnData
import datetime
import numpy as np
import wandb
from torch import nn
from torch.utils.data import DataLoader,random_split
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import argparse
sys.path.append("../")
from regformer.utils.utils import get_reduced,seed_all
from regformer.model.mambaLM import MambaModel
from regformer.data.pretrain_testdataset import prepare_test
from regformer.data.gene_tokenizer import GeneVocab
from regformer.data.dataset import Load_Data
from regformer.data.dataloader import Get_DataLoader
import regformer as scm
from regformer.model.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from regformer.utils import eval_scib_metrics

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
# most important
parser.add_argument("--task", type=str, default='Pretraining',choices=['Cell_annotation','Integration','Pretraining'], help='Name of task')#
parser.add_argument("--data_name", type=str, default='cellxgene', choices=['panglao','cellxgene'],help='Name of dataset')#
parser.add_argument("--model_name", type=str, default='mamba', help='name of model type.')#
parser.add_argument("--run_name", type=str, default='debug', help='name of experiment.')#
parser.add_argument("--distributed", type=bool, default=False, help='debug mode, single gpu device')#
parser.add_argument("--single_gpu", type=bool, default=False, help='single gpu device, but not debug mode')#
parser.add_argument("--do_train", type=bool, default=True, help='Train or inference')#
parser.add_argument("--GEPC", type=bool, default=False, help='Masked value prediction for cell embedding.')#
# general hyperparameters
parser.add_argument("--epochs", type=int, default=5, help='Number of epochs.')#
parser.add_argument("--seed", type=int, default=1927, help='Random seed.')#
parser.add_argument("--dropout", type=float, default=0.2, help='dropout rate.')#
parser.add_argument("--batch_size", type=int, default=8, help='Number of batch size.')#
parser.add_argument("--lr", type=float, default=1e-6, help='Learning rate.')#
parser.add_argument("--num_workers", type=int, default=0, help='number of workers when processing.')#
parser.add_argument("--log_interval",type=int, default=1000,help='interval of log.')#
# parser.add_argument("--forcing_wandb", type=bool, default=False, help='whether open wandb by force')#
parser.add_argument("--save_eval_interval",type=int, default=1,help='interval of evaluation')#
parser.add_argument("--schedule_ratio",type=float, default=0.9,help='ratio of epochs for learning rate schedule')#
parser.add_argument("--amp",type=bool, default=True,help='Automatic Mixed Precision')#
parser.add_argument("--token_emb_freeze",type=bool, default=False,help='freezing token-emb when predicting value')#
# path-related
parser.add_argument("--data_path", type=str, default='/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/data', help='Path of preprocessed data')#
parser.add_argument("--source_path", type=str, default='/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/train_data/cellxgene_6w', help='Path of source lmdb&h5ad data')#2种数据形式？
parser.add_argument("--lmdb",type=bool, default=True,help='use lmdb dataset or not')#
parser.add_argument("--load_model", type=str, default='/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/ckpts/whole_human', help='Path of pretrained model.')
parser.add_argument("--save_dir", type=str, default='/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/saves', help='Directory of checkpoint and result to save.')
parser.add_argument("--vocab_file", type=str, default='/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/ckpts/whole_human/vocab.json', help='Path of vocab, available if load_model is None')
parser.add_argument("--gene_array_file", type=str, default='/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/data/Pretraining/panglao/binned/panglao_gene_ids.pk', help='Path of vocab, available if load_model is None')
parser.add_argument("--graph_path", type=str, default="/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/graph", help='Path of graph')#

# if load model, batch_size, layer_size, nlayers, nhead will be ignored
parser.add_argument("--embsize", type=int, default=512, help='Size of embedding.')#
parser.add_argument("--d_hid", type=int, default=512, help='Size of hidden state.')#
parser.add_argument("--nheads", type=int, default=8, help='number of attention head')#
parser.add_argument("--nlayers", type=int, default=12, help='number of transformer layers')#
parser.add_argument("--mask_ratio", type=float, default=0.25, help='ratio of masked token.')#
parser.add_argument("--append_cls", type=bool, default=False, help='append <cls> token as first token')#
parser.add_argument("--pre_norm", type=bool, default=False, help='normalize previously')#
parser.add_argument("--n_layers_cls", type=int, default=3, help='number of transformer layers')#
parser.add_argument("--graph_sort", type=bool, default=False, help='using graph sorting')#
parser.add_argument("--sampling_etype", default='co_expression',type=str, choices=['share_pathway_with','interact_with','co_expression','ori'], help='choice of edge type when sampling')
parser.add_argument("--layer_mask", type=bool, default=False, help='using layer mask or not when using graph sort')#
parser.add_argument("--layer_emb", type=bool, default=False, help='using layer emb or not when using graph sort')#
parser.add_argument("--generative_pretraining", type=bool, default=False, help='using generative token precidtion in pretraining or masked token prediction in pretraining')#

# data preprocessing related hyper-params
parser.add_argument("--n_bins", type=int, default=51, help='Number of bins.')#
parser.add_argument("--n_hvg", type=int,default=-1, help='whether to subset the raw data to highly variable genes. -1: turn off hvg, positive: number of hvg')#
parser.add_argument("--ecs_thres", type=float, default=0, help='Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable.')#
parser.add_argument("--dab_weight", type=float, default=1.0, help='weight of dab.')#
parser.add_argument("--per_seq_batch_sample", type=bool, default=False, help='whether sort the adata by batch_id')#
parser.add_argument("--DSBN", type=bool, default=False, help='Domain-spec batchnorm')#
parser.add_argument("--explicit_zero_prob", type=bool, default=False, help='whether explicit bernoulli for zeros')#
parser.add_argument("--include_zero_gene", type=bool, default=False, help='whether include gene with zero expression value')
parser.add_argument("--use_batch_labels", type=bool, default=False, help='use batch emb or not, turn it off when pretraining')#
parser.add_argument("--max_seq_len", type=int, default=3000, help='max length of gene sequence')
parser.add_argument("--input_emb_style", type=str, default='continuous',choices=['continuous','category','scaling'], help='the style of input emb')#
parser.add_argument("--cell_emb_style", type=str, default="cls",choices=['final','cls','avg-pool','w-pol','attn'], help='method for generating cell emb')#
parser.add_argument("--mvc_decoder_style", type=str, default='inner product',choices=['inner product','concat query','sum query'], help='architecture style of the decoder')#
parser.add_argument("--bimamba_type", type=str, default='none',choices=['v1','v2','none'], help='tpye of bimamba')#



args = parser.parse_args()
sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
assert not (args.single_gpu and args.distributed)
debug=False
if args.distributed:# multi GPU mode
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ['LOCAL_RANK'])
    is_master = rank == 0 # 分布式训练（多GPU/多节点）环境下，is_master 是一个布尔标志，用于标识当前进程是否是主进程（Master Process）。主进程（Master）：通常对应 rank=0 的进程，负责全局协调任务（如日志记录、模型保存、WandB监控等）。下面会用if is_master
    dist.init_process_group(backend='nccl')
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_all(args.seed + torch.distributed.get_rank())
    if is_master:
        print("world size:",world_size)
else:
    if args.single_gpu:# single GPU mode
        local_rank=int(os.environ['LOCAL_RANK'])
        rank = int(os.environ["RANK"])
    else: #debug mode
        os.environ["WANDB_MODE"] = "offline"
        os.environ["CUDA_VISIBLE_DEVICES"]='2'
        local_rank = 0
        rank=0
        debug=True
        #args.input_emb_style='category'
        from analysis_tools.debug_mode import config_correction
        args = config_correction(args, version='HPC_new')
        args.log_interval=100
        # args.append_cls=True
        args.graph_sort=False
        # args.embsize=2048
        # args.d_hid=2048
        args.batch_size=8
        # "/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/saves/Pretraining/cellxgene/Mamba/gst_emb_attnMVC_mr3"
        args.load_model="none"
        args.generative_pretraining=True
        args.GEPC=True
        args.bimamba_type='none'
        args.cell_emb_style='attn'
        args.layer_emb=False
        #args.n_bins=0
    is_master=True
    world_size=1
    #device = torch.device("cpu", local_rank)
    device = torch.device("cuda", local_rank)
    seed_all(args.seed)
if is_master:
    ## wandb setting
    now=datetime.datetime.now().strftime("%Y-%m-%d")
    wandb_name=f'{args.data_name}_{args.model_name}{str(args.bimamba_type) if args.bimamba_type!="none" else ""}_{"lye" if args.layer_emb else ""}_{"CLM" if args.generative_pretraining else "MLM"}_{args.run_name}_G{world_size}_{now}'
    wandb_tags=[args.task,f'G{world_size}',
                args.data_name,args.model_name,'gts' if args.graph_sort else 'wogts',f'layer_mask{args.mask_ratio}' if args.layer_mask else f'random_mask{args.mask_ratio}',
                'layer_positional_emb' if args.layer_emb else 'w/o lyemb']
    run = wandb.init(
        config=args.__dict__,
        job_type=args.task,
        project="scLLM-Pretrain",
        name=wandb_name,
        tags=wandb_tags,
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    print(args)
    logger = scm.logger
print(f'current devices: {device}')

# set_seed(args.seed)
# %%
# settings for input and preprocessing
pad_token = "<pad>"
mask_ratio = args.mask_ratio
n_input_bins = args.n_bins
mask_value = -1
pad_value = -2
valid_ratio=0.1
# number of highly variable genes

per_seq_batch_sample = args.per_seq_batch_sample
DSBN = args.DSBN  # Domain-spec batchnorm
explicit_zero_prob = args.explicit_zero_prob  # whether explicit bernoulli for zeros


# %%


if args.load_model != "none":
    model_dir = Path(args.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json" #作者给了词汇表
    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    if is_master:
        logger.info(
            f"Resume model from {model_file}, the model args will be overriden by the "
            f"config {model_config_file}."
        )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    args.n_layers_cls = model_configs["n_layers_cls"]
else:
    if is_master:
        logger.info(f"Training model from scratch.")
    embsize = args.embsize
    nhead = args.nheads
    nlayers = args.nlayers
    d_hid = args.d_hid
    vocab_file=args.vocab_file
    # %%
vocab = GeneVocab.from_file(vocab_file)
mask_token='<mask>' if '<mask>' in vocab.vocab.itos_ else '<eoc>'
unk_token='<unk>' if args.append_cls else '<cls>'
special_tokens = [pad_token, mask_token]
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
if is_master:
    save_dir=os.path.join(args.save_dir,args.task,args.data_name,args.model_name,args.run_name)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    # save the whole script to the dir
    os.system(f"cp {__file__} {save_dir}")
    os.system(f"cp {vocab_file} {os.path.join(save_dir,'vocab.json')}")

    scm.utils.add_file_handler(logger, save_dir / "run.log")
    with open(os.path.join(save_dir,'args.json'),'w') as f:
        json.dump(args.__dict__,f)
        f.close()
else:
    logger=None

# %% [markdown]
# ## Loading and preparing data  加载数据 确实是两种数据类型可以选择
if not args.lmdb:
    if is_master:print('Load h5ad dataset......') 
    data_path=os.path.join(args.data_path,args.task,args.data_name)
    total_data=Load_Data(data_path=data_path,args=args,
                         vocab=vocab,mask_ratio=args.mask_ratio,append_cls=args.append_cls,
                         include_zero_gene=False,need_length=True,max_seq_len=args.max_seq_len)
    if debug:
        print(total_data[100000])
    valid_size, train_size = int(len(total_data) * valid_ratio), len(total_data) - int(len(total_data) * valid_ratio)
    train_dataset, valid_dataset = random_split(total_data, [train_size, valid_size])
    train_loader =Get_DataLoader(
        train_dataset,
        args=args,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    valid_loader = Get_DataLoader(
        valid_dataset,
        args=args,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    if is_master:
        logger.info(
            f"Total set number of cells: {valid_size+train_size}, from {total_data.n_files} h5ad files."
            f"\n\t Train: {train_size}, Valid: {valid_size}")
        if not args.include_zero_gene:
            logger.info(f"\n\tOnly using non-zero genes:"
                        f"\n\tThe max length of non-zero gene: {total_data.max_non_zero_count}"
                        f"\n\tThe min length of non-zero gene: {total_data.min_non_zero_count}"
                        f"\n\tUniform the length into max_seq_len: {args.max_seq_len}")
        else:
            logger.info(f"\n\t Using all the genes, the length of whole gene: {total_data.max_non_zero_count}")
else:
    if is_master: print('Load lmdb dataset......')
    data_path = os.path.join(args.data_path, args.task, args.data_name)
    train_dataset,valid_dataset = Load_Data(data_path=data_path, args=args,
                           vocab=vocab, mask_ratio=args.mask_ratio, append_cls=args.append_cls,
                           include_zero_gene=args.include_zero_gene,max_seq_len=args.max_seq_len,mask_token=mask_token,unk_token=unk_token)
    if debug:
        print(train_dataset[100000],train_dataset[6526354],valid_dataset[1927])
    train_loader = Get_DataLoader(
        dataset=train_dataset,
        args=args,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    valid_loader = Get_DataLoader(
        dataset=valid_dataset,
        args=args,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    if is_master:
        logger.info(
            f"Total set number of cells: {len(train_dataset)+len(valid_dataset)}."
            f"\n\t Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
        if not args.include_zero_gene:
            logger.info(f"\n\t Only using non-zero genes:"
                        f"\n\tUniform the length into max_seq_len: {args.max_seq_len}")
        else:
            logger.info(f"\n\t Using all the genes, the length of whole gene, uniform the length into max_seq_len: {args.max_seq_len}")

test_path=os.path.join(args.data_path,'Pretraining',"test")
adata,gene_ids,gene_ids_in_vocab=prepare_test(test_path,vocab,is_master,args,logger)




#关键类
ntokens = len(vocab)  # size of vocabulary
model = MambaModel(
    ntoken=ntokens,  # 词汇表大小(基因数量)
    d_model=embsize,  # 嵌入维度(默认512)
    nlayers=nlayers,  # Mamba层数(默认12)
    nlayers_cls=args.n_layers_cls,  # 分类头的层数(默认3)
    vocab=vocab, # 基因词汇表对象
    dropout=args.dropout, # dropout率(默认0.2)
    pad_token=pad_token,  # 填充token("<pad>")
    pad_value=pad_value,  # 填充值(-2)
    do_mvc=args.GEPC,  # 是否进行基因表达预测(默认False)
    do_dab=False,  # 是否进行批次效应校正(默认关闭)
    use_batch_labels=args.use_batch_labels, # 是否使用批次标签
    domain_spec_batchnorm=DSBN, # 是否使用领域特定批归一化
    n_input_bins=n_input_bins, # 输入分箱数(默认51)
    ecs_threshold=args.ecs_thres, # 弹性细胞相似性阈值
    input_emb_style=args.input_emb_style, # 输入嵌入风格('continuous'/'category'/'scaling')
    cell_emb_style=args.cell_emb_style, # 细胞嵌入生成方式('cls'/'avg-pool'等)
    mvc_decoder_style=args.mvc_decoder_style, # MVC解码器风格
    explicit_zero_prob=explicit_zero_prob,  # 显式零概率处理
    pre_norm=args.pre_norm, # 是否使用预归一化
    do_pretrain=True, # 预训练模式
    topo_graph=args.graph_sort, # 是否使用图排序
    if_bimamba=args.bimamba_type!="none", # 是否使用双向Mamba
    bimamba_type=args.bimamba_type, # 双向Mamba类型('v1'/'v2')
    if_devide_out=False, # 是否分头输出
    init_layer_scale=None, # 层缩放初始化
    token_emb_freeze=args.token_emb_freeze # 是否冻结token嵌入
)
if args.load_model !="none": # 加载模型就是预训练模式？
    try: # 尝试完整加载模型参数
        model.load_state_dict(torch.load(model_file))
        if is_master:
            logger.info(f"Loading all model params from {model_file}")
    except: # 部分加载兼容机制
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        ckpt_emb_shape = pretrained_dict['encoder.embedding.weight'].size()
        # 只加载形状匹配的参数
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        if is_master:
            if not 'encoder.embedding.weight' in pretrained_dict:
                logger.warning(f'{"!" * 30}Embeddings Unavailable{"!" * 30}\n'
                               f'Expected shape: {model_dict["encoder.embedding.weight"].size()}\n'
                               f'But got shape: {ckpt_emb_shape} from ckpt {model_file}')
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict) # 更新并加载参数
        model.load_state_dict(model_dict)
if is_master:
    # params={}
    # for name, param in model.named_parameters():
    #     params.update({name:param.numel()})
    total_params = sum(p.numel() for p in model.parameters())
    params={'total_params':total_params} ## 计算并记录模型参数总量
    print(params)
    wandb.log(params)
model.to(device) ## 将模型移至指定设备(GPU/CPU)
if args.distributed: # 分布式训练设置
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=False)

#训练关键组件初始化
#损失函数设置
criterion = masked_mse_loss # 主损失函数(掩码MSE)
if args.graph_sort:
    lm_criterion = nn.CrossEntropyLoss(ignore_index=-100) # 图排序的交叉熵损失
    
# 优化器配置
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, eps=1e-4 if args.amp else 1e-8
)
# 学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.schedule_ratio) # 学习率衰减系数(默认0.9)
# 混合精度梯度缩放器
scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
# WandB监控设置
if is_master:
    wandb.watch(model) # 跟踪模型参数和梯度

#实现了 Mamba 模型在单细胞 RNA 测序数据上的一个完整训练周期(epoch)。
def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    #初始化设置
    model.train() # 设置为训练模式
    total_loss, total_mse, total_gepc,total_topo = 0.0, 0.0, 0.0,0.0 # 初始化损失统计
    total_error = 0.0
    log_interval = args.log_interval # 日志记录间隔
    start_time = time.time() # 计时开始
    
    #数据批次处理循环
    num_batches = len(loader)
    for batch_idx,batch_data in enumerate(loader):
        # if debug and batch_idx%100==0:
        #     print(batch_idx)
        # if debug and batch_idx<2200:
        #     continue
        model.zero_grad()  # 清空梯度
        # 准备输入数据
        input_gene_ids = batch_data["gene_ids"].to(device) # 基因ID序列
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token]) # 填充位置掩码
        input_values = batch_data["masked_values"].to(device)  # masked-> -1  掩码后的表达值(-1表示被掩码)
        target_values = batch_data["target_values"].to(device)  # 真实表达值
        # batch_labels = batch_data["batch_labels"].to(device)

        if args.graph_sort: # 图排序相关数据准备(如果启用)
            target_sorted_gene=batch_data['sorted_gene_ids'].to(device)
            input_sorted_gene=batch_data['masked_sorted_gene_ids'].to(device)
            sorted_layer_idx = batch_data['sorted_layer_idx'].to(device)
            # if args.sampling_etype == 'ori':
            #     input_gene_ids = target_sorted_gene.clone()
            topo_padding_mask= input_sorted_gene.eq(vocab[pad_token]).to(device)
        else:
            input_sorted_gene=None
            topo_padding_mask=None
            target_sorted_gene=None

        #混合精度前向传播
        with torch.cuda.amp.autocast(enabled=args.amp): # 混合精度上下文
            output_dict = model(
                src=input_gene_ids,
                values=input_values,
                batch_labels=None,
                MVC=args.GEPC,  # 是否进行基因表达预测
                src_key_padding_mask=src_key_padding_mask,
                input_sorted_gene=input_sorted_gene if not args.generative_pretraining else target_sorted_gene,
                topo_padding_mask=topo_padding_mask,
                sorted_layer_idx=sorted_layer_idx if (args.graph_sort and args.layer_emb) else None
            )
            #多任务损失计算
            ## masked_value_prediction 掩码值预测损失(主任务)
            masked_positions = input_values.eq(mask_value) # 找出被掩码的位置
            loss = loss_mse = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )
            metrics_to_log = {"train/mlm": loss_mse.item()}
            if explicit_zero_prob: #零值概率损失(可选)
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
            if args.graph_sort: #图排序损失(可选)
                if args.generative_pretraining: # 生成式预训练
                    logit = output_dict['lm_logit'][:, :-1, :].clone() # 预测下一个token
                    logit = logit.view(-1, output_dict['lm_logit'].size(-1)) 
                    label = target_sorted_gene[:,1:].clone() # 真实下一个token
                    label=label.view(-1).long()
                    padded_positions = label.eq(vocab[pad_token])
                    label[padded_positions]=-100 # 忽略不需要预测的位置
                else:# randolym masked token prediction
                    logit = output_dict['lm_logit'].view(-1, output_dict['lm_logit'].size(-1))
                    label = target_sorted_gene.view(-1).long()
                    topo_needed_to_pred = input_sorted_gene.eq(vocab[mask_token]).to(device)
                    masked_pos = torch.logical_or(topo_padding_mask, ~topo_needed_to_pred).view(-1)
                    label[masked_pos] = -100
                topo_sorting_loss = lm_criterion(logit, label)
                # if debug:
                #     print(f'loss: {topo_sorting_loss.item()},total needed prediction num:{label[label!=-100].__len__()},logit mask token (60696) num:{(logit[label!=-100].argmax(dim=1)==60696).sum()}')
                #     if label[label!=-100].__len__()!=(logit[label!=-100].argmax(dim=1)==60696).sum():
                #         print(label[label!=-100])
                #         print(logit[label!=-100].argmax(dim=1))
                #
                # if debug and topo_sorting_loss.item()==0:
                #     print(f'label:{label[10:60]}\nsize:{label.size()}\nmask_num:{topo_needed_to_pred.sum()}\npad_num:{topo_padding_mask.sum()}\ntotal_mask_num:{masked_pos.sum()}\n')
                #     print(f'logit:{logit[10:60]}\nsize:{logit.size()}')
                #     pass
                weight = loss_mse.item() / topo_sorting_loss.item()
                loss = loss + weight * topo_sorting_loss
                metrics_to_log.update({"train/topo_loss": topo_sorting_loss.item()})
            if args.GEPC: #基因表达预测损失(可选)
                loss_gepc = criterion(
                    output_dict["mvc_output"], target_values,masked_positions
                )
                weight = loss_mse.item()/loss_gepc.item()
                loss = loss + weight*loss_gepc
                metrics_to_log.update({"train/mvc": loss_gepc.item()})
            if args.GEPC and explicit_zero_prob:
                loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values,masked_positions
                )

                loss = loss + loss_gepc_zero_log_prob

                metrics_to_log.update(
                    {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                )
            if args.ecs_thres > 0:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
            metrics_to_log.update({"train/loss": loss.item()})
        # 反向传播与优化
        scaler.scale(loss).backward() # 混合精度反向传播
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            ## 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                if is_master:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
        scaler.step(optimizer) ## 更新参数
        scaler.update() # 更新缩放器
        # 记录指标到WandB(仅主进程)
        if is_master:
            wandb.log(metrics_to_log)
        with torch.no_grad():# 计算相对误差 这点比较重要呢
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )
        # 累积统计量
        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_gepc += loss_gepc.item() if args.GEPC else 0.0
        total_topo += topo_sorting_loss.item() if args.graph_sort else 0.0
        total_error += mre.item()
        #定期日志输出
        if batch_idx % log_interval == 0 and batch_idx > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            ## 计算平均指标
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_gepc = total_gepc / log_interval if args.GEPC else 0.0
            cur_topo = total_topo/log_interval if args.graph_sort else 0.0
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            if args.distributed: # 分布式训练下聚合所有进程的结果
                cur_loss=get_reduced(cur_loss, local_rank, 0, world_size)
                cur_mse = get_reduced(cur_mse, local_rank, 0, world_size)
                cur_gepc = get_reduced(cur_gepc, local_rank, 0, world_size)
                cur_error= get_reduced(cur_error, local_rank, 0, world_size)
                cur_topo = get_reduced(cur_topo, local_rank, 0, world_size)
            ## 主进程打印日志
            if is_master:
                logger.info(
                    f"| epoch {epoch:3d} | {batch_idx:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.6f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                    +(f"gepc {cur_gepc:5.2f} |" if args.GEPC else "")
                    +(f"topo {cur_topo:5.2f} |" if args.graph_sort else "")
                )
            # 重置统计量
            total_loss = 0
            total_mse = 0
            total_gepc = 0
            total_error = 0
            total_topo=0
            start_time = time.time()
            if debug:
                break
                #pass

#定义 WandB (Weights & Biases) 实验跟踪工具中各项指标的显示方式和聚合方式
def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")
#summary：指定指标的汇总方式 "min"：显示该指标的最小值（适合损失类指标） "max"：显示该指标的最大值（适合准确率等正向指标）  step_metric：指定作为x轴的指标（这里统一使用"epoch"）


#evaluate() 函数用于在验证集上评估模型性能，是训练过程中关键的质量监控环节。
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.在给定数据加载器(loader)上评估模型性能
    """
    model.eval() # 切换到评估模式(关闭dropout等)
    if args.distributed:
        dist.barrier() # 分布式训练下的进程同步
    ## 初始化统计变量
    total_loss = 0.0
    total_error = 0.0
    total_num = 0
    total_topo=0
    total_mvc=0
    #批次数据处理核心逻辑
    #数据准备
    with torch.no_grad(): # 禁用梯度计算
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])  # 填充位置掩码
            input_values = batch_data["masked_values"].to(device)  # masked-> -1  输入值(-1表示掩码)
            target_values = batch_data["target_values"].to(device)  # 真实值
            if args.graph_sort: #图排序数据处理（可选）
                target_sorted_gene = batch_data['sorted_gene_ids'].to(device) # 目标基因序列
                input_sorted_gene = batch_data['masked_sorted_gene_ids'].to(device) # 掩码后的输入序列
                sorted_layer_idx = batch_data['sorted_layer_idx'].to(device) # 序列填充掩码
                # if args.sampling_etype=='ori':
                #     input_gene_ids=target_sorted_gene.clone()
                topo_padding_mask = input_sorted_gene.eq(vocab[pad_token]).to(device)
                # topo_needed_to_pred = input_sorted_gene.eq(vocab[mask_token]).to(device)
                # masked_pos = torch.logical_or(topo_padding_mask, ~topo_needed_to_pred)
                # target_sorted_gene[masked_pos] = -100
            else:
                input_sorted_gene = None
                topo_padding_mask = None
                sorted_layer_idx = None
                target_sorted_gene=None
            #混合精度前向传播
            with torch.cuda.amp.autocast(enabled=args.amp):
                output_dict=model(
                    src=input_gene_ids,
                    values=input_values,
                    MVC=args.GEPC, ## 是否启用基因表达预测
                    batch_labels=None,
                    src_key_padding_mask=src_key_padding_mask,
                    input_sorted_gene=input_sorted_gene if not args.generative_pretraining else target_sorted_gene,
                    topo_padding_mask=topo_padding_mask,
                    sorted_layer_idx=sorted_layer_idx if (args.graph_sort and args.layer_emb) else None
                )
                # causal loss
                output_values = output_dict["mlm_output"] #这个是什么
                # masked_positions = input_values.eq(mask_value)
                # padded_positions = input_values.eq(pad_value)
                #基础掩码预测损失
                masked_positions = input_values.eq(mask_value)
                loss = criterion(output_values, target_values, masked_positions)
                # mask = torch.logical_or(padded_positions, masked_positions)
                # loss = criterion(output_values[:,:-1], target_values[:,1:], mask[:,:-1])
                # if args.graph_sort:
                #     logit=output_dict['lm_logit'].view(-1, output_dict['lm_logit'].size(-1))
                #     label=target_sorted_gene.view(-1).long()
                #     topo_sorting_loss = lm_criterion(logit, label)
                #     loss = loss + topo_sorting_loss
                if args.graph_sort: #图排序损失（可选）
                    if args.generative_pretraining: # 生成式预训练：预测下一个token
                        logit = output_dict['lm_logit'][:, :-1, :].clone()
                        logit = logit.view(-1, output_dict['lm_logit'].size(-1)) # 预测序列
                        label = target_sorted_gene[:, 1:].clone() # 目标序列（右移一位）
                        label = label.view(-1).long()
                        padded_positions = label.eq(vocab[pad_token])
                        label[padded_positions] = -100
                    else:  # randolym masked token prediction # 掩码预测模式
                        logit = output_dict['lm_logit'].view(-1, output_dict['lm_logit'].size(-1))
                        label = target_sorted_gene.view(-1).long()
                        # 计算需要预测的位置（非填充且被掩码）
                        topo_needed_to_pred = input_sorted_gene.eq(vocab[mask_token]).to(device)
                        masked_pos = torch.logical_or(topo_padding_mask, ~topo_needed_to_pred).view(-1)
                        label[masked_pos] = -100 # 忽略不需要预测的位置
                    topo_sorting_loss = lm_criterion(logit, label)
                    weight = loss.item() / topo_sorting_loss.item()
                    loss = loss + weight * topo_sorting_loss
                    # loss = loss + topo_sorting_loss
                if args.GEPC: #基因表达预测损失（可选）
                    loss_gepc = criterion(
                        output_dict["mvc_output"], target_values, masked_positions
                    )
                    weight = loss.item() / loss_gepc.item()
                    loss = loss + weight * loss_gepc  # 动态加权
                    # loss = loss + loss_gepc
                # loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            #指标统计与聚合
            #累积批次指标
            total_loss += loss.item() * len(input_gene_ids) # 加权损失
            total_topo+=topo_sorting_loss.item()*len(input_gene_ids) if args.graph_sort else 0 # 相对误差
            total_mvc += loss_gepc.item() * len(input_gene_ids) if args.GEPC else 0
            # total_error += masked_relative_error(
            #     output_values[:,:-1], target_values[:,1:], mask[:,:-1]
            # ).item() * len(input_gene_ids)
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item() * len(input_gene_ids)
            #total_dab += loss_dab.item() * len(input_gene_ids)
            total_num += len(input_gene_ids) # 总样本数
            if debug:
                break
                #pass
    #分布式训练归约
    if args.distributed:
        mse = get_reduced(total_loss / total_num,local_rank, 0, world_size) # 跨进程聚合
        mre = get_reduced(total_error / total_num, local_rank, 0, world_size)
        topo_mse=get_reduced(total_topo / total_num, local_rank, 0, world_size)
        mvc_mse=get_reduced(total_mvc / total_num, local_rank, 0, world_size)
    else: ## 计算平均指标
        mse=total_loss / total_num #返回：三个关键指标 - 均方误差(MSE)
        mre=total_error / total_num #返回：三个关键指标 - 相对误差(MRE)和综合损失(total_mse)
        topo_mse = total_topo / total_num
        mvc_mse = total_mvc / total_num
    total_mse=mse+topo_mse+mvc_mse #返回：三个关键指标 - 综合损失(total_mse)
    if is_master: # 主进程记录日志
        wandb.log(
            {
                "valid/mse": mse,
                "valid/mre": mre,
                "valid/topo": topo_mse,
                "valid/mvc":mvc_mse,
                #/ total_num,
                "epoch": epoch,
            })


    return mse, mre,total_mse

#在测试集(adata_t)上评估模型生成的细胞嵌入质量
#计算细胞嵌入 评估生物学指标  生成可视化结果
def eval_testdata(
    model: nn.Module, #训练好的Mamba模型
    adata_t: AnnData,  #AnnData格式的测试数据集
    include_types: List[str] = ["cls"],  #评估类型列表（默认只评估"cls"方式生成的嵌入）
) -> Optional[Dict]:
    """evaluate the model on test dataset of adata_t"""
    model.eval() # 切换到评估模式

    # copy adata_t to avoid reuse previously computed results stored in adata_t  创建数据副本避免污染原始数据
    adata_t = adata_t.copy()

    # Evaluate cls cell embeddings 细胞嵌入生成（CLS方式）
    if "cls" in include_types:
        if is_master:
            logger.info("Evaluating cls cell embeddings")
        #  获取基因嵌入
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
            if args.distributed:
                available_gene_emb=model.cpu().module.encoder.embedding(torch.tensor(gene_ids)).numpy()#[gene_num,emb]
            else:
                available_gene_emb = model.cpu().encoder.embedding(
                    torch.tensor(gene_ids)).numpy()  # [gene_num,emb]
            #print(f'adata_t: {adata_t.shape}, gene: {available_gene_emb.shape}')
            #计算细胞嵌入 矩阵乘法：通过基因表达矩阵与基因嵌入的线性组合得到细胞嵌入，只使用存在于词汇表中的基因(gene_ids_in_vocab>=0)
            cell_embeddings=adata_t.X@available_gene_emb[gene_ids_in_vocab>=0,:]
        #L2归一化
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )
        #存储嵌入结果 存入AnnData的obsm
        adata_t.obsm["X_scGPT"] = cell_embeddings
       
        #生物学指标评估
        results = {}
        try:
            results = eval_scib_metrics(adata_t) # 调用scIB库计算指标
        except Exception as e:
            traceback.print_exc()
            if is_master:
                logger.error(e)
        
        #可视化分析
        #批次效应可视化
        sc.pp.neighbors(adata_t, use_rep="X_scGPT") # 基于嵌入计算邻域图
        sc.tl.umap(adata_t, min_dist=0.3) ## UMAP降维
        fig = sc.pl.umap(
            adata_t,
            color=["str_batch"], # 按批次着色
            title=[f"batch, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
            frameon=False,
            return_fig=True,
            show=False,
        )

        results["batch_umap"] = fig # 存储图像对象
        #细胞类型可视化
        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["celltype"], # 按细胞类型着色
            title=[
                f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",
            ],
            frameon=False,
            return_fig=True,
            show=False,
        )

        results["celltype_umap"] = fig

    if len(include_types) == 1: #目前仅实现"cls"方式，直接返回结果字典
        return results #结果包含：量化指标(来自eval_scib_metrics)  UMAP可视化图像对象
    

#callback 回调函数  在测试数据上评估最佳模型
def callback(save_dir,adata,best_model,best_model_epoch):
    # eval on testdata
    with torch.no_grad():
        results = eval_testdata(
            best_model,
            adata_t=adata,
            include_types=["cls"],
        )
    ## 保存UMAP可视化结果
    results["batch_umap"].savefig(
        save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png", dpi=300
    )
    results["celltype_umap"].savefig(
        save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png", dpi=300
    )
    ## 准备WandB日志数据
    metrics_to_log = {"test/" + k: v for k, v in results.items()} # 所有量化指标
    metrics_to_log["test/batch_umap"] = wandb.Image(
        str(save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png"),
        caption=f"celltype avg_bio epoch {best_model_epoch}",
    ) # 批次效应UMAP

    metrics_to_log["test/celltype_umap"] = wandb.Image(
        str(save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png"),
        caption=f"celltype avg_bio epoch {best_model_epoch}",
    ) # 细胞类型UMAP 
    metrics_to_log["test/best_model_epoch"] = best_model_epoch
    wandb.log(metrics_to_log)
    wandb.log({"avg_bio": results.get("avg_bio", 0.0)}) # 单独记录关键指标
    return

#主训练流程 把上面定义的函数都用起来  其实这在所有预训练流程都适用 只是
# %%
best_val_loss = float("inf") # 初始化最佳损失
best_avg_bio = 0.0 # 初始化最佳生物学指标 
best_model = None # 初始化最佳模型
if is_master:
    define_wandb_metrcis() # 主进程定义WandB指标
#训练循环 重点重点
for epoch in range(1, args.epochs + 1):
    # 分布式训练同步点
    if args.distributed:
        dist.barrier()
    epoch_start_time = time.time()
    # 训练阶段
    if args.do_train:
        train(
            model,
            loader=train_loader,
        )
    ## 验证阶段
    val_loss, val_mre,val_total_mse = evaluate(
        model,
        loader=valid_loader,
    )
    elapsed = time.time() - epoch_start_time
    # 日志记录
    if is_master:
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
        )
        logger.info("-" * 89)
    # 更新最佳模型
    if val_total_mse < best_val_loss:
        best_val_loss = val_total_mse
        best_model = copy.deepcopy(model) # 深拷贝避免后续训练污染
        best_model_epoch = epoch
        if is_master:
            logger.info(f"Best model with total mse score {best_val_loss:5.4f}")
    #模型保存策略
    # 定期保存或最终保存
    if epoch ==1 or epoch % args.save_eval_interval == 0 or epoch == args.epochs:
        if is_master:
            logger.info(f"Saving model to {save_dir}")
             # 分布式训练需特殊处理模型保存
            if args.distributed:
                torch.save(best_model.module.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")
            else:
                torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")
            #callback(save_dir, adata, best_model, best_model_epoch) 可选回调
    if debug:
        # results = eval_testdata(
        #     best_model,
        #     adata_t=adata,
        #     include_types=["cls"],
        # )
        break
        # pass

    if args.distributed:
        dist.barrier()
    #print(torch.cuda.max_memory_allocated())
    scheduler.step()
    if is_master:
        print(f'invalid datapoint: {train_dataset.invalid_datapoint_count}')
        train_dataset.invalid_datapoint_count=0

#训练收尾
# %%
# save the best model  最终保存最佳模型
if is_master:
    # 处理分布式模型保存
    if args.distributed:
        torch.save(best_model.module.state_dict(), save_dir / "best_model.pt")
    else:
        torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    # WandB模型归档
    artifact = wandb.Artifact(f"best_model", type="model")
    glob_str = os.path.join(save_dir, "best_model.pt")
    artifact.add_file(glob_str)
    run.log_artifact(artifact)
    # 资源清理
    run.finish()
    wandb.finish()
    gc.collect()

# %% [markdown]
# ## Gene embeddings

