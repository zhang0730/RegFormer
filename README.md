# RegFormer

## Introduction
This is the official codebase for **RegFormer**.

## [ckpt] Newly released
**Note:** For running BiMamba (bidirectional mamba), you need to additionally set 'bimamba_type=v1'.

## [Log] Updates
- [2024/03/07] Attention-based aggregation for cell embeddings; revision for Bidirectional Mamba Pretraining script.
- [2024/03/06] Bidirectional Mamba for MLM pretrainig.
- [2024/03/01] Integration module has completed. Note that distributed pipeline (DDP) for integration is unavailable, left for future update.

## [Optional] Usage of wandb
We recommend using [wandb](https://wandb.ai/) for logging and visualization.

```bash
$ pip install wandb
```
In the case of first run in your device, you need to login with your API key of [W&B](https://wandb.ai/home).

```bash
wandb login
```

## [Optional] Flash-attention

**Note**: The `flash-attn` dependency usually requires specific GPU and CUDA version. If you encounter any issues, please refer to the [flash-attn](https://github.com/HazyResearch/flash-attention/tree/main) repository for installation instructions. For now, May 2023, we recommend using CUDA 11.7 and flash-attn<1.0.5 due to various issues reported about installing new versions of flash-attn.

You can turn off flash-attn by setting *fast_transformer=False* in your command.
```bash
python3 srcipt.py --fast_transformer False
```

For running cell annotation for example:
```bash
python3 regformer/mamba_CA.py --model_name mamba --epochs 15 \
--cell_emb_style avg-pool --batch_size 64 --data_name your_data \
--load_model your_model_checkpoint \ 
# train from scratch if ckpt is not provided

--save_dir your_save_dir
--run_name your_run_name
# the result will be save in: your_save_dir/your_model_name/your_data_name/your_run_name

--single_gpu True
--distributed True
# Note: single_gpu and distribute shouldn't be set to True at the same time, you have to choose one from them.
# If both of them are set to False, this script will jump into debug mode.
```


## Fine-tune scMamba for Cell Annotation **[For Developer]**

Please see example pipeline for Cell Annotation in [regformer/mamba_CA.py](downstream_task/mamba_CA.py).

Set *single_gpu=False, distributed=False* for jumping into the debug mode.

For developing new downstream task, you also need to implement corresponding module in the [dataset](regformer/data/dataset.py) and [dataloader](regformer/data/dataloader.py).

