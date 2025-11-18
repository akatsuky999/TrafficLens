"""
统一配置与模型构建函数
=================================

本文件提供：
- 一个可复现实验的默认配置 `get_default_config`；
- 根据配置与数据形状动态构建模型的工厂函数 `build_model`。

训练脚本 `train.py` 与预测脚本 `predict.py` 只需要依赖这里的 API，
便于以后统一修改实验设置。
"""

from typing import Dict, Any
import torch

from .model.LSTM import LSTM
from .model.GWNet import gwnet
from .model.STGformer import STGformer, STGWrapper


def get_default_config() -> Dict[str, Any]:
    """返回一个用于时空预测任务的默认配置字典."""

    # 默认优先使用 cuda:0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg: Dict[str, Any] = {
        "seed": 42,
        "device": device,
        # -------------------- 数据相关 -------------------- #
        "data": {
            # 你的时空数据路径，shape (T, N)
            "data_path": "data/TW.npy",
            "input_steps": 5,   # 历史序列长度 T_in
            "pred_steps": 5,    # 预测序列长度 T_out
            "batch_size": 32,
            # (train_ratio, val_ratio) 相对于所有样本数的比例
            "split_ratios": (0.8, 0.9),
            "shuffle": True,
        },
        # -------------------- 模型相关 -------------------- #
        "model": {
            # 可选: "gwnet" / "lstm" / "stgformer"
            "name": "gwnet",

            # 通用
            "in_dim": 1,

            # LSTM 超参数
            "lstm_hidden_dim": 64,
            "lstm_num_layers": 5,

            # GWNet 超参数
            "gwnet_dropout": 0.3,
            "gwnet_gcn_bool": True,
            "gwnet_addaptadj": True,
            "gwnet_residual_channels": 32,
            "gwnet_dilation_channels": 32,
            "gwnet_skip_channels": 256,
            "gwnet_end_channels": 512,
            "gwnet_kernel_size": 2,
            "gwnet_blocks": 5,
            "gwnet_layers": 2,

            # STGformer 超参数
            "stg_steps_per_day": 288,
            "stg_input_dim": 1,
            "stg_output_dim": 1,
            "stg_input_embedding_dim": 24,
            "stg_tod_embedding_dim": 0,
            "stg_dow_embedding_dim": 0,
            "stg_spatial_embedding_dim": 0,
            "stg_adaptive_embedding_dim": 12,
            "stg_num_heads": 4,
            "stg_num_layers": 3,
            "stg_dropout": 0.1,
            "stg_mlp_ratio": 2.0,
            "stg_use_mixed_proj": True,
            "stg_dropout_a": 0.3,
            "stg_kernel_size": 1,
        },
        # -------------------- 训练相关 -------------------- #
        "train": {
            "epochs": 10000,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "grad_clip": 5.0,       # 0 或 None 表示不裁剪
            "patience": 10,         # 早停
            "loss": "mae",          # 可选: "mae" / "mse"
        },
        # -------------------- 日志与保存 -------------------- #
        "log": {
            "print_every": 50,         # 每多少个 iteration 打印一次训练日志
            "save_dir": "checkpoint",  # 模型保存目录
            "experiment_name": "spatiotemporal_forecast",
        },
    }

    return cfg


def build_model(config: Dict[str, Any],
                num_nodes: int,
                device: torch.device) -> torch.nn.Module:
    """
    根据配置与节点数构建模型。

    Args:
        config: `get_default_config()` 返回的配置或其修改版本
        num_nodes: 数据中的节点数量 N
        device: torch.device
    """
    model_cfg = config["model"]
    data_cfg = config["data"]
    name = model_cfg["name"].lower()

    in_dim = model_cfg["in_dim"]
    pred_steps = data_cfg["pred_steps"]

    if name == "lstm":
        model = LSTM(
            num_nodes=num_nodes,
            in_dim=in_dim,
            hidden_dim=model_cfg["lstm_hidden_dim"],
            num_layers=model_cfg["lstm_num_layers"],
            seq_length=data_cfg["input_steps"],
            pre_len=pred_steps,
        )
    elif name == "gwnet":
        model = gwnet(
            device=device,
            num_nodes=num_nodes,
            dropout=model_cfg["gwnet_dropout"],
            supports=None,               # 默认使用纯自适应邻接
            gcn_bool=model_cfg["gwnet_gcn_bool"],
            addaptadj=model_cfg["gwnet_addaptadj"],
            aptinit=None,
            in_dim=in_dim,
            out_dim=pred_steps,
            residual_channels=model_cfg["gwnet_residual_channels"],
            dilation_channels=model_cfg["gwnet_dilation_channels"],
            skip_channels=model_cfg["gwnet_skip_channels"],
            end_channels=model_cfg["gwnet_end_channels"],
            kernel_size=model_cfg["gwnet_kernel_size"],
            blocks=model_cfg["gwnet_blocks"],
            layers=model_cfg["gwnet_layers"],
        )
    elif name == "stgformer":
        core = STGformer(
            num_nodes=num_nodes,
            in_steps=data_cfg["input_steps"],
            out_steps=pred_steps,
            steps_per_day=model_cfg["stg_steps_per_day"],
            input_dim=model_cfg["stg_input_dim"],
            output_dim=model_cfg["stg_output_dim"],
            input_embedding_dim=model_cfg["stg_input_embedding_dim"],
            tod_embedding_dim=model_cfg["stg_tod_embedding_dim"],
            dow_embedding_dim=model_cfg["stg_dow_embedding_dim"],
            spatial_embedding_dim=model_cfg["stg_spatial_embedding_dim"],
            adaptive_embedding_dim=model_cfg["stg_adaptive_embedding_dim"],
            num_heads=model_cfg["stg_num_heads"],
            supports=None,
            num_layers=model_cfg["stg_num_layers"],
            dropout=model_cfg["stg_dropout"],
            mlp_ratio=model_cfg["stg_mlp_ratio"],
            use_mixed_proj=model_cfg["stg_use_mixed_proj"],
            dropout_a=model_cfg["stg_dropout_a"],
            kernel_size=[model_cfg["stg_kernel_size"]],
        )
        model = STGWrapper(core)
    else:
        raise ValueError(f"Unsupported model name: {name}. Use 'gwnet' or 'lstm'.")

    return model.to(device)
