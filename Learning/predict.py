import argparse
import glob
import os
import random
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils.data_factory import build_dataloaders_single_feature, _load_raw_array
from .model_config import get_default_config, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatiotemporal forecasting – inference script")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to trained model checkpoint (.pth). If empty, automatically use the latest checkpoint for this model.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="prediction.npy",
        help="Path to save predictions (NumPy .npy file)",
    )
    parser.add_argument(
        "--use_test_split",
        action="store_true",
        help="If set, run prediction on the test split defined in data_factory.",
    )
    return parser.parse_args()


def find_latest_checkpoint(cfg: Dict[str, Any]) -> str:
    """查找当前模型在保存目录下（对应子文件夹）最新的 .pth checkpoint."""
    save_root = cfg["log"]["save_dir"]
    model_name = cfg["model"]["name"]
    model_dir = os.path.join(save_root, model_name)

    pattern = os.path.join(model_dir, "*.pth")
    candidates = glob.glob(pattern)

    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {model_dir} for model '{model_name}'.")

    # 优先选择包含 'best' 的文件；如不存在，再从所有文件中选择最新
    best_candidates = [p for p in candidates if "best" in os.path.basename(p)]
    target_list = best_candidates if best_candidates else candidates

    latest = max(target_list, key=os.path.getmtime)
    return latest


@torch.no_grad()
def run_inference(
    cfg: Dict[str, Any],
    checkpoint_path: str,
    output_path: str,
    use_test_split: bool,
    node_idx: Optional[int] = None,
    save_plot: bool = True,
    return_figure: bool = False,
    return_series: bool = False,
    save_outputs: bool = True,
) -> object:
    """
    使用训练好的模型在完整 raw data 上跑一次滑动窗口预测，
    然后在完整时间轴（train+val+test）上对比真实值与预测值。

    注意：DataLoader 仍按 train/val/test 拆分，但这里会顺序遍历三个 split，
    并将所有窗口的预测映射回原始时间轴。
    """
    device = torch.device(cfg["device"])

    data_cfg = cfg["data"]
    train_loader, val_loader, test_loader, stats, shapes = build_dataloaders_single_feature(
        raw=data_cfg["data_path"],
        input_steps=data_cfg["input_steps"],
        pred_steps=data_cfg["pred_steps"],
        batch_size=data_cfg["batch_size"],
        shuffle=False,  # 预测时不需要打乱，保持窗口顺序
        DEVICE=device,
        split_ratios=tuple(data_cfg["split_ratios"]),
    )

    # 载入完整原始序列 (T_raw, N)
    raw_data = _load_raw_array(data_cfg["data_path"])
    T_raw, N_nodes = raw_data.shape

    num_nodes = shapes["N_nodes"]
    T_in = shapes["T_in"]
    horizon = shapes["T_out"]

    model = build_model(cfg, num_nodes=num_nodes, device=device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load checkpoint into model. "
            "Please check that the selected .pth matches "
            "the current model type and hyper-parameters."
        ) from exc
    model.eval()

    # 准备完整时间轴上的预测（对重叠窗口取平均）
    # 形状 (N, T_raw)
    pred_full = np.zeros((N_nodes, T_raw), dtype=np.float32)
    count_full = np.zeros((N_nodes, T_raw), dtype=np.float32)

    S_train = shapes["train_shape"][0]
    S_val = shapes["val_shape"][0]
    # S_test = shapes["test_shape"][0]

    loaders_with_offsets = [
        (train_loader, 0),
        (val_loader, S_train),
        (test_loader, S_train + S_val),
    ]

    for loader, split_offset in loaders_with_offsets:
        current_sample_global = split_offset
        for x, _ in loader:
            x = x.to(device)  # (B, N, 1, T_in)
            pred = model(x)   # (B, N, T_out)
            pred_np = pred.cpu().numpy()

            batch_size = pred_np.shape[0]
            for b in range(batch_size):
                global_s = current_sample_global + b  # 第 global_s 个窗口
                for t in range(horizon):
                    time_idx = T_in + global_s + t
                    if 0 <= time_idx < T_raw:
                        # 对所有节点同时累加
                        pred_full[:, time_idx] += pred_np[b, :, t]
                        count_full[:, time_idx] += 1.0

            current_sample_global += batch_size

    # 对重叠部分取平均
    mask = count_full > 0
    pred_full_avg = np.full_like(pred_full, np.nan, dtype=np.float32)
    pred_full_avg[mask] = pred_full[mask] / count_full[mask]

    # ----------------- 保存完整预测与真实值（可选） ----------------- #
    if save_outputs:
        np.save(output_path, {"pred_full": pred_full_avg, "true_full": raw_data.T})
        print(f"Saved full-sequence predictions to {output_path}")

    # 选择一个节点，在完整 raw 时间轴上画 true vs pred
    if node_idx is None:
        node_idx = random.randint(0, N_nodes - 1)
    if not (0 <= node_idx < N_nodes):
        raise ValueError(f"node_idx {node_idx} is out of range [0, {N_nodes - 1}]")

    true_full_node = raw_data[:, node_idx]            # (T_raw,)
    pred_full_node = pred_full_avg[node_idx, :]       # (T_raw,)

    if return_series:
        return int(node_idx), true_full_node, pred_full_node

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(true_full_node, label="True", linewidth=1.2)
    ax.plot(pred_full_node, label="Pred", linewidth=1.2)
    ax.set_title(f"Node {node_idx} – Full Series")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()

    fig_path = ""
    if save_plot:
        pic_root = "pic"
        model_name = cfg["model"]["name"]
        model_pic_dir = os.path.join(pic_root, model_name)
        os.makedirs(model_pic_dir, exist_ok=True)
        fig_name = os.path.splitext(os.path.basename(output_path))[0] + "_node_full_raw_series.png"
        fig_path = os.path.join(model_pic_dir, fig_name)
        fig.savefig(fig_path, dpi=150)
        print(f"Saved full raw-series comparison plot for node {node_idx} to {fig_path}")

    if return_figure:
        return fig
    else:
        plt.close(fig)
        return fig_path


def main() -> None:
    args = parse_args()
    cfg = get_default_config()

    # 若未显式指定 checkpoint，则自动查找最新的该模型 checkpoint
    checkpoint_path = args.checkpoint or find_latest_checkpoint(cfg)
    print(f"Using checkpoint: {checkpoint_path}")

    run_inference(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        output_path=args.output_path,
        use_test_split=args.use_test_split,
        save_outputs=True,
    )


if __name__ == "__main__":
    main()


