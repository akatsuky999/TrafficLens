"""
Early stopping utility
======================

典型的早停实现，用于在验证集指标不再提升时提前终止训练并保存最佳模型。
"""

import os
from typing import Optional

import numpy as np
import torch


class EarlyStopping:
    """
    Args:
        patience: 容忍验证指标不提升的 epoch 数
        verbose: 是否打印每次改进的信息
        delta: 最小改进幅度，小于该幅度视为没有改进
        path: 最佳模型保存路径；若为 None，则不自动保存
        mode: "min" 表示指标越小越好 (如 loss, MAE)，"max" 表示越大越好 (如 accuracy)
    """

    def __init__(
        self,
        patience: int = 10,
        verbose: bool = True,
        delta: float = 0.0,
        path: Optional[str] = None,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_value = np.inf if mode == "min" else -np.inf

    def __call__(self, value: float, model: torch.nn.Module) -> None:
        """
        更新早停状态。

        Args:
            value: 当前 epoch 的验证指标（例如 val_loss）
            model: 当前模型（用于在指标提升时保存）
        """
        score = -value if self.mode == "min" else value

        if self.best_score is None:
            self.best_score = score
            self.best_value = value
            self._save_checkpoint(value, model)
        elif score < self.best_score + self.delta:
            # 没有显著提升
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 指标有显著提升
            self.best_score = score
            self.best_value = value
            self._save_checkpoint(value, model)
            self.counter = 0

    def _save_checkpoint(self, value: float, model: torch.nn.Module) -> None:
        """保存当前最佳模型."""
        if self.path is None:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if self.verbose:
            print(f"Validation metric improved to {value:.6f}. Saving model to {self.path}")
        torch.save(model.state_dict(), self.path)


