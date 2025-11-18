from typing import Union, Tuple, Optional
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def _load_raw_array(raw: Union[str, np.ndarray]) -> np.ndarray:
    """Load raw array from path or accept ndarray. Expect shape (T, N)."""
    if isinstance(raw, str):
        if not os.path.exists(raw):
            raise FileNotFoundError(f"{raw} not found")
        ext = os.path.splitext(raw)[1].lower()
        if ext == '.npy':
            arr = np.load(raw)
        elif ext == '.npz':
            loaded = np.load(raw)
            # prefer key 'data' else first key
            if 'data' in loaded:
                arr = loaded['data']
            else:
                keys = list(loaded.keys())
                arr = loaded[keys[0]]
        else:
            raise ValueError("Unsupported file extension. Use .npy or .npz")
    elif isinstance(raw, np.ndarray):
        arr = raw
    else:
        raise ValueError("raw must be file path or numpy ndarray")

    if arr.ndim == 1:
        # interpret as (T,) single node -> convert to (T,1)
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"raw array must be 2D (T, N). Got shape {arr.shape}")
    return arr.astype(np.float32)


def build_dataloaders_single_feature(raw: Union[str, np.ndarray],
                                     input_steps: int = 12,
                                     pred_steps: int = 12,
                                     batch_size: int = 64,
                                     shuffle: bool = True,
                                     DEVICE: Optional[torch.device] = None,
                                     split_ratios: Tuple[float, float] = (0.6, 0.8)
                                     ):
    """
    Args:
        raw: file path (.npy/.npz) or numpy ndarray with shape (T, N)
        input_steps: number of historical time steps used as input
        pred_steps: forecast horizon
        batch_size, shuffle, DEVICE
        split_ratios: (train_end_ratio, val_end_ratio) relative to total samples
    Returns:
        train_loader, val_loader, test_loader, stats_dict, shapes_dict
    """
    # load
    data = _load_raw_array(raw)  # (T, N)
    T, N = data.shape

    # interpret as (T, N, 1) for feature axis
    data3 = data[:, :, None]  # (T, N, 1)

    # sliding windows: label_start runs from input_steps to T - pred_steps
    X_samples = []
    Y_samples = []
    for label_start in range(input_steps, T - pred_steps + 1):
        x_slice = data3[label_start - input_steps: label_start]   # (input_steps, N, 1)
        y_slice = data[label_start: label_start + pred_steps]     # (pred_steps, N)
        # transpose x to (N, F=1, T_in)
        x_sample = x_slice.transpose(1, 2, 0)  # (N, 1, input_steps)
        y_sample = y_slice.transpose(1, 0)     # (N, pred_steps)
        X_samples.append(x_sample)
        Y_samples.append(y_sample)

    if len(X_samples) == 0:
        raise ValueError("No samples created. Check input_steps/pred_steps relative to T")

    X_arr = np.stack(X_samples, axis=0)  # (S, N, 1, T_in)
    Y_arr = np.stack(Y_samples, axis=0)  # (S, N, T_out)

    S = X_arr.shape[0]
    s1 = int(S * split_ratios[0])
    s2 = int(S * split_ratios[1])

    train_X = X_arr[:s1]; val_X = X_arr[s1:s2]; test_X = X_arr[s2:]
    train_Y = Y_arr[:s1]; val_Y = Y_arr[s1:s2]; test_Y = Y_arr[s2:]

    # compute z-score stats on train_X over axes (samples, nodes, time) -> keep dims (1, N, 1, 1)
    mean = train_X.mean(axis=(0, 1, 3), keepdims=True)  # shape (1, N, 1, 1)
    std = train_X.std(axis=(0, 1, 3), keepdims=True)
    std[std == 0] = 1.0

    def zscore(x): return (x - mean) / std

    train_X_norm = zscore(train_X)
    val_X_norm = zscore(val_X)
    test_X_norm = zscore(test_X)

    device = DEVICE if DEVICE is not None else torch.device('cpu')

    train_X_t = torch.from_numpy(train_X_norm).float().to(device)
    train_Y_t = torch.from_numpy(train_Y).float().to(device)
    val_X_t = torch.from_numpy(val_X_norm).float().to(device)
    val_Y_t = torch.from_numpy(val_Y).float().to(device)
    test_X_t = torch.from_numpy(test_X_norm).float().to(device)
    test_Y_t = torch.from_numpy(test_Y).float().to(device)

    train_ds = TensorDataset(train_X_t, train_Y_t)
    val_ds = TensorDataset(val_X_t, val_Y_t)
    test_ds = TensorDataset(test_X_t, test_Y_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    stats = {'mean': mean, 'std': std}  # numpy arrays
    shapes = {
        'raw_shape': (T, N),
        'S_total': S,
        'train_shape': train_X.shape,
        'val_shape': val_X.shape,
        'test_shape': test_X.shape,
        'N_nodes': N,
        'T_in': input_steps,
        'T_out': pred_steps
    }

    return train_loader, val_loader, test_loader, stats, shapes