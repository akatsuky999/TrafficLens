from __future__ import annotations

from typing import Dict, Optional

from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path



plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 100,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)

try:
    project_root = Path(__file__).resolve().parent.parent
    fonts_dir = project_root / "Fonts"
    custom_font: Optional[Path] = None

    preferred = fonts_dir / "JetBrainsMono-Regular.ttf"
    if preferred.exists():
        custom_font = preferred
    else:
        for p in fonts_dir.glob("*.ttf"):
            custom_font = p
            break

    if custom_font is not None:
        font_manager.fontManager.addfont(str(custom_font))
        prop = font_manager.FontProperties(fname=str(custom_font))
        plt.rcParams["font.family"] = prop.get_name()
except Exception:
    pass


def overview_stats(df: pd.DataFrame) -> Dict[str, int]:
    if df is None or df.empty:
        return {"total_rows": 0, "total_cols": 0, "distinct_vehicle_types": 0}

    total_rows = int(len(df))
    total_cols = int(len(df.columns))
    distinct_vehicle = int(
        df["VehicleType"].nunique(dropna=True) if "VehicleType" in df.columns else 0
    )
    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "distinct_vehicle_types": distinct_vehicle,
    }



def vehicle_type_pie(df: pd.DataFrame) -> Dict[str, object]:
    if df is None or df.empty or "VehicleType" not in df.columns:
        raise ValueError("VehicleType column not found or dataframe is empty.")

    counts = df["VehicleType"].value_counts(dropna=False).sort_values(ascending=False)
    if counts.empty:
        raise ValueError("No valid data in VehicleType column.")

    fig, ax = plt.subplots(figsize=(5, 5))
    counts.plot(
        kind="pie",
        ax=ax,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        pctdistance=0.8,
    )
    ax.set_ylabel("")
    ax.set_title("VehicleType distribution")

    total = int(counts.sum())
    n_unique = int(counts.size)
    top_counts = counts.head(5).to_dict()
    text_lines = [f"Total: {total}", f"Unique: {n_unique}", "Top classes:"]
    text_lines += [f"{k}: {v}" for k, v in top_counts.items()]
    ax.text(
        1.05,
        0.5,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="center",
        fontsize=9,
    )

    fig.tight_layout()

    return {
        "stats": {
            "total": total,
            "n_unique": n_unique,
            "top_counts": top_counts,
        },
        "figure": fig,
    }



def time_hist_5min(df: pd.DataFrame, col: str) -> Dict[str, object]:
    if df is None or df.empty or col not in df.columns:
        raise ValueError(f"Column {col} not found or dataframe is empty.")

    dt = pd.to_datetime(df[col], errors="coerce")
    dt = dt.dropna()
    if dt.empty:
        raise ValueError(f"No valid datetime values in column {col}.")

    bins = dt.dt.floor("5min")
    counts = bins.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index.astype("datetime64[ns]"), counts.values, width=4 / (24 * 60))
    ax.set_title(f"{col} frequency per 5-minute interval")
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    fig.autofmt_xdate()
    stats = {
        "count": int(len(dt)),
        "start": dt.min(),
        "end": dt.max(),
    }
    text_lines = [
        f"Count: {stats['count']}",
        f"Start: {stats['start']}",
        f"End:   {stats['end']}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    fig.tight_layout()

    return {
        "stats": stats,
        "figure": fig,
    }



def trip_length_hist(df: pd.DataFrame, bins: int = 30) -> Optional[Dict[str, object]]:
    if df is None or df.empty or "TripLength" not in df.columns:
        return None

    ser = pd.to_numeric(df["TripLength"], errors="coerce").dropna()
    if ser.empty:
        return None

    bins = max(bins, 60)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ser.values, bins=bins, edgecolor="black", linewidth=0.5)
    ax.set_title("TripLength distribution")
    ax.set_xlabel("TripLength")
    ax.set_ylabel("Frequency")

    stats = {
        "count": float(ser.count()),
        "min": float(ser.min()),
        "mean": float(ser.mean()),
        "std": float(ser.std(ddof=1)) if ser.count() > 1 else 0.0,
        "max": float(ser.max()),
    }

    text_lines = [
        f"n = {stats['count']:.0f}",
        f"min = {stats['min']:.2f}",
        f"mean = {stats['mean']:.2f}",
        f"std = {stats['std']:.2f}",
        f"max = {stats['max']:.2f}",
    ]
    ax.text(
        0.98,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    fig.tight_layout()

    return {
        "stats": stats,
        "figure": fig,
    }



def gantry_code_bar(df: pd.DataFrame, col: str, top_n: int = 30) -> Dict[str, object]:
    if df is None or df.empty or col not in df.columns:
        raise ValueError(f"Column {col} not found or dataframe is empty.")

    counts = df[col].astype(str).value_counts(dropna=False)
    if counts.empty:
        raise ValueError(f"No valid data in column {col}.")

    top_counts = counts.head(top_n)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.bar(top_counts.index, top_counts.values, edgecolor="black", linewidth=0.6)
    ax.set_title(f"{col} frequency (top {top_n})")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="x", rotation=60)

    stats = {
        "n_unique": int(counts.size),
        "top_counts": top_counts.to_dict(),
    }

    text_lines = [
        f"Unique codes: {stats['n_unique']}",
        f"Top shown: {len(top_counts)}",
    ]
    ax.text(
        0.98,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    fig.tight_layout()

    return {
        "stats": stats,
        "figure": fig,
    }



def flow_timeseries(st_df: pd.DataFrame, node_index: int) -> Dict[str, object]:
    if st_df is None or st_df.empty:
        raise ValueError("Spatio-temporal dataframe is empty.")

    if not isinstance(st_df.index, pd.DatetimeIndex):
        try:
            st_df = st_df.copy()
            st_df.index = pd.to_datetime(st_df.index, errors="coerce")
            if st_df.index.isna().all():
                raise ValueError
        except Exception:
            raise ValueError("Index of spatio-temporal dataframe is not datetime-like.")

    cols = sorted([str(c) for c in st_df.columns])
    if node_index < 1 or node_index > len(cols):
        raise ValueError(f"node_index must be in [1, {len(cols)}].")

    node_name = cols[node_index - 1]
    ser = st_df[node_name].astype(float)

    stats = {
        "count": float(ser.count()),
        "sum": float(ser.sum()),
        "min": float(ser.min()),
        "mean": float(ser.mean()),
        "max": float(ser.max()),
    }

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(st_df.index, ser.values, color="#1f77b4", linewidth=1.2)
    ax.set_title(f"Flow time series - node {node_name} (index {node_index})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Flow count")
    fig.autofmt_xdate()

    text_lines = [
        f"n = {stats['count']:.0f}",
        f"sum = {stats['sum']:.0f}",
        f"min = {stats['min']:.0f}",
        f"mean = {stats['mean']:.2f}",
        f"max = {stats['max']:.0f}",
    ]
    ax.text(
        0.98,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    fig.tight_layout()

    return {
        "node": node_name,
        "stats": stats,
        "figure": fig,
    }


def st_3d_surface(
    st_df: pd.DataFrame, max_time_bins: int = 60, max_nodes: int = 40
) -> Dict[str, object]:
    if st_df is None or st_df.empty:
        raise ValueError("Spatio-temporal dataframe is empty.")

    if not isinstance(st_df.index, pd.DatetimeIndex):
        try:
            st_df = st_df.copy()
            st_df.index = pd.to_datetime(st_df.index, errors="coerce")
            if st_df.index.isna().all():
                raise ValueError
        except Exception:
            raise ValueError("Index of spatio-temporal dataframe is not datetime-like.")

    st = st_df.sort_index()
    cols = sorted([str(c) for c in st.columns])
    st = st[cols]

    if st.empty:
        raise ValueError("Spatio-temporal dataframe subset is empty.")

    z = st.to_numpy(dtype=float)
    n_time, n_nodes = z.shape

    x_old = np.arange(n_nodes)
    y_old = np.arange(n_time)

    up_x = min(n_nodes * 3, 200)
    up_y = min(n_time * 3, 200)

    if n_nodes > 1 and n_time > 1 and up_x > 1 and up_y > 1:
        x_new = np.linspace(0, n_nodes - 1, up_x)
        y_new = np.linspace(0, n_time - 1, up_y)

        z_tmp = np.empty((n_time, up_x), dtype=float)
        for i in range(n_time):
            z_tmp[i, :] = np.interp(x_new, x_old, z[i, :])

        z_smooth = np.empty((up_y, up_x), dtype=float)
        for j in range(up_x):
            z_smooth[:, j] = np.interp(y_new, y_old, z_tmp[:, j])

        kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float)
        kernel /= kernel.sum()

        for i in range(up_y):
            z_smooth[i, :] = np.convolve(z_smooth[i, :], kernel, mode="same")

        for j in range(up_x):
            z_smooth[:, j] = np.convolve(z_smooth[:, j], kernel, mode="same")

        X, Y = np.meshgrid(x_new, y_new)
        z_plot = z_smooth
    else:
        X, Y = np.meshgrid(x_old, y_old)
        z_plot = z

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X,
        Y,
        z_plot,
        cmap="viridis",
        edgecolor="none",
        antialiased=True,
        linewidth=0,
    )
    ax.set_xlabel("Node index")
    ax.set_ylabel("Time index")
    ax.set_zlabel("Flow count")
    ax.set_title(
        f"Spatio-temporal flow (full: {n_time} time bins x {n_nodes} nodes)"
    )

    try:
        ax.contour(
            X,
            Y,
            z_plot,
            zdir="z",
            offset=z_plot.min(),
            cmap="viridis",
            linewidths=0.3,
        )
    except Exception:
        pass
    fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.1)

    fig.tight_layout()

    return {
        "shape": (n_time, n_nodes),
        "figure": fig,
    }


