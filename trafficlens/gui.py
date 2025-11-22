from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
import threading

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from .config import DEFAULT_COLUMNS
from .data_loader import TrafficDataStore
from .stats import (
    overview_stats,
    vehicle_type_pie,
    time_hist_5min,
    trip_length_hist,
    gantry_code_bar,
    flow_timeseries,
    st_3d_surface,
)
from .view_filters import build_view_hint, apply_view_filter, get_field_kind
from .ST_generator import generate_spatiotemporal, generate_spatiotemporal_from_tripinfo
from Learning.model_config import get_default_config
from Learning.train import run_training
from Learning.predict import run_inference

class _DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


logger = _DummyLogger()


class TrafficLensApp(tk.Tk):
    def __init__(self, store: Optional[TrafficDataStore] = None) -> None:
        super().__init__()
        self.title("TrafficLens - Traffic Data Explorer for Taiwan Highway")
        self.geometry("1100x600")
        self.minsize(10, 10)

        self.store: Optional[TrafficDataStore] = store
        if self.store is not None:
            self.current_df: pd.DataFrame = self.store.dataframe.copy()
        else:
            self.current_df = pd.DataFrame(columns=DEFAULT_COLUMNS)

        self.page_size: int = 1000
        self.current_page: int = 1

        self.sorted_col: Optional[str] = None
        self.sorted_ascending: bool = True
        self.selected_sort_col: str = DEFAULT_COLUMNS[0]
        self.sort_col_label_var = tk.StringVar(value=DEFAULT_COLUMNS[0])

        self.page_input_var: tk.StringVar | None = None

        self.search_var = tk.StringVar()
        self.search_scope_var = tk.StringVar(value="All fields")
        self.search_mode_var = tk.StringVar(value="Fuzzy")

        self.toolbar_mode = tk.StringVar(value="Actions")

        self.table_columns = list(DEFAULT_COLUMNS)

        self.st_df: Optional[pd.DataFrame] = None
        self.st_prev_df: Optional[pd.DataFrame] = None
        self.st_prev_page: int = 1
        self.st_shape_info: str = ""

        self.plot_canvas: FigureCanvasTkAgg | None = None
        self.learning_log_widget: tk.Text | None = None
        self.learning_model_overrides: dict[str, dict[str, object]] = {}
        self.learning_view_mode: str = "log"
        self.learning_stop_training: bool = False
        self.learning_checkpoint_path: str | None = None

        self.last_search_keyword: Optional[str] = None
        self.last_search_column: Optional[str] = None
        self.last_search_strict: bool = False
        self.last_match_count: Optional[int] = None
        self._build_widgets()
        self._refresh_view()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        logger.info("TrafficLens GUI started with no data loaded.")

    def _build_widgets(self) -> None:

        tab_frame = ttk.Frame(self)
        tab_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 0))

        self.tab_style = ttk.Style(self)
        self.tab_style.configure(
            "Tab.TButton",
            padding=(10, 4),
            font=("Segoe UI", 9),
        )
        self.tab_style.map(
            "Tab.TButton",
            background=[("active", "#E5F1FB")],
        )
        self.tab_style.configure(
            "TabActive.TButton",
            padding=(10, 4),
            font=("Segoe UI", 9, "bold"),
            background="#0078D7",
            foreground="#000000",
            relief="sunken",
        )
        self.tab_style.map(
            "TabActive.TButton",
            background=[("!disabled", "#0078D7")],
            foreground=[("!disabled", "#000000")],
        )
        self.tab_style.configure(
            "GlobalBack.TButton",
            padding=(10, 4),
            foreground="#d9534f",
        )

        self.file_tab_btn = ttk.Button(
            tab_frame,
            text="File",
            command=lambda: self._on_tab_click("File"),
            width=8,
        )
        self.file_tab_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.action_tab_btn = ttk.Button(
            tab_frame,
            text="Actions",
            command=lambda: self._on_tab_click("Actions"),
            width=8,
        )
        self.action_tab_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.view_tab_btn = ttk.Button(
            tab_frame,
            text="View",
            command=lambda: self._on_tab_click("View"),
            width=8,
        )
        self.view_tab_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.st_tab_btn = ttk.Button(
            tab_frame,
            text="Spatio-temporal",
            command=lambda: self._on_tab_click("Spatio-temporal"),
            width=16,
        )
        self.st_tab_btn.pack(side=tk.LEFT)

        self.learning_tab_btn = ttk.Button(
            tab_frame,
            text="Learning",
            command=lambda: self._on_tab_click("Learning"),
            width=10,
        )
        self.learning_tab_btn.pack(side=tk.LEFT, padx=(4, 0))

        self.global_back_btn = ttk.Button(
            tab_frame,
            text="Back",
            command=self.on_close_st_view,
            style="GlobalBack.TButton",
            width=8,
        )
        self.global_back_btn.pack(side=tk.RIGHT)

        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 4))
        self._rebuild_toolbar()

        self.table_frame = ttk.Frame(self)
        self.table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        style = ttk.Style(self)
        try:
            if "vista" in style.theme_names():
                style.theme_use("vista")
        except Exception:
            pass

        style.configure(
            "Excel.Treeview",
            background="white",
            foreground="#202020",
            rowheight=22,
            fieldbackground="white",
            font=("Segoe UI", 9),
        )
        style.configure(
            "Excel.Treeview.Heading",
            font=("Segoe UI", 9, "bold"),
            background="#F3F3F3",
            foreground="#202020",
        )
        style.map(
            "Excel.Treeview",
            background=[("selected", "#CCE8FF")],
            foreground=[("selected", "#000000")],
        )

        self.tree = ttk.Treeview(
            self.table_frame,
            columns=self.table_columns,
            show="headings",
            height=20,
            style="Excel.Treeview",
        )

        self._configure_table_columns()

        vsb = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.tag_configure("evenrow", background="#FFFFFF")
        self.tree.tag_configure("oddrow", background="#F7F7F7")

        self.tree.bind("<ButtonRelease-1>", self.on_tree_click)

        self.tree.tag_configure("matchrow", background="#FFF4CC")

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        self.table_frame.rowconfigure(0, weight=1)
        self.table_frame.columnconfigure(0, weight=1)

        self.plot_frame = ttk.Frame(self)

        status_frame = ttk.Frame(self)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 8))

        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)


        self.page_input_var = tk.StringVar(value="1")
        ttk.Label(status_frame, text="  Go to page").pack(side=tk.LEFT)
        page_entry = ttk.Entry(status_frame, textvariable=self.page_input_var, width=5)
        page_entry.pack(side=tk.LEFT)
        ttk.Button(status_frame, text="Go", command=self.on_jump_page).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        self.st_shape_label_var = tk.StringVar()
        self.st_shape_label = ttk.Label(
            status_frame, textvariable=self.st_shape_label_var
        )
        self.st_shape_label.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Button(status_frame, text="Prev", command=self.on_prev_page).pack(
            side=tk.RIGHT, padx=(4, 0)
        )
        ttk.Button(status_frame, text="Next", command=self.on_next_page).pack(
            side=tk.RIGHT
        )

    def _get_page_df(self) -> pd.DataFrame:
        """Return the slice of current_df for the current page."""
        total = len(self.current_df)
        if total == 0:
            self.current_page = 1
            return self.current_df

        max_page = max((total - 1) // self.page_size + 1, 1)
        if self.current_page > max_page:
            self.current_page = max_page
        if self.current_page < 1:
            self.current_page = 1

        start = (self.current_page - 1) * self.page_size
        end = start + self.page_size
        return self.current_df.iloc[start:end]

    def _populate_table(self, df: pd.DataFrame) -> None:
        self.tree.delete(*self.tree.get_children())
        keyword = (self.last_search_keyword or "").strip()
        highlight = bool(keyword)
        strict = self.last_search_strict
        column = self.last_search_column

        for idx, (_, row) in enumerate(df.iterrows()):
            values = [row.get(col, "") for col in self.table_columns]
            tags = ["evenrow" if idx % 2 == 0 else "oddrow"]

            if highlight:
                if column:
                    cell_val = row.get(column, "")
                    cell_str = "" if cell_val is None else str(cell_val)
                    if strict:
                        is_match = cell_str == keyword
                    else:
                        is_match = keyword.lower() in cell_str.lower()
                else:
                    is_match = False
                    for col in DEFAULT_COLUMNS:
                        cell_val = row.get(col, "")
                        cell_str = "" if cell_val is None else str(cell_val)
                        if strict:
                            if cell_str == keyword:
                                is_match = True
                                break
                        else:
                            if keyword.lower() in cell_str.lower():
                                is_match = True
                                break

                if is_match:
                    tags.append("matchrow")

            self.tree.insert("", tk.END, values=values, tags=tags)

    def _refresh_view(self) -> None:
        """Refresh table based on current_df and current page."""
        page_df = self._get_page_df()
        self._populate_table(page_df)
        self._update_status()

    def _update_status(self) -> None:
        total = len(self.current_df)
        if total == 0:
            text = "No data"
        else:
            max_page = max((total - 1) // self.page_size + 1, 1)
            text = (
                f"Page {self.current_page}/{max_page}, "
                f"{self.page_size} rows per page, total {total} rows"
            )
            if self.last_match_count is not None and self.last_search_keyword:
                text += f"; search matched {self.last_match_count} rows"

        self.status_var.set(text)

        self.st_shape_label_var.set(self.st_shape_info)

        if self.page_input_var is not None:
            self.page_input_var.set(str(self.current_page))

    def _ensure_data_loaded(self) -> bool:
        """Return True if data is available, otherwise show a hint and return False."""
        if self.store is None or self.store.dataframe.empty:
            messagebox.showinfo("Info", "Please import data first in the 'File' tab.")
            return False
        return True

    def _clear_toolbar(self) -> None:
        for child in self.toolbar_frame.winfo_children():
            child.destroy()

    def _build_file_toolbar(self) -> None:
        """Toolbar content for the 'File' tab: import / merge / export / reset."""
        self._clear_toolbar()
        ttk.Button(
            self.toolbar_frame, text="Import file", command=self.on_import_file
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            self.toolbar_frame, text="Import folder", command=self.on_import_folder
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            self.toolbar_frame, text="Merge files", command=self.on_merge
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            self.toolbar_frame, text="Merge folder", command=self.on_merge_folder
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            self.toolbar_frame, text="Export current result", command=self.on_export
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            self.toolbar_frame, text="Reset to original order", command=self.on_reset
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            self.toolbar_frame, text="Clear all data", command=self.on_clear_data
        ).pack(side=tk.LEFT, padx=(8, 0))

    def _build_action_toolbar(self) -> None:
        """Toolbar content for the 'Actions' tab: search and sort."""
        self._clear_toolbar()

        ttk.Label(self.toolbar_frame, text="Keyword:").pack(side=tk.LEFT)
        search_entry = ttk.Entry(
            self.toolbar_frame, textvariable=self.search_var, width=24
        )
        search_entry.pack(side=tk.LEFT, padx=(4, 4))
        search_entry.bind("<Return>", lambda _e: self.on_search())
        ttk.Button(self.toolbar_frame, text="Search", command=self.on_search).pack(
            side=tk.LEFT, padx=(0, 8)
        )

        ttk.Label(self.toolbar_frame, text="Scope:").pack(side=tk.LEFT)
        search_scope_values = ["All fields"] + DEFAULT_COLUMNS
        scope_combo = ttk.Combobox(
            self.toolbar_frame,
            textvariable=self.search_scope_var,
            values=search_scope_values,
            state="readonly",
            width=12,
        )
        scope_combo.pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(self.toolbar_frame, text="Mode:").pack(side=tk.LEFT)
        mode_combo = ttk.Combobox(
            self.toolbar_frame,
            textvariable=self.search_mode_var,
            values=["Fuzzy", "Exact"],
            state="readonly",
            width=6,
        )
        mode_combo.pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(self.toolbar_frame, text="Sort column:").pack(side=tk.LEFT)
        ttk.Label(
            self.toolbar_frame, textvariable=self.sort_col_label_var
        ).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Button(
            self.toolbar_frame,
            text="Ascending",
            command=lambda: self.on_sort_button(True),
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            self.toolbar_frame,
            text="Descending",
            command=lambda: self.on_sort_button(False),
        ).pack(side=tk.LEFT, padx=(0, 4))

    def _build_view_toolbar(self) -> None:
        """Toolbar content for the 'View' tab: field selection, value range filter, and statistics plot."""
        self._clear_toolbar()

        top_row = ttk.Frame(self.toolbar_frame)
        top_row.pack(side=tk.TOP, fill=tk.X)
        bottom_row = ttk.Frame(self.toolbar_frame)
        bottom_row.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        ttk.Label(top_row, text="Field:").pack(side=tk.LEFT)
        self.view_field_var = tk.StringVar(value="DetectionTime_O")
        field_combo = ttk.Combobox(
            top_row,
            textvariable=self.view_field_var,
            values=[
                "DetectionTime_O",
                "DetectionTime_D",
                "VehicleType",
                "TripLength",
            ],
            state="readonly",
            width=15,
        )
        field_combo.pack(side=tk.LEFT, padx=(4, 8))
        field_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_view_controls())

        self.view_controls_frame = ttk.Frame(top_row)
        self.view_controls_frame.pack(side=tk.LEFT, padx=(4, 4))

        self.view_value_var = tk.StringVar()
        self.view_from_var = tk.StringVar()
        self.view_to_var = tk.StringVar()

        ttk.Button(
            top_row,
            text="Apply filter",
            command=self.on_apply_view_filter,
        ).pack(side=tk.LEFT, padx=(4, 4))
        ttk.Button(
            top_row,
            text="Clear filter",
            command=self.on_clear_view_filter,
        ).pack(side=tk.LEFT, padx=(0, 4))

        self.view_info_var = tk.StringVar()
        ttk.Label(top_row, textvariable=self.view_info_var).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        ttk.Label(bottom_row, text="Stats field:").pack(side=tk.LEFT)
        self.stats_field_var = tk.StringVar(value="VehicleType")
        stats_combo = ttk.Combobox(
            bottom_row,
            textvariable=self.stats_field_var,
            values=DEFAULT_COLUMNS,
            state="readonly",
            width=18,
        )
        stats_combo.pack(side=tk.LEFT, padx=(4, 8))

        ttk.Button(
            bottom_row,
            text="Plot statistics",
            command=self.on_plot_stats,
        ).pack(side=tk.LEFT, padx=(0, 4))

        self._update_view_controls()

    def _build_st_toolbar(self) -> None:
        """Toolbar content for the 'Spatio-temporal' tab: choose time column, node column and time bin size."""
        self._clear_toolbar()

        top_row = ttk.Frame(self.toolbar_frame)
        top_row.pack(side=tk.TOP, fill=tk.X)
        bottom_row = ttk.Frame(self.toolbar_frame)
        bottom_row.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        ttk.Label(top_row, text="Time column:").pack(side=tk.LEFT)
        self.st_time_col_var = tk.StringVar(value="DetectionTime_O")
        time_cols = [c for c in DEFAULT_COLUMNS if "Time" in c or "time" in c]
        if not time_cols:
            time_cols = DEFAULT_COLUMNS
        time_combo = ttk.Combobox(
            top_row,
            textvariable=self.st_time_col_var,
            values=time_cols,
            state="readonly",
            width=18,
        )
        time_combo.pack(side=tk.LEFT, padx=(4, 8))

        ttk.Label(top_row, text="Node column:").pack(side=tk.LEFT)
        self.st_node_col_var = tk.StringVar(value="GantryID_O")
        node_cols = [c for c in DEFAULT_COLUMNS if "GantryID" in c or "ID" in c]
        if not node_cols:
            node_cols = DEFAULT_COLUMNS
        node_combo = ttk.Combobox(
            top_row,
            textvariable=self.st_node_col_var,
            values=node_cols,
            state="readonly",
            width=18,
        )
        node_combo.pack(side=tk.LEFT, padx=(4, 8))

        ttk.Label(bottom_row, text="Bin size (min):").pack(side=tk.LEFT)
        self.st_freq_var = tk.StringVar(value="5")
        freq_entry = ttk.Entry(
            bottom_row, textvariable=self.st_freq_var, width=6
        )
        freq_entry.pack(side=tk.LEFT, padx=(2, 8))

        ttk.Button(
            bottom_row, text="Generate O/D ST matrix", command=self.on_generate_st
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            bottom_row,
            text="Generate trajectory ST matrix",
            command=self.on_generate_st_traj,
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(bottom_row, text="Node index:").pack(side=tk.LEFT)
        self.st_node_index_var = tk.StringVar(value="")
        node_idx_entry = ttk.Entry(
            bottom_row, textvariable=self.st_node_index_var, width=8
        )
        node_idx_entry.pack(side=tk.LEFT, padx=(2, 4))

        ttk.Button(
            bottom_row, text="Flow plot", command=self.on_plot_flow
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            bottom_row, text="3D plot", command=self.on_plot_st_3d
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.st_info_var = tk.StringVar()
        ttk.Label(bottom_row, textvariable=self.st_info_var).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        third_row = ttk.Frame(self.toolbar_frame)
        third_row.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))
        ttk.Button(
            third_row,
            text="Load ST file and 3D plot",
            command=self.on_load_st_3d_file,
        ).pack(side=tk.LEFT, padx=(0, 8))

    def _build_learning_toolbar(self) -> None:
        self._clear_toolbar()

        top_row = ttk.Frame(self.toolbar_frame)
        top_row.pack(side=tk.TOP, fill=tk.X)
        middle_row = ttk.Frame(self.toolbar_frame)
        middle_row.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))
        bottom_row = ttk.Frame(self.toolbar_frame)
        bottom_row.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        ttk.Label(top_row, text="Model:").pack(side=tk.LEFT)
        self.learning_model_var = tk.StringVar(value="gwnet")
        model_combo = ttk.Combobox(
            top_row,
            textvariable=self.learning_model_var,
            values=["gwnet", "lstm", "stgformer"],
            state="readonly",
            width=10,
        )
        model_combo.pack(side=tk.LEFT, padx=(4, 8))

        ttk.Label(top_row, text="Device:").pack(side=tk.LEFT)
        self.learning_device_var = tk.StringVar(value="Auto")
        device_combo = ttk.Combobox(
            top_row,
            textvariable=self.learning_device_var,
            values=["Auto", "CPU only"],
            state="readonly",
            width=10,
        )
        device_combo.pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(top_row, text="Data path:").pack(side=tk.LEFT)
        self.learning_data_path_var = tk.StringVar(value="Learning/data/TW.npy")
        data_entry = ttk.Entry(
            top_row, textvariable=self.learning_data_path_var, width=32
        )
        data_entry.pack(side=tk.LEFT, padx=(4, 4))
        ttk.Button(
            top_row, text="Browse", command=self.on_learning_browse_data
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(top_row, text="Input steps:").pack(side=tk.LEFT)
        self.learning_input_steps_var = tk.StringVar(value="5")
        ttk.Entry(
            top_row, textvariable=self.learning_input_steps_var, width=4
        ).pack(side=tk.LEFT, padx=(2, 4))

        ttk.Label(top_row, text="Pred steps:").pack(side=tk.LEFT)
        self.learning_pred_steps_var = tk.StringVar(value="5")
        ttk.Entry(
            top_row, textvariable=self.learning_pred_steps_var, width=4
        ).pack(side=tk.LEFT, padx=(2, 4))

        ttk.Label(middle_row, text="Mode:").pack(side=tk.LEFT)
        self.learning_mode_var = tk.StringVar(value="Fixed epochs")
        mode_combo = ttk.Combobox(
            middle_row,
            textvariable=self.learning_mode_var,
            values=["Fixed epochs", "Early-stopping only"],
            state="readonly",
            width=18,
        )
        mode_combo.pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(middle_row, text="Epochs:").pack(side=tk.LEFT)
        self.learning_epochs_var = tk.StringVar(value="100")
        ttk.Entry(
            middle_row, textvariable=self.learning_epochs_var, width=6
        ).pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(middle_row, text="Patience:").pack(side=tk.LEFT)
        self.learning_patience_var = tk.StringVar(value="10")
        ttk.Entry(
            middle_row, textvariable=self.learning_patience_var, width=6
        ).pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(middle_row, text="Batch size:").pack(side=tk.LEFT)
        self.learning_batch_size_var = tk.StringVar(value="32")
        ttk.Entry(
            middle_row, textvariable=self.learning_batch_size_var, width=6
        ).pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(middle_row, text="Learning rate:").pack(side=tk.LEFT)
        self.learning_lr_var = tk.StringVar(value="0.001")
        ttk.Entry(
            middle_row, textvariable=self.learning_lr_var, width=8
        ).pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(bottom_row, text="Node index:").pack(side=tk.LEFT)
        self.learning_node_index_var = tk.StringVar(value="22")
        ttk.Entry(
            bottom_row, textvariable=self.learning_node_index_var, width=6
        ).pack(side=tk.LEFT, padx=(2, 4))

        ttk.Button(
            bottom_row, text="Load .pth", command=self.on_learning_load_checkpoint
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            bottom_row, text="Predict", command=self.on_learning_predict
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            bottom_row, text="Save outputs", command=self.on_learning_save_outputs
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            bottom_row, text="Train", command=self.on_learning_train
        ).pack(side=tk.LEFT, padx=(8, 4))

        ttk.Button(
            bottom_row, text="Stop", command=self.on_learning_stop
        ).pack(side=tk.LEFT, padx=(0, 4))

        ttk.Button(
            bottom_row, text="Hyper-parameters", command=self.on_learning_hyperparams
        ).pack(side=tk.LEFT, padx=(0, 4))

        if self.learning_view_mode == "log":
            self._ensure_learning_log_view()

    def _rebuild_toolbar(self) -> None:
        mode = self.toolbar_mode.get()
        if mode == "File":
            self._build_file_toolbar()
        elif mode == "Actions":
            self._build_action_toolbar()
        elif mode == "View":
            self._build_view_toolbar()
        elif mode == "Spatio-temporal":
            self._build_st_toolbar()
        else:
            self._build_learning_toolbar()

        self._update_tab_styles()

    def _on_tab_click(self, mode: str) -> None:
        """Handle clicks on the tab buttons."""
        if self.toolbar_mode.get() == mode:
            return
        self.toolbar_mode.set(mode)
        self._rebuild_toolbar()

    def _update_tab_styles(self) -> None:
        """Refresh tab button styles so that the active tab is highlighted."""
        current = self.toolbar_mode.get()
        if current == "File":
            self.file_tab_btn.configure(style="TabActive.TButton")
            self.action_tab_btn.configure(style="Tab.TButton")
            self.view_tab_btn.configure(style="Tab.TButton")
            self.st_tab_btn.configure(style="Tab.TButton")
            self.learning_tab_btn.configure(style="Tab.TButton")
        elif current == "Actions":
            self.file_tab_btn.configure(style="Tab.TButton")
            self.action_tab_btn.configure(style="TabActive.TButton")
            self.view_tab_btn.configure(style="Tab.TButton")
            self.st_tab_btn.configure(style="Tab.TButton")
            self.learning_tab_btn.configure(style="Tab.TButton")
        elif current == "View":
            self.file_tab_btn.configure(style="Tab.TButton")
            self.action_tab_btn.configure(style="Tab.TButton")
            self.view_tab_btn.configure(style="TabActive.TButton")
            self.st_tab_btn.configure(style="Tab.TButton")
            self.learning_tab_btn.configure(style="Tab.TButton")
        elif current == "Spatio-temporal":
            self.file_tab_btn.configure(style="Tab.TButton")
            self.action_tab_btn.configure(style="Tab.TButton")
            self.view_tab_btn.configure(style="Tab.TButton")
            self.st_tab_btn.configure(style="TabActive.TButton")
            self.learning_tab_btn.configure(style="Tab.TButton")
        else:
            self.file_tab_btn.configure(style="Tab.TButton")
            self.action_tab_btn.configure(style="Tab.TButton")
            self.view_tab_btn.configure(style="Tab.TButton")
            self.st_tab_btn.configure(style="Tab.TButton")
            self.learning_tab_btn.configure(style="TabActive.TButton")

    def _build_learning_config(self) -> dict:
        cfg = get_default_config()
        model_name = self.learning_model_var.get().strip().lower()
        if model_name:
            cfg["model"]["name"] = model_name

        data_path = self.learning_data_path_var.get().strip()
        if data_path:
            cfg["data"]["data_path"] = data_path

        device_mode = self.learning_device_var.get().strip()
        if device_mode == "CPU only":
            cfg["device"] = "cpu"

        try:
            input_steps = int(self.learning_input_steps_var.get().strip())
            if input_steps > 0:
                cfg["data"]["input_steps"] = input_steps
        except ValueError:
            pass

        try:
            pred_steps = int(self.learning_pred_steps_var.get().strip())
            if pred_steps > 0:
                cfg["data"]["pred_steps"] = pred_steps
        except ValueError:
            pass

        try:
            batch_size = int(self.learning_batch_size_var.get().strip())
            if batch_size > 0:
                cfg["data"]["batch_size"] = batch_size
        except ValueError:
            pass

        mode = self.learning_mode_var.get().strip()

        try:
            patience_val = int(self.learning_patience_var.get().strip())
        except ValueError:
            patience_val = cfg["train"].get("patience", 10)

        if mode == "Fixed epochs":
            try:
                epochs = int(self.learning_epochs_var.get().strip())
                if epochs > 0:
                    cfg["train"]["epochs"] = epochs
            except ValueError:
                pass
            cfg["train"]["patience"] = 0
        else:
            cfg["train"]["patience"] = max(patience_val, 1)
            cfg["train"]["epochs"] = 10**9

        try:
            lr = float(self.learning_lr_var.get().strip())
            if lr > 0:
                cfg["train"]["learning_rate"] = lr
        except ValueError:
            pass

        overrides = self.learning_model_overrides.get(model_name)
        if overrides:
            cfg["model"].update(overrides)

        return cfg

    def on_learning_hyperparams(self) -> None:
        model_name = self.learning_model_var.get().strip().lower()
        if not model_name:
            messagebox.showerror("Error", "Please select a model first.")
            return

        base_cfg = get_default_config()
        base_model_cfg = base_cfg["model"]
        current_overrides = self.learning_model_overrides.get(model_name, {})

        if model_name == "gwnet":
            fields = [
                ("gwnet_dropout", float),
                ("gwnet_residual_channels", int),
                ("gwnet_dilation_channels", int),
                ("gwnet_skip_channels", int),
                ("gwnet_end_channels", int),
                ("gwnet_kernel_size", int),
                ("gwnet_blocks", int),
                ("gwnet_layers", int),
            ]
        elif model_name == "stgformer":
            fields = [
                ("stg_steps_per_day", int),
                ("stg_input_dim", int),
                ("stg_output_dim", int),
                ("stg_input_embedding_dim", int),
                ("stg_tod_embedding_dim", int),
                ("stg_dow_embedding_dim", int),
                ("stg_spatial_embedding_dim", int),
                ("stg_adaptive_embedding_dim", int),
                ("stg_num_heads", int),
                ("stg_num_layers", int),
                ("stg_dropout", float),
                ("stg_mlp_ratio", float),
                ("stg_use_mixed_proj", int),
                ("stg_dropout_a", float),
                ("stg_kernel_size", int),
            ]
        else:
            fields = [
                ("lstm_hidden_dim", int),
                ("lstm_num_layers", int),
            ]

        win = tk.Toplevel(self)
        win.title(f"{model_name.upper()} hyper-parameters")
        win.resizable(False, False)

        entries: dict[str, tk.Entry] = {}
        for row, (key, _typ) in enumerate(fields):
            ttk.Label(win, text=key).grid(row=row, column=0, sticky="w", padx=8, pady=4)
            if key in current_overrides:
                val = current_overrides[key]
            else:
                val = base_model_cfg.get(key, "")
            var = tk.StringVar(value=str(val))
            entry = ttk.Entry(win, textvariable=var, width=16)
            entry.grid(row=row, column=1, sticky="w", padx=8, pady=4)
            entries[key] = entry

        def on_ok() -> None:
            new_params: dict[str, object] = {}
            for key, typ in fields:
                raw = entries[key].get().strip()
                if not raw:
                    continue
                try:
                    value: object
                    if typ is int:
                        value = int(raw)
                    else:
                        value = float(raw)
                    new_params[key] = value
                except ValueError:
                    messagebox.showerror("Error", f"Invalid value for {key}: {raw}")
                    return
            self.learning_model_overrides[model_name] = new_params
            messagebox.showinfo(
                "Hyper-parameters",
                "Hyper-parameters updated for this session.\nThey will reset to defaults when you restart the program.",
            )
            win.destroy()

        btn_row = len(fields)
        ttk.Button(win, text="OK", command=on_ok).grid(
            row=btn_row, column=0, padx=8, pady=8, sticky="e"
        )
        ttk.Button(win, text="Cancel", command=win.destroy).grid(
            row=btn_row, column=1, padx=8, pady=8, sticky="w"
        )

    def _ensure_learning_log_view(self) -> None:
        if self.table_frame.winfo_ismapped():
            self.table_frame.pack_forget()
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))
        for child in self.plot_frame.winfo_children():
            child.destroy()
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        text = tk.Text(self.plot_frame, wrap="word")
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(self.plot_frame, orient="vertical", command=text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set)
        self.learning_log_widget = text

    def _append_learning_log(self, line: str) -> None:
        if self.learning_log_widget is None:
            return
        self.learning_log_widget.insert(tk.END, line + "\n")
        self.learning_log_widget.see(tk.END)

    def on_learning_browse_data(self) -> None:
        path = filedialog.askopenfilename(
            title="Select training data (.npy/.npz)",
            filetypes=[("NumPy file", "*.npy *.npz"), ("All Files", "*.*")],
        )
        if not path:
            return
        self.learning_data_path_var.set(path)

    def on_learning_train(self) -> None:
        cfg = self._build_learning_config()
        self.learning_stop_training = False
        self.learning_view_mode = "log"
        self._ensure_learning_log_view()
        if self.learning_log_widget is not None:
            self.learning_log_widget.delete("1.0", tk.END)

        def gui_log(msg: str) -> None:
            self.after(0, lambda: self._append_learning_log(msg))

        def worker() -> None:
            try:
                paths = run_training(
                    cfg,
                    log_callback=gui_log,
                    stop_callback=lambda: self.learning_stop_training,
                )

                def show_ok() -> None:
                    msg_lines = ["Training finished."]
                    if isinstance(paths, dict):
                        best = paths.get("best")
                        last = paths.get("last")
                        if best:
                            msg_lines.append(f"Best model file: {best}")
                        if last:
                            msg_lines.append(f"Last model file: {last}")
                    messagebox.showinfo("Training", "\n".join(msg_lines))

                self.after(0, show_ok)
            except Exception as exc:

                logger.exception("Learning training failed: %s", exc)

                def show_error() -> None:
                    messagebox.showerror("Error", f"Training failed: {exc}")

                self.after(0, show_error)

        threading.Thread(target=worker, daemon=True).start()

    def on_learning_load_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="Select checkpoint (.pth)",
            filetypes=[("PyTorch checkpoint", "*.pth"), ("All Files", "*.*")],
        )
        if not path:
            return
        self.learning_checkpoint_path = path
        messagebox.showinfo("Checkpoint", f"Loaded checkpoint:\n{path}")

    def on_learning_stop(self) -> None:
        self.learning_stop_training = True

    def on_learning_predict(self) -> None:
        cfg = self._build_learning_config()
        checkpoint = self.learning_checkpoint_path
        if not checkpoint:
            messagebox.showinfo(
                "Info", "Please load a checkpoint (.pth) first using 'Load .pth'."
            )
            return

        output_path = "prediction.npy"

        idx_str = self.learning_node_index_var.get().strip()
        node_idx = None
        if idx_str:
            try:
                node_idx = int(idx_str)
            except ValueError:
                messagebox.showerror("Error", "Node index must be an integer (0-based).")
                return

        def worker() -> None:
            try:
                result = run_inference(
                    cfg=cfg,
                    checkpoint_path=checkpoint,
                    output_path=output_path,
                    use_test_split=False,
                    node_idx=node_idx,
                    save_plot=False,
                    return_figure=False,
                    return_series=True,
                    save_outputs=False,
                )

                def show_fig() -> None:
                    if result is not None:
                        node_idx_local, true_series, pred_series = result
                        fig = Figure(figsize=(10, 4))
                        ax = fig.add_subplot(111)
                        ax.plot(true_series, label="True", linewidth=1.2)
                        ax.plot(pred_series, label="Pred", linewidth=1.2)
                        ax.set_title(f"Node {node_idx_local} – Full Series")
                        ax.set_xlabel("Time index")
                        ax.set_ylabel("Value")
                        ax.legend()
                        fig.tight_layout()
                        self.learning_view_mode = "plot"
                        self._show_figure(
                            fig,
                            "Prediction result (True vs Pred) for selected node.",
                        )

                self.after(0, show_fig)
            except RuntimeError as exc:
                logger.exception("Learning prediction failed (model mismatch): %s", exc)

                def show_error_model() -> None:
                    messagebox.showerror(
                        "Model checkpoint mismatch",
                        "Failed to load checkpoint into current model.\n"
                        "Please check that:\n"
                        "- The checkpoint type (GWNet / LSTM) matches current model,\n"
                        "- Hyper-parameters are consistent with the training settings.",
                    )

                self.after(0, show_error_model)
            except Exception as exc:
                logger.exception("Learning prediction failed: %s", exc)

                def show_error() -> None:
                    messagebox.showerror("Error", f"Prediction failed: {exc}")

                self.after(0, show_error)

        threading.Thread(target=worker, daemon=True).start()

    def on_learning_save_outputs(self) -> None:
        cfg = self._build_learning_config()
        checkpoint = self.learning_checkpoint_path
        if not checkpoint:
            messagebox.showinfo(
                "Info", "Please load a checkpoint (.pth) first using 'Load .pth'."
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save prediction outputs (.npy)",
            defaultextension=".npy",
            filetypes=[("NumPy file", "*.npy"), ("All Files", "*.*")],
        )
        if not path:
            return

        def worker() -> None:
            try:
                run_inference(
                    cfg=cfg,
                    checkpoint_path=checkpoint,
                    output_path=path,
                    use_test_split=False,
                    node_idx=None,
                    save_plot=False,
                    return_figure=False,
                    return_series=False,
                    save_outputs=True,
                )

                def show_ok() -> None:
                    messagebox.showinfo(
                        "Save outputs",
                        f"Full prediction outputs have been saved to:\n{path}",
                    )

                self.after(0, show_ok)
            except RuntimeError as exc:
                logger.exception("Saving prediction outputs failed (model mismatch): %s", exc)

                def show_error_model() -> None:
                    messagebox.showerror(
                        "Model checkpoint mismatch",
                        "Failed to load checkpoint into current model.\n"
                        "Please check that:\n"
                        "- The checkpoint type (GWNet / LSTM / STGformer) matches current model,\n"
                        "- Hyper-parameters are consistent with the training settings.",
                    )

                self.after(0, show_error_model)
            except Exception as exc:
                logger.exception("Saving prediction outputs failed: %s", exc)

                def show_error() -> None:
                    messagebox.showerror("Error", f"Save outputs failed: {exc}")

                self.after(0, show_error)

        threading.Thread(target=worker, daemon=True).start()

    def _get_current_dataframe(self) -> pd.DataFrame:
        """Helper to get current full dataframe (not only current page)."""
        if not self._ensure_data_loaded():
            return pd.DataFrame(columns=DEFAULT_COLUMNS)
        return self.current_df

    def _configure_table_columns(self) -> None:
        """
        Configure Treeview columns/headings based on self.table_columns.

        For the default traffic table we keep special widths; for other
        tables (e.g. spatio-temporal) we use a compact, generic layout.
        """
        self.tree["columns"] = self.table_columns

        default_widths = {
            "VehicleType": 90,
            "DetectionTime_O": 160,
            "GantryID_O": 110,
            "DetectionTime_D": 160,
            "GantryID_D": 110,
            "TripLength": 90,
            "TripEnd": 80,
            "TripInformation": 420,
        }

        is_default = set(self.table_columns) == set(DEFAULT_COLUMNS)

        for col in self.table_columns:
            self.tree.heading(col, text=col)
            if is_default:
                width = default_widths.get(col, 120)
                stretch = col == "TripInformation"
            else:
                if col.lower().startswith("time"):
                    width = 160
                else:
                    width = 80
                stretch = False
            self.tree.column(col, width=width, minwidth=60, anchor=tk.W, stretch=stretch)

    def _update_view_controls(self) -> None:
        """Rebuild view controls according to selected field."""
        for child in self.view_controls_frame.winfo_children():
            child.destroy()

        df = self._get_current_dataframe()
        col_name = self.view_field_var.get()

        if df.empty or col_name not in df.columns:
            self.view_info_var.set("No data available. Please import data first.")
            return

        kind, hint, categories = build_view_hint(df, col_name)

        if kind == "category":
            unique_vals = categories or []
            ttk.Label(self.view_controls_frame, text="Value:").pack(side=tk.LEFT)
            value_combo = ttk.Combobox(
                self.view_controls_frame,
                textvariable=self.view_value_var,
                values=unique_vals,
                state="readonly",
                width=16,
            )
            value_combo.pack(side=tk.LEFT, padx=(2, 4))
            self.view_info_var.set(hint)
        else:
            ttk.Label(self.view_controls_frame, text="From:").pack(side=tk.LEFT)
            from_entry = ttk.Entry(
                self.view_controls_frame, textvariable=self.view_from_var, width=14
            )
            from_entry.pack(side=tk.LEFT, padx=(2, 4))
            ttk.Label(self.view_controls_frame, text="To:").pack(side=tk.LEFT)
            to_entry = ttk.Entry(
                self.view_controls_frame, textvariable=self.view_to_var, width=14
            )
            to_entry.pack(side=tk.LEFT, padx=(2, 4))
            self.view_info_var.set(hint)

    def _clear_plot_content(self) -> None:
        """Remove existing matplotlib canvas and stats from the plot frame, but keep layout."""
        if self.plot_canvas is not None:
            widget = self.plot_canvas.get_tk_widget()
            widget.destroy()
            self.plot_canvas = None
        for child in self.plot_frame.winfo_children():
            child.destroy()

    def _close_plot(self) -> None:
        """Completely close the plot area and restore the table view."""
        self._clear_plot_content()
        self.plot_frame.pack_forget()
        if not self.table_frame.winfo_ismapped():
            self.table_frame.pack(
                side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 4)
            )

    def _show_figure(self, fig, stats_text: str | None = None) -> None:
        """Embed a matplotlib Figure and an optional stats panel into the GUI."""
        self._clear_plot_content()
        if fig is None:
            return

        if self.table_frame.winfo_ismapped():
            self.table_frame.pack_forget()

        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))
        self.plot_frame.columnconfigure(0, weight=3)
        self.plot_frame.columnconfigure(1, weight=1)

        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.plot_canvas.draw()
        widget = self.plot_canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=4)

        if stats_text:
            stats_label = ttk.Label(
                self.plot_frame,
                text=stats_text,
                justify="left",
                anchor="nw",
            )
            stats_label.grid(row=0, column=1, sticky="nsew", pady=4)

    def on_plot_stats(self) -> None:
        """Call backend statistics and plotting functions based on selected field."""
        df = self._get_current_dataframe()
        if df.empty:
            messagebox.showinfo("Statistics", "No data available for statistics and plotting.")
            return

        col_var = getattr(self, "stats_field_var", None)
        col_name = col_var.get() if col_var is not None else None
        if not col_name or col_name not in df.columns:
            messagebox.showerror("Error", "Please select a valid field.")
            return

        ov = overview_stats(df)
        overview_text = (
            "Overview:\n"
            f"  Rows: {ov['total_rows']}\n"
            f"  Columns: {ov['total_cols']}\n"
            f"  Vehicle types: {ov['distinct_vehicle_types']}\n\n"
        )

        try:
            if col_name == "VehicleType":
                result = vehicle_type_pie(df)
                stats = result["stats"]
                fig = result["figure"]
                stats_text = overview_text + (
                    "VehicleType stats:\n"
                    f"Total samples: {stats['total']}\n"
                    f"Unique types: {stats['n_unique']}\n\n"
                    "Top classes:\n"
                    + "\n".join(f"{k}: {v}" for k, v in stats['top_counts'].items())
                )
                self._show_figure(fig, stats_text)

            elif col_name in ("DetectionTime_O", "DetectionTime_D"):
                result = time_hist_5min(df, col_name)
                stats = result["stats"]
                fig = result["figure"]
                stats_text = overview_text + (
                    f"{col_name} time stats:\n"
                    f"Valid timestamps: {stats['count']}\n"
                    f"Start: {stats['start']}\n"
                    f"End:   {stats['end']}\n"
                )
                self._show_figure(fig, stats_text)

            elif col_name == "TripLength":
                result = trip_length_hist(df)
                if result is None:
                    messagebox.showinfo("Statistics", "No valid TripLength values in current data.")
                    return
                stats = result["stats"]
                fig = result["figure"]
                stats_text = overview_text + (
                    "TripLength stats:\n"
                    f"n:   {stats['count']:.0f}\n"
                    f"min: {stats['min']:.2f}\n"
                    f"mean:{stats['mean']:.2f}\n"
                    f"std: {stats['std']:.2f}\n"
                    f"max: {stats['max']:.2f}\n"
                )
                self._show_figure(fig, stats_text)

            elif col_name in ("GantryID_O", "GantryID_D"):
                result = gantry_code_bar(df, col_name)
                stats = result["stats"]
                fig = result["figure"]
                stats_text = overview_text + (
                    f"{col_name} stats:\n"
                    f"Unique codes: {stats['n_unique']}\n"
                    f"Top codes:\n"
                    + "\n".join(f"{k}: {v}" for k, v in stats['top_counts'].items())
                )
                self._show_figure(fig, stats_text)

            else:
                messagebox.showinfo("Statistics", f"Field {col_name} is not supported for dedicated plots.")

        except Exception as exc:
            logger.exception("Statistics plotting failed: %s", exc)
            messagebox.showerror("Error", f"Statistics plotting failed: {exc}")

    def on_generate_st(self) -> None:
        """Generate spatio-temporal dataset from current data."""
        if self.store is None or self.store.dataframe.empty:
            messagebox.showinfo("Info", "Please import data first in the 'File' tab.")
            return

        base_df = self._get_current_dataframe()
        time_col = self.st_time_col_var.get()
        node_col = self.st_node_col_var.get()
        freq_str = self.st_freq_var.get().strip()

        if time_col not in base_df.columns or node_col not in base_df.columns:
            base_df = self.store.dataframe
            if time_col not in base_df.columns or node_col not in base_df.columns:
                messagebox.showerror("Error", "Invalid time column or node column.")
                return
        try:
            freq_min = int(freq_str)
            if freq_min <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Bin size must be a positive integer (minutes).")
            return

        self.st_shape_info = ""
        self.st_info_var.set("Generating spatio-temporal data, please wait...")
        self._update_status()

        try:
            st_df = generate_spatiotemporal(base_df, time_col, node_col, freq_min)
        except Exception as exc:
            logger.exception("Spatio-temporal generation failed: %s", exc)
            messagebox.showerror("Error", f"Spatio-temporal generation failed: {exc}")
            self.st_info_var.set("Spatio-temporal generation failed.")
            self._update_status()
            return

        if self.st_prev_df is None:
            self.st_prev_df = self.current_df.copy()
            self.st_prev_page = self.current_page

        self.st_df = st_df
        logger.info(
            "Generated spatio-temporal dataset: %d time bins x %d nodes",
            st_df.shape[0],
            st_df.shape[1],
        )
        self.current_df = st_df.reset_index()
        columns = list(self.current_df.columns)
        if len(columns) > 11:
            columns = columns[:11]
            self.current_df = self.current_df[columns]
        self.table_columns = columns
        self._configure_table_columns()
        self.current_page = 1
        self._refresh_view()

        self.st_shape_info = (
            f"Spatio-temporal dataset: {st_df.shape[0]} time bins x {st_df.shape[1]} nodes."
        )
        self.st_info_var.set("Spatio-temporal dataset generated.")
        self._update_status()

    def on_generate_st_traj(self) -> None:
        if self.store is None or self.store.dataframe.empty:
            messagebox.showinfo("Info", "Please import data first in the 'File' tab.")
            return

        base_df = self.store.dataframe
        trip_col = "TripInformation"
        freq_str = self.st_freq_var.get().strip()

        if trip_col not in base_df.columns:
            messagebox.showerror(
                "Error", "Column 'TripInformation' was not found in current data."
            )
            return
        try:
            freq_min = int(freq_str)
            if freq_min <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Bin size must be a positive integer (minutes).")
            return

        self.st_shape_info = ""
        self.st_info_var.set(
            "Generating trajectory-based spatio-temporal data, please wait..."
        )
        self._update_status()
        progress_win = tk.Toplevel(self)
        progress_win.title("Processing")
        progress_win.resizable(False, False)
        ttk.Label(
            progress_win,
            text="Generating trajectory-based spatio-temporal data...",
        ).pack(side=tk.TOP, padx=12, pady=(10, 4))
        pb = ttk.Progressbar(progress_win, mode="indeterminate")
        pb.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(0, 10))
        pb.start(10)
        progress_win.transient(self)
        progress_win.grab_set()

        def worker() -> None:
            try:
                st_df = generate_spatiotemporal_from_tripinfo(
                    base_df, trip_col, freq_min
                )
            except Exception as exc:
                def on_error() -> None:
                    pb.stop()
                    progress_win.grab_release()
                    progress_win.destroy()
                    logger.exception(
                        "Trajectory spatio-temporal generation failed: %s", exc
                    )
                    messagebox.showerror(
                        "Error", f"Trajectory spatio-temporal generation failed: {exc}"
                    )
                    self.st_info_var.set(
                        "Trajectory spatio-temporal generation failed."
                    )
                    self._update_status()

                self.after(0, on_error)
                return

            def on_success() -> None:
                pb.stop()
                progress_win.grab_release()
                progress_win.destroy()
                if self.st_prev_df is None:
                    self.st_prev_df = self.current_df.copy()
                    self.st_prev_page = self.current_page
                self.st_df = st_df
                logger.info(
                    "Generated trajectory-based spatio-temporal dataset: %d time bins x %d nodes",
                    st_df.shape[0],
                    st_df.shape[1],
                )
                self.current_df = st_df.reset_index()
                columns = list(self.current_df.columns)
                if len(columns) > 11:
                    columns = columns[:11]
                    self.current_df = self.current_df[columns]
                self.table_columns = columns
                self._configure_table_columns()
                self.current_page = 1
                self._refresh_view()
                self.st_shape_info = (
                    f"Trajectory ST dataset: {st_df.shape[0]} time bins x {st_df.shape[1]} nodes."
                )
                self.st_info_var.set(
                    "Trajectory-based spatio-temporal dataset generated."
                )
                self._update_status()

            self.after(0, on_success)

        threading.Thread(target=worker, daemon=True).start()

    def on_plot_flow(self) -> None:
        """Plot flow time series for a given node index from spatio-temporal dataset."""
        if self.st_df is None or self.st_df.empty:
            messagebox.showinfo("Info", "Please generate spatio-temporal data first.")
            return

        idx_str = self.st_node_index_var.get().strip()
        if not idx_str:
            messagebox.showinfo("Info", "Please enter a node index (1-based integer).")
            return
        try:
            node_idx = int(idx_str)
        except ValueError:
            messagebox.showerror("Error", "Node index must be an integer.")
            return

        try:
            result = flow_timeseries(self.st_df, node_idx)
        except Exception as exc:
            logger.exception("Flow plot failed: %s", exc)
            messagebox.showerror("Error", f"Flow plot failed: {exc}")
            return

        node_name = result["node"]
        stats = result["stats"]
        fig = result["figure"]

        stats_text = (
            f"Node index: {node_idx}\n"
            f"Node label: {node_name}\n\n"
            "Flow stats:\n"
            f"  n:   {stats['count']:.0f}\n"
            f"  sum: {stats['sum']:.0f}\n"
            f"  min: {stats['min']:.0f}\n"
            f"  mean:{stats['mean']:.2f}\n"
            f"  max: {stats['max']:.0f}\n"
        )

        self._show_figure(fig, stats_text)

    def on_plot_st_3d(self) -> None:
        """Plot 3D spatio-temporal surface from current spatio-temporal dataset."""
        if self.st_df is None or self.st_df.empty:
            messagebox.showinfo("Info", "Please generate spatio-temporal data first.")
            return

        try:
            result = st_3d_surface(self.st_df)
        except Exception as exc:
            logger.exception("3D plot failed: %s", exc)
            messagebox.showerror("Error", f"3D plot failed: {exc}")
            return

        shape = result.get("shape", self.st_df.shape)
        fig = result["figure"]

        stats_text = (
            "Spatio-temporal 3D surface\n\n"
            f"Time bins (shown): {shape[0]}\n"
            f"Nodes (shown): {shape[1]}\n"
        )

        self._show_figure(fig, stats_text)

    def on_load_st_3d_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select spatio-temporal matrix file (.npy or .csv)",
            filetypes=[
                ("NumPy array", "*.npy"),
                ("CSV file", "*.csv"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            lower = path.lower()
            if lower.endswith(".npy"):
                data = np.load(path, allow_pickle=True)
                if isinstance(data, dict):
                    if "st" in data:
                        arr = data["st"]
                    elif "pred_full" in data:
                        arr = data["pred_full"].T
                    else:
                        arr = np.array(list(data.values())[0])
                else:
                    arr = data
            else:
                arr = np.loadtxt(path, delimiter=",")

            if arr.ndim == 1:
                arr = arr[:, None]
            t_len, n_nodes = arr.shape
            index = pd.date_range("2000-01-01", periods=t_len, freq="min")
            df = pd.DataFrame(arr, index=index)
            df.columns = [str(i) for i in range(n_nodes)]

            result = st_3d_surface(df)
            fig = result["figure"]
            # For 3D plots loaded from file, swap the displayed time/node axis labels
            # to match the user's interpretation for prediction exports, without
            # changing the internal data orientation or the normal 3D plot behavior.
            try:
                if fig.axes:
                    ax = fig.axes[0]
                    x_label = ax.get_xlabel()
                    y_label = ax.get_ylabel()
                    ax.set_xlabel(y_label)
                    ax.set_ylabel(x_label)
            except Exception:
                # If anything goes wrong while adjusting labels, fall back silently.
                pass

            shape = result.get("shape", df.shape)
            stats_text = (
                "Spatio-temporal 3D surface (from file)\n\n"
                f"Time bins (shown): {shape[0]}\n"
                f"Nodes (shown): {shape[1]}\n"
                f"Source file: {Path(path).name}"
            )
            self._show_figure(fig, stats_text)
        except Exception as exc:
            logger.exception("Loading ST matrix file failed: %s", exc)
            messagebox.showerror(
                "Error", f"Failed to load spatio-temporal matrix file:\n{exc}"
            )

    def on_close_st_view(self) -> None:
        if self.plot_frame.winfo_ismapped():
            logger.info("Closing embedded plot and returning to table view")
            self._close_plot()
            if self.toolbar_mode.get() == "Spatio-temporal":
                self.st_info_var.set("Returned to spatio-temporal table view.")
                self._update_status()
            return

        if self.st_prev_df is not None:
            logger.info("Closing spatio-temporal view and restoring previous table view")
            self.current_df = self.st_prev_df
            self.current_page = self.st_prev_page
            self.table_columns = list(DEFAULT_COLUMNS)
            self._configure_table_columns()
            self._refresh_view()
            self.st_shape_info = ""
            self.st_info_var.set("Closed spatio-temporal table and restored original view.")
            self._update_status()
            self.st_prev_df = None
            self.st_prev_page = 1
            return

        messagebox.showinfo("Info", "No content to go back to.")

    def on_apply_view_filter(self) -> None:
        """Apply view filter based on selected field and range/value."""
        if self.store is None or self.store.dataframe.empty:
            messagebox.showinfo("Info", "Please import data first in the 'File' tab.")
            return

        base_df = self.store.dataframe
        col_name = self.view_field_var.get()
        if not col_name or col_name not in base_df.columns:
            messagebox.showerror("Error", "Invalid view field.")
            return

        try:
            kind = get_field_kind(col_name)
            if kind is None:
                messagebox.showerror("Error", f"Field {col_name} is not supported for view filtering.")
                return

            value = self.view_value_var.get().strip()
            from_str = self.view_from_var.get().strip()
            to_str = self.view_to_var.get().strip()

            filtered = apply_view_filter(
                base_df, col_name, kind, value=value, from_str=from_str, to_str=to_str
            )

            self.current_df = filtered
            self.current_page = 1
            self._refresh_view()
            self.view_info_var.set(
                f"{col_name} filter applied: {len(filtered)} rows."
            )
        except Exception as exc:
            logger.exception("View filtering failed: %s", exc)
            messagebox.showerror("Error", f"View filtering failed: {exc}")

    def on_clear_view_filter(self) -> None:
        """Clear view filter and show full dataset from store."""
        if self.store is None or self.store.dataframe.empty:
            messagebox.showinfo("Info", "No data available; nothing to clear.")
            return
        self.current_df = self.store.dataframe.copy()
        self.current_page = 1
        self.view_from_var.set("")
        self.view_to_var.set("")
        self.view_value_var.set("")
        self._refresh_view()
        self.view_info_var.set("View filter cleared.")

    def on_close(self) -> None:
        """Handle window close: destroy GUI and terminate the process."""
        logger.info("TrafficLens GUI window closed by user")
        self.destroy()
        sys.exit(0)

    def on_clear_data(self) -> None:
        """Clear all loaded data and reset the view."""
        if not messagebox.askyesno("Confirm", "Are you sure you want to clear all loaded data?"):
            return
        logger.info("Clearing all loaded data")
        self.store = None
        self.current_df = pd.DataFrame(columns=DEFAULT_COLUMNS)
        self.current_page = 1
        self.search_var.set("")
        self.last_search_keyword = None
        self.last_search_column = None
        self.last_search_strict = False
        self.last_match_count = None
        self._close_plot()
        self._refresh_view()

    def on_export(self) -> None:
        """Export current result with header rows removed; spatio-temporal export outputs full matrix."""
        if self.current_df is None or self.current_df.empty:
            messagebox.showinfo("Info", "No data to export.")
            return

        path = filedialog.asksaveasfilename(
            title="Export current result",
            defaultextension=".csv",
            filetypes=[
                ("CSV file", "*.csv"),
                ("Excel file", "*.xlsx"),
                ("NumPy NPY", "*.npy"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return

        lower_path = path.lower()
        if lower_path.endswith(".xlsx"):
            fmt = "xlsx"
        elif lower_path.endswith(".npy"):
            fmt = "npy"
        else:
            fmt = "csv"

        try:
            is_st_view = (
                self.st_prev_df is not None
                and self.st_df is not None
                and not self.st_df.empty
            )

            if is_st_view:
                matrix_values = self.st_df.values
                if fmt == "csv":
                    np.savetxt(path, matrix_values, delimiter=",", fmt="%s")
                elif fmt == "xlsx":
                    pd.DataFrame(matrix_values).to_excel(
                        path, index=False, header=False
                    )
                else:
                    np.save(path, matrix_values)
            else:
                data_values = self.current_df.values
                if fmt == "csv":
                    np.savetxt(path, data_values, delimiter=",", fmt="%s")
                elif fmt == "xlsx":
                    pd.DataFrame(data_values).to_excel(path, index=False, header=False)
                else:
                    np.save(path, data_values)
            messagebox.showinfo("Success", f"Data exported to:\n{path}")
        except Exception as exc:
            logger.exception("Export current result failed: %s", exc)
            messagebox.showerror("Error", f"Export current result failed: {exc}")

    def _strip_header_like_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove any remaining header-like rows (should already be handled at load time)."""
        return df
    def show_overview_stats(self) -> None:
        df = self._get_current_dataframe()
        if df.empty:
            messagebox.showinfo("Statistics", "No data available for statistics.")
            return
        stats = overview_stats(df)
        total_rows = stats["total_rows"]
        total_cols = stats["total_cols"]
        distinct_vehicle = stats["distinct_vehicle_types"]
        msg = (
            "Overall overview:\n\n"
            f"- Total rows: {total_rows}\n"
            f"- Total columns: {total_cols}\n"
            f"- Distinct vehicle types: {distinct_vehicle}\n"
        )
        messagebox.showinfo("Statistics - Overview", msg)

    def show_vehicle_stats(self) -> None:
        df = self._get_current_dataframe()
        if df.empty:
            messagebox.showinfo("Statistics", "No data available for statistics.")
            return
        try:
            stats = vehicle_type_pie(df)
        except Exception as exc:
            logger.exception("VehicleType statistics failed: %s", exc)
            messagebox.showerror("Error", f"VehicleType statistics failed: {exc}")
            return

        top_lines = [f"{k}: {v}" for k, v in stats["top_counts"].items()]
        msg = (
            "VehicleType frequency statistics:\n\n"
            f"- Total samples: {stats['total']}\n"
            f"- Distinct VehicleType count: {stats['n_unique']}\n"
            f"- Most common classes:\n  " + "\n  ".join(top_lines)
        )
        messagebox.showinfo("Statistics - VehicleType", msg)

    def show_triplength_stats(self) -> None:
        df = self._get_current_dataframe()
        if df.empty:
            messagebox.showinfo("Statistics", "No data available for statistics.")
            return
        stats = trip_length_hist(df)
        if stats is None:
            messagebox.showinfo("Statistics", "No valid TripLength values in current data.")
            return
        msg = (
            "TripLength statistics:\n\n"
            f"- Count: {stats['count']:.0f}\n"
            f"- Min: {stats['min']:.2f}\n"
            f"- Mean: {stats['mean']:.2f}\n"
            f"- Std: {stats['std']:.2f}\n"
            f"- Max: {stats['max']:.2f}\n"
        )
        messagebox.showinfo("Statistics - TripLength", msg)


    def on_tree_click(self, event) -> None:
        """Track which column user clicked, for button-based sorting."""
        col_id = self.tree.identify_column(event.x)
        try:
            idx = int(col_id.replace("#", "")) - 1
        except ValueError:
            return
        if 0 <= idx < len(DEFAULT_COLUMNS):
            self.selected_sort_col = DEFAULT_COLUMNS[idx]
            if hasattr(self, "sort_col_label_var"):
                self.sort_col_label_var.set(self.selected_sort_col)

    def on_jump_page(self) -> None:
        """Jump to the page number specified in the bottom input box."""
        if not self._ensure_data_loaded():
            return
        if self.page_input_var is None:
            return

        raw = self.page_input_var.get().strip()
        total = len(self.current_df)
        if total == 0:
            messagebox.showinfo("Info", "No data available, cannot jump to a page.")
            return

        max_page = max((total - 1) // self.page_size + 1, 1)
        try:
            page = int(raw)
        except ValueError:
            messagebox.showerror("Error", "Page number must be a positive integer.")
            self.page_input_var.set(str(self.current_page))
            return

        if page < 1:
            page = 1
        if page > max_page:
            page = max_page

        self.current_page = page
        self._refresh_view()

    def on_import_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select CSV file to import",
            filetypes=[("CSV file", "*.csv"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            file_path = Path(path)
            logger.info("Import file: %s", file_path)
            self.store = TrafficDataStore.from_files([file_path])
            self.current_df = self.store.dataframe.copy()
            self.current_page = 1
            self._refresh_view()
            messagebox.showinfo("Success", f"Imported file: {file_path.name}")
        except Exception as exc:
            logger.exception("Import file failed: %s", exc)
            messagebox.showerror("Error", f"Import file failed: {exc}")

    def on_import_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select folder to import")
        if not folder:
            return
        try:
            folder_path = Path(folder)
            files = sorted(folder_path.glob("*.csv"))
            if not files:
                messagebox.showinfo("Info", "No CSV files found in this folder.")
                return
            logger.info("Import folder: %s (%d files)", folder_path, len(files))
            self.store = TrafficDataStore.from_files(files)
            self.current_df = self.store.dataframe
            self.current_page = 1
            self._refresh_view()
            messagebox.showinfo("Success", f"Imported and merged {len(files)} CSV files from folder.")
        except Exception as exc:
            logger.exception("Import folder failed: %s", exc)
            messagebox.showerror("Error", f"Import folder failed: {exc}")

    def on_search(self) -> None:
        keyword = self.search_var.get().strip()
        try:
            if not self._ensure_data_loaded():
                return
            logger.info("Search with keyword: %s", keyword)
            base_df = self.store.dataframe
            if keyword:
                scope = getattr(self, "search_scope_var", None)
                scope_value = scope.get() if scope is not None else "All fields"
                search_column = None if scope_value == "All fields" else scope_value
                mode = getattr(self, "search_mode_var", None)
                mode_value = mode.get() if mode is not None else "Fuzzy"
                strict = mode_value == "Exact"

                filtered = self.store.search(
                    keyword, column=search_column, strict=strict
                )
            else:
                filtered = base_df.copy()

            self.last_search_keyword = keyword if keyword else None
            self.last_search_column = search_column if keyword else None
            self.last_search_strict = strict if keyword else False
            self.last_match_count = len(filtered) if keyword else None

            self.current_df = filtered
            self.current_page = 1
            self._refresh_view()
        except Exception as exc:
            logger.exception("Search failed: %s", exc)
            messagebox.showerror("Error", f"Search failed: {exc}")

    def on_sort_button(self, ascending: bool) -> None:
        """Sort using the column selected in the table, triggered by buttons."""
        try:
            if not self._ensure_data_loaded():
                return
            col = getattr(self, "selected_sort_col", None) or DEFAULT_COLUMNS[0]
            logger.info("Button sort column=%s ascending=%s", col, ascending)

            # VehicleType is semantically numeric (e.g. 31, 32, 41), so for sorting
            # we use its numeric value rather than lexicographic string order.
            if col == "VehicleType" and col in self.current_df.columns:
                tmp = self.current_df.copy()
                tmp["_sort_key"] = pd.to_numeric(tmp[col], errors="coerce")
                tmp = tmp.sort_values(
                    by="_sort_key",
                    ascending=ascending,
                    na_position="last",
                )
                self.current_df = tmp.drop(columns="_sort_key")
            else:
                self.current_df = self.current_df.sort_values(by=col, ascending=ascending)

            self.sorted_col = col
            self.sorted_ascending = ascending
            self.current_page = 1
            self._refresh_view()
        except Exception as exc:
            logger.exception("Button sort failed: %s", exc)
            messagebox.showerror("Error", f"Sort failed: {exc}")

    def on_merge(self) -> None:
        if not self._ensure_data_loaded():
            return
        paths = filedialog.askopenfilenames(
            title="Select CSV files to merge",
            filetypes=[("CSV file", "*.csv"), ("All Files", "*.*")],
        )
        if not paths:
            return
        try:
            files = [Path(p) for p in paths]
            logger.info("Merging CSV files: %s", files)
            self.store = self.store.merge_with_files(files)
            self.current_df = self.store.dataframe.copy()
            self.current_page = 1
            self.on_search()
            messagebox.showinfo("Success", "Selected CSV files merged successfully.")
        except Exception as exc:
            logger.exception("Merge failed: %s", exc)
            messagebox.showerror("Error", f"Merge CSV failed: {exc}")

    def on_merge_folder(self) -> None:
        if not self._ensure_data_loaded():
            return
        folder = filedialog.askdirectory(title="Select folder to merge")
        if not folder:
            return
        files = sorted(Path(folder).glob("*.csv"))
        if not files:
            messagebox.showinfo("Info", "No CSV files found in this folder.")
            return
        try:
            logger.info("Merging CSV folder: %s (%d files)", folder, len(files))
            self.store = self.store.merge_with_files(files)
            self.current_df = self.store.dataframe.copy()
            self.current_page = 1
            self.on_search()
            messagebox.showinfo("Success", "CSV files from folder merged into current data.")
        except Exception as exc:
            logger.exception("Merge folder failed: %s", exc)
            messagebox.showerror("Error", f"Merge folder failed: {exc}")

    def on_reset(self) -> None:
        logger.info("Reset view to original data")
        if not self._ensure_data_loaded():
            return
        self.search_var.set("")
        self.last_search_keyword = None
        self.last_search_column = None
        self.last_search_strict = False
        self.last_match_count = None
        self.current_df = self.store.dataframe.copy()
        self.current_page = 1
        self._refresh_view()

    def on_prev_page(self) -> None:
        if not self._ensure_data_loaded():
            return
        if self.current_page > 1:
            self.current_page -= 1
            logger.info("Go to previous page: %s", self.current_page)
            self._refresh_view()

    def on_next_page(self) -> None:
        if not self._ensure_data_loaded():
            return
        total = len(self.current_df)
        if total == 0:
            return
        max_page = max((total - 1) // self.page_size + 1, 1)
        if self.current_page < max_page:
            self.current_page += 1
            logger.info("Go to next page: %s", self.current_page)
            self._refresh_view()


def run_app() -> None:
    app = TrafficLensApp()
    app.mainloop()


