"""
Tkinter GUI for TrafficLens.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
from .ST_generator import generate_spatiotemporal


logger = logging.getLogger(__name__)


class TrafficLensApp(tk.Tk):
    def __init__(self, store: Optional[TrafficDataStore] = None) -> None:
        super().__init__()
        self.title("TrafficLens - 交通数据查询系统")
        self.geometry("1100x600")
        self.minsize(900, 500)

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
        self.search_scope_var = tk.StringVar(value="全部字段")
        self.search_mode_var = tk.StringVar(value="模糊")

        self.toolbar_mode = tk.StringVar(value="操作")

        self.table_columns = list(DEFAULT_COLUMNS)

        self.st_df: Optional[pd.DataFrame] = None
        self.st_prev_df: Optional[pd.DataFrame] = None
        self.st_prev_page: int = 1
        self.st_shape_info: str = ""

        self.plot_canvas: FigureCanvasTkAgg | None = None

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
        self.tab_style.configure("Tab.TButton", padding=(6, 2))
        self.tab_style.configure(
            "TabActive.TButton", padding=(6, 2), relief="sunken"
        )
        self.tab_style.configure(
            "GlobalBack.TButton",
            padding=(6, 2),
            foreground="#d9534f",
        )

        self.file_tab_btn = ttk.Button(
            tab_frame,
            text="文件",
            command=lambda: self._on_tab_click("文件"),
            width=8,
        )
        self.file_tab_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.action_tab_btn = ttk.Button(
            tab_frame,
            text="操作",
            command=lambda: self._on_tab_click("操作"),
            width=8,
        )
        self.action_tab_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.view_tab_btn = ttk.Button(
            tab_frame,
            text="视图",
            command=lambda: self._on_tab_click("视图"),
            width=8,
        )
        self.view_tab_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.st_tab_btn = ttk.Button(
            tab_frame,
            text="时空统计",
            command=lambda: self._on_tab_click("时空统计"),
            width=10,
        )
        self.st_tab_btn.pack(side=tk.LEFT)

        self.global_back_btn = ttk.Button(
            tab_frame,
            text="回退",
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
        ttk.Label(status_frame, text="  跳转到第").pack(side=tk.LEFT)
        page_entry = ttk.Entry(status_frame, textvariable=self.page_input_var, width=5)
        page_entry.pack(side=tk.LEFT)
        ttk.Label(status_frame, text="页").pack(side=tk.LEFT)
        ttk.Button(status_frame, text="跳转", command=self.on_jump_page).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        self.st_shape_label_var = tk.StringVar()
        self.st_shape_label = ttk.Label(
            status_frame, textvariable=self.st_shape_label_var
        )
        self.st_shape_label.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Button(status_frame, text="上一页", command=self.on_prev_page).pack(
            side=tk.RIGHT, padx=(4, 0)
        )
        ttk.Button(status_frame, text="下一页", command=self.on_next_page).pack(
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
            text = "无数据"
        else:
            max_page = max((total - 1) // self.page_size + 1, 1)
            text = (
                f"当前第 {self.current_page}/{max_page} 页，"
                f"每页 {self.page_size} 条，共 {total} 条"
            )
            if self.last_match_count is not None and self.last_search_keyword:
                text += f"；本次搜索匹配 {self.last_match_count} 条记录"

        self.status_var.set(text)

        self.st_shape_label_var.set(self.st_shape_info)

        if self.page_input_var is not None:
            self.page_input_var.set(str(self.current_page))

    def _ensure_data_loaded(self) -> bool:
        """Return True if data is available, otherwise show a hint and return False."""
        if self.store is None or self.store.dataframe.empty:
            messagebox.showinfo("提示", "请先在“文件”选项卡中导入数据。")
            return False
        return True

    def _clear_toolbar(self) -> None:
        for child in self.toolbar_frame.winfo_children():
            child.destroy()

    def _build_file_toolbar(self) -> None:
        """Toolbar content for '文件'选项卡：导入 / 合并 / 导出 / 重置。"""
        self._clear_toolbar()
        ttk.Button(
            self.toolbar_frame, text="导入文件", command=self.on_import_file
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            self.toolbar_frame, text="导入文件夹", command=self.on_import_folder
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            self.toolbar_frame, text="合并文件", command=self.on_merge
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            self.toolbar_frame, text="合并文件夹", command=self.on_merge_folder
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            self.toolbar_frame, text="导出当前结果", command=self.on_export
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            self.toolbar_frame, text="恢复原始顺序", command=self.on_reset
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            self.toolbar_frame, text="清除所有数据", command=self.on_clear_data
        ).pack(side=tk.LEFT, padx=(8, 0))

    def _build_action_toolbar(self) -> None:
        """Toolbar content for '操作'选项卡：搜索 + 排序。"""
        self._clear_toolbar()

        ttk.Label(self.toolbar_frame, text="关键字:").pack(side=tk.LEFT)
        search_entry = ttk.Entry(
            self.toolbar_frame, textvariable=self.search_var, width=24
        )
        search_entry.pack(side=tk.LEFT, padx=(4, 4))
        search_entry.bind("<Return>", lambda _e: self.on_search())
        ttk.Button(self.toolbar_frame, text="搜索", command=self.on_search).pack(
            side=tk.LEFT, padx=(0, 8)
        )

        ttk.Label(self.toolbar_frame, text="范围:").pack(side=tk.LEFT)
        search_scope_values = ["全部字段"] + DEFAULT_COLUMNS
        scope_combo = ttk.Combobox(
            self.toolbar_frame,
            textvariable=self.search_scope_var,
            values=search_scope_values,
            state="readonly",
            width=12,
        )
        scope_combo.pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(self.toolbar_frame, text="模式:").pack(side=tk.LEFT)
        mode_combo = ttk.Combobox(
            self.toolbar_frame,
            textvariable=self.search_mode_var,
            values=["模糊", "严格"],
            state="readonly",
            width=6,
        )
        mode_combo.pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(self.toolbar_frame, text="排序列:").pack(side=tk.LEFT)
        ttk.Label(
            self.toolbar_frame, textvariable=self.sort_col_label_var
        ).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Button(
            self.toolbar_frame,
            text="升序",
            command=lambda: self.on_sort_button(True),
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            self.toolbar_frame,
            text="降序",
            command=lambda: self.on_sort_button(False),
        ).pack(side=tk.LEFT, padx=(0, 4))

    def _build_view_toolbar(self) -> None:
        """Toolbar content for '视图'选项卡：字段选择 + 显示范围过滤 + 统计绘图。"""
        self._clear_toolbar()

        top_row = ttk.Frame(self.toolbar_frame)
        top_row.pack(side=tk.TOP, fill=tk.X)
        bottom_row = ttk.Frame(self.toolbar_frame)
        bottom_row.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        ttk.Label(top_row, text="字段:").pack(side=tk.LEFT)
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
            text="应用过滤",
            command=self.on_apply_view_filter,
        ).pack(side=tk.LEFT, padx=(4, 4))
        ttk.Button(
            top_row,
            text="清除过滤",
            command=self.on_clear_view_filter,
        ).pack(side=tk.LEFT, padx=(0, 4))

        self.view_info_var = tk.StringVar()
        ttk.Label(top_row, textvariable=self.view_info_var).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        ttk.Label(bottom_row, text="统计字段:").pack(side=tk.LEFT)
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
            text="绘制统计图",
            command=self.on_plot_stats,
        ).pack(side=tk.LEFT, padx=(0, 4))

        self._update_view_controls()

    def _build_st_toolbar(self) -> None:
        """Toolbar content for '时空统计'选项卡：选择时间列、节点列和时间粒度。"""
        self._clear_toolbar()

        top_row = ttk.Frame(self.toolbar_frame)
        top_row.pack(side=tk.TOP, fill=tk.X)
        bottom_row = ttk.Frame(self.toolbar_frame)
        bottom_row.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        ttk.Label(top_row, text="时间列:").pack(side=tk.LEFT)
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

        ttk.Label(top_row, text="节点列:").pack(side=tk.LEFT)
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

        ttk.Label(bottom_row, text="粒度(min):").pack(side=tk.LEFT)
        self.st_freq_var = tk.StringVar(value="5")
        freq_entry = ttk.Entry(
            bottom_row, textvariable=self.st_freq_var, width=6
        )
        freq_entry.pack(side=tk.LEFT, padx=(2, 8))

        ttk.Button(
            bottom_row, text="生成时空数据", command=self.on_generate_st
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(bottom_row, text="节点序号:").pack(side=tk.LEFT)
        self.st_node_index_var = tk.StringVar(value="")
        node_idx_entry = ttk.Entry(
            bottom_row, textvariable=self.st_node_index_var, width=8
        )
        node_idx_entry.pack(side=tk.LEFT, padx=(2, 4))

        ttk.Button(
            bottom_row, text="流量图", command=self.on_plot_flow
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            bottom_row, text="3D图", command=self.on_plot_st_3d
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.st_info_var = tk.StringVar()
        ttk.Label(bottom_row, textvariable=self.st_info_var).pack(
            side=tk.LEFT, padx=(8, 0)
        )

    def _rebuild_toolbar(self) -> None:
        mode = self.toolbar_mode.get()
        if mode == "文件":
            self._build_file_toolbar()
        elif mode == "操作":
            self._build_action_toolbar()
        elif mode == "视图":
            self._build_view_toolbar()
        else:
            self._build_st_toolbar()

        self._update_tab_styles()

    def _on_tab_click(self, mode: str) -> None:
        """Handle clicks on the '文件' / '操作' tab buttons."""
        if self.toolbar_mode.get() == mode:
            return
        self.toolbar_mode.set(mode)
        self._rebuild_toolbar()

    def _update_tab_styles(self) -> None:
        """Refresh tab button styles so 当前选中的标签更突出。"""
        current = self.toolbar_mode.get()
        if current == "文件":
            self.file_tab_btn.configure(style="TabActive.TButton")
            self.action_tab_btn.configure(style="Tab.TButton")
            self.view_tab_btn.configure(style="Tab.TButton")
            self.st_tab_btn.configure(style="Tab.TButton")
        elif current == "操作":
            self.file_tab_btn.configure(style="Tab.TButton")
            self.action_tab_btn.configure(style="TabActive.TButton")
            self.view_tab_btn.configure(style="Tab.TButton")
            self.st_tab_btn.configure(style="Tab.TButton")
        elif current == "视图":
            self.file_tab_btn.configure(style="Tab.TButton")
            self.action_tab_btn.configure(style="Tab.TButton")
            self.view_tab_btn.configure(style="TabActive.TButton")
            self.st_tab_btn.configure(style="Tab.TButton")
        else:
            self.file_tab_btn.configure(style="Tab.TButton")
            self.action_tab_btn.configure(style="Tab.TButton")
            self.view_tab_btn.configure(style="Tab.TButton")
            self.st_tab_btn.configure(style="TabActive.TButton")

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
            self.view_info_var.set("当前无数据可用，请先导入。")
            return

        kind, hint, categories = build_view_hint(df, col_name)

        if kind == "category":
            unique_vals = categories or []
            ttk.Label(self.view_controls_frame, text="值:").pack(side=tk.LEFT)
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
        """根据字段类型调用后端统计绘图函数。"""
        df = self._get_current_dataframe()
        if df.empty:
            messagebox.showinfo("统计", "当前没有数据可用于统计和绘图。")
            return

        col_var = getattr(self, "stats_field_var", None)
        col_name = col_var.get() if col_var is not None else None
        if not col_name or col_name not in df.columns:
            messagebox.showerror("错误", "请选择一个有效的字段。")
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
                    messagebox.showinfo("统计", "当前数据中没有有效的 TripLength 数值。")
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
                messagebox.showinfo("统计", f"暂不支持字段 {col_name} 的专门统计图。")

        except Exception as exc:
            logger.exception("统计绘图失败: %s", exc)
            messagebox.showerror("错误", f"统计绘图失败: {exc}")

    def on_generate_st(self) -> None:
        """Generate spatio-temporal dataset from current data."""
        if self.store is None or self.store.dataframe.empty:
            messagebox.showinfo("提示", "请先在“文件”选项卡中导入数据。")
            return

        base_df = self._get_current_dataframe()
        time_col = self.st_time_col_var.get()
        node_col = self.st_node_col_var.get()
        freq_str = self.st_freq_var.get().strip()

        if time_col not in base_df.columns or node_col not in base_df.columns:
            base_df = self.store.dataframe
            if time_col not in base_df.columns or node_col not in base_df.columns:
                messagebox.showerror("错误", "时间列或节点列无效。")
                return
        try:
            freq_min = int(freq_str)
            if freq_min <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "粒度必须是正整数（分钟）。")
            return

        self.st_shape_info = ""
        self.st_info_var.set("生成时空数据中，请稍候...")
        self._update_status()

        try:
            st_df = generate_spatiotemporal(base_df, time_col, node_col, freq_min)
        except Exception as exc:
            logger.exception("生成时空数据失败: %s", exc)
            messagebox.showerror("错误", f"生成时空数据失败: {exc}")
            self.st_info_var.set("生成时空数据失败。")
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
        self.st_info_var.set("时空数据生成完成。")
        self._update_status()

    def on_plot_flow(self) -> None:
        """Plot flow time series for a given node index from spatio-temporal dataset."""
        if self.st_df is None or self.st_df.empty:
            messagebox.showinfo("提示", "请先生成时空数据。")
            return

        idx_str = self.st_node_index_var.get().strip()
        if not idx_str:
            messagebox.showinfo("提示", "请输入节点序号（1 开始的整数）。")
            return
        try:
            node_idx = int(idx_str)
        except ValueError:
            messagebox.showerror("错误", "节点序号必须是整数。")
            return

        try:
            result = flow_timeseries(self.st_df, node_idx)
        except Exception as exc:
            logger.exception("流量图绘制失败: %s", exc)
            messagebox.showerror("错误", f"流量图绘制失败: {exc}")
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
            messagebox.showinfo("提示", "请先生成时空数据。")
            return

        try:
            result = st_3d_surface(self.st_df)
        except Exception as exc:
            logger.exception("3D 图绘制失败: %s", exc)
            messagebox.showerror("错误", f"3D 图绘制失败: {exc}")
            return

        shape = result.get("shape", self.st_df.shape)
        fig = result["figure"]

        stats_text = (
            "Spatio-temporal 3D surface\n\n"
            f"Time bins (shown): {shape[0]}\n"
            f"Nodes (shown): {shape[1]}\n"
        )

        self._show_figure(fig, stats_text)

    def on_close_st_view(self) -> None:
        """
        Global back action:

        - If当前有内嵌图像（数据/视图/时空统计任意模块），先关闭图像并回到当前表格视图；
        - 否则若当前处于“时空统计”视图，则恢复生成前的原始交通记录表；
        - 若以上条件均不满足，则提示当前没有可回退的内容。
        """
        if self.plot_frame.winfo_ismapped():
            logger.info("Closing embedded plot and returning to table view")
            self._close_plot()
            if self.toolbar_mode.get() == "时空统计":
                self.st_info_var.set("已返回时空表视图。")
                self._update_status()
            return

        if self.toolbar_mode.get() == "时空统计" and self.st_prev_df is not None:
            logger.info("Closing spatio-temporal view and restoring previous table view")
            self.current_df = self.st_prev_df
            self.current_page = self.st_prev_page
            self.table_columns = list(DEFAULT_COLUMNS)
            self._configure_table_columns()
            self._refresh_view()
            self.st_shape_info = ""
            self.st_info_var.set("已关闭时空表，恢复原始视图。")
            self._update_status()
            self.st_prev_df = None
            self.st_prev_page = 1
            return

        messagebox.showinfo("提示", "当前没有可回退的内容。")

    def on_apply_view_filter(self) -> None:
        """Apply view filter based on selected field and range/value."""
        if self.store is None or self.store.dataframe.empty:
            messagebox.showinfo("提示", "请先在“文件”选项卡中导入数据。")
            return

        base_df = self.store.dataframe
        col_name = self.view_field_var.get()
        if not col_name or col_name not in base_df.columns:
            messagebox.showerror("错误", "视图字段无效。")
            return

        try:
            kind = get_field_kind(col_name)
            if kind is None:
                messagebox.showerror("错误", f"暂不支持字段 {col_name} 的视图过滤。")
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
            logger.exception("视图过滤失败: %s", exc)
            messagebox.showerror("错误", f"视图过滤失败: {exc}")

    def on_clear_view_filter(self) -> None:
        """Clear view filter and show full dataset from store."""
        if self.store is None or self.store.dataframe.empty:
            messagebox.showinfo("提示", "当前没有数据需要清除过滤。")
            return
        self.current_df = self.store.dataframe.copy()
        self.current_page = 1
        self.view_from_var.set("")
        self.view_to_var.set("")
        self.view_value_var.set("")
        self._refresh_view()
        self.view_info_var.set("视图过滤已清除。")

    def on_close(self) -> None:
        """Handle window close: destroy GUI and terminate the process."""
        logger.info("TrafficLens GUI window closed by user")
        self.destroy()
        sys.exit(0)

    def on_clear_data(self) -> None:
        """Clear all loaded data and reset the view."""
        if not messagebox.askyesno("确认", "确定要清除当前加载的所有数据吗？"):
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
        """Export current result (current_df) in user-selected format."""
        if self.current_df is None or self.current_df.empty:
            messagebox.showinfo("提示", "当前没有可导出的数据。")
            return

        path = filedialog.asksaveasfilename(
            title="导出当前结果",
            defaultextension=".csv",
            filetypes=[
                ("CSV 文件", "*.csv"),
                ("Excel 文件", "*.xlsx"),
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
            df_to_export = self.current_df.copy()
            if fmt == "csv":
                df_to_export.to_csv(path, index=False, encoding="utf-8-sig")
            elif fmt == "xlsx":
                df_to_export.to_excel(path, index=False)
            else:
                np.save(path, df_to_export.to_numpy())
            messagebox.showinfo("成功", f"已导出数据：\n{path}")
        except Exception as exc:
            logger.exception("导出当前结果失败: %s", exc)
            messagebox.showerror("错误", f"导出当前结果失败: {exc}")
    def show_overview_stats(self) -> None:
        df = self._get_current_dataframe()
        if df.empty:
            messagebox.showinfo("统计", "当前没有数据可用于统计。")
            return
        stats = overview_stats(df)
        total_rows = stats["total_rows"]
        total_cols = stats["total_cols"]
        distinct_vehicle = stats["distinct_vehicle_types"]
        msg = (
            f"总体概览：\n\n"
            f"- 总行数: {total_rows}\n"
            f"- 总列数: {total_cols}\n"
            f"- 不同车型数量: {distinct_vehicle}\n"
        )
        messagebox.showinfo("统计 - 总体概览", msg)

    def show_vehicle_stats(self) -> None:
        df = self._get_current_dataframe()
        if df.empty:
            messagebox.showinfo("统计", "当前没有数据可用于统计。")
            return
        try:
            stats = vehicle_type_pie(df)
        except Exception as exc:
            logger.exception("VehicleType 统计失败: %s", exc)
            messagebox.showerror("错误", f"VehicleType 统计失败: {exc}")
            return

        top_lines = [f"{k}: {v}" for k, v in stats["top_counts"].items()]
        msg = (
            "按车型统计（VehicleType 频数）：\n\n"
            f"- 总样本数: {stats['total']}\n"
            f"- 不同 VehicleType 数量: {stats['n_unique']}\n"
            f"- 最常见的几类:\n  " + "\n  ".join(top_lines)
        )
        messagebox.showinfo("统计 - 按车型", msg)

    def show_triplength_stats(self) -> None:
        df = self._get_current_dataframe()
        if df.empty:
            messagebox.showinfo("统计", "当前没有数据可用于统计。")
            return
        stats = trip_length_hist(df)
        if stats is None:
            messagebox.showinfo("统计", "当前数据中没有有效的 TripLength 数值。")
            return
        msg = (
            "行程长度统计（TripLength）：\n\n"
            f"- 计数: {stats['count']:.0f}\n"
            f"- 最小值: {stats['min']:.2f}\n"
            f"- 平均值: {stats['mean']:.2f}\n"
            f"- 标准差: {stats['std']:.2f}\n"
            f"- 最大值: {stats['max']:.2f}\n"
        )
        messagebox.showinfo("统计 - 行程长度", msg)


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
            messagebox.showinfo("提示", "当前没有数据，无法跳转页码。")
            return

        max_page = max((total - 1) // self.page_size + 1, 1)
        try:
            page = int(raw)
        except ValueError:
            messagebox.showerror("错误", "页码必须是正整数。")
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
            title="选择要导入的 CSV 文件",
            filetypes=[("CSV 文件", "*.csv"), ("All Files", "*.*")],
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
            messagebox.showinfo("成功", f"已导入文件：{file_path.name}")
        except Exception as exc:
            logger.exception("Import file failed: %s", exc)
            messagebox.showerror("错误", f"导入文件失败: {exc}")

    def on_import_folder(self) -> None:
        folder = filedialog.askdirectory(title="选择要导入的文件夹")
        if not folder:
            return
        try:
            folder_path = Path(folder)
            files = sorted(folder_path.glob("*.csv"))
            if not files:
                messagebox.showinfo("提示", "该文件夹下没有找到 CSV 文件。")
                return
            logger.info("Import folder: %s (%d files)", folder_path, len(files))
            self.store = TrafficDataStore.from_files(files)
            self.current_df = self.store.dataframe
            self.current_page = 1
            self._refresh_view()
            messagebox.showinfo("成功", f"已从文件夹导入并合并 {len(files)} 个 CSV 文件。")
        except Exception as exc:
            logger.exception("Import folder failed: %s", exc)
            messagebox.showerror("错误", f"导入文件夹失败: {exc}")

    def on_search(self) -> None:
        keyword = self.search_var.get().strip()
        try:
            if not self._ensure_data_loaded():
                return
            logger.info("Search with keyword: %s", keyword)
            base_df = self.store.dataframe
            if keyword:
                scope = getattr(self, "search_scope_var", None)
                scope_value = scope.get() if scope is not None else "全部字段"
                search_column = None if scope_value == "全部字段" else scope_value
                mode = getattr(self, "search_mode_var", None)
                mode_value = mode.get() if mode is not None else "模糊"
                strict = mode_value == "严格"

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
            messagebox.showerror("错误", f"搜索失败: {exc}")

    def on_sort_button(self, ascending: bool) -> None:
        """Sort using the column selected in the table, triggered by buttons."""
        try:
            if not self._ensure_data_loaded():
                return
            col = getattr(self, "selected_sort_col", None) or DEFAULT_COLUMNS[0]
            logger.info("Button sort column=%s ascending=%s", col, ascending)
            self.current_df = self.current_df.sort_values(by=col, ascending=ascending)
            self.sorted_col = col
            self.sorted_ascending = ascending
            self.current_page = 1
            self._refresh_view()
        except Exception as exc:
            logger.exception("Button sort failed: %s", exc)
            messagebox.showerror("错误", f"排序失败: {exc}")

    def on_merge(self) -> None:
        if not self._ensure_data_loaded():
            return
        paths = filedialog.askopenfilenames(
            title="选择要合并的 CSV 文件",
            filetypes=[("CSV 文件", "*.csv"), ("All Files", "*.*")],
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
            messagebox.showinfo("成功", "已成功合并选中的 CSV 文件。")
        except Exception as exc:
            logger.exception("Merge failed: %s", exc)
            messagebox.showerror("错误", f"合并 CSV 失败: {exc}")

    def on_merge_folder(self) -> None:
        if not self._ensure_data_loaded():
            return
        folder = filedialog.askdirectory(title="选择要合并的文件夹")
        if not folder:
            return
        files = sorted(Path(folder).glob("*.csv"))
        if not files:
            messagebox.showinfo("提示", "该文件夹下没有找到 CSV 文件。")
            return
        try:
            logger.info("Merging CSV folder: %s (%d files)", folder, len(files))
            self.store = self.store.merge_with_files(files)
            self.current_df = self.store.dataframe.copy()
            self.current_page = 1
            self.on_search()
            messagebox.showinfo("成功", "已成功将文件夹中的 CSV 合并到当前数据。")
        except Exception as exc:
            logger.exception("Merge folder failed: %s", exc)
            messagebox.showerror("错误", f"合并文件夹失败: {exc}")

    def on_export(self) -> None:
        if not self._ensure_data_loaded():
            return
        path = filedialog.asksaveasfilename(
            title="导出当前结果为 CSV",
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            logger.info("Exporting current view to %s", path)
            self.store.export_csv(Path(path), df=self.current_df)
            messagebox.showinfo("成功", "导出成功。")
        except Exception as exc:
            logger.exception("Export failed: %s", exc)
            messagebox.showerror("错误", f"导出失败: {exc}")

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


