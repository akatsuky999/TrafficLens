"""
Tkinter GUI for TrafficLens.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .config import DEFAULT_COLUMNS
from .data_loader import TrafficDataStore
from .stats import overview_stats, vehicle_type_counts, trip_length_stats


logger = logging.getLogger(__name__)


class TrafficLensApp(tk.Tk):
    def __init__(self, store: Optional[TrafficDataStore] = None) -> None:
        super().__init__()
        self.title("TrafficLens - 交通数据查询系统")
        self.geometry("1100x600")
        # 设置窗口最小尺寸，避免缩得太小时底部状态栏被完全挤掉
        self.minsize(900, 500)

        # Do not load any data at startup; wait for user to import.
        self.store: Optional[TrafficDataStore] = store
        if self.store is not None:
            self.current_df: pd.DataFrame = self.store.dataframe.copy()
        else:
            self.current_df = pd.DataFrame(columns=DEFAULT_COLUMNS)

        # Pagination settings to avoid loading huge tables into the widget
        self.page_size: int = 1000  # rows per page
        self.current_page: int = 1

        # Sorting state: last sorted column + direction
        self.sorted_col: Optional[str] = None
        self.sorted_ascending: bool = True
        # Column chosen by user (via clicking the table) for button-based sorting
        self.selected_sort_col: str = DEFAULT_COLUMNS[0]
        self.sort_col_label_var = tk.StringVar(value=DEFAULT_COLUMNS[0])

        # Page input state (for bottom jump-to-page control)
        self.page_input_var: tk.StringVar | None = None

        # Search state variables (used by the search/sort toolbar)
        self.search_var = tk.StringVar()
        self.search_scope_var = tk.StringVar(value="全部字段")
        self.search_mode_var = tk.StringVar(value="模糊")

        # Toolbar(tab) mode: '文件' / '操作' / '数据'
        self.toolbar_mode = tk.StringVar(value="操作")

        # Last search state (for highlighting and match count)
        self.last_search_keyword: Optional[str] = None
        self.last_search_column: Optional[str] = None
        self.last_search_strict: bool = False
        self.last_match_count: Optional[int] = None

        self._build_widgets()
        self._refresh_view()
        logger.info("TrafficLens GUI started with no data loaded.")

    # ------------------------------------------------------------------ UI
    def _build_widgets(self) -> None:
        # 不使用传统菜单栏，改用自定义“选项卡 + 工具栏”布局

        # Toolbar tabs（类似 Excel 顶部选项卡：文件 / 操作）
        tab_frame = ttk.Frame(self)
        tab_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 0))

        # 自定义选项卡按钮：通过样式高亮当前选中的标签
        self.tab_style = ttk.Style(self)
        self.tab_style.configure("Tab.TButton", padding=(6, 2))
        self.tab_style.configure(
            "TabActive.TButton", padding=(6, 2), relief="sunken"
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

        self.data_tab_btn = ttk.Button(
            tab_frame,
            text="数据",
            command=lambda: self._on_tab_click("数据"),
            width=8,
        )
        self.data_tab_btn.pack(side=tk.LEFT)

        # Actual toolbar area under the tabs
        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 4))
        self._rebuild_toolbar()

        # Table frame
        table_frame = ttk.Frame(self)
        table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        # Excel-like Treeview style
        style = ttk.Style(self)
        try:
            # 在 Windows 上优先使用更现代的主题
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
            table_frame,
            columns=DEFAULT_COLUMNS,
            show="headings",
            height=20,
            style="Excel.Treeview",
        )

        # 为不同列设置更合适的初始宽度，最后一列 TripInformation 给更大空间
        column_widths = {
            "VehicleType": 90,
            "DetectionTime_O": 160,
            "GantryID_O": 110,
            "DetectionTime_D": 160,
            "GantryID_D": 110,
            "TripLength": 90,
            "TripEnd": 80,
            "TripInformation": 420,
        }

        for col in DEFAULT_COLUMNS:
            # 先显示纯文本表头，排序由上方按钮触发
            self.tree.heading(col, text=col)
            width = column_widths.get(col, 120)
            # 最后一列允许随窗口伸缩，便于查看长文本
            stretch = col == "TripInformation"
            self.tree.column(col, width=width, minwidth=80, anchor=tk.W, stretch=stretch)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # 条纹行效果，类似 Excel 交替行底色
        self.tree.tag_configure("evenrow", background="#FFFFFF")
        self.tree.tag_configure("oddrow", background="#F7F7F7")

        # 记录用户点击的列，用于按钮排序
        self.tree.bind("<ButtonRelease-1>", self.on_tree_click)

        # 高亮匹配行的样式（搜索结果）
        self.tree.tag_configure("matchrow", background="#FFF4CC")

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        # Status and pagination controls
        status_frame = ttk.Frame(self)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 8))

        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)

        # Jump-to-page controls
        self.page_input_var = tk.StringVar(value="1")
        ttk.Label(status_frame, text="  跳转到第").pack(side=tk.LEFT)
        page_entry = ttk.Entry(status_frame, textvariable=self.page_input_var, width=5)
        page_entry.pack(side=tk.LEFT)
        ttk.Label(status_frame, text="页").pack(side=tk.LEFT)
        ttk.Button(status_frame, text="跳转", command=self.on_jump_page).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        ttk.Button(status_frame, text="上一页", command=self.on_prev_page).pack(
            side=tk.RIGHT, padx=(4, 0)
        )
        ttk.Button(status_frame, text="下一页", command=self.on_next_page).pack(
            side=tk.RIGHT
        )

    # ------------------------------------------------------------------ Data operations
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
            values = [row.get(col, "") for col in DEFAULT_COLUMNS]
            tags = ["evenrow" if idx % 2 == 0 else "oddrow"]

            # Highlight rows that match the last search criteria
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

        # 同步底部页码输入框
        if self.page_input_var is not None:
            self.page_input_var.set(str(self.current_page))

    def _ensure_data_loaded(self) -> bool:
        """Return True if data is available, otherwise show a hint and return False."""
        if self.store is None or self.store.dataframe.empty:
            messagebox.showinfo("提示", "请先通过菜单“文件 -> 导入...”导入数据。")
            return False
        return True

    # ------------------------------------------------------------------ Toolbar builders
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

    def _build_action_toolbar(self) -> None:
        """Toolbar content for '操作'选项卡：搜索 + 排序。"""
        self._clear_toolbar()

        # Search controls
        ttk.Label(self.toolbar_frame, text="关键字:").pack(side=tk.LEFT)
        search_entry = ttk.Entry(
            self.toolbar_frame, textvariable=self.search_var, width=24
        )
        search_entry.pack(side=tk.LEFT, padx=(4, 4))
        search_entry.bind("<Return>", lambda _e: self.on_search())
        ttk.Button(self.toolbar_frame, text="搜索", command=self.on_search).pack(
            side=tk.LEFT, padx=(0, 8)
        )

        # Scope
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

        # Mode
        ttk.Label(self.toolbar_frame, text="模式:").pack(side=tk.LEFT)
        mode_combo = ttk.Combobox(
            self.toolbar_frame,
            textvariable=self.search_mode_var,
            values=["模糊", "严格"],
            state="readonly",
            width=6,
        )
        mode_combo.pack(side=tk.LEFT, padx=(2, 8))

        # Sort buttons
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

    def _build_data_toolbar(self) -> None:
        """Toolbar content for '数据'选项卡：基本统计信息。"""
        self._clear_toolbar()

        ttk.Button(
            self.toolbar_frame,
            text="总体概览",
            command=self.show_overview_stats,
        ).pack(side=tk.LEFT, padx=(0, 4))

        ttk.Button(
            self.toolbar_frame,
            text="按车型统计",
            command=self.show_vehicle_stats,
        ).pack(side=tk.LEFT, padx=(0, 4))

        ttk.Button(
            self.toolbar_frame,
            text="行程长度统计",
            command=self.show_triplength_stats,
        ).pack(side=tk.LEFT, padx=(0, 4))

    def _rebuild_toolbar(self) -> None:
        mode = self.toolbar_mode.get()
        if mode == "文件":
            self._build_file_toolbar()
        elif mode == "操作":
            self._build_action_toolbar()
        else:
            self._build_data_toolbar()

        # 更新 tab 高亮状态
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
            self.data_tab_btn.configure(style="Tab.TButton")
        elif current == "操作":
            self.file_tab_btn.configure(style="Tab.TButton")
            self.action_tab_btn.configure(style="TabActive.TButton")
            self.data_tab_btn.configure(style="Tab.TButton")
        else:  # 数据
            self.file_tab_btn.configure(style="Tab.TButton")
            self.action_tab_btn.configure(style="Tab.TButton")
            self.data_tab_btn.configure(style="TabActive.TButton")

    # ------------------------------------------------------------------ Data stats handlers
    def _get_current_dataframe(self) -> pd.DataFrame:
        """Helper to get current full dataframe (not only current page)."""
        if not self._ensure_data_loaded():
            return pd.DataFrame(columns=DEFAULT_COLUMNS)
        # current_df 可能已经被过滤/排序，这里按当前视图统计即可
        return self.current_df

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
        counts = vehicle_type_counts(df)
        lines = [f"{idx}: {val}" for idx, val in counts.items()]
        msg = "按车型统计（VehicleType 频数）：\n\n" + "\n".join(lines[:50])
        if len(lines) > 50:
            msg += f"\n\n(仅显示前 50 项，共 {len(lines)} 项)"
        messagebox.showinfo("统计 - 按车型", msg)

    def show_triplength_stats(self) -> None:
        df = self._get_current_dataframe()
        if df.empty:
            messagebox.showinfo("统计", "当前没有数据可用于统计。")
            return
        stats = trip_length_stats(df)
        if stats is None:
            messagebox.showinfo("统计", "当前数据中没有有效的 TripLength 数值。")
            return
        msg = (
            "行程长度统计（TripLength）：\n\n"
            f"- 计数: {stats['count']:.0f}\n"
            f"- 最小值: {stats['min']:.2f}\n"
            f"- 平均值: {stats['mean']:.2f}\n"
            f"- 最大值: {stats['max']:.2f}\n"
        )
        messagebox.showinfo("统计 - 行程长度", msg)

    # ------------------------------------------------------------------ Event handlers

    def on_tree_click(self, event) -> None:
        """Track which column user clicked, for button-based sorting."""
        col_id = self.tree.identify_column(event.x)  # e.g. '#1'
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
            # 没有数据时直接返回
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
            self.current_df = self.store.dataframe.copy()
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
                # Determine search scope: global or specific column
                scope = getattr(self, "search_scope_var", None)
                scope_value = scope.get() if scope is not None else "全部字段"
                search_column = None if scope_value == "全部字段" else scope_value
                # Determine search mode
                mode = getattr(self, "search_mode_var", None)
                mode_value = mode.get() if mode is not None else "模糊"
                strict = mode_value == "严格"

                # Use store.search with optional column and mode
                filtered = self.store.search(
                    keyword, column=search_column, strict=strict
                )
            else:
                filtered = base_df.copy()

            # Update last search state for highlighting and status
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
            # 记录最后一次排序状态
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
            # After merge, reset view and re-apply current search and sort
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
        # 清空搜索状态和高亮
        self.last_search_keyword = None
        self.last_search_column = None
        self.last_search_strict = False
        self.last_match_count = None
        self.current_df = self.store.dataframe.copy()
        self.current_page = 1
        self._refresh_view()

    # ------------------------------------------------------------------ Pagination handlers
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


