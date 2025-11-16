"""
Utilities for loading and querying traffic CSV data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import DATA_DIR, DEFAULT_COLUMNS


@dataclass
class TrafficDataStore:
    """In-memory store for traffic records."""

    dataframe: pd.DataFrame

    @classmethod
    def empty(cls) -> "TrafficDataStore":
        """Create an empty store with the default columns."""
        df = pd.DataFrame(columns=DEFAULT_COLUMNS)
        return cls(df)

    @classmethod
    def from_files(cls, files: List[Path]) -> "TrafficDataStore":
        frames: List[pd.DataFrame] = []
        for f in files:
            p = Path(f)
            if not p.exists():
                continue
            df = pd.read_csv(p, header=None)
            if len(df.columns) == len(DEFAULT_COLUMNS):
                df.columns = DEFAULT_COLUMNS
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=DEFAULT_COLUMNS
        )
        return cls(combined)

    @classmethod
    def from_data_dir(cls, pattern: str = "*.csv") -> "TrafficDataStore":
        files = sorted(DATA_DIR.glob(pattern))
        return cls.from_files(files)

    def search(
        self, keyword: str, column: Optional[str] = None, strict: bool = False
    ) -> pd.DataFrame:
        """
        Search keyword in the dataframe.

        - If column is None: search across all string columns (全局搜索)
        - If column is given: only search in that specific column
        - If strict is True: 使用严格匹配（等于）；否则使用模糊匹配（包含）
        """
        if not keyword:
            return self.dataframe.copy()
        keyword_lower = str(keyword).lower()

        # 搜索指定列
        if column:
            if column not in self.dataframe.columns:
                return self.dataframe.iloc[0:0].copy()
            ser_str = self.dataframe[column].astype(str)
            if strict:
                mask = ser_str == keyword
            else:
                mask = ser_str.str.lower().str.contains(keyword_lower, na=False)
            return self.dataframe[mask].copy()

        # 全局搜索：遍历所有字符串列
        mask = pd.Series(False, index=self.dataframe.index)
        for col in self.dataframe.columns:
            if self.dataframe[col].dtype == object:
                ser_str = self.dataframe[col].astype(str)
                if strict:
                    mask |= ser_str == keyword
                else:
                    mask |= ser_str.str.lower().str.contains(
                        keyword_lower, na=False
                    )
        return self.dataframe[mask].copy()

    def sort(self, column: str, ascending: bool = True) -> pd.DataFrame:
        if column not in self.dataframe.columns:
            raise ValueError(f"Column '{column}' not found.")
        return self.dataframe.sort_values(by=column, ascending=ascending).copy()

    def merge_with_files(self, files: List[Path]) -> "TrafficDataStore":
        """Return a new store with current data plus additional CSV files."""
        if not files:
            return self
        extra = TrafficDataStore.from_files(files)
        merged_df = pd.concat([self.dataframe, extra.dataframe], ignore_index=True)
        return TrafficDataStore(merged_df)

    def export_csv(self, path: Path, df: Optional[pd.DataFrame] = None) -> None:
        target = df if df is not None else self.dataframe
        path.parent.mkdir(parents=True, exist_ok=True)
        target.to_csv(path, index=False, encoding="utf-8-sig")


