"""
로컬 저장 모듈

변환된 13개 테이블을 CSV + Parquet 형식으로 로컬에 저장한다.

저장 경로:
  02_processed_data/csv/final/     ← CSV (utf-8-sig)
  02_processed_data/parquet/final/  ← Parquet
"""
from pathlib import Path
from typing import Dict
import pandas as pd

try:
    from .transformer import TABLE_ORDER
except ImportError:
    from transformer import TABLE_ORDER


class LocalStorage:
    """로컬 CSV + Parquet 저장"""

    def __init__(self, base_dir: str = "02_processed_data"):
        self.base_dir = Path(base_dir)
        self.csv_dir = self.base_dir / "csv" / "final"
        self.parquet_dir = self.base_dir / "parquet" / "final"

    def save_all(
        self,
        tables: Dict[str, pd.DataFrame],
        save_csv: bool = True,
        save_parquet: bool = True,
    ) -> Dict[str, dict]:
        """
        13개 테이블을 로컬에 저장

        Parameters
        ----------
        tables : {테이블명: DataFrame} dict
        save_csv : CSV 저장 여부
        save_parquet : Parquet 저장 여부

        Returns
        -------
        {테이블명: {"csv": path, "parquet": path, "rows": int}}
        """
        results = {}

        if save_csv:
            self.csv_dir.mkdir(parents=True, exist_ok=True)
        if save_parquet:
            self.parquet_dir.mkdir(parents=True, exist_ok=True)

        for name in TABLE_ORDER:
            df = tables.get(name)
            if df is None:
                continue

            info = {"rows": len(df)}

            if save_csv:
                csv_path = self.csv_dir / f"{name}.csv"
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                info["csv"] = str(csv_path)

            if save_parquet:
                parquet_path = self.parquet_dir / f"{name}.parquet"
                df.to_parquet(parquet_path, index=False)
                info["parquet"] = str(parquet_path)

            results[name] = info

        return results

    def save_table(
        self,
        name: str,
        df: pd.DataFrame,
        save_csv: bool = True,
        save_parquet: bool = True,
    ):
        """단일 테이블 저장"""
        if save_csv:
            self.csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = self.csv_dir / f"{name}.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        if save_parquet:
            self.parquet_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = self.parquet_dir / f"{name}.parquet"
            df.to_parquet(parquet_path, index=False)
