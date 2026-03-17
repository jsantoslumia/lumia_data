"""Merge Excel 'Class' sheet onto visits by visit_rate."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd


def _find_col_ci(df: pd.DataFrame, name: str) -> Optional[str]:
    w = name.lower()
    for c in df.columns:
        if str(c).strip().lower() == w:
            return c
    return None


def _visit_rate_merge_key(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    str_key = series.astype("string").str.strip()
    return num.where(num.notna(), str_key)


def merge_visit_class_from_excel(
    visits: pd.DataFrame, mapping_excel: Union[str, Path]
) -> pd.DataFrame:
    """
    Left-join sheet 'Class' (columns visit_rate, Class) onto visits.
    """
    path = Path(mapping_excel)
    vr_visits = _find_col_ci(visits, "visit_rate")
    if not vr_visits:
        raise ValueError(
            "Visits must contain visit_rate when applying class mapping."
        )
    try:
        class_df = pd.read_excel(path, sheet_name="Class")
    except ValueError as e:
        raise ValueError(f"Could not read sheet 'Class' from {path}: {e}") from e
    vr_map = _find_col_ci(class_df, "visit_rate")
    class_col = _find_col_ci(class_df, "Class")
    if not vr_map or not class_col:
        raise ValueError("Sheet 'Class' must contain visit_rate and Class columns.")
    m = class_df[[vr_map, class_col]].copy()
    m = m.rename(columns={vr_map: "_vr_map", class_col: "Class"})
    m = m.loc[m["_vr_map"].notna()].drop_duplicates(subset=["_vr_map"], keep="first")
    m["_vrk"] = _visit_rate_merge_key(m["_vr_map"])
    m = m.loc[m["_vrk"].notna()].drop_duplicates(subset=["_vrk"], keep="first")

    out = visits.copy()
    for c in list(out.columns):
        if str(c).strip().lower() == "class":
            out = out.drop(columns=[c])
            break
    out["_vrk"] = _visit_rate_merge_key(out[vr_visits])
    out = out.merge(m[["Class", "_vrk"]], on="_vrk", how="left")
    return out.drop(columns=["_vrk"])
