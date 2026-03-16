"""Common utilities for shift profitability: CSV read, ID/string cleaning, rounding, column lookup."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def clean_id_series(s: pd.Series) -> pd.Series:
    raw = s.astype("string").str.strip()
    raw = raw.replace(["", "nan", "None", "<NA>"], pd.NA)

    num = pd.to_numeric(raw, errors="coerce")
    is_int_like = num.notna() & (np.floor(num) == num)
    raw = raw.where(~is_int_like, num.astype("Int64").astype("string"))
    raw = raw.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
    return raw


def clean_str_series(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.replace("", pd.NA)
    return s


def safe_round(df: pd.DataFrame, col: str, ndigits: int) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(ndigits)


def first_existing_col_case_insensitive(
    df: pd.DataFrame, candidates: List[str]
) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        hit = lower_map.get(cand.lower())
        if hit is not None:
            return hit
    return None


def to_bool_series(s: pd.Series) -> pd.Series:
    """
    Convert a Series to reliable boolean without triggering pandas FutureWarning
    about silent downcasting during fillna on object dtype.
    """
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    ss = s.astype("string")
    ss = ss.str.strip().str.lower()
    ss = ss.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "<na>": pd.NA})

    true_vals = {"true", "t", "yes", "y", "1"}
    false_vals = {"false", "f", "no", "n", "0"}

    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out = out.mask(ss.isin(true_vals), True)
    out = out.mask(ss.isin(false_vals), False)

    num = pd.to_numeric(ss, errors="coerce")
    out = out.mask(num.notna(), num != 0)

    return out.fillna(False).astype(bool)
