"""CSV write and enriched visit export write (for CLI)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from shift_profitability_lib.cost_allocation import (
    apply_helper_hours_cost_allocation_to_visits,
)
from shift_profitability_lib.utils import clean_id_series
from shift_profitability_lib.visits import build_enriched_visits_export

def _norm_col(c: object) -> str:
    return str(c).strip().lower().replace(" ", "_")


# Strip these (and case/space variants) before writing canonical sah_* columns
_SAH_DROP_NORMALIZED = frozenset(
    {"sah_revenue", "sah_visit_revenue", "sah_revenue_x", "sah_revenue_y"}
)


def write_csv(df: pd.DataFrame, path: Path, utf8_bom: bool) -> None:
    enc = "utf-8-sig" if utf8_bom else "utf-8"
    df.to_csv(path, index=False, encoding=enc)


def _dataframe_without_sah_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any prior SAH-related columns so we write exactly one sah_revenue / sah_visit_revenue."""
    keep = [c for c in df.columns if _norm_col(c) not in _SAH_DROP_NORMALIZED]
    return df.loc[:, keep].copy()


def _sah_revenue_after_merge(df: pd.DataFrame) -> pd.Series:
    """Resolve sah_revenue after merge (handles duplicate column rename)."""
    for name in ("sah_revenue", "sah_revenue_y"):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def write_enriched_visits_export(
    out_path: Path,
    visits_csv: str,
    shift_profitability_feed: pd.DataFrame,
    exclude_zero_revenue_visits: bool,
    dva_claims_csv: Optional[str],
    vhc_claims_csv: Optional[str],
    chsp_claims_csv: Optional[str],
    sah_revenue_by_membership: Optional[pd.DataFrame],
    utf8_bom: bool,
    *,
    class_mapping_excel: Optional[str] = None,
    costs_csv: Optional[str] = None,
) -> None:
    """Build enriched visits (claim pricing + cost allocation), optionally add SAH revenue, ensure columns, write CSV."""
    visits_enriched = build_enriched_visits_export(
        visits_csv=visits_csv,
        exclude_zero_revenue_visits=exclude_zero_revenue_visits,
        dva_claims_csv=dva_claims_csv,
        vhc_claims_csv=vhc_claims_csv,
        chsp_claims_csv=chsp_claims_csv,
        class_mapping_excel=class_mapping_excel,
    )
    visits_enriched = apply_helper_hours_cost_allocation_to_visits(
        visits_enriched=visits_enriched,
        shift_profitability_feed=shift_profitability_feed,
        costs_csv=costs_csv,
        class_mapping_excel=class_mapping_excel,
    )

    if sah_revenue_by_membership is not None:
        if "membership_uuid" not in visits_enriched.columns:
            raise ValueError(
                "Visits export must contain membership_uuid to add sah_revenue."
            )
        visits_enriched["membership_uuid"] = clean_id_series(
            visits_enriched["membership_uuid"]
        )
        visits_enriched = visits_enriched.merge(
            sah_revenue_by_membership, on="membership_uuid", how="left"
        )
        if "actual_visit_hours" not in visits_enriched.columns:
            raise ValueError(
                "Visits export must contain actual_visit_hours to allocate sah_revenue to sah_visit_revenue."
            )
        sah_rev = _sah_revenue_after_merge(visits_enriched)
        visits_enriched["actual_visit_hours"] = pd.to_numeric(
            visits_enriched["actual_visit_hours"], errors="coerce"
        ).fillna(0.0)
        total_hours = visits_enriched.groupby("membership_uuid")[
            "actual_visit_hours"
        ].transform("sum")
        share = np.where(
            total_hours > 0,
            visits_enriched["actual_visit_hours"] / total_hours,
            0.0,
        )
        sah_visit = pd.to_numeric(sah_rev, errors="coerce").fillna(0.0) * share
        sah_rev = pd.to_numeric(sah_rev, errors="coerce")
        sah_rev = sah_rev.round(2)
        sah_visit = pd.Series(sah_visit, index=visits_enriched.index).round(2)
    else:
        sah_rev = pd.Series(np.nan, index=visits_enriched.index, dtype=float)
        sah_visit = pd.Series(np.nan, index=visits_enriched.index, dtype=float)

    visits_enriched = _dataframe_without_sah_columns(visits_enriched)
    visits_enriched["sah_revenue"] = sah_rev.values
    visits_enriched["sah_visit_revenue"] = sah_visit.values

    if "allocation_ok" not in visits_enriched.columns:
        visits_enriched["allocation_ok"] = False
    else:
        visits_enriched["allocation_ok"] = visits_enriched["allocation_ok"].astype(bool)
    if "allocation_method" not in visits_enriched.columns:
        visits_enriched["allocation_method"] = ""
    if "allocation_reason" not in visits_enriched.columns:
        visits_enriched["allocation_reason"] = ""

    if "sah_visit_revenue" not in visits_enriched.columns or "sah_revenue" not in visits_enriched.columns:
        raise RuntimeError("Internal error: sah columns missing before CSV write.")
    write_csv(visits_enriched, out_path, utf8_bom=utf8_bom)
    print(
        f"Wrote: {out_path.resolve()}  (rows={len(visits_enriched):,}, unique visit_shift_id={visits_enriched['visit_shift_id'].nunique():,})"
    )
