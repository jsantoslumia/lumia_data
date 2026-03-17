"""CSV write and enriched visit export write (for CLI)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from shift_profitability_lib.cost_allocation import (
    apply_helper_hours_cost_allocation_to_visits,
)
from shift_profitability_lib.utils import clean_id_series, safe_round
from shift_profitability_lib.visits import build_enriched_visits_export


def write_csv(df: pd.DataFrame, path: Path, utf8_bom: bool) -> None:
    enc = "utf-8-sig" if utf8_bom else "utf-8"
    df.to_csv(path, index=False, encoding=enc)


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
        visits_enriched["sah_revenue"] = pd.to_numeric(
            visits_enriched["sah_revenue"], errors="coerce"
        )
        safe_round(visits_enriched, "sah_revenue", 2)

        if "actual_visit_hours" not in visits_enriched.columns:
            raise ValueError(
                "Visits export must contain actual_visit_hours to allocate sah_revenue to sah_visit_revenue."
            )
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
        visits_enriched["sah_visit_revenue"] = visits_enriched["sah_revenue"] * share
        safe_round(visits_enriched, "sah_visit_revenue", 2)

    if "allocation_ok" not in visits_enriched.columns:
        visits_enriched["allocation_ok"] = False
    else:
        visits_enriched["allocation_ok"] = visits_enriched["allocation_ok"].astype(bool)
    if "allocation_method" not in visits_enriched.columns:
        visits_enriched["allocation_method"] = ""
    if "allocation_reason" not in visits_enriched.columns:
        visits_enriched["allocation_reason"] = ""

    write_csv(visits_enriched, out_path, utf8_bom=utf8_bom)
    print(
        f"Wrote: {out_path.resolve()}  (rows={len(visits_enriched):,}, unique visit_shift_id={visits_enriched['visit_shift_id'].nunique():,})"
    )
