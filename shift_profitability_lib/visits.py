"""Visit normalization and enriched visit export (claim pricing applied)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from shift_profitability_lib.claim_pricing import apply_claim_pricing_to_visits
from shift_profitability_lib.utils import clean_id_series, clean_str_series, read_csv


def normalize_visits_for_feed(
    visits: pd.DataFrame,
    *,
    ensure_helper_id: bool = False,
    ensure_hours_columns: bool = False,
) -> pd.DataFrame:
    """
    Standardize visit_shift_id, helper_id, visit_projected_price, hours, and
    membership columns on a visits DataFrame.
    """
    if "visit_shift_id" not in visits.columns and "shift_id" in visits.columns:
        visits = visits.rename(columns={"shift_id": "visit_shift_id"})
    if "visit_shift_id" not in visits.columns:
        raise ValueError("Visits CSV must contain visit_shift_id (or shift_id).")

    visits["visit_shift_id"] = clean_id_series(visits["visit_shift_id"])

    if "helper_id" in visits.columns:
        visits["helper_id"] = clean_id_series(visits["helper_id"])
    elif ensure_helper_id:
        visits["helper_id"] = pd.NA

    if "visit_projected_price" not in visits.columns:
        raise ValueError("Visits CSV must contain visit_projected_price.")
    visits["visit_projected_price"] = pd.to_numeric(
        visits["visit_projected_price"], errors="coerce"
    ).fillna(0.0)

    if "projected_visit_hours" in visits.columns:
        visits["projected_visit_hours"] = pd.to_numeric(
            visits["projected_visit_hours"], errors="coerce"
        )
    elif ensure_hours_columns:
        visits["projected_visit_hours"] = np.nan

    if "actual_visit_hours" in visits.columns:
        visits["actual_visit_hours"] = pd.to_numeric(
            visits["actual_visit_hours"], errors="coerce"
        )
    elif ensure_hours_columns:
        visits["actual_visit_hours"] = np.nan

    membership_community_col = (
        "membership_community_name"
        if "membership_community_name" in visits.columns
        else None
    )
    membership_scheme_col = (
        "membership_funding_scheme"
        if "membership_funding_scheme" in visits.columns
        else None
    )
    if membership_community_col:
        visits[membership_community_col] = clean_str_series(
            visits[membership_community_col]
        )
    if membership_scheme_col:
        visits[membership_scheme_col] = clean_str_series(visits[membership_scheme_col])

    return visits


def build_enriched_visits_export(
    visits_csv: str,
    exclude_zero_revenue_visits: bool = False,
    dva_claims_csv: Optional[str] = None,
    vhc_claims_csv: Optional[str] = None,
    chsp_claims_csv: Optional[str] = None,
    class_mapping_excel: Optional[str] = None,
) -> pd.DataFrame:
    """Read the raw visit export and return an enriched visit-level table with claim pricing applied."""
    visits = read_csv(visits_csv)
    if class_mapping_excel:
        from shift_profitability_lib.class_mapping import merge_visit_class_from_excel

        visits = merge_visit_class_from_excel(visits, class_mapping_excel)

    if "visit_shift_id" not in visits.columns and "shift_id" in visits.columns:
        visits = visits.rename(columns={"shift_id": "visit_shift_id"})
    if "visit_shift_id" not in visits.columns:
        raise ValueError("Visits CSV must contain visit_shift_id (or shift_id).")

    original_cols = list(visits.columns)
    visits = normalize_visits_for_feed(
        visits, ensure_helper_id=False, ensure_hours_columns=False
    )
    membership_scheme_col = (
        "membership_funding_scheme"
        if "membership_funding_scheme" in visits.columns
        else None
    )
    visits = apply_claim_pricing_to_visits(
        visits,
        dva_claims_csv=dva_claims_csv,
        vhc_claims_csv=vhc_claims_csv,
        chsp_claims_csv=chsp_claims_csv,
        membership_scheme_col=membership_scheme_col,
    )

    if exclude_zero_revenue_visits:
        visits = visits.loc[visits["visit_projected_price"] != 0].copy()

    ordered = [c for c in original_cols if c in visits.columns]
    remaining = [c for c in visits.columns if c not in ordered]
    visits = visits[ordered + remaining]

    return visits
