"""Shift-level profitability feed: load visits/costs, aggregate, merge, reconcile."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from shift_profitability_lib.utils import (
    clean_id_series,
    first_existing_col_case_insensitive,
    read_csv,
    safe_round,
    to_bool_series,
)
from shift_profitability_lib.visits import (
    apply_claim_pricing_to_visits,
    normalize_visits_for_feed,
)


def _load_visits_and_costs(
    visits_csv: str,
    costs_csv: str,
    class_mapping_excel: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Load visits and costs CSVs, standardize shift IDs, drop rows with missing shift_id."""
    from shift_profitability_lib.class_mapping import (
        merge_costs_gl_from_excel,
        merge_visit_class_from_excel,
    )

    visits = read_csv(visits_csv)
    costs = read_csv(costs_csv)
    if class_mapping_excel:
        visits = merge_visit_class_from_excel(visits, class_mapping_excel)
        costs = merge_costs_gl_from_excel(costs, class_mapping_excel)

    if "visit_shift_id" not in visits.columns and "shift_id" in visits.columns:
        visits = visits.rename(columns={"shift_id": "visit_shift_id"})
    if "visit_shift_id" not in visits.columns:
        raise ValueError("Visits CSV must contain visit_shift_id (or shift_id).")
    if "shift_id" not in costs.columns:
        raise ValueError("Costs CSV must contain shift_id.")

    visits["visit_shift_id"] = clean_id_series(visits["visit_shift_id"])
    costs["shift_id"] = clean_id_series(costs["shift_id"])

    if "helper_id" in visits.columns:
        visits["helper_id"] = clean_id_series(visits["helper_id"])
    else:
        visits["helper_id"] = pd.NA

    visits_rows_before = len(visits)
    costs_rows_before = len(costs)
    visits = visits.loc[visits["visit_shift_id"].notna()].copy()
    costs = costs.loc[costs["shift_id"].notna()].copy()
    dropped_visits_missing_shift = visits_rows_before - len(visits)
    dropped_costs_missing_shift = costs_rows_before - len(costs)

    return visits, costs, dropped_visits_missing_shift, dropped_costs_missing_shift


def _aggregate_visits_to_shifts(
    visits: pd.DataFrame,
    costs: pd.DataFrame,
    exclude_zero_revenue_visits: bool,
    billable_only: bool,
    dva_claims_csv: Optional[str],
    vhc_claims_csv: Optional[str],
    chsp_claims_csv: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize visits, apply claim pricing, aggregate by shift_id, optionally filter billable_only."""
    visits = normalize_visits_for_feed(
        visits, ensure_helper_id=False, ensure_hours_columns=True
    )
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
    visits = apply_claim_pricing_to_visits(
        visits,
        dva_claims_csv=dva_claims_csv,
        vhc_claims_csv=vhc_claims_csv,
        chsp_claims_csv=chsp_claims_csv,
        membership_scheme_col=membership_scheme_col,
    )

    if exclude_zero_revenue_visits:
        visits = visits.loc[visits["visit_projected_price"] != 0].copy()

    visit_count_agg = (
        ("visit_id", "nunique")
        if "visit_id" in visits.columns
        else ("visit_shift_id", "size")
    )
    visits_agg = (
        visits.groupby("visit_shift_id", as_index=False)
        .agg(
            revenue=("visit_projected_price", "sum"),
            visit_count=visit_count_agg,
            projected_hours=("projected_visit_hours", "sum"),
            actual_hours=("actual_visit_hours", "sum"),
            helper_id=("helper_id", "first"),
            membership_community_name_distinct_count=(
                membership_community_col,
                lambda s: pd.Series(s).nunique(dropna=True),
            )
            if membership_community_col
            else ("visit_shift_id", lambda s: 0),
            membership_funding_scheme_distinct_count=(
                membership_scheme_col,
                lambda s: pd.Series(s).nunique(dropna=True),
            )
            if membership_scheme_col
            else ("visit_shift_id", lambda s: 0),
        )
        .rename(columns={"visit_shift_id": "shift_id"})
    )
    visits_agg["has_multiple_membership_community_name"] = (
        visits_agg["membership_community_name_distinct_count"] > 1
    )
    visits_agg["has_multiple_membership_funding_scheme"] = (
        visits_agg["membership_funding_scheme_distinct_count"] > 1
    )

    if billable_only:
        billable_shift_ids = (
            visits_agg.loc[visits_agg["revenue"] > 0, "shift_id"]
            .dropna()
            .astype("string")
            .unique()
            .tolist()
        )
        visits_agg = visits_agg.loc[
            visits_agg["shift_id"].isin(billable_shift_ids)
        ].copy()
        costs = costs.loc[costs["shift_id"].isin(billable_shift_ids)].copy()

    allowed_shift_ids = (
        visits_agg["shift_id"].dropna().astype("string").unique().tolist()
    )
    costs = costs.loc[costs["shift_id"].isin(allowed_shift_ids)].copy()

    return visits_agg, costs


def _aggregate_costs(costs: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Compute row_cost, flags, aggregate by shift_id with rollups. Returns (costs_agg, raw_cost_total)."""
    amount_candidates = [
        "shift_cost_line_amount",
        "cost_amount",
        "amount",
        "rate",
    ]
    units_candidates = ["shift_cost_line_units", "units", "Units", "qty", "quantity"]

    amount_col = first_existing_col_case_insensitive(costs, amount_candidates)
    units_col = first_existing_col_case_insensitive(costs, units_candidates)

    if amount_col is None:
        raise ValueError(
            f"Costs CSV missing a cost amount column. Tried: {amount_candidates}"
        )
    if units_col is None:
        costs["shift_cost_line_units"] = 1.0
        units_col = "shift_cost_line_units"

    if amount_col != "shift_cost_line_amount":
        costs = costs.rename(columns={amount_col: "shift_cost_line_amount"})
    if units_col != "shift_cost_line_units":
        costs = costs.rename(columns={units_col: "shift_cost_line_units"})

    costs["shift_cost_line_amount"] = pd.to_numeric(
        costs["shift_cost_line_amount"], errors="coerce"
    ).fillna(0.0)
    costs["shift_cost_line_units"] = pd.to_numeric(
        costs["shift_cost_line_units"], errors="coerce"
    ).fillna(0.0)
    costs["row_cost"] = costs["shift_cost_line_amount"] * costs["shift_cost_line_units"]

    desc_candidates = ["Rule Name"]
    desc_col = first_existing_col_case_insensitive(costs, desc_candidates)
    if desc_col:
        desc = costs[desc_col].astype(str)
        costs["is_allowance"] = desc.str.contains(r"allowance", case=False, na=False)
        costs["is_travel_allowance"] = (
            desc.str.contains(r"travel", case=False, na=False) & costs["is_allowance"]
        )
        costs["is_other_allowance"] = (
            costs["is_allowance"] & ~costs["is_travel_allowance"]
        )
        costs["is_vehicle_cost_line"] = desc.str.contains(
            r"vehicle", case=False, na=False
        )
        costs["is_overtime_line"] = desc.str.contains(r"overtime", case=False, na=False)
        costs["is_weekend_line"] = desc.str.contains(
            r"saturday|sunday", case=False, na=False
        )
        costs["is_public_holiday_line"] = desc.str.contains(
            r"holiday", case=False, na=False
        )
        costs["is_afternoon_line"] = desc.str.contains(
            r"afternoon", case=False, na=False
        )
        costs["is_night_line"] = desc.str.contains(r"night", case=False, na=False)
        costs["is_broken_shift_line"] = desc.str.contains(
            r"brokenshift", case=False, na=False
        )
        costs["is_casual_loading_line"] = desc.str.contains(
            r"casual\s*loading", case=False, na=False
        )
        costs["is_minimum_shift_length_line"] = desc.str.contains(
            r"minimum\s*shift\s*length", case=False, na=False
        )
    else:
        costs["is_allowance"] = False
        costs["is_travel_allowance"] = False
        costs["is_other_allowance"] = False
        costs["is_vehicle_cost_line"] = False
        costs["is_overtime_line"] = False
        costs["is_weekend_line"] = False
        costs["is_public_holiday_line"] = False
        costs["is_afternoon_line"] = False
        costs["is_night_line"] = False
        costs["is_broken_shift_line"] = False
        costs["is_casual_loading_line"] = False
        costs["is_minimum_shift_length_line"] = False

    meta_cols = [
        "Employee ID",
        "Helper Name",
        "Helper Region",
        "Employee Type",
        "Date",
        "Shift start date and time",
        "Shift end date and time",
        "Award Name",
        "Payroll Category",
    ]
    meta_cols_present = [c for c in meta_cols if c in costs.columns]

    costs_agg_dict: Dict[str, Tuple[str, str]] = {"total_cost": ("row_cost", "sum")}
    for c in meta_cols_present:
        costs_agg_dict[c] = (c, "first")

    if "GL" in costs.columns:

        def _combine_gl(s: pd.Series) -> object:
            u = {str(x).strip() for x in s.dropna() if str(x).strip()}
            return ", ".join(sorted(u)) if u else pd.NA

        costs_agg_dict["GL"] = ("GL", _combine_gl)

    costs_agg = costs.groupby("shift_id", as_index=False).agg(**costs_agg_dict)

    rollups = (
        costs.assign(
            allowance_cost=np.where(costs["is_allowance"], costs["row_cost"], 0.0),
            travel_allowance_cost=np.where(
                costs["is_travel_allowance"], costs["row_cost"], 0.0
            ),
            other_allowance_cost=np.where(
                costs["is_other_allowance"], costs["row_cost"], 0.0
            ),
            vehicle_cost=np.where(
                costs["is_vehicle_cost_line"], costs["row_cost"], 0.0
            ),
        )
        .groupby("shift_id", as_index=False)
        .agg(
            allowance_cost_total=("allowance_cost", "sum"),
            travel_allowance_cost=("travel_allowance_cost", "sum"),
            other_allowance_cost=("other_allowance_cost", "sum"),
            vehicle_cost=("vehicle_cost", "sum"),
            has_allowance=("is_allowance", "max"),
            has_travel_allowance=("is_travel_allowance", "max"),
            has_other_allowance=("is_other_allowance", "max"),
            has_vehicle_cost_lines=("is_vehicle_cost_line", "max"),
            has_overtime=("is_overtime_line", "max"),
            is_weekend=("is_weekend_line", "max"),
            is_public_holiday=("is_public_holiday_line", "max"),
            is_afternoon=("is_afternoon_line", "max"),
            is_night=("is_night_line", "max"),
            is_broken_shift=("is_broken_shift_line", "max"),
            is_casual_loading=("is_casual_loading_line", "max"),
            is_minimum_shift_length=("is_minimum_shift_length_line", "max"),
        )
    )
    costs_agg = costs_agg.merge(rollups, on="shift_id", how="left")

    if "Date" not in costs_agg.columns:
        costs_agg["Date"] = pd.NA

    raw_cost_total = float(
        pd.to_numeric(costs["row_cost"], errors="coerce").fillna(0.0).sum()
    )
    return costs_agg, raw_cost_total


def _merge_and_finalize_shift_fact(
    costs_agg: pd.DataFrame,
    visits_agg: pd.DataFrame,
    billable_only: bool,
    dropped_visits_missing_shift: int,
    dropped_costs_missing_shift: int,
    raw_cost_total: float,
) -> pd.DataFrame:
    """Merge costs_agg and visits_agg, add derived columns, column order, reconcile print."""
    how = "left" if billable_only else "outer"
    shift_fact = costs_agg.merge(visits_agg, on="shift_id", how=how)

    for c in [
        "revenue",
        "total_cost",
        "allowance_cost_total",
        "travel_allowance_cost",
        "other_allowance_cost",
        "vehicle_cost",
    ]:
        if c not in shift_fact.columns:
            shift_fact[c] = 0.0
        shift_fact[c] = pd.to_numeric(shift_fact[c], errors="coerce").fillna(0.0)

    for c in [
        "visit_count",
        "membership_community_name_distinct_count",
        "membership_funding_scheme_distinct_count",
    ]:
        if c not in shift_fact.columns:
            shift_fact[c] = 0
        shift_fact[c] = (
            pd.to_numeric(shift_fact[c], errors="coerce").fillna(0).astype(int)
        )

    for c in ["projected_hours", "actual_hours"]:
        if c not in shift_fact.columns:
            shift_fact[c] = np.nan
        shift_fact[c] = pd.to_numeric(shift_fact[c], errors="coerce")

    bool_cols = [
        "has_allowance",
        "has_travel_allowance",
        "has_other_allowance",
        "has_vehicle_cost_lines",
        "has_overtime",
        "is_weekend",
        "is_public_holiday",
        "is_afternoon",
        "is_night",
        "is_broken_shift",
        "is_casual_loading",
        "is_minimum_shift_length",
        "has_multiple_membership_community_name",
        "has_multiple_membership_funding_scheme",
    ]
    for c in bool_cols:
        if c not in shift_fact.columns:
            shift_fact[c] = False
        shift_fact[c] = to_bool_series(shift_fact[c])

    shift_fact["base_cost_without_allowances_vehicle_cost"] = (
        shift_fact["total_cost"]
        - shift_fact["allowance_cost_total"]
        - shift_fact["vehicle_cost"]
    )
    shift_fact["profit"] = shift_fact["revenue"] - shift_fact["total_cost"]
    shift_fact["margin"] = np.where(
        shift_fact["revenue"] > 0, shift_fact["profit"] / shift_fact["revenue"], np.nan
    )
    shift_fact["margin_pct"] = shift_fact["margin"] * 100.0

    for c in [
        "revenue",
        "total_cost",
        "base_cost_without_allowances_vehicle_cost",
        "profit",
        "allowance_cost_total",
        "travel_allowance_cost",
        "other_allowance_cost",
        "vehicle_cost",
    ]:
        safe_round(shift_fact, c, 2)
    safe_round(shift_fact, "margin", 6)
    safe_round(shift_fact, "margin_pct", 2)
    safe_round(shift_fact, "projected_hours", 6)
    safe_round(shift_fact, "actual_hours", 6)

    preferred_order = [
        "shift_id",
        "helper_id",
        "revenue",
        "total_cost",
        "base_cost_without_allowances_vehicle_cost",
        "allowance_cost_total",
        "travel_allowance_cost",
        "other_allowance_cost",
        "vehicle_cost",
        "has_vehicle_cost_lines",
        "has_overtime",
        "is_weekend",
        "is_public_holiday",
        "is_afternoon",
        "is_night",
        "is_broken_shift",
        "is_casual_loading",
        "is_minimum_shift_length",
        "has_allowance",
        "has_travel_allowance",
        "has_other_allowance",
        "profit",
        "margin",
        "margin_pct",
        "visit_count",
        "projected_hours",
        "actual_hours",
        "membership_community_name_distinct_count",
        "has_multiple_membership_community_name",
        "membership_funding_scheme_distinct_count",
        "has_multiple_membership_funding_scheme",
        "Employee ID",
        "Helper Name",
        "Helper Region",
        "Employee Type",
        "Date",
        "Shift start date and time",
        "Shift end date and time",
        "Award Name",
        "Payroll Category",
        "GL",
    ]
    existing = [c for c in preferred_order if c in shift_fact.columns]
    remaining = [c for c in shift_fact.columns if c not in existing]
    shift_fact = shift_fact[existing + remaining]

    agg_cost_total = float(
        pd.to_numeric(shift_fact["total_cost"], errors="coerce").fillna(0.0).sum()
    )
    delta = raw_cost_total - agg_cost_total

    print(
        f"[reconcile] costs rows dropped (missing shift_id): {dropped_costs_missing_shift}"
    )
    print(
        f"[reconcile] visits rows dropped (missing visit_shift_id): {dropped_visits_missing_shift}"
    )
    print(f"[reconcile] total row_cost from costs file: {raw_cost_total:,.2f}")
    print(f"[reconcile] total_cost in output table:   {agg_cost_total:,.2f}")
    print(f"[reconcile] delta (raw - output):        {delta:,.2f}")

    return shift_fact


def build_shift_profitability_feed(
    visits_csv: str,
    costs_csv: str,
    exclude_zero_revenue_visits: bool = False,
    billable_only: bool = False,
    dva_claims_csv: Optional[str] = None,
    vhc_claims_csv: Optional[str] = None,
    chsp_claims_csv: Optional[str] = None,
    class_mapping_excel: Optional[str] = None,
) -> pd.DataFrame:
    """Build the shift-level profitability feed from visits and costs CSVs."""
    visits, costs, dropped_visits_missing_shift, dropped_costs_missing_shift = (
        _load_visits_and_costs(visits_csv, costs_csv, class_mapping_excel)
    )
    visits_agg, costs = _aggregate_visits_to_shifts(
        visits,
        costs,
        exclude_zero_revenue_visits,
        billable_only,
        dva_claims_csv,
        vhc_claims_csv,
        chsp_claims_csv,
    )
    costs_agg, raw_cost_total = _aggregate_costs(costs)
    return _merge_and_finalize_shift_fact(
        costs_agg,
        visits_agg,
        billable_only,
        dropped_visits_missing_shift,
        dropped_costs_missing_shift,
        raw_cost_total,
    )
