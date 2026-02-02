#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ALLOWED_FUNDING_SCHEMES = {"sah", "hcp"}

SERVICE_TYPE_CARE_MGMT = "40103 - HCP Revenue - Care Management"
SERVICE_TYPE_OTHER = "40105 - HCP Revenue - Other"
SERVICE_TYPE_SERVICES = "40101 - HCP Revenue - Services"


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _clean_id_series(s: pd.Series) -> pd.Series:
    raw = s.astype("string").str.strip()
    raw = raw.replace(["", "nan", "None", "<NA>"], pd.NA)

    num = pd.to_numeric(raw, errors="coerce")
    is_int_like = num.notna() & (np.floor(num) == num)
    raw = raw.where(~is_int_like, num.astype("Int64").astype("string"))
    raw = raw.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
    return raw


def _clean_str_series(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.replace("", pd.NA)
    return s


def _safe_round(df: pd.DataFrame, col: str, ndigits: int) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(ndigits)


def _first_existing_col_case_insensitive(
    df: pd.DataFrame, candidates: List[str]
) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        hit = lower_map.get(cand.lower())
        if hit is not None:
            return hit
    return None


def _to_bool_series(s: pd.Series) -> pd.Series:
    """Convert a Series to reliable boolean without triggering pandas FutureWarning."""
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

    # Numeric fallback: non-zero => True
    num = pd.to_numeric(ss, errors="coerce")
    out = out.mask(num.notna(), num != 0)

    return out.fillna(False).astype(bool)


def _filter_visits_to_sah(visits: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where membership_funding_scheme is SAH or HCP (case-insensitive)."""
    if "membership_funding_scheme" not in visits.columns:
        raise ValueError(
            "Visits CSV must contain membership_funding_scheme for SAH/HCP filtering."
        )

    scheme = (
        visits["membership_funding_scheme"]
        .astype("string")
        .str.strip()
        .str.lower()
        .fillna("")
    )
    out = visits.loc[scheme.isin(ALLOWED_FUNDING_SCHEMES)].copy()
    return out


def build_enriched_visits_export(
    visits_csv: str,
    exclude_zero_revenue_visits: bool = False,
    allowed_membership_uuids: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Read the raw visit export and return an enriched visit-level table (SAH/HCP only).

    Additional filtering (when allowed_membership_uuids is provided):
      - Keep only rows whose membership_uuid appears in the SAH transactions input.

    Enrichment performed:
      - Normalizes ID columns (visit_shift_id, helper_id, visit_id where present)
      - Cleans membership_* string columns (trims whitespace)
      - Ensures visit_projected_price is numeric

    Notes:
      - This function intentionally preserves the incoming schema (column set and
        order) except for any in-place normalization.
      - This does NOT touch shift_profitability_feed output schema.
    """
    visits = _read_csv(visits_csv)

    # Standardize shift key name
    if "visit_shift_id" not in visits.columns and "shift_id" in visits.columns:
        visits = visits.rename(columns={"shift_id": "visit_shift_id"})

    if "visit_shift_id" not in visits.columns:
        raise ValueError("Visits CSV must contain visit_shift_id (or shift_id).")

    # Preserve original column order
    original_cols = list(visits.columns)

    # Basic cleanup / typing
    visits["visit_shift_id"] = _clean_id_series(visits["visit_shift_id"])

    if "helper_id" in visits.columns:
        visits["helper_id"] = _clean_id_series(visits["helper_id"])

    if "membership_uuid" in visits.columns:
        visits["membership_uuid"] = _clean_id_series(visits["membership_uuid"])
    elif allowed_membership_uuids is not None:
        raise ValueError(
            "Visits CSV must contain membership_uuid when filtering by SAH transactions."
        )

    if "visit_projected_price" not in visits.columns:
        raise ValueError("Visits CSV must contain visit_projected_price.")
    visits["visit_projected_price"] = pd.to_numeric(
        visits["visit_projected_price"], errors="coerce"
    ).fillna(0.0)

    # Clean optional visit hour fields if present
    if "projected_visit_hours" in visits.columns:
        visits["projected_visit_hours"] = pd.to_numeric(
            visits["projected_visit_hours"], errors="coerce"
        )
    if "actual_visit_hours" in visits.columns:
        visits["actual_visit_hours"] = pd.to_numeric(
            visits["actual_visit_hours"], errors="coerce"
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

    if membership_community_col:
        visits[membership_community_col] = _clean_str_series(
            visits[membership_community_col]
        )
    if membership_scheme_col:
        visits[membership_scheme_col] = _clean_str_series(visits[membership_scheme_col])

    # ✅ SAH/HCP-only filter
    visits = _filter_visits_to_sah(visits)

    # ✅ Optional: limit to membership_uuid present in the SAH transactions file
    if allowed_membership_uuids is not None:
        allowed = pd.Series(list(allowed_membership_uuids), dtype="string")
        visits = visits.loc[
            visits["membership_uuid"].notna()
            & visits["membership_uuid"].astype("string").isin(set(allowed.tolist()))
        ].copy()

    if exclude_zero_revenue_visits:
        visits = visits.loc[visits["visit_projected_price"] != 0].copy()

    # Re-apply original ordering where possible
    ordered = [c for c in original_cols if c in visits.columns]
    remaining = [c for c in visits.columns if c not in ordered]
    visits = visits[ordered + remaining]

    return visits


def apply_revenue_weighted_cost_allocation_to_visits(
    visits_enriched: pd.DataFrame,
    shift_profitability_feed: pd.DataFrame,
) -> pd.DataFrame:
    """Add HELPER-hours-weighted cost allocation columns to an enriched visits export.

    Output columns preserved:
      - shift_total_cost
      - shift_total_revenue (computed for reference/debugging)
      - visit_cost_allocated
      - allocation_method
      - allocation_ok
      - allocation_reason

    Allocation logic (per helper), using actual_visit_hours:

      helper_total_cost  = SUM(total_cost) across shifts for that helper
                          (restricted to shift_ids present in visits_enriched)
      helper_total_hours = SUM(actual_visit_hours) across visits for that helper

      visit_cost_allocated = helper_total_cost * (actual_visit_hours / helper_total_hours)

    This satisfies the accountant pivot formula at (helper, membership_uuid) grain:
      SUM(visit_cost_allocated) over helper+membership
        = helper_total_cost * (A / B)
      where A = SUM(actual_visit_hours) for (helper, membership_uuid),
            B = SUM(actual_visit_hours) for (helper).

    Rows with missing helper_id, helper_total_hours == 0, or shifts not found in the
    shift_profitability_feed are allocated 0 and flagged.
    """
    visits = visits_enriched.copy()

    if "visit_shift_id" not in visits.columns:
        raise ValueError("visits_enriched must contain visit_shift_id")
    if "visit_projected_price" not in visits.columns:
        raise ValueError("visits_enriched must contain visit_projected_price")
    if "actual_visit_hours" not in visits.columns:
        raise ValueError(
            "visits_enriched must contain actual_visit_hours for hours-weighted allocation"
        )
    if "helper_id" not in visits.columns:
        raise ValueError("visits_enriched must contain helper_id for helper allocation")
    if "shift_id" not in shift_profitability_feed.columns:
        raise ValueError("shift_profitability_feed must contain shift_id")
    if "total_cost" not in shift_profitability_feed.columns:
        raise ValueError("shift_profitability_feed must contain total_cost")

    # Clean join keys / types
    visits["visit_shift_id"] = _clean_id_series(visits["visit_shift_id"])
    visits["helper_id"] = _clean_id_series(visits["helper_id"])
    visits["actual_visit_hours"] = pd.to_numeric(
        visits["actual_visit_hours"], errors="coerce"
    ).fillna(0.0)

    # Shift-level lookup (restricted to shifts present in the visit export)
    shift_ids_in_visits = (
        visits["visit_shift_id"].dropna().astype("string").unique().tolist()
    )

    shift_lookup_cols = ["shift_id", "total_cost"]
    if "helper_id" in shift_profitability_feed.columns:
        shift_lookup_cols.append("helper_id")

    shift_lookup = shift_profitability_feed[shift_lookup_cols].copy()
    shift_lookup["shift_id"] = _clean_id_series(shift_lookup["shift_id"])
    if "helper_id" in shift_lookup.columns:
        shift_lookup["helper_id"] = _clean_id_series(shift_lookup["helper_id"])

    shift_lookup = shift_lookup.loc[
        shift_lookup["shift_id"].isin(set(shift_ids_in_visits))
    ].copy()

    # Precompute shift_total_revenue from the visits export (debugging only)
    valid_shift = visits["visit_shift_id"].notna()
    shift_rev = (
        visits.loc[valid_shift]
        .groupby("visit_shift_id", as_index=False)["visit_projected_price"]
        .sum()
        .rename(columns={"visit_projected_price": "shift_total_revenue"})
    )

    # Merge shift_total_cost and shift_total_revenue onto visit rows
    visits = visits.merge(
        shift_lookup[["shift_id", "total_cost"]].rename(
            columns={"shift_id": "visit_shift_id", "total_cost": "shift_total_cost"}
        ),
        on="visit_shift_id",
        how="left",
    )
    visits = visits.merge(shift_rev, on="visit_shift_id", how="left")

    visits["shift_total_cost"] = pd.to_numeric(
        visits["shift_total_cost"], errors="coerce"
    ).fillna(0.0)
    visits["shift_total_revenue"] = pd.to_numeric(
        visits["shift_total_revenue"], errors="coerce"
    ).fillna(0.0)

    has_shift_id = visits["visit_shift_id"].notna()
    has_helper_id = visits["helper_id"].notna()

    # Shift match check (exists in shift_lookup, not based on cost != 0)
    shift_ids_set = set(shift_lookup["shift_id"].dropna().astype("string").tolist())
    exists_in_shift_feed = has_shift_id & visits["visit_shift_id"].astype(
        "string"
    ).isin(shift_ids_set)

    # Helper total hours (B)
    helper_hours = (
        visits.loc[has_helper_id]
        .groupby("helper_id", as_index=False)["actual_visit_hours"]
        .sum()
        .rename(columns={"actual_visit_hours": "_helper_total_hours"})
    )

    # Helper total cost: SUM(total_cost) across shifts for that helper (restricted to shifts_in_visits)
    if "helper_id" in shift_lookup.columns:
        helper_cost = (
            shift_lookup.loc[shift_lookup["helper_id"].notna()]
            .drop_duplicates(subset=["shift_id"])  # just-in-case
            .groupby("helper_id", as_index=False)["total_cost"]
            .sum()
            .rename(columns={"total_cost": "_helper_total_cost"})
        )
    else:
        # Fallback: derive helper_total_cost by deduping (helper_id, shift) from visits after merge
        shift_cost_dedup = visits.loc[
            has_helper_id & has_shift_id,
            ["helper_id", "visit_shift_id", "shift_total_cost"],
        ].drop_duplicates(subset=["helper_id", "visit_shift_id"])
        helper_cost = (
            shift_cost_dedup.groupby("helper_id", as_index=False)["shift_total_cost"]
            .sum()
            .rename(columns={"shift_total_cost": "_helper_total_cost"})
        )

    visits = visits.merge(helper_hours, on="helper_id", how="left")
    visits = visits.merge(helper_cost, on="helper_id", how="left")

    visits["_helper_total_hours"] = pd.to_numeric(
        visits["_helper_total_hours"], errors="coerce"
    ).fillna(0.0)
    visits["_helper_total_cost"] = pd.to_numeric(
        visits["_helper_total_cost"], errors="coerce"
    ).fillna(0.0)

    visits["visit_cost_allocated"] = 0.0
    ok_mask = has_helper_id & exists_in_shift_feed & (visits["_helper_total_hours"] > 0)
    visits.loc[ok_mask, "visit_cost_allocated"] = visits.loc[
        ok_mask, "_helper_total_cost"
    ] * (
        visits.loc[ok_mask, "actual_visit_hours"]
        / visits.loc[ok_mask, "_helper_total_hours"]
    )

    # Apply oncosts directly to visit_cost_allocated (accountant factor)
    visits.loc[ok_mask, "visit_cost_allocated"] = (
        visits.loc[ok_mask, "visit_cost_allocated"] * 1.2075
    )

    visits["allocation_method"] = "helper_hours_weighted"
    visits["allocation_ok"] = ok_mask

    reason = pd.Series("ok", index=visits.index, dtype="string")
    reason = reason.mask(~has_shift_id, "missing_shift_id")
    reason = reason.mask(~has_helper_id, "missing_helper_id")
    reason = reason.mask(has_shift_id & ~exists_in_shift_feed, "no_shift_match")
    reason = reason.mask(
        has_helper_id & (visits["_helper_total_hours"] <= 0), "zero_helper_hours"
    )
    visits["allocation_reason"] = reason

    for c in ["shift_total_cost", "shift_total_revenue", "visit_cost_allocated"]:
        _safe_round(visits, c, 2)

    # Drop internal helper totals (do not change exported schema)
    visits = visits.drop(
        columns=[
            c
            for c in ["_helper_total_hours", "_helper_total_cost"]
            if c in visits.columns
        ]
    )

    return visits


def build_shift_profitability_feed(
    visits_csv: str,
    costs_csv: str,
    exclude_zero_revenue_visits: bool = False,
    billable_only: bool = False,
    allowed_membership_uuids: Optional[set[str]] = None,
) -> pd.DataFrame:
    visits = _read_csv(visits_csv)
    costs = _read_csv(costs_csv)

    if "visit_shift_id" not in visits.columns and "shift_id" in visits.columns:
        visits = visits.rename(columns={"shift_id": "visit_shift_id"})

    if "visit_shift_id" not in visits.columns:
        raise ValueError("Visits CSV must contain visit_shift_id (or shift_id).")
    if "shift_id" not in costs.columns:
        raise ValueError("Costs CSV must contain shift_id.")

    # ✅ SAH/HCP-only filter early
    visits = _filter_visits_to_sah(visits)

    # Optional: only keep visits whose membership_uuid exists in the SAH transactions input
    if allowed_membership_uuids is not None:
        if "membership_uuid" not in visits.columns:
            raise ValueError(
                "Visits CSV must contain membership_uuid when filtering by SAH transactions."
            )
        visits["membership_uuid"] = _clean_id_series(visits["membership_uuid"])
        allowed = pd.Series(list(allowed_membership_uuids), dtype="string")
        visits = visits.loc[
            visits["membership_uuid"].notna()
            & visits["membership_uuid"].astype("string").isin(set(allowed.tolist()))
        ].copy()

    visits["visit_shift_id"] = _clean_id_series(visits["visit_shift_id"])
    costs["shift_id"] = _clean_id_series(costs["shift_id"])

    if "helper_id" in visits.columns:
        visits["helper_id"] = _clean_id_series(visits["helper_id"])
    else:
        visits["helper_id"] = pd.NA

    costs_rows_before = len(costs)
    visits_rows_before = len(visits)

    visits = visits.loc[visits["visit_shift_id"].notna()].copy()
    costs = costs.loc[costs["shift_id"].notna()].copy()

    dropped_visits_missing_shift = visits_rows_before - len(visits)
    dropped_costs_missing_shift = costs_rows_before - len(costs)

    if "visit_projected_price" not in visits.columns:
        raise ValueError("Visits CSV must contain visit_projected_price.")
    visits["visit_projected_price"] = pd.to_numeric(
        visits["visit_projected_price"], errors="coerce"
    ).fillna(0.0)

    if "projected_visit_hours" in visits.columns:
        visits["projected_visit_hours"] = pd.to_numeric(
            visits["projected_visit_hours"], errors="coerce"
        )
    else:
        visits["projected_visit_hours"] = np.nan

    if "actual_visit_hours" in visits.columns:
        visits["actual_visit_hours"] = pd.to_numeric(
            visits["actual_visit_hours"], errors="coerce"
        )
    else:
        visits["actual_visit_hours"] = np.nan

    membership_community_col = (
        "membership_community_name"
        if "membership_community_name" in visits.columns
        else None
    )
    membership_scheme_col = "membership_funding_scheme"

    if membership_community_col:
        visits[membership_community_col] = _clean_str_series(
            visits[membership_community_col]
        )
    if membership_scheme_col in visits.columns:
        visits[membership_scheme_col] = _clean_str_series(visits[membership_scheme_col])

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
            if membership_scheme_col in visits.columns
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

    billable_shift_ids: Optional[List[str]] = None
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

    # ✅ IMPORTANT: costs must be restricted to the same shift population as filtered visits
    allowed_shift_ids = (
        visits_agg["shift_id"].dropna().astype("string").unique().tolist()
    )
    costs = costs.loc[costs["shift_id"].isin(allowed_shift_ids)].copy()

    amount_candidates = [
        "shift_cost_line_amount",
        "cost_amount",
        "amount",
        "rate",
    ]
    units_candidates = ["shift_cost_line_units", "units", "Units", "qty", "quantity"]

    amount_col = _first_existing_col_case_insensitive(costs, amount_candidates)
    units_col = _first_existing_col_case_insensitive(costs, units_candidates)

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

    desc_candidates = [
        "Rule Name",
    ]
    desc_col = _first_existing_col_case_insensitive(costs, desc_candidates)
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
        shift_fact[c] = _to_bool_series(shift_fact[c])

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
        _safe_round(shift_fact, c, 2)
    _safe_round(shift_fact, "margin", 6)
    _safe_round(shift_fact, "margin_pct", 2)
    _safe_round(shift_fact, "projected_hours", 6)
    _safe_round(shift_fact, "actual_hours", 6)

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
    ]
    existing = [c for c in preferred_order if c in shift_fact.columns]
    remaining = [c for c in shift_fact.columns if c not in existing]
    shift_fact = shift_fact[existing + remaining]

    raw_cost_total = float(
        pd.to_numeric(costs["row_cost"], errors="coerce").fillna(0.0).sum()
    )
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


def read_and_enrich_sah_transactions(sah_transactions_csv: str) -> pd.DataFrame:
    """Read SAH transactions and derive fields used by purchases + revenue.

    Required columns:
      - membership_uuid
      - line_sah_service_type
      - product_name
      - line_net_amount

    Optional columns (used for revenue sign handling):
      - invoice_category  (invoice|credit_note). Missing => treated as invoice.
    """
    tx = _read_csv(sah_transactions_csv)

    required = [
        "membership_uuid",
        "line_sah_service_type",
        "product_name",
        "line_net_amount",
    ]
    missing = [c for c in required if c not in tx.columns]
    if missing:
        raise ValueError(f"SAH transactions CSV missing required columns: {missing}")

    tx["membership_uuid"] = _clean_id_series(tx["membership_uuid"])
    tx["line_sah_service_type"] = _clean_str_series(tx["line_sah_service_type"])

    # product_name can be numeric or string; treat 0/blank as 'Other' branch
    prod_s = tx["product_name"].astype("string").str.strip().fillna("")
    prod_is_blank = prod_s.eq("")
    prod_is_zeroish_str = prod_s.str.lower().isin({"0", "0.0"})
    prod_num = pd.to_numeric(prod_s, errors="coerce")
    prod_is_zero_num = prod_num.eq(0).fillna(False)
    prod_is_zero_or_blank = prod_is_blank | prod_is_zeroish_str | prod_is_zero_num

    line_type = tx["line_sah_service_type"].astype("string").str.strip().fillna("")
    is_care_mgmt = line_type.str.casefold().eq("care management")

    # Set defaults then override with masks (priority: care mgmt wins)
    tx["service_type"] = SERVICE_TYPE_SERVICES
    tx.loc[prod_is_zero_or_blank, "service_type"] = SERVICE_TYPE_OTHER
    tx.loc[is_care_mgmt, "service_type"] = SERVICE_TYPE_CARE_MGMT

    tx["line_net_amount"] = pd.to_numeric(
        tx["line_net_amount"], errors="coerce"
    ).fillna(0.0)

    # Normalize invoice_category
    if "invoice_category" in tx.columns:
        tx["invoice_category"] = (
            tx["invoice_category"]
            .astype("string")
            .str.strip()
            .str.lower()
            .fillna("invoice")
        )
    else:
        tx["invoice_category"] = "invoice"

    return tx


def build_memberships_sah_purchases_from_tx(tx: pd.DataFrame) -> pd.DataFrame:
    """Aggregate SAH purchases (service_type == OTHER) by membership_uuid."""
    other = tx.loc[
        tx["service_type"].eq(SERVICE_TYPE_OTHER) & tx["membership_uuid"].notna()
    ].copy()

    agg = other.groupby("membership_uuid", as_index=False).agg(
        total_cost=("line_net_amount", "sum")
    )
    agg["purchases"] = agg["total_cost"] / 1.05

    _safe_round(agg, "total_cost", 2)
    _safe_round(agg, "purchases", 2)

    return agg[["membership_uuid", "total_cost", "purchases"]]


def build_memberships_sah_revenue_from_tx(tx: pd.DataFrame) -> pd.DataFrame:
    """Aggregate SAH revenue (service_type == care mgmt OR services) by membership_uuid.

    Sign handling:
      - invoice_category == 'invoice'     => +line_net_amount
      - invoice_category == 'credit_note' => -line_net_amount
    """
    cat = (
        tx["invoice_category"]
        .astype("string")
        .str.strip()
        .str.lower()
        .fillna("invoice")
    )
    amt = pd.to_numeric(tx["line_net_amount"], errors="coerce").fillna(0.0)

    signed_amt = amt.abs()
    signed_amt = signed_amt.where(~cat.eq("credit_note"), -signed_amt)

    rev_lines = tx["service_type"].isin({SERVICE_TYPE_CARE_MGMT, SERVICE_TYPE_SERVICES})
    rev = tx.loc[rev_lines & tx["membership_uuid"].notna(), ["membership_uuid"]].copy()
    rev["signed_amount"] = signed_amt.loc[rev.index].to_numpy()

    out = rev.groupby("membership_uuid", as_index=False).agg(
        sah_revenue=("signed_amount", "sum")
    )

    _safe_round(out, "sah_revenue", 2)
    return out


def build_memberships_sah_purchases(sah_transactions_csv: str) -> pd.DataFrame:
    """Backward-compatible wrapper: read tx then build purchases."""
    tx = read_and_enrich_sah_transactions(sah_transactions_csv)
    return build_memberships_sah_purchases_from_tx(tx)


def _write_csv(df: pd.DataFrame, path: Path, utf8_bom: bool) -> None:
    enc = "utf-8-sig" if utf8_bom else "utf-8"
    df.to_csv(path, index=False, encoding=enc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build shift_profitability_feed.csv (and optionally an enriched visit export). SAH/HCP-only; optional membership_uuid restriction via SAH transactions."
    )
    parser.add_argument("--visits", required=True, help="Visits CSV path.")
    parser.add_argument("--costs", required=True, help="Shift costs CSV path.")
    parser.add_argument("--out-dir", default=".", help="Output directory.")
    parser.add_argument(
        "--out", default="shift_profitability_feed.csv", help="Output filename."
    )
    parser.add_argument(
        "--out-visits",
        default=None,
        help="Optional output filename for an enriched visit export CSV (visit-level). If provided, the script will write this file alongside shift_profitability_feed.",
    )
    parser.add_argument(
        "--sah-transactions",
        default=None,
        help="Optional SAH transactions CSV path. If provided, the script will (a) filter visits to membership_uuid present in this file, (b) add sah_revenue to the visit export, and (c) generate memberships_sah_purchases.csv.",
    )
    parser.add_argument(
        "--out-sah-purchases",
        default="memberships_sah_purchases.csv",
        help="Output filename for SAH purchases (used when --sah-transactions is provided).",
    )
    parser.add_argument("--exclude-zero-revenue-visits", action="store_true")
    parser.add_argument("--billable-only", action="store_true")
    parser.add_argument(
        "--utf8-bom", action="store_true", help="Write UTF-8 with BOM (Excel-friendly)."
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_memberships: Optional[set[str]] = None
    sah_revenue_by_membership: Optional[pd.DataFrame] = None
    purchases: Optional[pd.DataFrame] = None

    if args.sah_transactions:
        tx = read_and_enrich_sah_transactions(args.sah_transactions)

        allowed_memberships = set(
            tx["membership_uuid"].dropna().astype("string").unique().tolist()
        )
        sah_revenue_by_membership = build_memberships_sah_revenue_from_tx(tx)
        purchases = build_memberships_sah_purchases_from_tx(tx)

        # Enrich purchases with membership_name (from visits) when available
        try:
            vmap = _read_csv(args.visits)
            vmap = _filter_visits_to_sah(vmap)
            if allowed_memberships is not None and "membership_uuid" in vmap.columns:
                vmap = vmap.loc[
                    vmap["membership_uuid"].astype("string").isin(allowed_memberships)
                ].copy()

            if "membership_uuid" in vmap.columns and "membership_name" in vmap.columns:
                name_map = vmap[["membership_uuid", "membership_name"]].copy()
                name_map["membership_uuid"] = name_map["membership_uuid"].astype(
                    "string"
                )
                name_map["membership_name"] = (
                    name_map["membership_name"].astype("string").fillna("").str.strip()
                )
                name_map = name_map.loc[name_map["membership_uuid"].notna()].copy()
                name_map = name_map.loc[name_map["membership_name"].ne("")].copy()

                if not name_map.empty:
                    name_map = name_map.groupby("membership_uuid", as_index=False).agg(
                        membership_name=("membership_name", "first")
                    )
                    purchases = purchases.merge(
                        name_map, on="membership_uuid", how="left"
                    )
        except Exception:
            # If we can't read/derive membership_name, continue without it.
            pass

        if "membership_name" not in purchases.columns:
            purchases["membership_name"] = pd.NA

        # Reorder columns
        ordered_cols = ["membership_uuid", "membership_name", "total_cost", "purchases"]
        purchases = purchases[[c for c in ordered_cols if c in purchases.columns]]

        out_purchases_path = out_dir / args.out_sah_purchases
        _write_csv(purchases, out_purchases_path, utf8_bom=args.utf8_bom)
        print(
            f"Wrote: {out_purchases_path.resolve()}  (rows={len(purchases):,}, unique memberships={purchases['membership_uuid'].nunique():,})"
        )

    df = build_shift_profitability_feed(
        visits_csv=args.visits,
        costs_csv=args.costs,
        exclude_zero_revenue_visits=args.exclude_zero_revenue_visits,
        billable_only=args.billable_only,
        allowed_membership_uuids=allowed_memberships,
    )

    out_path = out_dir / args.out
    _write_csv(df, out_path, utf8_bom=args.utf8_bom)

    # Optional: write enriched visit export (visit-level)
    if args.out_visits:
        visits_enriched = build_enriched_visits_export(
            visits_csv=args.visits,
            exclude_zero_revenue_visits=args.exclude_zero_revenue_visits,
            allowed_membership_uuids=allowed_memberships,
        )

        # ✅ Filter visit export to Central Coast only (per request)
        if "membership_community_name" not in visits_enriched.columns:
            raise ValueError(
                "Visits export must contain membership_community_name to filter to NSW - Central Coast."
            )
        visits_enriched = visits_enriched.loc[
            visits_enriched["membership_community_name"]
            .astype("string")
            .str.strip()
            .eq("NSW - Central Coast")
        ].copy()

        # Add allocated cost columns at visit grain for simpler Power BI modelling
        visits_enriched = apply_revenue_weighted_cost_allocation_to_visits(
            visits_enriched=visits_enriched,
            shift_profitability_feed=df,
        )

        # Optional: add sah_revenue per membership_uuid (same value repeated per membership)
        if sah_revenue_by_membership is not None:
            if "membership_uuid" not in visits_enriched.columns:
                raise ValueError(
                    "Visits export must contain membership_uuid to add sah_revenue."
                )
            visits_enriched["membership_uuid"] = _clean_id_series(
                visits_enriched["membership_uuid"]
            )
            visits_enriched = visits_enriched.merge(
                sah_revenue_by_membership, on="membership_uuid", how="left"
            )
            visits_enriched["sah_revenue"] = pd.to_numeric(
                visits_enriched["sah_revenue"], errors="coerce"
            ).fillna(0.0)
            _safe_round(visits_enriched, "sah_revenue", 2)

            # Allocate membership-level SAH revenue down to visits by actual_visit_hours
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
            visits_enriched["sah_visit_revenue"] = (
                visits_enriched["sah_revenue"] * share
            )
            _safe_round(visits_enriched, "sah_visit_revenue", 2)

        out_visits_path = out_dir / args.out_visits
        _write_csv(visits_enriched, out_visits_path, utf8_bom=args.utf8_bom)
        print(
            f"Wrote: {out_visits_path.resolve()}  (rows={len(visits_enriched):,}, unique visit_shift_id={visits_enriched['visit_shift_id'].nunique():,})"
        )

    print(
        f"Wrote: {out_path.resolve()}  (rows={len(df):,}, unique shifts={df['shift_id'].nunique():,})"
    )
    print(f"Total revenue: {df['revenue'].sum():,.2f}")
    print(f"Total cost:    {df['total_cost'].sum():,.2f}")
    print(f"Total profit:  {df['profit'].sum():,.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
