#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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

    # Numeric fallback: non-zero => True
    num = pd.to_numeric(ss, errors="coerce")
    out = out.mask(num.notna(), num != 0)

    return out.fillna(False).astype(bool)


def _apply_dva_claim_pricing(
    visits: pd.DataFrame,
    dva_claims_csv: str,
    membership_scheme_col: str,
    amount_col: str = "visit_projected_price",
) -> pd.DataFrame:
    """
    If membership_funding_scheme == 'dva' (case-insensitive),
    override visits[visit_projected_price] from dva_claims_expanded ChargeAmount* by visit_id.
    If visit_id not found => set to 0 for those dva rows.

    Requires visits to have a visit_id column.
    """
    # Find/standardize visit_id column
    visit_id_col = _first_existing_col_case_insensitive(
        visits, ["visit_id", "Visit ID", "visitId", "visit id"]
    )
    if visit_id_col is None:
        print("[dva] Skipping DVA pricing: visits missing visit_id column.")
        return visits

    if visit_id_col != "visit_id":
        visits = visits.rename(columns={visit_id_col: "visit_id"})

    visits["visit_id"] = _clean_id_series(visits["visit_id"])

    # Load DVA claims expanded file
    dva = _read_csv(dva_claims_csv)

    dva_visit_col = _first_existing_col_case_insensitive(
        dva, ["VisitId", "visit_id", "visit id"]
    )
    dva_amount_col = _first_existing_col_case_insensitive(
        dva, ["ChargeAmount*", "ChargeAmount", "charge_amount"]
    )

    if dva_visit_col is None or dva_amount_col is None:
        print(
            f"[dva] Skipping DVA pricing: dva claims missing columns. "
            f"Need VisitId + ChargeAmount*. Found visit_col={dva_visit_col}, amount_col={dva_amount_col}"
        )
        return visits

    if dva_visit_col != "VisitId":
        dva = dva.rename(columns={dva_visit_col: "VisitId"})
    if dva_amount_col != "ChargeAmount*":
        dva = dva.rename(columns={dva_amount_col: "ChargeAmount*"})

    dva["VisitId"] = _clean_id_series(dva["VisitId"])
    dva["ChargeAmount*"] = pd.to_numeric(dva["ChargeAmount*"], errors="coerce").fillna(
        0.0
    )

    # In case of duplicates, sum per VisitId (safer)
    dva_map = (
        dva.loc[dva["VisitId"].notna()]
        .groupby("VisitId", as_index=True)["ChargeAmount*"]
        .sum()
    )

    scheme = visits[membership_scheme_col].astype("string").str.strip().str.lower()
    mask_dva = scheme.eq("dva") & visits["visit_id"].notna()

    if not mask_dva.any():
        print("[dva] No rows with membership_funding_scheme == 'dva' found in visits.")
        return visits

    mapped = visits.loc[mask_dva, "visit_id"].map(dva_map)
    matched = mapped.notna().sum()
    unmatched = int(mask_dva.sum() - matched)

    # Override
    visits.loc[mask_dva, amount_col] = pd.to_numeric(mapped, errors="coerce").fillna(
        0.0
    )

    print(
        f"[dva] Applied DVA pricing from {Path(dva_claims_csv).name}: "
        f"dva_rows={int(mask_dva.sum()):,} matched={int(matched):,} unmatched={unmatched:,} (unmatched set to 0)"
    )

    return visits


def build_enriched_visits_export(
    visits_csv: str,
    exclude_zero_revenue_visits: bool = False,
    dva_claims_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Read the raw visit export and return an enriched visit-level table.

    Enrichment performed:
      - Normalizes ID columns (visit_shift_id, helper_id, visit_id where present)
      - Cleans membership_* string columns (trims whitespace)
      - Ensures visit_projected_price is numeric
      - If dva_claims_csv is provided and membership_funding_scheme exists, applies
        DVA claim pricing overrides via _apply_dva_claim_pricing.

    Notes:
      - This function intentionally preserves the incoming schema (column set and
        order) except for any in-place normalization (e.g., visit_id column
        standardization) and the possible override of visit_projected_price.
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

    visits["visit_shift_id"] = _clean_id_series(visits["visit_shift_id"])

    if "helper_id" in visits.columns:
        visits["helper_id"] = _clean_id_series(visits["helper_id"])

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

    # Apply DVA pricing overrides at visit grain (if applicable)
    if dva_claims_csv and membership_scheme_col:
        visits = _apply_dva_claim_pricing(
            visits=visits,
            dva_claims_csv=dva_claims_csv,
            membership_scheme_col=membership_scheme_col,
            amount_col="visit_projected_price",
        )

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
    """Add HOURS-weighted cost allocation columns to an enriched visits export.

    IMPORTANT (Power BI schema stability): this function preserves the *same output columns*
    as the prior revenue-weighted implementation:
      - shift_total_cost
      - shift_total_revenue (still computed for reference/debugging)
      - visit_cost_allocated
      - allocation_method
      - allocation_ok
      - allocation_reason

    Allocation logic (within each shift), using actual_visit_hours:
      visit_cost_allocated = shift_total_cost * (actual_visit_hours / shift_total_hours)

    Rows with missing/blank visit_shift_id, shifts not found in shift_profitability_feed,
    or shifts with zero shift_total_hours are allocated 0 and flagged.

    This function does NOT modify the schema of shift_profitability_feed.
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
    if "shift_id" not in shift_profitability_feed.columns:
        raise ValueError("shift_profitability_feed must contain shift_id")
    if "total_cost" not in shift_profitability_feed.columns:
        raise ValueError("shift_profitability_feed must contain total_cost")

    # Ensure clean join keys
    visits["visit_shift_id"] = _clean_id_series(visits["visit_shift_id"])
    visits["actual_visit_hours"] = pd.to_numeric(
        visits["actual_visit_hours"], errors="coerce"
    ).fillna(0.0)
    shift_lookup = shift_profitability_feed[["shift_id", "total_cost"]].copy()
    shift_lookup["shift_id"] = _clean_id_series(shift_lookup["shift_id"])

    # Precompute shift_total_revenue from the (already DVA-priced) visits export
    # Only consider nonblank shift ids.
    valid_mask = visits["visit_shift_id"].notna()
    shift_rev = (
        visits.loc[valid_mask]
        .groupby("visit_shift_id", as_index=False)["visit_projected_price"]
        .sum()
        .rename(columns={"visit_projected_price": "shift_total_revenue"})
    )

    # Precompute shift_total_hours from actual_visit_hours.
    shift_hours = (
        visits.loc[valid_mask]
        .groupby("visit_shift_id", as_index=False)["actual_visit_hours"]
        .sum()
        .rename(columns={"actual_visit_hours": "_shift_total_hours"})
    )

    # Merge shift_total_cost and shift_total_revenue onto the visit rows
    visits = visits.merge(
        shift_lookup.rename(
            columns={"shift_id": "visit_shift_id", "total_cost": "shift_total_cost"}
        ),
        on="visit_shift_id",
        how="left",
    )
    visits = visits.merge(shift_rev, on="visit_shift_id", how="left")
    visits = visits.merge(shift_hours, on="visit_shift_id", how="left")

    # Defaults
    visits["shift_total_cost"] = pd.to_numeric(
        visits["shift_total_cost"], errors="coerce"
    )
    visits["shift_total_revenue"] = pd.to_numeric(
        visits["shift_total_revenue"], errors="coerce"
    )
    visits["_shift_total_hours"] = pd.to_numeric(
        visits["_shift_total_hours"], errors="coerce"
    )
    visits["shift_total_cost"] = visits["shift_total_cost"].fillna(0.0)
    visits["shift_total_revenue"] = visits["shift_total_revenue"].fillna(0.0)
    visits["_shift_total_hours"] = visits["_shift_total_hours"].fillna(0.0)

    # Allocation flags
    has_shift_id = visits["visit_shift_id"].notna()

    # NOTE: A shift can legitimately have total_cost == 0. In that case, allocation is 0 and ok.
    # We treat shift match as: shift_id exists in shift_lookup (not cost != 0).
    shift_ids_set = set(shift_lookup["shift_id"].dropna().astype("string").tolist())
    exists_in_shift_feed = has_shift_id & visits["visit_shift_id"].astype(
        "string"
    ).isin(shift_ids_set)

    # Allocation computation
    visits["visit_cost_allocated"] = 0.0
    ok_mask = exists_in_shift_feed & (visits["_shift_total_hours"] > 0)
    visits.loc[ok_mask, "visit_cost_allocated"] = visits.loc[
        ok_mask, "shift_total_cost"
    ] * (
        visits.loc[ok_mask, "actual_visit_hours"]
        / visits.loc[ok_mask, "_shift_total_hours"]
    )

    visits["allocation_method"] = "hours_weighted"
    visits["allocation_ok"] = ok_mask

    # Reason codes for debugging in Power BI
    reason = pd.Series("ok", index=visits.index, dtype="string")
    reason = reason.mask(~has_shift_id, "missing_shift_id")
    reason = reason.mask(has_shift_id & ~exists_in_shift_feed, "no_shift_match")
    reason = reason.mask(
        exists_in_shift_feed & (visits["_shift_total_hours"] <= 0), "zero_shift_hours"
    )
    visits["allocation_reason"] = reason

    # Round monetary fields (keep visit_projected_price as-is rounding managed elsewhere)
    for c in ["shift_total_cost", "shift_total_revenue", "visit_cost_allocated"]:
        _safe_round(visits, c, 2)

    # IMPORTANT: preserve output schema vs prior versions (do not add new exported columns).
    if "_shift_total_hours" in visits.columns:
        visits = visits.drop(columns=["_shift_total_hours"])

    return visits


def build_shift_profitability_feed(
    visits_csv: str,
    costs_csv: str,
    exclude_zero_revenue_visits: bool = False,
    billable_only: bool = False,
    dva_claims_csv: Optional[str] = None,
) -> pd.DataFrame:
    visits = _read_csv(visits_csv)
    costs = _read_csv(costs_csv)

    if "visit_shift_id" not in visits.columns and "shift_id" in visits.columns:
        visits = visits.rename(columns={"shift_id": "visit_shift_id"})

    if "visit_shift_id" not in visits.columns:
        raise ValueError("Visits CSV must contain visit_shift_id (or shift_id).")
    if "shift_id" not in costs.columns:
        raise ValueError("Costs CSV must contain shift_id.")

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

    # ✅ Preprocess visit_export prices for DVA before aggregating
    if dva_claims_csv and membership_scheme_col:
        visits = _apply_dva_claim_pricing(
            visits=visits,
            dva_claims_csv=dva_claims_csv,
            membership_scheme_col=membership_scheme_col,
            amount_col="visit_projected_price",
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
        # "shift_cost_line_description",
        # "description",
        # "Shift Cost Line Description",
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

        # Vehicle cost lines (case-insensitive match on Rule Name / description)
        costs["is_vehicle_cost_line"] = desc.str.contains(
            r"vehicle", case=False, na=False
        )
        # Additional flags derived from Rule Name / description (case-insensitive)
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

    # Reconcile printout
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


def _write_csv(df: pd.DataFrame, path: Path, utf8_bom: bool) -> None:
    enc = "utf-8-sig" if utf8_bom else "utf-8"
    df.to_csv(path, index=False, encoding=enc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build shift_profitability_feed.csv (and optionally an enriched visit export)."
    )
    parser.add_argument("--visits", required=True, help="Visits CSV path.")
    parser.add_argument("--costs", required=True, help="Shift costs CSV path.")
    parser.add_argument(
        "--dva-claims",
        default=None,
        help="Optional path to dva_claims_expanded.csv. If provided, DVA visits will be priced from this file by visit_id.",
    )
    parser.add_argument("--out-dir", default=".", help="Output directory.")
    parser.add_argument(
        "--out", default="shift_profitability_feed.csv", help="Output filename."
    )
    parser.add_argument(
        "--out-visits",
        default=None,
        help="Optional output filename for an enriched visit export CSV (visit-level). If provided, the script will write this file alongside shift_profitability_feed.",
    )
    parser.add_argument("--exclude-zero-revenue-visits", action="store_true")
    parser.add_argument("--billable-only", action="store_true")
    parser.add_argument(
        "--utf8-bom", action="store_true", help="Write UTF-8 with BOM (Excel-friendly)."
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_shift_profitability_feed(
        visits_csv=args.visits,
        costs_csv=args.costs,
        exclude_zero_revenue_visits=args.exclude_zero_revenue_visits,
        billable_only=args.billable_only,
        dva_claims_csv=args.dva_claims,
    )

    out_path = out_dir / args.out
    _write_csv(df, out_path, utf8_bom=args.utf8_bom)

    # Optional: write enriched visit export (visit-level) with DVA pricing applied
    if args.out_visits:
        visits_enriched = build_enriched_visits_export(
            visits_csv=args.visits,
            exclude_zero_revenue_visits=args.exclude_zero_revenue_visits,
            dva_claims_csv=args.dva_claims,
        )
        # Add revenue-weighted allocated cost columns at visit grain for simpler Power BI modelling
        visits_enriched = apply_revenue_weighted_cost_allocation_to_visits(
            visits_enriched=visits_enriched,
            shift_profitability_feed=df,
        )
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
