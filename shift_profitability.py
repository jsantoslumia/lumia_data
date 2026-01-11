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
        "Rate",
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
        "shift_cost_line_description",
        "description",
        "Shift Cost Line Description",
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
    else:
        costs["is_allowance"] = False
        costs["is_travel_allowance"] = False
        costs["is_other_allowance"] = False

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
        "Rate",
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
        )
        .groupby("shift_id", as_index=False)
        .agg(
            allowance_cost_total=("allowance_cost", "sum"),
            travel_allowance_cost=("travel_allowance_cost", "sum"),
            other_allowance_cost=("other_allowance_cost", "sum"),
            has_allowance=("is_allowance", "max"),
            has_travel_allowance=("is_travel_allowance", "max"),
            has_other_allowance=("is_other_allowance", "max"),
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
        "has_multiple_membership_community_name",
        "has_multiple_membership_funding_scheme",
    ]
    for c in bool_cols:
        if c not in shift_fact.columns:
            shift_fact[c] = False
        shift_fact[c] = _to_bool_series(shift_fact[c])

    shift_fact["base_cost"] = (
        shift_fact["total_cost"] - shift_fact["allowance_cost_total"]
    )
    shift_fact["profit"] = shift_fact["revenue"] - shift_fact["total_cost"]
    shift_fact["margin"] = np.where(
        shift_fact["revenue"] > 0, shift_fact["profit"] / shift_fact["revenue"], np.nan
    )
    shift_fact["margin_pct"] = shift_fact["margin"] * 100.0

    for c in [
        "revenue",
        "total_cost",
        "base_cost",
        "profit",
        "allowance_cost_total",
        "travel_allowance_cost",
        "other_allowance_cost",
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
        "base_cost",
        "allowance_cost_total",
        "travel_allowance_cost",
        "other_allowance_cost",
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
        "Rate",
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
        description="Build shift_profitability_feed.csv only."
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

    print(
        f"Wrote: {out_path.resolve()}  (rows={len(df):,}, unique shifts={df['shift_id'].nunique():,})"
    )
    print(f"Total revenue: {df['revenue'].sum():,.2f}")
    print(f"Total cost:    {df['total_cost'].sum():,.2f}")
    print(f"Total profit:  {df['profit'].sum():,.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
