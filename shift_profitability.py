#!/usr/bin/env python3
"""
Shift profitability exports for Power BI (CSV outputs)

Outputs 3 CSVs:
1) shift_profitability_feed.csv        (1 row per shift)
2) shift_cost_lines_feed.csv           (many rows per shift)
3) shift_membership_bridge.csv         (many rows per shift; 1 row per shift + membership combo)

Flags:
- --exclude-zero-revenue-visits:
    Exclude individual visits where visit_projected_price == 0 before building revenue + bridge.
- --billable-only:
    Keep only shifts with aggregated revenue > 0 AND drop cost lines + bridge rows for non-billable shifts.

Notes:
- Uses read_csv(low_memory=False) to avoid mixed dtype warnings.
- Keeps membership as a proper bridge table (no "primary" guessing).
- Writes UTF-8 CSVs by default; you can opt into Excel-friendly UTF-8 with BOM via --utf8-bom.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------- helpers -----------------


def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def _require_col(df: pd.DataFrame, col: str, df_name: str) -> None:
    if col not in df.columns:
        raise KeyError(
            f"Expected column '{col}' not found in {df_name}. "
            f"Available columns: {list(df.columns)}"
        )


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _safe_round(df: pd.DataFrame, col: str, decimals: int) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(decimals)


def write_csv(df: pd.DataFrame, path: Path, *, utf8_bom: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    encoding = "utf-8-sig" if utf8_bom else "utf-8"
    df.to_csv(path, index=False, encoding=encoding)


def _clean_str_series(s: pd.Series) -> pd.Series:
    # Keep blanks as NA so groupby distinct counts behave consistently
    s = s.astype("string")
    s = s.str.strip()
    s = s.replace("", pd.NA)
    return s


# ----------------- core logic -----------------


def build_three_tables(
    visits_csv: str | Path,
    costs_csv: str | Path,
    *,
    exclude_zero_revenue_visits: bool = False,
    billable_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Load
    visits = _strip_cols(pd.read_csv(visits_csv, low_memory=False))
    costs = _strip_cols(pd.read_csv(costs_csv, low_memory=False))

    # Required
    _require_col(visits, "visit_shift_id", "visits CSV")
    _require_col(visits, "visit_projected_price", "visits CSV")
    _require_col(visits, "helper_id", "visits CSV")

    _require_col(costs, "shift_id", "shift costs CSV")
    _require_col(costs, "Units", "shift costs CSV")
    _require_col(costs, "shift_cost_line_amount", "shift costs CSV")

    # Coerce numerics used in calcs/joins
    visits = _to_numeric(
        visits,
        [
            "visit_shift_id",
            "visit_projected_price",
            "projected_visit_hours",
            "actual_visit_hours",
        ],
    )
    costs = _to_numeric(costs, ["shift_id", "Units", "shift_cost_line_amount"])

    # Optional: exclude zero revenue visits
    if exclude_zero_revenue_visits:
        visits["visit_projected_price"] = pd.to_numeric(
            visits["visit_projected_price"], errors="coerce"
        ).fillna(0)
        visits = visits.loc[visits["visit_projected_price"] != 0].copy()

    # -------------------------
    # Visits aggregation (shift revenue + helper + hours)
    # -------------------------
    visits_agg_dict = {
        "helper_id": ("helper_id", "first"),
        "visit_count": ("visit_id", "nunique")
        if "visit_id" in visits.columns
        else ("visit_shift_id", "size"),
        "revenue": ("visit_projected_price", "sum"),
    }
    if "projected_visit_hours" in visits.columns:
        visits_agg_dict["projected_hours"] = ("projected_visit_hours", "sum")
    else:
        visits_agg_dict["projected_hours"] = ("visit_projected_price", lambda s: np.nan)

    if "actual_visit_hours" in visits.columns:
        visits_agg_dict["actual_hours"] = ("actual_visit_hours", "sum")
    else:
        visits_agg_dict["actual_hours"] = ("visit_projected_price", lambda s: np.nan)

    visits_agg = (
        visits.groupby("visit_shift_id", as_index=False)
        .agg(**visits_agg_dict)
        .rename(columns={"visit_shift_id": "shift_id"})
    )
    visits_agg["revenue"] = pd.to_numeric(
        visits_agg["revenue"], errors="coerce"
    ).fillna(0)

    # Billable shift set
    if billable_only:
        billable_shift_ids = set(
            visits_agg.loc[visits_agg["revenue"] > 0, "shift_id"].dropna().tolist()
        )
    else:
        billable_shift_ids = None

    # -------------------------
    # Membership bridge table
    # -------------------------
    membership_fields = [
        "membership_community_name",
        "membership_funding_scheme",
        "membership_custom_funding_scheme_name",
    ]
    membership_fields = [c for c in membership_fields if c in visits.columns]

    if membership_fields:
        bridge = visits.copy().rename(columns={"visit_shift_id": "shift_id"})
        for c in membership_fields:
            bridge[c] = _clean_str_series(bridge[c])

        if billable_only:
            bridge = bridge.loc[bridge["shift_id"].isin(billable_shift_ids)].copy()

        group_cols = ["shift_id", "helper_id"] + membership_fields

        agg_map = {
            "visits_in_combo": ("visit_id", "nunique")
            if "visit_id" in bridge.columns
            else ("shift_id", "size"),
            "revenue_in_combo": ("visit_projected_price", "sum"),
        }
        if "projected_visit_hours" in bridge.columns:
            agg_map["projected_hours_in_combo"] = ("projected_visit_hours", "sum")
        else:
            agg_map["projected_hours_in_combo"] = (
                "visit_projected_price",
                lambda s: np.nan,
            )

        if "actual_visit_hours" in bridge.columns:
            agg_map["actual_hours_in_combo"] = ("actual_visit_hours", "sum")
        else:
            agg_map["actual_hours_in_combo"] = (
                "visit_projected_price",
                lambda s: np.nan,
            )

        membership_bridge = bridge.groupby(group_cols, as_index=False).agg(**agg_map)

        for c in [
            "revenue_in_combo",
            "projected_hours_in_combo",
            "actual_hours_in_combo",
        ]:
            if c in membership_bridge.columns:
                membership_bridge[c] = pd.to_numeric(
                    membership_bridge[c], errors="coerce"
                ).fillna(0)

        _safe_round(membership_bridge, "revenue_in_combo", 2)
        _safe_round(membership_bridge, "projected_hours_in_combo", 6)
        _safe_round(membership_bridge, "actual_hours_in_combo", 6)
    else:
        membership_bridge = pd.DataFrame(
            columns=["shift_id", "helper_id", "visits_in_combo", "revenue_in_combo"]
        )

    # -------------------------
    # Membership diagnostics per shift (no primary)
    # -------------------------
    if membership_fields and not membership_bridge.empty:
        diag = membership_bridge.groupby("shift_id", as_index=False).agg(
            membership_combo_count=("shift_id", "size"),
            membership_community_name_distinct_count=(
                "membership_community_name",
                "nunique",
            )
            if "membership_community_name" in membership_bridge.columns
            else ("shift_id", lambda s: 0),
            membership_funding_scheme_distinct_count=(
                "membership_funding_scheme",
                "nunique",
            )
            if "membership_funding_scheme" in membership_bridge.columns
            else ("shift_id", lambda s: 0),
            membership_custom_funding_scheme_name_distinct_count=(
                "membership_custom_funding_scheme_name",
                "nunique",
            )
            if "membership_custom_funding_scheme_name" in membership_bridge.columns
            else ("shift_id", lambda s: 0),
        )

        for col in [
            "membership_combo_count",
            "membership_community_name_distinct_count",
            "membership_funding_scheme_distinct_count",
            "membership_custom_funding_scheme_name_distinct_count",
        ]:
            if col in diag.columns:
                diag[col] = (
                    pd.to_numeric(diag[col], errors="coerce").fillna(0).astype(int)
                )

        diag["has_multiple_membership_combos"] = diag["membership_combo_count"] > 1
        diag["has_multiple_membership_community_name"] = (
            diag["membership_community_name_distinct_count"] > 1
        )
        diag["has_multiple_membership_funding_scheme"] = (
            diag["membership_funding_scheme_distinct_count"] > 1
        )
        diag["has_multiple_membership_custom_funding_scheme_name"] = (
            diag["membership_custom_funding_scheme_name_distinct_count"] > 1
        )
    else:
        diag = pd.DataFrame(
            columns=[
                "shift_id",
                "membership_combo_count",
                "membership_community_name_distinct_count",
                "membership_funding_scheme_distinct_count",
                "membership_custom_funding_scheme_name_distinct_count",
                "has_multiple_membership_combos",
                "has_multiple_membership_community_name",
                "has_multiple_membership_funding_scheme",
                "has_multiple_membership_custom_funding_scheme_name",
            ]
        )

    # -------------------------
    # Cost lines (detail)
    # -------------------------
    cost_lines = costs.copy()
    if billable_only:
        cost_lines = cost_lines.loc[
            cost_lines["shift_id"].isin(billable_shift_ids)
        ].copy()

    cost_lines["row_cost"] = cost_lines["Units"].fillna(0) * cost_lines[
        "shift_cost_line_amount"
    ].fillna(0)

    # Attach helper_id from visits aggregation
    cost_lines = cost_lines.merge(
        visits_agg[["shift_id", "helper_id"]],
        on="shift_id",
        how="left",
        suffixes=("", "_from_visits"),
    )
    if "helper_id_from_visits" in cost_lines.columns:
        cost_lines["helper_id"] = cost_lines["helper_id_from_visits"].combine_first(
            cost_lines.get("helper_id")
        )
        cost_lines = cost_lines.drop(columns=["helper_id_from_visits"])

    _safe_round(cost_lines, "shift_cost_line_amount", 6)
    _safe_round(cost_lines, "Units", 6)
    _safe_round(cost_lines, "row_cost", 2)

    # Allowance flags based on description
    desc_col = "shift_cost_line_description"
    if desc_col in cost_lines.columns:
        desc = cost_lines[desc_col].astype(str)
        cost_lines["is_allowance"] = desc.str.contains(
            r"allowance", case=False, na=False
        )
        cost_lines["is_travel_allowance"] = desc.str.contains(
            r"travel", case=False, na=False
        ) & desc.str.contains(r"allowance", case=False, na=False)
        cost_lines["is_other_allowance"] = (
            cost_lines["is_allowance"] & ~cost_lines["is_travel_allowance"]
        )
    else:
        cost_lines["is_allowance"] = False
        cost_lines["is_travel_allowance"] = False
        cost_lines["is_other_allowance"] = False

    # -------------------------
    # Costs aggregation (shift total cost + metadata)
    # -------------------------
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
    meta_cols = [c for c in meta_cols if c in cost_lines.columns]

    agg_dict = {
        "total_cost": ("row_cost", "sum"),
        "cost_line_count": ("shift_cost_line_id", "nunique")
        if "shift_cost_line_id" in cost_lines.columns
        else ("row_cost", "size"),
    }
    for c in meta_cols:
        agg_dict[c] = (c, "first")

    costs_agg = cost_lines.groupby("shift_id", as_index=False).agg(**agg_dict)

    rollups = (
        cost_lines.assign(
            allowance_cost=np.where(
                cost_lines["is_allowance"], cost_lines["row_cost"], 0.0
            ),
            travel_allowance_cost=np.where(
                cost_lines["is_travel_allowance"], cost_lines["row_cost"], 0.0
            ),
            other_allowance_cost=np.where(
                cost_lines["is_other_allowance"], cost_lines["row_cost"], 0.0
            ),
        )
        .groupby("shift_id", as_index=False)
        .agg(
            allowance_cost_total=("allowance_cost", "sum"),
            travel_allowance_cost=("travel_allowance_cost", "sum"),
            other_allowance_cost=("other_allowance_cost", "sum"),
            has_allowance=("is_allowance", "any"),
            has_travel_allowance=("is_travel_allowance", "any"),
            has_other_allowance=("is_other_allowance", "any"),
        )
    )
    costs_agg = costs_agg.merge(rollups, on="shift_id", how="left")

    # -------------------------
    # Shift fact = costs_agg + visits_agg (+ membership diagnostics)
    # -------------------------
    shift_fact = costs_agg.merge(
        visits_agg, how="left" if billable_only else "outer", on="shift_id"
    )
    if billable_only:
        shift_fact = shift_fact.loc[
            shift_fact["shift_id"].isin(billable_shift_ids)
        ].copy()

    if not diag.empty:
        shift_fact = shift_fact.merge(diag, on="shift_id", how="left")

    # Ensure numerics
    for c in [
        "total_cost",
        "revenue",
        "visit_count",
        "projected_hours",
        "actual_hours",
        "cost_line_count",
        "allowance_cost_total",
        "travel_allowance_cost",
        "other_allowance_cost",
        "membership_combo_count",
        "membership_community_name_distinct_count",
        "membership_funding_scheme_distinct_count",
        "membership_custom_funding_scheme_name_distinct_count",
    ]:
        if c in shift_fact.columns:
            shift_fact[c] = pd.to_numeric(shift_fact[c], errors="coerce").fillna(0)

    # Ensure booleans
    bool_cols = [
        "has_allowance",
        "has_travel_allowance",
        "has_other_allowance",
        "has_multiple_membership_combos",
        "has_multiple_membership_community_name",
        "has_multiple_membership_funding_scheme",
        "has_multiple_membership_custom_funding_scheme_name",
    ]
    for c in bool_cols:
        if c in shift_fact.columns:
            shift_fact[c] = shift_fact[c].astype("boolean").fillna(False).astype(bool)
        else:
            shift_fact[c] = False

    # Base cost + profitability
    shift_fact["base_cost"] = (
        shift_fact["total_cost"] - shift_fact["allowance_cost_total"]
    )
    shift_fact["profit"] = shift_fact["revenue"] - shift_fact["total_cost"]
    shift_fact["margin"] = np.where(
        shift_fact["revenue"] > 0, shift_fact["profit"] / shift_fact["revenue"], np.nan
    )
    shift_fact["margin_pct"] = shift_fact["margin"] * 100.0

    # Rounding
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

    # Order columns
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
        "cost_line_count",
        "membership_combo_count",
        "has_multiple_membership_combos",
        "membership_community_name_distinct_count",
        "has_multiple_membership_community_name",
        "membership_funding_scheme_distinct_count",
        "has_multiple_membership_funding_scheme",
        "membership_custom_funding_scheme_name_distinct_count",
        "has_multiple_membership_custom_funding_scheme_name",
    ] + meta_cols

    existing = [c for c in preferred_order if c in shift_fact.columns]
    remaining = [c for c in shift_fact.columns if c not in existing]
    shift_fact = shift_fact[existing + remaining]

    return shift_fact, cost_lines, membership_bridge


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create shift profitability + cost lines + membership bridge CSV feeds."
    )
    parser.add_argument("--visits", required=True, help="Visits CSV path.")
    parser.add_argument("--costs", required=True, help="Shift costs CSV path.")
    parser.add_argument("--out-dir", default=".", help="Output directory.")
    parser.add_argument(
        "--shift-out",
        default="shift_profitability_feed.csv",
        help="Shift fact output filename.",
    )
    parser.add_argument(
        "--lines-out",
        default="shift_cost_lines_feed.csv",
        help="Cost lines output filename.",
    )
    parser.add_argument(
        "--bridge-out",
        default="shift_membership_bridge.csv",
        help="Membership bridge output filename.",
    )
    parser.add_argument("--exclude-zero-revenue-visits", action="store_true")
    parser.add_argument("--billable-only", action="store_true")
    parser.add_argument(
        "--utf8-bom",
        action="store_true",
        help="Write CSVs as UTF-8 with BOM (helps Excel on Windows).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    shift_fact, cost_lines, membership_bridge = build_three_tables(
        args.visits,
        args.costs,
        exclude_zero_revenue_visits=args.exclude_zero_revenue_visits,
        billable_only=args.billable_only,
    )

    write_csv(shift_fact, out_dir / args.shift_out, utf8_bom=args.utf8_bom)
    write_csv(cost_lines, out_dir / args.lines_out, utf8_bom=args.utf8_bom)
    write_csv(membership_bridge, out_dir / args.bridge_out, utf8_bom=args.utf8_bom)

    print(
        f"Wrote shift fact:        {(out_dir / args.shift_out).resolve()}  (rows={len(shift_fact):,})"
    )
    print(
        f"Wrote shift cost lines:  {(out_dir / args.lines_out).resolve()}  (rows={len(cost_lines):,})"
    )
    print(
        f"Wrote membership bridge: {(out_dir / args.bridge_out).resolve()}  (rows={len(membership_bridge):,})"
    )

    if "revenue" in shift_fact.columns and "total_cost" in shift_fact.columns:
        print(f"Total revenue: {shift_fact['revenue'].sum():,.2f}")
        print(f"Total cost:    {shift_fact['total_cost'].sum():,.2f}")
        print(f"Total profit:  {shift_fact['profit'].sum():,.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
