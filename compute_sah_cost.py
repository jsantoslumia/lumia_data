#!/usr/bin/env python3
"""
Compute correct SAH cost using the same logic as shift_profitability_sah (SAH/HCP
visits only, SAH-only shift feed). Outputs a CSV for Power BI.

Run after or alongside shift_profitability. Uses the same visit and cost inputs.

Power BI:
  - Load sah_cost_export.csv as a separate table (e.g. sah_cost_export).
  - SAH Cost = SUM(sah_cost_export[sah_cost_allocated]).
  - To slice by membership: relate sah_cost_export to your memberships table
    on membership_uuid (or use membership_uuid in visuals and sum sah_cost_allocated).

To Run:

py -m compute_sah_cost --visits ./input_files/visit_export_jan.csv --costs ./input_files/shift_costs_jan.csv --out-dir . --out sah_cost_export.csv
py -m compute_sah_cost --visits ./input_files/visit_export_feb.csv --costs ./input_files/shift_costs_feb.csv --out-dir . --out sah_cost_export.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shift_profitability_sah import (
    apply_revenue_weighted_cost_allocation_to_visits,
    build_enriched_visits_export,
    build_shift_profitability_feed,
)


def _write_csv(df, path: Path, utf8_bom: bool = False) -> None:
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig" if utf8_bom else "utf-8",
        date_format="%Y-%m-%d",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute SAH cost (same logic as shift_profitability_sah). Output CSV for Power BI."
    )
    parser.add_argument("--visits", required=True, help="Path to visit export CSV")
    parser.add_argument("--costs", required=True, help="Path to shift costs CSV")
    parser.add_argument(
        "--out",
        default="sah_cost_export.csv",
        help="Output CSV path (default: sah_cost_export.csv)",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Output directory (default: current)",
    )
    parser.add_argument(
        "--exclude-zero-revenue-visits",
        action="store_true",
        help="Exclude visits with zero projected price",
    )
    parser.add_argument(
        "--billable-only",
        action="store_true",
        help="Restrict to shifts with revenue > 0",
    )
    parser.add_argument(
        "--utf8-bom",
        action="store_true",
        help="Write UTF-8 with BOM (Excel-friendly)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same pipeline as shift_profitability_sah: SAH/HCP visits + SAH-only shift feed
    df_sah = build_shift_profitability_feed(
        visits_csv=args.visits,
        costs_csv=args.costs,
        exclude_zero_revenue_visits=args.exclude_zero_revenue_visits,
        billable_only=args.billable_only,
    )
    visits_sah = build_enriched_visits_export(
        visits_csv=args.visits,
        exclude_zero_revenue_visits=args.exclude_zero_revenue_visits,
    )
    visits_sah = apply_revenue_weighted_cost_allocation_to_visits(
        visits_enriched=visits_sah,
        shift_profitability_feed=df_sah,
    )

    # Export columns useful for Power BI: link by membership_uuid and/or use SUM(sah_cost_allocated)
    id_cols = [
        c
        for c in [
            "membership_uuid",
            "visit_shift_id",
            "visit_id",
            "membership_funding_scheme",
        ]
        if c in visits_sah.columns
    ]
    out_df = visits_sah[id_cols + ["visit_cost_allocated"]].copy()
    out_df = out_df.rename(columns={"visit_cost_allocated": "sah_cost_allocated"})

    out_path = out_dir / args.out
    _write_csv(out_df, out_path, utf8_bom=args.utf8_bom)
    total = out_df["sah_cost_allocated"].sum()
    print(f"Wrote: {out_path.resolve()}  (rows={len(out_df):,})")
    print(f"Total SAH cost: {total:,.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
