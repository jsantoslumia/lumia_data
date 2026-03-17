#!/usr/bin/env python3
"""
py -m shift_profitability_new_allocation --visits ./input_files/visit_export_feb.csv --costs ./input_files/shift_costs_feb.csv --out-dir . --out-visits visits_export_enriched.csv --sah-transactions ./input_files/sah_transactions_feb.csv --out-sah-purchases memberships_sah_purchases.csv --dva-claims ./dva_claims_expanded.csv --vhc-claims ./vhc_claims.csv --chsp-claims ./chsp_dex_report.csv --mapping ./wages_allocation_mapping.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from shift_profitability_lib import (
    build_shift_profitability_feed,
    clean_id_series,
    read_csv,
    write_csv,
    write_enriched_visits_export,
)

try:
    from shift_profitability_sah import (
        build_memberships_sah_purchases_from_tx,
        build_memberships_sah_revenue_from_tx,
        read_and_enrich_sah_transactions,
    )
except ImportError:
    read_and_enrich_sah_transactions = None
    build_memberships_sah_purchases_from_tx = None
    build_memberships_sah_revenue_from_tx = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build shift_profitability_feed.csv (and optionally an enriched visit export)."
    )
    parser.add_argument("--visits", required=True, help="Visits CSV path.")
    parser.add_argument("--costs", required=True, help="Shift costs CSV path.")
    parser.add_argument(
        "--dva-claims",
        default=None,
        help="Optional path to dva_claims_expanded.csv.",
    )
    parser.add_argument(
        "--vhc-claims",
        default=None,
        help="Optional path to VHC claims CSV.",
    )
    parser.add_argument(
        "--chsp-claims",
        default=None,
        help="Optional path to CHSP DEX report CSV.",
    )
    parser.add_argument("--out-dir", default=".", help="Output directory.")
    parser.add_argument(
        "--out", default="shift_profitability_feed.csv", help="Output filename."
    )
    parser.add_argument(
        "--out-visits",
        default=None,
        help="Optional output filename for enriched visit export CSV.",
    )
    parser.add_argument(
        "--sah-transactions",
        default=None,
        help="Optional SAH transactions CSV path.",
    )
    parser.add_argument(
        "--out-sah-purchases",
        default="memberships_sah_purchases.csv",
        help="Output filename for SAH purchases.",
    )
    parser.add_argument("--exclude-zero-revenue-visits", action="store_true")
    parser.add_argument("--billable-only", action="store_true")
    parser.add_argument(
        "--utf8-bom", action="store_true", help="Write UTF-8 with BOM (Excel-friendly)."
    )
    parser.add_argument(
        "--mapping",
        default=None,
        help="Optional Excel: sheet 'Class' with visit_rate + Class (merged on load).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_mapping: Optional[str] = None
    if args.mapping:
        mp = Path(args.mapping)
        if not mp.is_file():
            raise SystemExit(f"--mapping file not found: {mp.resolve()}")
        class_mapping = str(mp.resolve())

    sah_revenue_by_membership: Optional[pd.DataFrame] = None

    if args.sah_transactions:
        if (
            read_and_enrich_sah_transactions is None
            or build_memberships_sah_purchases_from_tx is None
            or build_memberships_sah_revenue_from_tx is None
        ):
            raise SystemExit(
                "SAH transactions require shift_profitability_sah to be importable. "
                "Ensure shift_profitability_sah.py is on the path."
            )
        tx = read_and_enrich_sah_transactions(args.sah_transactions)
        sah_revenue_by_membership = build_memberships_sah_revenue_from_tx(tx)
        purchases = build_memberships_sah_purchases_from_tx(tx)

        try:
            vmap = read_csv(args.visits)
            if "membership_uuid" in vmap.columns and "membership_name" in vmap.columns:
                vmap["membership_uuid"] = clean_id_series(vmap["membership_uuid"])
                name_map = vmap.loc[
                    vmap["membership_uuid"].notna(),
                    ["membership_uuid", "membership_name"],
                ].copy()
                name_map["membership_name"] = (
                    name_map["membership_name"].astype("string").fillna("").str.strip()
                )
                name_map = name_map.loc[
                    name_map["membership_name"].ne("")
                ].drop_duplicates(subset=["membership_uuid"], keep="first")
                if not name_map.empty:
                    purchases = purchases.merge(
                        name_map, on="membership_uuid", how="left"
                    )
        except Exception:
            pass
        if "membership_name" not in purchases.columns:
            purchases["membership_name"] = pd.NA
        ordered_cols = ["membership_uuid", "membership_name", "total_cost", "purchases"]
        purchases = purchases[[c for c in ordered_cols if c in purchases.columns]]

        out_purchases_path = out_dir / args.out_sah_purchases
        write_csv(purchases, out_purchases_path, utf8_bom=args.utf8_bom)
        print(
            f"Wrote: {out_purchases_path.resolve()}  (rows={len(purchases):,}, unique memberships={purchases['membership_uuid'].nunique():,})"
        )

    df = build_shift_profitability_feed(
        visits_csv=args.visits,
        costs_csv=args.costs,
        exclude_zero_revenue_visits=args.exclude_zero_revenue_visits,
        billable_only=args.billable_only,
        dva_claims_csv=args.dva_claims,
        vhc_claims_csv=args.vhc_claims,
        chsp_claims_csv=args.chsp_claims,
        class_mapping_excel=class_mapping,
    )

    out_path = out_dir / args.out
    write_csv(df, out_path, utf8_bom=args.utf8_bom)

    if args.out_visits:
        write_enriched_visits_export(
            out_dir / args.out_visits,
            args.visits,
            df,
            args.exclude_zero_revenue_visits,
            args.dva_claims,
            args.vhc_claims,
            args.chsp_claims,
            sah_revenue_by_membership,
            args.utf8_bom,
            class_mapping_excel=class_mapping,
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
