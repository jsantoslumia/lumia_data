#!/usr/bin/env python3
"""
Build separate outputs:
- shift_costs_allocation.csv (from new_model allocation pipeline)
- visit_revenue.csv (visit-level revenue with DVA/VHC/CHSP claim logic)
- memberships_sah_purchases.csv (optional SAH purchases)

py -m new_model_revenue \
  --visits ./input_files/visit_export_feb.csv \
  --costs ./input_files/shift_costs_feb.csv \
  --mapping ./wages_allocation_mapping.xlsx \
  --out-dir ./test_files/test \
  --out-allocation shift_costs_allocation.csv \
  --out-visit-revenue visit_revenue.csv \
  --sah-transactions ./input_files/sah_transactions_feb.csv \
  --out-sah-purchases memberships_sah_purchases.csv \
  --dva-claims ./dva_claims_expanded.csv \
  --vhc-claims ./vhc_claims.csv \
  --chsp-claims ./chsp_dex_report.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from new_model import (
    build_final_all_from_sources,
    load_source_files,
    prepare_class_mapping,
)

try:
    from shift_profitability_sah import (
        build_memberships_sah_purchases_from_tx,
        read_and_enrich_sah_transactions,
    )
except ImportError:
    read_and_enrich_sah_transactions = None
    build_memberships_sah_purchases_from_tx = None


VHC_SERVICE_TYPE_RATES: Dict[str, float] = {
    "da": 98.72,
    "pc": 115.05,
    "ri": 73.0,
}

CHSP_OUTLET_RATES: Dict[str, float] = {
    "Allied Health and Therapy Services": 136.52,
    "Domestic Assistance": 65.24,
    "Home or Community General Respite": 61.84,
    "Individual Social Support": 55.74,
    "Personal Care": 64.69,
}


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _write_csv(df: pd.DataFrame, path: Path, utf8_bom: bool) -> None:
    enc = "utf-8-sig" if utf8_bom else "utf-8"
    df.to_csv(path, index=False, encoding=enc)


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
    return s.replace("", pd.NA)


def _first_existing_col_case_insensitive(
    df: pd.DataFrame, candidates: List[str]
) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        hit = lower_map.get(cand.lower())
        if hit is not None:
            return hit
    return None


def _chsp_rate_from_row(outlet_id: str, service_type_id: str) -> float:
    outlet = (outlet_id or "").strip()
    service = (service_type_id or "").strip()
    if "Community Home Support" in outlet:
        return 65.24 if "Domestic Assistance" in service else 0.0
    for label, rate in CHSP_OUTLET_RATES.items():
        if label in outlet:
            return rate
    return 0.0


def _apply_dva_claim_pricing(
    visits: pd.DataFrame,
    dva_claims_csv: str,
    membership_scheme_col: str,
    amount_col: str = "visit_projected_price",
) -> pd.DataFrame:
    visit_id_col = _first_existing_col_case_insensitive(
        visits, ["visit_id", "Visit ID", "visitId", "visit id"]
    )
    if visit_id_col is None:
        print("[dva] Skipping DVA pricing: visits missing visit_id column.")
        return visits
    if visit_id_col != "visit_id":
        visits = visits.rename(columns={visit_id_col: "visit_id"})
    visits["visit_id"] = _clean_id_series(visits["visit_id"])

    dva = _read_csv(dva_claims_csv)
    dva_visit_col = _first_existing_col_case_insensitive(
        dva, ["VisitId", "visit_id", "visit id"]
    )
    dva_amount_col = _first_existing_col_case_insensitive(
        dva, ["ChargeAmount*", "ChargeAmount", "charge_amount"]
    )
    if dva_visit_col is None or dva_amount_col is None:
        print("[dva] Skipping DVA pricing: missing VisitId or ChargeAmount*.")
        return visits

    if dva_visit_col != "VisitId":
        dva = dva.rename(columns={dva_visit_col: "VisitId"})
    if dva_amount_col != "ChargeAmount*":
        dva = dva.rename(columns={dva_amount_col: "ChargeAmount*"})

    dva["VisitId"] = _clean_id_series(dva["VisitId"])
    dva["ChargeAmount*"] = pd.to_numeric(dva["ChargeAmount*"], errors="coerce").fillna(
        0.0
    )
    dva_map = (
        dva.loc[dva["VisitId"].notna()]
        .groupby("VisitId", as_index=True)["ChargeAmount*"]
        .sum()
    )

    scheme = visits[membership_scheme_col].astype("string").str.strip().str.lower()
    mask_dva = scheme.eq("dva") & visits["visit_id"].notna()
    if not mask_dva.any():
        return visits

    mapped = visits.loc[mask_dva, "visit_id"].map(dva_map)
    visits.loc[mask_dva, amount_col] = pd.to_numeric(mapped, errors="coerce").fillna(
        0.0
    )
    return visits


def _apply_vhc_claim_pricing(
    visits: pd.DataFrame,
    vhc_claims_csv: str,
    membership_scheme_col: str,
    amount_col: str = "visit_projected_price",
) -> pd.DataFrame:
    visit_id_col = _first_existing_col_case_insensitive(
        visits, ["visit_id", "Visit ID", "visitId", "visit id"]
    )
    if visit_id_col is None:
        print("[vhc] Skipping VHC pricing: visits missing visit_id column.")
        return visits

    if visit_id_col != "visit_id":
        visits = visits.rename(columns={visit_id_col: "visit_id"})
    visits["visit_id"] = _clean_id_series(visits["visit_id"])

    if "actual_visit_hours" not in visits.columns:
        print("[vhc] Skipping VHC pricing: visits missing actual_visit_hours.")
        return visits
    actual_hours = pd.to_numeric(visits["actual_visit_hours"], errors="coerce").fillna(
        0.0
    )

    vhc = _read_csv(vhc_claims_csv)
    vhc_visit_col = _first_existing_col_case_insensitive(
        vhc, ["Visit ID", "VisitId", "visit_id", "visit id"]
    )
    vhc_service_col = _first_existing_col_case_insensitive(
        vhc, ["Service Type", "ServiceType", "service_type"]
    )
    vhc_copay_col = _first_existing_col_case_insensitive(
        vhc, ["Co-payment amount", "Co-payment", "copayment", "co_payment_amount"]
    )
    if vhc_visit_col is None or vhc_service_col is None or vhc_copay_col is None:
        print("[vhc] Skipping VHC pricing: required columns missing.")
        return visits

    vhc = vhc.rename(
        columns={
            vhc_visit_col: "_vhc_visit_id",
            vhc_service_col: "_vhc_service_type",
            vhc_copay_col: "_vhc_copay",
        }
    )
    vhc["_vhc_visit_id"] = _clean_id_series(vhc["_vhc_visit_id"])
    vhc["_vhc_rate"] = (
        vhc["_vhc_service_type"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(VHC_SERVICE_TYPE_RATES)
        .fillna(0.0)
    )
    vhc["_vhc_copay"] = pd.to_numeric(vhc["_vhc_copay"], errors="coerce").fillna(0.0)
    vhc_agg = (
        vhc.loc[vhc["_vhc_visit_id"].notna()]
        .groupby("_vhc_visit_id", as_index=True)
        .agg({"_vhc_rate": "first", "_vhc_copay": "sum"})
    )

    scheme = visits[membership_scheme_col].astype("string").str.strip().str.lower()
    price = pd.to_numeric(visits[amount_col], errors="coerce").fillna(-1)
    mask_vhc = scheme.eq("vhc") & visits["visit_id"].notna() & (price == 0)
    if not mask_vhc.any():
        return visits

    vhc_visit_ids = visits.loc[mask_vhc, "visit_id"]
    rates = vhc_visit_ids.map(vhc_agg["_vhc_rate"])
    copays = vhc_visit_ids.map(vhc_agg["_vhc_copay"])
    hours = actual_hours.loc[mask_vhc]
    visits.loc[mask_vhc, amount_col] = (rates.fillna(0.0) * hours) + copays.fillna(0.0)
    return visits


def _apply_chsp_claim_pricing(
    visits: pd.DataFrame,
    chsp_claims_csv: str,
    membership_scheme_col: str,
    amount_col: str = "visit_projected_price",
) -> pd.DataFrame:
    visit_id_col = _first_existing_col_case_insensitive(
        visits, ["visit_id", "Visit ID", "visitId", "visit id"]
    )
    if visit_id_col is None:
        print("[chsp] Skipping CHSP pricing: visits missing visit_id column.")
        return visits
    if visit_id_col != "visit_id":
        visits = visits.rename(columns={visit_id_col: "visit_id"})
    visits["visit_id"] = _clean_id_series(visits["visit_id"])

    chsp = _read_csv(chsp_claims_csv)
    outlet_col = _first_existing_col_case_insensitive(
        chsp, ["Outlet ID", "Outlet Id", "outlet_id"]
    )
    service_type_col = _first_existing_col_case_insensitive(
        chsp,
        ["Service type ID", "Service Type ID", "Service type Id", "service_type_id"],
    )
    time_col = _first_existing_col_case_insensitive(
        chsp, ["Time minutes", "Time Minutes", "time_minutes"]
    )
    client_contrib_col = _first_existing_col_case_insensitive(
        chsp, ["Client contribution", "Client Contribution", "client_contribution"]
    )
    session_col = _first_existing_col_case_insensitive(
        chsp, ["Session ID", "Session Id", "session_id"]
    )
    if not all([outlet_col, time_col, client_contrib_col, session_col]):
        print("[chsp] Skipping CHSP pricing: required columns missing.")
        return visits

    outlet_vals = chsp[outlet_col].astype("string").fillna("")
    service_vals = (
        chsp[service_type_col].astype("string").fillna("")
        if service_type_col
        else pd.Series("", index=chsp.index)
    )
    chsp["_chsp_rate"] = [
        _chsp_rate_from_row(o, s) for o, s in zip(outlet_vals, service_vals)
    ]
    chsp["_chsp_time_hours"] = (
        pd.to_numeric(chsp[time_col], errors="coerce").fillna(0.0) / 60.0
    )
    chsp["_chsp_client_contrib"] = pd.to_numeric(
        chsp[client_contrib_col], errors="coerce"
    ).fillna(0.0)
    chsp["_chsp_revenue"] = (
        chsp["_chsp_rate"] * chsp["_chsp_time_hours"] + chsp["_chsp_client_contrib"]
    )

    raw_session = chsp[session_col].astype("string").str.strip()
    chsp["_chsp_visit_id"] = raw_session.str.replace(
        r"_visit_fee$", "", regex=True
    ).str.strip()
    chsp["_chsp_visit_id"] = _clean_id_series(chsp["_chsp_visit_id"])

    chsp_map = (
        chsp.loc[chsp["_chsp_visit_id"].notna()]
        .groupby("_chsp_visit_id", as_index=True)["_chsp_revenue"]
        .sum()
    )

    scheme = visits[membership_scheme_col].astype("string").str.strip().str.lower()
    mask_chsp = scheme.eq("chsp") & visits["visit_id"].notna()
    if not mask_chsp.any():
        return visits

    mapped = visits.loc[mask_chsp, "visit_id"].map(chsp_map)
    visits.loc[mask_chsp, amount_col] = pd.to_numeric(mapped, errors="coerce").fillna(
        0.0
    )
    return visits


def build_visit_revenue(
    visits_csv: str,
    exclude_zero_revenue_visits: bool = False,
    dva_claims_csv: Optional[str] = None,
    vhc_claims_csv: Optional[str] = None,
    chsp_claims_csv: Optional[str] = None,
    class_map_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    visits = _read_csv(visits_csv)

    if "visit_projected_price" not in visits.columns:
        raise ValueError("Visits CSV must contain visit_projected_price.")

    if "visit_shift_id" not in visits.columns and "shift_id" in visits.columns:
        visits = visits.rename(columns={"shift_id": "visit_shift_id"})
    if "visit_shift_id" not in visits.columns:
        raise ValueError("Visits CSV must contain visit_shift_id (or shift_id).")

    visits["visit_shift_id"] = _clean_id_series(visits["visit_shift_id"])
    if "helper_id" in visits.columns:
        visits["helper_id"] = _clean_id_series(visits["helper_id"])

    visits["visit_projected_price"] = pd.to_numeric(
        visits["visit_projected_price"], errors="coerce"
    ).fillna(0.0)

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

    if membership_scheme_col and dva_claims_csv:
        visits = _apply_dva_claim_pricing(visits, dva_claims_csv, membership_scheme_col)
    if membership_scheme_col and vhc_claims_csv:
        visits = _apply_vhc_claim_pricing(visits, vhc_claims_csv, membership_scheme_col)
    if membership_scheme_col and chsp_claims_csv:
        visits = _apply_chsp_claim_pricing(
            visits, chsp_claims_csv, membership_scheme_col
        )

    if class_map_df is not None and "visit_rate" in visits.columns:
        class_lookup = prepare_class_mapping(class_map_df)
        class_lookup["visit_rate"] = class_lookup["visit_rate"].astype(str).str.strip()
        visits["visit_rate"] = visits["visit_rate"].astype(str).str.strip()
        visits = visits.merge(class_lookup, on="visit_rate", how="left")

    if exclude_zero_revenue_visits:
        visits = visits.loc[visits["visit_projected_price"] != 0].copy()

    return visits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build new_model allocation + separate visit revenue and SAH purchases outputs."
    )
    parser.add_argument("--visits", required=True, help="Visits CSV path.")
    parser.add_argument("--costs", required=True, help="Shift costs CSV path.")
    parser.add_argument(
        "--mapping",
        required=True,
        help="Excel mapping file path (sheets: Class, EH Mapping).",
    )
    parser.add_argument("--out-dir", default=".", help="Output directory.")
    parser.add_argument(
        "--out-allocation",
        default="shift_costs_allocation.csv",
        help="Allocation output CSV filename.",
    )
    parser.add_argument(
        "--out-visit-revenue",
        default="visit_revenue.csv",
        help="Visit revenue CSV filename.",
    )
    parser.add_argument(
        "--sah-transactions",
        default=None,
        help="Optional SAH transactions CSV path.",
    )
    parser.add_argument(
        "--out-sah-purchases",
        default="memberships_sah_purchases.csv",
        help="SAH purchases CSV filename.",
    )
    parser.add_argument("--dva-claims", default=None, help="Optional DVA claims CSV.")
    parser.add_argument("--vhc-claims", default=None, help="Optional VHC claims CSV.")
    parser.add_argument("--chsp-claims", default=None, help="Optional CHSP claims CSV.")
    parser.add_argument("--exclude-zero-revenue-visits", action="store_true")
    parser.add_argument("--utf8-bom", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    visits_df, shift_costs_df, class_map_df, eh_map_df = load_source_files(
        visits_csv_path=args.visits,
        shift_costs_csv_path=args.costs,
        mapping_excel_path=args.mapping,
    )

    allocation_outputs = build_final_all_from_sources(
        visits_df=visits_df,
        shift_costs_df=shift_costs_df,
        class_map_df=class_map_df,
        eh_map_df=eh_map_df,
    )

    allocation_path = out_dir / args.out_allocation
    _write_csv(
        allocation_outputs["Final_All"],
        allocation_path,
        utf8_bom=args.utf8_bom,
    )
    print(
        f"Wrote: {allocation_path.resolve()}  (rows={len(allocation_outputs['Final_All']):,})"
    )

    visit_revenue = build_visit_revenue(
        visits_csv=args.visits,
        exclude_zero_revenue_visits=args.exclude_zero_revenue_visits,
        dva_claims_csv=args.dva_claims,
        vhc_claims_csv=args.vhc_claims,
        chsp_claims_csv=args.chsp_claims,
        class_map_df=class_map_df,
    )
    visit_revenue_path = out_dir / args.out_visit_revenue
    _write_csv(visit_revenue, visit_revenue_path, utf8_bom=args.utf8_bom)
    print(
        f"Wrote: {visit_revenue_path.resolve()}  (rows={len(visit_revenue):,}, total_revenue={visit_revenue['visit_projected_price'].sum():,.2f})"
    )

    if args.sah_transactions:
        if (
            read_and_enrich_sah_transactions is None
            or build_memberships_sah_purchases_from_tx is None
        ):
            raise SystemExit(
                "SAH transactions require shift_profitability_sah to be importable."
            )
        tx = read_and_enrich_sah_transactions(args.sah_transactions)
        purchases = build_memberships_sah_purchases_from_tx(tx)

        try:
            vmap = _read_csv(args.visits)
            if "membership_uuid" in vmap.columns and "membership_name" in vmap.columns:
                vmap["membership_uuid"] = _clean_id_series(vmap["membership_uuid"])
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

        purchases_path = out_dir / args.out_sah_purchases
        _write_csv(purchases, purchases_path, utf8_bom=args.utf8_bom)
        print(
            f"Wrote: {purchases_path.resolve()}  (rows={len(purchases):,}, unique memberships={purchases['membership_uuid'].nunique():,})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
