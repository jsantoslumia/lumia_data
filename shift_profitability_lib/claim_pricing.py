"""DVA, VHC, and CHSP claim pricing: apply claim amounts to visit_projected_price by scheme."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from shift_profitability_lib.utils import (
    clean_id_series,
    first_existing_col_case_insensitive,
    read_csv,
)

# VHC Service Type rates (per hour); blank/unknown => 0
VHC_SERVICE_TYPE_RATES: Dict[str, float] = {
    "da": 98.72,
    "pc": 115.05,
    "ri": 73.0,
}

# CHSP: rate by Outlet ID (and Service type ID when Outlet contains "Community Home Support")
CHSP_OUTLET_RATES: Dict[str, float] = {
    "Allied Health and Therapy Services": 136.52,
    "Domestic Assistance": 65.24,
    "Home or Community General Respite": 61.84,
    "Individual Social Support": 55.74,
    "Personal Care": 64.69,
}


def _chsp_rate_from_row(outlet_id: str, service_type_id: str) -> float:
    """Determine CHSP rate from Outlet ID and optionally Service type ID."""
    outlet = (outlet_id or "").strip()
    service = (service_type_id or "").strip()
    if "Community Home Support" in outlet:
        return 65.24 if "Domestic Assistance" in service else 0.0
    for label, rate in CHSP_OUTLET_RATES.items():
        if label in outlet:
            return rate
    return 0.0


def apply_dva_claim_pricing(
    visits: pd.DataFrame,
    dva_claims_csv: str,
    membership_scheme_col: str,
    amount_col: str = "visit_projected_price",
) -> pd.DataFrame:
    """
    If membership_funding_scheme == 'dva' (case-insensitive),
    override visits[visit_projected_price] from dva_claims_expanded ChargeAmount* by visit_id.
    """
    visit_id_col = first_existing_col_case_insensitive(
        visits, ["visit_id", "Visit ID", "visitId", "visit id"]
    )
    if visit_id_col is None:
        print("[dva] Skipping DVA pricing: visits missing visit_id column.")
        return visits

    if visit_id_col != "visit_id":
        visits = visits.rename(columns={visit_id_col: "visit_id"})

    visits["visit_id"] = clean_id_series(visits["visit_id"])

    dva = read_csv(dva_claims_csv)

    dva_visit_col = first_existing_col_case_insensitive(
        dva, ["VisitId", "visit_id", "visit id"]
    )
    dva_amount_col = first_existing_col_case_insensitive(
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

    dva["VisitId"] = clean_id_series(dva["VisitId"])
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
        print("[dva] No rows with membership_funding_scheme == 'dva' found in visits.")
        return visits

    mapped = visits.loc[mask_dva, "visit_id"].map(dva_map)
    matched = mapped.notna().sum()
    unmatched = int(mask_dva.sum() - matched)

    visits.loc[mask_dva, amount_col] = pd.to_numeric(mapped, errors="coerce").fillna(
        0.0
    )

    print(
        f"[dva] Applied DVA pricing from {Path(dva_claims_csv).name}: "
        f"dva_rows={int(mask_dva.sum()):,} matched={int(matched):,} unmatched={unmatched:,} (unmatched set to 0)"
    )

    return visits


def apply_vhc_claim_pricing(
    visits: pd.DataFrame,
    vhc_claims_csv: str,
    membership_scheme_col: str,
    amount_col: str = "visit_projected_price",
) -> pd.DataFrame:
    """
    For rows with membership_funding_scheme == 'vhc' and visit_projected_price == 0,
    set visit claim from vhc_claims CSV: (rate * actual_visit_hours) + Co-payment amount.
    """
    visit_id_col = first_existing_col_case_insensitive(
        visits, ["visit_id", "Visit ID", "visitId", "visit id"]
    )
    if visit_id_col is None:
        print("[vhc] Skipping VHC pricing: visits missing visit_id column.")
        return visits

    if visit_id_col != "visit_id":
        visits = visits.rename(columns={visit_id_col: "visit_id"})
    visits["visit_id"] = clean_id_series(visits["visit_id"])

    if "actual_visit_hours" not in visits.columns:
        print("[vhc] Skipping VHC pricing: visits missing actual_visit_hours column.")
        return visits
    actual_hours = pd.to_numeric(visits["actual_visit_hours"], errors="coerce").fillna(
        0.0
    )

    vhc = read_csv(vhc_claims_csv)
    vhc_visit_col = first_existing_col_case_insensitive(
        vhc, ["Visit ID", "VisitId", "visit_id", "visit id"]
    )
    vhc_service_col = first_existing_col_case_insensitive(
        vhc, ["Service Type", "ServiceType", "service_type"]
    )
    vhc_copay_col = first_existing_col_case_insensitive(
        vhc, ["Co-payment amount", "Co-payment", "copayment", "co_payment_amount"]
    )

    if vhc_visit_col is None or vhc_service_col is None or vhc_copay_col is None:
        print(
            f"[vhc] Skipping VHC pricing: vhc claims missing columns. "
            f"Need Visit ID, Service Type, Co-payment amount. "
            f"Found visit_col={vhc_visit_col}, service_col={vhc_service_col}, copay_col={vhc_copay_col}"
        )
        return visits

    vhc = vhc.rename(
        columns={
            vhc_visit_col: "_vhc_visit_id",
            vhc_service_col: "_vhc_service_type",
            vhc_copay_col: "_vhc_copay",
        }
    )
    vhc["_vhc_visit_id"] = clean_id_series(vhc["_vhc_visit_id"])
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
        print(
            "[vhc] No rows with membership_funding_scheme == 'vhc' and visit_projected_price == 0 found in visits."
        )
        return visits

    vhc_visit_ids = visits.loc[mask_vhc, "visit_id"]
    hours = actual_hours.loc[mask_vhc]
    rates = vhc_visit_ids.map(vhc_agg["_vhc_rate"])
    copays = vhc_visit_ids.map(vhc_agg["_vhc_copay"])
    amounts = (rates.fillna(0.0) * hours) + copays.fillna(0.0)
    matched = vhc_visit_ids.isin(vhc_agg.index).sum()
    unmatched = int(mask_vhc.sum() - matched)

    visits.loc[mask_vhc, amount_col] = pd.to_numeric(amounts, errors="coerce").fillna(
        0.0
    )

    print(
        f"[vhc] Applied VHC pricing from {Path(vhc_claims_csv).name}: "
        f"vhc_rows={int(mask_vhc.sum()):,} matched={int(matched):,} unmatched={unmatched:,} (unmatched set to 0)"
    )
    return visits


def apply_chsp_claim_pricing(
    visits: pd.DataFrame,
    chsp_claims_csv: str,
    membership_scheme_col: str,
    amount_col: str = "visit_projected_price",
) -> pd.DataFrame:
    """
    For rows with membership_funding_scheme == 'chsp', set visit_projected_price from
    CHSP DEX report: (rate * time_hours) + Client contribution.
    """
    visit_id_col = first_existing_col_case_insensitive(
        visits, ["visit_id", "Visit ID", "visitId", "visit id"]
    )
    if visit_id_col is None:
        print("[chsp] Skipping CHSP pricing: visits missing visit_id column.")
        return visits

    if visit_id_col != "visit_id":
        visits = visits.rename(columns={visit_id_col: "visit_id"})
    visits["visit_id"] = clean_id_series(visits["visit_id"])

    chsp = read_csv(chsp_claims_csv)
    outlet_col = first_existing_col_case_insensitive(
        chsp, ["Outlet ID", "Outlet Id", "outlet_id"]
    )
    service_type_col = first_existing_col_case_insensitive(
        chsp,
        ["Service type ID", "Service Type ID", "Service type Id", "service_type_id"],
    )
    time_col = first_existing_col_case_insensitive(
        chsp, ["Time minutes", "Time Minutes", "time_minutes"]
    )
    client_contrib_col = first_existing_col_case_insensitive(
        chsp, ["Client contribution", "Client Contribution", "client_contribution"]
    )
    session_col = first_existing_col_case_insensitive(
        chsp, ["Session ID", "Session Id", "session_id"]
    )

    if not all([outlet_col, time_col, client_contrib_col, session_col]):
        print(
            f"[chsp] Skipping CHSP pricing: CHSP claims missing columns. "
            f"Need Outlet ID, Time minutes, Client contribution, Session ID. "
            f"Found outlet={outlet_col}, time={time_col}, client_contrib={client_contrib_col}, session={session_col}"
        )
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
    chsp["_chsp_visit_id"] = clean_id_series(chsp["_chsp_visit_id"])

    chsp_map = (
        chsp.loc[chsp["_chsp_visit_id"].notna()]
        .groupby("_chsp_visit_id", as_index=True)["_chsp_revenue"]
        .sum()
    )

    scheme = visits[membership_scheme_col].astype("string").str.strip().str.lower()
    mask_chsp = scheme.eq("chsp") & visits["visit_id"].notna()

    if not mask_chsp.any():
        print(
            "[chsp] No rows with membership_funding_scheme == 'chsp' found in visits."
        )
        return visits

    mapped = visits.loc[mask_chsp, "visit_id"].map(chsp_map)
    matched = mapped.notna().sum()
    unmatched = int(mask_chsp.sum() - matched)

    visits.loc[mask_chsp, amount_col] = pd.to_numeric(mapped, errors="coerce").fillna(
        0.0
    )

    print(
        f"[chsp] Applied CHSP pricing from {Path(chsp_claims_csv).name}: "
        f"chsp_rows={int(mask_chsp.sum()):,} matched={int(matched):,} unmatched={unmatched:,} (unmatched set to 0)"
    )
    return visits


def apply_claim_pricing_to_visits(
    visits: pd.DataFrame,
    *,
    dva_claims_csv: Optional[str] = None,
    vhc_claims_csv: Optional[str] = None,
    chsp_claims_csv: Optional[str] = None,
    membership_scheme_col: Optional[str] = None,
) -> pd.DataFrame:
    """Apply DVA, VHC, and CHSP claim pricing when the corresponding CSV and membership_scheme_col are provided."""
    if dva_claims_csv and membership_scheme_col:
        visits = apply_dva_claim_pricing(
            visits=visits,
            dva_claims_csv=dva_claims_csv,
            membership_scheme_col=membership_scheme_col,
            amount_col="visit_projected_price",
        )
    if vhc_claims_csv and membership_scheme_col:
        visits = apply_vhc_claim_pricing(
            visits=visits,
            vhc_claims_csv=vhc_claims_csv,
            membership_scheme_col=membership_scheme_col,
            amount_col="visit_projected_price",
        )
    if chsp_claims_csv and membership_scheme_col:
        visits = apply_chsp_claim_pricing(
            visits=visits,
            chsp_claims_csv=chsp_claims_csv,
            membership_scheme_col=membership_scheme_col,
            amount_col="visit_projected_price",
        )
    return visits
