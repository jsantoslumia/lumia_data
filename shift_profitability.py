#!/usr/bin/env python3
"""
py -m shift_profitability --visits ./input_files/visit_export_feb.csv --costs ./input_files/shift_costs_feb.csv --out-dir . --out-visits visits_export_enriched.csv --sah-transactions ./input_files/sah_transactions_feb.csv --out-sah-purchases memberships_sah_purchases.csv --dva-claims ./dva_claims_expanded.csv --vhc-claims ./vhc_claims.csv --chsp-claims ./chsp_dex_report.csv
"""

# TODO: need to compute sum of Units from keypay report and add somewhere

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# SAH purchases and revenue (same logic as shift_profitability_sah)
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


# VHC Service Type rates (per hour); blank/unknown => 0
VHC_SERVICE_TYPE_RATES: Dict[str, float] = {
    "da": 98.72,
    "pc": 115.05,
    "ri": 73.0,
}


def _apply_vhc_claim_pricing(
    visits: pd.DataFrame,
    vhc_claims_csv: str,
    membership_scheme_col: str,
    amount_col: str = "visit_projected_price",
) -> pd.DataFrame:
    """
    For rows with membership_funding_scheme == 'vhc' and visit_projected_price == 0,
    set visit claim from vhc_claims CSV: (rate * actual_visit_hours) + Co-payment amount,
    where rate comes from Service Type (DA=$98.72, PC=$115.05, RI=$73). Requires Visit ID,
    Service Type, and Co-payment amount columns in the CSV.
    """
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
        print("[vhc] Skipping VHC pricing: visits missing actual_visit_hours column.")
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

    # One row per Visit ID: take first rate and sum co-payments if duplicates
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
    # claim = rate * actual_visit_hours + co_payment
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


def _apply_chsp_claim_pricing(
    visits: pd.DataFrame,
    chsp_claims_csv: str,
    membership_scheme_col: str,
    amount_col: str = "visit_projected_price",
) -> pd.DataFrame:
    """
    For rows with membership_funding_scheme == 'chsp', set visit_projected_price from
    CHSP DEX report: (rate * time_hours) + Client contribution, where rate is derived
    from Outlet ID (and Service type ID when Outlet contains 'Community Home Support').
    Visit id is matched via Session ID (strip '_visit_fee' suffix if present).
    """
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

    # Normalize Session ID to visit id: strip trailing '_visit_fee' if present
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


def build_enriched_visits_export(
    visits_csv: str,
    exclude_zero_revenue_visits: bool = False,
    dva_claims_csv: Optional[str] = None,
    vhc_claims_csv: Optional[str] = None,
    chsp_claims_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Read the raw visit export and return an enriched visit-level table.

    Enrichment performed:
      - Normalizes ID columns (visit_shift_id, helper_id, visit_id where present)
      - Cleans membership_* string columns (trims whitespace)
      - Ensures visit_projected_price is numeric
      - If dva_claims_csv is provided and membership_funding_scheme exists, applies
        DVA claim pricing overrides via _apply_dva_claim_pricing.
      - If vhc_claims_csv is provided and membership_funding_scheme exists, applies
        VHC claim pricing via _apply_vhc_claim_pricing (rate * actual_visit_hours + co-payment).
      - If chsp_claims_csv is provided and membership_funding_scheme exists, applies
        CHSP claim pricing via _apply_chsp_claim_pricing ((rate * time_hours) + client contribution).

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
    # Apply VHC claim pricing (rate * actual_visit_hours + co-payment) at visit grain
    if vhc_claims_csv and membership_scheme_col:
        visits = _apply_vhc_claim_pricing(
            visits=visits,
            vhc_claims_csv=vhc_claims_csv,
            membership_scheme_col=membership_scheme_col,
            amount_col="visit_projected_price",
        )
    # Apply CHSP claim pricing ((rate * time_hours) + client contribution) at visit grain
    if chsp_claims_csv and membership_scheme_col:
        visits = _apply_chsp_claim_pricing(
            visits=visits,
            chsp_claims_csv=chsp_claims_csv,
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
    """Add HELPER-hours-weighted cost allocation columns to an enriched visits export.

    Same logic as shift_profitability_sah: allocation is per helper using actual_visit_hours,
    then oncost factor 1.2075 is applied. Preserves Power BI output columns:
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
                          (restricted to visits whose shift is in the shift feed)
      visit_cost_allocated = helper_total_cost * (actual_visit_hours / helper_total_hours)
      then visit_cost_allocated *= 1.2075 (oncosts).

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
        visits["helper_id"] = pd.NA
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

    # Helper total hours (B): only count hours from visits whose shift is in the feed
    # (match shift_profitability_sah so SUM(visit_cost_allocated) per helper = helper_total_cost * 1.2075)
    helper_hours = (
        visits.loc[has_helper_id & exists_in_shift_feed]
        .groupby("helper_id", as_index=False)["actual_visit_hours"]
        .sum()
        .rename(columns={"actual_visit_hours": "_helper_total_hours"})
    )

    # Helper total cost: SUM(total_cost) across shifts for that helper (restricted to shifts_in_visits)
    if "helper_id" in shift_lookup.columns:
        helper_cost = (
            shift_lookup.loc[shift_lookup["helper_id"].notna()]
            .drop_duplicates(subset=["shift_id"])
            .groupby("helper_id", as_index=False)["total_cost"]
            .sum()
            .rename(columns={"total_cost": "_helper_total_cost"})
        )
    else:
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

    # Apply oncosts directly to visit_cost_allocated (accountant factor, same as shift_profitability_sah)
    visits.loc[ok_mask, "visit_cost_allocated"] = (
        visits.loc[ok_mask, "visit_cost_allocated"] * 1.2075
    )

    visits["allocation_method"] = "helper_hours_weighted"
    # Ensure allocation_ok is a proper column (aligned to visits.index, bool dtype for stable CSV export)
    visits["allocation_ok"] = ok_mask.reindex(visits.index, fill_value=False).astype(
        bool
    )

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
    dva_claims_csv: Optional[str] = None,
    vhc_claims_csv: Optional[str] = None,
    chsp_claims_csv: Optional[str] = None,
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
    # ✅ Preprocess visit_export prices for VHC (rate * actual_visit_hours + co-payment)
    if vhc_claims_csv and membership_scheme_col:
        visits = _apply_vhc_claim_pricing(
            visits=visits,
            vhc_claims_csv=vhc_claims_csv,
            membership_scheme_col=membership_scheme_col,
            amount_col="visit_projected_price",
        )
    # ✅ Preprocess visit_export prices for CHSP ((rate * time_hours) + client contribution)
    if chsp_claims_csv and membership_scheme_col:
        visits = _apply_chsp_claim_pricing(
            visits=visits,
            chsp_claims_csv=chsp_claims_csv,
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

    # Restrict costs to shifts that appear in the visit export (same as shift_profitability_sah)
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
    parser.add_argument(
        "--vhc-claims",
        default=None,
        help="Optional path to VHC claims CSV (columns: Visit ID, Service Type [DA/PC/RI], Co-payment amount). VHC visits are priced as rate*actual_visit_hours + co-payment.",
    )
    parser.add_argument(
        "--chsp-claims",
        default=None,
        help="Optional path to CHSP DEX report CSV (e.g. chsp_dex_report.csv). Columns: Outlet ID, Service type ID, Time minutes, Client contribution, Session ID. CHSP visits are priced as (rate*time_hours) + client contribution.",
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
    parser.add_argument(
        "--sah-transactions",
        default=None,
        help="Optional SAH transactions CSV path. If provided, generates memberships_sah_purchases.csv (and adds sah_revenue/sah_visit_revenue to --out-visits when used). Requires shift_profitability_sah to be importable.",
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

        # Enrich purchases with membership_name from visits (all visits, no SAH filter)
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
        dva_claims_csv=args.dva_claims,
        vhc_claims_csv=args.vhc_claims,
        chsp_claims_csv=args.chsp_claims,
    )

    out_path = out_dir / args.out
    _write_csv(df, out_path, utf8_bom=args.utf8_bom)

    # Optional: write enriched visit export (visit-level) with DVA/VHC pricing applied.
    if args.out_visits:
        visits_enriched = build_enriched_visits_export(
            visits_csv=args.visits,
            exclude_zero_revenue_visits=args.exclude_zero_revenue_visits,
            dva_claims_csv=args.dva_claims,
            vhc_claims_csv=args.vhc_claims,
            chsp_claims_csv=args.chsp_claims,
        )
        visits_enriched = apply_revenue_weighted_cost_allocation_to_visits(
            visits_enriched=visits_enriched,
            shift_profitability_feed=df,
        )

        # Add SAH revenue columns when SAH transactions were provided (same as shift_profitability_sah).
        # sah_revenue and sah_visit_revenue are null for visits that are not HCP/SAH or don't match a membership.
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
            )
            _safe_round(visits_enriched, "sah_revenue", 2)

            # Allocate membership-level SAH revenue down to visits by actual_visit_hours (same as shift_profitability_sah)
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

        # Ensure allocation columns are present for Power BI / schema stability
        if "allocation_ok" not in visits_enriched.columns:
            visits_enriched["allocation_ok"] = False
        else:
            visits_enriched["allocation_ok"] = visits_enriched["allocation_ok"].astype(
                bool
            )
        if "allocation_method" not in visits_enriched.columns:
            visits_enriched["allocation_method"] = ""
        if "allocation_reason" not in visits_enriched.columns:
            visits_enriched["allocation_reason"] = ""

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
