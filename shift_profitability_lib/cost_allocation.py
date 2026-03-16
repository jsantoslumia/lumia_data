"""Helper-hours-weighted cost allocation from shift feed to visits."""

from __future__ import annotations

import pandas as pd

from shift_profitability_lib.utils import clean_id_series, safe_round


def apply_helper_hours_cost_allocation_to_visits(
    visits_enriched: pd.DataFrame,
    shift_profitability_feed: pd.DataFrame,
) -> pd.DataFrame:
    """Add helper-hours-weighted cost allocation columns to an enriched visits export.

    Allocation logic (per helper), using actual_visit_hours:
      helper_total_cost  = SUM(total_cost) across shifts for that helper
      helper_total_hours = SUM(actual_visit_hours) across visits for that helper
      visit_cost_allocated = helper_total_cost * (actual_visit_hours / helper_total_hours)
      then visit_cost_allocated *= 1.2075 (oncosts).
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

    visits["visit_shift_id"] = clean_id_series(visits["visit_shift_id"])
    visits["helper_id"] = clean_id_series(visits["helper_id"])
    visits["actual_visit_hours"] = pd.to_numeric(
        visits["actual_visit_hours"], errors="coerce"
    ).fillna(0.0)

    shift_ids_in_visits = (
        visits["visit_shift_id"].dropna().astype("string").unique().tolist()
    )

    shift_lookup_cols = ["shift_id", "total_cost"]
    if "helper_id" in shift_profitability_feed.columns:
        shift_lookup_cols.append("helper_id")

    shift_lookup = shift_profitability_feed[shift_lookup_cols].copy()
    shift_lookup["shift_id"] = clean_id_series(shift_lookup["shift_id"])
    if "helper_id" in shift_lookup.columns:
        shift_lookup["helper_id"] = clean_id_series(shift_lookup["helper_id"])

    shift_lookup = shift_lookup.loc[
        shift_lookup["shift_id"].isin(set(shift_ids_in_visits))
    ].copy()

    valid_shift = visits["visit_shift_id"].notna()
    shift_rev = (
        visits.loc[valid_shift]
        .groupby("visit_shift_id", as_index=False)["visit_projected_price"]
        .sum()
        .rename(columns={"visit_projected_price": "shift_total_revenue"})
    )

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

    shift_ids_set = set(shift_lookup["shift_id"].dropna().astype("string").tolist())
    exists_in_shift_feed = has_shift_id & visits["visit_shift_id"].astype(
        "string"
    ).isin(shift_ids_set)

    helper_hours = (
        visits.loc[has_helper_id & exists_in_shift_feed]
        .groupby("helper_id", as_index=False)["actual_visit_hours"]
        .sum()
        .rename(columns={"actual_visit_hours": "_helper_total_hours"})
    )

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

    visits.loc[ok_mask, "visit_cost_allocated"] = (
        visits.loc[ok_mask, "visit_cost_allocated"] * 1.2075
    )

    visits["allocation_method"] = "helper_hours_weighted"
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
        safe_round(visits, c, 2)

    visits = visits.drop(
        columns=[
            c
            for c in ["_helper_total_hours", "_helper_total_cost"]
            if c in visits.columns
        ]
    )

    return visits
