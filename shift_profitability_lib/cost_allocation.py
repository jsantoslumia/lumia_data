"""Cost allocation from shift costs (GL/Rate lines) to visits; optional legacy helper-hours mode."""

from __future__ import annotations

import re
from typing import Optional, Set

import numpy as np
import pandas as pd

from shift_profitability_lib.utils import (
    clean_id_series,
    first_existing_col_case_insensitive,
    read_csv,
    safe_round,
)

# GL accounts: wages / overtime / public holiday → Phase 1 + Phase 2 when cross-group + Aged Care
WAGE_GLS: Set[int] = frozenset({50001, 50010, 50011})
# Always split by visit hours only (no Phase 2)
HOURS_ONLY_GLS: Set[int] = frozenset({50007, 50008, 50012, 50013})

ONCOST_FACTOR = 1.2075


def _gl_to_int(val: object) -> Optional[int]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "<na>"):
        return None
    n = pd.to_numeric(s, errors="coerce")
    if pd.notna(n):
        return int(round(float(n)))
    m = re.search(r"\b(500\d{2})\b", s)
    if m:
        return int(m.group(1))
    return None


def _visit_class_is_aged_care(class_series: pd.Series) -> np.ndarray:
    s = class_series.astype("string").fillna("")
    return s.str.contains("12 aged care", case=False, regex=True).to_numpy()


def load_cost_lines_for_allocation(
    costs_csv: str,
    class_mapping_excel: Optional[str] = None,
    allowed_shift_ids: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Load costs CSV; optional EH Mapping merge for GL; return shift_id, gl, rate, row_cost.
    """
    costs = read_csv(costs_csv)
    if "shift_id" not in costs.columns:
        raise ValueError("Costs CSV must contain shift_id.")
    costs = costs.copy()
    costs["shift_id"] = clean_id_series(costs["shift_id"])
    costs = costs.loc[costs["shift_id"].notna()].copy()

    if allowed_shift_ids is not None:
        sid = costs["shift_id"].astype("string")
        costs = costs.loc[sid.isin(allowed_shift_ids)].copy()

    rule_col = first_existing_col_case_insensitive(costs, ["Rule Name"])
    if class_mapping_excel and rule_col:
        from shift_profitability_lib.class_mapping import merge_costs_gl_from_excel

        costs = merge_costs_gl_from_excel(costs, class_mapping_excel)

    gl_col = first_existing_col_case_insensitive(
        costs, ["GL", "GL_account", "gl_account"]
    )
    if gl_col is None:
        return pd.DataFrame(columns=["shift_id", "gl", "rate", "row_cost"])

    amount_candidates = [
        "shift_cost_line_amount",
        "cost_amount",
        "amount",
        "rate",
    ]
    units_candidates = ["shift_cost_line_units", "units", "Units", "qty", "quantity"]
    amount_col = first_existing_col_case_insensitive(costs, amount_candidates)
    units_col = first_existing_col_case_insensitive(costs, units_candidates)
    if amount_col is None:
        raise ValueError(f"Costs CSV missing amount column. Tried: {amount_candidates}")
    if units_col is None:
        costs["_units"] = 1.0
        units_col = "_units"

    amt = pd.to_numeric(costs[amount_col], errors="coerce").fillna(0.0)
    uts = pd.to_numeric(costs[units_col], errors="coerce").fillna(0.0)
    row_cost = amt * uts

    rate_col = first_existing_col_case_insensitive(costs, ["Rate", "rate"])
    if rate_col:
        rate = pd.to_numeric(costs[rate_col], errors="coerce").fillna(0.0)
    else:
        rate = pd.Series(0.0, index=costs.index)

    gl_int = costs[gl_col].map(_gl_to_int)
    out = pd.DataFrame(
        {
            "shift_id": costs["shift_id"].astype("string"),
            "gl": gl_int,
            "rate": rate,
            "row_cost": row_cost,
        }
    )
    out = out.loc[out["gl"].notna() & (out["row_cost"] != 0)].copy()
    out["gl"] = out["gl"].astype(int)
    return out


def _allocate_shift_lines_to_visits(
    hours: np.ndarray,
    is_aged: np.ndarray,
    single_class: bool,
    lines: pd.DataFrame,
) -> np.ndarray:
    """Return per-visit pre-scale allocation for one shift (same order as hours/is_aged)."""
    n = len(hours)
    alloc = np.zeros(n, dtype=float)
    H = float(hours.sum())
    H_a = float(hours[is_aged].sum())
    H_o = float(hours[~is_aged].sum())
    has_aged = bool(is_aged.any())

    if n == 0:
        return alloc

    def hours_split(total: float) -> np.ndarray:
        if total == 0:
            return np.zeros(n, dtype=float)
        if H > 0:
            return total * (hours / H)
        return np.full(n, total / n, dtype=float)

    for gl, gdf in lines.groupby("gl", sort=False):
        gl = int(gl)
        wage = gl in WAGE_GLS
        hours_only = (gl in HOURS_ONLY_GLS) or (not wage)

        if hours_only or single_class or not has_aged:
            alloc += hours_split(float(gdf["row_cost"].sum()))
            continue

        # Cross-group + wage GL + at least one Aged Care visit on shift
        rates = gdf["rate"].to_numpy(dtype=float)
        costs_gl = gdf["row_cost"].to_numpy(dtype=float)
        max_r = float(np.nanmax(rates)) if len(rates) else 0.0
        if not np.isfinite(max_r):
            max_r = 0.0
        at_max = np.isclose(rates, max_r, rtol=1e-9, atol=1e-6)
        high_c = float(costs_gl[at_max].sum())
        low_c = float(costs_gl[~at_max].sum())

        if H_a > 0:
            alloc[is_aged] += high_c * (hours[is_aged] / H_a)
        else:
            alloc += hours_split(high_c)

        if H_o > 0:
            alloc[~is_aged] += low_c * (hours[~is_aged] / H_o)
        elif H_a > 0:
            alloc[is_aged] += low_c * (hours[is_aged] / H_a)
        else:
            alloc += hours_split(low_c)

    return alloc


def apply_helper_hours_cost_allocation_to_visits(
    visits_enriched: pd.DataFrame,
    shift_profitability_feed: pd.DataFrame,
    *,
    costs_csv: Optional[str] = None,
    class_mapping_excel: Optional[str] = None,
) -> pd.DataFrame:
    """Add cost allocation columns to enriched visits.

    If ``costs_csv`` is provided (and GL is available after optional mapping merge),
    allocation is **per shift** from cost lines (GL + Rate): Phase 1 single-class by
    hours; Phase 2 for 50001/50010/50011 on cross-group shifts with Aged Care
    (highest-rate lines → Aged Care visits by hours; other lines → non–Aged Care by
    hours); 50007/50008/50012/50013 and unknown GLs always by hours. Per-shift line
    totals are scaled to match ``shift_profitability_feed.total_cost`` when they differ.

    Otherwise (no costs file): legacy helper-level hours weighting × oncost.

    Oncost 1.2075 is applied to ``visit_cost_allocated`` in both modes.
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

    shift_ids_in_visits = set(
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
        shift_lookup["shift_id"].isin(shift_ids_in_visits)
    ].copy()

    # One row per shift_id for merge (feed can duplicate helper_id)
    shift_cost_one = (
        shift_lookup.groupby("shift_id", as_index=False)["total_cost"]
        .first()
        .rename(columns={"total_cost": "_feed_total"})
    )
    shift_feed_totals = dict(
        zip(
            shift_cost_one["shift_id"].astype("string"),
            pd.to_numeric(shift_cost_one["_feed_total"], errors="coerce").fillna(0.0),
        )
    )

    valid_shift = visits["visit_shift_id"].notna()
    shift_rev = (
        visits.loc[valid_shift]
        .groupby("visit_shift_id", as_index=False)["visit_projected_price"]
        .sum()
        .rename(columns={"visit_projected_price": "shift_total_revenue"})
    )

    visits = visits.merge(
        shift_lookup[["shift_id", "total_cost"]]
        .drop_duplicates(subset=["shift_id"])
        .rename(
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

    cost_lines: Optional[pd.DataFrame] = None
    if costs_csv:
        cost_lines = load_cost_lines_for_allocation(
            costs_csv,
            class_mapping_excel=class_mapping_excel,
            allowed_shift_ids=shift_ids_set & shift_ids_in_visits,
        )

    use_line_allocation = cost_lines is not None and not cost_lines.empty

    if use_line_allocation:
        if "Class" not in visits.columns:
            visits["Class"] = pd.NA

        visit_pre = pd.Series(0.0, index=visits.index, dtype=float)

        for sid, grp in visits.groupby("visit_shift_id", sort=False):
            if sid is pd.NA or str(sid) == "<NA>":
                continue
            sid_s = str(sid)
            if sid_s not in shift_ids_set:
                continue

            lines = cost_lines.loc[cost_lines["shift_id"].astype("string") == sid_s]
            feed_total = float(shift_feed_totals.get(sid_s, 0.0))
            line_sum = float(lines["row_cost"].sum()) if not lines.empty else 0.0

            idx = grp.index
            hours = grp["actual_visit_hours"].to_numpy(dtype=float)
            cls = grp["Class"]
            is_aged = _visit_class_is_aged_care(cls)
            cls_key = cls.fillna("__NA__").astype("string").str.strip()
            single_class = cls_key.nunique() <= 1

            if lines.empty:
                if feed_total > 0 and hours.sum() > 0:
                    visit_pre.loc[idx] += feed_total * (hours / hours.sum())
                continue

            alloc = _allocate_shift_lines_to_visits(hours, is_aged, single_class, lines)
            pre_sum = float(alloc.sum())
            if line_sum > 0 and pre_sum > 0 and abs(feed_total - line_sum) > 1e-6:
                scale = feed_total / line_sum
                alloc = alloc * scale
            elif line_sum == 0 and feed_total > 0 and hours.sum() > 0:
                alloc = np.full(len(hours), feed_total / len(hours), dtype=float)
            elif pre_sum == 0 and feed_total > 0 and hours.sum() > 0:
                alloc = feed_total * (hours / hours.sum())

            visit_pre.loc[idx] += alloc

        visits["visit_cost_allocated"] = visit_pre * ONCOST_FACTOR
        visits["allocation_method"] = "shift_gl_class_weighted"
        helper_hours = (
            visits.loc[has_helper_id & exists_in_shift_feed]
            .groupby("helper_id", as_index=False)["actual_visit_hours"]
            .sum()
            .rename(columns={"actual_visit_hours": "_helper_total_hours"})
        )
        visits = visits.merge(helper_hours, on="helper_id", how="left")
        visits["_helper_total_hours"] = pd.to_numeric(
            visits["_helper_total_hours"], errors="coerce"
        ).fillna(0.0)
        ok_mask = (
            has_helper_id & exists_in_shift_feed & (visits["_helper_total_hours"] > 0)
        )
        zero_h = has_helper_id & (visits["_helper_total_hours"] <= 0)
        visits = visits.drop(
            columns=[c for c in ["_helper_total_hours"] if c in visits.columns]
        )
    else:
        # Legacy: helper-level hours × total_cost
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
                shift_cost_dedup.groupby("helper_id", as_index=False)[
                    "shift_total_cost"
                ]
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
        ok_mask = (
            has_helper_id & exists_in_shift_feed & (visits["_helper_total_hours"] > 0)
        )
        zero_h = has_helper_id & (visits["_helper_total_hours"] <= 0)
        visits.loc[ok_mask, "visit_cost_allocated"] = (
            visits.loc[ok_mask, "_helper_total_cost"]
            * (
                visits.loc[ok_mask, "actual_visit_hours"]
                / visits.loc[ok_mask, "_helper_total_hours"]
            )
            * ONCOST_FACTOR
        )
        visits["allocation_method"] = "helper_hours_weighted"
        visits = visits.drop(
            columns=[
                c
                for c in ["_helper_total_hours", "_helper_total_cost"]
                if c in visits.columns
            ]
        )

    visits["allocation_ok"] = ok_mask.reindex(visits.index, fill_value=False).astype(
        bool
    )
    reason = pd.Series("ok", index=visits.index, dtype="string")
    reason = reason.mask(~has_shift_id, "missing_shift_id")
    reason = reason.mask(~has_helper_id, "missing_helper_id")
    reason = reason.mask(has_shift_id & ~exists_in_shift_feed, "no_shift_match")
    reason = reason.mask(zero_h, "zero_helper_hours")
    visits["allocation_reason"] = reason

    for c in ["shift_total_cost", "shift_total_revenue", "visit_cost_allocated"]:
        safe_round(visits, c, 2)

    return visits
