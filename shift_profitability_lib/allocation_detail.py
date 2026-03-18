"""Per-shift GL × Class allocation detail export (Power BI / reconciliation)."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from shift_profitability_lib.cost_allocation import (
    HOURS_ONLY_GLS,
    WAGE_GLS,
    load_cost_lines_for_allocation,
    _visit_class_is_aged_care,
)
from shift_profitability_lib.utils import (
    clean_id_series,
    first_existing_col_case_insensitive,
    read_csv,
    safe_round,
)


def _allocate_one_gl_to_visits(
    hours: np.ndarray,
    is_aged: np.ndarray,
    single_class: bool,
    gl: int,
    gdf: pd.DataFrame,
) -> np.ndarray:
    """Pre-oncost allocation of one GL's cost lines to visits (same rules as visit export)."""
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
        return np.full(n, total / max(n, 1), dtype=float)

    wage = gl in WAGE_GLS
    hours_only = (gl in HOURS_ONLY_GLS) or (not wage)

    if hours_only or single_class or not has_aged:
        return hours_split(float(gdf["row_cost"].sum()))

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


def _class_group_value(class_val: object) -> str:
    s = str(class_val).strip() if class_val is not None and not pd.isna(class_val) else ""
    if not s:
        return ""
    m = re.match(r"^\s*(\d+)", s)
    return m.group(1) if m else ""


def build_shift_gl_class_allocation_detail(
    visits_csv: str,
    costs_csv: str,
    shift_profitability_feed: pd.DataFrame,
    *,
    class_mapping_excel: Optional[str] = None,
    exclude_zero_revenue_visits: bool = False,
) -> pd.DataFrame:
    """
    Build a table with columns aligned to the allocation detail layout:

    shift_id, total_cost, allocated_cost, Class, Location, class_group,
    GL_account, Rate (max rate for that GL on the shift).

    ``total_cost`` = sum of cost lines for that shift × GL (pre-oncost).
    ``allocated_cost`` = amount attributed to that Class after Phase 1/2 rules,
    scaled to match ``shift_profitability_feed.total_cost`` per shift (pre-oncost).
    """
    visits = read_csv(visits_csv)
    if class_mapping_excel:
        from shift_profitability_lib.class_mapping import merge_visit_class_from_excel

        visits = merge_visit_class_from_excel(visits, class_mapping_excel)

    if "visit_shift_id" not in visits.columns and "shift_id" in visits.columns:
        visits = visits.rename(columns={"shift_id": "visit_shift_id"})
    if "visit_shift_id" not in visits.columns:
        raise ValueError("Visits CSV must contain visit_shift_id (or shift_id).")
    visits = visits.copy()
    visits["visit_shift_id"] = clean_id_series(visits["visit_shift_id"])
    visits = visits.loc[visits["visit_shift_id"].notna()].copy()

    if exclude_zero_revenue_visits and "visit_projected_price" in visits.columns:
        vp = pd.to_numeric(visits["visit_projected_price"], errors="coerce").fillna(0.0)
        visits = visits.loc[vp != 0].copy()

    loc_col = first_existing_col_case_insensitive(
        visits,
        ["Location", "location", "Helper Region", "helper_region", "membership_community_name"],
    )
    if loc_col:
        visits["_loc"] = visits[loc_col].astype("string").fillna("")
    else:
        visits["_loc"] = ""

    cg_col = first_existing_col_case_insensitive(visits, ["class_group", "Class Group"])
    if cg_col:
        visits["_cg"] = visits[cg_col].astype("string").fillna("").str.strip()
    else:
        visits["_cg"] = ""

    if "Class" not in visits.columns:
        visits["Class"] = pd.NA
    visits["Class"] = visits["Class"].astype("string").fillna("")

    visits["actual_visit_hours"] = pd.to_numeric(
        visits.get("actual_visit_hours", 0), errors="coerce"
    ).fillna(0.0)

    mask_cg = visits["_cg"].eq("")
    visits.loc[mask_cg, "_cg"] = visits.loc[mask_cg, "Class"].map(_class_group_value)

    shift_ids_visits = set(visits["visit_shift_id"].astype("string").unique().tolist())
    cost_lines = load_cost_lines_for_allocation(
        costs_csv,
        class_mapping_excel=class_mapping_excel,
        allowed_shift_ids=shift_ids_visits,
    )
    if cost_lines.empty:
        return pd.DataFrame(
            columns=[
                "shift_id",
                "total_cost",
                "allocated_cost",
                "Class",
                "Location",
                "class_group",
                "GL_account",
                "Rate",
            ]
        )

    feed = shift_profitability_feed.copy()
    feed["shift_id"] = clean_id_series(feed["shift_id"])
    feed_tot = (
        feed.groupby("shift_id", as_index=False)["total_cost"]
        .first()
        .assign(
            total_cost=lambda x: pd.to_numeric(x["total_cost"], errors="coerce").fillna(
                0.0
            )
        )
    )
    feed_map = dict(
        zip(feed_tot["shift_id"].astype("string"), feed_tot["total_cost"].astype(float))
    )

    rows: list[dict] = []
    for sid, grp in visits.groupby("visit_shift_id", sort=False):
        sid_s = str(sid)
        lines = cost_lines.loc[cost_lines["shift_id"].astype("string") == sid_s]
        if lines.empty:
            continue

        hours = grp["actual_visit_hours"].to_numpy(dtype=float)
        is_aged = _visit_class_is_aged_care(grp["Class"])
        cls_key = grp["Class"].astype("string").str.strip()
        single_class = cls_key.nunique() <= 1

        line_sum = float(lines["row_cost"].sum())
        feed_total = float(feed_map.get(sid_s, 0.0))
        if line_sum > 0:
            scale = feed_total / line_sum
        else:
            scale = 1.0

        idx = grp.index.tolist()
        for gl, gdf in lines.groupby("gl", sort=False):
            gl = int(gl)
            alloc = _allocate_one_gl_to_visits(hours, is_aged, single_class, gl, gdf)
            alloc = alloc * scale
            sub = grp.copy()
            sub["_alloc"] = alloc
            total_gl = float(gdf["row_cost"].sum())
            rate_max = float(np.nanmax(gdf["rate"].to_numpy())) if len(gdf) else 0.0
            if not np.isfinite(rate_max):
                rate_max = 0.0

            agg = (
                sub.groupby(["Class", "_loc", "_cg"], as_index=False, dropna=False)
                .agg(allocated_cost=("_alloc", "sum"))
                .rename(columns={"_loc": "Location", "_cg": "class_group"})
            )
            for _, r in agg.iterrows():
                alc = float(r["allocated_cost"])
                if alc == 0.0 and total_gl == 0.0:
                    continue
                rows.append(
                    {
                        "shift_id": sid_s,
                        "total_cost": total_gl,
                        "allocated_cost": alc,
                        "Class": str(r["Class"]) if pd.notna(r["Class"]) else "",
                        "Location": str(r["Location"]) if pd.notna(r["Location"]) else "",
                        "class_group": str(r["class_group"])
                        if pd.notna(r["class_group"])
                        else "",
                        "GL_account": gl,
                        "Rate": rate_max,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "shift_id",
                "total_cost",
                "allocated_cost",
                "Class",
                "Location",
                "class_group",
                "GL_account",
                "Rate",
            ]
        )

    col_order = [
        "shift_id",
        "total_cost",
        "allocated_cost",
        "Class",
        "Location",
        "class_group",
        "GL_account",
        "Rate",
    ]
    out = out[col_order]
    for c in ["total_cost", "allocated_cost", "Rate"]:
        safe_round(out, c, 2)
    out = out.sort_values(["shift_id", "GL_account", "Class"], kind="mergesort")
    return out.reset_index(drop=True)
