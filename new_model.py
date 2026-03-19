from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GL_MAP = {
    "50001": "50001 - Direct Wages - General",
    "50007": "50007 - Direct Wages - Broken Shift Allowance",
    "50008": "50008 - Direct Wages - Minimum 2 hour Allowance",
    "50010": "50010 - Direct Wages - Overtime",
    "50011": "50011 - Direct Wages - Public Holiday",
    "50012": "50012 - Direct Wages - Travel",
    "50013": "50013 - Direct Wages - Other Allowances",
}


def round_2(series: pd.Series) -> pd.Series:
    return series.round(2)


def derive_class_group(class_value: object) -> str:
    if pd.isna(class_value):
        return "other"
    s = str(class_value)
    if s.startswith("12"):
        return "12"
    if s.startswith("13"):
        return "13"
    if s.startswith("14"):
        return "14"
    return "other"


# -----------------------------
# Input loading + mapping prep
# -----------------------------


def load_source_files(
    visits_csv_path: str | Path,
    shift_costs_csv_path: str | Path,
    mapping_excel_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    visits_df = pd.read_csv(visits_csv_path, low_memory=False)
    shift_costs_df = pd.read_csv(shift_costs_csv_path, low_memory=False)

    class_map_df = pd.read_excel(mapping_excel_path, sheet_name="Class")
    eh_map_df = pd.read_excel(mapping_excel_path, sheet_name="EH Mapping")

    return visits_df, shift_costs_df, class_map_df, eh_map_df


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def prepare_class_mapping(class_map_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the 'Class' sheet from the mapping workbook.

    Required columns:
      - visit_rate
      - Class

    Deduplicates by visit_rate to mimic a simple lookup table.
    """
    required = {"visit_rate", "Class"}
    missing = required - set(class_map_df.columns)
    if missing:
        raise ValueError(
            f"Class mapping sheet missing required columns: {sorted(missing)}"
        )

    df = class_map_df[["visit_rate", "Class"]].copy()
    df["visit_rate"] = df["visit_rate"].astype(str).str.strip()
    df["Class"] = df["Class"].apply(
        lambda x: None if pd.isna(x) or x == "#N/A" else str(x)
    )

    # Keep first mapping per visit_rate
    df = df.drop_duplicates(subset=["visit_rate"], keep="first").copy()
    return df


def prepare_eh_mapping(eh_map_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the 'EH Mapping' sheet from the mapping workbook.

    Required columns:
      - Rule Name
      - GL

    Deduplicates by Rule Name to mimic a simple lookup table.
    """
    required = {"Rule Name", "GL"}
    missing = required - set(eh_map_df.columns)
    if missing:
        raise ValueError(
            f"EH Mapping sheet missing required columns: {sorted(missing)}"
        )

    df = eh_map_df[["Rule Name", "GL"]].copy()
    df["Rule Name"] = df["Rule Name"].astype(str).str.strip()
    df["GL"] = df["GL"].apply(lambda x: None if pd.isna(x) else str(x).strip())

    # Keep first mapping per Rule Name
    df = df.drop_duplicates(subset=["Rule Name"], keep="first").copy()
    return df


def build_job_extract_from_visits(
    visits_df: pd.DataFrame,
    class_map_df: pd.DataFrame,
) -> pd.DataFrame:
    required = {"visit_shift_id", "actual_visit_hours", "visit_rate"}
    missing = required - set(visits_df.columns)
    if missing:
        raise ValueError(
            f"visits_report.csv missing required columns: {sorted(missing)}"
        )

    df = visits_df.copy()

    df["visit_rate"] = df["visit_rate"].astype(str).str.strip()
    class_map_df = class_map_df.copy()
    class_map_df["visit_rate"] = class_map_df["visit_rate"].astype(str).str.strip()

    df["actual_visit_hours"] = clean_numeric(df["actual_visit_hours"])

    df = df.merge(
        class_map_df,
        on="visit_rate",
        how="left",
    )

    return df


def build_cost_line_from_shift_costs(
    shift_costs_df: pd.DataFrame,
    eh_mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    required = {"shift_id", "Rate", "Units", "Rule Name"}
    missing = required - set(shift_costs_df.columns)
    if missing:
        raise ValueError(f"shift_costs.csv missing required columns: {sorted(missing)}")

    df = shift_costs_df.copy()

    df["Rule Name"] = df["Rule Name"].astype(str).str.strip()
    eh_mapping_df = eh_mapping_df.copy()
    eh_mapping_df["Rule Name"] = eh_mapping_df["Rule Name"].astype(str).str.strip()

    df["Rate"] = clean_numeric(df["Rate"])
    df["Units"] = clean_numeric(df["Units"])

    df["$"] = df["Rate"] * df["Units"]

    df = df.merge(
        eh_mapping_df,
        on="Rule Name",
        how="left",
    )

    df = df.rename(columns={"GL": "GL account"})

    return df


# -----------------------------
# Query replicas
# -----------------------------


def query_job_extract_base(job_df: pd.DataFrame) -> pd.DataFrame:
    """
    Power Query replica of JobExtract_Base, minus Location.
    """
    df = job_df.copy()

    df["visit_shift_id"] = pd.to_numeric(df["visit_shift_id"], errors="coerce").astype(
        "Int64"
    )
    df["actual_visit_hours"] = clean_numeric(df["actual_visit_hours"])

    df = df[df["visit_shift_id"].notna()].copy()

    df["Class"] = df["Class"].apply(
        lambda x: None if pd.isna(x) or x == "#N/A" else str(x)
    )

    df = df[["visit_shift_id", "Class", "actual_visit_hours"]].copy()
    return df


def query_class_hours_per_shift(job_extract_base: pd.DataFrame) -> pd.DataFrame:
    """
    Power Query replica of ClassHoursPerShift, minus Location.
    """
    df = job_extract_base.copy()

    total_hours = (
        df.groupby("visit_shift_id", dropna=False)["actual_visit_hours"]
        .sum()
        .reset_index(name="total_shift_hours")
    )

    df = df.merge(total_hours, on="visit_shift_id", how="left")

    df["proportion"] = np.where(
        df["total_shift_hours"] == 0,
        np.nan,
        df["actual_visit_hours"] / df["total_shift_hours"],
    )

    df["class_group"] = df["Class"].apply(derive_class_group)

    result = (
        df.groupby(["visit_shift_id", "Class", "class_group"], dropna=False)
        .agg(proportion=("proportion", "sum"))
        .reset_index()
    )

    return result


def query_shift_group_flags(class_hours_per_shift: pd.DataFrame) -> pd.DataFrame:
    """
    Power Query replica of ShiftGroupFlags.
    """
    counts = (
        class_hours_per_shift[["visit_shift_id", "class_group"]]
        .drop_duplicates()
        .groupby("visit_shift_id", dropna=False)
        .size()
        .reset_index(name="distinct_group_count")
    )

    counts["shift_type"] = np.where(
        counts["distinct_group_count"] == 1,
        "Phase 1",
        "Phase 2",
    )

    return counts[["visit_shift_id", "shift_type"]].copy()


def query_costline_grouped(cost_df: pd.DataFrame, gl_code: str) -> pd.DataFrame:
    """
    Replica of CostLine_500xx grouped queries.
    """
    gl_text = GL_MAP[gl_code]

    df = cost_df.copy()
    df = df[df["GL account"] == gl_text].copy()

    df["shift_id"] = pd.to_numeric(df["shift_id"], errors="coerce").astype("Int64")
    df["$"] = clean_numeric(df["$"])
    if "Rate" in df.columns:
        df["Rate"] = clean_numeric(df["Rate"])

    df = df[["shift_id", "$"]].copy()

    result = (
        df.groupby("shift_id", dropna=False)["$"].sum().reset_index(name="total_cost")
    )

    return result


def query_final_simple(
    costline_grouped: pd.DataFrame,
    class_hours_per_shift: pd.DataFrame,
    gl_code: str,
) -> pd.DataFrame:
    """
    Replica of Final_50007 / Final_50008 / Final_50012 / Final_50013.
    """
    df = costline_grouped.merge(
        class_hours_per_shift,
        left_on="shift_id",
        right_on="visit_shift_id",
        how="left",
    )

    df["allocated_cost"] = np.where(
        df["proportion"].isna(),
        df["total_cost"],
        round_2(df["total_cost"] * df["proportion"]),
    )

    df["GL_account"] = gl_code

    result = df[
        [
            "shift_id",
            "total_cost",
            "allocated_cost",
            "Class",
            "class_group",
            "GL_account",
        ]
    ].copy()

    return result


def query_phase1_final(
    costline_grouped: pd.DataFrame,
    shift_group_flags: pd.DataFrame,
    class_hours_per_shift: pd.DataFrame,
    gl_code: str,
) -> pd.DataFrame:
    """
    Replica of Final_50001_Phase1 / Final_50010_Phase1 / Final_50011_Phase1.
    """
    df = costline_grouped.merge(
        shift_group_flags,
        left_on="shift_id",
        right_on="visit_shift_id",
        how="left",
    )

    df = df[df["shift_type"] == "Phase 1"].copy()

    df = df.merge(
        class_hours_per_shift,
        left_on="shift_id",
        right_on="visit_shift_id",
        how="left",
    )

    df["allocated_cost"] = np.where(
        df["proportion"].isna(),
        df["total_cost"],
        round_2(df["total_cost"] * df["proportion"]),
    )

    df["GL_account"] = gl_code

    result = df[
        [
            "shift_id",
            "total_cost",
            "allocated_cost",
            "Class",
            "class_group",
            "GL_account",
        ]
    ].copy()

    return result


def query_costline_phase2_raw(
    cost_df: pd.DataFrame,
    shift_group_flags: pd.DataFrame,
    gl_code: str,
) -> pd.DataFrame:
    """
    Replica of CostLine_50001_Phase2 / CostLine_50010_Phase2 / CostLine_50011_Phase2.
    """
    gl_text = GL_MAP[gl_code]

    df = cost_df.copy()
    df = df[df["GL account"] == gl_text].copy()

    df["shift_id"] = pd.to_numeric(df["shift_id"], errors="coerce").astype("Int64")
    df["$"] = clean_numeric(df["$"])
    df["Rate"] = clean_numeric(df["Rate"])

    df = df[["shift_id", "$", "Rate"]].copy()

    df = df.merge(
        shift_group_flags,
        left_on="shift_id",
        right_on="visit_shift_id",
        how="left",
    )

    df = df[df["shift_type"] == "Phase 2"].copy()

    return df[["shift_id", "$", "Rate"]].copy()


def query_phase2_ranked(costline_phase2_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Replica of Phase2_Ranked / Phase2_Ranked_50010 / Phase2_Ranked_50011.
    """
    max_rate = (
        costline_phase2_raw.groupby("shift_id", dropna=False)["Rate"]
        .max()
        .reset_index(name="max_rate")
    )

    df = costline_phase2_raw.merge(max_rate, on="shift_id", how="left")

    df["line_type"] = np.where(
        df["Rate"] == df["max_rate"],
        "Group 12",
        "Other groups",
    )

    return df


def query_final_phase2(
    phase2_ranked: pd.DataFrame,
    class_hours_per_shift: pd.DataFrame,
    gl_code: str,
) -> pd.DataFrame:
    """
    Replica of Final_50001_Phase2 / Final_50010_Phase2 / Final_50011_Phase2.
    """
    df = phase2_ranked.merge(
        class_hours_per_shift,
        left_on="shift_id",
        right_on="visit_shift_id",
        how="left",
    )

    mask = ((df["line_type"] == "Group 12") & (df["class_group"] == "12")) | (
        (df["line_type"] == "Other groups") & (df["class_group"] != "12")
    )
    df = df[mask].copy()

    line_hours = (
        df.groupby(["shift_id", "line_type"], dropna=False)["proportion"]
        .sum()
        .reset_index(name="line_hours")
    )

    df = df.merge(line_hours, on=["shift_id", "line_type"], how="left")

    df["allocated_cost"] = np.where(
        df["line_hours"] == 0,
        np.nan,
        round_2(df["$"] * (df["proportion"] / df["line_hours"])),
    )

    df["GL_account"] = gl_code

    result = df[
        ["shift_id", "$", "allocated_cost", "Class", "class_group", "GL_account"]
    ].copy()

    return result


def query_phase2_no12(
    phase2_ranked: pd.DataFrame,
    class_hours_per_shift: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replica of Phase2_No12 / Phase2_No12_50010 / Phase2_No12_50011.
    """
    df = phase2_ranked.merge(
        class_hours_per_shift,
        left_on="shift_id",
        right_on="visit_shift_id",
        how="left",
    )

    grouped = (
        df.groupby("shift_id", dropna=False)
        .agg(
            groups=("class_group", lambda s: " + ".join(pd.unique(s.astype(str)))),
            total_cost=("$", "sum"),
        )
        .reset_index()
    )

    grouped = grouped[~grouped["groups"].str.contains("12", na=False)].copy()
    return grouped


def query_phase2_no12_allocated(
    phase2_ranked: pd.DataFrame,
    phase2_no12: pd.DataFrame,
    class_hours_per_shift: pd.DataFrame,
    gl_code: str,
) -> pd.DataFrame:
    """
    Replica of Phase2_No12_Allocated / _50010 / _50011.
    """
    df = phase2_ranked.merge(
        phase2_no12[["shift_id"]],
        on="shift_id",
        how="inner",
    )

    df = df.merge(
        class_hours_per_shift,
        left_on="shift_id",
        right_on="visit_shift_id",
        how="left",
    )

    df["allocated_cost"] = np.where(
        df["proportion"].isna(),
        df["$"],
        round_2(df["$"] * df["proportion"]),
    )

    df["GL_account"] = gl_code

    result = df[
        ["shift_id", "$", "allocated_cost", "Class", "class_group", "GL_account"]
    ].copy()

    return result


def query_phase2_combined(
    final_phase2: pd.DataFrame,
    phase2_no12_allocated: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replica of Final_50001_Phase2_Combined / Final_50010_Phase2_Combined /
    Final_50011_Phase2_Combined.

    Intentionally preserves workbook behavior exactly, including any duplicates.
    """
    return pd.concat([final_phase2, phase2_no12_allocated], ignore_index=True)


# -----------------------------
# Pipeline
# -----------------------------


def build_final_all_from_sources(
    visits_df: pd.DataFrame,
    shift_costs_df: pd.DataFrame,
    class_map_df: pd.DataFrame,
    eh_map_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    # Pre-build equivalent source tables
    class_map_prepared = prepare_class_mapping(class_map_df)
    eh_map_prepared = prepare_eh_mapping(eh_map_df)

    job_df = build_job_extract_from_visits(visits_df, class_map_prepared)
    cost_df = build_cost_line_from_shift_costs(shift_costs_df, eh_map_prepared)

    # Base queries
    job_extract_base = query_job_extract_base(job_df)
    class_hours_per_shift = query_class_hours_per_shift(job_extract_base)
    shift_group_flags = query_shift_group_flags(class_hours_per_shift)

    # Simple GLs
    costline_50007 = query_costline_grouped(cost_df, "50007")
    final_50007 = query_final_simple(costline_50007, class_hours_per_shift, "50007")

    costline_50008 = query_costline_grouped(cost_df, "50008")
    final_50008 = query_final_simple(costline_50008, class_hours_per_shift, "50008")

    costline_50012 = query_costline_grouped(cost_df, "50012")
    final_50012 = query_final_simple(costline_50012, class_hours_per_shift, "50012")

    costline_50013 = query_costline_grouped(cost_df, "50013")
    final_50013 = query_final_simple(costline_50013, class_hours_per_shift, "50013")

    # 50001
    costline_50001 = query_costline_grouped(cost_df, "50001")
    final_50001_phase1 = query_phase1_final(
        costline_50001, shift_group_flags, class_hours_per_shift, "50001"
    )

    costline_50001_phase2 = query_costline_phase2_raw(
        cost_df, shift_group_flags, "50001"
    )
    phase2_ranked = query_phase2_ranked(costline_50001_phase2)
    final_50001_phase2 = query_final_phase2(
        phase2_ranked, class_hours_per_shift, "50001"
    )
    phase2_no12 = query_phase2_no12(phase2_ranked, class_hours_per_shift)
    phase2_no12_allocated = query_phase2_no12_allocated(
        phase2_ranked, phase2_no12, class_hours_per_shift, "50001"
    )
    final_50001_phase2_combined = query_phase2_combined(
        final_50001_phase2, phase2_no12_allocated
    )

    # 50010
    costline_50010 = query_costline_grouped(cost_df, "50010")
    final_50010_phase1 = query_phase1_final(
        costline_50010, shift_group_flags, class_hours_per_shift, "50010"
    )

    costline_50010_phase2 = query_costline_phase2_raw(
        cost_df, shift_group_flags, "50010"
    )
    phase2_ranked_50010 = query_phase2_ranked(costline_50010_phase2)
    final_50010_phase2 = query_final_phase2(
        phase2_ranked_50010, class_hours_per_shift, "50010"
    )
    phase2_no12_50010 = query_phase2_no12(phase2_ranked_50010, class_hours_per_shift)
    phase2_no12_allocated_50010 = query_phase2_no12_allocated(
        phase2_ranked_50010, phase2_no12_50010, class_hours_per_shift, "50010"
    )
    final_50010_phase2_combined = query_phase2_combined(
        final_50010_phase2, phase2_no12_allocated_50010
    )

    # 50011
    costline_50011 = query_costline_grouped(cost_df, "50011")
    final_50011_phase1 = query_phase1_final(
        costline_50011, shift_group_flags, class_hours_per_shift, "50011"
    )

    costline_50011_phase2 = query_costline_phase2_raw(
        cost_df, shift_group_flags, "50011"
    )
    phase2_ranked_50011 = query_phase2_ranked(costline_50011_phase2)
    final_50011_phase2 = query_final_phase2(
        phase2_ranked_50011, class_hours_per_shift, "50011"
    )
    phase2_no12_50011 = query_phase2_no12(phase2_ranked_50011, class_hours_per_shift)
    phase2_no12_allocated_50011 = query_phase2_no12_allocated(
        phase2_ranked_50011, phase2_no12_50011, class_hours_per_shift, "50011"
    )
    final_50011_phase2_combined = query_phase2_combined(
        final_50011_phase2, phase2_no12_allocated_50011
    )

    # Final append order exactly matches workbook
    final_all = pd.concat(
        [
            final_50007,
            final_50008,
            final_50001_phase1,
            final_50001_phase2_combined,
            final_50010_phase1,
            final_50010_phase2_combined,
            final_50012,
            final_50013,
            final_50011_phase1,
            final_50011_phase2_combined,
        ],
        ignore_index=True,
    )

    return {
        "Job_Input_Mapped": job_df,
        "Cost_Input_Mapped": cost_df,
        "JobExtract_Base": job_extract_base,
        "ClassHoursPerShift": class_hours_per_shift,
        "ShiftGroupFlags": shift_group_flags,
        "Final_50007": final_50007,
        "Final_50008": final_50008,
        "Final_50001_Phase1": final_50001_phase1,
        "Final_50001_Phase2_Combined": final_50001_phase2_combined,
        "Final_50010_Phase1": final_50010_phase1,
        "Final_50010_Phase2_Combined": final_50010_phase2_combined,
        "Final_50012": final_50012,
        "Final_50013": final_50013,
        "Final_50011_Phase1": final_50011_phase1,
        "Final_50011_Phase2_Combined": final_50011_phase2_combined,
        "Final_All": final_all,
    }


def main() -> None:
    visits_csv = "visits_report.csv"
    shift_costs_csv = "shift_costs.csv"
    mapping_file = "wages_allocation_mapping.xlsx"

    visits_df, shift_costs_df, class_map_df, eh_map_df = load_source_files(
        visits_csv_path=visits_csv,
        shift_costs_csv_path=shift_costs_csv,
        mapping_excel_path=mapping_file,
    )

    outputs = build_final_all_from_sources(
        visits_df=visits_df,
        shift_costs_df=shift_costs_df,
        class_map_df=class_map_df,
        eh_map_df=eh_map_df,
    )

    outputs["Final_All"].to_excel("shift_costs_allocation.xlsx", index=False)

    mapped_job = outputs["Job_Input_Mapped"]
    mapped_cost = outputs["Cost_Input_Mapped"]

    print("Job rows:", len(mapped_job))
    print("Cost rows:", len(mapped_cost))

    print("Mapped Class nulls:", mapped_job["Class"].isna().sum())
    print("Mapped GL account nulls:", mapped_cost["GL account"].isna().sum())

    print("Cost $ summary:")
    print(mapped_cost["$"].describe())

    print("GL account sample:")
    print(mapped_cost["GL account"].dropna().value_counts().head(20))

    with pd.ExcelWriter("Replicated_Query_Output.xlsx", engine="openpyxl") as writer:
        for name, df in outputs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    print("Done.")
    print(f"Final_All rows: {len(outputs['Final_All'])}")
    print("Files written:")
    print(" - shift_costs_allocation.xlsx")
    print(" - Replicated_Query_Output.xlsx")


if __name__ == "__main__":
    main()
