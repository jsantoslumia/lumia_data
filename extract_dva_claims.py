import argparse
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

VISIT_IDS_RE = re.compile(
    r"\bVisit IDs\b\s*:\s*\[(.*?)\]", flags=re.IGNORECASE | re.DOTALL
)
VISIT_ID_RE = re.compile(r"\bVisit ID\b\s*:\s*(\d{6,})", flags=re.IGNORECASE)
ID_NUMBER_RE = re.compile(r"\b\d{6,}\b")


def parse_decimal_amount(x) -> Decimal | None:
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return Decimal(str(x))
    s = str(x).strip()
    s = re.sub(r"[^\d\.\-]", "", s)
    if not s:
        return None
    return Decimal(s)


def quantize_2dp(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def split_amount_to_cents(total: Decimal, n: int) -> list[Decimal]:
    """Split total into n 2dp amounts that sum EXACTLY to total."""
    if n <= 0:
        return []
    total_2dp = quantize_2dp(total)
    cents = int((total_2dp * 100).to_integral_value())
    q, r = divmod(cents, n)
    parts_cents = [q + (1 if i < r else 0) for i in range(n)]
    return [Decimal(p) / Decimal(100) for p in parts_cents]


def extract_visit_ids(description: str) -> tuple[list[str], bool, bool]:
    """
    Returns (ids, is_plural, is_truncated)
      - ids: list of extracted visit ids
      - is_plural: True if matched "Visit IDs: [...]"
      - is_truncated: True if the bracket content contains "..."
    """
    if not isinstance(description, str):
        return [], False, False

    m = VISIT_IDS_RE.search(description)
    if m:
        inside = m.group(1)
        is_truncated = "..." in inside
        ids = ID_NUMBER_RE.findall(inside)
        return ids, True, is_truncated

    m = VISIT_ID_RE.search(description)
    if m:
        return [m.group(1)], False, False

    return [], False, False


def expand_rows(
    df: pd.DataFrame,
    description_col: str = "Description",
    amount_col: str = "ChargeAmount*",
    visit_id_col: str = "VisitId",
    allow_truncated: bool = False,
) -> pd.DataFrame:
    out_rows: list[pd.Series] = []

    for _, row in df.iterrows():
        desc = row.get(description_col)
        ids, is_plural, is_truncated = extract_visit_ids(desc)

        # Ignore descriptions we don't parse
        if not ids:
            new_row = row.copy()
            new_row[visit_id_col] = pd.NA
            out_rows.append(new_row)
            continue

        # If the visit list is truncated (contains "..."), splitting is unsafe by default
        if is_plural and is_truncated and not allow_truncated:
            new_row = row.copy()
            new_row[visit_id_col] = pd.NA
            out_rows.append(new_row)
            continue

        total = parse_decimal_amount(row.get(amount_col))

        # If we can't parse the amount, still expand IDs but leave amount unchanged
        if total is None:
            for vid in ids:
                new_row = row.copy()
                new_row[visit_id_col] = vid
                out_rows.append(new_row)
            continue

        parts = (
            split_amount_to_cents(total, len(ids))
            if is_plural
            else [quantize_2dp(total)]
        )

        for vid, part in zip(ids, parts):
            new_row = row.copy()
            new_row[visit_id_col] = vid
            # keep cents-correct
            new_row[amount_col] = float(quantize_2dp(part))
            out_rows.append(new_row)

    return pd.DataFrame(out_rows)


def iter_input_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    files = [p for p in input_path.glob(pattern) if p.is_file()]
    # Include uppercase .CSV if pattern ends in .csv
    if pattern.lower().endswith(".csv"):
        files += [p for p in input_path.glob(pattern[:-4] + ".CSV") if p.is_file()]

    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "input_path", help="Path to a CSV file OR a directory containing CSV files"
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write dva_claims_expanded.csv into",
    )
    ap.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern when input_path is a directory (default: *.csv)",
    )
    ap.add_argument("--description-col", default="Description")
    ap.add_argument("--amount-col", default="ChargeAmount*")
    ap.add_argument("--visit-id-col", default="VisitId")
    ap.add_argument(
        "--allow-truncated",
        action="store_true",
        help='If "Visit IDs: [...]" contains "...", still split using the IDs that are present.',
    )

    args = ap.parse_args()

    # Support accidental "input_path=./path" usage (argparse treats it as one positional)
    raw = getattr(args, "input_path", "")
    if isinstance(raw, str) and raw.startswith("input_path="):
        raw = raw.split("=", 1)[1].strip()
    input_path = Path(raw)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = iter_input_files(input_path, args.pattern)
    if not files:
        raise SystemExit(
            f"No input files found in {input_path} matching pattern {args.pattern!r}"
        )

    expanded_frames: list[pd.DataFrame] = []
    total_in = 0
    total_out = 0

    for f in files:
        df = pd.read_csv(f)
        expanded = expand_rows(
            df,
            description_col=args.description_col,
            amount_col=args.amount_col,
            visit_id_col=args.visit_id_col,
            allow_truncated=args.allow_truncated,
        )
        expanded_frames.append(expanded)

        total_in += len(df)
        total_out += len(expanded)
        print(f"Processed: {f.name}  (rows: {len(df)} -> {len(expanded)})")

    combined = pd.concat(expanded_frames, ignore_index=True)
    out_path = output_dir / "dva_claims_expanded.csv"
    combined.to_csv(out_path, index=False)

    print(f"\nWrote: {out_path}")
    print(f"Done. Files: {len(files)} | Total rows: {total_in} -> {total_out}")


if __name__ == "__main__":
    main()
