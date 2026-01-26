from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

from config import API_KEY, COMPANY_ID


def get_request_headers(
    api_key: Optional[str] = API_KEY, is_create: bool = False
) -> Dict[str, str]:
    if not api_key:
        raise ValueError(
            "Missing API key. Set LOOKOUT_API_KEY env var (or API_KEY in config.py)."
        )

    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_key}",
    }
    if is_create:
        headers["content-type"] = "application/json"
    return headers


def retrieve_client(
    client_id: str,
    company_id: Optional[str] = COMPANY_ID,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    NOTE: Your pseudo-code calls this endpoint /clients/{client_id}.
    Here we treat membership_id as the identifier you want to query.
    """
    if not company_id:
        raise ValueError(
            "Missing company id. Set LOOKOUT_COMPANY_ID env var (or COMPANY_ID in config.py)."
        )

    url = f"https://api.thelookoutapp.com/api/{company_id}/clients/{client_id}"
    sess = session or requests.Session()
    resp = sess.get(url, headers=get_request_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def _extract_lonlat(payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Extracts profile.lonlat.latitude / longitude from response JSON.
    Returns (lat, lon) or (None, None) if missing.
    """
    try:
        prof = payload.get("profile") or {}
        lonlat = prof.get("lonlat") or {}
        print(lonlat)
        lat = lonlat.get("latitude")
        lon = lonlat.get("longitude")
        lat_f = float(lat) if lat is not None else None
        lon_f = float(lon) if lon is not None else None
        return lat_f, lon_f
    except Exception:
        return None, None


@dataclass
class LonLatEnrichmentStats:
    total_rows: int
    rows_with_membership_id: int
    unique_ids: int
    fetched: int
    matched: int
    failed: int


def enrich_visits_with_lonlat(
    visits_df,
    membership_id_col: str = "membership_id",
    test_max_rows: int = 0,
    sleep_seconds: float = 0.0,
) -> Tuple[Any, LonLatEnrichmentStats]:
    """
    Adds 'latitude' and 'longitude' columns to visits_df by looking up each membership_id
    via retrieve_client() and extracting profile.lonlat.latitude/longitude.

    - test_max_rows: if > 0, only processes the first N rows of visits_df (useful for testing).
    - sleep_seconds: optional throttle between API calls (set e.g. 0.05–0.2 if rate limits apply).

    Returns (df, stats).
    """
    import pandas as pd  # local import to keep utils standalone if needed

    df = visits_df.copy()
    if membership_id_col not in df.columns:
        # No membership_id column -> return unchanged, but still add empty cols for schema consistency if you want.
        if "latitude" not in df.columns:
            df["latitude"] = pd.NA
        if "longitude" not in df.columns:
            df["longitude"] = pd.NA
        stats = LonLatEnrichmentStats(
            total_rows=len(df),
            rows_with_membership_id=0,
            unique_ids=0,
            fetched=0,
            matched=0,
            failed=0,
        )
        return df, stats

    # Ensure columns exist
    if "latitude" not in df.columns:
        df["latitude"] = pd.NA
    if "longitude" not in df.columns:
        df["longitude"] = pd.NA

    work_df = df.head(test_max_rows) if test_max_rows and test_max_rows > 0 else df

    # Build unique id list (stringy, nonblank)
    ids = (
        work_df[membership_id_col]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )
    mask = ids.notna()
    unique_ids = ids[mask].unique().tolist()

    cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    fetched = 0
    matched = 0
    failed = 0

    with requests.Session() as session:
        for mid in unique_ids:
            try:
                payload = retrieve_client(str(mid), session=session)
                lat, lon = _extract_lonlat(payload)
                print(f"[lonlat] membership_id={mid} -> lat={lat}, lon={lon}")
                cache[str(mid)] = (lat, lon)
                fetched += 1
                if lat is not None or lon is not None:
                    matched += 1
            except Exception:
                cache[str(mid)] = (None, None)
                fetched += 1
                failed += 1
            if sleep_seconds and sleep_seconds > 0:
                time.sleep(sleep_seconds)

    # Apply back to rows
    lat_series = ids.map(
        lambda x: cache.get(str(x), (None, None))[0] if x is not None else None
    )
    lon_series = ids.map(
        lambda x: cache.get(str(x), (None, None))[1] if x is not None else None
    )

    df.loc[work_df.index, "latitude"] = lat_series.values
    df.loc[work_df.index, "longitude"] = lon_series.values

    # Optional: place latitude/longitude next to membership_id (preserve column order otherwise)
    cols = list(df.columns)
    if membership_id_col in cols:
        # Move lat/lon to directly after membership_id
        for c in ["latitude", "longitude"]:
            if c in cols:
                cols.remove(c)
        insert_at = cols.index(membership_id_col) + 1
        cols[insert_at:insert_at] = ["latitude", "longitude"]
        df = df[cols]

    stats = LonLatEnrichmentStats(
        total_rows=len(df),
        rows_with_membership_id=int(mask.sum()) if hasattr(mask, "sum") else 0,
        unique_ids=len(unique_ids),
        fetched=fetched,
        matched=matched,
        failed=failed,
    )
    return df, stats
