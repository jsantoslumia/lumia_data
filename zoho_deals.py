from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# ---- Domains (AU) ----
ACCOUNTS_BASE = os.getenv("ZOHO_ACCOUNTS_BASE", "https://accounts.zoho.com.au")
API_BASE = os.getenv("ZOHO_API_BASE", "https://www.zohoapis.com.au")

# Use v2 by default (works for /crm/v2/Deals etc). Set to v8 if you prefer.
CRM_VERSION = os.getenv("ZOHO_CRM_VERSION", "v2")

# OAuth app creds (required to refresh)
CLIENT_ID = os.environ["ZOHO_CLIENT_ID"]
CLIENT_SECRET = os.environ["ZOHO_CLIENT_SECRET"]
REFRESH_TOKEN = os.environ["ZOHO_REFRESH_TOKEN"]

# Start with an access token you already have (optional). If missing, we’ll refresh immediately.
ACCESS_TOKEN = os.getenv("ZOHO_ACCESS_TOKEN")

# Token cache
_access_token: Optional[str] = ACCESS_TOKEN
_access_token_expiry: float = 0.0  # unknown; we’ll refresh on 401


def refresh_access_token() -> str:
    url = f"{ACCOUNTS_BASE}/oauth/v2/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": REFRESH_TOKEN,
    }
    r = requests.post(url, data=data, timeout=30)
    payload = r.json() if r.content else {}
    if r.status_code >= 400 or "error" in payload:
        raise RuntimeError(f"Zoho refresh failed ({r.status_code}): {payload}")
    return payload["access_token"]


def crm_get(
    path: str, token: str, params: Optional[Dict[str, Any]] = None
) -> requests.Response:
    url = f"{API_BASE}{path}"
    return requests.get(
        url,
        headers={"Authorization": f"Zoho-oauthtoken {token}"},
        params=params or {},
        timeout=30,
    )


def get_json_with_auto_refresh(
    path: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Tries current token; if 401, refresh token once and retry.
    """
    global _access_token

    if not _access_token:
        _access_token = refresh_access_token()

    r = crm_get(path, _access_token, params=params)
    if r.status_code == 401:
        _access_token = refresh_access_token()
        r = crm_get(path, _access_token, params=params)

    payload = r.json() if r.content else {}
    if r.status_code >= 400:
        raise RuntimeError(f"Zoho GET failed ({r.status_code}): {payload}")

    return payload


def fetch_all_records(
    module: str, per_page: int = 200, fields: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Generic paginator for Zoho CRM list endpoints.
    module: "Deals" | "Leads" | "Accounts" | "Contacts"
    """
    page = 1
    all_rows: List[Dict[str, Any]] = []

    while True:
        params: Dict[str, Any] = {"page": page, "per_page": per_page}
        if fields:
            params["fields"] = ",".join(fields)

        resp = get_json_with_auto_refresh(f"/crm/{CRM_VERSION}/{module}", params=params)
        rows = resp.get("data", []) or []
        all_rows.extend(rows)

        info = resp.get("info", {}) or {}
        if not info.get("more_records"):
            break
        page += 1

    return all_rows


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flattens nested dicts using json_normalize.
    - Owner -> Owner__name, Owner__id, Owner__email
    - Account_Name -> Account_Name__name, Account_Name__id
    - etc.
    """
    if not records:
        return pd.DataFrame()

    df = pd.json_normalize(records, sep="__")

    # Optional: make column order a bit nicer (id first if present)
    cols = list(df.columns)
    if "id" in cols:
        cols = ["id"] + [c for c in cols if c != "id"]
        df = df[cols]

    return df


def export_module_to_csv(
    module: str, out_path: str, per_page: int = 200, fields: Optional[List[str]] = None
) -> pd.DataFrame:
    records = fetch_all_records(module, per_page=per_page, fields=fields)
    df = records_to_dataframe(records)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return df


if __name__ == "__main__":
    # Example: export Deals
    # df_deals = export_module_to_csv("Deals", "zoho_deals.csv", per_page=200)
    # print(
    #     f"Wrote zoho_deals.csv with {len(df_deals)} rows and {len(df_deals.columns)} columns."
    # )
    # print("Sample columns:", list(df_deals.columns)[:30])

    # Uncomment for other modules:
    # export_module_to_csv("Leads", "zoho_leads.csv")
    export_module_to_csv("Accounts", "zoho_accounts.csv")
    export_module_to_csv("Contacts", "zoho_contacts.csv")
