import os

import requests

ACCOUNTS_DOMAIN = "https://accounts.zoho.com.au"


def exchange_code_for_tokens(code: str) -> dict:
    url = f"{ACCOUNTS_DOMAIN}/oauth/v2/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": os.environ["ZOHO_CLIENT_ID"],
        "client_secret": os.environ["ZOHO_CLIENT_SECRET"],
        "code": code,
    }
    r = requests.post(url, data=data, timeout=30)
    # Zoho returns JSON even on errors; show it clearly:
    try:
        payload = r.json()
    except Exception:
        r.raise_for_status()
        raise

    if r.status_code >= 400 or "error" in payload:
        raise RuntimeError(f"Token exchange failed: {payload}")

    return payload


if __name__ == "__main__":
    # Put the auth code in env so you don’t hardcode secrets
    code = os.environ["ZOHO_AUTH_CODE"]

    tokens = exchange_code_for_tokens(code)
    print("✅ Token exchange success.")
    print("access_token:", tokens.get("access_token"))
    print("refresh_token:", tokens.get("refresh_token"))
    print("expires_in:", tokens.get("expires_in"))
    print("\nSAVE the refresh_token securely (env/KeyVault).")
