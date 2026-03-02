import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone

API_URL = "https://dh2.efir-net.ru/v2"

API_LOGIN = os.getenv("API_LOGIN", "accentam-api1")
API_PASS  = os.getenv("API_PASS",  "653Bsw")

DATE_FROM = os.getenv("DATE_FROM", "2025-01-01T00:00:00Z")
DATE_TO   = os.getenv("DATE_TO") or datetime.now(timezone.utc).strftime("%Y-%m-%dT23:59:59Z")

# Инструменты: где нужно, используем MOEX SECID вместо ISIN
INSTRUMENTS = [
    "RU000A100WZ5",  # Акцент IV (ISIN)
    "XACCSK",        # Акцент 5 (SECID)
]

NAME_MAP = {
    "RU000A100WZ5": "АКЦЕНТ IV",
    "RU000A10DQF7": "Акцент 5",
}

SECID_TO_ISIN = {
    "XACCSK": "RU000A10DQF7",
}


def do_post(url: str, body: dict, token: str | None) -> list[dict] | dict | None:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.post(url, json=body, headers=headers, timeout=60)
    if r.status_code == 200:
        return r.json()
    raise RuntimeError(f"HTTP {r.status_code}: {r.text}")


def get_token(login: str, password: str) -> str:
    data = do_post(f"{API_URL}/Account/Login", {"login": login, "password": password}, token=None)
    token = (data or {}).get("token")
    if not token:
        raise RuntimeError("Не удалось получить токен (проверьте API_LOGIN/API_PASS).")
    return token


def fetch_history(token: str, instruments: list[str], date_from: str, date_to: str, page_size: int = 100) -> list[dict]:
    url = f"{API_URL}/Moex/History"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    out: list[dict] = []
    page = 0
    while True:
        body = {
            "engine": "stock",
            "market": "shares",
            "boardid": ["TQIF"],
            "instruments": instruments,
            "dateFrom": date_from,
            "dateTo": date_to,
            "tradingSessions": [],
            "pageNum": page,
            "pageSize": page_size,
        }
        r = requests.post(url, json=body, headers=headers, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"API error page={page}: {r.status_code} {r.text}")

        data = r.json()
        if not data:
            break

        out.extend(data)
        page += 1
        if len(data) < page_size:
            break

    return out


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def sum_or_single(s: pd.Series, decimals: int = 0) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return np.nan
    xr = x.round(decimals)
    if xr.nunique() == 1:
        return float(x.iloc[0])
    return float(x.sum())


def build_daily_ohlcv(rows: list[dict]) -> pd.DataFrame:
    raw = pd.DataFrame(rows)

    need = [
        "shortname", "secid", "isin", "tradedate",
        "open", "high", "low", "close", "waprice",
        "volume", "value", "numtrades",
    ]
    for c in need:
        if c not in raw.columns:
            raw[c] = np.nan

    df = raw[need].copy()

    # восстановление isin по secid (для Акцент 5)
    df["isin"] = df["isin"].fillna(df["secid"].map(SECID_TO_ISIN))
    df["isin"] = df["isin"].replace(SECID_TO_ISIN)

    # типы
    for c in ["open", "high", "low", "close", "waprice", "volume", "value", "numtrades"]:
        df[c] = to_num(df[c])

    df["tradedate"] = pd.to_datetime(df["tradedate"], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=["tradedate", "secid"]).copy()

    # схлопываем возможные дубли дневных строк
    df = (
        df.sort_values(["secid", "tradedate"])
          .groupby(["secid", "isin", "tradedate"], as_index=False)
          .agg(
              shortname=("shortname", "last"),
              open=("open", "first"),
              high=("high", "max"),
              low=("low", "min"),
              close=("close", "last"),
              waprice=("waprice", "last"),
              volume=("volume", lambda s: sum_or_single(s, 0)),
              value=("value", lambda s: sum_or_single(s, 0)),
              numtrades=("numtrades", lambda s: sum_or_single(s, 0)),
          )
    )

    # имена фондов
    df["fund"] = df["isin"].map(NAME_MAP).fillna(df["shortname"].astype(str))

    # рублевые метрики
    df["rub_value_waprice"] = df["waprice"] * df["volume"]
    df["rub_value_close"] = df["close"] * df["volume"]

    # финальныи порядок колонок
    df = df[[
        "tradedate", "fund", "isin", "secid",
        "volume", "open", "high", "low", "close", "waprice",
        "rub_value_waprice", "rub_value_close",
        "numtrades", "value",
    ]].sort_values(["tradedate", "fund"])

    return df


def main():
    token = get_token(API_LOGIN, API_PASS)
    rows = fetch_history(token, INSTRUMENTS, DATE_FROM, DATE_TO)
    if not rows:
        print("Нет данных за период.")
        return

    df = build_daily_ohlcv(rows)

    # вывод в консоль (первые строки)
    with pd.option_context("display.max_rows", 30, "display.max_columns", 30, "display.width", 140):
        print(df.head(20))

    out_path = "accent_ohlcv_daily.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved: {out_path} | rows={len(df)} | period={df['tradedate'].min()}..{df['tradedate'].max()}")


if __name__ == "__main__":
    main()
