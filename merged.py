#!/usr/bin/env python
# coding: utf-8

import os
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
from typing import Any, Dict, List, Optional

import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import openpyxl  # noqa: F401
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="Торги ЗПИФ", layout="wide")

# -----------------------
# Конфигурация
# -----------------------
API_URL = "https://dh2.efir-net.ru/v2"

ACCENT_IV_ISIN = "RU000A100WZ5"
ACCENT_5_ISIN = "RU000A10DQF7"

FUND_MAP = {
    # Парус
    "RU000A105328": "ПАРУС-ЛОГ",
    "RU000A1068X9": "ПАРУС-ДВН",
    "RU000A108UH0": "ПАРУС-КРАС",
    "RU000A108VR7": "ПАРУС-МАКС",
    "RU000A104KU3": "ПАРУС-НОРД",
    "RU000A1022Z1": "ПАРУС-ОЗН",
    "RU000A104172": "ПАРУС-СБЛ",
    "RU000A108BZ2": "ПАРУС-ТРМ",
    "RU000A10CFM8": "ПАРУС-ЗОЛЯ",

    # Акцент
    ACCENT_IV_ISIN: "АКЦЕНТ IV",
    ACCENT_5_ISIN: "Акцент 5",

    # УК Первая
    "RU000A102AH5": "ЗПИФСовр 6",
    "RU000A0ZYC64": "СоврАрБизн 2",

    # Велес
    "RU000A0JWCE7": "Перловский",
    "RU000A0JRHC0": "Атриум",

    # Самолет
    "RU000A10A117": "ЗПИФ СМЛТ",

    # СФН
    "RU000A1034U7": "ЗПИФСовр 7",
    "RU000A0JWAW3": "СоврАрБизн",
    "RU000A0ZZ5R2": "СоврАрБизн 3",
    "RU000A104YX8": "СоврКоммНедвиж",
    "RU000A1099U0": "ЗПИФСовр 9",

    # ВИМ-Инвестиции
    "RU000A102N77": "РД",
    "RU000A103B62": "РД ПРО",
    "RU000A103HD7": "РД 2",

    # Рентал-Про
    "RU000A108157": "Рентал-Про",

    # Активо
    "RU000A10CLY1": "АКТИВО ДВАДЦАТЬ ОДИН",
    "RU000A1092L4": "АКТИВО ДВАДЦАТЬ",
    "RU000A10ATA8": "АКТИВО ФЛИППИНГ",
    "RU000A101YY2": "Арендный поток-2",
}

ISIN_TO_MOEX_CODE = {
    ACCENT_5_ISIN: "XACCSK",   # Акцент 5
    "RU000A10A117": "XHOUSE",  # Самолет
    "RU000A108BZ2": "XTRIUMF", # Триумф
}
MOEX_CODE_TO_ISIN = {secid: isin for isin, secid in ISIN_TO_MOEX_CODE.items()}

TARGET_ISINS = list(FUND_MAP.keys())
ZPIF_SECIDS = [ISIN_TO_MOEX_CODE.get(isin, isin) for isin in TARGET_ISINS]

ACCENT_TARGET_ISINS = [ACCENT_IV_ISIN, ACCENT_5_ISIN]
ACCENT_INSTRUMENTS_FOR_API = [ACCENT_IV_ISIN, ISIN_TO_MOEX_CODE[ACCENT_5_ISIN]]

FUND_NAME_BY_ISIN = {
    ACCENT_IV_ISIN: "АКЦЕНТ IV",
    ACCENT_5_ISIN: "Акцент 5",
}

QUAL_BY_ISIN = {
    "RU000A105328": "квал",
    "RU000A1068X9": "квал",
    "RU000A108UH0": "квал",
    "RU000A108VR7": "квал",
    "RU000A104KU3": "квал",
    "RU000A1022Z1": "квал",
    "RU000A104172": "квал",
    "RU000A108BZ2": "квал",
    "RU000A10CFM8": "квал",
    ACCENT_IV_ISIN: "квал",
    ACCENT_5_ISIN: "неквал",
    "RU000A10A117": "неквал",
    "RU000A1099U0": "неквал",
    "RU000A1034U7": "неквал",
    "RU000A0JWAW3": "неквал",
    "RU000A102N77": "неквал",
    "RU000A103B62": "квал",
    "RU000A108157": "квал",
    "RU000A10CLY1": "квал",
    "RU000A1092L4": "неквал",
    "RU000A10ATA8": "неквал",
    "RU000A102AH5": "неквал",
    "RU000A0ZYC64": "неквал",
    "RU000A0JWCE7": "неквал",
    "RU000A0JRHC0": "неквал",
    "RU000A0ZZ5R2": "квал",
    "RU000A104YX8": "квал",
    "RU000A103HD7": "неквал",
    "RU000A101YY2": "неквал",
}

RU_MONTHS = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
    5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
    9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь",
}

SELECT_KEY = "fund_select"
AVAILABLE_FUNDS_KEY = "_available_funds"
GROUPS_QUAL_KEY = "_groups_qual"

GROUPS = {
    "Акцент": [name for _, name in FUND_MAP.items()
               if str(name).startswith("Акцент") or str(name).upper().startswith("АКЦЕНТ")],
    "Парус": [name for _, name in FUND_MAP.items() if str(name).startswith("ПАРУС-")],
    "СФН": [
        FUND_MAP.get("RU000A1099U0", "ЗПИФСовр 9"),
        FUND_MAP.get("RU000A1034U7", "ЗПИФСовр 7"),
        FUND_MAP.get("RU000A0JWAW3", "СоврАрБизн"),
        FUND_MAP.get("RU000A102AH5", "ЗПИФСовр 6"),
        FUND_MAP.get("RU000A0ZYC64", "СоврАрБизн 2"),
        FUND_MAP.get("RU000A0ZZ5R2", "СоврАрБизн 3"),
        FUND_MAP.get("RU000A104YX8", "СоврКоммНедвиж"),
    ],
    "Самолет": [FUND_MAP.get("RU000A10A117", "ЗПИФ СМЛТ")],
    "Велес": [
        FUND_MAP.get("RU000A0JWCE7", "Перловский"),
        FUND_MAP.get("RU000A0JRHC0", "Атриум"),
    ],
    "ВИМ-Инвестиции": [
        FUND_MAP.get("RU000A102N77", "РД"),
        FUND_MAP.get("RU000A103HD7", "РД 2"),
        FUND_MAP.get("RU000A103B62", "РД ПРО"),
    ],
    "АБ-Капитал (Рентал Про)": [FUND_MAP.get("RU000A108157", "Рентал-Про")],
    "Активо": [
        FUND_MAP.get("RU000A10CLY1", "АКТИВО ДВАДЦАТЬ ОДИН"),
        FUND_MAP.get("RU000A1092L4", "АКТИВО ДВАДЦАТЬ"),
        FUND_MAP.get("RU000A101YY2", "Арендный поток-2"),
        FUND_MAP.get("RU000A10ATA8", "АКТИВО ФЛИППИНГ"),
    ],
}

# -----------------------
# Secrets / Env
# -----------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            v = st.secrets.get(name)
            if v is not None and str(v).strip() != "":
                return str(v)
    except Exception:
        pass

    v = os.getenv(name)
    if v is not None and str(v).strip() != "":
        return str(v)

    return default

API_LOGIN = get_secret("API_LOGIN", "")
API_PASS = get_secret("API_PASS", "")

# -----------------------
# Базовые утилиты
# -----------------------
def _sum_or_single(s: pd.Series, decimals: int = 0) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return np.nan

    xr = x.round(decimals)
    if xr.nunique() == 1:
        return float(x.iloc[0])

    return float(x.sum())

def _first_non_na(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(x.iloc[0]) if not x.empty else np.nan

def _last_non_na(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(x.iloc[-1]) if not x.empty else np.nan

def _vwap(waprice: pd.Series, volume: pd.Series) -> float:
    wp = pd.to_numeric(waprice, errors="coerce")
    vv = pd.to_numeric(volume, errors="coerce")
    m = wp.notna() & vv.notna() & (vv > 0)

    if not m.any():
        return np.nan

    num = (wp[m] * vv[m]).sum()
    den = vv[m].sum()
    return float(num / den) if den != 0 else np.nan

def chunk_list(lst: List[str], size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def _to_api_dt(d: date, end_of_day: bool = False) -> str:
    if end_of_day:
        return f"{d:%Y-%m-%d}T23:59:59Z"
    return f"{d:%Y-%m-%d}T00:00:00Z"

def _to_date(d: str) -> date:
    return pd.to_datetime(d, utc=True).date()

def save_snapshot_csv(df: pd.DataFrame) -> None:
    try:
        out_dir = Path("snapshots")
        out_dir.mkdir(parents=True, exist_ok=True)
        snap_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_path = out_dir / f"zpif_history_{snap_date}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
    except Exception:
        pass

def df_to_xlsx_bytes(df: pd.DataFrame, sheet_name: str) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        ws.freeze_panes = "A2"
        for col_cells in ws.columns:
            max_len = 0
            col_letter = col_cells[0].column_letter
            for cell in col_cells:
                v = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(v))
            ws.column_dimensions[col_letter].width = min(max_len + 2, 60)
    buf.seek(0)
    return buf.read()

# -----------------------
# API helpers
# -----------------------
def do_post_request(url: str, body: Dict[str, Any], token: Optional[str]) -> Any:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = requests.post(url, json=body, headers=headers, timeout=60)
    if r.status_code == 200:
        return r.json()

    raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

def get_token(login: str, password: str) -> str:
    data = do_post_request(
        f"{API_URL}/Account/Login",
        {"login": login, "password": password},
        token=None,
    )
    token = data.get("token") if isinstance(data, dict) else None
    if not token:
        raise RuntimeError("Не удалось получить token из ответа /Account/Login")
    return str(token)

def fetch_all_trading_results(
    token: str,
    instruments: List[str],
    date_from: str,
    date_to: str,
    page_size: int = 100,
) -> List[Dict[str, Any]]:
    url = f"{API_URL}/Moex/History"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    all_data: List[Dict[str, Any]] = []
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
            raise RuntimeError(f"API error on page {page}: {r.status_code} {r.text}")

        page_data = r.json()
        if not page_data:
            break

        all_data.extend(page_data)
        page += 1

        if len(page_data) < page_size:
            break

    return all_data

def fetch_accent_history(
    token: str,
    instruments: List[str],
    date_from: str,
    date_to: str,
    page_size: int = 100,
) -> List[Dict[str, Any]]:
    url = f"{API_URL}/Moex/History"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    out: List[Dict[str, Any]] = []

    requests_plan = [
        {"market": "shares", "boardid": ["TQIF"], "mode_label": "Основной режим"},
        {"market": "ndm", "boardid": None, "mode_label": "РПС"},
    ]

    for cfg in requests_plan:
        page = 0
        while True:
            body = {
                "engine": "stock",
                "market": cfg["market"],
                "instruments": instruments,
                "dateFrom": date_from,
                "dateTo": date_to,
                "tradingSessions": [],
                "pageNum": page,
                "pageSize": page_size,
            }
            if cfg["boardid"] is not None:
                body["boardid"] = cfg["boardid"]

            r = requests.post(url, json=body, headers=headers, timeout=60)
            if r.status_code != 200:
                raise RuntimeError(
                    f"API error for market={cfg['market']} page={page}: "
                    f"{r.status_code} {r.text}"
                )

            data = r.json()
            if not data:
                break

            for row in data:
                row["_request_market"] = cfg["market"]
                row["_mode_label"] = cfg["mode_label"]

            out.extend(data)

            if len(data) < page_size:
                break

            page += 1

    return out

def _fetch_range_safe(
    token: str,
    instruments: List[str],
    start_d: date,
    end_d: date,
    page_size: int = 100,
    min_days: int = 31,
) -> List[Dict[str, Any]]:
    try:
        return fetch_all_trading_results(
            token=token,
            instruments=instruments,
            date_from=_to_api_dt(start_d, end_of_day=False),
            date_to=_to_api_dt(end_d, end_of_day=True),
            page_size=page_size,
        )
    except Exception as e:
        if (end_d - start_d).days <= min_days:
            raise RuntimeError(f"Failed on small range {start_d}..{end_d}: {e}")

        mid = start_d + timedelta(days=((end_d - start_d).days // 2))
        left = _fetch_range_safe(token, instruments, start_d, mid, page_size=page_size, min_days=min_days)
        right = _fetch_range_safe(token, instruments, mid + timedelta(days=1), end_d, page_size=page_size, min_days=min_days)
        return left + right

# -----------------------
# Кешируемые загрузчики
# -----------------------
@st.cache_data(ttl=24 * 60 * 60)
def load_df(secids: List[str], date_from: str, date_to: str) -> pd.DataFrame:
    token = get_token(API_LOGIN, API_PASS)

    all_results: List[Dict[str, Any]] = []
    for chunk in chunk_list(secids, 100):
        all_results.extend(fetch_all_trading_results(token, chunk, date_from, date_to))

    if not all_results:
        return pd.DataFrame(columns=["shortname", "fund", "isin", "volume", "value", "numtrades", "open", "close", "waprice", "tradedate"])

    raw = pd.DataFrame(all_results)

    need_cols = ["shortname", "secid", "isin", "volume", "value", "numtrades", "open", "close", "waprice", "tradedate"]
    for c in need_cols:
        if c not in raw.columns:
            raw[c] = np.nan

    df = raw[need_cols].copy()
    df["isin"] = df["isin"].fillna(df["secid"].map(MOEX_CODE_TO_ISIN))
    df["isin"] = df["isin"].replace(MOEX_CODE_TO_ISIN)

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["numtrades"] = pd.to_numeric(df["numtrades"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["waprice"] = pd.to_numeric(df["waprice"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")

    df["tradedate"] = pd.to_datetime(df["tradedate"], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=["isin", "tradedate"])

    df = (
        df.sort_values(["isin", "tradedate", "secid"])
          .groupby(["isin", "tradedate"], as_index=False)
          .agg(
              shortname=("shortname", "last"),
              secid=("secid", "last"),
              open=("open", "last"),
              close=("close", "last"),
              waprice=("waprice", "last"),
              volume=("volume", lambda s: _sum_or_single(s, 0)),
              value=("value", lambda s: _sum_or_single(s, 0)),
              numtrades=("numtrades", lambda s: _sum_or_single(s, 0)),
          )
    )

    df["fund"] = df["isin"].map(FUND_MAP).fillna(df["shortname"].astype(str))
    return df

@st.cache_data(ttl=24 * 60 * 60)
def load_df_long_history(
    secids: List[str],
    date_from: str,
    date_to: str,
    chunk_size: int = 30,
    page_size: int = 100,
) -> pd.DataFrame:
    token = get_token(API_LOGIN, API_PASS)

    start_d = _to_date(date_from)
    end_d = _to_date(date_to)

    all_results: List[Dict[str, Any]] = []
    for sec_chunk in chunk_list(secids, chunk_size):
        all_results.extend(
            _fetch_range_safe(
                token=token,
                instruments=list(sec_chunk),
                start_d=start_d,
                end_d=end_d,
                page_size=page_size,
                min_days=31,
            )
        )

    if not all_results:
        return pd.DataFrame(columns=["shortname", "fund", "isin", "volume", "value", "numtrades", "open", "close", "waprice", "tradedate"])

    raw = pd.DataFrame(all_results)
    need_cols = ["shortname", "secid", "isin", "volume", "value", "numtrades", "open", "close", "waprice", "tradedate"]
    for c in need_cols:
        if c not in raw.columns:
            raw[c] = np.nan

    df = raw[need_cols].copy()
    df["isin"] = df["isin"].fillna(df["secid"].map(MOEX_CODE_TO_ISIN))
    df["isin"] = df["isin"].replace(MOEX_CODE_TO_ISIN)

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["numtrades"] = pd.to_numeric(df["numtrades"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["waprice"] = pd.to_numeric(df["waprice"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")

    df["tradedate"] = pd.to_datetime(df["tradedate"], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=["isin", "tradedate"])

    df = (
        df.sort_values(["isin", "tradedate", "secid"])
          .groupby(["isin", "tradedate"], as_index=False)
          .agg(
              shortname=("shortname", "last"),
              secid=("secid", "last"),
              open=("open", "last"),
              close=("close", "last"),
              waprice=("waprice", "last"),
              volume=("volume", lambda s: _sum_or_single(s, 0)),
              value=("value", lambda s: _sum_or_single(s, 0)),
              numtrades=("numtrades", lambda s: _sum_or_single(s, 0)),
          )
    )

    df["fund"] = df["isin"].map(FUND_MAP).fillna(df["shortname"].astype(str))
    return df

@st.cache_data(ttl=24 * 60 * 60)
def load_accent_raw(d_from: date, d_to: date) -> pd.DataFrame:
    token = get_token(API_LOGIN, API_PASS)

    all_rows = fetch_accent_history(
        token=token,
        instruments=ACCENT_INSTRUMENTS_FOR_API,
        date_from=_to_api_dt(d_from, end_of_day=False),
        date_to=_to_api_dt(d_to, end_of_day=True),
        page_size=100,
    )

    if not all_rows:
        return pd.DataFrame()

    raw = pd.DataFrame(all_rows).drop_duplicates().copy()

    need = [
        "shortname",
        "secid",
        "isin",
        "tradedate",
        "open",
        "high",
        "low",
        "close",
        "waprice",
        "volume",
        "value",
        "numtrades",
        "boardid",
        "_request_market",
        "_mode_label",
    ]
    for c in need:
        if c not in raw.columns:
            raw[c] = np.nan

    raw["tradedate"] = pd.to_datetime(raw["tradedate"], errors="coerce", utc=True).dt.date
    for c in ["open", "high", "low", "close", "waprice", "volume", "value", "numtrades"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw["isin"] = raw["isin"].replace(MOEX_CODE_TO_ISIN)
    raw["isin"] = raw["isin"].fillna(raw["secid"].replace(MOEX_CODE_TO_ISIN))
    raw = raw.dropna(subset=["isin", "tradedate"])
    raw = raw[raw["isin"].isin(ACCENT_TARGET_ISINS)].copy()
    raw["fund"] = raw["isin"].map(FUND_NAME_BY_ISIN).fillna(raw["shortname"].astype(str))

    def mark_mode(row: pd.Series) -> str:
        req_market = row.get("_request_market")
        if req_market == "shares":
            return "Основной режим"
        if req_market == "ndm":
            return "РПС"

        board = row.get("boardid")
        if board == "TQIF":
            return "Основной режим"
        return f"Прочие ({board})"

    raw["mode"] = raw.apply(mark_mode, axis=1)

    raw = raw.drop_duplicates(
        subset=[
            "tradedate",
            "isin",
            "secid",
            "boardid",
            "mode",
            "open",
            "high",
            "low",
            "close",
            "waprice",
            "volume",
            "value",
            "numtrades",
        ]
    ).copy()

    return raw

# -----------------------
# Функции для вкладки Ф4/Ф5
# -----------------------
def calc_period_window(end_d: date, period_kind: str) -> tuple[date, date]:
    if period_kind == "Неделя":
        return end_d - timedelta(days=6), end_d
    if period_kind == "Месяц":
        return (end_d - relativedelta(months=1)) + timedelta(days=1), end_d
    if period_kind == "Квартал":
        return (end_d - relativedelta(months=3)) + timedelta(days=1), end_d
    if period_kind == "Год":
        return (end_d - relativedelta(years=1)) + timedelta(days=1), end_d
    raise ValueError(f"Неизвестный период: {period_kind}")

def shift_date_by_period(cur: date, period_kind: str, forward: bool) -> date:
    sign = 1 if forward else -1
    if period_kind == "Неделя":
        return cur + timedelta(days=7 * sign)
    if period_kind == "Месяц":
        return cur + relativedelta(months=sign)
    if period_kind == "Квартал":
        return cur + relativedelta(months=3 * sign)
    if period_kind == "Год":
        return cur + relativedelta(years=sign)
    return cur

def build_accent_daily_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    d = df_raw.copy().drop_duplicates()
    d = d.sort_values(["isin", "tradedate", "mode", "secid", "boardid"], na_position="last")

    def _agg(g: pd.DataFrame) -> pd.Series:
        vol = _sum_or_single(g["volume"], decimals=0)
        val_api = _sum_or_single(g["value"], decimals=0)
        trades = _sum_or_single(g["numtrades"], decimals=0)

        open_ = _first_non_na(g["open"])
        close_ = _last_non_na(g["close"])
        high_ = pd.to_numeric(g["high"], errors="coerce").max(skipna=True)
        low_ = pd.to_numeric(g["low"], errors="coerce").min(skipna=True)

        wap = _vwap(g["waprice"], g["volume"])
        if pd.isna(wap):
            wap = _last_non_na(g["waprice"])

        rub_wap = float(vol * wap) if pd.notna(vol) and pd.notna(wap) else np.nan
        rub_close = float(vol * close_) if pd.notna(vol) and pd.notna(close_) else np.nan

        return pd.Series({
            "Кол-во бумаг, шт": vol,
            "Open": open_,
            "High": high_,
            "Low": low_,
            "Close": close_,
            "Средняя цена (waprice)": wap,
            "Рубли (volume*waprice)": rub_wap,
            "Рубли (close*volume)": rub_close,
            "Рубли как в API (value)": val_api,
            "Сделок, шт": trades,
        })

    out = (
        d.groupby(["tradedate", "isin", "fund", "mode"], as_index=False)
         .apply(_agg, include_groups=False)
         .rename(columns={
             "tradedate": "Дата",
             "isin": "ISIN",
             "fund": "Фонд",
             "mode": "Режим торгов",
         })
         .sort_values(["Дата", "Фонд", "Режим торгов"])
         .reset_index(drop=True)
    )

    out = out.drop(columns=["index", "level_0", "level_1"], errors="ignore")

    for c in ["Open", "High", "Low", "Close", "Средняя цена (waprice)"]:
        out[c] = out[c].round(2)
    for c in ["Рубли (volume*waprice)", "Рубли (close*volume)", "Рубли как в API (value)"]:
        out[c] = out[c].round(0)

    return out

def build_range_summary(df_raw: pd.DataFrame, start_d: date, end_d: date) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    d = df_raw.copy()
    d = d.dropna(subset=["fund", "mode", "value", "volume", "numtrades", "tradedate"]).copy()

    grp = (
        d.groupby(["fund", "mode"], as_index=False)
         .agg(
             volume=("volume", "sum"),
             value=("value", "sum"),
             numtrades=("numtrades", "sum"),
         )
         .rename(columns={
             "fund": "Фонд",
             "mode": "Режим торгов",
             "volume": "Кол-во бумаг, шт",
             "value": "Оборот, руб",
             "numtrades": "Сделок, шт",
         })
    )

    grp.insert(0, "Начало периода", start_d)
    grp.insert(1, "Конец периода", end_d)
    return grp

def pivot_period_summary(period_df: pd.DataFrame) -> pd.DataFrame:
    if period_df is None or period_df.empty:
        return pd.DataFrame()

    metrics = ["Кол-во бумаг, шт", "Оборот, руб", "Сделок, шт"]

    pv = period_df.pivot_table(
        index=["Начало периода", "Конец периода", "Фонд"],
        columns="Режим торгов",
        values=metrics,
        aggfunc="sum",
        fill_value=0,
    )

    pv.columns = [f"{m} — {mode}" for m, mode in pv.columns]
    pv = pv.reset_index()

    for m in metrics:
        col_main = f"{m} — Основной режим"
        col_rps = f"{m} — РПС"
        if col_main not in pv.columns:
            pv[col_main] = 0
        if col_rps not in pv.columns:
            pv[col_rps] = 0
        pv[f"{m} — Итого"] = pv[col_main] + pv[col_rps]

    ordered = ["Начало периода", "Конец периода", "Фонд"]
    for m in metrics:
        ordered += [f"{m} — Основной режим", f"{m} — РПС", f"{m} — Итого"]

    ordered = [c for c in ordered if c in pv.columns]
    pv = pv[ordered].sort_values(["Начало периода", "Фонд"], ascending=[False, True]).reset_index(drop=True)
    return pv

def _fmt_ru_1(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")

def _month_bucket(dt: pd.Series) -> pd.Series:
    return dt.dt.to_period("M").dt.start_time.dt.normalize()

def _month_label(ts: pd.Timestamp) -> str:
    d = ts.date()
    return f"{RU_MONTHS[d.month]} {d.year}"

def build_turnover_stacked_chart(df_raw: pd.DataFrame, value_mode: str = "Основной режим") -> go.Figure:
    fig = go.Figure()
    if df_raw is None or df_raw.empty:
        return fig

    d = df_raw.copy()
    d = d[d["isin"].isin([ACCENT_IV_ISIN, ACCENT_5_ISIN])].copy()
    if d.empty:
        return fig

    if "mode" in d.columns:
        if value_mode == "Основной режим":
            d = d[d["mode"] == "Основной режим"].copy()
        elif value_mode == "РПС":
            d = d[d["mode"] == "РПС"].copy()
        else:
            d = d[d["mode"].isin(["Основной режим", "РПС"])].copy()

    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["tradedate", "value"]).copy()
    dt = pd.to_datetime(d["tradedate"], errors="coerce")
    d = d.loc[dt.notna()].copy()
    dt = pd.to_datetime(d["tradedate"], errors="coerce")
    d["bucket"] = _month_bucket(dt)

    label_map = {ACCENT_5_ISIN: "ЗПИФ 5", ACCENT_IV_ISIN: "ЗПИФ 4"}
    d["fund_label"] = d["isin"].map(label_map).fillna(d.get("fund", d["isin"].astype(str)))

    g = d.groupby(["bucket", "fund_label"], as_index=False)["value"].sum()
    g["mln"] = g["value"] / 1e6

    pv = (
        g.pivot_table(index="bucket", columns="fund_label", values="mln", aggfunc="sum", fill_value=0)
         .reset_index()
         .sort_values("bucket")
    )

    if "ЗПИФ 5" not in pv.columns:
        pv["ЗПИФ 5"] = 0.0
    if "ЗПИФ 4" not in pv.columns:
        pv["ЗПИФ 4"] = 0.0

    pv["total"] = pv["ЗПИФ 5"] + pv["ЗПИФ 4"]

    x_labels = [_month_label(pd.Timestamp(x)) for x in pv["bucket"]]
    y5 = pv["ЗПИФ 5"].astype(float).tolist()
    y4 = pv["ЗПИФ 4"].astype(float).tolist()
    yt = pv["total"].astype(float).tolist()

    fig.add_bar(
        name="ЗПИФ 5",
        x=x_labels,
        y=y5,
        marker_color="#7A1F1F",
        text=[_fmt_ru_1(v) if v > 0 else "" for v in y5],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white"),
    )
    fig.add_bar(
        name="ЗПИФ 4",
        x=x_labels,
        y=y4,
        marker_color="#BFBFBF",
        text=[_fmt_ru_1(v) if v > 0 else "" for v in y4],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="black"),
    )
    fig.add_scatter(
        x=x_labels,
        y=yt,
        mode="text",
        text=[_fmt_ru_1(v) if v > 0 else "" for v in yt],
        textposition="top center",
        showlegend=False,
        textfont=dict(color="black"),
    )

    suffix = {
        "Основной режим": " (основной режим)",
        "РПС": " (РПС)",
    }.get(value_mode, " (итого: основной + РПС)")

    fig.update_layout(
        barmode="stack",
        title=f"Оборот на Мосбирже, млн руб.{suffix}",
        xaxis_title=None,
        yaxis_title=None,
        legend_title_text=None,
        separators=". ",
        margin=dict(l=30, r=30, t=60, b=30),
    )

    max_y = max(yt) if yt else 0.0
    fig.update_yaxes(range=[0, max_y * 1.25 if max_y > 0 else 1])
    return fig

# -----------------------
# Функции для общих фильтров
# -----------------------
def _add_group(group_name: str):
    available_funds = st.session_state.get(AVAILABLE_FUNDS_KEY, [])
    gf = [f for f in GROUPS.get(group_name, []) if f in available_funds]
    st.session_state[SELECT_KEY] = sorted(set(st.session_state.get(SELECT_KEY, [])).union(gf))

def _remove_group(group_name: str):
    available_funds = st.session_state.get(AVAILABLE_FUNDS_KEY, [])
    gf = [f for f in GROUPS.get(group_name, []) if f in available_funds]
    st.session_state[SELECT_KEY] = sorted(set(st.session_state.get(SELECT_KEY, [])) - set(gf))

def _add_qual(qual_key: str):
    available_funds = st.session_state.get(AVAILABLE_FUNDS_KEY, [])
    groups_qual = st.session_state.get(GROUPS_QUAL_KEY, {})
    gf = [f for f in groups_qual.get(qual_key, []) if f in available_funds]
    st.session_state[SELECT_KEY] = sorted(set(st.session_state.get(SELECT_KEY, [])).union(gf))

def _remove_qual(qual_key: str):
    available_funds = st.session_state.get(AVAILABLE_FUNDS_KEY, [])
    groups_qual = st.session_state.get(GROUPS_QUAL_KEY, {})
    gf = [f for f in groups_qual.get(qual_key, []) if f in available_funds]
    st.session_state[SELECT_KEY] = sorted(set(st.session_state.get(SELECT_KEY, [])) - set(gf))

def render_fund_selector(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["qual"] = df["isin"].map(QUAL_BY_ISIN).fillna("неизвестно")

    available_funds = sorted(df["fund"].dropna().unique().tolist())
    groups_qual = {
        "Квал": sorted(df.loc[df["qual"] == "квал", "fund"].dropna().unique().tolist()),
        "Неквал": sorted(df.loc[df["qual"] == "неквал", "fund"].dropna().unique().tolist()),
    }

    st.session_state[AVAILABLE_FUNDS_KEY] = available_funds
    st.session_state[GROUPS_QUAL_KEY] = groups_qual

    if SELECT_KEY not in st.session_state:
        st.session_state[SELECT_KEY] = available_funds[:]
    else:
        st.session_state[SELECT_KEY] = [
            f for f in st.session_state[SELECT_KEY]
            if f in available_funds
        ]

    with st.expander("Быстрый выбор: УК и квал/неквал", expanded=False):
        st.markdown("#### По УК")
        cols = st.columns(len(GROUPS))
        for i, gname in enumerate(GROUPS.keys()):
            with cols[i]:
                st.markdown(f"**{gname}**")
                st.button(
                    f"+ {gname}",
                    key=f"btn_add_{gname}",
                    width="stretch",
                    on_click=_add_group,
                    args=(gname,),
                )
                st.button(
                    f"− {gname}",
                    key=f"btn_rm_{gname}",
                    width="stretch",
                    on_click=_remove_group,
                    args=(gname,),
                )

        st.markdown("#### По квалификации")
        q1, q2 = st.columns(2)
        with q1:
            st.button("+ Квал", key="btn_add_qual", width="stretch", on_click=_add_qual, args=("Квал",))
            st.button("− Квал", key="btn_rm_qual", width="stretch", on_click=_remove_qual, args=("Квал",))
        with q2:
            st.button("+ Неквал", key="btn_add_nonqual", width="stretch", on_click=_add_qual, args=("Неквал",))
            st.button("− Неквал", key="btn_rm_nonqual", width="stretch", on_click=_remove_qual, args=("Неквал",))

        a, b = st.columns(2)
        with a:
            st.button(
                "Выбрать все",
                key="btn_all",
                width="stretch",
                on_click=lambda: st.session_state.__setitem__(SELECT_KEY, st.session_state.get(AVAILABLE_FUNDS_KEY, [])[:]),
            )
        with b:
            st.button(
                "Снять все",
                key="btn_none",
                width="stretch",
                on_click=lambda: st.session_state.__setitem__(SELECT_KEY, []),
            )

    st.multiselect(
        "Выберите фонды",
        options=available_funds,
        key=SELECT_KEY,
    )

    selected_funds = st.session_state.get(SELECT_KEY, [])
    df_sel = df[df["fund"].isin(selected_funds)].copy().sort_values(["tradedate", "fund"])
    df_sel["label"] = df_sel["fund"].astype(str) + " (" + df_sel["isin"].astype(str) + ")"
    return df_sel

# -----------------------
# Рендер вкладки "Доходность"
# -----------------------
def render_returns(df_sel: pd.DataFrame, date_to: str) -> None:
    st.subheader("Доходность (накопленная, %): 1 пай, купленный в разные периоды")

    available_dates = sorted(df_sel["tradedate"].dropna().unique().tolist())
    if not available_dates:
        st.warning("Нет дат для расчета доходности.")
        return

    end_date_ret: date = st.select_slider(
        "Конечная дата (доходность)",
        options=available_dates,
        value=available_dates[-1],
        key="ret_end_date",
    )

    strict_horizon = st.checkbox(
        "Показывать только фонды с полной историей на выбранном горизонте",
        value=True,
        key="ret_strict",
    )

    price_base = (
        df_sel.dropna(subset=["close"])
             .sort_values(["label", "tradedate"])
             .groupby(["label", "fund", "isin", "tradedate"], as_index=False)
             .agg(close=("close", "last"))
    )
    price_base = price_base[price_base["tradedate"] <= end_date_ret].copy()

    if price_base.empty:
        st.info("Недостаточно данных close для расчета доходности.")
        return

    def filter_full_horizon(prices: pd.DataFrame, start_date: date) -> pd.DataFrame:
        first_dates = prices.groupby("label")["tradedate"].min()
        ok_labels = first_dates[first_dates <= start_date].index
        return prices[prices["label"].isin(ok_labels)].copy()

    def build_cum_return(prices: pd.DataFrame, start_date: Optional[date], end_date_: date) -> pd.DataFrame:
        d = prices.copy()
        d = d[d["tradedate"] <= end_date_]
        if start_date is not None:
            d = d[d["tradedate"] >= start_date]
        if d.empty:
            return d

        d = d.sort_values(["label", "tradedate"]).copy()
        d["base_close"] = d.groupby("label")["close"].transform("first")
        d["base_date"] = d.groupby("label")["tradedate"].transform("first")
        d["base_date_str"] = d["base_date"].astype(str)
        d["ret_pct"] = (d["close"] / d["base_close"] - 1.0) * 100.0
        return d

    def plot_return_tab(ret_df: pd.DataFrame, title_suffix: str) -> None:
        if ret_df.empty:
            st.info("Недостаточно данных для построения доходности по выбранному горизонту.")
            return

        ret_df = ret_df.copy()
        ret_df["ret_pct"] = pd.to_numeric(ret_df["ret_pct"], errors="coerce").round(2)

        fig_ret = px.line(
            ret_df,
            x="tradedate",
            y="ret_pct",
            color="label",
            markers=True,
            custom_data=["fund", "isin", "close", "base_date_str", "base_close", "ret_pct"],
            labels={"ret_pct": "Доходность, %", "tradedate": "Дата"},
            title=None,
        )
        fig_ret.update_layout(separators=". ")
        fig_ret.update_traces(
            hovertemplate=(
                "Дата: %{x|%Y-%m-%d}<br>"
                "Фонд: %{customdata[0]}<br>"
                "Цена закрытия: %{customdata[2]:,.0f}<br>"
                "Базовая дата: %{customdata[3]}<br>"
                "Накопленная доходность: %{customdata[5]:+.2f}%<br>"
                f"<extra>{title_suffix}</extra>"
            )
        )
        st.plotly_chart(fig_ret, width="stretch")

        last_points = (
            ret_df.sort_values(["label", "tradedate"])
                  .groupby(["label", "fund", "isin"], as_index=False)
                  .tail(1)
        )
        out = last_points[["fund", "isin", "base_date_str", "base_close", "close", "ret_pct"]].copy()
        out = out.rename(columns={
            "fund": "Фонд",
            "base_date_str": "Дата покупки 1 пая",
            "base_close": "Цена на базовую дату, руб",
            "close": "Цена на конечную дату, руб",
            "ret_pct": "Доходность, %",
        }).sort_values("Доходность, %", ascending=False, na_position="last")

        disp = out.copy()
        disp["Цена на базовую дату, руб"] = disp["Цена на базовую дату, руб"].map(
            lambda x: "—" if pd.isna(x) else f"{x:,.2f}".replace(",", " ")
        )
        disp["Цена на конечную дату, руб"] = disp["Цена на конечную дату, руб"].map(
            lambda x: "—" if pd.isna(x) else f"{x:,.2f}".replace(",", " ")
        )
        disp["Доходность, %"] = disp["Доходность, %"].map(
            lambda x: "—" if pd.isna(x) else f"{x:+.2f}%"
        )
        st.dataframe(disp, width="stretch", hide_index=True)

    end_ts = pd.to_datetime(end_date_ret)
    start_1y = (end_ts - pd.DateOffset(years=1)).date()
    start_3y = (end_ts - pd.DateOffset(years=3)).date()

    tab_all, tab_1y, tab_3y = st.tabs(["За все время", "За 1 год", "За 3 года"])

    with tab_all:
        ret_all = build_cum_return(price_base, start_date=None, end_date_=end_date_ret)
        plot_return_tab(ret_all, "Горизонт: все время")

    with tab_1y:
        pb = filter_full_horizon(price_base, start_1y) if strict_horizon else price_base
        ret_1y = build_cum_return(pb, start_date=start_1y, end_date_=end_date_ret)
        plot_return_tab(ret_1y, "Горизонт: 1 год")

    with tab_3y:
        pb = filter_full_horizon(price_base, start_3y) if strict_horizon else price_base
        ret_3y = build_cum_return(pb, start_date=start_3y, end_date_=end_date_ret)
        plot_return_tab(ret_3y, "Горизонт: 3 года")

    st.caption(f"История для доходности: 2020-01-01T00:00:00Z — {date_to} (UTC). Кеш 24 часа.")

# -----------------------
# Рендер вкладки "Основные графики"
# -----------------------
def render_main_graphs(df_sel: pd.DataFrame, date_from: str, date_to: str) -> None:
    mode = st.radio(
        "Режим просмотра",
        options=["Режим истории", "Режим сравнения (сегодня vs предыдущий торговый день)"],
        horizontal=True,
    )

    if mode == "Режим истории":
        available_dates = sorted(df_sel["tradedate"].dropna().unique().tolist())
        if not available_dates:
            st.warning("Нет дат для выбранных фондов.")
            return

        end_date: date = st.select_slider(
            "Конечная дата",
            options=available_dates,
            value=available_dates[-1],
        )

        max_window = min(252, len(available_dates))
        if max_window <= 1:
            window = 1
        else:
            window = st.slider(
                "Длина периода (торговые дни)",
                min_value=2,
                max_value=max_window,
                value=min(30, max_window),
                step=1,
            )

        date_to_idx = {d: i for i, d in enumerate(available_dates)}
        end_idx = date_to_idx[end_date]
        start_idx = max(0, end_idx - window + 1)
        start_date = available_dates[start_idx]

        period_df = df_sel[(df_sel["tradedate"] >= start_date) & (df_sel["tradedate"] <= end_date)].copy()

        st.subheader("Изменение цены закрытия (%)")
        st.caption(f"Период: {start_date} — {end_date} (торговых дней в окне: {window})")

        if period_df.empty:
            st.info("За выбранный период нет данных.")
        else:
            period_df = period_df.dropna(subset=["close"]).copy()
            period_df = (
                period_df.sort_values(["label", "tradedate"])
                         .groupby(["label", "fund", "isin", "tradedate"], as_index=False)
                         .agg(close=("close", "last"), volume=("volume", "sum"), value=("value", "sum"))
            )

            if period_df.empty:
                st.info("Нет данных close для построения графика.")
            else:
                base_close = period_df.groupby("label")["close"].transform("first")
                period_df["close_change_pct"] = (period_df["close"] / base_close - 1.0) * 100.0

                fig_close_pct = px.line(
                    period_df,
                    x="tradedate",
                    y="close_change_pct",
                    color="label",
                    hover_data=["fund", "isin", "close", "volume", "value"],
                    markers=True,
                    labels={"close_change_pct": "Изменение цены закрытия, %", "tradedate": "Дата"},
                )
                fig_close_pct.update_yaxes(hoverformat=".2f")
                fig_close_pct.update_layout(separators=". ")
                fig_close_pct.update_traces(
                    hovertemplate=(
                        "Дата: %{x|%Y-%m-%d}<br>"
                        "Фонд: %{customdata[0]}<br>"
                        "Цена закрытия: %{customdata[2]:,.2f}<br>"
                        "Изменение цены закрытия: %{y:.2f}%<br>"
                        "Объем бумаг: %{customdata[3]:,.0f}<br>"
                        "Оборот (руб): %{customdata[4]:,.0f}<br>"
                        "<extra>%{fullData.name}</extra>"
                    )
                )
                st.plotly_chart(fig_close_pct, width="stretch")

        st.subheader("Волатильность цены")

        vol_roll_n = 21
        price_base = df_sel[df_sel["tradedate"] <= end_date].copy()
        price_base = (
            price_base.dropna(subset=["close"])
                      .sort_values(["label", "tradedate"])
                      .groupby(["label", "fund", "isin", "tradedate"], as_index=False)
                      .agg(open=("open", "last"), close=("close", "last"))
        )

        all_dates = sorted(price_base["tradedate"].dropna().unique().tolist())
        if len(all_dates) < 2:
            st.info("Недостаточно торговых дат для расчета волатильности.")
        else:
            end_date_eff = end_date if end_date in set(all_dates) else all_dates[-1]
            end_i = all_dates.index(end_date_eff)
            prev_date = all_dates[end_i - 1] if end_i - 1 >= 0 else None

            tab_roll, tab_month = st.tabs(["Дневная", "Месячная (цена открытия к цене закрытия)"])

            with tab_roll:
                st.markdown("""
**Методология расчета:**
1) Для каждого фонда берем цену закрытия по торговым дням.  
2) Считаем дневные изменения в виде **лог-доходности**, сравнивая сегодняшнее закрытие с предыдущим днем.  
3) На каждом дне считаем **стандартное отклонение** этих изменений на окне из 21 торгового дня.  
4) Переводим в проценты.
""")
                if prev_date is None:
                    st.info("Нет предыдущей даты для расчета.")
                else:
                    st.markdown(
                        f"**Выбранная дата:** {end_date_eff}&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;"
                        f"**Предыдущая дата:** {prev_date}"
                    )

                    price_base["ret_cc"] = np.log(
                        price_base["close"] / price_base.groupby("label")["close"].shift(1)
                    )
                    price_base["vol_roll_daily"] = (
                        price_base.groupby("label")["ret_cc"]
                                  .rolling(vol_roll_n)
                                  .std()
                                  .reset_index(level=0, drop=True)
                    )
                    price_base["vol_roll_daily_pct"] = price_base["vol_roll_daily"] * 100.0

                    asof = (
                        price_base[price_base["tradedate"] <= prev_date]
                        .sort_values(["label", "tradedate"])
                        .dropna(subset=["vol_roll_daily"])
                        .groupby(["label", "fund", "isin"], as_index=False)
                        .tail(1)
                    )

                    skeleton = df_sel[["label", "fund", "isin"]].drop_duplicates()
                    out = skeleton.merge(asof[["label", "vol_roll_daily_pct"]], on="label", how="left")

                    out = out.rename(columns={
                        "fund": "Фонд",
                        "isin": "ISIN",
                        "vol_roll_daily_pct": "Дневная волатильность, %",
                    }).sort_values("Дневная волатильность, %", ascending=False, na_position="last")

                    display = out.drop(columns=["label"], errors="ignore").copy()
                    display["Дневная волатильность, %"] = display["Дневная волатильность, %"].map(
                        lambda x: "—" if pd.isna(x) else f"{x:.2f}%"
                    )
                    st.dataframe(display, width="stretch", hide_index=True)

            with tab_month:
                st.markdown("""
Показывает, насколько сильно цена менялась **внутри торгового дня** (от открытия до закрытия) за выбранный отрезок дней в текущем месяце.

**Методология расчета:**
1) Берем только дни внутри выбранного месяца: от 1-го числа до конечной даты.  
2) Ползунком выбирается, сколько **последних торговых дней** внутри месяца берется в расчет.  
3) Для каждого дня считаем внутридневное изменение как **лог-доходность цены открытия к цене закрытия**.  
4) Считаем **стандартное отклонение** этих изменений по выбранным дням.  
5) Переводим в проценты.
""")
                month_start = end_date_eff.replace(day=1)
                st.markdown(f"**Месяц:** {month_start} — {end_date_eff}")

                month_df = price_base[
                    (price_base["tradedate"] >= month_start) &
                    (price_base["tradedate"] <= end_date_eff)
                ].copy()

                month_df = month_df.dropna(subset=["open", "close"]).copy()
                month_df = month_df[(month_df["open"] > 0) & (month_df["close"] > 0)].copy()

                if month_df.empty:
                    st.info("Нет данных open/close для расчета в выбранном месяце.")
                else:
                    month_dates = sorted(month_df["tradedate"].unique().tolist())
                    if len(month_dates) < 2:
                        st.info("Недостаточно торговых дат в месяце для расчета волатильности.")
                    else:
                        n_days = st.slider(
                            "Период внутри месяца (торговые дни)",
                            min_value=2,
                            max_value=len(month_dates),
                            value=min(10, len(month_dates)),
                            step=1,
                            key="vol_month_n_days",
                        )

                        selected_dates = set(month_dates[-n_days:])
                        month_cut = month_df[month_df["tradedate"].isin(selected_dates)].copy()
                        month_cut["ret_oc"] = np.log(month_cut["close"] / month_cut["open"])

                        month_tbl = (
                            month_cut.groupby(["label", "fund", "isin"], as_index=False)
                                     .agg(vol_oc_daily=("ret_oc", "std"))
                        )
                        month_tbl["vol_oc_daily_pct"] = month_tbl["vol_oc_daily"] * 100.0

                        skeleton = df_sel[["label", "fund", "isin"]].drop_duplicates()
                        out2 = skeleton.merge(
                            month_tbl[["label", "vol_oc_daily_pct"]],
                            on="label",
                            how="left"
                        )

                        col_name = f"Волатильность цены открытия к цене закрытия, % (за {n_days} дней)"
                        out2 = out2.rename(columns={
                            "fund": "Фонд",
                            "isin": "ISIN",
                            "vol_oc_daily_pct": col_name,
                        }).sort_values(col_name, ascending=False, na_position="last")

                        display2 = out2.drop(columns=["label"], errors="ignore").copy()
                        display2[col_name] = display2[col_name].map(
                            lambda x: "—" if pd.isna(x) else f"{x:.2f}%"
                        )
                        st.dataframe(display2, width="stretch", hide_index=True)

        st.subheader("Оборот торгов")
        st.caption(f"Период: {start_date} — {end_date} (торговых дней в окне: {window})")

        vol_df = df_sel[(df_sel["tradedate"] >= start_date) & (df_sel["tradedate"] <= end_date)].copy()
        vol_df = vol_df.dropna(subset=["value"]).copy()
        vol_df = (
            vol_df.sort_values(["label", "tradedate"])
                  .groupby(["label", "fund", "isin", "tradedate"], as_index=False)
                  .agg(
                      value=("value", "sum"),
                      volume=("volume", "sum"),
                      numtrades=("numtrades", "sum"),
                      close=("close", "last"),
                  )
        )

        if vol_df.empty:
            st.info("За выбранный период нет данных по обороту.")
        else:
            tab_table, tab_log, tab_hist = st.tabs(["Таблица", "Логарифмический график", "Гистограмма"])

            with tab_table:
                change_mode = st.radio(
                    "Изменение оборота считать как",
                    options=["День к дню", "Месяц к месяцу (21 торговый день)"],
                    horizontal=True,
                )

                vol_dates = sorted(vol_df["tradedate"].unique().tolist())
                vol_dates_idx = {d: i for i, d in enumerate(vol_dates)}
                end_date_eff = end_date if end_date in vol_dates_idx else vol_dates[-1]
                end_i = vol_dates_idx[end_date_eff]

                summary_table = None
                caption_text = ""

                if change_mode == "День к дню":
                    if end_i - 1 < 0:
                        st.info("Недостаточно дат для сравнения день к дню.")
                    else:
                        prev_date = vol_dates[end_i - 1]
                        today_df = vol_df[vol_df["tradedate"] == end_date_eff][["label", "fund", "isin", "value"]].copy()
                        today_df = today_df.rename(columns={"value": "value_today"})
                        prev_df = vol_df[vol_df["tradedate"] == prev_date][["label", "value"]].copy()
                        prev_df = prev_df.rename(columns={"value": "value_prev"})
                        caption_text = f"Сравнение: {end_date_eff} vs {prev_date}"

                        summary = today_df.merge(prev_df, on="label", how="left")
                        summary["change_pct"] = np.where(
                            (summary["value_prev"].notna()) & (summary["value_prev"] > 0),
                            (summary["value_today"] / summary["value_prev"] - 1.0) * 100.0,
                            np.nan,
                        )
                        summary_table = summary[["fund", "isin", "value_today", "change_pct"]].copy()

                else:
                    if end_i - 21 + 1 < 0:
                        st.info("Недостаточно дат для окна в 21 торговый день.")
                    else:
                        cur_start_i = end_i - 21 + 1
                        cur_dates = set(vol_dates[cur_start_i:end_i + 1])

                        prev_end_i = cur_start_i - 1
                        prev_start_i = prev_end_i - 21 + 1

                        if prev_start_i < 0:
                            st.info("Недостаточно дат для сравнения двух окон по 21 торговому дню.")
                        else:
                            prev_dates = set(vol_dates[prev_start_i:prev_end_i + 1])

                            today_df = (
                                vol_df[vol_df["tradedate"].isin(cur_dates)]
                                .groupby(["label", "fund", "isin"], as_index=False)["value"]
                                .sum()
                                .rename(columns={"value": "value_today"})
                            )
                            prev_df = (
                                vol_df[vol_df["tradedate"].isin(prev_dates)]
                                .groupby(["label"], as_index=False)["value"]
                                .sum()
                                .rename(columns={"value": "value_prev"})
                            )
                            caption_text = (
                                f"Окна по 21 торговому дню: "
                                f"{vol_dates[cur_start_i]} — {end_date_eff} vs "
                                f"{vol_dates[prev_start_i]} — {vol_dates[prev_end_i]}"
                            )

                            summary = today_df.merge(prev_df, on="label", how="left")
                            summary["change_pct"] = np.where(
                                (summary["value_prev"].notna()) & (summary["value_prev"] > 0),
                                (summary["value_today"] / summary["value_prev"] - 1.0) * 100.0,
                                np.nan,
                            )
                            summary_table = summary[["fund", "isin", "value_today", "change_pct"]].copy()

                if caption_text:
                    st.caption(caption_text)

                if summary_table is not None and not summary_table.empty:
                    summary_table = summary_table.rename(columns={
                        "fund": "Фонд",
                        "isin": "ISIN",
                        "value_today": "Оборот, руб (за текущий период)",
                        "change_pct": "Изменение оборота, %",
                    }).sort_values("Оборот, руб (за текущий период)", ascending=False)

                    display_table = summary_table.copy()
                    display_table["Оборот, руб (за текущий период)"] = display_table["Оборот, руб (за текущий период)"].map(
                        lambda x: f"{x:,.0f}".replace(",", " ") if pd.notna(x) else "—"
                    )
                    display_table["Изменение оборота, %"] = display_table["Изменение оборота, %"].map(
                        lambda x: "—" if pd.isna(x) else f"{x:+.2f}%"
                    )
                    st.dataframe(display_table, width="stretch", hide_index=True)

            with tab_log:
                val_pos = vol_df[vol_df["value"] > 0].copy()
                if val_pos.empty:
                    st.warning("Для логарифмической шкалы нужны положительные значения value > 0.")
                else:
                    val_pos["log10_value"] = np.log10(val_pos["value"])
                    fig_val = px.line(
                        val_pos,
                        x="tradedate",
                        y="value",
                        color="label",
                        custom_data=["log10_value", "fund", "close", "volume", "numtrades"],
                        markers=True,
                        labels={"value": "Оборот, руб", "tradedate": "Дата"},
                    )
                    fig_val.update_yaxes(type="log")
                    fig_val.update_layout(separators=". ")
                    fig_val.update_traces(
                        hovertemplate=(
                            "Дата: %{x|%Y-%m-%d}<br>"
                            "Фонд: %{customdata[1]}<br>"
                            "Оборот (руб): %{y:,.0f}<br>"
                            "log10(оборота): %{customdata[0]:.3f}<br>"
                            "Цена закрытия: %{customdata[2]:,.2f}<br>"
                            "Объем бумаг: %{customdata[3]:,.0f}<br>"
                            "Сделок: %{customdata[4]:,.0f}<br>"
                            "<extra>%{fullData.name}</extra>"
                        )
                    )
                    st.plotly_chart(fig_val, width="stretch")

            with tab_hist:
                sum_df = (
                    vol_df.groupby(["label", "fund", "isin"], as_index=False)
                          .agg(value_sum=("value", "sum"))
                          .sort_values("value_sum", ascending=True)
                )

                if sum_df.empty:
                    st.info("Нет данных для гистограммы.")
                else:
                    max_val = float(sum_df["value_sum"].max())
                    if max_val >= 1e9:
                        divisor, unit, decimals = 1e9, "млрд руб", 2
                    elif max_val >= 1e6:
                        divisor, unit, decimals = 1e6, "млн руб", 2
                    elif max_val >= 1e3:
                        divisor, unit, decimals = 1e3, "тыс руб", 0
                    else:
                        divisor, unit, decimals = 1.0, "руб", 0

                    sum_df["value_sum_unit"] = sum_df["value_sum"] / divisor
                    sum_df["window_days"] = int(window)

                    fig_hist = px.bar(
                        sum_df,
                        x="value_sum_unit",
                        y="label",
                        orientation="h",
                        custom_data=["fund", "isin", "window_days"],
                        labels={"value_sum_unit": f"Суммарный оборот за период, {unit}", "label": "Фонд"},
                        text="value_sum_unit",
                    )
                    fig_hist.update_layout(separators=". ")
                    fig_hist.update_xaxes(tickformat=f",.{decimals}f")
                    fig_hist.update_traces(
                        texttemplate=f"%{{x:,.{decimals}f}} {unit}",
                        textposition="outside",
                        hovertemplate=(
                            "Фонд: %{y}<br>"
                            "Период: %{customdata[2]} торговых дней<br>"
                            f"Суммарный оборот за период: %{{x:,.{decimals}f}} {unit}<br>"
                            "<extra></extra>"
                        ),
                    )
                    st.plotly_chart(fig_hist, width="stretch")

        st.subheader("Изменение среднего размера сделки (руб/сделку) за выбранный период")
        st.caption(f"Период: {start_date} — {end_date} (торговых дней в окне: {window})")

        avg_df = df_sel[(df_sel["tradedate"] >= start_date) & (df_sel["tradedate"] <= end_date)].copy()
        avg_df = avg_df.dropna(subset=["value", "numtrades"]).copy()
        avg_df = (
            avg_df.sort_values(["label", "tradedate"])
                  .groupby(["label", "fund", "isin", "tradedate"], as_index=False)
                  .agg(value=("value", "sum"), numtrades=("numtrades", "sum"))
        )
        avg_df = avg_df[avg_df["numtrades"] > 0].copy()
        avg_df["avg_trade_rub"] = avg_df["value"] / avg_df["numtrades"]

        if avg_df.empty:
            st.info("Нет данных для расчета среднего размера сделки.")
        else:
            def _first_last_in_window(g: pd.DataFrame) -> pd.Series:
                g = g.sort_values("tradedate")
                first = g.iloc[0]
                last = g.iloc[-1]
                return pd.Series({
                    "avg_first": first["avg_trade_rub"],
                    "avg_last": last["avg_trade_rub"],
                    "date_first": first["tradedate"],
                    "date_last": last["tradedate"],
                })

            win = (
                avg_df.groupby(["label", "fund", "isin"], as_index=False)
                      .apply(_first_last_in_window, include_groups=False)
            )

            win["change_abs"] = win["avg_last"] - win["avg_first"]
            win["change_pct"] = np.where(
                (win["avg_first"].notna()) & (win["avg_first"] > 0),
                (win["avg_last"] / win["avg_first"] - 1.0) * 100.0,
                np.nan,
            )

            out = win[["fund", "isin", "avg_first", "avg_last", "change_abs", "change_pct"]].copy()
            out = out.rename(columns={
                "fund": "Фонд",
                "isin": "ISIN",
                "avg_first": "Средний размер сделки (начало выбранного периода), руб/сделку",
                "avg_last": "Средний размер сделки (конец выбранного периода), руб/сделку",
                "change_abs": "Изменение, руб",
                "change_pct": "Изменение, %",
            }).sort_values("Изменение, %", ascending=False, na_position="last")

            display = out.copy()
            display["Средний размер сделки (начало выбранного периода), руб/сделку"] = display[
                "Средний размер сделки (начало выбранного периода), руб/сделку"
            ].map(lambda x: f"{x:,.0f}".replace(",", " ") if pd.notna(x) else "—")
            display["Средний размер сделки (конец выбранного периода), руб/сделку"] = display[
                "Средний размер сделки (конец выбранного периода), руб/сделку"
            ].map(lambda x: f"{x:,.0f}".replace(",", " ") if pd.notna(x) else "—")
            display["Изменение, руб"] = display["Изменение, руб"].map(
                lambda x: "—" if pd.isna(x) else f"{x:+,.0f}".replace(",", " ")
            )
            display["Изменение, %"] = display["Изменение, %"].map(
                lambda x: "—" if pd.isna(x) else f"{x:+.2f}%"
            )
            st.dataframe(display, width="stretch", hide_index=True)

    else:
        st.subheader("Изменение цены закрытия, средневзвешенная цена")

        cmp_df = df_sel.dropna(subset=["close"]).copy()
        cmp_df = (
            cmp_df.sort_values(["label", "tradedate"])
                  .groupby(["label", "fund", "isin", "tradedate"], as_index=False)
                  .agg(close=("close", "last"), waprice=("waprice", "last"))
        )

        all_dates = sorted(cmp_df["tradedate"].dropna().unique().tolist())
        if len(all_dates) < 2:
            st.info("Недостаточно торговых дат для сравнения (нужно минимум 2).")
        else:
            sel_date: date = st.select_slider(
                "Дата для сравнения с предыдущим торговым днем",
                options=all_dates[1:],
                value=all_dates[-1],
            )
            sel_i = all_dates.index(sel_date)
            prev_date = all_dates[sel_i - 1]

            st.markdown(
                f"**Выбранная дата:** {sel_date}&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;"
                f"**Предыдущая торговая дата:** {prev_date}"
            )

            last_df = cmp_df[cmp_df["tradedate"] == sel_date][["label", "fund", "isin", "close", "waprice"]].copy()
            last_df = last_df.rename(columns={"close": "close_last", "waprice": "waprice_last"})

            prev_df = cmp_df[cmp_df["tradedate"] == prev_date][["label", "close", "waprice"]].copy()
            prev_df = prev_df.rename(columns={"close": "close_prev", "waprice": "waprice_prev"})

            summary = last_df.merge(prev_df, on="label", how="left")
            summary["change_pct"] = np.where(
                (summary["close_prev"].notna()) & (summary["close_prev"] > 0),
                (summary["close_last"] / summary["close_prev"] - 1.0) * 100.0,
                np.nan,
            )
            summary["waprice_change_pct"] = np.where(
                (summary["waprice_prev"].notna()) & (summary["waprice_prev"] > 0),
                (summary["waprice_last"] / summary["waprice_prev"] - 1.0) * 100.0,
                np.nan,
            )

            out = summary[[
                "fund", "isin",
                "close_last", "close_prev", "change_pct",
                "waprice_last", "waprice_prev", "waprice_change_pct",
            ]].copy()

            out = out.rename(columns={
                "fund": "Фонд",
                "isin": "ISIN",
                "close_last": "Последняя цена закрытия, руб",
                "close_prev": "Предыдущая цена закрытия, руб",
                "change_pct": "Изменение цены закрытия, %",
                "waprice_last": "Последняя средневзвешенная цена, руб",
                "waprice_prev": "Предыдущая средневзвешенная цена, руб",
                "waprice_change_pct": "Изменение средневзвешенной цены, %",
            }).sort_values("Изменение цены закрытия, %", ascending=False, na_position="last")

            display = out.copy()
            for c in [
                "Последняя цена закрытия, руб",
                "Предыдущая цена закрытия, руб",
                "Последняя средневзвешенная цена, руб",
                "Предыдущая средневзвешенная цена, руб",
            ]:
                display[c] = display[c].map(lambda x: f"{x:,.2f}".replace(",", " ") if pd.notna(x) else "—")

            display["Изменение цены закрытия, %"] = display["Изменение цены закрытия, %"].map(
                lambda x: "—" if pd.isna(x) else f"{x:+.2f}%"
            )
            display["Изменение средневзвешенной цены, %"] = display["Изменение средневзвешенной цены, %"].map(
                lambda x: "—" if pd.isna(x) else f"{x:+.2f}%"
            )

            st.dataframe(display, width="stretch", hide_index=True)

    st.caption(f"Период загрузки: {date_from} — {date_to} (UTC). Кеш обновляется раз в сутки.")

# -----------------------
# Рендер вкладки "Торги Ф4 и Ф5"
# -----------------------
def render_accent_tab() -> None:
    with st.sidebar:
        st.header("Параметры Ф4 / Ф5")

        today = date.today()
        default_from = date(today.year, 1, 1)

        d_from = st.date_input("Начало", value=default_from, key="accent_d_from")
        d_to = st.date_input("Конец", value=today, key="accent_d_to")

        period_kind = st.radio(
            "Итоги: период отображения",
            options=["Неделя", "Месяц", "Квартал", "Год"],
            horizontal=True,
            index=0,
            key="accent_period_kind",
        )

        min_end = date(2010, 1, 1)
        state_key = "accent_summary_end_input"

        if state_key not in st.session_state:
            st.session_state[state_key] = d_to

        if st.session_state[state_key] > d_to:
            st.session_state[state_key] = d_to
        if st.session_state[state_key] < min_end:
            st.session_state[state_key] = min_end

        st.date_input(
            "Конец периода итогов",
            key=state_key,
            min_value=min_end,
            max_value=d_to,
        )

        def _shift_summary_end(forward: bool):
            cur = st.session_state[state_key]
            new_end = shift_date_by_period(cur, period_kind, forward=forward)
            new_end = min(max(new_end, min_end), d_to)
            st.session_state[state_key] = new_end

        c_prev, c_next = st.columns(2)
        with c_prev:
            st.button(
                "← Предыдущий период",
                width="stretch",
                key="accent_prev_period",
                on_click=_shift_summary_end,
                args=(False,),
            )
        with c_next:
            st.button(
                "Следующий период →",
                width="stretch",
                key="accent_next_period",
                on_click=_shift_summary_end,
                args=(True,),
            )

        period_start, period_end = calc_period_window(st.session_state[state_key], period_kind)
        st.caption(f"Окно итогов: {period_start} — {period_end}")

        if st.button("Очистить кеш", width="stretch", key="accent_clear_cache"):
            st.cache_data.clear()
            st.rerun()

    if d_from > d_to:
        st.error("Некорректный период: начало позже конца.")
        return

   st.title("Таблица торгов по дням: Акцент IV и Акцент 5")

    with st.spinner("Загружаю данные из API..."):
        df_raw_period = load_accent_raw(period_start, period_end)
        df_raw_day = load_accent_raw(d_from, d_to)

    if df_raw_period.empty and df_raw_day.empty:
        st.warning("Нет данных в выбранном диапазоне.")
        return

    st.markdown("### Оборот на Мосбирже, млн руб")
    value_mode = st.radio(
        "Оборот на графике",
        options=["Основной режим", "РПС", "Итого"],
        horizontal=True,
        index=0,
        key="accent_turnover_value_mode",
    )

    fig_turnover = build_turnover_stacked_chart(df_raw=df_raw_day, value_mode=value_mode)
    if fig_turnover.data:
        st.plotly_chart(fig_turnover, width="stretch")
    else:
        st.info("Нет данных для построения графика.")

    st.markdown(f"### Итоги, {period_kind.lower()}: Основной режим vs РПС")
    period_long = build_range_summary(df_raw_period, period_start, period_end)
    period_pivot = pivot_period_summary(period_long)

    if period_pivot.empty:
        st.warning("Не удалось построить таблицу итогов за период.")
    else:
        with st.popover("Выбрать колонки для отображения"):
            all_cols = period_pivot.columns.tolist()
            default_cols = [c for c in all_cols if c in [
                "Начало периода", "Конец периода", "Фонд",
                "Оборот, руб — Основной режим", "Оборот, руб — РПС", "Оборот, руб — Итого",
            ]]
            if not default_cols:
                default_cols = all_cols

            selected_cols = st.multiselect(
                "Колонки",
                options=all_cols,
                default=default_cols,
                key="accent_selected_period_cols",
            )

        period_show = period_pivot[selected_cols].copy() if selected_cols else period_pivot.copy()

        fmt = {}
        for c in period_show.columns:
            if "Кол-во бумаг" in c or "Сделок" in c or "Оборот" in c:
                fmt[c] = "{:.0f}"

        st.dataframe(
            period_show.style.format(fmt),
            width="stretch",
            hide_index=True,
        )

        xlsx_period = df_to_xlsx_bytes(period_show, sheet_name=f"Итоги_{period_kind}")
        st.download_button(
            f"Скачать Excel: итоги, {period_kind.lower()}",
            data=xlsx_period,
            file_name=f"accent_summary_{period_kind.lower()}_{d_from}_{d_to}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
            key="accent_download_summary",
        )

    st.markdown("### Детальные торги по дням")
    accent_daily = build_accent_daily_table(df_raw_day)

    if accent_daily.empty:
        st.warning("Не удалось построить детальную таблицу.")
    else:
        mode_options = ["Все режимы"] + accent_daily["Режим торгов"].dropna().unique().tolist()
        selected_mode = st.radio(
            "Показывать в дневной таблице",
            options=mode_options,
            horizontal=True,
            key="accent_daily_mode",
        )

        if selected_mode != "Все режимы":
            accent_daily_show = accent_daily[accent_daily["Режим торгов"] == selected_mode].copy()
        else:
            accent_daily_show = accent_daily.copy()

        accent_daily_show = accent_daily_show.reset_index(drop=True)

        st.dataframe(
            accent_daily_show,
            width="stretch",
            hide_index=True,
            column_config={"_index": None},
        )

        xlsx_daily = df_to_xlsx_bytes(accent_daily_show, sheet_name="Детальные торги")
        st.download_button(
            "Скачать Excel: детальные торги",
            data=xlsx_daily,
            file_name=f"accent_daily_{d_from}_{d_to}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
            key="accent_download_daily",
        )

# -----------------------
# Главный UI
# -----------------------
st.title("Торги ЗПИФ")

if not API_LOGIN or not API_PASS:
    st.error("Не заданы API_LOGIN / API_PASS. Добавьте их в Secrets или переменные окружения.")
    st.stop()

section = st.segmented_control(
    "Вкладка",
    options=["Торги Ф4 и Ф5", "Основные графики", "Доходность"],
    default="Торги Ф4 и Ф5",
    key="main_section",
)

if section == "Торги Ф4 и Ф5":
    render_accent_tab()

elif section == "Основные графики":
    utc_now = datetime.now(timezone.utc)
    date_to = utc_now.strftime("%Y-%m-%dT23:59:59Z")
    date_from = "2025-01-01T00:00:00Z"

    df = load_df(ZPIF_SECIDS, date_from, date_to)
    df = df[df["isin"].isin(TARGET_ISINS)].copy()

    if df.empty:
        st.warning("Данных не найдено за выбранный период.")
    else:
        save_snapshot_csv(df)
        df_sel = render_fund_selector(df)
        if df_sel.empty:
            st.warning("По выбранным фондам нет данных.")
        else:
            render_main_graphs(df_sel, date_from, date_to)

else:
    utc_now = datetime.now(timezone.utc)
    date_to = utc_now.strftime("%Y-%m-%dT23:59:59Z")
    date_from = "2020-01-01T00:00:00Z"

    try:
        df = load_df_long_history(ZPIF_SECIDS, date_from, date_to, chunk_size=30)
    except Exception as e:
        st.error(f"Ошибка загрузки длинной истории: {e}")
        st.stop()

    df = df[df["isin"].isin(TARGET_ISINS)].copy()

    if df.empty:
        st.warning("Данных не найдено за выбранный период.")
    else:
        save_snapshot_csv(df)
        df_sel = render_fund_selector(df)
        if df_sel.empty:
            st.warning("По выбранным фондам нет данных.")
        else:
            render_returns(df_sel, date_to)
