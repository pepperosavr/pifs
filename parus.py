import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

# =========================
# Конфигурация
# =========================
API_URL = "https://dh2.efir-net.ru/v2"

ACCENT_IV_ISIN = "RU000A100WZ5"
ACCENT_5_ISIN  = "RU000A10DQF7"

# Акцент 5 передавать как тикер 
ISIN_TO_MOEX_CODE = {
    ACCENT_5_ISIN: "XACCSK",
}
MOEX_CODE_TO_ISIN = {v: k for k, v in ISIN_TO_MOEX_CODE.items()}

ACCENT_TARGET_ISINS = [ACCENT_IV_ISIN, ACCENT_5_ISIN]
ACCENT_INSTRUMENTS_FOR_API = [
    ACCENT_IV_ISIN,                 # Акцент IV можно слать ISIN
    ISIN_TO_MOEX_CODE[ACCENT_5_ISIN] # Акцент 5 слать тикер
]

st.set_page_config(page_title="Акцент IV/5: дневные торги", layout="wide")
st.title("Таблица торгов по дням: Акцент IV и Акцент 5")

# =========================
# Secrets / Env
# =========================
def get_secret(name: str, default: str = "") -> str:
    # 1) Streamlit secrets
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            v = st.secrets.get(name)
            if v is not None and str(v).strip() != "":
                return str(v)
    except Exception:
        pass
    # 2) Env
    v = os.getenv(name)
    if v is not None and str(v).strip() != "":
        return str(v)
    return default

API_LOGIN = get_secret("API_LOGIN", "")
API_PASS  = get_secret("API_PASS", "")

if not API_LOGIN or not API_PASS:
    st.error("Не заданы API_LOGIN / API_PASS. Добавьте их в Secrets или переменные окружения.")
    st.stop()

# =========================
# Утилиты агрегации дублей
# =========================
def _sum_or_single(s: pd.Series, decimals: int = 0) -> float:
    """
    Если значения одинаковые (после округления) -> считаем, что это дубль и берем одно.
    Иначе -> суммируем.
    """
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

# =========================
# API: auth + history
# =========================
def do_post_request(url: str, body: Dict[str, Any], token: Optional[str]) -> Optional[Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.post(url, json=body, headers=headers, timeout=60)
    if r.status_code == 200:
        return r.json()
    raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

def get_token(login: str, password: str) -> str:
    url = f"{API_URL}/Account/Login"
    body = {"login": login, "password": password}
    data = do_post_request(url, body, token=None)
    token = data.get("token") if isinstance(data, dict) else None
    if not token:
        raise RuntimeError("Не удалось получить token из ответа /Account/Login")
    return str(token)

def fetch_moex_history(
    token: str,
    instruments: List[str],
    date_from: str,
    date_to: str,
    page_size: int = 100
) -> List[Dict[str, Any]]:
    url = f"{API_URL}/Moex/History"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    out: List[Dict[str, Any]] = []
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
            raise RuntimeError(f"API error page {page}: {r.status_code} {r.text}")

        data = r.json()
        if not data:
            break

        out.extend(data)
        if len(data) < page_size:
            break

        page += 1

    return out

def _to_api_dt(d: date, end_of_day: bool) -> str:
    return f"{d:%Y-%m-%d}T23:59:59Z" if end_of_day else f"{d:%Y-%m-%d}T00:00:00Z"

@st.cache_data(ttl=24 * 60 * 60)
def load_accent_daily(d_from: str, d_to: str) -> pd.DataFrame:
    """
    Возвращает ТОЛЬКО дневную таблицу (1 строка на isin+tradedate),
    уже без дублеи и готовую для UI/расчетов.
    """
    token = get_token(API_LOGIN, API_PASS)
    if not token:
        raise RuntimeError("Ошибка авторизации: не удалось получить токен")

    # 1) качаем (внутри функции сырье существует, но наружу не отдаём)
    all_rows = fetch_all_trading_results(
        token=token,
        instruments=["RU000A100WZ5", "XACCSK"],   # Акцент IV + Акцент 5 (тикер)
        date_from=d_from,
        date_to=d_to,
        page_size=100,
    )
    if not all_rows:
        return pd.DataFrame()

    raw = pd.DataFrame(all_rows)

    # 2) гарантируем колонки
    need = ["shortname","secid","isin","tradedate","open","high","low","close","waprice","volume","value","numtrades"]
    for c in need:
        if c not in raw.columns:
            raw[c] = np.nan

    # 3) нормализация типов
    raw["tradedate"] = pd.to_datetime(raw["tradedate"], errors="coerce", utc=True).dt.date
    for c in ["open","high","low","close","waprice","volume","value","numtrades"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # 4) восстановление ISIN для XACCSK
    raw["isin"] = raw["isin"].replace({"XACCSK": "RU000A10DQF7"})
    raw["isin"] = raw["isin"].fillna(raw["secid"].replace({"XACCSK": "RU000A10DQF7"}))

    # 5) фильтр только 2 фондов
    raw = raw[raw["isin"].isin(["RU000A100WZ5", "RU000A10DQF7"])].dropna(subset=["isin","tradedate"])
    if raw.empty:
        return pd.DataFrame()

    # 6) схлопываем дубли по дню
    day = collapse_day_duplicates(raw)

    # 7) приводим названия
    day["fund"] = day["isin"].map({
        "RU000A100WZ5": "АКЦЕНТ IV",
        "RU000A10DQF7": "Акцент 5",
    }).fillna(day.get("fund"))

    return day

def build_accent_daily_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Делает 1 строку на день на фонд с учетом дублей (например, сессии).
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    d = df_raw.copy()
    d = d.sort_values(["isin", "tradedate", "secid"], na_position="last")

    def _agg(g: pd.DataFrame) -> pd.Series:
        vol = _sum_or_single(g["volume"], decimals=0)
        val_api = _sum_or_single(g["value"], decimals=0)
        trades = _sum_or_single(g["numtrades"], decimals=0)

        open_  = _first_non_na(g["open"])
        close_ = _last_non_na(g["close"])

        high_ = pd.to_numeric(g["high"], errors="coerce").max(skipna=True)
        low_  = pd.to_numeric(g["low"], errors="coerce").min(skipna=True)

        wap = _vwap(g["waprice"], g["volume"])
        # fallback, если VWAP не собрался
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
        d.groupby(["tradedate", "isin", "fund"], as_index=False)
         .apply(_agg, include_groups=False)
         .reset_index(drop=True)
         .rename(columns={"tradedate": "Дата", "isin": "ISIN", "fund": "Фонд"})
         .sort_values(["Дата", "Фонд"])
    )

    # Округления для читабельности
    for c in ["Open", "High", "Low", "Close", "Средняя цена (waprice)"]:
        out[c] = out[c].round(2)
    for c in ["Рубли (volume*waprice)", "Рубли (close*volume)", "Рубли как в API (value)"]:
        out[c] = out[c].round(0)

    return out

# =========================
# UI: параметры
# =========================
with st.sidebar:
    st.header("Параметры")

    today = date.today()
    default_from = date(today.year, 1, 1)

    d_from = st.date_input("Начало", value=default_from, min_value=date(2010, 1, 1))
    d_to   = st.date_input("Конец", value=today, min_value=date(2010, 1, 1))

    st.caption("Если приложение загружается слишком медленно, уменьшите период.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Очистить кеш", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

if d_from > d_to:
    st.error("Некорректный период: начало позже конца.")
    st.stop()

# =========================
# Загрузка и расчет
# =========================
with st.spinner("Загружаю данные из API..."):
    df_raw = load_accent_raw(d_from, d_to)

if df_raw.empty:
    st.warning("Нет данных в выбранном диапазоне.")
    st.stop()

# Диагностика: есть ли дубли на (isin, tradedate)
dup_cnt = (
    df_raw.groupby(["isin", "tradedate"]).size().reset_index(name="rows")
)
has_dups = (dup_cnt["rows"] > 1).any()

c1, c2, c3 = st.columns(3)
c1.metric("Строк (сырых)", f"{len(df_raw):,}".replace(",", " "))
c2.metric("Дат", f"{df_raw['tradedate'].nunique():,}".replace(",", " "))
c3.metric("Есть дубли по дню", "да" if has_dups else "нет")

accent_daily = build_accent_daily_table(df_raw)

st.subheader("Дневная таблица (1 строка на день на фонд)")
st.dataframe(accent_daily, use_container_width=True, hide_index=True)

csv_bytes = accent_daily.to_csv(index=False, encoding="utf-8").encode("utf-8")
st.download_button(
    "Скачать CSV",
    data=csv_bytes,
    file_name="accent_iv_5_daily.csv",
    mime="text/csv",
    use_container_width=True,
)

st.caption("Поля: volume (шт), Open/High/Low/Close, рубли через waprice, рубли через close, value от API (контроль), numtrades.")
