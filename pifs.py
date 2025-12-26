#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime, timezone, date

# -----------------------
# Конфигурация
# -----------------------
API_URL = "https://dh2.efir-net.ru/v2"

def get_secret(name: str, default: str = "") -> str:
    if hasattr(st, "secrets") and name in st.secrets:
        return str(st.secrets[name])
    return os.getenv(name, default)

API_LOGIN = get_secret("API_LOGIN", "accentam-api-test1")
API_PASS  = get_secret("API_PASS",  "652Dsw")

ZPIF_SECIDS = [
    "RU000A105328", # парус
    "RU000A0JRHC0", # атриум
    "RU000A1034U7", # ар бизн
    "RU000A1099U0", # совр 9
    "RU000A100WZ5", # акцент
]

# -----------------------
# API helpers
# -----------------------
def do_post_request(url: str, body: dict, token: str | None) -> dict | None:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.post(url, json=body, headers=headers, timeout=60)
    if r.status_code == 200:
        return r.json()
    return None

def get_token(login: str, password: str) -> str | None:
    url = f"{API_URL}/Account/Login"
    body = {"login": login, "password": password}
    data = do_post_request(url, body, None)
    if not data:
        return None
    return data.get("token")

def chunk_list(lst, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]

def fetch_all_trading_results(token: str, instruments: list[str], date_from: str, date_to: str, page_size: int = 100):
    url = f"{API_URL}/Moex/History"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    all_data = []
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

        # защита от бесконечного цикла
        if len(page_data) < page_size:
            break

    return all_data

# -----------------------
# Загрузка данных (кеширование)
# -----------------------
@st.cache_data(ttl=24 * 60 * 60)
def load_df(secids: list[str], date_from: str, date_to: str) -> pd.DataFrame:
    token = get_token(API_LOGIN, API_PASS)
    if not token:
        raise RuntimeError("Ошибка авторизации: не удалось получить токен")

    all_results = []
    for chunk in chunk_list(secids, 100):
        all_results.extend(fetch_all_trading_results(token, chunk, date_from, date_to))

    if not all_results:
        return pd.DataFrame(columns=["shortname", "isin", "volume", "close", "tradedate"])

    df = pd.DataFrame(all_results)[["shortname", "isin", "volume", "close", "tradedate"]]
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["close"]  = pd.to_numeric(df["close"], errors="coerce")

    # tradedate -> datetime.date (для корректной работы виджетов)
    df["tradedate"] = pd.to_datetime(df["tradedate"], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=["shortname", "isin", "tradedate"])

    return df

# -----------------------
# Streamlit UI
# -----------------------
st.title("Торги ЗПИФ")

# 1) Сначала формируем период загрузки и грузим df
utc_now = datetime.now(timezone.utc)
date_from = "2025-01-01T00:00:00Z"
date_to   = utc_now.strftime("%Y-%m-%dT23:59:59Z")

df = load_df(ZPIF_SECIDS, date_from, date_to)

if df.empty:
    st.warning("Данных не найдено за выбранный период.")
    st.stop()

# 2) Выбор фондов
available_funds = sorted(df["shortname"].unique().tolist())
selected_funds = st.multiselect(
    "Выберите ЗПИФы",
    available_funds,
    default=available_funds[: min(5, len(available_funds))],
)

df_sel = df[df["shortname"].isin(selected_funds)].copy()
df_sel = df_sel.sort_values(["tradedate", "shortname"])

# 3) Даты только по выбранным фондам (чтобы ползунок не показывал даты без данных)
available_dates = sorted(df_sel["tradedate"].dropna().unique().tolist())
if not available_dates:
    st.warning("Нет дат для выбранных ЗПИФов. Проверьте фильтр фондов.")
    st.stop()

# 4) Ползунок конечной даты (по торговым датам)
end_date: date = st.select_slider(
    "Конечная дата",
    options=available_dates,
    value=available_dates[-1],
)

# 5) Ползунок длины периода (в торговых днях) с защитой от коротких рядов
max_window = min(252, len(available_dates))
if max_window <= 1:
    window = 1
else:
    default_window = min(30, max_window)
    window = st.slider(
        "Длина периода (торговые дни)",
        min_value=2,
        max_value=max_window,
        value=default_window,
        step=1,
    )

# 6) Рассчитываем старт окна
date_to_idx = {d: i for i, d in enumerate(available_dates)}
end_idx = date_to_idx[end_date]
start_idx = max(0, end_idx - window + 1)
start_date = available_dates[start_idx]

# 7) Данные за период для линии close
period_df = df_sel[(df_sel["tradedate"] >= start_date) & (df_sel["tradedate"] <= end_date)].copy()

# 7) Данные за период для линии close
period_df = df_sel[(df_sel["tradedate"] >= start_date) & (df_sel["tradedate"] <= end_date)].copy()

st.subheader("Изменение цены закрытия (%)")
st.caption(f"Период: {start_date} — {end_date} (торговых дней в окне: {window})")

if period_df.empty:
    st.info("За выбранный период нет данных.")
else:
    # Чтобы не схлопывались фонды с одинаковым shortname, используем label = shortname + (isin)
    period_df = period_df.dropna(subset=["close"]).copy()
    period_df["label"] = period_df["shortname"].astype(str) + " (" + period_df["isin"].astype(str) + ")"
    period_df = period_df.sort_values(["label", "tradedate"])

    # Накопленное изменение цены к первой доступной дате в окне
    base_close = period_df.groupby("label")["close"].transform("first")
    period_df["close_change_pct"] = (period_df["close"] / base_close - 1.0) * 100.0

    fig_close_pct = px.line(
        period_df,
        x="tradedate",
        y="close_change_pct",
        color="label",
        hover_data=["shortname", "isin", "close", "volume"],
        markers=True,
        labels={"close_change_pct": "Изменение цены, %", "tradedate": "Дата"},
    )
    st.plotly_chart(fig_close_pct, use_container_width=True)

# 7b) Линейный график объемов за тот же период + два режима шкалы
st.subheader("Динамика объема торгов (volume) по выбранным ЗПИФам")
st.caption(f"Период: {start_date} — {end_date} (торговых дней в окне: {window})")

scale_mode = st.radio(
    "Шкала по оси Y для объемов",
    options=["Обычная", "Логарифмическая"],
    horizontal=True,
)

vol_df = df_sel[(df_sel["tradedate"] >= start_date) & (df_sel["tradedate"] <= end_date)].copy()
vol_df = vol_df.dropna(subset=["volume"]).copy()
vol_df["label"] = vol_df["shortname"].astype(str) + " (" + vol_df["isin"].astype(str) + ")"
vol_df = vol_df.sort_values(["label", "tradedate"])

if vol_df.empty:
    st.info("За выбранный период нет данных по volume.")
else:
    fig_vol_line = px.line(
        vol_df,
        x="tradedate",
        y="volume",
        color="label",
        hover_data=["shortname", "isin", "close"],
        markers=True,
        labels={"volume": "Объем торгов", "tradedate": "Дата"},
    )

    if scale_mode == "Логарифмическая":
        # log не работает для нулей/отрицательных значений: отфильтруем нули
        vol_pos = vol_df[vol_df["volume"] > 0].copy()
        if vol_pos.empty:
            st.warning("Для логарифмической шкалы нужны положительные объемы (volume > 0).")
        else:
            fig_vol_line = px.line(
                vol_pos,
                x="tradedate",
                y="volume",
                color="label",
                hover_data=["shortname", "isin", "close"],
                markers=True,
                labels={"volume": "Объем торгов (лог. шкала)", "tradedate": "Дата"},
            )
            fig_vol_line.update_yaxes(type="log")

    st.plotly_chart(fig_vol_line, use_container_width=True)

st.caption(f"Период загрузки: {date_from} — {date_to} (UTC). Кеш обновляется раз в сутки.")
