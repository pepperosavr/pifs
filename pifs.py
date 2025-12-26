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

# Рекомендуется хранить секреты в .streamlit/secrets.toml или переменных окружения.
# Пример secrets.toml:
# API_LOGIN = "..."
# API_PASS  = "..."
def get_secret(name: str, default: str = "") -> str:
    if hasattr(st, "secrets") and name in st.secrets:
        return str(st.secrets[name])
    return os.getenv(name, default)

API_LOGIN = get_secret("API_LOGIN", "accentam-api-test1")  # при желании уберите дефолт
API_PASS  = get_secret("API_PASS",  "652Dsw")              # при желании уберите дефолт


ZPIF_SECIDS = [
    "RU000A105328",
    "RU000A1099U0",
    "RU000A1034U7",
    "RU000A10A117",
    "RU000A100WZ5",
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

        # Дополнительная защита от бесконечного цикла, если API всегда возвращает непустой список
        if len(page_data) < page_size:
            break

    return all_data


# -----------------------
# Загрузка данных (кеширование)
# -----------------------
@st.cache_data(ttl=24 * 60 * 60)  # обновление не чаще 1 раза в сутки
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
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # Приводим дату к datetime.date (это устраняет проблему Streamlit с pandas.Timestamp)
    df["tradedate"] = pd.to_datetime(df["tradedate"], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=["shortname", "isin", "tradedate"])

    return df


# -----------------------
# Streamlit UI
# -----------------------
st.title("Торги ЗПИФ: объем по выбранной дате")

# Автоматическое обновление dateTo: текущая дата в UTC (до конца дня)
utc_now = datetime.now(timezone.utc)
date_from = "2025-01-01T00:00:00Z"
date_to = utc_now.strftime("%Y-%m-%dT23:59:59Z")  # авто-обновляется каждый день

df = load_df(ZPIF_SECIDS, date_from, date_to)

if df.empty:
    st.warning("Данных не найдено за выбранный период.")
    st.stop()

available_funds = sorted(df["shortname"].unique().tolist())
selected_funds = st.multiselect(
    "Выберите ЗПИФы",
    available_funds,
    default=available_funds[: min(5, len(available_funds))],
)

# Чтобы выбирать только реальные торговые даты (без выходных), используем select_slider
available_dates = sorted(df["tradedate"].unique().tolist())
selected_date: date = st.select_slider(
    "Выберите дату",
    options=available_dates,
    value=available_dates[-1],
)

filtered_df = df[(df["shortname"].isin(selected_funds)) & (df["tradedate"] == selected_date)]

st.subheader("Объем торгов по выбранным ЗПИФам")
if filtered_df.empty:
    st.info("По выбранным ЗПИФам на эту дату нет строк (проверьте фильтры).")
else:
    fig = px.bar(
        filtered_df,
        x="shortname",
        y="volume",
        hover_data=["isin", "close", "tradedate"],
        color="shortname",
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption(f"Период загрузки: {date_from} — {date_to} (UTC). Кеш обновляется раз в сутки.")
