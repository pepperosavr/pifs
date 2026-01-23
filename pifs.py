#!/usr/bin/env python
# coding: utf-8

# In[5]:

import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime, timezone, date
from pathlib import Path

# -----------------------
# Конфигурация
# -----------------------
API_URL = "https://dh2.efir-net.ru/v2"

def get_secret(name: str, default: str = "") -> str:
    if hasattr(st, "secrets") and name in st.secrets:
        return str(st.secrets[name])
    return os.getenv(name, default)

API_LOGIN = get_secret("API_LOGIN", "accentam-api1")
API_PASS  = get_secret("API_PASS",  "653Bsw")

# 1) НУЖНЫЕ ФОНДЫ (ISIN + названия)
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
    "RU000A100WZ5": "АКЦЕНТ IV",
    "RU000A10DQF7": "Акцент 5",

    # Самолет 
    "RU000A10A117": "ЗПИФ СМЛТ",
    "RU000A1099U0": "ЗПИФСовр 9",
    # СФН 
    "RU000A1034U7": "ЗПИФСовр 7",
    "RU000A0JWAW3": "СоврАрБизн",

    # ВИМ-Инвестиции 
    "RU000A102N77": "РД",
    "RU000A103B62": "РД ПРО",

    # Рентал-Про 
    "RU000A108157": "Рентал-Про",

    # Активо ----
    "RU000A10CLY1": "АКТИВО ДВАДЦАТЬ ОДИН",
    "RU000A1092L4": "АКТИВО ДВАДЦАТЬ",
    "RU000A10ATA8": "АКТИВО ФЛИППИНГ",
}

# ВАЖНО: некоторым фондам в API нужно передавать не ISIN, а торговыи код MOEX (SECID/ticker).
# Для "Акцент 5" это XACCSK при ISIN RU000A10DQF7. :contentReference[oaicite:1]{index=1}
ISIN_TO_MOEX_CODE = {
    "RU000A10DQF7": "XACCSK", # Акцент 5
    "RU000A10A117": "XHOUSE", # самолет
    "RU000A108BZ2": "XTRIUMF" # триумф
}

# То, что хотим видеть в итоговых данных (ISIN-ы)
TARGET_ISINS = list(FUND_MAP.keys())

# То, что реально отправляем в API (MOEX-коды, где нужно; иначе ISIN как раньше)
ZPIF_SECIDS = [ISIN_TO_MOEX_CODE.get(isin, isin) for isin in TARGET_ISINS]

# Обратно: SECID -> ISIN (чтобы восстановить isin в ответе API)
MOEX_CODE_TO_ISIN = {secid: isin for isin, secid in ISIN_TO_MOEX_CODE.items()}
    
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

        if len(page_data) < page_size:
            break

    return all_data

def _to_num_series(s: pd.Series) -> pd.Series:
    # Нормализуем типичные форматы чисел: пробелы/неразрывные пробелы, запятая как десятичный разделитель
    if s is None:
        return pd.Series(dtype="float64")
    s = s.astype(str)
    s = s.str.replace("\xa0", "", regex=False)  # NBSP
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    s = s.replace(["", "nan", "None"], np.nan)
    return pd.to_numeric(s, errors="coerce")

# -----------------------
# Загрузка данных (кеширование)
# -----------------------

@st.cache_data(ttl=24 * 60 * 60)
def load_df(secids: list[str], date_from: str, date_to: str) -> pd.DataFrame:
    token = get_token(API_LOGIN, API_PASS)
    if not token:
        raise RuntimeError("Ошибка авторизации: не удалось получить токен")

    all_results = []
    for chunk in chunk_list(secids, 30):
        all_results.extend(fetch_all_trading_results(token, chunk, date_from, date_to))

    if not all_results:
        return pd.DataFrame(columns=["shortname", "fund", "isin", "volume", "value", "numtrades", "close", "tradedate"])

    raw = pd.DataFrame(all_results)

    need_cols = ["shortname", "secid", "isin", "volume", "value", "numtrades", "open", "close", "waprice", "tradedate"]
    for c in need_cols:
        if c not in raw.columns:
            raw[c] = np.nan

    df = raw[need_cols].copy()

    # Нормализуем пустые значения
    df["isin"] = df["isin"].replace(["", "nan", "None"], np.nan)
    df["secid"] = df["secid"].replace(["", "nan", "None"], np.nan)

# 1) если isin пустои, берем secid (часто secid уже является ISIN)
    df["isin"] = df["isin"].fillna(df["secid"])

# 2) если в isin лежит MOEX-код (XACCSK/XTRIUMF/...), переводим в ISIN
    df["isin"] = df["isin"].replace(MOEX_CODE_TO_ISIN)

    df["volume"]    = _to_num_series(df["volume"])
    df["value"]     = _to_num_series(df["value"])
    df["numtrades"] = _to_num_series(df["numtrades"])
    df["close"]     = _to_num_series(df["close"])
    df["waprice"]   = _to_num_series(df["waprice"])
    df["open"]      = _to_num_series(df["open"])

    df["tradedate"] = pd.to_datetime(df["tradedate"], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=["isin", "tradedate"])

    # имя фонда 
    df["fund"] = df["isin"].map(FUND_MAP).fillna(df["shortname"].astype(str))

    return df

from dateutil.relativedelta import relativedelta  # pip install python-dateutil

def _to_date(d: str) -> date:
    # "2018-01-01T00:00:00Z" -> date(2018,1,1)
    return pd.to_datetime(d, utc=True).date()

def _to_api_dt(d: date, end_of_day: bool = False) -> str:
    if end_of_day:
        return f"{d:%Y-%m-%d}T23:59:59Z"
    return f"{d:%Y-%m-%d}T00:00:00Z"

def _iter_month_ranges(date_from: str, date_to: str, step_months: int):
    """
    Генерирует полуинтервалы [start, end] в формате API.
    """
    start = _to_date(date_from)
    end = _to_date(date_to)

    cur = start
    while cur <= end:
        nxt = (cur + relativedelta(months=step_months))
        # делаем end включительно, но не выходим за date_to
        seg_end = min(end, (nxt - relativedelta(days=1)))
        yield _to_api_dt(cur, end_of_day=False), _to_api_dt(seg_end, end_of_day=True)
        cur = nxt

@st.cache_data(ttl=24 * 60 * 60)
def load_df_long_history(
    secids: list[str],
    date_from: str,
    date_to: str,
    chunk_size: int = 30,
    step_months: int = 6,
    page_size: int = 100,
) -> pd.DataFrame:
    """
    Длинная история: дробим запрос по времени и по инструментам.
    Возвращает те же поля, что и load_df, чтобы дальнеи код не менялся.
    """
    token = get_token(API_LOGIN, API_PASS)
    if not token:
        raise RuntimeError("Ошибка авторизации: не удалось получить токен")

    all_results = []

    # 1) режем по инструментам
    for sec_chunk in chunk_list(secids, chunk_size):
        # 2) режем по времени
        for d_from, d_to in _iter_month_ranges(date_from, date_to, step_months):
            part = fetch_all_trading_results(
                token=token,
                instruments=list(sec_chunk),
                date_from=d_from,
                date_to=d_to,
                page_size=page_size,
            )
            if part:
                all_results.extend(part)

    if not all_results:
        return pd.DataFrame(columns=["shortname", "fund", "isin", "volume", "value", "numtrades", "open", "close", "waprice", "tradedate"])

    raw = pd.DataFrame(all_results)

    need_cols = ["shortname", "secid", "isin", "volume", "value", "numtrades", "open", "close", "waprice", "tradedate"]
    for c in need_cols:
        if c not in raw.columns:
            raw[c] = np.nan

    df = raw[need_cols].copy()

    df["isin"] = df["isin"].replace(["", "nan", "None"], np.nan)
    df["secid"] = df["secid"].replace(["", "nan", "None"], np.nan)

    df["isin"] = df["isin"].fillna(df["secid"])
    df["isin"] = df["isin"].replace(MOEX_CODE_TO_ISIN)

    df["volume"]    = _to_num_series(df["volume"])
    df["value"]     = _to_num_series(df["value"])
    df["numtrades"] = _to_num_series(df["numtrades"])
    df["close"]     = _to_num_series(df["close"])
    df["waprice"]   = _to_num_series(df["waprice"])
    df["open"]      = _to_num_series(df["open"])

    df["tradedate"] = pd.to_datetime(df["tradedate"], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=["isin", "tradedate"])

    # имя фонда (как в load_df)
    df["fund"] = df["isin"].map(FUND_MAP).fillna(df["shortname"].astype(str))

    return df

# -----------------------
# Streamlit UI
# -----------------------
st.title("Торги ЗПИФ")

    # Выбор раздела: segmented_control есть не во всех версиях Streamlit
if hasattr(st, "segmented_control"):
    section = st.segmented_control(
        "Раздел",
        options=["Основные графики", "Доходность"],
        default="Основные графики",
    )
else:
    section = st.radio(
        "Раздел",
        options=["Основные графики", "Доходность"],
        index=0,
        horizontal=True,
    )

# Период загрузки

utc_now = datetime.now(timezone.utc)
date_to = utc_now.strftime("%Y-%m-%dT23:59:59Z")

if section == "Доходность":
    date_from = "2018-01-01T00:00:00Z"
    try:
        df = load_df_long_history(
            ZPIF_SECIDS,
            date_from,
            date_to,
            chunk_size=30,   # при необходимости уменьшить до 20/10
            step_months=6    # при необходимости уменьшить до 3
        )
    except Exception as e:
        st.error(f"Ошибка загрузки длиннои истории: {e}")
        st.stop()
else:
    date_from = "2025-01-01T00:00:00Z"
    df = load_df(ZPIF_SECIDS, date_from, date_to)

# На всякии случаи: оставляем только целевые ISIN
df = df[df["isin"].isin(TARGET_ISINS)].copy()

if df.empty:
    st.warning("Данных не найдено за выбранныи период.")
    st.stop()



# Выбор фондов

# квал/неквал

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
    "RU000A100WZ5": "квал",
    "RU000A10DQF7": "неквал",
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
    
}

df["qual"] = df["isin"].map(QUAL_BY_ISIN).fillna("неизвестно")

available_funds = sorted(df["fund"].unique().tolist())

SELECT_KEY = "fund_select"

# инициализация выбранных фондов (один раз)
if SELECT_KEY not in st.session_state:
    st.session_state[SELECT_KEY] = available_funds[:]  # по умолчанию все

# --- Группы УК ---

GROUPS = {
    
    "Акцент": [name for _, name in FUND_MAP.items()
               if str(name).startswith("Акцент") or str(name).upper().startswith("АКЦЕНТ")],
    "Парус":  [name for _, name in FUND_MAP.items() if str(name).startswith("ПАРУС-")],

    # СФН
    "СФН":    [
        FUND_MAP.get("RU000A1099U0", "ЗПИФСовр 9"),
        FUND_MAP.get("RU000A1034U7", "ЗПИФСовр 7"),
        FUND_MAP.get("RU000A0JWAW3", "СоврАрБизн"),
    ],

    "Самолет": [FUND_MAP.get("RU000A10A117", "ЗПИФ СМЛТ")],

    # ВИМ-Инвестиции
    "ВИМ-Инвестиции": [
        FUND_MAP.get("RU000A102N77", "РД"),
        FUND_MAP.get("RU000A103B62", "РД ПРО"),
    ],

    # АБ Капитал
    "АБ-Капитал": [FUND_MAP.get("RU000A108157", "Рентал-Про")],

    # Активо
    "Активо": [
        FUND_MAP.get("RU000A10CLY1", "АКТИВО ДВАДЦАТЬ ОДИН"),
        FUND_MAP.get("RU000A1092L4", "АКТИВО ДВАДЦАТЬ"),
        FUND_MAP.get("RU000A10ATA8", "АКТИВО ФЛИППИНГ"),
    ],
}

GROUPS_QUAL = {
    "Квал":   sorted(df.loc[df["qual"] == "квал", "fund"].dropna().unique().tolist()),
    "Неквал": sorted(df.loc[df["qual"] == "неквал", "fund"].dropna().unique().tolist()),
}

def _add_group(group_name: str):
    gf = [f for f in GROUPS.get(group_name, []) if f in available_funds]
    st.session_state[SELECT_KEY] = sorted(set(st.session_state[SELECT_KEY]).union(gf))

def _remove_group(group_name: str):
    gf = [f for f in GROUPS.get(group_name, []) if f in available_funds]
    st.session_state[SELECT_KEY] = sorted(set(st.session_state[SELECT_KEY]) - set(gf))

def _add_qual(qual_key: str):
    gf = [f for f in GROUPS_QUAL.get(qual_key, []) if f in available_funds]
    st.session_state[SELECT_KEY] = sorted(set(st.session_state[SELECT_KEY]).union(gf))

def _remove_qual(qual_key: str):
    gf = [f for f in GROUPS_QUAL.get(qual_key, []) if f in available_funds]
    st.session_state[SELECT_KEY] = sorted(set(st.session_state[SELECT_KEY]) - set(gf))

st.multiselect(
    "Выберите фонды",
    options=available_funds,
    key=SELECT_KEY,
)

with st.expander("Быстрый выбор: УК и квал/неквал", expanded=False):

    st.markdown("#### По УК")
    cols = st.columns(len(GROUPS))
    for i, gname in enumerate(GROUPS.keys()):
        with cols[i]:
            st.markdown(f"**{gname}**")
            st.button(f"+ {gname}", key=f"btn_add_{gname}", use_container_width=True,
                      on_click=_add_group, args=(gname,))
            st.button(f"− {gname}", key=f"btn_rm_{gname}", use_container_width=True,
                      on_click=_remove_group, args=(gname,))

    st.markdown("#### По квалификации")
    q1, q2 = st.columns(2)
    with q1:
        st.button("+ Квал", key="btn_add_qual", use_container_width=True,
                  on_click=_add_qual, args=("Квал",))
        st.button("− Квал", key="btn_rm_qual", use_container_width=True,
                  on_click=_remove_qual, args=("Квал",))
    with q2:
        st.button("+ Неквал", key="btn_add_nonqual", use_container_width=True,
                  on_click=_add_qual, args=("Неквал",))
        st.button("− Неквал", key="btn_rm_nonqual", use_container_width=True,
                  on_click=_remove_qual, args=("Неквал",))

    a, b = st.columns(2)
    with a:
        st.button("Выбрать все", key="btn_all", use_container_width=True,
                  on_click=lambda: st.session_state.__setitem__(SELECT_KEY, available_funds[:]))
    with b:
        st.button("Снять все", key="btn_none", use_container_width=True,
                  on_click=lambda: st.session_state.__setitem__(SELECT_KEY, []))

available_funds = sorted(df["fund"].unique().tolist())

selected_funds = st.session_state[SELECT_KEY]
df_sel = df[df["fund"].isin(selected_funds)].copy().sort_values(["tradedate", "fund"])

if df_sel.empty:
    st.warning("По выбранным фондам нет данных.")
    st.stop()

# Единая метка для графиков/таблиц
df_sel["label"] = df_sel["fund"].astype(str) + " (" + df_sel["isin"].astype(str) + ")"

# =========================================================
# РАЗДЕЛ: ДОХОДНОСТЬ (глубокая история с 2018 года)
# =========================================================
if section == "Доходность":
    st.subheader("Доходность (накопленная, %): 1 пай, купленный в разные периоды")
    st.caption("Считается по цене (close): без учета возможных выплат/распределений.")

     # df уже загружен выше в виде длиннои истории (с 2018)
    date_from_long = "2018-01-01T00:00:00Z"
    df_long = df.copy()

    # 2) Те же названия и label
    df_long["fund"] = df_long["isin"].map(FUND_MAP).fillna(df_long["shortname"].astype(str))
    df_long["label"] = df_long["fund"].astype(str) + " (" + df_long["isin"].astype(str) + ")"

    # 3) Применяем выбранные фонды
    selected_funds = st.session_state[SELECT_KEY]
    df_long_sel = df_long[df_long["fund"].isin(selected_funds)].copy().sort_values(["tradedate", "fund"])

    if df_long_sel.empty:
        st.warning("По выбранным фондам нет данных в длинной истории (с 2018).")
        st.stop()

    # 4) Выбор конечной даты (отдельно для доходности)
    available_dates_long = sorted(df_long_sel["tradedate"].dropna().unique().tolist())
    if not available_dates_long:
        st.warning("Нет дат для расчета доходности.")
        st.stop()

    end_date_ret: date = st.select_slider(
        "Конечная дата (доходность)",
        options=available_dates_long,
        value=available_dates_long[-1],
        key="ret_end_date",
    )

    # 5) База цен: одна close на фонд/дату (схлопываем повторы)
    price_base = df_long_sel.dropna(subset=["close"]).copy()
    price_base = (
        price_base[price_base["tradedate"] <= end_date_ret]
        .sort_values(["label", "tradedate"])
        .groupby(["label", "fund", "isin", "tradedate"], as_index=False)
        .agg(close=("close", "last"))
    )

    if price_base.empty:
        st.info("Недостаточно данных close для расчета доходности.")
        st.stop()

    def filter_full_horizon(prices: pd.DataFrame, start_date: date) -> pd.DataFrame:
        """Оставляет только фонды, у которых первая доступная дата <= start_date."""
        first_dates = prices.groupby("label")["tradedate"].min()
        ok_labels = first_dates[first_dates <= start_date].index
        return prices[prices["label"].isin(ok_labels)].copy()

    def build_cum_return(prices: pd.DataFrame, start_date: date | None, end_date_: date) -> pd.DataFrame:
        """
        tradedate, label, fund, isin, close, base_date_str, base_close, ret_pct
        """
        d = prices.copy()
        d = d[d["tradedate"] <= end_date_]
        if start_date is not None:
            d = d[d["tradedate"] >= start_date]

        if d.empty:
            return d

        d = d.sort_values(["label", "tradedate"]).copy()

        # базовая дата и базовая цена = первая точка внутри горизонта
        d["base_close"] = d.groupby("label")["close"].transform("first")
        d["base_date"] = d.groupby("label")["tradedate"].transform("first")
        d["base_date_str"] = d["base_date"].astype(str)

        d["ret_pct"] = (d["close"] / d["base_close"] - 1.0) * 100.0
        return d

    # 6) Горизонты: "все время", "1 год", "3 года" от end_date_ret
    end_ts = pd.to_datetime(end_date_ret)
    start_1y = (end_ts - pd.DateOffset(years=1)).date()
    start_3y = (end_ts - pd.DateOffset(years=3)).date()

    tab_all, tab_1y, tab_3y = st.tabs(["За все время", "За 1 год", "За 3 года"])

    def plot_return_tab(ret_df: pd.DataFrame, title_suffix: str):
        if ret_df.empty:
            st.info("Недостаточно данных для построения доходности по выбранному горизонту.")
            return

        fig_ret = px.line(
            ret_df,
            x="tradedate",
            y="ret_pct",
            color="label",
            markers=True,
            custom_data=["fund", "isin", "close", "base_date_str", "base_close"],
            labels={"ret_pct": "Доходность, %", "tradedate": "Дата"},
            title=None,
        )
        fig_ret.update_layout(separators=". ")
        fig_ret.update_traces(
            hovertemplate=(
                "Дата: %{x|%Y-%m-%d}<br>"
                "Фонд: %{customdata[0]}<br>"
                "Цена закрытия: %{customdata[2]:,.2f}<br>"
                "Базовая дата: %{customdata[3]} (close=%{customdata[4]:,.2f})<br>"
                "Накопленная доходность: %{y:+.2f}%<br>"
                f"<extra>{title_suffix}</extra>"
            )
        )
        st.plotly_chart(fig_ret, use_container_width=True)

        # Итоговая доходность на end_date_ret
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
        st.dataframe(disp, use_container_width=True, hide_index=True)

    with tab_all:
        ret_all = build_cum_return(price_base, start_date=None, end_date_=end_date_ret)
        plot_return_tab(ret_all, "Горизонт: все время")

    with tab_1y:
        pb = filter_full_horizon(price_base, start_1y) if st.session_state.get("ret_strict", True) else price_base
        ret_1y = build_cum_return(pb, start_date=start_1y, end_date_=end_date_ret)
        plot_return_tab(ret_1y, "Горизонт: 1 год")

    with tab_3y:
        pb = filter_full_horizon(price_base, start_3y) if st.session_state.get("ret_strict", True) else price_base
        ret_3y = build_cum_return(pb, start_date=start_3y, end_date_=end_date_ret)
        plot_return_tab(ret_3y, "Горизонт: 3 года")

    st.caption(f"История для доходности: {date_from_long} — {date_to} (UTC). Кеш 24 часа.")
    st.stop()

# 2) КНОПКА: История/Сравнение (только для "Основных графиков")
mode = st.radio(
    "Режим просмотра",
    options=["Режим истории", "Режим сравнения (сегодня vs предыдущий торговый день)"],
    horizontal=True,
)

# ---------- РЕЖИМ 1: ИСТОРИЯ ----------
if mode == "Режим истории":
    available_dates = sorted(df_sel["tradedate"].dropna().unique().tolist())
    if not available_dates:
        st.warning("Нет дат для выбранных фондов.")
        st.stop()

    # Конечная дата
    end_date: date = st.select_slider(
        "Конечная дата",
        options=available_dates,
        value=available_dates[-1],
    )

    # Длина периода
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

    # Старт периода
    date_to_idx = {d: i for i, d in enumerate(available_dates)}
    end_idx = date_to_idx[end_date]
    start_idx = max(0, end_idx - window + 1)
    start_date = available_dates[start_idx]

    # -------- 7) Изменение цены закрытия (%) за период --------
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

        base_close = period_df.groupby("label")["close"].transform("first")
        period_df["close_change_pct"] = (period_df["close"] / base_close - 1.0) * 100.0

    fig_close_pct = px.line(
        period_df,
        x="tradedate",
        y="close_change_pct",
        color="label",
        markers=True,
        custom_data=["fund", "isin", "close", "volume", "value"],
        labels={"close_change_pct": "Изменение цены закрытия, %", "tradedate": "Дата"},
    )

    fig_close_pct.update_yaxes(hoverformat=".2f")
    fig_close_pct.update_layout(separators=". ")

    fig_close_pct.update_traces(
        hovertemplate=(
            "Дата: %{x|%Y-%m-%d}<br>"
            "Фонд: %{customdata[0]}<br>"
            "ISIN: %{customdata[1]}<br>"
            "Цена закрытия: %{customdata[2]:,.2f}<br>"
            "Изменение цены закрытия: %{y:.2f}%<br>"
            "Объем бумаг: %{customdata[3]:,.0f}<br>"
            "Оборот (руб): %{customdata[4]:,.0f}<br>"
            "<extra>%{fullData.name}</extra>"
        )
    )

    st.plotly_chart(fig_close_pct, use_container_width=True)

    # -------- 7a2) Волатильность цены (по close) --------
    
    st.subheader("Волатильность цены")

# Фиксированная длина rolling-окна (жестко задана)
    VOL_ROLL_N = 21

# Берем историю до выбранной конечной даты, чтобы rolling не зависел от длины window
    price_base = df_sel[df_sel["tradedate"] <= end_date].copy()

# Оставляем только нужное и схлопываем дубли на фонд/дату
    price_base = (
        price_base.dropna(subset=["close"])
                  .sort_values(["label", "tradedate"])
                  .groupby(["label", "fund", "isin", "tradedate"], as_index=False)
                  .agg(
                      open=("open", "last"),   # нужен для табл. 2
                      close=("close", "last"),
                  )
    )

    all_dates = sorted(price_base["tradedate"].dropna().unique().tolist())
    if len(all_dates) < 2:
        st.info("Недостаточно торговых дат для расчета волатильности.")
    else:
    # эффективная конечная дата (если end_date вдруг отсутствует в price_base)
        end_date_eff = end_date if end_date in set(all_dates) else all_dates[-1]
        end_i = all_dates.index(end_date_eff)
        prev_date = all_dates[end_i - 1] if end_i - 1 >= 0 else None

        tab_roll, tab_month = st.tabs(
            [f"Дневная", "Месячная (цена открытия к цене закрытия)"]
        )

    # ---------------------------------------------------------
    # TAB 1: дневная волатильность rolling, значение "за прошлый день"
    # ---------------------------------------------------------
        with tab_roll:
            st.markdown("""
        **Методология расчета:**
        1) Для каждого фонда берем цену закрытия по торговым дням.  
        2) Считаем дневные изменения в виде **лог-доходности**, сравнивая сегодняшнее закрытие с предыдущим днем.
        3) На каждом дне считаем **стандартное отклонение** этих изменений на окне из 21 торгового дня.  
        4) Переводим в проценты.
        """)
            if prev_date is None:
                st.info("Нет предыдущей даты для расчета (слишком короткая история).")
            else:
                st.markdown(
                    f"**Выбранная дата:** {end_date_eff}&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;"
                    f"**Предыдущая дата:** {prev_date}"
                )

            # лог-доходность close-close
                price_base["ret_cc"] = np.log(
                    price_base["close"] / price_base.groupby("label")["close"].shift(1)
                )

            # rolling std по доходностям
                price_base["vol_roll_daily"] = (
                    price_base.groupby("label")["ret_cc"]
                              .rolling(VOL_ROLL_N)
                              .std()
                              .reset_index(level=0, drop=True)
                )

            # дневная волатильность в %
                price_base["vol_roll_daily_pct"] = price_base["vol_roll_daily"] * 100.0

            # Берем значение на дату <= prev_date (если у фонда нет ровно prev_date, берем последнее до нее)
                asof = (
                    price_base[price_base["tradedate"] <= prev_date]
                    .sort_values(["label", "tradedate"])
                    .dropna(subset=["vol_roll_daily"])
                    .groupby(["label", "fund", "isin"], as_index=False)
                    .tail(1)
                )

            # "скелет" всех фондов, чтобы не пропадали строки при merge
                skeleton = df_sel[["label", "fund", "isin"]].drop_duplicates()

                out = skeleton.merge(
                    asof[["label", "vol_roll_daily_pct"]],
                    on="label",
                    how="left"
                )

                out = out.rename(columns={
                    "fund": "Фонд",
                    "isin": "ISIN",
                    "vol_roll_daily_pct": f"Дневная волатильность, %",
                }).sort_values(
                    f"Дневная волатильность, %",
                    ascending=False,
                    na_position="last"
                )

                display = out.copy()
                display = display.drop(columns=["label"], errors="ignore")
                display[f"Дневная волатильность, %"] = display[
                    f"Дневная волатильность, %"
                ].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}%")

                st.dataframe(display, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # TAB 2: месячная волатильность по внутридневным изменениям Open->Close

        
    # ---------------------------------------------------------
        with tab_month:
            st.markdown("""
        Показывает, насколько сильно цена менялась **внутри торгового дня** (от открытия до закрытия) за выбранный отрезок дней в текущем месяце.

        **Методология расчета:**
        1) Берем только дни внутри выбранного месяца: от 1-го числа до конечной даты.  
        2) Ползунком выбирается, сколько **последних торговых дней** внутри месяца берется в расчет.  
        3) Для каждого дня считаем внутридневное изменение как **лог-доходность цены открытия к цене закрытия**.  
        4) Считаем **стандартное отклонение** этих изменений по выбранным дням.  
        5) Переводим в проценты.
        """
        )
    # Месяц по выбранной конечной дате
            month_start = end_date_eff.replace(day=1)

            st.markdown(
                f"**Месяц:** {month_start} — {end_date_eff}"
            )

    # Данные внутри месяца до end_date_eff
            month_df = price_base[
                (price_base["tradedate"] >= month_start) &
                (price_base["tradedate"] <= end_date_eff)
            ].copy()

    # Нужны open и close
            month_df = month_df.dropna(subset=["open", "close"]).copy()
            month_df = month_df[(month_df["open"] > 0) & (month_df["close"] > 0)].copy()

            if month_df.empty:
                st.info("Нет данных open/close для расчета (Open→Close) в выбранном месяце.")
            else:
        # Список торговых дат в этом месяце (по факту данных)
                month_dates = sorted(month_df["tradedate"].unique().tolist())
                if len(month_dates) < 2:
                    st.info("Недостаточно торговых дат в месяце для расчета волатильности.")
                else:
            # Ползунок: сколько торговых дней брать внутри месяца
                    max_n = len(month_dates)
                    n_days = st.slider(
                "Период внутри месяца (торговые дни)",
                        min_value=2,
                        max_value=max_n,
                        value=min(10, max_n),
                        step=1,
                        key="vol_month_n_days",
                    )

            # Берем последние N торговых дат месяца (заканчивая end_date_eff)
                    selected_dates = set(month_dates[-n_days:])
                    month_cut = month_df[month_df["tradedate"].isin(selected_dates)].copy()

            # внутридневная доходность Open->Close
                    month_cut["ret_oc"] = np.log(month_cut["close"] / month_cut["open"])

            # std внутри выбранного отрезка
                    month_tbl = (
                        month_cut.groupby(["label", "fund", "isin"], as_index=False)
                             .agg(vol_oc_daily=("ret_oc", "std"))
                    )

                    month_tbl["vol_oc_daily_pct"] = month_tbl["vol_oc_daily"] * 100.0

            # "скелет" всех фондов, чтобы строки не пропадали
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
                    })

            # сортируем по реально существующей колонке
                    out2 = out2.sort_values(col_name, ascending=False, na_position="last")

                    display2 = out2.copy()
                    display2 = display2.drop(columns=["label"], errors="ignore")
                    display2[col_name] = display2[col_name].map(
                        lambda x: "—" if pd.isna(x) else f"{x:.2f}%"
                    )

                    st.dataframe(display2, use_container_width=True, hide_index=True)

    # -------- 7b) Оборот торгов: Таблица + Логарифм. график + Гистограмма --------
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

        # --- TAB 1: Таблица (day/day или window/window) ---
        
        with tab_table:
            change_mode = st.radio(
                "Изменение оборота считать как",
                options=["День к дню", "Месяц к месяцу (21 торговый день)"],
                horizontal=True,
            )

    # торговые даты именно для оборота (на случаи пропусков)
            vol_dates = sorted(vol_df["tradedate"].unique().tolist())
            vol_dates_idx = {d: i for i, d in enumerate(vol_dates)}

    # если end_date отсутствует в обороте — берем последнюю доступную
            end_date_eff = end_date if end_date in vol_dates_idx else vol_dates[-1]
            end_i = vol_dates_idx[end_date_eff]

            if change_mode == "День к дню":
                if end_i - 1 < 0:
                    st.caption("Недостаточно дат для сравнения день к дню.")
                    st.stop()

                prev_date = vol_dates[end_i - 1]

                today_df = vol_df[vol_df["tradedate"] == end_date_eff][["label", "fund", "isin", "value"]].copy()
                today_df = today_df.rename(columns={"value": "value_today"})

                prev_df = vol_df[vol_df["tradedate"] == prev_date][["label", "value"]].copy()
                prev_df = prev_df.rename(columns={"value": "value_prev"})

                caption_text = f"Сравнение: {end_date_eff} vs {prev_date}"

            else:
        # окно 21 торговыи день
                if end_i - 21 + 1 < 0:
                    st.caption("Недостаточно дат для окна (21 торговыи день).")
                    st.stop()

                cur_start_i = end_i - 21 + 1
                cur_dates = set(vol_dates[cur_start_i : end_i + 1])

                prev_end_i = cur_start_i - 1
                prev_start_i = prev_end_i - 21 + 1
                if prev_start_i < 0:
                    st.caption("Недостаточно дат для сравнения двух окон по 21 торговому дню.")
                    st.stop()

                prev_dates = set(vol_dates[prev_start_i : prev_end_i + 1])

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

            st.caption(caption_text)

    # ---- Общая часть: собираем таблицу для обоих режимов ----
            summary = today_df.merge(prev_df, on="label", how="left")

            summary["change_pct"] = np.where(
                (summary["value_prev"].notna()) & (summary["value_prev"] > 0),
                (summary["value_today"] / summary["value_prev"] - 1.0) * 100.0,
                np.nan,
            )

            summary_table = summary[["fund", "isin", "value_today", "change_pct"]].copy()
            summary_table = summary_table.rename(
                columns={
                    "fund": "Фонд",
                    "isin": "ISIN",
                    "value_today": "Оборот, руб (за текущий период)",
                    "change_pct": "Изменение оборота, %",
                }
            ).sort_values("Оборот, руб (за текущий период)", ascending=False)

            display_table = summary_table.copy()

    # пробел как разделитель тысяч
            display_table["Оборот, руб (за текущий период)"] = display_table["Оборот, руб (за текущий период)"] \
                if "Оборот, руб (за текущий период)" in display_table.columns else display_table["Оборот, руб (за текущий период)"]
            display_table["Оборот, руб (за текущий период)"] = display_table["Оборот, руб (за текущий период)"].map(
                lambda x: (f"{x:,.0f}".replace(",", " ") if pd.notna(x) else "—")
            )

    # проценты с 2 знаками (и знаком + при росте)
            display_table["Изменение оборота, %"] = display_table["Изменение оборота, %"].map(
                lambda x: ("—" if pd.isna(x) else f"{x:+.2f}%")
            )

            st.dataframe(display_table, use_container_width=True, hide_index=True)


        # --- TAB 2: Логарифмический график оборота (value) ---
        with tab_log:
            val_pos = vol_df[vol_df["value"] > 0].copy()
            if val_pos.empty:
                st.warning("Для логарифмической шкалы нужны положительные значения value > 0.")
                st.stop()

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
            st.plotly_chart(fig_val, use_container_width=True)

        # --- TAB 3: Гистограмма (горизонтальная) суммарного оборота за окно ---
        with tab_hist:
            sum_df = (
                vol_df.groupby(["label", "fund", "isin"], as_index=False)
                      .agg(value_sum=("value", "sum"))
                      .sort_values("value_sum", ascending=True)
            )

        with tab_hist:
            sum_df = (
                vol_df.groupby(["label", "fund", "isin"], as_index=False)
                      .agg(value_sum=("value", "sum"))
                      .sort_values("value_sum", ascending=True)
            )

            if sum_df.empty:
                st.info("Нет данных для гистограммы.")
                st.stop()

    # 1) Выбираем единицу (млрд/млн/тыс) по максимуму
            max_val = float(sum_df["value_sum"].max())
            if max_val >= 1e9:
                divisor = 1e9
                unit = "млрд руб"
                decimals = 2
            elif max_val >= 1e6:
                divisor = 1e6
                unit = "млн руб"
                decimals = 2
            elif max_val >= 1e3:
                divisor = 1e3
                unit = "тыс руб"
                decimals = 0
            else:
                divisor = 1.0
                unit = "руб"
                decimals = 0

            sum_df["value_sum_unit"] = sum_df["value_sum"] / divisor

    # 2) Строка для рублевого значения с пробелами
            sum_df["value_sum_rub_fmt"] = sum_df["value_sum"].map(
                lambda v: f"{v:,.0f}".replace(",", " ") if pd.notna(v) else "—"
            )

    # 3) Период (число торговых дней) как колонка, чтобы можно было использовать в custom_data
            sum_df["window_days"] = int(window)

            fig_hist = px.bar(
                sum_df,
                x="value_sum_unit",
                y="label",
                orientation="h",
                custom_data=["fund", "isin", "value_sum_rub_fmt", "window_days"],
                labels={"value_sum_unit": f"Суммарный оборот за период, {unit}", "label": "Фонд"},
                color_discrete_sequence=["red"],
                text="value_sum_unit",
            )

    # десятичная ".", тысячи " "
            fig_hist.update_layout(separators=". ")

    # Отключаем SI-формат (B/M) и задаем нормальные подписи
            fig_hist.update_xaxes(tickformat=f",.{decimals}f")

            fig_hist.update_traces(
                texttemplate=f"%{{x:,.{decimals}f}} {unit}",
                textposition="outside",
                hovertemplate=(
                    "Фонд: %{y}<br>"
                    "Период: %{customdata[3]} торговых дней<br>"
                    f"Суммарный оборот за период: %{{x:,.{decimals}f}} {unit}<br>"
            
                    "<extra></extra>"
                ),
            )

            st.plotly_chart(fig_hist, use_container_width=True)

    # -------- 7c) Средний размер сделки: ТАБЛИЦА (изменение за выбранное окно) --------
        st.subheader("Изменение среднего размера сделки (руб/сделку) за выбранный период")
        st.caption(f"Период: {start_date} — {end_date} (торговых дней в окне: {window})")

        avg_df = df_sel[(df_sel["tradedate"] >= start_date) & (df_sel["tradedate"] <= end_date)].copy()
        avg_df = avg_df.dropna(subset=["value", "numtrades"]).copy()

# схлопываем повторы на фонд/дату
        avg_df = (
            avg_df.sort_values(["label", "tradedate"])
                  .groupby(["label", "fund", "isin", "tradedate"], as_index=False)
                  .agg(value=("value", "sum"), numtrades=("numtrades", "sum"))
        )

# средний размер сделки (руб/сделку) по каждой дате
        avg_df = avg_df[avg_df["numtrades"] > 0].copy()
        avg_df["avg_trade_rub"] = avg_df["value"] / avg_df["numtrades"]

        if avg_df.empty:
            st.info("Нет данных для расчета среднего размера сделки (value/numtrades) в выбранном окне.")
        else:
    # Для каждого фонда берем первую и последнюю доступную дату ВНУТРИ ОКНА
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

    # изменения за окно: конец окна (последняя доступная дата) vs начало окна (первая доступная дата)
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
            })

            out = out.sort_values("Изменение, %", ascending=False, na_position="last")

    # форматирование: пробелы как разделитель тысяч
            display = out.copy()
            display["Средний размер сделки (начало выбранного периода), руб/сделку"] = display["Средний размер сделки (начало выбранного периода), руб/сделку"].map(
                lambda x: f"{x:,.0f}".replace(",", " ") if pd.notna(x) else "—"
            )
            display["Средний размер сделки (конец выбранного периода), руб/сделку"] = display["Средний размер сделки (конец выбранного периода), руб/сделку"].map(
                lambda x: f"{x:,.0f}".replace(",", " ") if pd.notna(x) else "—"
            )
            display["Изменение, руб"] = display["Изменение, руб"].map(
                lambda x: "—" if pd.isna(x) else f"{x:+,.0f}".replace(",", " ")
            )
            display["Изменение, %"] = display["Изменение, %"].map(
                lambda x: "—" if pd.isna(x) else f"{x:+.2f}%"
            )

            st.dataframe(display, use_container_width=True, hide_index=True)

        # мосбиржа

        ISS_BASE = "https://iss.moex.com/iss"

        from datetime import timedelta

        INDEX_MAP = {
            "RGBI": "RGBI",
            "RGBITR": "RGBITR",
            "RUCBCPNS": "RUCBCPNS",
            "RUCBTRNS": "RUCBTRNS",
            "RUSFAR": "RUSFAR",
            "CREI": "CREI",
            "MREF": "MREF",
        }

        def _iss_get(url: str, params: dict | None = None) -> dict:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()

        @st.cache_data(ttl=24 * 60 * 60)
        def resolve_board(secid: str) -> tuple[str, str, str]:
            j = _iss_get(f"{ISS_BASE}/securities/{secid}.json", params={"iss.meta": "off", "iss.only": "boards"})
            boards = pd.DataFrame(j["boards"]["data"], columns=j["boards"]["columns"])

    # наиболее частый случай: engine=stock, market=index, is_traded=1
            if "is_traded" in boards.columns:
                boards = boards.sort_values("is_traded", ascending=False)

            cand = boards[boards.get("engine").astype(str).eq("stock")] if "engine" in boards.columns else boards
            pref = cand[cand.get("market").astype(str).eq("index")] if "market" in cand.columns else cand

            pick = pref.iloc[0] if len(pref) else cand.iloc[0]
            return str(pick["engine"]), str(pick["market"]), str(pick["boardid"])

        @st.cache_data(ttl=24 * 60 * 60)
        def load_index_candles(secid: str, d_from: date, d_to: date) -> pd.DataFrame:
            engine, market, board = resolve_board(secid)
            url = f"{ISS_BASE}/engines/{engine}/markets/{market}/boards/{board}/securities/{secid}/candles.json"

            frames = []
            start = 0

            while True:
                j = _iss_get(url, params={
                    "from": d_from.isoformat(),
                    "till": d_to.isoformat(),
                    "interval": 24,         # дневные свечи
                    "start": start,         # пагинация
                    "iss.meta": "off",
                })

                candles_block = j.get("candles", {})
                data = candles_block.get("data", [])
                cols = candles_block.get("columns", [])

                if not data or not cols:
                    break

                part = pd.DataFrame(data, columns=cols)
                if part.empty:
                    break

                frames.append(part)
                got = len(part)
                if got < 100:
                    break

                start += got

            if not frames:
                return pd.DataFrame(columns=["secid", "tradedate", "close"])

            candles = pd.concat(frames, ignore_index=True)

    # типичные поля: begin, open, close, high, low, value, volume
            if "begin" not in candles.columns or "close" not in candles.columns:
                return pd.DataFrame(columns=["secid", "tradedate", "close"])

            candles["tradedate"] = pd.to_datetime(candles["begin"], errors="coerce").dt.date
            candles["close"] = pd.to_numeric(candles["close"], errors="coerce")

            out = candles[["tradedate", "close"]].dropna().copy()
            out["secid"] = secid
            return out

        st.subheader("Индексы Московской биржи за выбранный период")

        IDX_KEY = "idx_select"
        idx_selected = st.multiselect(
            "Выберите индексы",
            options=list(INDEX_MAP.keys()),
            default=["RGBI", "RGBITR", "RUCBCPNS", "RUCBTRNS", "RUSFAR", "CREI", "MREF"],
            key=IDX_KEY,
        )

        if not idx_selected:
            st.info("Выберите хотя бы один индекс.")
            st.stop()

# --- грузим данные по выбранным индексам в idx_df ---
        errors = {}
        idx_frames = []

        idx_to_full = date.today()
        idx_from_full = idx_to_full - timedelta(days=365 * 5)  # 5 лет

        for secid in idx_selected:
            try:
                tmp = load_index_candles(secid, idx_from_full, idx_to_full)
                if tmp.empty:
                    errors[secid] = "пустои ответ candles за выбранный диапазон"
                else:
                    idx_frames.append(tmp)
            except Exception as e:
                errors[secid] = str(e)

        if errors:
            st.warning(
                "Не удалось загрузить часть индексов:\n" +
                "\n".join([f"{k}: {v}" for k, v in errors.items()])
            )

        if not idx_frames:
            st.info("Нет данных по выбранным индексам за доступный период.")
            st.stop()

        idx_df = pd.concat(idx_frames, ignore_index=True)
        idx_df = idx_df.dropna(subset=["tradedate", "close"]).copy()

# метка для легенды/цветов
        idx_df["label"] = idx_df["secid"]

# Даты по выбранным индексам
        idx_dates = sorted(idx_df["tradedate"].unique().tolist())
        if len(idx_dates) < 2:
            st.warning("Недостаточно дат по выбранным индексам, чтобы построить период.")
            st.stop()

# --- отдельный выбор периода для индексов ---
        idx_end_date = st.select_slider(
            "Конечная дата (индексы)",
            options=idx_dates,
            value=idx_dates[-1],
            key="idx_end_date",
        )

        idx_max_window = min(252, len(idx_dates))
        idx_window = st.slider(
            "Длина периода (торговые дни) (индексы)",
            min_value=2,
            max_value=idx_max_window,
            value=min(30, idx_max_window),
            step=1,
            key="idx_window",
        )

        idx_date_to_i = {d: i for i, d in enumerate(idx_dates)}
        idx_end_i = idx_date_to_i[idx_end_date]
        idx_start_i = max(0, idx_end_i - idx_window + 1)
        idx_start_date = idx_dates[idx_start_i]

        st.caption(f"Период: {idx_start_date} — {idx_end_date} (торговых дней: {idx_window})")

# фильтр по окну индексов
        idx_period = idx_df[
            (idx_df["tradedate"] >= idx_start_date) &
            (idx_df["tradedate"] <= idx_end_date)
        ].copy()

# схлопывание дублеи по дате (если есть)
        idx_period = (
            idx_period.sort_values(["secid", "tradedate"])
                      .groupby(["secid", "label", "tradedate"], as_index=False)
                      .agg(close=("close", "last"))
        )

        fig_idx = px.line(
            idx_period,
            x="tradedate",
            y="close",
            color="label",
            markers=True,
            labels={"close": "Значение индекса", "tradedate": "Дата"},
            custom_data=["secid"],
        )

        fig_idx.update_layout(separators=". ")
        fig_idx.update_traces(
            hovertemplate=(
                "Дата: %{x|%Y-%m-%d}<br>"
                "Индекс: %{customdata[0]}<br>"
                "Значение: %{y:,.2f}<br>"
                "<extra></extra>"
            )
        )

        st.plotly_chart(fig_idx, use_container_width=True)

# ---------- РЕЖИМ 2: СРАВНЕНИЕ (сегодня vs предыдущий торговыи день) ----------
else:
    st.subheader("Изменение цены закрытия, средневзвешенная цена")

    # 3) Таблица: изменение close 
    cmp_df = df_sel.dropna(subset=["close"]).copy()

    cmp_df = (
        cmp_df.sort_values(["label", "tradedate"])
              .groupby(["label", "fund", "isin", "tradedate"], as_index=False)
              .agg(
                  close=("close", "last"),
                  waprice=("waprice", "last"),
              )
    )
    
    all_dates = sorted(cmp_df["tradedate"].dropna().unique().tolist())
    if len(all_dates) < 2:
        st.info("Недостаточно торговых дат для сравнения (нужно минимум 2).")
        st.stop()

# Ползунок: выбираем ЛЮБУЮ дату, кроме самой первой (иначе не с чем сравнивать)
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

# Берем значения на выбранную дату и на предыдущую
    last_df = cmp_df[cmp_df["tradedate"] == sel_date][["label", "fund", "isin", "close", "waprice"]].copy()
    last_df = last_df.rename(columns={
        "close": "close_last",
        "waprice": "waprice_last",
    })

    prev_df = cmp_df[cmp_df["tradedate"] == prev_date][["label", "close", "waprice"]].copy()
    prev_df = prev_df.rename(columns={
        "close": "close_prev",
        "waprice": "waprice_prev",
    })

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
        "waprice_last", "waprice_prev", "waprice_change_pct"
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
    })

    out = out.sort_values("Изменение цены закрытия, %", ascending=False, na_position="last")

    display = out.copy()
    display["Последняя цена закрытия, руб"] = display["Последняя цена закрытия, руб"].map(
        lambda x: f"{x:,.2f}".replace(",", " ") if pd.notna(x) else "—"
    )
    display["Предыдущая цена закрытия, руб"] = display["Предыдущая цена закрытия, руб"].map(
        lambda x: f"{x:,.2f}".replace(",", " ") if pd.notna(x) else "—"
    )
    display["Последняя средневзвешенная цена, руб"] = display["Последняя средневзвешенная цена, руб"].map(
        lambda x: f"{x:,.2f}".replace(",", " ") if pd.notna(x) else "—"
    )
    display["Предыдущая средневзвешенная цена, руб"] = display["Предыдущая средневзвешенная цена, руб"].map(
        lambda x: f"{x:,.2f}".replace(",", " ") if pd.notna(x) else "—"
    )
    display["Изменение цены закрытия, %"] = display["Изменение цены закрытия, %"].map(
        lambda x: "—" if pd.isna(x) else f"{x:+.2f}%"
    )
    display["Изменение средневзвешенной цены, %"] = display["Изменение средневзвешенной цены, %"].map(
        lambda x: "—" if pd.isna(x) else f"{x:+.2f}%"
    )

    st.dataframe(display, use_container_width=True, hide_index=True)

st.caption(f"Период загрузки: {date_from} — {date_to} (UTC). Кеш обновляется раз в сутки.")

