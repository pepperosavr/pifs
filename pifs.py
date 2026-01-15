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
    "RU000A105328": "ПАРУС-ЛОГ",
    "RU000A1068X9": "ПАРУС-ДВН",
    "RU000A108UH0": "ПАРУС-КРАС",
    "RU000A108VR7": "ПАРУС-МАКС",
    "RU000A104KU3": "ПАРУС-НОРД",
    "RU000A1022Z1": "ПАРУС-ОЗН",
    "RU000A104172": "ПАРУС-СБЛ",
    "RU000A108BZ2": "ПАРУС-ТРМ",
    "RU000A10CFM8": "ПАРУС-ЗОЛЯ",
    "RU000A100WZ5": "АКЦЕНТ IV",
    "RU000A10DQF7": "Акцент 5",
    "RU000A10A117": "ЗПИФ СМЛТ",
    "RU000A1099U0": "ЗПИФСовр 9",
}

# ВАЖНО: некоторым фондам в API нужно передавать не ISIN, а торговыи код MOEX (SECID/ticker).
# Для "Акцент 5" это XACCSK при ISIN RU000A10DQF7. :contentReference[oaicite:1]{index=1}
ISIN_TO_MOEX_CODE = {
    "RU000A10DQF7": "XACCSK",  # Акцент 5
}

# То, что хотим видеть в итоговых данных (ISIN-ы)
TARGET_ISINS = list(FUND_MAP.keys())

# То, что реально отправляем в API (MOEX-коды, где нужно; иначе ISIN как раньше)
ZPIF_SECIDS = [ISIN_TO_MOEX_CODE.get(isin, isin) for isin in TARGET_ISINS]

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
        return pd.DataFrame(columns=["shortname", "fund", "isin", "volume", "value", "numtrades", "close", "tradedate"])

    df = pd.DataFrame(all_results)[["shortname", "isin", "volume", "value", "numtrades", "close", "waprice", "tradedate"]]


    df = pd.DataFrame(all_results)[["shortname", "isin", "volume", "value", "numtrades", "close", "waprice", "tradedate"]]

    df["volume"]    = pd.to_numeric(df["volume"], errors="coerce")
    df["value"]     = pd.to_numeric(df["value"], errors="coerce")        # денежныи оборот
    df["numtrades"] = pd.to_numeric(df["numtrades"], errors="coerce")    # число сделок
    df["close"]     = pd.to_numeric(df["close"], errors="coerce")
    df["waprice"] = pd.to_numeric(df["waprice"], errors="coerce")

    df["tradedate"] = pd.to_datetime(df["tradedate"], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=["isin", "tradedate"])

    # имя фонда 
    df["fund"] = df["isin"].map(FUND_MAP).fillna(df["shortname"].astype(str))

    return df

# -----------------------
# Streamlit UI
# -----------------------
st.title("Торги ЗПИФ")

# 2) КНОПКА: История/Сравнение 
mode = st.radio(
    "Режим просмотра",
    options=["Режим истории", "Режим сравнения (сегодня vs предыдущий торговый день)"],
    horizontal=True,
)

# Период загрузки
utc_now = datetime.now(timezone.utc)
date_from = "2025-01-01T00:00:00Z"
date_to   = utc_now.strftime("%Y-%m-%dT23:59:59Z")

df = load_df(ZPIF_SECIDS, date_from, date_to)
# На всякии случаи: оставляем только целевые ISIN (если в ответе вдруг будут лишние инструменты)
df = df[df["isin"].isin(TARGET_ISINS)].copy()

if df.empty:
    st.warning("Данных не найдено за выбранныи период.")
    st.stop()

# Снапшот
out_dir = Path("snapshots")
out_dir.mkdir(parents=True, exist_ok=True)
snap_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
out_path = out_dir / f"zpif_history_{snap_date}.csv"
df.to_csv(out_path, index=False, encoding="utf-8")
print(f"Saved snapshot to: {out_path.resolve()}")

# Выбор фондов
available_funds = sorted(df["fund"].unique().tolist())
selected_funds = st.multiselect(
    "Выберите фонды",
    available_funds,
    default=available_funds,   # по умолчанию все 13
)

df_sel = df[df["fund"].isin(selected_funds)].copy()
df_sel = df_sel.sort_values(["tradedate", "fund"])

if df_sel.empty:
    st.warning("По выбранным фондам нет данных.")
    st.stop()

# Единая метка для графиков/таблиц
df_sel["label"] = df_sel["fund"].astype(str) + " (" + df_sel["isin"].astype(str) + ")"

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
        hover_data=["fund", "isin", "close", "volume", "value"],
        markers=True,
        labels={"close_change_pct": "Изменение цены закрытия, %", "tradedate": "Дата"},
)

# Показываем изменение цены как процент в hover
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
    st.plotly_chart(fig_close_pct, use_container_width=True)

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

# -------- 7c) Среднии размер сделки: ТАБЛИЦА (последняя дата vs предыдущая) --------
st.subheader("Среднии размер сделки (руб/сделку) — таблица изменений")
st.caption(f"Период: {start_date} — {end_date} (торговых дней в окне: {window})")

avg_df = df_sel[(df_sel["tradedate"] >= start_date) & (df_sel["tradedate"] <= end_date)].copy()
avg_df = avg_df.dropna(subset=["value", "numtrades"]).copy()

# схлопываем повторы на фонд/дату
avg_df = (
    avg_df.sort_values(["label", "tradedate"])
          .groupby(["label", "fund", "isin", "tradedate"], as_index=False)
          .agg(value=("value", "sum"), numtrades=("numtrades", "sum"))
)

# среднии размер сделки (руб/сделку)
avg_df = avg_df[avg_df["numtrades"] > 0].copy()
avg_df["avg_trade_rub"] = avg_df["value"] / avg_df["numtrades"]

if avg_df.empty:
    st.info("Нет данных для расчета среднего размера сделки (value/numtrades) в выбранном периоде.")
else:
    # торговые даты именно для avg_df (на случаи пропусков)
    avg_dates = sorted(avg_df["tradedate"].dropna().unique().tolist())
    if len(avg_dates) == 0:
        st.info("Нет торговых дат для расчета таблицы.")
        st.stop()

    last_date = avg_dates[-1]
    prev_date = avg_dates[-2] if len(avg_dates) >= 2 else None

    st.markdown(
        f"**Последняя дата:** {last_date}&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;"
        f"**Предыдущая дата:** {prev_date if prev_date is not None else '—'}"
    )

    # значения на последнюю дату
    last_df = avg_df[avg_df["tradedate"] == last_date][["label", "fund", "isin", "avg_trade_rub"]].copy()
    last_df = last_df.rename(columns={"avg_trade_rub": "avg_last"})

    # значения на предыдущую дату (если есть)
    if prev_date is not None:
        prev_df = avg_df[avg_df["tradedate"] == prev_date][["label", "avg_trade_rub"]].copy()
        prev_df = prev_df.rename(columns={"avg_trade_rub": "avg_prev"})
    else:
        prev_df = pd.DataFrame(columns=["label", "avg_prev"])

    out = last_df.merge(prev_df, on="label", how="left")

    # абсолютное изменение
    out["change_abs"] = out["avg_last"] - out["avg_prev"]

    # процентное изменение
    out["change_pct"] = np.where(
        (out["avg_prev"].notna()) & (out["avg_prev"] > 0),
        (out["avg_last"] / out["avg_prev"] - 1.0) * 100.0,
        np.nan,
    )

    # финальная таблица + порядок колонок
    out = out[["fund", "isin", "avg_last", "avg_prev", "change_abs", "change_pct"]].copy()
    out = out.rename(columns={
        "fund": "Фонд",
        "isin": "ISIN",
        "avg_last": "Последнии среднии размер сделки, руб/сделку",
        "avg_prev": "Предыдущии среднии размер сделки, руб/сделку",
        "change_abs": "Изменение, руб",
        "change_pct": "Изменение, %",
    })

    out = out.sort_values("Изменение, %", ascending=False, na_position="last")

    # форматирование (пробелы как разделитель тысяч)
    display = out.copy()

    display["Последнии среднии размер сделки, руб/сделку"] = display["Последнии среднии размер сделки, руб/сделку"].map(
        lambda x: f"{x:,.0f}".replace(",", " ") if pd.notna(x) else "—"
    )
    display["Предыдущии среднии размер сделки, руб/сделку"] = display["Предыдущии среднии размер сделки, руб/сделку"].map(
        lambda x: f"{x:,.0f}".replace(",", " ") if pd.notna(x) else "—"
    )
    display["Изменение, руб"] = display["Изменение, руб"].map(
        lambda x: "—" if pd.isna(x) else f"{x:+,.0f}".replace(",", " ")
    )
    display["Изменение, %"] = display["Изменение, %"].map(
        lambda x: "—" if pd.isna(x) else f"{x:+.2f}%"
    )

    st.dataframe(display, use_container_width=True, hide_index=True)

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
    if len(all_dates) == 0:
        st.info("Нет торговых дат для расчета сравнения.")
        st.stop()

    last_date = all_dates[-1]
    prev_date = all_dates[-2] if len(all_dates) >= 2 else None

    st.markdown(
        f"**Последняя дата:** {last_date}&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;"
        f"**Предыдущая дата:** {prev_date if prev_date is not None else '—'}"
    )

    def _last_prev(group: pd.DataFrame) -> pd.Series:
        g = group.sort_values("tradedate")
        last = g.iloc[-1]
        if len(g) >= 2:
            prev = g.iloc[-2]
            return pd.Series({
                "date_last": last["tradedate"],
                "close_last": last["close"],
                "waprice_last": last["waprice"],
                "date_prev": prev["tradedate"],
                "close_prev": prev["close"],
                "waprice_prev": prev["waprice"],
            })
        return pd.Series({
            "date_last": last["tradedate"],
            "close_last": last["close"],
            "waprice_last": last["waprice"],
            "date_prev": pd.NaT,
            "close_prev": np.nan,
            "waprice_prev": np.nan,
        })

    summary = (
        cmp_df.groupby(["label", "fund", "isin"], as_index=False)
              .apply(_last_prev, include_groups=False)
    )

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
        "fund", "isin", "close_prev", "close_last", "change_pct",
        "waprice_prev", "waprice_last",
        "waprice_change_pct"
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

# сортируем по реально существующей колонке
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
# проценты
    display["Изменение цены закрытия, %"] = display["Изменение цены закрытия, %"].map(
        lambda x: "—" if pd.isna(x) else f"{x:+.2f}%"
    )

    display["Изменение средневзвешенной цены, %"] = display["Изменение средневзвешенной цены, %"].map(
        lambda x: "—" if pd.isna(x) else f"{x:+.2f}%"
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

st.caption(f"Период загрузки: {date_from} — {date_to} (UTC). Кеш обновляется раз в сутки.")

