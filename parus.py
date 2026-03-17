import os
from io import BytesIO
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from typing import Optional, Dict, Any, List

import numpy as np
import openpyxl
import pandas as pd
import requests
import streamlit as st


# =========================
# Конфигурация
# =========================
API_URL = "https://dh2.efir-net.ru/v2"

ACCENT_IV_ISIN = "RU000A100WZ5"
ACCENT_5_ISIN = "RU000A10DQF7"

# Акцент 5 слать тикером
ISIN_TO_MOEX_CODE = {
    ACCENT_5_ISIN: "XACCSK",
}
MOEX_CODE_TO_ISIN = {v: k for k, v in ISIN_TO_MOEX_CODE.items()}

ACCENT_TARGET_ISINS = [ACCENT_IV_ISIN, ACCENT_5_ISIN]
ACCENT_INSTRUMENTS_FOR_API = [ACCENT_IV_ISIN, ISIN_TO_MOEX_CODE[ACCENT_5_ISIN]]

FUND_NAME_BY_ISIN = {
    ACCENT_IV_ISIN: "АКЦЕНТ IV",
    ACCENT_5_ISIN: "Акцент 5",
}

st.set_page_config(page_title="Акцент IV/5: дневные торги", layout="wide")
st.title("Таблица торгов по дням: Акцент IV и Акцент 5")


# =========================
# Secrets / Env
# =========================
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

if not API_LOGIN or not API_PASS:
    st.error("Не заданы API_LOGIN / API_PASS. Добавьте их в Secrets или переменные окружения.")
    st.stop()


# =========================
# Утилиты агрегации дублей
# =========================
def _sum_or_single(s: pd.Series, decimals: int = 0) -> float:
    """
    Если значения одинаковые (после округления), считаем дублем и берем одно.
    Иначе суммируем.
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
def do_post_request(url: str, body: Dict[str, Any], token: Optional[str]) -> Any:
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
    page_size: int = 100,
) -> List[Dict[str, Any]]:
    url = f"{API_URL}/Moex/History"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    out: List[Dict[str, Any]] = []

    requests_plan = [
        {
            "market": "shares",
            "boardid": ["TQIF"],
            "mode_label": "Основной режим",
        },
        {
            "market": "ndm",
            "boardid": None,
            "mode_label": "РПС",
        },
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


def _to_api_dt(d: date, end_of_day: bool) -> str:
    if end_of_day:
        return f"{d:%Y-%m-%d}T23:59:59Z"
    return f"{d:%Y-%m-%d}T00:00:00Z"

def _week_monday(d: date) -> date:
    return date.fromordinal(d.toordinal() - d.weekday())

def calc_period_window(end_d: date, period_kind: str) -> tuple[date, date]:
    """
    Скользящее окно, привязанное к end_d (включительно).
    Неделя: 7 днеи
    Месяц: 1 месяц назад
    Квартал: 3 месяца назад
    Год: 1 год назад
    """
    if period_kind == "Неделя":
        start_d = end_d - timedelta(days=6)
        return start_d, end_d

    if period_kind == "Месяц":
        start_d = (end_d - relativedelta(months=1)) + timedelta(days=1)
        return start_d, end_d

    if period_kind == "Квартал":
        start_d = (end_d - relativedelta(months=3)) + timedelta(days=1)
        return start_d, end_d

    if period_kind == "Год":
        start_d = (end_d - relativedelta(years=1)) + timedelta(days=1)
        return start_d, end_d

    raise ValueError(f"Неизвестный период: {period_kind}")

def step_for_period(period_kind: str) -> relativedelta | timedelta:
    if period_kind == "Неделя":
        return timedelta(days=7)
    if period_kind == "Месяц":
        return relativedelta(months=1)
    if period_kind == "Квартал":
        return relativedelta(months=3)
    if period_kind == "Год":
        return relativedelta(years=1)
    return timedelta(days=0)


# =========================
# Загрузка сырых данных
# =========================
@st.cache_data(ttl=24 * 60 * 60)
def load_accent_raw(d_from: date, d_to: date) -> pd.DataFrame:
    token = get_token(API_LOGIN, API_PASS)

    date_from = _to_api_dt(d_from, end_of_day=False)
    date_to = _to_api_dt(d_to, end_of_day=True)

    all_rows = fetch_moex_history(
        token=token,
        instruments=ACCENT_INSTRUMENTS_FOR_API,
        date_from=date_from,
        date_to=date_to,
        page_size=100,
    )

    if not all_rows:
        return pd.DataFrame()

    raw = pd.DataFrame(all_rows)
    raw = raw.drop_duplicates().copy()

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

    # восстановление ISIN, если пришел тикер
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


# =========================
# Схлопывание дублей -> дневная таблица
# =========================

def build_accent_daily_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    1 строка на (Дата, ISIN, Фонд, Режим торгов).
    Основной режим и РПС не суммируются между собой.
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    d = df_raw.copy()
    d = d.drop_duplicates().copy()

    d = d.sort_values(
        ["isin", "tradedate", "mode", "secid", "boardid"],
        na_position="last"
    )

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
        .rename(
            columns={
                "tradedate": "Дата",
                "isin": "ISIN",
                "fund": "Фонд",
                "mode": "Режим торгов",
            }
        )
        .sort_values(["Дата", "Фонд", "Режим торгов"])
    )

    # если pandas все же создал технические колонки — убираем их
    out = out.drop(columns=["index", "level_0", "level_1"], errors="ignore")

    # чистый индекс
    out = out.reset_index(drop=True)

    for c in ["Open", "High", "Low", "Close", "Средняя цена (waprice)"]:
        out[c] = out[c].round(2)

    for c in ["Рубли (volume*waprice)", "Рубли (close*volume)", "Рубли как в API (value)"]:
        out[c] = out[c].round(0)

    return out

def build_period_summary(df_raw: pd.DataFrame, period_kind: str) -> pd.DataFrame:
    
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    d = df_raw.copy()

    dt = pd.to_datetime(d["tradedate"], errors="coerce")
    d = d.dropna(subset=["fund", "mode", "value", "volume", "numtrades", "tradedate"]).copy()
    dt = pd.to_datetime(d["tradedate"], errors="coerce")

    if period_kind == "Неделя":
        d["period_start"] = (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.date
        d["period_end"] = (pd.to_datetime(d["period_start"]) + pd.to_timedelta(6, unit="D")).dt.date
    elif period_kind == "Месяц":
        d["period_start"] = dt.dt.to_period("M").dt.start_time.dt.date
        d["period_end"] = (pd.to_datetime(d["period_start"]) + pd.offsets.MonthEnd(0)).dt.date
    elif period_kind == "Квартал":
        d["period_start"] = dt.dt.to_period("Q").dt.start_time.dt.date
        d["period_end"] = (pd.to_datetime(d["period_start"]) + pd.offsets.QuarterEnd(0)).dt.date
    elif period_kind == "Год":
        d["period_start"] = dt.dt.to_period("Y").dt.start_time.dt.date
        d["period_end"] = (pd.to_datetime(d["period_start"]) + pd.offsets.YearEnd(0)).dt.date
    else:
        raise ValueError(f"Неизвестный период: {period_kind}")

    grp = (
        d.groupby(["period_start", "period_end", "fund", "mode"], as_index=False)
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

    return grp.sort_values(["period_start", "Фонд", "Режим торгов"], ascending=[False, True, True])


def pivot_period_summary(period_df: pd.DataFrame) -> pd.DataFrame:
    if period_df is None or period_df.empty:
        return pd.DataFrame()

    metrics = ["Кол-во бумаг, шт", "Оборот, руб", "Сделок, шт"]

    # 1) Определяем, какие колонки периода есть в данных
    if {"period_start", "period_end"}.issubset(period_df.columns):
        idx_cols = ["period_start", "period_end", "Фонд"]
        rename_after = {"period_start": "Начало периода", "period_end": "Конец периода"}
    elif {"Начало периода", "Конец периода"}.issubset(period_df.columns):
        idx_cols = ["Начало периода", "Конец периода", "Фонд"]
        rename_after = {}
    else:
        raise KeyError(
            "В period_df нет колонок периода. Ожидаю либо "
            "period_start/period_end, либо Начало периода/Конец периода."
        )

    # 2) Pivot
    pv = period_df.pivot_table(
        index=idx_cols,
        columns="Режим торгов",
        values=metrics,
        aggfunc="sum",
        fill_value=0,
    )

    # 3) Выпрямляем MultiIndex колонок
    pv.columns = [f"{m} — {mode}" for m, mode in pv.columns]
    pv = pv.reset_index()

    # 4) Приводим названия к единому виду (если были period_start/period_end)
    if rename_after:
        pv = pv.rename(columns=rename_after)

    # 5) Итого = Основной режим + РПС
    for m in metrics:
        col_main = f"{m} — Основной режим"
        col_rps = f"{m} — РПС"
        if col_main not in pv.columns:
            pv[col_main] = 0
        if col_rps not in pv.columns:
            pv[col_rps] = 0
        pv[f"{m} — Итого"] = pv[col_main] + pv[col_rps]

    # 6) Красивый порядок колонок
    ordered = ["Начало периода", "Конец периода", "Фонд"]
    for m in metrics:
        ordered += [f"{m} — Основной режим", f"{m} — РПС", f"{m} — Итого"]

    ordered = [c for c in ordered if c in pv.columns]
    pv = (
        pv[ordered]
        .sort_values(["Начало периода", "Фонд"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return pv
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


# =========================
# UI: параметры
# =========================
with st.sidebar:
    st.header("Параметры")

    today = date.today()
    default_from = date(today.year, 1, 1)

    d_from = st.date_input("Начало", value=default_from)
    d_to = st.date_input("Конец", value=today)

    period_kind = st.radio(
        "Итоги: период отображения",
        options=["Неделя", "Месяц", "Квартал", "Год"],
        horizontal=True,
        index=0,
    )

    # --- конечная дата для скользящего окна ---
    MIN_END = date(2010, 1, 1)

    if "summary_end_input" not in st.session_state:
        st.session_state["summary_end_input"] = d_to

    # если пользователь поменял d_to (общий конец), поджимаем summary_end
    if st.session_state["summary_end_input"] > d_to:
        st.session_state["summary_end_input"] = d_to
    if st.session_state["summary_end_input"] < MIN_END:
        st.session_state["summary_end_input"] = MIN_END

    # сам виджет читает/пишет в session_state по своему key
    st.date_input(
        "Конец периода итогов",
        key="summary_end_input",
        min_value=MIN_END,
        max_value=d_to,
    )

    def _shift_summary_end(delta):
        cur = st.session_state["summary_end_input"]
        new_end = cur + delta
        if new_end < MIN_END:
            new_end = MIN_END
        if new_end > d_to:
            new_end = d_to
        st.session_state["summary_end_input"] = new_end

    c_prev, c_next = st.columns(2)
    step = step_for_period(period_kind)

    with c_prev:
        st.button(
            "← Предыдущий период",
            use_container_width=True,
            on_click=_shift_summary_end,
            args=(-step,),
        )

    with c_next:
        st.button(
            "Следующий период →",
            use_container_width=True,
            on_click=_shift_summary_end,
            args=(step,),
        )

    # окно итогов (скользящее)
    period_start, period_end = calc_period_window(st.session_state["summary_end_input"], period_kind)
    st.caption(f"Окно итогов: {period_start} — {period_end}")

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
    df_raw_period = load_accent_raw(period_start, period_end)  
    df_raw_day    = load_accent_raw(d_from, d_to)              

if df_raw_period.empty and df_raw_day.empty:
    st.warning("Нет данных в выбранном диапазоне.")
    st.stop()

# 1. Итоги за период (Неделя/Месяц/Квартал/Год)
st.subheader(f"Итоги за {period_kind.lower()}: Основной режим vs РПС")

period_long = build_range_summary(df_raw_period, period_start, period_end)
period_pivot = pivot_period_summary(period_long)

if period_pivot.empty:
    st.warning("Не удалось построить таблицу итогов за период.")
else:
    # --- выбор отображаемых колонок через кнопку ---
    with st.popover("Выбрать колонки для отображения"):
        all_cols = period_pivot.columns.tolist()

        # разумный дефолт: даты + фонд + оборот (3 колонки)
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
        )

    if not selected_cols:
        st.info("Выберите хотя бы одну колонку.")
        period_show = period_pivot.copy()
    else:
        period_show = period_pivot[selected_cols].copy()

    # формат чисел (без Styler тоже можно, но так удобнее)
    fmt = {}
    for c in period_show.columns:
        if "Кол-во бумаг" in c or "Сделок" in c:
            fmt[c] = "{:.0f}"
        if "Оборот" in c:
            fmt[c] = "{:.0f}"

    st.dataframe(
        period_show.style.format(fmt),
        use_container_width=True,
        hide_index=True,
    )

    # --- Excel сразу под таблицей итогов ---
    def _df_to_xlsx_bytes_single(df: pd.DataFrame, sheet_name: str) -> bytes:
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

    xlsx_period = _df_to_xlsx_bytes_single(period_show, sheet_name=f"Итоги_{period_kind}")
    st.download_button(
        f"Скачать Excel: итоги, {period_kind.lower()}",
        data=xlsx_period,
        file_name=f"accent_summary_{period_kind.lower()}_{d_from}_{d_to}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


# 2. Детальная дневная таблица

st.subheader("Детальные торги по дням")

accent_daily = build_accent_daily_table(df_raw_day)
accent_daily_show = pd.DataFrame()  # важно: создаем заранее

if not accent_daily.empty:
    mode_options = ["Все режимы"] + accent_daily["Режим торгов"].dropna().unique().tolist()

    selected_mode = st.radio(
        "Показывать в дневной таблице",
        options=mode_options,
        horizontal=True,
    )

    if selected_mode != "Все режимы":
        accent_daily_show = accent_daily[
            accent_daily["Режим торгов"] == selected_mode
        ].copy()
    else:
        accent_daily_show = accent_daily.copy()

    accent_daily_show = accent_daily_show.reset_index(drop=True)

    st.dataframe(
        accent_daily_show,
        use_container_width=True,
        hide_index=True,
        column_config={"_index": None},
    )
else:
    st.warning("Не удалось построить детальную таблицу.")

# --- Excel выгрузка: детальные торги ---
def _df_to_xlsx_bytes_single(df: pd.DataFrame, sheet_name: str) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        ws.freeze_panes = "A2"

        # автоширина колонок
        for col_cells in ws.columns:
            max_len = 0
            col_letter = col_cells[0].column_letter
            for cell in col_cells:
                v = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(v))
            ws.column_dimensions[col_letter].width = min(max_len + 2, 60)

    buf.seek(0)
    return buf.read()

# Кнопку показываем только если есть что выгружать
xlsx_daily = _df_to_xlsx_bytes_single(accent_daily_show, sheet_name="Детальные торги")
st.download_button(
    "Скачать Excel: детальные торги",
    data=xlsx_daily,
    file_name=f"accent_daily_{d_from}_{d_to}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)
