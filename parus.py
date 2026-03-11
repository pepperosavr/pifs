import os
from io import BytesIO
from datetime import date
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
    Основной режим и РПС НЕ суммируются между собой.
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    d = df_raw.copy()

    # на всякий случай снимаем полные дубли
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

        return pd.Series(
            {
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
            }
        )

    out = (
        d.groupby(["tradedate", "isin", "fund", "mode"], as_index=False)
        .apply(_agg, include_groups=False)
        .reset_index()
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

    for c in ["Open", "High", "Low", "Close", "Средняя цена (waprice)"]:
        out[c] = out[c].round(2)

    for c in ["Рубли (volume*waprice)", "Рубли (close*volume)", "Рубли как в API (value)"]:
        out[c] = out[c].round(0)

    return out


def build_weekly_summary(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    d = df_raw.copy()
    d["tradedate"] = pd.to_datetime(d["tradedate"])

    # дата начала недели = понедельник
    d["Начало недели"] = (
        d["tradedate"] - pd.to_timedelta(d["tradedate"].dt.weekday, unit="D")
    ).dt.date

    grp = (
        d.groupby(["Начало недели", "fund", "mode"], as_index=False)
        .agg(
            volume=("volume", "sum"),
            value=("value", "sum"),
            numtrades=("numtrades", "sum"),
        )
    )

    grp = grp.rename(
        columns={
            "fund": "Фонд",
            "mode": "Режим торгов",
            "volume": "Кол-во бумаг, шт",
            "value": "Оборот, руб",
            "numtrades": "Сделок, шт",
        }
    )

    return grp.sort_values(["Начало недели", "Фонд", "Режим торгов"], ascending=[False, True, True])


# =========================
# UI: параметры
# =========================
with st.sidebar:
    st.header("Параметры")

    today = date.today()
    default_from = date(today.year, 1, 1)

    d_from = st.date_input("Начало", value=default_from)
    d_to = st.date_input("Конец", value=today)

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

# 1. Таблица итогов за неделю
st.subheader("Итоги за неделю: Основной режим vs РПС")
weekly_df = build_weekly_summary(df_raw)

if not weekly_df.empty:
    st.dataframe(
        weekly_df.style.format(
            {
                "Кол-во бумаг, шт": "{:.0f}",
                "Оборот, руб": "{:.0f}",
                "Сделок, шт": "{:.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning("Не удалось построить недельную таблицу.")


# 2. Детальная дневная таблица

st.subheader("Детальные торги по дням")

accent_daily = build_accent_daily_table(df_raw)

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

    # ← ключевая строка
    accent_daily_show = accent_daily_show.reset_index(drop=True)

    st.dataframe(
        accent_daily_show,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning("Не удалось построить детальную таблицу.")


# =========================
# Выгрузка в Excel
# =========================
def df_to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "Accent_IV_5") -> bytes:
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


st.divider()

try:
    xlsx_bytes = df_to_xlsx_bytes(accent_daily)
    st.download_button(
        "Скачать Excel (.xlsx)",
        data=xlsx_bytes,
        file_name=f"accent_daily_{d_from}_{d_to}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
except Exception:
    st.warning("Excel выгрузка недоступна.")
    csv_bytes = accent_daily.to_csv(index=False, encoding="utf-8").encode("utf-8")
    st.download_button(
        "Скачать как CSV",
        data=csv_bytes,
        file_name="accent_daily.csv",
        use_container_width=True,
    )
