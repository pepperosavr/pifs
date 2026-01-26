import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import date, datetime
from typing import Dict, Tuple, List
import os

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

# =========================
# Настройки приложения
# =========================
st.set_page_config(page_title="MOEX индексы: % изменение с 2010", layout="wide")
st.title("Индексы Московской биржи: процентное изменение (с 2010 года)")

ISS_BASE = "https://iss.moex.com/iss"

INDEX_MAP: Dict[str, str] = {
    "RGBI": "RGBI",
    "RGBITR": "RGBITR",
    "RUCBCPNS": "RUCBCPNS",
    "RUCBTRNS": "RUCBTRNS",
    "RUSFAR": "RUSFAR",
    "CREI": "CREI",
    "MREF": "MREF",
    "MREDC": "MREDC",
}

DEFAULT_FROM = date(2010, 1, 1)

# =========================
# ISS helpers
# =========================
@st.cache_data(ttl=24 * 60 * 60)
def _iss_get(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=24 * 60 * 60)
def resolve_board(secid: str) -> Tuple[str, str, str]:
    """
    Подбирает наиболее подходящии режим торгов для индекса.
    Логика: engine=stock, market=index; приоритет is_traded=1, если есть.
    """
    j = _iss_get(
        f"{ISS_BASE}/securities/{secid}.json",
        params={"iss.meta": "off", "iss.only": "boards"},
    )
    boards = pd.DataFrame(j["boards"]["data"], columns=j["boards"]["columns"])
    if boards.empty:
        raise RuntimeError(f"Не нашел boards для {secid}")

    if "is_traded" in boards.columns:
        boards = boards.sort_values("is_traded", ascending=False)

    cand = boards[boards.get("engine").astype(str).eq("stock")] if "engine" in boards.columns else boards
    pref = cand[cand.get("market").astype(str).eq("index")] if "market" in cand.columns else cand

    pick = pref.iloc[0] if len(pref) else cand.iloc[0]
    return str(pick["engine"]), str(pick["market"]), str(pick["boardid"])

@st.cache_data(ttl=24 * 60 * 60)
def load_index_candles_daily(secid: str, d_from: date, d_to: date) -> pd.DataFrame:
    """
    Грузит дневные свечи (interval=24) с пагинациеи start.
    Возвращает tradedate (date), close (float), secid.
    """
    engine, market, board = resolve_board(secid)
    url = f"{ISS_BASE}/engines/{engine}/markets/{market}/boards/{board}/securities/{secid}/candles.json"

    frames: List[pd.DataFrame] = []
    start = 0

    while True:
        j = _iss_get(
            url,
            params={
                "from": d_from.isoformat(),
                "till": d_to.isoformat(),
                "interval": 24,      # дневные свечи
                "start": start,      # пагинация
                "iss.meta": "off",
            },
        )

        block = j.get("candles", {})
        data = block.get("data", [])
        cols = block.get("columns", [])

        if not data or not cols:
            break

        part = pd.DataFrame(data, columns=cols)
        if part.empty:
            break

        frames.append(part)
        got = len(part)

        # у ISS типично порции по 100 строк; если пришло меньше, значит конец
        if got < 100:
            break

        start += got

    if not frames:
        return pd.DataFrame(columns=["secid", "tradedate", "close"])

    candles = pd.concat(frames, ignore_index=True)

    if "begin" not in candles.columns or "close" not in candles.columns:
        return pd.DataFrame(columns=["secid", "tradedate", "close"])

    tradedt = pd.to_datetime(candles["begin"], errors="coerce")
    close = pd.to_numeric(candles["close"], errors="coerce")

    out = pd.DataFrame({"tradedate": tradedt.dt.date, "close": close})
    out = out.dropna(subset=["tradedate", "close"]).copy()
    out["secid"] = secid

    # схлопнем возможные дубли дат (берем last)
    out = (
        out.sort_values(["secid", "tradedate"])
           .groupby(["secid", "tradedate"], as_index=False)
           .agg(close=("close", "last"))
    )
    return out

@st.cache_data(ttl=24 * 60 * 60)
def load_index_history_from_2010(secid: str) -> pd.DataFrame:
    """
    Единая функция для кеша: одна загрузка "с 2010 до сегодня" на индекс.
    Далее в интерфеисе режем период без повторных запросов.
    """
    return load_index_candles_daily(secid, DEFAULT_FROM, date.today())

ACCENT_SERIES = {
    # Акцент IV: торгуется по ISIN
    "RU000A100WZ5": ("ACCENT_IV", "Акцент IV"),
    # Акцент 5: часто нужен тикер XACCSK, но иногда в ответах встречается и ISIN
    "XACCSK": ("ACCENT_5", "Акцент 5"),
    "RU000A10DQF7": ("ACCENT_5", "Акцент 5"),
}

ACCENT_INSTRUMENTS = ["RU000A100WZ5", "XACCSK"]

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
        yield lst[i:i+size]

def fetch_all_trading_results(token: str, instruments: list[str], date_from: str, date_to: str, page_size: int = 100):
    url = f"{API_URL}/Moex/History"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    out = []
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
        out.extend(page_data)
        page += 1
        if len(page_data) < page_size:
            break
    return out

def _to_api_dt(d: date, end_of_day: bool = False) -> str:
    return f"{d:%Y-%m-%d}T23:59:59Z" if end_of_day else f"{d:%Y-%m-%d}T00:00:00Z"

@st.cache_data(ttl=24 * 60 * 60)
def load_accent_close(d_from: date, d_to: date) -> pd.DataFrame:
    """
    Возвращает tradedate, close, secid (серии), label.
    secid тут делаем искусственным: ACCENT_IV / ACCENT_5,
    чтобы не конфликтовать с индексами и одинаково считать pct.
    """
    if not API_LOGIN or not API_PASS:
        # не падаем, а возвращаем пусто (на HF это типичная ситуация без secrets)
        return pd.DataFrame(columns=["secid", "label", "tradedate", "close"])

    token = get_token(API_LOGIN, API_PASS)
    if not token:
        raise RuntimeError("Не удалось получить токен (проверьте API_LOGIN/API_PASS)")

    date_from = _to_api_dt(d_from, end_of_day=False)
    date_to   = _to_api_dt(d_to, end_of_day=True)

    all_rows = []
    for chunk in chunk_list(ACCENT_INSTRUMENTS, 30):
        all_rows.extend(fetch_all_trading_results(token, list(chunk), date_from, date_to))

    if not all_rows:
        return pd.DataFrame(columns=["secid", "label", "tradedate", "close"])

    raw = pd.DataFrame(all_rows)

    # гарантируем поля
    for c in ["secid", "isin", "tradedate", "close"]:
        if c not in raw.columns:
            raw[c] = np.nan

    raw["tradedate"] = pd.to_datetime(raw["tradedate"], errors="coerce", utc=True).dt.date
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")

    # выбираем идентификатор инструмента: secid приоритетнее, иначе isin
    raw["instr"] = raw["secid"].where(raw["secid"].notna(), raw["isin"])

    # маппим в серии Акцент
    raw["series_id"] = raw["instr"].map(lambda x: ACCENT_SERIES.get(str(x), (None, None))[0] if pd.notna(x) else None)
    raw["label"]     = raw["instr"].map(lambda x: ACCENT_SERIES.get(str(x), (None, None))[1] if pd.notna(x) else None)

    df = raw.dropna(subset=["tradedate", "close", "series_id"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["secid", "label", "tradedate", "close"])

    # схлопываем дубли по дате в серии
    df = (
        df.sort_values(["series_id", "tradedate"])
          .groupby(["series_id", "label", "tradedate"], as_index=False)
          .agg(close=("close", "last"))
    )
    df = df.rename(columns={"series_id": "secid"})
    return df[["secid", "label", "tradedate", "close"]]

def _resample_close(df_one: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    freq: 'D' (без ресемпла), 'W' (неделя), 'M' (месяц)
    Берем последнии close периода.
    """
    if df_one.empty:
        return df_one

    if freq == "D":
        return df_one

    d = df_one.copy()
    d["dt"] = pd.to_datetime(d["tradedate"], errors="coerce")
    d = d.dropna(subset=["dt"]).set_index("dt").sort_index()

    rule = {"W": "W-FRI", "M": "M"}[freq]
    d_rs = d["close"].resample(rule).last().dropna().reset_index()
    d_rs["tradedate"] = d_rs["dt"].dt.date
    out = d_rs[["tradedate", "close"]].copy()
    out["secid"] = df_one["secid"].iloc[0]
    return out

def _calc_metric(df_all: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode:
      - 'cumulative': накопленное изменение к первои точке выбранного периода (%)
      - 'daily': дневное изменение (%)
    """
    d = df_all.sort_values(["secid", "tradedate"]).copy()

    if mode == "cumulative":
        first_close = d.groupby("secid")["close"].transform("first")
        d["pct"] = (d["close"] / first_close - 1.0) * 100.0
    else:
        d["pct"] = d.groupby("secid")["close"].pct_change() * 100.0

    return d

# =========================
# UI: выбор параметров
# =========================
with st.sidebar:
    st.header("Параметры")

    idx_selected = st.multiselect(
        "Индексы",
        options=list(INDEX_MAP.keys()),
        default=list(INDEX_MAP.keys()),
    )

    # --- Кнопка-переключатель для рядов Акцент ---
    if "show_accent" not in st.session_state:
        st.session_state.show_accent = True  # по умолчанию показываем

    btn_label = "Убрать Акцент" if st.session_state.show_accent else "Добавить Акцент"
    if st.button(btn_label, use_container_width=True):
        st.session_state.show_accent = not st.session_state.show_accent

    show_accent = st.session_state.show_accent

    c1, c2 = st.columns(2)
    with c1:
        d_from = st.date_input("Начало периода", value=DEFAULT_FROM, min_value=DEFAULT_FROM)
    with c2:
        d_to = st.date_input("Конец периода", value=date.today(), min_value=DEFAULT_FROM)

    freq_label = st.selectbox(
        "Частота на графике",
        options=["День", "Неделя", "Месяц"],
        index=1,  # неделя по умолчанию, чтобы график был легче
    )
    freq = {"День": "D", "Неделя": "W", "Месяц": "M"}[freq_label]

    metric_label = st.selectbox(
        "Показатель",
        options=[
            "Накопленное изменение к первой точке периода, %",
            "Дневное изменение, %",
        ],
        index=0,
    )
    metric_mode = "cumulative" if metric_label.startswith("Накопленное") else "daily"

    show_markers = st.checkbox("Маркеры на линиях", value=False)

    if st.button("Сбросить кеш данных"):
        st.cache_data.clear()
        st.success("Кеш очищен.")

if not idx_selected:
    st.info("Выберите хотя бы один индекс.")
    st.stop()

if d_from > d_to:
    st.error("Некорректный период: начало позже конца.")
    st.stop()

# =========================
# Загрузка данных
# =========================
errors = {}
frames = []

progress = st.progress(0)
for i, secid in enumerate(idx_selected, start=1):
    try:
        hist = load_index_history_from_2010(secid)
        if hist.empty:
            errors[secid] = "пустой ответ candles"
        else:
            # режем нужныи период
            cut = hist[(hist["tradedate"] >= d_from) & (hist["tradedate"] <= d_to)].copy()
            cut = _resample_close(cut, freq=freq)
            if cut.empty or len(cut) < 2:
                errors[secid] = "недостаточно точек в выбранном периоде"
            else:
                frames.append(cut)
    except Exception as e:
        errors[secid] = str(e)

    progress.progress(int(i / len(idx_selected) * 100))

progress.empty()

if errors:
    st.warning(
        "Часть индексов не удалось отобразить:\n" +
        "\n".join([f"{k}: {v}" for k, v in errors.items()])
    )

if not frames:
    st.error("Нет данных для построения графика по выбранным параметрам.")
    st.stop()

# --- базовыи набор: индексы ---
idx_df = pd.concat(frames, ignore_index=True)
idx_df = idx_df.dropna(subset=["tradedate", "close"]).copy()
idx_df["label"] = idx_df["secid"]

series_df_raw = idx_df[["secid", "label", "tradedate", "close"]].copy()

# --- опционально добавляем Акцент ---
if show_accent:
    try:
        accent_df = load_accent_close(d_from, d_to)
        if accent_df.empty:
            st.warning("Акцент не добавлен: нет данных (или не настроены API_LOGIN/API_PASS).")
        else:
            # приводим к выбраннои частоте так же, как индексы
            acc_parts = []
            for sid in accent_df["secid"].unique().tolist():
                one = accent_df[accent_df["secid"] == sid].copy()
                one = _resample_close(one[["secid", "tradedate", "close"]], freq=freq)
                # восстановим label после ресемпла
                one["label"] = "Акцент IV" if sid == "ACCENT_IV" else "Акцент 5"
                acc_parts.append(one[["secid", "label", "tradedate", "close"]])
            accent_rs = pd.concat(acc_parts, ignore_index=True) if acc_parts else pd.DataFrame()

            if not accent_rs.empty:
                series_df_raw = pd.concat([series_df_raw, accent_rs], ignore_index=True)
    except Exception as e:
        st.warning(f"Акцент не добавлен из-за ошибки: {e}")

# --- расчет метрики уже по общему набору ---
series_df = _calc_metric(series_df_raw, mode=metric_mode)

# =========================
# График
# =========================
st.subheader("График")
st.caption(
    f"Период: {d_from} — {d_to}. Частота: {freq_label}. "
    f"Если индекс начал публиковаться позже 2010 года, базовая дата берется как первая доступная точка в выбранном периоде."
)

fig = px.line(
    series_df,
    x="tradedate",
    y="pct",
    color="label",
    markers=show_markers,
    labels={"tradedate": "Дата", "pct": "Изменение, %", "label": "Индекс"},
    custom_data=["secid", "close"],
)

fig.update_layout(separators=". ")
fig.update_traces(
    hovertemplate=(
        "Дата: %{x|%Y-%m-%d}<br>"
        "Индекс: %{customdata[0]}<br>"
        "Close: %{customdata[1]:,.2f}<br>"
        "Изменение: %{y:+.2f}%<br>"
        "<extra></extra>"
    )
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# Таблица: итоговые значения
# =========================
st.subheader("Итог по выбранному периоду")

tmp = series_df.sort_values(["secid", "tradedate"]).copy()
first = tmp.groupby("secid", as_index=False).first()[["secid", "tradedate", "close", "pct"]].rename(
    columns={"tradedate": "base_date", "close": "base_close", "pct": "base_pct"}
)
last = tmp.groupby("secid", as_index=False).last()[["secid", "tradedate", "close", "pct"]].rename(
    columns={"tradedate": "last_date", "close": "last_close", "pct": "last_pct"}
)

summary = first.merge(last, on="secid", how="inner")

if metric_mode == "cumulative":
    summary["result_pct"] = summary["last_pct"]
else:
    # для дневного изменения "итог" как последняя дневная доходность
    summary["result_pct"] = summary["last_pct"]

summary = summary.sort_values("result_pct", ascending=False, na_position="last")

disp = summary.copy()
disp["base_close"] = disp["base_close"].map(lambda x: "—" if pd.isna(x) else f"{x:,.2f}".replace(",", " "))
disp["last_close"] = disp["last_close"].map(lambda x: "—" if pd.isna(x) else f"{x:,.2f}".replace(",", " "))
disp["result_pct"] = disp["result_pct"].map(lambda x: "—" if pd.isna(x) else f"{x:+.2f}%")

disp = disp.rename(
    columns={
        "secid": "Индекс",
        "base_date": "Базовая дата",
        "base_close": "Индекс (базовая дата)",
        "last_date": "Конечная дата",
        "last_close": "Индекс (конечная дата)",
        "result_pct": "Изменение, %",
    }
)

st.dataframe(disp[["Индекс", "Базовая дата", "Индекс (базовая дата)", "Конечная дата", "Индекс (конечная дата)", "Изменение, %"]],
             use_container_width=True, hide_index=True)
