import math
from datetime import date
from typing import Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st

# =========================
# Настройки приложения
# =========================
st.set_page_config(
    page_title="Портфель российских индексов",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ISS_BASE = "https://iss.moex.com/iss"
RF = 0.16  # как в исходном HTML

# =========================
# Индексы и логика приложения
# =========================
INDEX_META: Dict[str, dict] = {
    "IMOEX": {
        "name": "Индекс МосБиржи",
        "aliases": ["IMOEX"],
        "color": "#c8a86b",
        "max": 80,
    },
    "RGBI": {
        "name": "Гос. облигации (ОФЗ)",
        "aliases": ["RGBI"],
        "color": "#64748b",
        "max": 80,
    },
    "MCFTR": {
        "name": "МосБиржа полной доходности",
        "aliases": ["MCFTR"],
        "color": "#f59e0b",
        "max": 80,
    },
    "RUCBTR": {
        "name": "Корп. облигации",
        "aliases": ["RUCBTR", "RUCBTRNS"]],  # fallback
        "color": "#94a3b8",
        "max": 80,
    },
    "MREF": {
        "name": "Складская и индустриальная недвижимость",
        "aliases": ["MREF"],
        "color": "#2dd4bf",
        "max": 40,
    },
}

BASE = ["IMOEX", "RGBI", "MCFTR", "RUCBTR"]
ALL = ["IMOEX", "RGBI", "MCFTR", "RUCBTR", "MREF"]

BASELINE = {
    "IMOEX": 40,
    "RGBI": 25,
    "MCFTR": 20,
    "RUCBTR": 15,
    "MREF": 0,
}

MREF_PORTFOLIO = {
    "IMOEX": 34,
    "RGBI": 21,
    "MCFTR": 17,
    "RUCBTR": 13,
    "MREF": 15,
}

METRIC_DEFS = [
    {
        "label": "Годовая доходность",
        "key": "ret",
        "mult": 100,
        "dec": 1,
        "suf": "%",
        "dir": 1,
        "gt": 12,
        "ot": 7,
        "desc": "Историческая CAGR по выбранному периоду",
    },
    {
        "label": "Волатильность",
        "key": "vol",
        "mult": 100,
        "dec": 1,
        "suf": "%",
        "dir": -1,
        "gt": 12,
        "ot": 18,
        "desc": "Годовая историческая волатильность",
    },
    {
        "label": "Коэф. Шарпа",
        "key": "shr",
        "mult": 1,
        "dec": 2,
        "suf": "",
        "dir": 1,
        "gt": 0.3,
        "ot": 0,
        "desc": "Доходность сверх 16% на единицу риска",
    },
    {
        "label": "Коэф. Сортино",
        "key": "sor",
        "mult": 1,
        "dec": 2,
        "suf": "",
        "dir": 1,
        "gt": 0.4,
        "ot": 0,
        "desc": "Коэффициент с учетом downside deviation",
    },
    {
        "label": "Макс. просадка",
        "key": "mdd",
        "mult": 100,
        "dec": 1,
        "suf": "%",
        "dir": -1,
        "gt": -15,
        "ot": -30,
        "desc": "Максимальная историческая просадка",
    },
    {
        "label": "Коэф. Кальмара",
        "key": "cal",
        "mult": 1,
        "dec": 2,
        "suf": "",
        "dir": 1,
        "gt": 0.5,
        "ot": 0.2,
        "desc": "CAGR / |Max Drawdown|",
    },
]

# =========================
# State
# =========================
def init_state() -> None:
    if "weights" not in st.session_state:
        st.session_state.weights = BASELINE.copy()

    if "re_on" not in st.session_state:
        st.session_state.re_on = False

    if "prev_re_on" not in st.session_state:
        st.session_state.prev_re_on = False

    for t in ALL:
        slider_key = f"slider_{t}"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = st.session_state.weights[t]


def rebalance_for_toggle(enable_mref: bool) -> None:
    if enable_mref:
        st.session_state.weights = MREF_PORTFOLIO.copy()
    else:
        st.session_state.weights = BASELINE.copy()

    for t in ALL:
        st.session_state[f"slider_{t}"] = st.session_state.weights[t]


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
    Подбор режима торгов для индекса:
    security -> boards -> приоритет traded/index.
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

    cand = boards.copy()
    if "engine" in boards.columns:
        cand = cand[cand["engine"].astype(str).eq("stock")]

    pref = cand.copy()
    if "market" in cand.columns:
        pref = cand[cand["market"].astype(str).eq("index")]

    pick = pref.iloc[0] if len(pref) else cand.iloc[0]
    return str(pick["engine"]), str(pick["market"]), str(pick["boardid"])


@st.cache_data(ttl=24 * 60 * 60)
def load_index_candles_daily(secid: str, d_from: date, d_to: date) -> pd.DataFrame:
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
                "interval": 24,
                "start": start,
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

    out = (
        out.sort_values(["secid", "tradedate"])
        .groupby(["secid", "tradedate"], as_index=False)
        .agg(close=("close", "last"))
    )
    return out


@st.cache_data(ttl=24 * 60 * 60)
def load_one_index(logical_key: str, d_from: date, d_to: date) -> Tuple[pd.DataFrame, str]:
    aliases = INDEX_META[logical_key]["aliases"]
    last_error = None

    for secid in aliases:
        try:
            df = load_index_candles_daily(secid, d_from, d_to)
            if not df.empty:
                return df.copy(), secid
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Не удалось загрузить {logical_key}. Последняя ошибка: {last_error}")


@st.cache_data(ttl=24 * 60 * 60)
def load_all_index_data(d_from: date, d_to: date):
    raw_map: Dict[str, pd.DataFrame] = {}
    resolved_map: Dict[str, str] = {}

    merged = None
    for key in ALL:
        df, resolved = load_one_index(key, d_from, d_to)
        raw_map[key] = df.copy()
        resolved_map[key] = resolved

        s = df[["tradedate", "close"]].rename(columns={"close": key})
        merged = s if merged is None else merged.merge(s, on="tradedate", how="outer")

    merged = merged.sort_values("tradedate").drop_duplicates("tradedate").reset_index(drop=True)
    merged[ALL] = merged[ALL].ffill()

    return merged, resolved_map, raw_map


# =========================
# Метрики
# =========================
def asset_stats_from_raw(df: pd.DataFrame) -> Tuple[float, float] | None:
    s = df.sort_values("tradedate")["close"].astype(float)
    rets = s.pct_change().dropna()

    if len(rets) == 0:
        return None

    cagr = (s.iloc[-1] / s.iloc[0]) ** (252 / len(rets)) - 1
    vol = rets.std(ddof=0) * math.sqrt(252)
    return cagr, vol


def prepare_window(prices: pd.DataFrame, active_keys: List[str]) -> pd.DataFrame:
    """
    Для корректного сравнения сценариев:
    - берем общий ценовой DataFrame по всем индексам,
    - отбрасываем даты, где еще нет данных по активным индексам.
    """
    df = prices.copy().set_index("tradedate")
    df = df.dropna(subset=active_keys).copy()
    return df[ALL]


def portfolio_daily_returns(price_window: pd.DataFrame, weights_pct: Dict[str, int]) -> pd.Series:
    used = [k for k in ALL if weights_pct.get(k, 0) > 0]
    if not used:
        return pd.Series(dtype=float)

    rets = price_window[used].pct_change().dropna(how="any")
    if rets.empty:
        return pd.Series(dtype=float)

    w = pd.Series({k: weights_pct[k] / 100 for k in used})
    port = rets.mul(w, axis=1).sum(axis=1)
    return port


def calc_portfolio_metrics(price_window: pd.DataFrame, weights_pct: Dict[str, int]) -> Dict[str, float]:
    pr = portfolio_daily_returns(price_window, weights_pct)
    if pr.empty:
        return {"ret": 0.0, "vol": 0.0, "shr": 0.0, "sor": 0.0, "mdd": 0.0, "cal": 0.0}

    nav = (1 + pr).cumprod()
    ann_return = nav.iloc[-1] ** (252 / len(pr)) - 1
    ann_vol = pr.std(ddof=0) * math.sqrt(252)

    downside = pr.clip(upper=0)
    downside_dev = math.sqrt((downside ** 2).mean()) * math.sqrt(252)

    sharpe = (ann_return - RF) / ann_vol if ann_vol else 0.0
    sortino = (ann_return - RF) / downside_dev if downside_dev else 0.0

    drawdown = nav / nav.cummax() - 1
    mdd = drawdown.min() if not drawdown.empty else 0.0
    calmar = ann_return / abs(mdd) if mdd else 0.0

    return {
        "ret": ann_return,
        "vol": ann_vol,
        "shr": sharpe,
        "sor": sortino,
        "mdd": mdd,
        "cal": calmar,
    }


def metric_class(direction: int, value: float, good_thr: float, ok_thr: float) -> str:
    if direction == 1:
        if value >= good_thr:
            return "good"
        if value >= ok_thr:
            return "neutral"
        return "bad"
    else:
        if value <= good_thr:
            return "good"
        if value <= ok_thr:
            return "neutral"
        return "bad"


def format_delta(metric_key: str, current_value: float, base_value: float, dec: int, suffix: str) -> Tuple[str, str]:
    diff = current_value - base_value
    icon = "▲" if diff >= 0 else "▼"
    sign = "+" if diff >= 0 else ""

    if metric_key == "vol":
        improved = current_value < base_value
    elif metric_key == "mdd":
        improved = abs(current_value) < abs(base_value)
    else:
        improved = current_value > base_value

    css = "pos" if improved else "neg"
    return f"{icon} {sign}{diff:.{dec}f}{suffix} vs базовый", css


# =========================
# UI state
# =========================
init_state()

# =========================
# Стили
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: #0a0c10;
        color: #e2e8f0;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.toggle-title) {
        background: linear-gradient(135deg, #121825, #0f1520) !important;
        border: 1px solid rgba(200,168,107,0.55) !important;
        border-radius: 22px !important;
        padding: 12px 18px 10px 18px !important;
        margin-top: 24px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18) !important;
    }

    .toggle-title {
        color: #e2e8f0;
        font-size: 1.02rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 8px;
        line-height: 1.1;
    }

    .toggle-name {
        color: #e5e7eb;
        font-size: 0.98rem;
        font-weight: 700;
        margin-bottom: 4px;
        line-height: 1.2;
    }

    .toggle-sub {
        color: #94a3b8;
        font-size: 0.80rem;
        line-height: 1.35;
    }

    .section-title {
        font-size: 0.82rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #94a3b8;
        margin: 24px 0 12px 0;
        font-weight: 700;
    }

    .asset-card {
        background: #111419;
        border: 1px solid #1e2530;
        border-radius: 14px;
        padding: 14px 14px 12px 14px;
        margin-bottom: 12px;
        min-height: 185px;
    }

    .asset-ticker {
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }

    .asset-name {
        color: #94a3b8;
        font-size: 0.88rem;
        margin-bottom: 10px;
        min-height: 40px;
    }

    .asset-stat {
        color: #cbd5e1;
        font-size: 0.78rem;
        margin-top: 8px;
        line-height: 1.45;
    }

    .asset-stat-label {
        color: #cbd5e1;
    }

    .asset-stat-pos {
        color: #4ade80;
        font-weight: 600;
    }

    .asset-stat-light {
        color: #cbd5e1;
    }

    .asset-note {
        color: #64748b;
        font-size: 0.68rem;
        margin-top: 6px;
        line-height: 1.35;
    }

    .metric-card {
        background: #111419;
        border: 1px solid #1e2530;
        border-radius: 14px;
        padding: 18px 16px;
        min-height: 160px;
    }

    .metric-label {
        font-size: 0.72rem;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 8px;
    }

    .good { color: #4ade80; }
    .neutral { color: #e8c97a; }
    .bad { color: #f87171; }

    .metric-delta {
        font-size: 0.80rem;
        margin-bottom: 8px;
        min-height: 1.2em;
    }

    .pos { color: #4ade80; }
    .neg { color: #f87171; }
    .zero { color: #64748b; }

    .metric-desc {
        font-size: 0.78rem;
        color: #64748b;
        line-height: 1.45;
    }

    .banner {
        background: linear-gradient(135deg, rgba(200,168,107,0.08), rgba(200,168,107,0.02));
        border: 1px solid rgba(200,168,107,0.2);
        border-radius: 12px;
        padding: 14px 18px;
        color: #c8a86b;
        margin: 8px 0 22px 0;
    }

    .insight {
        background: linear-gradient(135deg, rgba(45,212,191,0.06), rgba(45,212,191,0.02));
        border: 1px solid rgba(45,212,191,0.18);
        border-radius: 14px;
        padding: 20px;
        margin-top: 14px;
        margin-bottom: 22px;
    }

    .footnote {
        border-top: 1px solid #1e2530;
        padding-top: 16px;
        margin-top: 16px;
        color: #64748b;
        font-size: 0.78rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Параметры периода
# =========================
default_from = date(2018, 1, 1)
default_to = date.today()

with st.expander("Период расчета", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        d_from = st.date_input("Дата начала", value=default_from, key="date_from")
    with c2:
        d_to = st.date_input("Дата окончания", value=default_to, key="date_to")

if d_from >= d_to:
    st.error("Дата начала должна быть раньше даты окончания.")
    st.stop()

# =========================
# Загрузка данных
# =========================
with st.spinner("Загружаю реальные ряды индексов из ISS..."):
    try:
        prices, resolved_map, raw_map = load_all_index_data(d_from, d_to)
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        st.stop()

asset_stats = {}
for key in ALL:
    stats = asset_stats_from_raw(raw_map[key])
    asset_stats[key] = stats

# =========================
# Заголовок
# =========================
col_left, col_right = st.columns([3.2, 2])

with col_left:
    st.markdown(
        "# Портфель <span style='color:#e8c97a'>Российских Индексов</span>",
        unsafe_allow_html=True,
    )
    st.caption("Реальные ряды MOEX ISS · Исторические метрики риска и доходности")

with col_right:
    with st.container(border=True):
        tcol1, tcol2 = st.columns([4.2, 1.1], vertical_alignment="center")
        with tcol1:
            st.markdown("<div class='toggle-title'>ДОБАВИТЬ НЕДВИЖИМОСТЬ</div>", unsafe_allow_html=True)
            st.markdown("<div class='toggle-name'>MREF • Складская недвижимость</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='toggle-sub'>Индекс складской и индустриальной недвижимости МосБиржи</div>",
                unsafe_allow_html=True,
            )
        with tcol2:
            st.toggle("toggle_mref", key="re_on", label_visibility="collapsed")

if st.session_state.re_on != st.session_state.prev_re_on:
    rebalance_for_toggle(st.session_state.re_on)
    st.session_state.prev_re_on = st.session_state.re_on

if st.session_state.re_on:
    st.markdown(
        "<div class='banner'><strong>Складская недвижимость добавлена.</strong> "
        "Метрики считаются уже по реальному историческому окну, где доступны данные MREF.</div>",
        unsafe_allow_html=True,
    )

# =========================
# Карточки активов
# =========================
st.markdown("<div class='section-title'>Распределение активов</div>", unsafe_allow_html=True)

active_tickers = ALL if st.session_state.re_on else BASE
cols = st.columns(5 if st.session_state.re_on else 4)

for idx, ticker in enumerate(active_tickers):
    meta = INDEX_META[ticker]
    with cols[idx]:
        st.markdown(
            f"<div class='asset-card'>"
            f"<div class='asset-ticker' style='color:{meta['color']}'>{ticker}</div>"
            f"<div class='asset-name'>{meta['name']}</div>",
            unsafe_allow_html=True,
        )

        slider_value = st.slider(
            label=f"Вес {ticker}",
            min_value=0,
            max_value=meta["max"],
            step=1,
            format="%d%%",
            key=f"slider_{ticker}",
            label_visibility="collapsed",
        )

        st.session_state.weights[ticker] = slider_value

        st.markdown(f"**{slider_value}%**")

        stat = asset_stats.get(ticker)
        if stat is not None:
            cagr, vol = stat
            st.markdown(
                f"<div class='asset-stat'>"
                f"<span class='asset-stat-label'>Доходность:</span> "
                f"<span class='asset-stat-pos'>{cagr * 100:.1f}% / год</span> · "
                f"<span class='asset-stat-label'>Вол:</span> "
                f"<span class='asset-stat-light'>{vol * 100:.1f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='asset-stat'>Недостаточно данных</div>", unsafe_allow_html=True)

        st.markdown(
            f"<div class='asset-note'>ISS тикер: {resolved_map.get(ticker, ticker)}</div></div>",
            unsafe_allow_html=True,
        )

if not st.session_state.re_on:
    st.session_state.weights["MREF"] = 0
    st.session_state["slider_MREF"] = 0

current_total = sum(st.session_state.weights[t] for t in active_tickers)

# =========================
# Окно анализа
# =========================
price_window = prepare_window(prices, active_tickers)

if price_window.empty or len(price_window) < 3:
    st.error("Недостаточно общих исторических данных для расчета метрик.")
    st.stop()

actual_start = price_window.index.min()
actual_end = price_window.index.max()

st.caption(
    f"Фактическое окно расчета: {actual_start} — {actual_end}. "
    f"Если включен MREF, начало может сдвигаться вперед из-за более короткой истории."
)

# =========================
# Метрики портфеля
# =========================
if abs(current_total - 100) > 1:
    st.error(f"Сумма весов не равна 100%. Текущая сумма: {current_total}%")
else:
    current_metrics = calc_portfolio_metrics(price_window, st.session_state.weights)
    baseline_metrics = calc_portfolio_metrics(price_window, BASELINE)

    st.markdown("<div class='section-title'>Портфельные метрики</div>", unsafe_allow_html=True)

    metric_cols = st.columns(3)

    for idx, md in enumerate(METRIC_DEFS):
        raw = current_metrics[md["key"]]
        value = raw * md["mult"]

        if value < 0:
            css_class = "bad"
        else:
            css_class = metric_class(md["dir"], value, md["gt"], md["ot"])

        delta_html = "<div class='metric-delta zero'></div>"
        if st.session_state.re_on:
            base_value = baseline_metrics[md["key"]] * md["mult"]
            delta_text, delta_css = format_delta(
                md["key"], value, base_value, md["dec"], md["suf"]
            )
            delta_html = f"<div class='metric-delta {delta_css}'>{delta_text}</div>"

        with metric_cols[idx % 3]:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{md['label']}</div>"
                f"<div class='metric-value {css_class}'>{value:.{md['dec']}f}{md['suf']}</div>"
                f"{delta_html}"
                f"<div class='metric-desc'>{md['desc']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

# =========================
# Объясняющий блок
# =========================
if st.session_state.re_on:
    st.markdown(
        """
        <div class='insight'>
            <h3 style='color:#2dd4bf; margin-top:0;'>Почему склады могут менять портфель?</h3>
            <div style='display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px;'>
                <div>📦 MREF в этом приложении учитывается уже не как вручную заданный параметр, а как реальный исторический ряд индекса.</div>
                <div>📈 Доходность и волатильность пересчитываются на фактическом окне данных, а не по зашитым константам.</div>
                <div>🛡 Эффект диверсификации теперь определяется наблюдаемой совместной динамикой рядов, а не вручную заданной корреляционной матрицей.</div>
                <div>⚖️ Поэтому различия с вашим первоначальным HTML будут нормальны: модель стала эмпирической, а не сценарно-иллюстративной.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class='footnote'>
    Источник данных: ISS Московской биржи. Метрики рассчитываются по историческим дневным значениям индексов за выбранный период.
    Расчеты носят аналитический характер и не являются инвестиционной рекомендацией.
    </div>
    """,
    unsafe_allow_html=True,
)
