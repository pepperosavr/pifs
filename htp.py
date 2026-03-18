import math
import streamlit as st

st.set_page_config(
    page_title="Портфель российских индексов",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ASSETS = {
    "IMOEX": {
        "name": "Индекс МосБиржи",
        "r": 0.142,
        "s": 0.228,
        "color": "#c8a86b",
        "max": 80,
        "corr": {"IMOEX": 1, "RGBI": -0.18, "MCFTR": 0.85, "RUCBTR": 0.22, "MREF": 0.21},
        "stat": "Доходность: +14.2% / год · Вол: 22.8%",
    },
    "RGBI": {
        "name": "Гос. облигации (ОФЗ)",
        "r": 0.082,
        "s": 0.055,
        "color": "#64748b",
        "max": 80,
        "corr": {"IMOEX": -0.18, "RGBI": 1, "MCFTR": -0.12, "RUCBTR": 0.62, "MREF": -0.05},
        "stat": "Доходность: +8.2% / год · Вол: 5.5%",
    },
    "MCFTR": {
        "name": "МосБиржа полной доходности",
        "r": 0.168,
        "s": 0.235,
        "color": "#f59e0b",
        "max": 80,
        "corr": {"IMOEX": 0.85, "RGBI": -0.12, "MCFTR": 1, "RUCBTR": 0.18, "MREF": 0.24},
        "stat": "Доходность: +16.8% / год · Вол: 23.5%",
    },
    "RUCBTR": {
        "name": "Корп. облигации",
        "r": 0.095,
        "s": 0.068,
        "color": "#94a3b8",
        "max": 80,
        "corr": {"IMOEX": 0.22, "RGBI": 0.62, "MCFTR": 0.18, "RUCBTR": 1, "MREF": 0.08},
        "stat": "Доходность: +9.5% / год · Вол: 6.8%",
    },
    "MREF": {
        "name": "Складская и индустриальная недвижимость",
        "r": 0.178,
        "s": 0.095,
        "color": "#2dd4bf",
        "max": 40,
        "corr": {"IMOEX": 0.21, "RGBI": -0.05, "MCFTR": 0.24, "RUCBTR": 0.08, "MREF": 1},
        "stat": "Доходность: +17.8% / год · Вол: 9.5%",
    },
}

BASE = ["IMOEX", "RGBI", "MCFTR", "RUCBTR"]
ALL = ["IMOEX", "RGBI", "MCFTR", "RUCBTR", "MREF"]
RF = 0.16
BASELINE = {"IMOEX": 40, "RGBI": 25, "MCFTR": 20, "RUCBTR": 15, "MREF": 0}

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
        "desc": "Среднегодовая взвешенная доходность портфеля",
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
        "desc": "Стандартное отклонение — мера нестабильности",
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
        "desc": "Доходность сверх ставки ЦБ (16%) на единицу риска",
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
        "desc": "Шарп, штрафующий только нисходящую волатильность",
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
        "desc": "Максимальное падение стоимости от исторического пика",
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
        "desc": "Доходность / |Макс. просадка| — качество риска/награды",
    },
]


def init_state() -> None:
    if "weights" not in st.session_state:
        st.session_state.weights = BASELINE.copy()
    if "re_on" not in st.session_state:
        st.session_state.re_on = False
    if "prev_re_on" not in st.session_state:
        st.session_state.prev_re_on = False


def rebalance_for_toggle(enable_mref: bool) -> None:
    w = st.session_state.weights
    if enable_mref:
        w["MREF"] = 15
        base_sum = sum(w[t] for t in BASE)
        acc = 0
        for i, t in enumerate(BASE):
            if i < len(BASE) - 1:
                w[t] = round(w[t] / base_sum * 85)
                acc += w[t]
            else:
                w[t] = 85 - acc
    else:
        w["MREF"] = 0
        base_sum = sum(w[t] for t in BASE)
        acc = 0
        for i, t in enumerate(BASE):
            if i < len(BASE) - 1:
                w[t] = round(w[t] / base_sum * 100)
                acc += w[t]
            else:
                w[t] = 100 - acc


def portfolio_return(weights_pct: dict[str, int]) -> float:
    return sum((weights_pct[t] / 100) * ASSETS[t]["r"] for t in ALL)


def portfolio_vol(weights_pct: dict[str, int]) -> float:
    v = 0.0
    for ti in ALL:
        for tj in ALL:
            wi = weights_pct[ti] / 100
            wj = weights_pct[tj] / 100
            v += wi * wj * ASSETS[ti]["s"] * ASSETS[tj]["s"] * ASSETS[ti]["corr"][tj]
    return math.sqrt(v)


def calc_metrics(weights_pct: dict[str, int]) -> dict[str, float]:
    r = portfolio_return(weights_pct)
    s = portfolio_vol(weights_pct)
    mdd = -(s * 2.3 * math.sqrt(1.5))
    dv = s * 0.62
    shr = (r - RF) / s if s else 0.0
    sor = (r - RF) / dv if dv else 0.0
    cal = r / abs(mdd) if mdd else 0.0
    return {"ret": r, "vol": s, "shr": shr, "sor": sor, "mdd": mdd, "cal": cal}


def metric_class(direction: int, value: float, good_thr: float, ok_thr: float) -> str:
    if direction == 1:
        if value >= good_thr:
            return "good"
        if value >= ok_thr:
            return "neutral"
        return "bad"
    if value <= good_thr:
        return "good"
    if value <= ok_thr:
        return "neutral"
    return "bad"


def format_delta(direction: int, diff: float, dec: int, suffix: str) -> tuple[str, str]:
    improved = diff > 0 if direction == 1 else diff < 0
    icon = "▲" if improved else "▼"
    sign = "+" if diff >= 0 else ""
    css = "pos" if improved else "neg"
    return f"{icon} {sign}{diff:.{dec}f}{suffix} vs базовый", css


init_state()
baseline_metrics = calc_metrics(BASELINE)

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

    h1, h2, h3 {
        color: #e2e8f0;
    }

    .top-card {
        background: #161b23;
        border: 1px solid #1e2530;
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 10px;
    }

    .banner {
        background: linear-gradient(135deg, rgba(200,168,107,0.08), rgba(200,168,107,0.02));
        border: 1px solid rgba(200,168,107,0.2);
        border-radius: 12px;
        padding: 14px 18px;
        color: #c8a86b;
        margin: 8px 0 22px 0;
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
        min-height: 170px;
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
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 8px;
        line-height: 1.4;
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

    div[data-testid="stMarkdownContainer"] p {
        color: inherit;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([3.2, 2])

with col_left:
    st.markdown(
        "# Портфель <span style='color:#e8c97a'>Российских Индексов</span>",
        unsafe_allow_html=True,
    )
    st.caption("Интерактивный анализ · Метрики риска и доходности")

with col_right:
    st.markdown("<div class='top-card'>", unsafe_allow_html=True)
    st.toggle(
        "Добавить недвижимость · MREF",
        key="re_on",
        help="Индекс складской и индустриальной недвижимости МосБиржи",
    )
    st.caption("При включении MREF получает 15%, а остальные активы автоматически масштабируются до 85%.")
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.re_on != st.session_state.prev_re_on:
    rebalance_for_toggle(st.session_state.re_on)
    st.session_state.prev_re_on = st.session_state.re_on

if st.session_state.re_on:
    st.markdown(
        "<div class='banner'><strong>Складская недвижимость добавлена.</strong> "
        "Низкая корреляция с акциями делает портфель устойчивее без потери доходности.</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='section-title'>Распределение активов</div>", unsafe_allow_html=True)

active_tickers = ALL if st.session_state.re_on else BASE
cols = st.columns(5 if st.session_state.re_on else 4)

for idx, ticker in enumerate(active_tickers):
    asset = ASSETS[ticker]
    with cols[idx]:
        st.markdown(
            f"<div class='asset-card'>"
            f"<div class='asset-ticker' style='color:{asset['color']}'>{ticker}</div>"
            f"<div class='asset-name'>{asset['name']}</div>",
            unsafe_allow_html=True,
        )

        slider_value = st.slider(
            label=f"Вес {ticker}",
            min_value=0,
            max_value=asset["max"],
            value=int(st.session_state.weights[ticker]),
            step=1,
            key=f"slider_{ticker}",
            label_visibility="collapsed",
        )

        st.session_state.weights[ticker] = slider_value
        st.markdown(f"**{slider_value}%**")
        st.markdown(f"<div class='asset-stat'>{asset['stat']}</div></div>", unsafe_allow_html=True)

if not st.session_state.re_on:
    st.session_state.weights["MREF"] = 0

current_total = sum(st.session_state.weights[t] for t in active_tickers)

if abs(current_total - 100) > 1:
    st.error(f"Сумма весов не равна 100%. Текущая сумма: {current_total}%")
else:
    current_metrics = calc_metrics(st.session_state.weights)
    st.markdown("<div class='section-title'>Портфельные метрики</div>", unsafe_allow_html=True)

    metric_cols = st.columns(3)

    for idx, md in enumerate(METRIC_DEFS):
        raw = current_metrics[md["key"]]
        value = raw * md["mult"]
        css_class = metric_class(md["dir"], value, md["gt"], md["ot"])

        delta_html = "<div class='metric-delta zero'></div>"
        if st.session_state.re_on:
            base_value = baseline_metrics[md["key"]] * md["mult"]
            diff = value - base_value
            delta_text, delta_css = format_delta(md["dir"], diff, md["dec"], md["suf"])
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

if st.session_state.re_on:
    st.markdown(
        """
        <div class='insight'>
            <h3 style='color:#2dd4bf; margin-top:0;'>Почему склады улучшают портфель?</h3>
            <div style='display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px;'>
                <div>📦 MREF отражает динамику фондов складской и индустриальной недвижимости.</div>
                <div>📈 Рентные выплаты поддерживают доходность даже при слабом рынке акций.</div>
                <div>🛡 Слабо коррелированный актив снижает совокупную волатильность портфеля.</div>
                <div>⚖️ Это соответствует эффекту диверсификации в логике портфельной теории Марковица.</div>
                <div>🏗 Рост e-commerce и 3PL-логистики формирует долгосрочный спрос на складские площади.</div>
                <div>🔒 Индексация арендных ставок создает частичную инфляционную защиту.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class='footnote'>
    Данные смоделированы на исторических параметрах индексов Московской биржи (2018–2024).<br>
    IMOEX, RGBI, MCFTR, RUCBTR — официальные индексы МосБиржи.<br>
    MREF — индекс складской и индустриальной недвижимости МосБиржи.<br>
    Безрисковая ставка для коэффициентов Шарпа и Сортино принята равной 16%.<br>
    Расчеты носят иллюстративный характер и не являются инвестиционной рекомендацией.
    </div>
    """,
    unsafe_allow_html=True,
)
