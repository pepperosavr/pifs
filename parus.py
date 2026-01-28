# app.py
# Запуск: streamlit run app.py
#
# Что делает:
# - Берет RSS/Atom URL (вставляете в сайдбар)
# - Парсит новые элементы, сохраняет в SQLite
# - По желанию скачивает полный текст страницы (для классификации)
# - Классифицирует события по ключевым словам/regex
# - Показывает витрину с фильтрами + экспортом
#
# Примечание:
# - SQLite создается рядом с app.py (smartlab_events.db)
# - Для надежности используем WAL и busy_timeout
# - Если не хотите трогать сеть из UI-процесса, запускайте обновление по кнопке реже

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import feedparser
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup


APP_TITLE = "Мониторинг событии (smart-lab) для конкурентного анализа"
DEFAULT_DB_PATH = "smartlab_events.db"

DEFAULT_FEEDS_TEXT = """parus_news|PASTE_RSS_URL_HERE
parus_blog|PASTE_RSS_URL_HERE
"""

# Базовые правила под конкурентный анализ ЗПИФ/УК.
# Вы можете расширять их в UI (кастомные паттерны).
BASE_EVENT_RULES: Dict[str, List[str]] = {
    "additional_units_or_spo": [
        r"\bSPO\b",
        r"доп(олнительн\w*)?\s+(инвестиц\w*\s+)?па(и|ев)\b",
        r"доп(олнительн\w*)?\s+выдач\w+\s+паев",
        r"доп(олнительн\w*)?\s+эмисси\w+",
        r"\bразмещен(ие|ия)\b",
        r"книга заявок",
        r"новая партия паев",
        r"доп(олнительн\w*)?\s+размещен(ие|ия)",
        r"доп(олнительн\w*)?\s+выпуск",
    ],
    "debt_event": [
        r"закрыл(и|а|о)\s+кредит",
        r"погашен(ие|)\s+кредит",
        r"погашен(а|о|ы)\s+задолженн\w+",
        r"рефинансир\w+",
        r"кредитн\w+\s+договор",
        r"сняти(е|я)\s+обременен\w+",
        r"ипотек\w+\s+снята",
    ],
    "asset_transaction": [
        r"приобр(е|и)л[аио]?\b",
        r"покупк\w+\s+(объект|актив|недвижимост\w+)",
        r"продал[аио]?\b",
        r"сделк\w+\b",
        r"заключен(а|о)\s+сделк\w+",
        r"выкуп\b",
        r"долю\b",
    ],
    "tenant_or_lease": [
        r"арендатор\w+",
        r"договор аренды",
        r"подписан(а|о)\s+аренд\w+",
        r"якорн\w+\s+арендатор\w+",
        r"ваканси\w+\s+сниж",
    ],
    "payouts": [
        r"выплат(а|или|ил)\s+доход",
        r"распределен(ие|)\s+доход",
        r"денежн\w+\s+поток",
        r"доходност\w+",
        r"выплат(а|ы)\s+инвестор\w+",
    ],
    "regulatory_or_disclosure": [
        r"раскрыт(ие|а)\s+информац\w+",
        r"сообщен(ие|ия)\s+о\s+существенн\w+\s+факт",
        r"отчетност\w+",
        r"аудит\w+",
        r"изменен(ие|ия)\s+правил",
        r"изменен(ие|ия)\s+услови\w+",
    ],
}


@dataclass
class FeedConfig:
    name: str
    url: str


def utcnow_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def parse_feeds_text(text: str) -> List[FeedConfig]:
    feeds: List[FeedConfig] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            name, url = line.split("|", 1)
            name, url = name.strip(), url.strip()
        else:
            name, url = "feed", line
        if url:
            feeds.append(FeedConfig(name=name or "feed", url=url))
    return feeds


def safe_requests_get(url: str, timeout: int = 30) -> requests.Response:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitSmartlabMonitor/1.0)"}
    r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r


def html_to_text(html: str) -> str:
    # Предпочитаем lxml, но если его нет, BeautifulSoup сам упадет на html.parser
    soup = BeautifulSoup(html, "lxml")
    # Убираем явный мусор: скрипты/стили
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return text


def feed_entry_datetime(entry) -> Optional[str]:
    # feedparser возвращает time.struct_time в published_parsed/updated_parsed
    # Сохраняем как ISO 8601 (UTC)
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                dt = datetime.fromtimestamp(time.mktime(t), tz=timezone.utc)
                return dt.isoformat()
            except Exception:
                pass
    # иногда есть published/updated строкой
    for attr in ("published", "updated"):
        s = getattr(entry, attr, None)
        if s and isinstance(s, str):
            # как есть; для фильтров используем fetched_at если не получится распарсить
            return s
    return None


def compile_rules(
    base_rules: Dict[str, List[str]], extra_rules_text: str
) -> Dict[str, List[re.Pattern]]:
    """
    extra_rules_text формат:
      category_name: pattern1
      category_name: pattern2
    или
      category_name|pattern
    Пустые строки и #комменты игнорируются.
    """
    rules: Dict[str, List[str]] = {k: list(v) for k, v in base_rules.items()}

    for line in (extra_rules_text or "").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        cat, pat = None, None
        if ":" in raw:
            cat, pat = raw.split(":", 1)
        elif "|" in raw:
            cat, pat = raw.split("|", 1)
        if cat and pat:
            cat, pat = cat.strip(), pat.strip()
            if cat and pat:
                rules.setdefault(cat, []).append(pat)

    compiled: Dict[str, List[re.Pattern]] = {}
    for cat, patterns in rules.items():
        compiled[cat] = []
        for p in patterns:
            try:
                compiled[cat].append(re.compile(p, flags=re.IGNORECASE))
            except re.error:
                # некорректный regex — игнорируем, но лучше показывать в UI (делаем это позже)
                continue
    return compiled


def classify_text(title: str, text: str, compiled_rules: Dict[str, List[re.Pattern]]) -> List[str]:
    blob = f"{title or ''} {text or ''}"
    labels = []
    for cat, patterns in compiled_rules.items():
        for rx in patterns:
            if rx.search(blob):
                labels.append(cat)
                break
    return labels


@st.cache_resource
def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            guid TEXT NOT NULL,
            title TEXT,
            link TEXT,
            published TEXT,
            fetched_at TEXT NOT NULL,
            labels TEXT,
            is_event INTEGER NOT NULL DEFAULT 0,
            summary TEXT,
            content TEXT,
            UNIQUE(source, guid)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_items_published
        ON items(published)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_items_is_event
        ON items(is_event)
        """
    )
    conn.commit()


def upsert_item(
    conn: sqlite3.Connection,
    source: str,
    guid: str,
    title: str,
    link: str,
    published: Optional[str],
    labels: List[str],
    summary: str,
    content: str,
) -> bool:
    """
    Возвращает True, если вставили новый элемент (а не проигнорировали).
    """
    labels_str = ",".join(labels)
    is_event = 1 if labels else 0
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO items
        (source, guid, title, link, published, fetched_at, labels, is_event, summary, content)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source,
            guid,
            title,
            link,
            published,
            utcnow_iso(),
            labels_str,
            is_event,
            summary,
            content,
        ),
    )
    conn.commit()
    return cur.rowcount == 1


def update_from_feeds(
    conn: sqlite3.Connection,
    feeds: List[FeedConfig],
    compiled_rules: Dict[str, List[re.Pattern]],
    fetch_fulltext: bool,
    only_events: bool,
    request_timeout: int,
    max_items_per_feed: int,
) -> Tuple[int, int, List[str]]:
    """
    Возвращает:
      (inserted_count, scanned_count, warnings)
    """
    inserted = 0
    scanned = 0
    warnings: List[str] = []

    for f in feeds:
        if not f.url or "PASTE_RSS_URL_HERE" in f.url:
            warnings.append(f"[{f.name}] пропущено: вставьте реальный RSS URL")
            continue

        try:
            parsed = feedparser.parse(f.url)
        except Exception as e:
            warnings.append(f"[{f.name}] ошибка feedparser: {e}")
            continue

        entries = getattr(parsed, "entries", []) or []
        if max_items_per_feed and len(entries) > max_items_per_feed:
            entries = entries[:max_items_per_feed]

        for e in entries:
            scanned += 1
            link = getattr(e, "link", "") or ""
            title = getattr(e, "title", "") or ""
            summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""
            guid = getattr(e, "id", None) or getattr(e, "guid", None) or link or f"{f.name}:{title}"

            published = feed_entry_datetime(e)

            content_text = ""
            if fetch_fulltext and link:
                try:
                    r = safe_requests_get(link, timeout=request_timeout)
                    content_text = html_to_text(r.text)
                except Exception as ex:
                    warnings.append(f"[{f.name}] не удалось скачать/разобрать {link}: {ex}")
                    content_text = summary
            else:
                content_text = summary

            labels = classify_text(title, content_text, compiled_rules)

            if only_events and not labels:
                # не сохраняем шум
                continue

            try:
                if upsert_item(
                    conn=conn,
                    source=f.name,
                    guid=str(guid),
                    title=title,
                    link=link,
                    published=published,
                    labels=labels,
                    summary=summary,
                    content=content_text if fetch_fulltext else "",
                ):
                    inserted += 1
            except sqlite3.OperationalError as oe:
                warnings.append(f"[{f.name}] SQLite ошибка при вставке: {oe}")
            except Exception as ex:
                warnings.append(f"[{f.name}] неожиданная ошибка: {ex}")

    return inserted, scanned, warnings


@st.cache_data(ttl=900)
def load_items_df(db_path: str) -> pd.DataFrame:
    conn = get_conn(db_path)
    df = pd.read_sql_query(
        """
        SELECT
            source, guid, title, link, published, fetched_at, labels, is_event
        FROM items
        ORDER BY
            CASE WHEN published IS NULL OR published = '' THEN fetched_at ELSE published END DESC
        """,
        conn,
    )
    # Попробуем распарсить published/fetched_at в datetime для фильтра по дате
    def to_dt(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="coerce", utc=True)

    df["published_dt"] = to_dt(df["published"])
    df["fetched_dt"] = to_dt(df["fetched_at"])
    df["effective_dt"] = df["published_dt"].fillna(df["fetched_dt"])
    return df


def apply_filters(
    df: pd.DataFrame,
    sources: List[str],
    labels_selected: List[str],
    only_events: bool,
    q: str,
    date_from: Optional[date],
    date_to: Optional[date],
) -> pd.DataFrame:
    out = df.copy()

    if only_events:
        out = out[out["is_event"] == 1]

    if sources:
        out = out[out["source"].isin(sources)]

    if labels_selected:
        def has_any_labels(s: str) -> bool:
            parts = [x for x in (s or "").split(",") if x]
            return any(l in parts for l in labels_selected)
        out = out[out["labels"].apply(has_any_labels)]

    if q:
        ql = q.strip().lower()
        out = out[out["title"].fillna("").str.lower().str.contains(ql)]

    if date_from or date_to:
        # effective_dt в UTC; приводим фильтр к UTC по дате
        if date_from:
            start = pd.Timestamp(datetime(date_from.year, date_from.month, date_from.day, tzinfo=timezone.utc))
            out = out[out["effective_dt"] >= start]
        if date_to:
            end = pd.Timestamp(datetime(date_to.year, date_to.month, date_to.day, tzinfo=timezone.utc)) + pd.Timedelta(days=1)
            out = out[out["effective_dt"] < end]

    return out


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    # --- Sidebar: конфигурация ---
    with st.sidebar:
        st.header("Конфигурация")

        db_path = st.text_input("SQLite файл", value=DEFAULT_DB_PATH)
        db_path = str(Path(db_path).expanduser())

        conn = get_conn(db_path)
        init_db(conn)

        st.caption("RSS URL можно взять, кликнув на иконку RSS на нужной ленте/блоге smart-lab.")

        feeds_text = st.text_area(
            "Ленты (формат: имя|url, по 1 на строку)",
            value=st.session_state.get("feeds_text", DEFAULT_FEEDS_TEXT),
            height=140,
        )
        st.session_state["feeds_text"] = feeds_text
        feeds = parse_feeds_text(feeds_text)

        st.subheader("Сбор данных")
        fetch_fulltext = st.checkbox(
            "Скачивать полный текст по ссылке (точнее, но медленнее)",
            value=st.session_state.get("fetch_fulltext", True),
        )
        st.session_state["fetch_fulltext"] = fetch_fulltext

        only_events_store = st.checkbox(
            "Сохранять в БД только события (не хранить шум)",
            value=st.session_state.get("only_events_store", False),
        )
        st.session_state["only_events_store"] = only_events_store

        request_timeout = st.number_input("Timeout запросов (сек)", min_value=5, max_value=120, value=25, step=5)
        max_items_per_feed = st.number_input("Макс. элементов на ленту за запуск", min_value=10, max_value=500, value=120, step=10)

        st.subheader("Правила событий")
        st.caption(
            "Можно добавить свои паттерны. Формат строк: category: regex  или  category|regex"
        )
        extra_rules = st.text_area(
            "Кастомные правила (опционально)",
            value=st.session_state.get("extra_rules", ""),
            height=120,
        )
        st.session_state["extra_rules"] = extra_rules

        # Компилируем правила один раз на ререндер
        compiled_rules = compile_rules(BASE_EVENT_RULES, extra_rules)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Обновить ленты"):
                with st.spinner("Собираю и сохраняю новые элементы..."):
                    inserted, scanned, warnings = update_from_feeds(
                        conn=conn,
                        feeds=feeds,
                        compiled_rules=compiled_rules,
                        fetch_fulltext=fetch_fulltext,
                        only_events=only_events_store,
                        request_timeout=int(request_timeout),
                        max_items_per_feed=int(max_items_per_feed),
                    )
                load_items_df.clear()
                st.success(f"Готово. Просканировано: {scanned}. Добавлено новых: {inserted}.")
                if warnings:
                    st.warning("Есть предупреждения. Смотрите ниже.")
                    for w in warnings[:20]:
                        st.write("-", w)
                    if len(warnings) > 20:
                        st.write(f"...и еще {len(warnings) - 20}")

        with col_b:
            if st.button("Сбросить кэш витрины"):
                load_items_df.clear()
                st.info("Кэш очищен.")

    # --- Main: витрина ---
    df = load_items_df(db_path)

    total = len(df)
    total_events = int(df["is_event"].sum()) if "is_event" in df.columns else 0
    last_dt = df["effective_dt"].max() if not df.empty else None

    m1, m2, m3 = st.columns(3)
    m1.metric("Всего записеи в БД", f"{total}")
    m2.metric("События (по правилам)", f"{total_events}")
    m3.metric("Последняя дата", "-" if last_dt is None else str(last_dt))

    st.divider()

    # Фильтры
    sources_all = sorted(df["source"].dropna().unique().tolist()) if total else []
    labels_all = sorted({lab for x in df["labels"].dropna() for lab in str(x).split(",") if lab})

    f1, f2, f3, f4 = st.columns([2, 2, 2, 2])

    with f1:
        sources_selected = st.multiselect("Источник", sources_all, default=sources_all[:])
    with f2:
        labels_selected = st.multiselect("Категории", labels_all, default=labels_all[:])
    with f3:
        only_events_view = st.checkbox("Показывать только события", value=True)
    with f4:
        q = st.text_input("Поиск по заголовку", value="")

    d1, d2 = st.columns(2)
    with d1:
        date_from = st.date_input("Дата с", value=None)
    with d2:
        date_to = st.date_input("Дата по", value=None)

    filtered = apply_filters(
        df=df,
        sources=sources_selected,
        labels_selected=labels_selected,
        only_events=only_events_view,
        q=q,
        date_from=date_from,
        date_to=date_to,
    )

    st.subheader("Таймлаин событий")
    if filtered.empty:
        st.info("По текущим фильтрам ничего не найдено.")
    else:
        # Группируем по дате (UTC), считаем число событии
        tmp = filtered.copy()
        tmp["day"] = tmp["effective_dt"].dt.date
        ts = tmp.groupby("day")["guid"].count().reset_index(name="count").sort_values("day")
        st.line_chart(ts.set_index("day")["count"])

    st.subheader("Список")
    st.caption("Колонка link кликабельна в режиме data editor. Можно экспортировать отфильтрованное в CSV.")

    # Таблица с кликабельными ссылками
    view = filtered.copy()
    view["published_or_fetched"] = view["effective_dt"].astype(str)

    cols = ["published_or_fetched", "title", "labels", "source", "link"]
    view = view[cols].rename(columns={"published_or_fetched": "dt"})

    st.data_editor(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "dt": st.column_config.TextColumn("Дата (published/fetched)"),
            "title": st.column_config.TextColumn("Заголовок"),
            "labels": st.column_config.TextColumn("Категории"),
            "source": st.column_config.TextColumn("Источник"),
            "link": st.column_config.LinkColumn("Ссылка"),
        },
        disabled=True,
    )

    # Экспорт
    csv_bytes = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Скачать CSV (отфильтрованное)",
        data=csv_bytes,
        file_name="smartlab_events_filtered.csv",
        mime="text/csv",
    )

    with st.expander("Как расширять правила классификации"):
        st.write(
            """
- Базовые правила заданы в коде (BASE_EVENT_RULES).
- В сайдбаре можно добавить кастомные строки вида:
  - category: ваш_regex
  - category|ваш_regex

Примеры:
- additional_units_or_spo: "доп\\.?\\s*размещ"
- debt_event: "погашен\\w+\\s+кредит"
- asset_transaction: "приобрел\\w+\\s+(объект|актив)"
"""
        )


if __name__ == "__main__":
    main()
