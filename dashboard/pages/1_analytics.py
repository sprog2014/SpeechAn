import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import sys
import os
import plotly.express as px
from datetime import datetime, timedelta

# Добавляем путь к src, чтобы найти config.py и db_utils.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import PG_CONFIG
from db_utils import get_all_prompts, get_default_prompt, get_call_file_path, get_call_transcript, format_dialogue

if not st.session_state.get("password_correct", False):
    st.error("Пожалуйста, авторизуйтесь на главной странице.")
    st.stop()

st.set_page_config(page_title="Аналитика звонков", layout="wide")
st.title("Аналитика звонков")

@st.cache_resource
def get_engine():
    db_url = f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}"
    return create_engine(db_url)

@st.cache_data(ttl=60)
def get_analytics_data(start_date, end_date, prompt_id, filters=None):
    engine = get_engine()

    where_clauses = ["c.calldate >= :start", "c.calldate < :end", "e.prompt_id = :pid"]
    params = {
        "start": start_date,
        "end": end_date + timedelta(days=1),
        "pid": prompt_id
    }

    # Словари для маппинга (используем в SQL)
    purpose_sql = """CASE e.call_purpose
        WHEN 'appointment' THEN 'Запись на прием'
        WHEN 'consultation' THEN 'Консультация'
        WHEN 'complaint' THEN 'Жалоба'
        WHEN 'cancel_appointment' THEN 'Отмена записи'
        WHEN 'reschedule_appointment' THEN 'Перенос записи'
        ELSE COALESCE(e.call_purpose, 'Другое') END"""

    sentiment_sql = """CASE e.client_sentiment
        WHEN 'positive' THEN 'Положительное'
        WHEN 'neutral' THEN 'Нейтральное'
        WHEN 'negative' THEN 'Отрицательное'
        WHEN 'conflict' THEN 'Конфликт'
        ELSE COALESCE(e.client_sentiment, 'Не определено') END"""

    direction_sql = """CASE LOWER(c.direction)
        WHEN 'incoming' THEN '📥'
        WHEN 'inbound' THEN '📥'
        WHEN 'outgoing' THEN '📤'
        WHEN 'outbound' THEN '📤'
        WHEN 'internal' THEN '🏠'
        ELSE '❓' END"""

    if filters:
        if filters.get("purpose"):
            where_clauses.append(f"{purpose_sql} = :f_purpose")
            params["f_purpose"] = filters["purpose"]
        if filters.get("sentiment"):
            where_clauses.append(f"{sentiment_sql} = :f_sentiment")
            params["f_sentiment"] = filters["sentiment"]
        if filters.get("hour") is not None:
            where_clauses.append("EXTRACT(HOUR FROM c.calldate) = :f_hour")
            params["f_hour"] = filters["hour"]
        if filters.get("type"):
            where_clauses.append(f"{direction_sql} = :f_type")
            params["f_type"] = filters["type"]
        if filters.get("date"):
            where_clauses.append("DATE(c.calldate) = :f_date")
            params["f_date"] = filters["date"]

    where_str = " AND ".join(where_clauses)

    # 1. Агрегация по часам и типам
    query_hourly = f"""
        SELECT
            EXTRACT(HOUR FROM c.calldate) as hour,
            {direction_sql} as call_type,
            COUNT(*) as count,
            AVG(c.billsec) as avg_billsec
        FROM calls c
        JOIN evaluations e ON c.linkedid = e.linkedid
        WHERE {where_str}
        GROUP BY 1, 2
    """

    # 2. Агрегация по целям
    query_purposes = f"""
        SELECT {purpose_sql} as purpose, COUNT(*) as count
        FROM calls c
        JOIN evaluations e ON c.linkedid = e.linkedid
        WHERE {where_str}
        GROUP BY 1
    """

    # 3. Агрегация по настроению
    query_sentiment = f"""
        SELECT {sentiment_sql} as sentiment, COUNT(*) as count
        FROM calls c
        JOIN evaluations e ON c.linkedid = e.linkedid
        WHERE {where_str}
        GROUP BY 1
    """

    # 4. Вежливость по операторам
    query_operators = f"""
        SELECT
            COALESCE(p.name, CASE WHEN c.direction = 'incoming' THEN c.answeredext ELSE c.src END) as operator_name,
            ROUND(AVG(e.politeness_score)::numeric, 2) as avg_politeness
        FROM calls c
        JOIN evaluations e ON c.linkedid = e.linkedid
        LEFT JOIN phones p ON p.number = (CASE WHEN c.direction = 'incoming' THEN c.answeredext ELSE c.src END)
        WHERE {where_str} AND e.politeness_score IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC
    """

    # 5. Список звонков (детализация)
    query_calls = f"""
        SELECT
            c.linkedid, c.calldate, c.billsec, c.processing_status,
            {direction_sql} as call_type,
            CASE WHEN c.direction = 'incoming' THEN c.src ELSE c.answeredext END as client_number,
            COALESCE(p.name, CASE WHEN c.direction = 'incoming' THEN c.answeredext ELSE c.src END) as operator_name,
            {purpose_sql} as purpose,
            {sentiment_sql} as sentiment,
            e.call_summary,
            e.checklist_json,
            EXTRACT(HOUR FROM c.calldate) as hour
        FROM calls c
        JOIN evaluations e ON c.linkedid = e.linkedid
        LEFT JOIN phones p ON p.number = (CASE WHEN c.direction = 'incoming' THEN c.answeredext ELSE c.src END)
        WHERE {where_str}
        ORDER BY c.calldate DESC
        LIMIT 1000
    """

    with engine.connect() as conn:
        df_hourly = pd.read_sql(text(query_hourly), conn, params=params)
        df_purposes = pd.read_sql(text(query_purposes), conn, params=params)
        df_sentiment = pd.read_sql(text(query_sentiment), conn, params=params)
        df_operators = pd.read_sql(text(query_operators), conn, params=params)
        df_calls = pd.read_sql(text(query_calls), conn, params=params)

    return {
        "hourly": df_hourly,
        "purposes": df_purposes,
        "sentiment": df_sentiment,
        "operators": df_operators,
        "calls": df_calls
    }

# --- Интерфейс ---
col_d, col_p = st.columns([1, 1])

with col_d:
    today = datetime.now().date()
    default_start = today - timedelta(days=7)
    date_range = st.date_input("Диапазон дат", (default_start, today))

if not (isinstance(date_range, tuple) and len(date_range) == 2):
    st.info("Выберите диапазон дат.")
    st.stop()

start_date, end_date = date_range

with col_p:
    all_prompts = get_all_prompts()
    default_p = get_default_prompt()
    default_p_id = default_p['id'] if default_p else (all_prompts[0]['id'] if all_prompts else None)
    prompt_options = {p['id']: p['name'] for p in all_prompts}
    selected_prompt_id = st.selectbox("Аналитический промпт", options=list(prompt_options.keys()),
                                      format_func=lambda x: prompt_options[x],
                                      index=list(prompt_options.keys()).index(default_p_id) if default_p_id in prompt_options else 0)

if not selected_prompt_id:
    st.warning("Промпты не найдены.")
    st.stop()

# Инициализация фильтров
if "filters" not in st.session_state:
    st.session_state.filters = {"purpose": None, "sentiment": None, "hour": None, "type": None, "date": None}
if "show_table" not in st.session_state:
    st.session_state.show_table = False

# Загрузка данных (с учетом активных фильтров для графиков, если нужно кросс-фильтровать всё)
# В оригинальной версии была сложная логика "исключения фильтра самого для себя".
# Для упрощения и скорости в SQL-версии будем просто фильтровать всё по текущим фильтрам.
data = get_analytics_data(start_date, end_date, selected_prompt_id, st.session_state.filters)

# Отображение фильтров
active_filters = [k for k, v in st.session_state.filters.items() if v is not None]
if active_filters:
    cols_f = st.columns(len(active_filters) + 1)
    for i, k in enumerate(active_filters):
        cols_f[i].info(f"{k}: {st.session_state.filters[k]}")
    if cols_f[-1].button("Сбросить"):
        st.session_state.filters = {"purpose": None, "sentiment": None, "hour": None, "type": None, "date": None}
        st.rerun()

col1, col2, col3 = st.columns(3)

# 1.1 Звонки по часам
df_h = data["hourly"]
fig_h_count = px.bar(df_h, x='hour', y='count', color='call_type',
                     labels={'count': 'Кол-во', 'hour': 'Час', 'call_type': 'Тип'},
                     color_discrete_map={'📥': 'green', '📤': 'blue', '🏠': 'orange', '❓': 'gray'},
                     custom_data=['hour'])
fig_h_count.update_layout(xaxis={'tickmode': 'linear', 'tick0': 6, 'dtick': 1}, barmode='stack')

# 1.2 Средняя длительность
def format_mmss(sec):
    if not sec or sec <= 0: return "00:00"
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"

df_h['billsec_str'] = df_h['avg_billsec'].apply(format_mmss)
fig_h_dur = px.bar(df_h.groupby('hour')['avg_billsec'].mean().reset_index(), x='hour', y='avg_billsec',
                   labels={'avg_billsec': 'Время (сек)', 'hour': 'Час'},
                   custom_data=['hour'])
fig_h_dur.update_traces(marker_color='indianred')

# 1.3 Цели
fig_purpose = px.bar(data["purposes"], x='purpose', y='count', labels={'count': 'Кол-во', 'purpose': 'Цель'}, custom_data=['purpose'])

with col1:
    st.subheader("Звонки по часам")
    ev = st.plotly_chart(fig_h_count, width='stretch', on_select="rerun", key="h_chart")
    if ev and ev.selection.get("points"):
        st.session_state.filters["hour"] = int(ev.selection["points"][0].get("customdata", [None])[0])
        st.session_state.show_table = True
        st.rerun()

with col2:
    st.subheader("Средняя длительность")
    st.plotly_chart(fig_h_dur, width='stretch')

with col3:
    st.subheader("Распределение целей")
    ev_p = st.plotly_chart(fig_purpose, width='stretch', on_select="rerun", key="p_chart")
    if ev_p and ev_p.selection.get("points"):
        st.session_state.filters["purpose"] = ev_p.selection["points"][0].get("customdata", [None])[0]
        st.session_state.show_table = True
        st.rerun()

col4, col5 = st.columns([1, 2])

# 2.1 Настроение
fig_sentiment = px.pie(data["sentiment"], values='count', names='sentiment',
                       color='sentiment',
                       color_discrete_map={'Положительное': 'green', 'Нейтральное': 'blue', 'Отрицательное': 'orange', 'Конфликт': 'red'},
                       custom_data=['sentiment'])

# 2.2 Вежливость
fig_poly = px.bar(data["operators"], x='operator_name', y='avg_politeness',
                  labels={'avg_politeness': 'Вежливость', 'operator_name': 'Оператор'},
                  range_y=[0, 10], text='avg_politeness')

with col4:
    st.subheader("Настроение")
    ev_s = st.plotly_chart(fig_sentiment, width='stretch', on_select="rerun", key="s_chart")
    if ev_s and ev_s.selection.get("points"):
        st.session_state.filters["sentiment"] = ev_s.selection["points"][0].get("customdata", [None])[0] or ev_s.selection["points"][0].get("label")
        st.session_state.show_table = True
        st.rerun()

with col5:
    st.subheader("Вежливость по операторам")
    st.plotly_chart(fig_poly, width='stretch')

if st.button("Показать все звонки"):
    st.session_state.filters = {"purpose": None, "sentiment": None, "hour": None, "type": None, "date": None}
    st.session_state.show_table = True
    st.rerun()

if st.session_state.show_table:
    df_calls = data["calls"]
    st.markdown("---")
    st.subheader(f"Список звонков ({len(df_calls)})")

    status_icons = {'done': '✅', 'processing': '⏳', 'skipped': '⏭️', 'error': '❌', 'new': '🆕', 'empty': '😶', 'stop': '🛑'}
    df_calls['Статус'] = df_calls['processing_status'].map(lambda x: status_icons.get(x, '❓'))

    # Флаги чек-листа
    def get_flag(c, k):
        return "✅" if (c and isinstance(c, dict) and c.get(k)) else "❌"

    keys = [('greeting', '👋'), ('introduced_himself', '🆔'), ('agreed_datetime', '📅'),
            ('identified_need', '🎯'), ('informed_price', '💰'), ('handled_objection', '🛠️'), ('farewell', '🤝')]

    for k, icon in keys:
        df_calls[icon] = df_calls['checklist_json'].apply(lambda x: get_flag(x, k))

    df_calls['Продолжительность'] = df_calls['billsec'].apply(format_mmss)

    display_cols = ['Статус', 'calldate', 'call_type', 'client_number', 'operator_name', 'Продолжительность', 'purpose', 'sentiment', 'call_summary'] + [icon for k, icon in keys]

    selection = st.dataframe(
        df_calls[display_cols],
        column_config={
            "calldate": st.column_config.DatetimeColumn("Дата/время", format="DD.MM.YYYY HH:mm"),
            "call_type": "📞", "Продолжительность": "⏱️", "purpose": "Цель", "sentiment": "Настроение", "call_summary": "Суть"
        },
        hide_index=True, on_select="rerun", selection_mode="single-row", key="calls_table"
    )

    if selection and selection.selection.rows:
        idx = selection.selection.rows[0]
        if idx < len(df_calls):
            lid = df_calls.iloc[idx]['linkedid']
            st.markdown("---")
            st.subheader(f"Прослушивание: {lid}")
            fpath = get_call_file_path(lid)
            if fpath and os.path.exists(fpath): st.audio(fpath)
            else: st.error("Файл не найден")

            st.markdown("#### Расшифровка")
            t_rows = get_call_transcript(lid)
            if t_rows:
                for tr in t_rows:
                    m, s = divmod(int(tr['start_time']), 60)
                    lbl = "👤 **Оператор**" if tr['channel'] == 'operator' else "👥 **Клиент**"
                    st.markdown(f"[{m:02d}:{s:02d}] {lbl}: {tr['text']}")
                with st.expander("Текст для копирования"):
                    st.text_area("Диалог", format_dialogue(t_rows), height=300)
