import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import sys
import os
import plotly.express as px
from datetime import datetime, timedelta
import json

# Добавляем путь к src, чтобы найти config.py и db_utils.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import PG_CONFIG
from db_utils import get_all_prompts, get_default_prompt, get_call_file_path, get_call_transcript, format_dialogue

# Проверка авторизации
if not st.session_state.get("password_correct", False):
    st.error("Пожалуйста, авторизуйтесь на главной странице.")
    st.stop()

st.set_page_config(page_title="Конструктор отчетов", layout="wide")
st.title("Конструктор отчетов")

@st.cache_resource
def get_engine():
    db_url = f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}"
    return create_engine(db_url)

@st.cache_data(ttl=60)
def get_data(start_date, end_date, prompt_id):
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT
                c.linkedid, c.calldate, c.src, c.answeredext, c.direction,
                c.duration, c.billsec,
                e.politeness_score, e.client_sentiment, e.call_purpose,
                e.checklist_json, e.metrics_json
            FROM calls c
            JOIN evaluations e ON c.linkedid = e.linkedid
            WHERE e.prompt_id = :pid
              AND c.calldate >= :start
              AND c.calldate < :end
        """), conn, params={
            "pid": prompt_id,
            "start": start_date,
            "end": end_date + timedelta(days=1)
        })
    return df

@st.cache_data(ttl=300)
def get_phone_names():
    engine = get_engine()
    with engine.connect() as conn:
        df_phones = pd.read_sql(text("SELECT number, name FROM phones"), conn)
    return dict(zip(df_phones['number'], df_phones['name']))

def get_sample_record(prompt_id):
    engine = get_engine()
    with engine.connect() as conn:
        res = conn.execute(text("""
            SELECT checklist_json, metrics_json
            FROM evaluations
            WHERE prompt_id = :pid
            LIMIT 1
        """), {"pid": prompt_id}).fetchone()
        return res

# Словарь для отображения имен колонок
column_labels = {
    "calldate": "Дата и время",
    "direction": "Направление",
    "duration": "Длительность (общая)",
    "billsec": "Длительность (разговор)",
    "politeness_score": "Вежливость",
    "client_sentiment": "Настроение",
    "call_purpose": "Цель звонка",
    "operator_name": "Имя оператора",
    "client_number": "Номер клиента"
}

def format_col_name(col):
    return column_labels.get(col, col)

# --- Инициализация состояния ---
if "report_filters" not in st.session_state:
    st.session_state.report_filters = []
if "last_prompt_id" not in st.session_state:
    st.session_state.last_prompt_id = None
if "applied_settings" not in st.session_state:
    st.session_state.applied_settings = None

# --- Боковая панель (Настройки отчета) ---
with st.sidebar:
    st.header("Параметры отчета")

    # 1. Выбор периода
    today = datetime.now().date()
    default_start = today - timedelta(days=7)
    date_range = st.date_input("Диапазон дат", (default_start, today))
    if not (isinstance(date_range, tuple) and len(date_range) == 2):
        st.info("Выберите диапазон дат.")
        st.stop()
    start_date, end_date = date_range

    # 2. Выбор промпта
    all_prompts = get_all_prompts()
    if not all_prompts:
        st.warning("Промпты не найдены. Создайте промпт в настройках.")
        st.stop()

    prompt_options = {p['id']: p['name'] for p in all_prompts}
    default_p = get_default_prompt()
    default_p_id = default_p['id'] if default_p else all_prompts[0]['id']

    selected_prompt_id = st.selectbox(
        "Аналитический промпт",
        options=list(prompt_options.keys()),
        format_func=lambda x: prompt_options[x],
        index=list(prompt_options.keys()).index(default_p_id) if default_p_id in prompt_options else 0
    )

    if st.session_state.last_prompt_id != selected_prompt_id:
        st.session_state.report_filters = []
        st.session_state.last_prompt_id = selected_prompt_id

    # Загружаем данные для получения ключей JSON и уникальных значений
    df_for_metadata = get_data(start_date, end_date, selected_prompt_id)
    phone_names = get_phone_names()

    def process_metadata_df(df_m):
        if df_m.empty: return df_m
        def _row(row):
            if row['direction'] == 'incoming':
                op_num = row['answeredext']
                cl_num = row['src']
            else:
                op_num = row['src']
                cl_num = row['answeredext']
            row['operator_name'] = phone_names.get(op_num, op_num)
            row['client_number'] = cl_num
            return row
        df_m = df_m.apply(_row, axis=1)

        # Распаковка JSON
        sample = get_sample_record(selected_prompt_id)
        if sample:
            checklist = sample[0] if isinstance(sample[0], dict) else {}
            metrics = sample[1] if isinstance(sample[1], dict) else {}
            for k in checklist.keys():
                df_m[f"checklist.{k}"] = df_m["checklist_json"].apply(lambda x: x.get(k) if isinstance(x, dict) else None)
            for k in metrics.keys():
                df_m[f"metrics.{k}"] = df_m["metrics_json"].apply(lambda x: x.get(k) if isinstance(x, dict) else None)
        return df_m

    df_for_metadata = process_metadata_df(df_for_metadata)

    # Получаем ключи JSON
    sample = get_sample_record(selected_prompt_id)
    json_keys = []
    if sample:
        checklist = sample[0] if isinstance(sample[0], dict) else {}
        metrics = sample[1] if isinstance(sample[1], dict) else {}
        json_keys = [f"checklist.{k}" for k in checklist.keys()] + [f"metrics.{k}" for k in metrics.keys()]

    # Список доступных полей
    base_columns = [
        "calldate", "direction", "duration", "billsec",
        "politeness_score", "client_sentiment", "call_purpose",
        "operator_name", "client_number"
    ]
    all_available_columns = base_columns + json_keys

    st.subheader("Фильтры")
    if st.button("Добавить фильтр"):
        st.session_state.report_filters.append({"column": all_available_columns[0], "op": "равно", "value": "", "not": False})

    for i, f in enumerate(st.session_state.report_filters):
        with st.expander(f"Фильтр {i+1}: {format_col_name(f['column'])}"):
            f['column'] = st.selectbox(f"Поле", all_available_columns,
                                      index=all_available_columns.index(f['column']),
                                      format_func=format_col_name,
                                      key=f"col_{i}")

            c1, c2 = st.columns([1, 4])
            with c1:
                f['not'] = st.checkbox("НЕ", value=f.get('not', False), key=f"not_{i}")
            with c2:
                ops = ["равно", "больше", "меньше", "содержит", "начинается с", "заполнено", "не заполнено"]
                f['op'] = st.selectbox(f"Операция", ops, index=ops.index(f['op']) if f['op'] in ops else 0, key=f"op_{i}")

            if f['op'] not in ["заполнено", "не заполнено"]:
                # Если "равно", предлагаем выбор из списка (мультиселект)
                if f['op'] == "равно":
                    unique_vals = []
                    if not df_for_metadata.empty and f['column'] in df_for_metadata.columns:
                        unique_vals = sorted([str(x) for x in df_for_metadata[f['column']].unique() if x is not None])

                    if unique_vals:
                        # Если текущее значение (строка или список) содержит элементы не из списка, добавляем их
                        if isinstance(f['value'], list):
                            current_vals = [str(v) for v in f['value']]
                        else:
                            current_vals = [str(f['value'])] if f['value'] else []

                        for v in current_vals:
                            if v not in unique_vals:
                                unique_vals = [v] + unique_vals

                        f['value'] = st.multiselect(f"Значение", unique_vals, default=current_vals, key=f"val_{i}")
                    else:
                        f['value'] = st.text_input(f"Значение", value=f['value'], key=f"val_{i}")
                else:
                    f['value'] = st.text_input(f"Значение", value=f['value'], key=f"val_{i}")

            if st.button(f"Удалить", key=f"del_{i}"):
                st.session_state.report_filters.pop(i)
                st.rerun()

    st.subheader("Визуализация")
    agg_col = st.selectbox("Группировка (X-ось)", all_available_columns,
                           index=all_available_columns.index("call_purpose") if "call_purpose" in all_available_columns else 0,
                           format_func=format_col_name)
    agg_type = st.selectbox("Тип агрегации (Y-ось)", ["Количество", "Сумма", "Среднее", "Процент"])

    y_axis_col = None
    if agg_type in ["Сумма", "Среднее"]:
        y_axis_options = [c for c in base_columns if c in ["duration", "billsec", "politeness_score"]] + json_keys
        y_axis_col = st.selectbox("Поле для расчета", y_axis_options, format_func=format_col_name)

    chart_type_map = {
        "Столбчатая": "bar",
        "Линейная": "line",
        "Круговая": "pie",
        "Область": "area"
    }
    selected_chart_label = st.selectbox("Тип диаграммы", list(chart_type_map.keys()))
    chart_type = chart_type_map[selected_chart_label]

    time_toggle = st.checkbox("Ось времени (Дни/Часы)")
    time_res = "День"
    if time_toggle:
        time_res = st.radio("Детализация", ["День", "Час"])

    if st.button("Применить", type="primary", width="stretch"):
        st.session_state.applied_settings = {
            "start_date": start_date,
            "end_date": end_date,
            "prompt_id": selected_prompt_id,
            "filters": [f.copy() for f in st.session_state.report_filters],
            "agg_col": agg_col,
            "agg_type": agg_type,
            "y_axis_col": y_axis_col,
            "chart_type": chart_type,
            "time_toggle": time_toggle,
            "time_res": time_res,
            "json_keys": json_keys
        }

# --- Основная область ---
if st.session_state.applied_settings is None:
    st.info("Настройте параметры в боковой панели и нажмите 'Применить'.")
    st.stop()

settings = st.session_state.applied_settings

df_raw = get_data(settings["start_date"], settings["end_date"], settings["prompt_id"])

if df_raw.empty:
    st.info("Данные не найдены для выбранных параметров.")
else:
    phone_names = get_phone_names()
    df = df_raw.copy()

    # Предварительная обработка
    def process_row_basic(row):
        if row['direction'] == 'incoming':
            op_num = row['answeredext']
            cl_num = row['src']
        else:
            op_num = row['src']
            cl_num = row['answeredext']
        row['operator_name'] = phone_names.get(op_num, op_num)
        row['client_number'] = cl_num
        return row

    df = df.apply(process_row_basic, axis=1)

    # Распаковка JSON
    for jk in settings["json_keys"]:
        prefix, key = jk.split('.')
        col_name = f"{prefix}_json"
        df[jk] = df[col_name].apply(lambda x: x.get(key) if isinstance(x, dict) else None)

    # Применение фильтров
    for f in settings["filters"]:
        col = f['column']
        op = f['op']
        val = f['value']
        is_not = f.get('not', False)

        mask = pd.Series(True, index=df.index)

        if op == "равно":
            if isinstance(val, list):
                if not val:
                    mask = pd.Series(True, index=df.index)
                else:
                    # Пытаемся сравнить как числа или как строки
                    try:
                        v_list = [float(x) for x in val]
                        mask = df[col].isin(v_list)
                    except:
                        mask = df[col].astype(str).isin([str(x) for x in val])
            else:
                try:
                    v = float(val)
                    mask = (df[col] == v)
                except:
                    mask = (df[col].astype(str) == str(val))
        elif op == "больше":
            mask = (pd.to_numeric(df[col], errors='coerce') > float(val))
        elif op == "меньше":
            mask = (pd.to_numeric(df[col], errors='coerce') < float(val))
        elif op == "содержит":
            mask = (df[col].astype(str).str.contains(str(val), case=False, na=False))
        elif op == "начинается с":
            mask = (df[col].astype(str).str.startswith(str(val), na=False))
        elif op == "заполнено":
            mask = (df[col].notnull())
        elif op == "не заполнено":
            mask = (df[col].isnull())

        if is_not:
            df = df[~mask]
        else:
            df = df[mask]

    if df.empty:
        st.warning("После применения фильтров данных не осталось.")
    else:
        # Подготовка данных для графика
        plot_df = df.copy()

        x_axis = settings["agg_col"]
        if settings["time_toggle"]:
            if settings["time_res"] == "День":
                plot_df['time_axis'] = plot_df['calldate'].dt.date
            else:
                plot_df['time_axis'] = plot_df['calldate'].dt.strftime('%Y-%m-%d %H:00')
            x_axis = 'time_axis'

        # Агрегация
        agg_type = settings["agg_type"]
        y_axis_col = settings["y_axis_col"]

        if agg_type == "Количество":
            res_df = plot_df.groupby(x_axis, observed=False).size().reset_index(name='value')
        elif agg_type == "Сумма":
            plot_df[y_axis_col] = pd.to_numeric(plot_df[y_axis_col], errors='coerce')
            res_df = plot_df.groupby(x_axis, observed=False)[y_axis_col].sum().reset_index(name='value')
        elif agg_type == "Среднее":
            plot_df[y_axis_col] = pd.to_numeric(plot_df[y_axis_col], errors='coerce')
            res_df = plot_df.groupby(x_axis, observed=False)[y_axis_col].mean().reset_index(name='value')
            res_df['value'] = res_df['value'].round(2)
        elif agg_type == "Процент":
            counts = plot_df.groupby(x_axis, observed=False).size().reset_index(name='count')
            total = counts['count'].sum()
            counts['value'] = (counts['count'] / total * 100).round(2)
            res_df = counts

        res_df = res_df.sort_values(x_axis)

        # Отрисовка графика
        fig = None
        labels_map = {'value': agg_type, x_axis: format_col_name(settings["agg_col"])}
        chart_type = settings["chart_type"]
        if chart_type == "bar":
            fig = px.bar(res_df, x=x_axis, y='value', text='value', labels=labels_map, custom_data=[x_axis])
        elif chart_type == "line":
            fig = px.line(res_df, x=x_axis, y='value', markers=True, labels=labels_map, custom_data=[x_axis])
        elif chart_type == "pie":
            fig = px.pie(res_df, names=x_axis, values='value', labels=labels_map, custom_data=[x_axis])
        elif chart_type == "area":
            fig = px.area(res_df, x=x_axis, y='value', labels=labels_map, custom_data=[x_axis])

        st.subheader(f"Отчет: {agg_type} по {format_col_name(settings['agg_col'])}")
        selected_points = st.plotly_chart(fig, width="stretch", on_select="rerun", key="report_chart")

        filtered_selection = df.copy()
        if selected_points and selected_points.selection.get("points"):
            point = selected_points.selection["points"][0]
            # Пытаемся получить значение из разных возможных ключей в зависимости от типа графика
            val = point.get("x")
            if val is None:
                val = point.get("label")
            if val is None:
                val = point.get("customdata", [None])[0]

            if val is not None:
                if settings["time_toggle"]:
                    if settings["time_res"] == "День":
                        # Приводим к строке для сравнения
                        filtered_selection = filtered_selection[filtered_selection['calldate'].dt.date.astype(str) == str(val)]
                    else:
                        filtered_selection = filtered_selection[filtered_selection['calldate'].dt.strftime('%Y-%m-%d %H:00') == str(val)]
                else:
                    filtered_selection = filtered_selection[filtered_selection[settings["agg_col"]].astype(str) == str(val)]

        # --- Детализация ---
        st.markdown("---")
        st.subheader(f"Список звонков ({len(filtered_selection)})")

        display_df = filtered_selection[['calldate', 'direction', 'client_number', 'operator_name', 'duration', 'call_purpose', 'politeness_score', 'linkedid']].copy()
        display_df.columns = ['Дата/время', 'Тип', 'Клиент', 'Оператор', 'Длительность', 'Цель', 'Вежливость', 'linkedid']

        dir_map = {'incoming': '📥', 'inbound': '📥', 'outgoing': '📤', 'outbound': '📤', 'internal': '🏠'}
        display_df['Тип'] = display_df['Тип'].apply(lambda x: dir_map.get(str(x).lower(), '❓'))

        event = st.dataframe(
            display_df,
            column_config={
                "Дата/время": st.column_config.DatetimeColumn(format="DD.MM.YYYY HH:mm"),
                "linkedid": None
            },
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="details_table"
        )

        if event and event.selection.rows:
            idx = event.selection.rows[0]
            if idx < len(display_df):
                selected_row = display_df.iloc[idx]
                linkedid = selected_row['linkedid']

            st.markdown("---")
            st.subheader(f"Детали звонка: {linkedid}")

            fpath = get_call_file_path(linkedid)
            if fpath and os.path.exists(fpath):
                st.audio(fpath)
            else:
                st.error("Аудиофайл не найден.")

            st.markdown("#### Расшифровка")
            transcript_rows = get_call_transcript(linkedid)
            if transcript_rows:
                for trow in transcript_rows:
                    m, s = divmod(int(trow['start_time']), 60)
                    time_str = f"[{m:02d}:{s:02d}]"
                    label = "👤 **Оператор**" if trow['channel'] == 'operator' else "👥 **Клиент**"
                    st.markdown(f"{time_str} {label}: {trow['text']}")

                with st.expander("Весь текст для копирования"):
                    st.text_area("Текст диалога", format_dialogue(transcript_rows), height=300)
            else:
                st.info("Расшифровка отсутствует.")
