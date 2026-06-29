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
from db_utils import (
    get_all_prompts, get_default_prompt, get_call_file_path,
    get_call_transcript, format_dialogue, get_value_mappings,
    build_case_sql
)

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
def get_report_data(start_date, end_date, prompt_id, filters, agg_col, agg_type, y_axis_col, time_toggle, time_res, color_col=None, sort_axis="x_val", sort_dir="ASC", is_periodic=False):
    engine = get_engine()

    # 1. Формируем WHERE
    where_clauses = ["e.prompt_id = :pid", "c.calldate >= :start", "c.calldate < :end"]
    params = {
        "pid": prompt_id,
        "start": start_date,
        "end": end_date + timedelta(days=1)
    }

    # Получаем динамические маппинги
    all_mappings = get_value_mappings()
    mapping = next((m for m in all_mappings if m['prompt_id'] == prompt_id), None)

    purpose_sql = build_case_sql('call_purpose', mapping['call_purpose'] if mapping else [], 'Другое')
    sentiment_sql = build_case_sql('client_sentiment', mapping['client_sentiment'] if mapping else [], 'Не определено')

    def get_sql_col(col, as_numeric=False):
        expr = ""
        if col.startswith("checklist."):
            key = col.split('.')[1]
            expr = f"e.checklist_json->>'{key}'"
            if as_numeric:
                # Конвертируем true/false в 1/0 для агрегаций
                expr = f"CASE WHEN {expr} = 'true' THEN 1 WHEN {expr} = 'false' THEN 0 ELSE NULL END"
        elif col.startswith("metrics."):
            key = col.split('.')[1]
            expr = f"e.metrics_json->>'{key}'"
            if as_numeric:
                # Обрабатываем как числа, так и булевы значения (true/false -> 1/0)
                # Оборачиваем в COALESCE для корректной работы агрегатных функций если ключа нет
                expr = f"CASE WHEN {expr} = 'true' THEN 1 WHEN {expr} = 'false' THEN 0 ELSE COALESCE(({expr})::numeric, 0) END"
        elif col == "operator_name":
            expr = "COALESCE(p.name, CASE WHEN c.direction = 'incoming' THEN c.answeredext ELSE c.src END)"
        elif col == "client_number":
            expr = "CASE WHEN c.direction = 'incoming' THEN c.src ELSE c.answeredext END"
        elif col == "call_purpose":
            expr = purpose_sql
        elif col == "client_sentiment":
            expr = sentiment_sql
        else:
            expr = f"c.{col}" if col in ["calldate", "direction", "duration", "billsec"] else f"e.{col}"
            if as_numeric and col in ["duration", "billsec", "politeness_score"]:
                expr = f"({expr})::numeric"
        return expr

    for i, f in enumerate(filters):
        col = f['column']
        op = f['op']

        # Для сравнений больше/меньше нам нужно числовое представление
        is_num_op = op in ["больше", "меньше"]
        col_sql = get_sql_col(col, as_numeric=is_num_op)
        val = f['value']
        is_not = f.get('not', False)

        p_name = f"v_{i}"
        clause = ""

        if op == "заполнено":
            clause = f"{col_sql} IS NOT NULL"
        elif op == "не заполнено":
            clause = f"{col_sql} IS NULL"
        elif op == "равно":
            if isinstance(val, list):
                if not val: continue
                clause = f"{col_sql} IN :v_{i}"
                params[p_name] = tuple(val)
            else:
                clause = f"{col_sql} = :v_{i}"
                params[p_name] = val
        elif op == "больше":
            clause = f"{col_sql} > :v_{i}"
            params[p_name] = float(val)
        elif op == "меньше":
            clause = f"{col_sql} < :v_{i}"
            params[p_name] = float(val)
        elif op == "содержит":
            clause = f"{col_sql} ILIKE :v_{i}"
            params[p_name] = f"%{val}%"
        elif op == "начинается с":
            clause = f"{col_sql} ILIKE :v_{i}"
            params[p_name] = f"{val}%"

        if clause:
            if is_not:
                where_clauses.append(f"NOT ({clause})")
            else:
                where_clauses.append(clause)

    where_str = " AND ".join(where_clauses)

    # 2. Формируем проекцию X-оси
    if agg_col == "total":
        x_sql = "'Все данные'"
    elif time_toggle:
        if time_res == "День":
            x_sql = "DATE(c.calldate)"
        elif time_res == "День недели":
            if is_periodic:
                x_sql = """CASE EXTRACT(DOW FROM c.calldate)
                    WHEN 1 THEN '1. Понедельник'
                    WHEN 2 THEN '2. Вторник'
                    WHEN 3 THEN '3. Среда'
                    WHEN 4 THEN '4. Четверг'
                    WHEN 5 THEN '5. Пятница'
                    WHEN 6 THEN '6. Суббота'
                    WHEN 0 THEN '7. Воскресенье'
                    ELSE 'Неизвестно' END"""
            else:
                x_sql = "DATE(c.calldate)"
        elif time_res == "Час":
            if is_periodic:
                # Экранируем двоеточие для SQLAlchemy, чтобы не считалось за именованный параметр
                x_sql = r"LPAD(EXTRACT(HOUR FROM c.calldate)::text, 2, '0') || '\:00'"
            else:
                x_sql = r"TO_CHAR(c.calldate, 'YYYY-MM-DD HH24\:00')"
        else:
            x_sql = "DATE(c.calldate)"
    else:
        x_sql = get_sql_col(agg_col)

    # 3. Формируем запрос для графиков (Агрегация на стороне БД)
    y_sql = "*"
    if agg_type == "Количество":
        y_sql = "COUNT(*)"
    elif agg_type == "Сумма":
        y_sql = f"SUM({get_sql_col(y_axis_col, as_numeric=True)})"
    elif agg_type == "Среднее":
        y_sql = f"ROUND(AVG({get_sql_col(y_axis_col, as_numeric=True)}), 2)"
    elif agg_type == "Процент":
        y_sql = "COUNT(*)" # Процент посчитаем в пандасе из долей

    color_select = ""
    color_group = ""
    if color_col:
        color_sql = get_sql_col(color_col)
        color_select = f", {color_sql} as color_val"
        color_group = ", 3"

    # Если есть сортировка по Y и сегментация, нам нужно сортировать по сумме всего стека
    final_sort = f"{sort_axis} {sort_dir}"
    if sort_axis == "y_val" and color_col:
        # Используем оконную функцию, чтобы получить общую величину всего X-значения для сортировки
        if agg_type == "Среднее":
            # Для среднего по группе X мы считаем взвешенное среднее по всем сегментам
            col_num = get_sql_col(y_axis_col, as_numeric=True)
            y_sort_expr = f"SUM(SUM({col_num})) OVER(PARTITION BY {x_sql}) / NULLIF(SUM(COUNT({col_num})) OVER(PARTITION BY {x_sql}), 0)"
        elif agg_type == "Процент":
            y_sort_expr = f"SUM(COUNT(*)) OVER(PARTITION BY {x_sql})"
        elif agg_type == "Сумма":
            col_num = get_sql_col(y_axis_col, as_numeric=True)
            y_sort_expr = f"SUM(SUM({col_num})) OVER(PARTITION BY {x_sql})"
        else:
            # Количество
            y_sort_expr = f"SUM(COUNT(*)) OVER(PARTITION BY {x_sql})"

        # Внешний SELECT должен использовать алиасы из подзапроса
        outer_color = ", color_val" if color_col else ""
        query_agg = f"""
            SELECT x_val, y_val {outer_color} FROM (
                SELECT
                    {x_sql} as x_val,
                    {y_sql} as y_val
                    {color_select},
                    {y_sort_expr} as total_y
                FROM calls c
                JOIN evaluations e ON c.linkedid = e.linkedid
                LEFT JOIN phones p ON p.number = (CASE WHEN c.direction = 'incoming' THEN c.answeredext ELSE c.src END)
                WHERE {where_str}
                GROUP BY 1 {color_group}
            ) sub
            ORDER BY total_y {sort_dir}, x_val, color_val
        """
    else:
        query_agg = f"""
            SELECT {x_sql} as x_val, {y_sql} as y_val {color_select}
            FROM calls c
            JOIN evaluations e ON c.linkedid = e.linkedid
            LEFT JOIN phones p ON p.number = (CASE WHEN c.direction = 'incoming' THEN c.answeredext ELSE c.src END)
            WHERE {where_str}
            GROUP BY 1 {color_group}
            ORDER BY {final_sort}
        """

    # 4. Запрос для детализации (Лимит 1000 для скорости)
    query_details = f"""
        SELECT
            c.calldate, c.direction,
            CASE WHEN c.direction = 'incoming' THEN c.src ELSE c.answeredext END as client_number,
            COALESCE(p.name, CASE WHEN c.direction = 'incoming' THEN c.answeredext ELSE c.src END) as operator_name,
            c.duration, c.billsec, {purpose_sql} as call_purpose, {sentiment_sql} as client_sentiment,
            e.politeness_score, e.call_summary, e.checklist_json, c.linkedid,
            {x_sql} as x_val
        FROM calls c
        JOIN evaluations e ON c.linkedid = e.linkedid
        LEFT JOIN phones p ON p.number = (CASE WHEN c.direction = 'incoming' THEN c.answeredext ELSE c.src END)
        WHERE {where_str}
        ORDER BY c.calldate DESC
        LIMIT 1000
    """

    with engine.connect() as conn:
        df_agg = pd.read_sql(text(query_agg), conn, params=params)
        df_details = pd.read_sql(text(query_details), conn, params=params)

    return df_agg, df_details

@st.cache_data(ttl=300)
def get_distinct_values(prompt_id, column):
    engine = get_engine()
    col_sql = column
    if column.startswith("checklist."):
        col_sql = f"checklist_json->>'{column.split('.')[1]}'"
    elif column.startswith("metrics."):
        col_sql = f"metrics_json->>'{column.split('.')[1]}'"
    elif column == "operator_name":
        col_sql = "COALESCE(p.name, CASE WHEN c.direction = 'incoming' THEN c.answeredext ELSE c.src END)"
    elif column == "client_number":
        col_sql = "CASE WHEN c.direction = 'incoming' THEN c.src ELSE c.answeredext END"
    else:
        col_sql = f"c.{column}" if column in ["direction"] else f"e.{column}"

    query = f"""
        SELECT DISTINCT {col_sql} as val
        FROM calls c
        JOIN evaluations e ON c.linkedid = e.linkedid
        LEFT JOIN phones p ON p.number = (CASE WHEN c.direction = 'incoming' THEN c.answeredext ELSE c.src END)
        WHERE e.prompt_id = :pid AND {col_sql} IS NOT NULL
        ORDER BY 1
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params={"pid": prompt_id})
    return sorted([str(x) for x in df['val'].tolist()])

def get_report_json_keys(prompt_id, start_date=None, end_date=None):
    engine = get_engine()
    where = "e.prompt_id = :pid"
    params = {"pid": prompt_id}

    if start_date and end_date:
        where += " AND c.calldate >= :start AND c.calldate < :end"
        params["start"] = start_date
        params["end"] = end_date + timedelta(days=1)

    with engine.connect() as conn:
        # Берем 50 последних записей за указанный период
        res = conn.execute(text(f"""
            SELECT e.checklist_json, e.metrics_json
            FROM evaluations e
            JOIN calls c ON c.linkedid = e.linkedid
            WHERE {where}
            ORDER BY c.calldate DESC
            LIMIT 50
        """), params).fetchall()

        checklist_keys = set()
        metrics_keys = set()
        for row in res:
            if row[0] and isinstance(row[0], dict):
                checklist_keys.update(row[0].keys())
            if row[1] and isinstance(row[1], dict):
                metrics_keys.update(row[1].keys())

        return sorted([f"checklist.{k}" for k in checklist_keys]) + sorted([f"metrics.{k}" for k in metrics_keys])

def get_saved_reports():
    engine = get_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT id, name, settings FROM reports ORDER BY name ASC"), conn)
            return df
    except:
        return pd.DataFrame(columns=['id', 'name', 'settings'])

def save_report(name, settings):
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO reports (name, settings) VALUES (:name, :settings)
            ON CONFLICT (name) DO UPDATE SET settings = EXCLUDED.settings
        """), {"name": name, "settings": json.dumps(settings)})
        conn.commit()

def delete_report(report_id):
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM reports WHERE id = :id"), {"id": report_id})
        conn.commit()

# Словарь для отображения имен колонок
column_labels = {
    "total": "-- Всего --",
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
if "active_report_name" not in st.session_state:
    st.session_state.active_report_name = None
if "viz_settings" not in st.session_state:
    st.session_state.viz_settings = {}

# --- Боковая панель (Настройки отчета) ---
with st.sidebar:
    st.header("Параметры отчета")

    # Сохраненные отчеты
    saved_reports = get_saved_reports()
    report_names = ["-- Новый отчет --"] + saved_reports['name'].tolist()
    selected_report_name = st.selectbox("Загрузить отчет", report_names)

    if selected_report_name != "-- Новый отчет --":
        if st.session_state.active_report_name != selected_report_name:
            report_data = saved_reports[saved_reports['name'] == selected_report_name].iloc[0]
            s = report_data['settings']
            if isinstance(s, str): s = json.loads(s)
            st.session_state.report_filters = s.get('filters', [])
            st.session_state.last_prompt_id = s.get('prompt_id')
            st.session_state.viz_settings = {
                "agg_col": s.get("agg_col"),
                "color_col": s.get("color_col"),
                "barmode": s.get("barmode", "stack"),
                "agg_type": s.get("agg_type"),
                "y_axis_col": s.get("y_axis_col"),
                "chart_type": s.get("chart_type"),
                "time_toggle": s.get("time_toggle"),
                "time_res": s.get("time_res"),
                "is_periodic": s.get("is_periodic", False),
                "sort_axis": s.get("sort_axis", "x_val"),
                "sort_dir": s.get("sort_dir", "ASC")
            }
            st.session_state.active_report_name = selected_report_name
            st.rerun()

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
        st.warning("Промпты не найдены.")
        st.stop()

    prompt_options = {p['id']: p['name'] for p in all_prompts}
    default_p = get_default_prompt()
    default_p_id = st.session_state.last_prompt_id or (default_p['id'] if default_p else all_prompts[0]['id'])

    selected_prompt_id = st.selectbox(
        "Аналитический промпт",
        options=list(prompt_options.keys()),
        format_func=lambda x: prompt_options[x],
        index=list(prompt_options.keys()).index(default_p_id) if default_p_id in prompt_options else 0
    )

    if st.session_state.last_prompt_id != selected_prompt_id:
        st.session_state.report_filters = []
        st.session_state.last_prompt_id = selected_prompt_id

    # Получаем ключи JSON за выбранный период
    json_keys = get_report_json_keys(selected_prompt_id, start_date, end_date)

    base_columns = ["direction", "duration", "billsec", "politeness_score", "client_sentiment", "call_purpose", "operator_name", "client_number"]
    all_available_columns = base_columns + json_keys

    # Для X-оси
    x_axis_columns = ["total"] + all_available_columns

    st.subheader("Фильтры")
    if st.button("Добавить фильтр"):
        st.session_state.report_filters.append({"column": all_available_columns[0], "op": "равно", "value": "", "not": False})

    for i, f in enumerate(st.session_state.report_filters):
        with st.expander(f"Фильтр {i+1}: {format_col_name(f['column'])}"):
            f['column'] = st.selectbox(f"Поле", all_available_columns,
                                      index=all_available_columns.index(f['column']) if f['column'] in all_available_columns else 0,
                                      format_func=format_col_name, key=f"col_{i}")

            c1, c2 = st.columns([1, 4])
            with c1:
                f['not'] = st.checkbox("НЕ", value=f.get('not', False), key=f"not_{i}")
            with c2:
                ops = ["равно", "больше", "меньше", "содержит", "начинается с", "заполнено", "не заполнено"]
                f['op'] = st.selectbox(f"Операция", ops, index=ops.index(f['op']) if f['op'] in ops else 0, key=f"op_{i}")

            if f['op'] not in ["заполнено", "не заполнено"]:
                if f['op'] == "равно":
                    unique_vals = get_distinct_values(selected_prompt_id, f['column'])
                    if unique_vals:
                        if isinstance(f['value'], list): current_vals = [str(v) for v in f['value']]
                        else: current_vals = [str(f['value'])] if f['value'] else []
                        for v in current_vals:
                            if v not in unique_vals: unique_vals = [v] + unique_vals
                        f['value'] = st.multiselect(f"Значение", unique_vals, default=current_vals, key=f"val_{i}")
                    else:
                        f['value'] = st.text_input(f"Значение", value=f['value'], key=f"val_{i}")
                else:
                    f['value'] = st.text_input(f"Значение", value=f['value'], key=f"val_{i}")

            if st.button(f"Удалить", key=f"del_{i}"):
                st.session_state.report_filters.pop(i)
                st.rerun()

    st.subheader("Визуализация")
    saved_agg_col = st.session_state.viz_settings.get("agg_col")
    agg_col = st.selectbox("Группировка (X-ось)", x_axis_columns,
                           index=x_axis_columns.index(saved_agg_col) if saved_agg_col in x_axis_columns else (x_axis_columns.index("call_purpose") if "call_purpose" in x_axis_columns else 0),
                           format_func=format_col_name)

    saved_color_col = st.session_state.viz_settings.get("color_col")
    color_options = [None] + all_available_columns
    def format_color_col(col):
        if col is None: return "-- Без сегментации --"
        return format_col_name(col)

    color_col = st.selectbox("Сегментация (Цвет)", color_options,
                             index=color_options.index(saved_color_col) if saved_color_col in color_options else 0,
                             format_func=format_color_col)

    agg_types = ["Количество", "Сумма", "Среднее", "Процент"]
    saved_agg_type = st.session_state.viz_settings.get("agg_type")
    agg_type = st.selectbox("Тип агрегации (Y-ось)", agg_types, index=agg_types.index(saved_agg_type) if saved_agg_type in agg_types else 0)

    y_axis_col = None
    if agg_type in ["Сумма", "Среднее"]:
        y_axis_options = [c for c in base_columns if c in ["duration", "billsec", "politeness_score"]] + json_keys
        saved_y_axis = st.session_state.viz_settings.get("y_axis_col")
        y_axis_col = st.selectbox("Поле для расчета", y_axis_options,
                                  index=y_axis_options.index(saved_y_axis) if saved_y_axis in y_axis_options else 0,
                                  format_func=format_col_name)

    chart_type_map = {
        "Столбчатая (Stacked Bar Chart)": "bar_stack",
        "Столбчатая (Grouped Bar Chart)": "bar_group",
        "Линейная (Line Chart)": "line",
        "Круговая (Pie Chart)": "pie",
        "Область (Area Chart)": "area"
    }
    saved_chart_type = st.session_state.viz_settings.get("chart_type")
    inv_chart_map = {v: k for k, v in chart_type_map.items()}
    default_chart_label = inv_chart_map.get(saved_chart_type, "Столбчатая (Stacked Bar Chart)")
    selected_chart_label = st.selectbox("Тип диаграммы", list(chart_type_map.keys()), index=list(chart_type_map.keys()).index(default_chart_label))
    chart_type = chart_type_map[selected_chart_label]

    # barmode определяется типом диаграммы
    barmode = "stack"
    if chart_type == "bar_group":
        barmode = "group"

    # Отключаем ось времени для 'total'
    is_total = agg_col == "total"
    time_toggle = st.checkbox("Ось времени", value=st.session_state.viz_settings.get("time_toggle", False) if not is_total else False, disabled=is_total)

    time_res_options = ["День", "День недели", "Час"]
    saved_time_res = st.session_state.viz_settings.get("time_res")
    time_res = st.radio("Детализация времени", time_res_options,
                        index=time_res_options.index(saved_time_res) if saved_time_res in time_res_options else 0,
                        label_visibility="collapsed", disabled=not time_toggle)

    is_periodic = st.checkbox("Периодически",
                              value=st.session_state.viz_settings.get("is_periodic", False),
                              disabled=not time_toggle or time_res == "День")

    st.subheader("Сортировка")
    c_sort1, c_sort2 = st.columns(2)
    with c_sort1:
        sort_axis_options = {"x_val": "Ось X", "y_val": "Ось Y"}
        saved_sort_axis = st.session_state.viz_settings.get("sort_axis", "x_val")
        # Обеспечиваем корректный индекс, если значение в сессии внезапно некорректно
        ax_idx = list(sort_axis_options.keys()).index(saved_sort_axis) if saved_sort_axis in sort_axis_options else 0
        sort_axis = st.radio("По оси", options=list(sort_axis_options.keys()),
                             format_func=lambda x: sort_axis_options[x],
                             index=ax_idx,
                             key=f"sort_axis_radio_{selected_report_name}",
                             label_visibility="collapsed")
    with c_sort2:
        sort_dir_options = {"ASC": "⬇️", "DESC": "⬆️"}
        saved_sort_dir = st.session_state.viz_settings.get("sort_dir", "ASC")
        dir_idx = list(sort_dir_options.keys()).index(saved_sort_dir) if saved_sort_dir in sort_dir_options else 0
        sort_dir = st.radio("Направление", options=list(sort_dir_options.keys()),
                            format_func=lambda x: sort_dir_options[x],
                            index=dir_idx,
                            key=f"sort_dir_radio_{selected_report_name}",
                            label_visibility="collapsed")

    st.markdown("---")
    if st.button("Применить", type="primary", width="stretch"):
        st.session_state.applied_settings = {
            "start_date": start_date, "end_date": end_date, "prompt_id": selected_prompt_id,
            "filters": [f.copy() for f in st.session_state.report_filters],
            "agg_col": agg_col, "color_col": color_col, "barmode": barmode, "agg_type": agg_type, "y_axis_col": y_axis_col,
            "chart_type": chart_type, "time_toggle": time_toggle, "time_res": time_res, "is_periodic": is_periodic,
            "sort_axis": sort_axis, "sort_dir": sort_dir,
            "report_name": st.session_state.active_report_name
        }

    col_save, col_del = st.columns(2)
    with col_save:
        if st.button("Сохранить"): st.session_state.show_save_dialog = True
    with col_del:
        if st.session_state.active_report_name and st.button("Удалить"):
            rep_id = saved_reports[saved_reports['name'] == st.session_state.active_report_name].iloc[0]['id']
            delete_report(rep_id)
            st.session_state.active_report_name = None
            st.rerun()

    if st.session_state.get("show_save_dialog"):
        with st.form("save_report_form"):
            new_name = st.text_input("Имя отчета", value=st.session_state.active_report_name or "")
            if st.form_submit_button("Подтвердить"):
                rep_settings = {
                    "prompt_id": selected_prompt_id, "filters": st.session_state.report_filters,
                    "agg_col": agg_col, "color_col": color_col, "barmode": barmode, "agg_type": agg_type, "y_axis_col": y_axis_col,
                    "chart_type": chart_type, "time_toggle": time_toggle, "time_res": time_res, "is_periodic": is_periodic,
                    "sort_axis": sort_axis, "sort_dir": sort_dir
                }
                save_report(new_name, rep_settings)

                # Обновляем состояние, чтобы после rerun всё отобразилось корректно
                st.session_state.active_report_name = new_name
                st.session_state.viz_settings = rep_settings.copy()
                if st.session_state.applied_settings:
                    st.session_state.applied_settings.update(rep_settings)
                    st.session_state.applied_settings["report_name"] = new_name

                st.session_state.show_save_dialog = False
                st.rerun()

# --- Основная область ---
if st.session_state.applied_settings is None:
    st.info("Настройте параметры в боковой панели и нажмите 'Применить'.")
    st.stop()

settings = st.session_state.applied_settings
try:
    df_agg, df_details = get_report_data(
        settings["start_date"], settings["end_date"], settings["prompt_id"],
        settings["filters"], settings["agg_col"], settings["agg_type"],
        settings["y_axis_col"], settings["time_toggle"], settings["time_res"],
        settings.get("color_col"),
        settings.get("sort_axis", "x_val"), settings.get("sort_dir", "ASC"),
        is_periodic=settings.get("is_periodic", False)
    )
except Exception as e:
    st.error(f"Ошибка при формировании отчета: {e}")
    st.stop()

if df_agg.empty:
    st.info("Данные не найдены.")
else:
    title_prefix = f"Отчет: {settings['report_name']}" if settings.get('report_name') else "Конструктор отчетов"

    fig = None
    labels_map = {
        'y_val': settings["agg_type"],
        'x_val': format_col_name(settings["agg_col"]),
        'color_val': format_col_name(settings.get("color_col")) if settings.get("color_col") else None
    }
    color_p = "color_val" if settings.get("color_col") else None

    # Общая предобработка данных для всех типов графиков
    df_agg['x_val'] = df_agg['x_val'].fillna("Не определено").astype(str).str.strip()
    if color_p:
        df_agg[color_p] = df_agg[color_p].fillna("Не определено").astype(str).str.strip()
        # Для сегментации тоже применяем ДА/НЕТ если это булево поле
        if settings["color_col"].startswith("checklist.") or (settings["color_col"].startswith("metrics.") and df_agg[color_p].isin(['0', '1', '0.0', '1.0', 'true', 'false']).all()):
            bool_map = {'1': 'ДА', '1.0': 'ДА', 'true': 'ДА', '0': 'НЕТ', '0.0': 'НЕТ', 'false': 'НЕТ'}
            df_agg[color_p] = df_agg[color_p].map(lambda x: bool_map.get(str(x).lower(), x))

    # Гарантируем уникальность пар (x_val, color_val) и заполняем пустоты нулями для правильного стекирования
    group_cols = ['x_val']
    if color_p: group_cols.append(color_p)

    if settings["agg_type"] == "Среднее":
        # Используем sort=False чтобы сохранить порядок сортировки из БД
        df_agg = df_agg.groupby(group_cols, as_index=False, sort=False)['y_val'].mean()
    else:
        df_agg = df_agg.groupby(group_cols, as_index=False, sort=False)['y_val'].sum()

    df_agg['y_val'] = df_agg['y_val'].fillna(0)

    # Проверка, является ли поле булевым (чеклист или булева метрика)
    # Если да, мы будем отображать ДА/НЕТ вместо 1/0
    is_boolean_field = settings["agg_col"].startswith("checklist.") or \
                        (settings["agg_col"].startswith("metrics.") and df_agg['x_val'].isin(['0', '1', '0.0', '1.0', 'true', 'false']).all())

    if settings["chart_type"] in ["bar_stack", "bar_group"]:
        bm = "stack" if settings["chart_type"] == "bar_stack" else "group"

        if is_boolean_field:
            bool_map = {'1': 'ДА', '1.0': 'ДА', 'true': 'ДА', '0': 'НЕТ', '0.0': 'НЕТ', 'false': 'НЕТ'}
            df_agg['x_val'] = df_agg['x_val'].map(lambda x: bool_map.get(str(x).lower(), x))

        if settings["agg_type"] == "Процент":
            if color_p:
                if bm == "stack":
                    # Для стекируемых процентов нормализуем на 100% внутри каждого столбца
                    fig = px.bar(df_agg, x='x_val', y='y_val', color=color_p,
                                 barmode=bm,
                                 labels=labels_map, custom_data=['x_val'])
                    fig.update_layout(yaxis_title="Доля (%) внутри категории", barnorm='percent')
                else:
                    # Для группированных - считаем процент внутри категории X
                    df_agg['y_val'] = df_agg.groupby('x_val', group_keys=False, sort=False)['y_val'].apply(lambda x: (x / x.sum() * 100).round(2))
                    fig = px.bar(df_agg, x='x_val', y='y_val', color=color_p, text='y_val',
                                 barmode=bm, labels=labels_map, custom_data=['x_val'])
                    fig.update_layout(yaxis_title="Доля (%) внутри категории")
            else:
                # Если сегментации нет - процент от общего итога
                total = df_agg['y_val'].sum()
                if total > 0:
                    df_agg['y_val'] = (df_agg['y_val'] / total * 100).round(2)
                fig = px.bar(df_agg, x='x_val', y='y_val', text='y_val',
                             barmode=bm, labels=labels_map, custom_data=['x_val'])
                fig.update_layout(yaxis_title="Доля (%) от общего итога")
        else:
            # Обычные значения (Количество, Сумма, Среднее)
            # Обработка подписей для булевых значений (ДА/НЕТ)
            if settings["y_axis_col"] and (settings["y_axis_col"].startswith("checklist.") or settings["y_axis_col"].startswith("metrics.")):
                # Если мы считаем среднее/сумму по булеву полю
                # но подпись 'ДА/НЕТ' применима скорее к группировке.
                # Для Y-оси оставим числа, но если это Percentage - они и так в %
                pass

            if bm == "stack":
                fig = px.bar(df_agg, x='x_val', y='y_val', color=color_p,
                             barmode=bm, labels=labels_map, custom_data=['x_val'])

                # Если Y это булево значение и мы не в режиме процента, можно попробовать подменить текст
                # Но обычно на Y-оси при агрегации (Сумма/Среднее) получаются дробные числа или суммы > 1
                fig.update_traces(texttemplate='%{y}', textposition='inside')
            else:
                fig = px.bar(df_agg, x='x_val', y='y_val', color=color_p, text='y_val',
                             barmode=bm, labels=labels_map, custom_data=['x_val'])

        # ЖЕСТКОЕ ПРАВИЛО ДЛЯ STREAMLIT 1.58.0+:
        fig.update_layout(
            barmode=bm,
            xaxis=dict(type='category', categoryorder='trace')
        )
        # Если это стекируемый график, принудительно объединяем в одну группу смещения
        if bm == 'stack':
            fig.update_traces(offsetgroup=0)
    elif settings["chart_type"] == "line":
        fig = px.line(df_agg, x='x_val', y='y_val', color=color_p, markers=True, labels=labels_map, custom_data=['x_val'])
    elif settings["chart_type"] == "pie":
        # Круговая диаграмма со вторым измерением — это сомнительно, но сделаем через 'names'/'values'
        fig = px.pie(df_agg, names='x_val', values='y_val', labels=labels_map, custom_data=['x_val'])
    elif settings["chart_type"] == "area":
        fig = px.area(df_agg, x='x_val', y='y_val', color=color_p, labels=labels_map, custom_data=['x_val'])
        fig.update_layout(xaxis=dict(type='category', categoryorder='trace'))

    st.subheader(f"{title_prefix}: {settings['agg_type']} по {format_col_name(settings['agg_col'])}")

    selected_points = st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        key="report_chart",
        theme=None
    )

    filtered_selection = df_details.copy()
    if selected_points and selected_points.selection.get("points"):
        point = selected_points.selection["points"][0]
        val = point.get("x") or point.get("label") or (point.get("customdata", [None])[0])
        if val is not None:
            filtered_selection = filtered_selection[filtered_selection['x_val'].astype(str) == str(val)]

    st.markdown("---")
    st.subheader(f"Список звонков ({len(filtered_selection)})")

    def format_mmss(sec):
        if not sec or sec <= 0: return "00:00"
        m, s = divmod(int(sec), 60)
        return f"{m:02d}:{s:02d}"

    def get_flag(c, k):
        return "✅" if (c and isinstance(c, dict) and c.get(k)) else "❌"

    checklist_keys = [
        ('greeting', '👋', 'Приветствие'),
        ('introduced_himself', '🆔', 'Представился'),
        ('agreed_datetime', '📅', 'Договорился о времени'),
        ('identified_need', '🎯', 'Выявил потребность'),
        ('informed_price', '💰', 'Проинформировал о цене'),
        ('handled_objection', '🛠️', 'Отработал возражение'),
        ('farewell', '🤝', 'Прощание')
    ]

    display_df = filtered_selection.copy()
    dir_map = {'incoming': '📥', 'inbound': '📥', 'outgoing': '📤', 'outbound': '📤', 'internal': '🏠'}
    display_df['direction'] = display_df['direction'].apply(lambda x: dir_map.get(str(x).lower(), '❓'))
    display_df['billsec'] = display_df['billsec'].apply(format_mmss)

    for k, icon, label in checklist_keys:
        display_df[icon] = display_df['checklist_json'].apply(lambda x: get_flag(x, k))

    # Порядок: Дата/Время, Направление, Номер клиента, Оператор, Длительность, Цель, Настроение, Суть, показатели
    cols_to_show = ['calldate', 'direction', 'client_number', 'operator_name', 'billsec', 'call_purpose', 'client_sentiment', 'call_summary'] + [icon for k, icon, label in checklist_keys]

    col_config = {
        "calldate": st.column_config.DatetimeColumn("Дата/время", format="DD.MM.YYYY HH:mm"),
        "direction": st.column_config.TextColumn("Напр.", help="Направление"),
        "client_number": "Номер клиента",
        "operator_name": "Оператор",
        "billsec": st.column_config.TextColumn("Длит.", help="Длительность разговора"),
        "call_purpose": "Цель",
        "client_sentiment": "Настроение",
        "call_summary": "Суть"
    }
    for k, icon, label in checklist_keys:
        col_config[icon] = st.column_config.TextColumn(icon, help=label)

    event = st.dataframe(display_df[cols_to_show + ['linkedid']],
                         column_config={**col_config, "linkedid": None},
                         hide_index=True, on_select="rerun", selection_mode="single-row", key="details_table")

    if event and event.selection.rows:
        idx = event.selection.rows[0]
        if idx < len(display_df):
            linkedid = display_df.iloc[idx]['linkedid']
            st.markdown("---")
            st.subheader(f"Детали звонка: {linkedid}")
            fpath = get_call_file_path(linkedid)
            if fpath and os.path.exists(fpath): st.audio(fpath)
            else: st.error("Аудиофайл не найден.")

            st.markdown("#### Расшифровка")
            transcript_rows = get_call_transcript(linkedid)
            if transcript_rows:
                for trow in transcript_rows:
                    m, s = divmod(int(trow['start_time']), 60)
                    st.markdown(f"[{m:02d}:{s:02d}] {'👤 **Оператор**' if trow['channel'] == 'operator' else '👥 **Клиент**'}: {trow['text']}")
                with st.expander("Весь текст для копирования"):
                    st.text_area("Текст диалога", format_dialogue(transcript_rows), height=300)
            else: st.info("Расшифровка отсутствует.")
