import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

# Добавляем путь к src, чтобы найти config.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import PG_CONFIG
from db_utils import get_all_prompts, get_default_prompt

if not st.session_state.get("password_correct", False):
    st.error("Пожалуйста, авторизуйтесь на главной странице.")
    st.stop()

st.title("Аналитика звонков")

from sqlalchemy import text
from datetime import datetime, timedelta

@st.cache_data(ttl=60)
def get_summary_data(start_date, end_date, prompt_id):
    # Формируем SQLAlchemy URL
    db_url = f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}"
    engine = create_engine(db_url)
    with engine.connect() as conn:
        # Увеличиваем end_date на 1 день для корректного захвата конца последнего дня
        df = pd.read_sql(text("""
        SELECT
            c.linkedid,
            c.calldate,
            c.direction,
            c.billsec,
            c.duration,
            c.src,
            c.answeredext,
            e.client_sentiment,
            e.call_purpose,
            e.call_summary,
            e.checklist_json,
            e.politeness_score
        FROM calls c
        LEFT JOIN evaluations e ON c.linkedid = e.linkedid AND e.prompt_id = :pid
        WHERE c.processing_status = 'done'
          AND c.calldate >= :start
          AND c.calldate < :end
        ORDER BY c.calldate DESC
    """), conn, params={
        "start": start_date,
        "end": end_date + timedelta(days=1),
        "pid": prompt_id
    })
    engine.dispose()
    return df

@st.cache_data(ttl=300)
def get_phone_names():
    db_url = f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}"
    engine = create_engine(db_url)
    with engine.connect() as conn:
        df_phones = pd.read_sql(text("SELECT number, name FROM phones"), conn)
    engine.dispose()
    return dict(zip(df_phones['number'], df_phones['name']))

# Фильтры в верхней панели
col_d, col_p = st.columns([1, 1])

with col_d:
    today = datetime.now().date()
    default_start = today - timedelta(days=7)
    date_range = st.date_input("Диапазон дат", (default_start, today))

if not (isinstance(date_range, tuple) and len(date_range) == 2):
    st.info("Выберите диапазон дат (начало и конец).")
    st.stop()

start_date, end_date = date_range

with col_p:
    all_prompts = get_all_prompts()
    default_p = get_default_prompt()
    default_p_id = default_p['id'] if default_p else (all_prompts[0]['id'] if all_prompts else None)

    prompt_options = {p['id']: p['name'] for p in all_prompts}
    selected_prompt_id = st.selectbox(
        "Аналитический промпт",
        options=list(prompt_options.keys()),
        format_func=lambda x: prompt_options[x],
        index=list(prompt_options.keys()).index(default_p_id) if default_p_id in prompt_options else 0
    )

if not selected_prompt_id:
    st.warning("Промпты не найдены. Создайте промпт в настройках.")
    st.stop()

df = get_summary_data(start_date, end_date, selected_prompt_id)

if df.empty:
    st.warning("Нет данных для отображения.")
else:
    phone_names = get_phone_names()

    # Обработка данных
    def process_row(row):
        # 1. Определение ролей
        # User: входящие называются incoming, остальные Operator is src
        if row['direction'] == 'incoming':
            op_num = row['answeredext']
            cl_num = row['src']
        else:
            op_num = row['src']
            cl_num = row['answeredext']

        row['Имя оператора'] = phone_names.get(op_num, op_num)
        row['Номер клиента'] = cl_num

        # 2. Определение типа звонка
        dir_map = {
            'incoming': 'Входящий',
            'inbound': 'Входящий',
            'outgoing': 'Исходящий',
            'outbound': 'Исходящий',
            'internal': 'Внутренний'
        }
        row['Тип звонка'] = dir_map.get(row['direction'], row['direction'])

        # 3. Продолжительность мм:сс
        if pd.notnull(row['billsec']):
            m, s = divmod(int(row['billsec']), 60)
            row['Продолжительность'] = f"{m:02d}:{s:02d}"
        else:
            row['Продолжительность'] = "00:00"

        # 4. Перевод целей и настроения
        purpose_map = {
            "appointment": "Запись на прием",
            "consultation": "Консультация",
            "complaint": "Жалоба",
            "cancel_appointment": "Отмена записи",
            "reschedule_appointment": "Перенос записи",
            "other": "Другое"
        }
        sentiment_map = {
            "positive": "Положительное",
            "neutral": "Нейтральное",
            "negative": "Отрицательное",
            "conflict": "Конфликт"
        }
        row['Цель звонка'] = purpose_map.get(row['call_purpose'], row['call_purpose'])
        row['Настроение'] = sentiment_map.get(row['client_sentiment'], row['client_sentiment'])

        # 5. Флаги
        checklist = row['checklist_json'] if isinstance(row['checklist_json'], dict) else {}
        row['Поздоровался'] = "✅" if checklist.get('greeting') else "❌"
        row['Представился'] = "✅" if checklist.get('introduced_himself') else "❌"
        row['Согласована дата'] = "✅" if checklist.get('agreed_datetime') else "❌"
        row['Определена цель'] = "✅" if checklist.get('identified_need') else "❌"
        row['Озвучена цена'] = "✅" if checklist.get('informed_price') else "❌"
        row['Жалоба решена'] = "✅" if checklist.get('handled_objection') else "❌"
        row['Попрощался'] = "✅" if checklist.get('farewell') else "❌"

        return row

    processed_df = df.apply(process_row, axis=1).reset_index(drop=True)
    processed_df['hour'] = processed_df['calldate'].dt.hour

    # Инициализация состояния фильтров
    if "filters" not in st.session_state:
        st.session_state.filters = {"purpose": None, "sentiment": None, "hour": None, "type": None, "date": None}
    if "show_table" not in st.session_state:
        st.session_state.show_table = False
    if "last_date_range" not in st.session_state:
        st.session_state.last_date_range = date_range

    # Сброс фильтров при смене дат
    if st.session_state.last_date_range != date_range:
        st.session_state.filters = {"purpose": None, "sentiment": None, "hour": None, "type": None, "date": None}
        st.session_state.show_table = False
        st.session_state.last_date_range = date_range

    def get_filtered_df(exclude=None):
        mask = pd.Series(True, index=processed_df.index)
        if st.session_state.filters["purpose"] and exclude != "purpose":
            mask &= (processed_df['Цель звонка'] == st.session_state.filters["purpose"])
        if st.session_state.filters["sentiment"] and exclude != "sentiment":
            mask &= (processed_df['Настроение'] == st.session_state.filters["sentiment"])
        if st.session_state.filters["hour"] is not None and exclude != "hour":
            mask &= (processed_df['hour'] == st.session_state.filters["hour"])
        if st.session_state.filters["type"] and exclude != "type":
            mask &= (processed_df['Тип звонка'] == st.session_state.filters["type"])
        if st.session_state.filters["date"] and exclude != "date":
            mask &= (processed_df['calldate'].dt.date == st.session_state.filters["date"])
        return processed_df[mask].copy()

    # Отображение активных фильтров
    active_filters_count = len([v for v in st.session_state.filters.values() if v is not None])
    if active_filters_count > 0:
        # Ограничиваем количество колонок, чтобы кнопки не были слишком узкими
        num_cols = min(active_filters_count + 1, 6)
        cols_f = st.columns(num_cols)
        cur_col = 0
        if st.session_state.filters["purpose"]:
            cols_f[cur_col % num_cols].info(f"Цель: {st.session_state.filters['purpose']}")
            cur_col += 1
        if st.session_state.filters["sentiment"]:
            cols_f[cur_col % num_cols].info(f"Настроение: {st.session_state.filters['sentiment']}")
            cur_col += 1
        if st.session_state.filters["hour"] is not None:
            cols_f[cur_col % num_cols].info(f"Час: {st.session_state.filters['hour']}:00")
            cur_col += 1
        if st.session_state.filters["type"]:
            cols_f[cur_col % num_cols].info(f"Тип: {st.session_state.filters['type']}")
            cur_col += 1
        if st.session_state.filters["date"]:
            cols_f[cur_col % num_cols].info(f"Дата: {st.session_state.filters['date'].strftime('%d.%m')}")
            cur_col += 1

        if cols_f[cur_col % num_cols].button("Сбросить"):
            st.session_state.filters = {"purpose": None, "sentiment": None, "hour": None, "type": None, "date": None}
            st.session_state.show_table = False
            st.rerun()

    col1, col2, col_poly = st.columns(3)

    # 1. График целей
    df_p = get_filtered_df(exclude="purpose")
    purpose_counts = df_p['Цель звонка'].value_counts().reset_index()
    purpose_counts.columns = ['Цель', 'Количество']
    fig_purpose = px.bar(purpose_counts, x='Цель', y='Количество',
                         labels={'Количество': 'Количество звонков', 'Цель': 'Цель звонка'},
                         custom_data=['Цель'])
    fig_purpose.update_layout(separators=", ")

    # 2. График настроения
    df_s = get_filtered_df(exclude="sentiment")
    sentiment_counts = df_s['Настроение'].value_counts().reset_index()
    sentiment_counts.columns = ['Настроение', 'Количество']
    fig_sentiment = px.pie(sentiment_counts, values='Количество', names='Настроение',
                           color='Настроение',
                           color_discrete_map={
                               'Положительное': 'green',
                               'Нейтральное': 'blue',
                               'Отрицательное': 'orange',
                               'Конфликт': 'red'
                           },
                           custom_data=['Настроение'])
    fig_sentiment.update_layout(separators=", ")

    with col1:
        st.subheader("Распределение целей звонков")
        purpose_event = st.plotly_chart(fig_purpose, use_container_width=True, on_select="rerun", key="purpose_chart")
        if purpose_event and purpose_event.selection.get("points"):
            point = purpose_event.selection["points"][0]
            sel = point.get("customdata", [None])[0] or point.get("x")
            if sel != st.session_state.filters["purpose"]:
                st.session_state.filters["purpose"] = sel
                st.session_state.show_table = True
                st.rerun()

    with col2:
        st.subheader("Настроение клиентов")
        sentiment_event = st.plotly_chart(fig_sentiment, use_container_width=True, on_select="rerun", key="sentiment_chart")
        if sentiment_event and sentiment_event.selection.get("points"):
            point = sentiment_event.selection["points"][0]
            sel = point.get("customdata", [None])[0] or point.get("label")
            if sel != st.session_state.filters["sentiment"]:
                st.session_state.filters["sentiment"] = sel
                st.session_state.show_table = True
                st.rerun()

    # Линейный график вежливости
    df_poly = get_filtered_df(exclude="date")
    df_poly['date'] = df_poly['calldate'].dt.date
    # Группируем и берем среднее, игнорируя NaN
    daily_politeness = df_poly.groupby('date')['politeness_score'].mean().reset_index()
    daily_politeness['politeness_score'] = daily_politeness['politeness_score'].round(2)

    fig_politeness = px.line(daily_politeness, x='date', y='politeness_score',
                             labels={'politeness_score': 'Вежливость', 'date': 'Дата'},
                             markers=True,
                             range_y=[0, 10]) # Оценка от 1 до 10
    fig_politeness.update_layout(separators=", ")

    with col_poly:
        st.subheader("Вежливость по дням")
        st.plotly_chart(fig_politeness, use_container_width=True)

    col3, col4 = st.columns(2)

    # Подготовка данных для почасовых графиков
    df_h = get_filtered_df(exclude="hour")
    num_days = (end_date - start_date).days + 1
    if num_days <= 0: num_days = 1

    all_hours = pd.DataFrame({'hour': range(6, 23)})

    # 3. Распределение по часам (количество)
    hourly_data = df_h[(df_h['hour'] >= 6) & (df_h['hour'] <= 22)]

    # Считаем общее кол-во по часам для отображения среднего значения на столбике
    hourly_totals = hourly_data.groupby('hour').size().reset_index(name='total_count')
    hourly_totals['avg'] = (hourly_totals['total_count'] / num_days).round(1)
    hourly_totals['avg_str'] = hourly_totals['avg'].apply(lambda x: str(x).replace('.', ','))

    # Группируем по часам и типам звонков
    hourly_counts = hourly_data.groupby(['hour', 'Тип звонка']).size().reset_index(name='count')

    # Добавляем все часы и типы, чтобы графики не "прыгали"
    call_types = ['Входящий', 'Исходящий', 'Внутренний']
    full_index = pd.MultiIndex.from_product([range(6, 23), call_types], names=['hour', 'Тип звонка'])
    hourly_counts = hourly_counts.set_index(['hour', 'Тип звонка']).reindex(full_index, fill_value=0).reset_index()

    fig_h_count = px.bar(hourly_counts, x='hour', y='count', color='Тип звонка',
                         labels={'count': 'Кол-во звонков', 'hour': 'Час', 'Тип звонка': 'Тип'},
                         custom_data=['hour', 'Тип звонка'],
                         color_discrete_map={
                             'Входящий': 'green',
                             'Исходящий': 'blue',
                             'Внутренний': 'orange'
                         })

    # Добавляем среднее значение посередине всего столбика
    for _, row_t in hourly_totals.iterrows():
        if row_t['total_count'] > 0:
            fig_h_count.add_annotation(
                x=row_t['hour'], y=row_t['total_count']/2, text=row_t['avg_str'],
                showarrow=False, font=dict(color="white", size=12)
            )

    fig_h_count.update_layout(xaxis={'tickmode': 'linear', 'tick0': 6, 'dtick': 1}, separators=", ", barmode='stack')

    # 4. Средняя продолжительность
    hourly_dur = df_h[(df_h['hour'] >= 6) & (df_h['hour'] <= 22)].groupby('hour')[['duration', 'billsec']].mean().reset_index()
    hourly_dur = all_hours.merge(hourly_dur, on='hour', how='left').fillna(0)

    def format_mmss(sec):
        if sec <= 0: return ""
        m, s = divmod(int(sec), 60)
        return f"{m:02d}:{s:02d}"

    hourly_dur['duration_str'] = hourly_dur['duration'].apply(format_mmss)
    hourly_dur['billsec_str'] = hourly_dur['billsec'].apply(format_mmss)

    fig_h_dur = go.Figure()
    fig_h_dur.add_trace(go.Bar(
        x=hourly_dur['hour'], y=hourly_dur['duration'], name='Ожидание + Разговор',
        marker_color='rgba(100, 149, 237, 0.6)',
        text=hourly_dur['duration_str'], textposition='inside',
        hovertemplate="Ожидание + Разговор: %{text}<extra></extra>",
        customdata=hourly_dur['hour']
    ))
    fig_h_dur.add_trace(go.Bar(
        x=hourly_dur['hour'], y=hourly_dur['billsec'], name='Разговор',
        marker_color='rgba(0, 0, 139, 0.8)',
        text=hourly_dur['billsec_str'], textposition='inside',
        hovertemplate="Разговор: %{text}<extra></extra>",
        customdata=hourly_dur['hour']
    ))
    fig_h_dur.update_layout(
        barmode='overlay',
        xaxis={'title': 'Час', 'tickmode': 'linear', 'tick0': 6, 'dtick': 1},
        yaxis={'title': 'Среднее время (сек)'},
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1},
        separators=", "
    )

    with col3:
        st.subheader("Звонки по часам")
        hour_event = st.plotly_chart(fig_h_count, use_container_width=True, on_select="rerun", key="hour_chart")
        if hour_event and hour_event.selection.get("points"):
            point = hour_event.selection["points"][0]
            h_sel = point.get("customdata", [None, None])[0] or point.get("x")
            t_sel = point.get("customdata", [None, None])[1]
            changed = False
            if h_sel is not None:
                h_sel = int(h_sel)
                if h_sel != st.session_state.filters["hour"]:
                    st.session_state.filters["hour"] = h_sel
                    changed = True
            if t_sel is not None:
                if t_sel != st.session_state.filters["type"]:
                    st.session_state.filters["type"] = t_sel
                    changed = True
            if changed:
                st.session_state.show_table = True
                st.rerun()

    with col4:
        st.subheader("Средняя длительность")
        hour_dur_event = st.plotly_chart(fig_h_dur, use_container_width=True, on_select="rerun", key="hour_dur_chart")
        if hour_dur_event and hour_dur_event.selection.get("points"):
            point = hour_dur_event.selection["points"][0]
            sel = point.get("customdata") or point.get("x")
            if sel is not None:
                sel = int(sel)
                if sel != st.session_state.filters["hour"]:
                    st.session_state.filters["hour"] = sel
                    st.session_state.show_table = True
                    st.rerun()

    if st.button("Показать все звонки"):
        st.session_state.filters = {"purpose": None, "sentiment": None, "hour": None, "type": None, "date": None}
        st.session_state.show_table = True
        st.rerun()

    if st.session_state.show_table:
        filtered_df = get_filtered_df()
        st.markdown("---")
        st.markdown(f"### Список звонков ({len(filtered_df)})")

        # Выбор и переименование колонок для отображения
        display_df = filtered_df[[
            'calldate', 'Тип звонка', 'Номер клиента', 'Имя оператора', 'Продолжительность',
            'Цель звонка', 'Настроение', 'call_summary',
            'Поздоровался', 'Представился', 'Согласована дата', 'Определена цель',
            'Озвучена цена', 'Жалоба решена', 'Попрощался'
        ]].copy()

        display_df.rename(columns={
            'calldate': 'Дата/время',
            'call_summary': 'Краткое содержание'
        }, inplace=True)

        # Отображение таблицы
        selection = st.dataframe(
            display_df,
            column_config={
                "Дата/время": st.column_config.DatetimeColumn("Дата/время", format="DD.MM.YYYY HH:mm"),
            },
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="calls_table"
        )

        # Обработка выбора строки и вывод плеера
        if selection and selection.selection.rows:
            selected_index = selection.selection.rows[0]
            selected_linkedid = filtered_df.iloc[selected_index]['linkedid']

            st.markdown("---")
            st.subheader(f"Прослушивание записи: {selected_linkedid}")

            db_url = f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}"
            engine = create_engine(db_url)
            with engine.connect() as conn:
                # 1. Получаем путь к файлу
                res = conn.execute(text("SELECT file_path FROM calls WHERE linkedid = :lid"), {"lid": selected_linkedid}).fetchone()
                if res and res[0] and os.path.exists(res[0]):
                    st.audio(res[0])
                else:
                    st.error("Файл записи не найден.")

                # 2. Добавляем расшифровку
                st.markdown("#### Расшифровка звонка")
                transcript_res = conn.execute(text("""
                    SELECT channel, start_time, text
                    FROM transcripts
                    WHERE linkedid = :lid
                    ORDER BY start_time ASC
                """), {"lid": selected_linkedid})

                rows = transcript_res.fetchall()
                if rows:
                    for row in rows:
                        m, s = divmod(int(row.start_time), 60)
                        time_str = f"[{m:02d}:{s:02d}]"
                        label = "👤 **Оператор**" if row.channel == 'operator' else "👥 **Клиент**"
                        st.markdown(f"{time_str} {label}: {row.text}")
                else:
                    st.info("Расшифровка для этого звонка отсутствует.")

            engine.dispose()
