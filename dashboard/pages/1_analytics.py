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
from db_utils import get_all_prompts, get_default_prompt, get_call_file_path, get_call_transcript, format_dialogue

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
            c.processing_status,
            e.client_sentiment,
            e.call_purpose,
            e.call_summary,
            e.checklist_json,
            e.politeness_score
        FROM calls c
        LEFT JOIN evaluations e ON c.linkedid = e.linkedid AND e.prompt_id = :pid
        WHERE c.calldate >= :start
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
            'incoming': '📥',
            'inbound': '📥',
            'outgoing': '📤',
            'outbound': '📤',
            'internal': '🏠'
        }
        row['Тип звонка'] = dir_map.get(str(row['direction']).lower(), '❓')

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

    col1, col2, col3 = st.columns(3)

    # ПОДГОТОВКА ДАННЫХ ДЛЯ ПЕРВОГО РЯДА
    num_days = (end_date - start_date).days + 1
    if num_days <= 0: num_days = 1
    all_hours = pd.DataFrame({'hour': range(6, 23)})

    # 1.1 Звонки по часам
    df_h = get_filtered_df(exclude="hour")
    hourly_data = df_h[(df_h['hour'] >= 6) & (df_h['hour'] <= 22)].copy()

    # Группировка по часам и типам для стека
    hourly_counts = hourly_data.groupby(['hour', 'Тип звонка'], observed=False).size().reset_index(name='count')

    # Считаем общее количество для аннотаций
    hourly_totals = hourly_data.groupby('hour', observed=False).size().reset_index(name='total_count')

    # Формируем полный индекс для всех часов
    call_types = ['📥', '📤', '🏠', '❓']
    full_index = pd.MultiIndex.from_product([range(6, 23), call_types], names=['hour', 'Тип звонка'])
    hourly_counts = hourly_counts.set_index(['hour', 'Тип звонка']).reindex(full_index, fill_value=0).reset_index()

    fig_h_count = px.bar(hourly_counts, x='hour', y='count', color='Тип звонка',
                         labels={'count': 'Кол-во звонков', 'hour': 'Час', 'Тип звонка': 'Тип'},
                         color_discrete_map={
                             '📥': 'green',
                             '📤': 'blue',
                             '🏠': 'orange',
                             '❓': 'gray'
                         },
                         custom_data=['hour'])

    for _, row_t in hourly_totals.iterrows():
        if row_t['total_count'] > 0:
            avg_val = round(row_t['total_count'] / num_days, 1)
            avg_str = str(avg_val).replace('.', ',')
            fig_h_count.add_annotation(
                x=row_t['hour'], y=row_t['total_count']/2, text=avg_str,
                showarrow=False, font=dict(color="white", size=12)
            )
    fig_h_count.update_layout(xaxis={'tickmode': 'linear', 'tick0': 6, 'dtick': 1}, separators=", ", barmode='stack')

    # 1.2 Средняя длительность
    hourly_dur = hourly_data.groupby('hour')[['duration', 'billsec']].mean().reset_index()
    hourly_dur = all_hours.merge(hourly_dur, on='hour', how='left').fillna(0)
    def format_mmss(sec):
        if sec <= 0: return "00:00"
        m, s = divmod(int(sec), 60)
        return f"{m:02d}:{s:02d}"
    hourly_dur['billsec_str'] = hourly_dur['billsec'].apply(format_mmss)
    fig_h_dur = px.bar(hourly_dur, x='hour', y='billsec',
                       labels={'billsec': 'Среднее время (сек)', 'hour': 'Час'},
                       text='billsec_str',
                       custom_data=['hour'])
    fig_h_dur.update_traces(marker_color='indianred', textposition='inside')
    fig_h_dur.update_layout(xaxis={'tickmode': 'linear', 'tick0': 6, 'dtick': 1}, separators=", ")

    # 1.3 Распределение целей
    df_p = get_filtered_df(exclude="purpose")
    purpose_counts = df_p['Цель звонка'].value_counts().reset_index()
    purpose_counts.columns = ['Цель', 'Количество']
    fig_purpose = px.bar(purpose_counts, x='Цель', y='Количество',
                         labels={'Количество': 'Количество звонков', 'Цель': 'Цель звонка'},
                         custom_data=['Цель'])
    fig_purpose.update_layout(separators=", ")

    with col1:
        st.subheader("Звонки по часам")
        hour_event = st.plotly_chart(fig_h_count, width='stretch', on_select="rerun", key="hour_chart")
        if hour_event and hour_event.selection.get("points"):
            point = hour_event.selection["points"][0]
            h_sel = point.get("customdata", [None])[0] or point.get("x")
            if h_sel is not None:
                st.session_state.filters["hour"] = int(h_sel)
                st.session_state.show_table = True
                st.rerun()

    with col2:
        st.subheader("Средняя длительность")
        hour_dur_event = st.plotly_chart(fig_h_dur, width='stretch', on_select="rerun", key="hour_dur_chart")
        if hour_dur_event and hour_dur_event.selection.get("points"):
            point = hour_dur_event.selection["points"][0]
            sel = point.get("customdata", [None])[0] or point.get("x")
            if sel is not None:
                st.session_state.filters["hour"] = int(sel)
                st.session_state.show_table = True
                st.rerun()

    with col3:
        st.subheader("Распределение целей")
        purpose_event = st.plotly_chart(fig_purpose, width='stretch', on_select="rerun", key="purpose_chart")
        if purpose_event and purpose_event.selection.get("points"):
            point = purpose_event.selection["points"][0]
            sel = point.get("customdata", [None])[0] or point.get("x")
            if sel != st.session_state.filters["purpose"]:
                st.session_state.filters["purpose"] = sel
                st.session_state.show_table = True
                st.rerun()

    # РЯД 2: Настроение (1/3) и Вежливость по операторам (2/3)
    col4, col5 = st.columns([1, 2])

    # 2.1 Настроение
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

    # 2.2 Вежливость по операторам
    df_poly = get_filtered_df() # Здесь можно не исключать фильтры, или исключать специфические
    # Оставляем только те звонки, где есть оценка вежливости
    df_poly_valid = df_poly[df_poly['politeness_score'].notnull()].copy()
    operator_politeness = df_poly_valid.groupby('Имя оператора')['politeness_score'].mean().reset_index()
    operator_politeness['politeness_score'] = operator_politeness['politeness_score'].round(2)
    operator_politeness = operator_politeness.sort_values('politeness_score', ascending=False)

    fig_poly_op = px.bar(operator_politeness, x='Имя оператора', y='politeness_score',
                         labels={'politeness_score': 'Средняя вежливость', 'Имя оператора': 'Оператор'},
                         range_y=[0, 10], text='politeness_score')
    fig_poly_op.update_layout(separators=", ")

    with col4:
        st.subheader("Настроение клиентов")
        sentiment_event = st.plotly_chart(fig_sentiment, width='stretch', on_select="rerun", key="sentiment_chart")
        if sentiment_event and sentiment_event.selection.get("points"):
            point = sentiment_event.selection["points"][0]
            sel = point.get("customdata", [None])[0] or point.get("label")
            if sel != st.session_state.filters["sentiment"]:
                st.session_state.filters["sentiment"] = sel
                st.session_state.show_table = True
                st.rerun()

    with col5:
        st.subheader("Вежливость по операторам")
        st.plotly_chart(fig_poly_op, width='stretch')

    if st.button("Показать все звонки"):
        st.session_state.filters = {"purpose": None, "sentiment": None, "hour": None, "type": None, "date": None}
        st.session_state.show_table = True
        st.rerun()

    if st.session_state.show_table:
        filtered_df = get_filtered_df()
        st.markdown("---")
        st.markdown(f"### Список звонков ({len(filtered_df)})")

        # Выбор и переименование колонок для отображения
        status_map = {
            'done': '✅',
            'processing': '⏳',
            'skipped': '⏭️',
            'error': '❌',
            'new': '🆕',
            'empty': '😶'
        }
        filtered_df['Статус'] = filtered_df['processing_status'].map(lambda x: status_map.get(x, '❓'))

        display_df = filtered_df[[
            'Статус', 'calldate', 'Тип звонка', 'Номер клиента', 'Имя оператора', 'Продолжительность',
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
                "Статус": st.column_config.TextColumn("⚙️", help="Статус обработки"),
                "Дата/время": st.column_config.DatetimeColumn("Дата/время", format="DD.MM.YYYY HH:mm"),
                "Тип звонка": st.column_config.TextColumn("📞", help="Тип звонка"),
                "Продолжительность": st.column_config.TextColumn("⏱️", help="Продолжительность"),
                "Поздоровался": st.column_config.TextColumn("👋", help="Поздоровался"),
                "Представился": st.column_config.TextColumn("🆔", help="Представился"),
                "Согласована дата": st.column_config.TextColumn("📅", help="Согласована дата"),
                "Определена цель": st.column_config.TextColumn("🎯", help="Определена цель"),
                "Озвучена цена": st.column_config.TextColumn("💰", help="Озвучена цена"),
                "Жалоба решена": st.column_config.TextColumn("🛠️", help="Жалоба решена"),
                "Попрощался": st.column_config.TextColumn("🤝", help="Попрощался")
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

            # 1. Получаем путь к файлу
            fpath = get_call_file_path(selected_linkedid)
            if fpath and os.path.exists(fpath):
                st.audio(fpath)
            else:
                st.error("Файл записи не найден.")

            # 2. Добавляем расшифровку
            st.markdown("#### Расшифровка звонка")
            transcript_rows = get_call_transcript(selected_linkedid)

            if transcript_rows:
                for trow in transcript_rows:
                    m, s = divmod(int(trow['start_time']), 60)
                    time_str = f"[{m:02d}:{s:02d}]"
                    label = "👤 **Оператор**" if trow['channel'] == 'operator' else "👥 **Клиент**"
                    st.markdown(f"{time_str} {label}: {trow['text']}")

                # Добавляем скрытый блок с полным текстом для копирования, если нужно
                # с использованием общей функции форматирования
                with st.expander("Весь текст для копирования"):
                    st.text_area("Текст диалога", format_dialogue(transcript_rows), height=300)
            else:
                st.info("Расшифровка для этого звонка отсутствует.")
