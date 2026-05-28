import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import sys
import os
import plotly.express as px

# Добавляем путь к src, чтобы найти config.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import PG_CONFIG

if not st.session_state.get("password_correct", False):
    st.error("Пожалуйста, авторизуйтесь на главной странице.")
    st.stop()

st.title("Аналитика звонков")

# Обработка запроса на прослушивание файла
if "linkedid" in st.query_params:
    linkedid = st.query_params["linkedid"]
    db_url = f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}"
    engine = create_engine(db_url)
    with engine.connect() as conn:
        # В SQLAlchemy 2.0+ нужно использовать text() для сырых запросов или правильные методы
        from sqlalchemy import text
        res = conn.execute(text("SELECT file_path FROM calls WHERE linkedid = :lid"), {"lid": linkedid}).fetchone()
        if res and res[0] and os.path.exists(res[0]):
            st.info(f"Прослушивание записи для звонка: {linkedid}")
            st.audio(res[0])
            if st.button("Закрыть плеер", key="close_player_analytics"):
                st.query_params.clear()
                st.rerun()
        else:
            st.error("Файл записи не найден.")
    engine.dispose()

@st.cache_data(ttl=60)
def get_summary_data():
    # Формируем SQLAlchemy URL
    db_url = f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}"
    engine = create_engine(db_url)
    with engine.connect() as conn:
        df = pd.read_sql("""
        SELECT
            c.linkedid,
            c.calldate,
            c.direction,
            c.billsec,
            c.src,
            c.answeredext,
            e.client_sentiment,
            e.call_purpose,
            e.call_summary,
            e.checklist_json
        FROM calls c
        LEFT JOIN evaluations e ON c.linkedid = e.linkedid
        WHERE c.processing_status = 'done'
        ORDER BY c.calldate DESC
        LIMIT 500
    """, conn)
    conn.close()
    return df

@st.cache_data(ttl=300)
def get_phone_names():
    db_url = f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}"
    engine = create_engine(db_url)
    with engine.connect() as conn:
        df_phones = pd.read_sql("SELECT number, name FROM phones", conn)
    engine.dispose()
    return dict(zip(df_phones['number'], df_phones['name']))

df = get_summary_data()

if df.empty:
    st.warning("Нет данных для отображения.")
else:
    phone_names = get_phone_names()

    # Обработка данных
    def process_row(row):
        # 1. Определение ролей
        if row['direction'] == 'inbound':
            op_num = row['answeredext']
            cl_num = row['src']
        else:
            op_num = row['src']
            cl_num = row['answeredext']

        row['Имя оператора'] = phone_names.get(op_num, op_num)

        # 2. Ссылка на запись
        # Используем параметр cl в URL для отображения через regex в LinkColumn
        row['Номер клиента'] = f"/?linkedid={row['linkedid']}&cl={cl_num}"

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

    processed_df = df.apply(process_row, axis=1)

    # Выбор и переименование колонок для отображения
    display_df = processed_df[[
        'calldate', 'Номер клиента', 'Имя оператора', 'Продолжительность',
        'Цель звонка', 'Настроение', 'call_summary',
        'Поздоровался', 'Представился', 'Согласована дата', 'Определена цель',
        'Озвучена цена', 'Жалоба решена', 'Попрощался'
    ]].copy()

    display_df.rename(columns={
        'calldate': 'Дата/время',
        'call_summary': 'Краткое содержание'
    }, inplace=True)

    # Отображение таблицы
    st.dataframe(
        display_df,
        column_config={
            "Дата/время": st.column_config.DatetimeColumn("Дата/время", format="DD.MM.YYYY HH:mm"),
            "Номер клиента": st.column_config.LinkColumn(
                "Номер клиента",
                help="Нажмите, чтобы прослушать запись",
                display_text=r"cl=(.*)$"
            )
        },
        hide_index=True,
    )

    st.markdown("### Визуализация")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Распределение целей звонков")
        purpose_counts = processed_df['Цель звонка'].value_counts().reset_index()
        purpose_counts.columns = ['Цель', 'Количество']
        fig_purpose = px.bar(purpose_counts, x='Цель', y='Количество', color='Цель',
                             labels={'Количество': 'Количество звонков', 'Цель': 'Цель звонка'})
        st.plotly_chart(fig_purpose, use_container_width=True)

    with col2:
        st.subheader("Настроение клиентов")
        sentiment_counts = processed_df['Настроение'].value_counts().reset_index()
        sentiment_counts.columns = ['Настроение', 'Количество']
        fig_sentiment = px.pie(sentiment_counts, values='Количество', names='Настроение',
                               color='Настроение',
                               color_discrete_map={
                                   'Положительное': 'green',
                                   'Нейтральное': 'blue',
                                   'Отрицательное': 'orange',
                                   'Конфликт': 'red'
                               })
        st.plotly_chart(fig_sentiment, use_container_width=True)
