import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from db_utils import get_processing_statistics
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Статистика работы", layout="wide")

st.title("📊 Статистика работы системы")

# Фильтры
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Дата начала", datetime.now() - timedelta(days=7))
with col2:
    end_date = st.date_input("Дата окончания", datetime.now())

if start_date > end_date:
    st.error("Дата начала не может быть больше даты окончания")
else:
    stats = get_processing_statistics(start_date, end_date)

    if not stats or not stats['daily_stats']:
        st.info("Нет данных за выбранный период")
    else:
        # 1. Основные показатели (карточки)
        df_daily = pd.DataFrame(stats['daily_stats'])

        total_calls = df_daily['total'].sum()
        total_processed = df_daily['processed'].sum()
        total_skipped = df_daily['skipped'].sum()
        total_waiting = df_daily['waiting'].sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Всего записей", total_calls)
        m2.metric("Обработано", total_processed)
        m3.metric("Пропущено", total_skipped)
        m4.metric("В ожидании", total_waiting)

        st.divider()

        # 2. График по дням
        st.subheader("Статистика по дням")
        fig_daily = px.bar(
            df_daily,
            x='date',
            y=['processed', 'skipped', 'waiting'],
            title="Статус обработки по дням",
            labels={'value': 'Количество', 'date': 'Дата', 'variable': 'Статус'},
            barmode='stack'
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        # Таблица данных
        st.dataframe(df_daily, use_container_width=True)

        st.divider()

        # 3. Скорость и время анализа
        st.subheader("Скорость и этапы анализа")

        if stats['speed_stats']:
            df_speed = pd.DataFrame(stats['speed_stats'])
            # Средняя скорость: сколько файлов в час (на основе среднего времени обработки 1 файла)
            # Если 1 файл обрабатывается X секунд, то в час 3600/X файлов
            df_speed['files_per_hour'] = df_speed['avg_total_duration'].apply(lambda x: 3600 / x if x > 0 else 0)

            avg_speed = df_speed['files_per_hour'].mean()
            st.info(f"**Средняя скорость анализа:** {avg_speed:.2f} файлов в час")

            fig_speed = px.line(
                df_speed,
                x='date',
                y='files_per_hour',
                title="Скорость анализа по дням (файлов в час)",
                labels={'files_per_hour': 'Файлов в час', 'date': 'Дата'}
            )
            st.plotly_chart(fig_speed, use_container_width=True)
        else:
            st.write("Недостаточно данных для расчета скорости")

        # 4. Распределение времени по этапам
        if stats['timings'] and stats['timings']['avg_total']:
            t = stats['timings']
            total = t['avg_total']

            labels = ['Транскрибация (ASR)', 'Эмоциональная оценка', 'LLM Анализ']
            values = [t['avg_asr'], t['avg_emo'], t['avg_llm']]

            # Процентное соотношение
            percentages = [v/total * 100 for v in values]

            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig_pie.update_layout(title_text="Распределение времени по этапам")

            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                st.write("**Среднее время по этапам (сек):**")
                st.write(f"- ASR: {t['avg_asr']:.2f} сек ({percentages[0]:.1f}%)")
                st.write(f"- Emotion: {t['avg_emo']:.2f} сек ({percentages[1]:.1f}%)")
                st.write(f"- LLM: {t['avg_llm']:.2f} сек ({percentages[2]:.1f}%)")
                st.write(f"**Итого в среднем на 1 файл:** {total:.2f} сек")
        else:
            st.write("Недостаточно данных для анализа этапов обработки")
