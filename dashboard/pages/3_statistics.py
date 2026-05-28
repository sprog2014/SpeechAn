import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from db_utils import get_processing_statistics, get_prompt_usage_statistics
from config import RECORDS_ROOT
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Статистика работы", layout="wide")

st.title("📊 Статистика работы системы")

def get_disk_file_counts(start_date, end_date):
    """Считает количество .mp3 файлов в RECORDS_ROOT в разрезе дат."""
    counts = {}
    curr = start_date
    while curr <= end_date:
        date_path = curr.strftime("%Y/%m/%d")
        full_path = os.path.join(RECORDS_ROOT, date_path)
        count = 0
        if os.path.exists(full_path):
            for root, dirs, files in os.walk(full_path):
                for f in files:
                    if f.lower().endswith('.mp3'):
                        count += 1
        counts[curr] = count
        curr += timedelta(days=1)
    return counts

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
    disk_counts = get_disk_file_counts(start_date, end_date)

    if not stats or not stats['daily_stats']:
        st.info("Нет данных за выбранный период")
    else:
        # 1. Основные показатели (карточки)
        df_daily = pd.DataFrame(stats['daily_stats'])
        df_daily['date'] = pd.to_datetime(df_daily['date']).dt.date

        # Добавляем данные с диска
        df_daily['total_disk'] = df_daily['date'].apply(lambda d: disk_counts.get(d, 0))

        # Считаем очередь: Всего на диске - (Обработано + Пропущено + В процессе + Ошибки)
        df_daily['queued'] = df_daily['total_disk'] - (df_daily['processed'] + df_daily['skipped'] + df_daily['in_progress'] + df_daily['error'])
        # Очередь не может быть отрицательной
        df_daily['queued'] = df_daily['queued'].apply(lambda x: max(0, x))

        total_calls = df_daily['total_disk'].sum()
        total_processed = df_daily['processed'].sum()
        total_skipped = df_daily['skipped'].sum()
        total_in_progress = df_daily['in_progress'].sum()
        total_queued = df_daily['queued'].sum()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Всего на диске", total_calls)
        m2.metric("Обработано", total_processed)
        m3.metric("Пропущено", total_skipped)
        m4.metric("В обработке", total_in_progress)
        m5.metric("В очереди", total_queued)

        st.divider()

        # 2. График по дням
        st.subheader("Статистика по дням")

        # Переименовываем столбцы для легенды
        df_plot_daily = df_daily.rename(columns={
            'processed': 'Обработано',
            'skipped': 'Пропущено',
            'in_progress': 'В обработке',
            'queued': 'В очереди',
            'error': 'Ошибка'
        })

        fig_daily = px.bar(
            df_plot_daily,
            x='date',
            y=['Обработано', 'Пропущено', 'В обработке', 'В очереди', 'Ошибка'],
            title="Статус обработки по дням",
            labels={'value': 'Количество', 'date': 'Дата', 'variable': 'Статус'},
            barmode='stack'
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        # Таблица данных
        st.dataframe(
            df_daily,
            use_container_width=True,
            column_config={
                "date": "Дата",
                "total_disk": "Всего",
                "skipped": "Пропущено",
                "processed": "Обработано",
                "in_progress": "В обработке",
                "queued": "В очереди",
                "error": "Ошибка",
                "avg_duration": "Среднее время (сек)",
                "total_duration": "Общее время (сек)"
            },
            hide_index=True
        )

        st.divider()

        # 2.5 Использование промптов
        st.subheader("Использование промптов")
        prompt_usage = get_prompt_usage_statistics()
        if prompt_usage:
            df_prompts = pd.DataFrame(prompt_usage)
            # Переименовываем для круговой диаграммы
            df_prompts_plot = df_prompts.rename(columns={'name': 'Название промпта', 'count': 'Кол-во использований'})

            fig_prompts = px.pie(
                df_prompts_plot,
                values='Кол-во использований',
                names='Название промпта',
                title="Распределение использования промптов"
            )

            pc1, pc2 = st.columns([1, 1])
            with pc1:
                st.plotly_chart(fig_prompts, use_container_width=True)
            with pc2:
                st.dataframe(
                    df_prompts,
                    use_container_width=True,
                    column_config={"name": "Название промпта", "count": "Использовано раз"},
                    hide_index=True
                )
        else:
            st.info("Нет данных об использовании промптов")

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
