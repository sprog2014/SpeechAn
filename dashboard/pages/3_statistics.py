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
today = datetime.now().date()
default_start = today - timedelta(days=7)
date_range = st.date_input("Выберите диапазон дат", (default_start, today))

if not (isinstance(date_range, tuple) and len(date_range) == 2):
    st.info("Выберите диапазон дат (начало и конец).")
    st.stop()

start_date, end_date = date_range

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

        # Считаем очередь: Всего на диске - (Обработано + Пропущено + В процессе + Транскрибировано + Ошибки + Пустые + Остановлено)
        df_daily['queued'] = df_daily['total_disk'] - (df_daily['processed'] + df_daily['skipped'] + df_daily['in_progress'] + df_daily['transcribed'] + df_daily['error'] + df_daily['empty'] + df_daily['stop'])
        # Очередь не может быть отрицательной
        df_daily['queued'] = df_daily['queued'].apply(lambda x: max(0, x))

        total_calls = df_daily['total_disk'].sum()
        total_processed = df_daily['processed'].sum()
        total_skipped = df_daily['skipped'].sum()
        total_transcribed = df_daily['transcribed'].sum()
        total_empty = df_daily['empty'].sum()
        total_stop = df_daily['stop'].sum()
        total_queued = df_daily['queued'].sum()

        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Всего на диске", f"{total_calls:,}".replace(",", " "))
        m2.metric("Обработано", f"{total_processed:,}".replace(",", " "))
        m3.metric("Пропущено", f"{total_skipped:,}".replace(",", " "))
        m4.metric("Транскрибировано", f"{total_transcribed:,}".replace(",", " "))
        m5.metric("Пустые", f"{total_empty:,}".replace(",", " "))
        m6.metric("Стоп", f"{total_stop:,}".replace(",", " "))
        m7.metric("В очереди", f"{total_queued:,}".replace(",", " "))

        st.divider()

        # 2. График по дням
        st.subheader("Статистика по дням")

        # Переименовываем столбцы для легенды
        df_plot_daily = df_daily.rename(columns={
            'processed': 'Обработано',
            'skipped': 'Пропущено',
            'transcribed': 'Транскрибировано',
            'empty': 'Пустые',
            'stop': 'Стоп',
            'queued': 'В очереди'
        })

        fig_daily = px.bar(
            df_plot_daily,
            x='date',
            y=['Обработано', 'Пропущено', 'Транскрибировано', 'Пустые', 'Стоп', 'В очереди'],
            title="Статус обработки по дням",
            labels={'value': 'Количество', 'date': 'Дата', 'variable': 'Статус'},
            barmode='stack',
            color_discrete_map={
                'Обработано': 'green',
                'Пропущено': 'gray',
                'Транскрибировано': 'blue',
                'Пустые': 'lightgray',
                'Стоп': 'red',
                'В очереди': 'orange'
            }
        )
        fig_daily.update_layout(separators=", ")
        st.plotly_chart(fig_daily, width='stretch')

        # Таблица данных
        st.dataframe(
            df_daily,
            width='stretch',
            column_config={
                "date": "Дата",
                "total_disk": st.column_config.NumberColumn("Всего", format="%d"),
                "skipped": st.column_config.NumberColumn("Пропущено", format="%d"),
                "transcribed": st.column_config.NumberColumn("Транскрибировано", format="%d"),
                "processed": st.column_config.NumberColumn("Обработано", format="%d"),
                "in_progress": st.column_config.NumberColumn("В обработке", format="%d"),
                "empty": st.column_config.NumberColumn("Пустые", format="%d"),
                "stop": st.column_config.NumberColumn("Остановлено", format="%d"),
                "queued": st.column_config.NumberColumn("В очереди", format="%d"),
                "error": st.column_config.NumberColumn("Ошибка", format="%d"),
                "avg_duration": st.column_config.NumberColumn("Среднее время (сек)", format="%.2f"),
                "total_duration": st.column_config.NumberColumn("Общее время (сек)", format="%.2f")
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
            fig_prompts.update_layout(separators=", ")

            pc1, pc2 = st.columns([1, 1])
            with pc1:
                st.plotly_chart(fig_prompts, width='stretch')
            with pc2:
                st.dataframe(
                    df_prompts,
                    width='stretch',
                    column_config={
                        "name": "Название промпта",
                        "count": st.column_config.NumberColumn("Использовано раз", format="%d")
                    },
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
            formatted_speed = f"{avg_speed:,.2f}".replace(",", " ").replace(".", ",")
            st.info(f"**Средняя скорость анализа:** {formatted_speed} файлов в час")

            fig_speed = px.line(
                df_speed,
                x='date',
                y='files_per_hour',
                title="Скорость анализа по дням (файлов в час)",
                labels={'files_per_hour': 'Файлов в час', 'date': 'Дата'}
            )
            fig_speed.update_layout(separators=", ")
            st.plotly_chart(fig_speed, width='stretch')
        else:
            st.write("Недостаточно данных для расчета скорости")

        # 4. Распределение времени по этапам
        if stats['timings'] and stats['timings']['avg_total']:
            t = stats['timings']
            total = t['avg_total']

            labels = ['Транскрибация (ASR)', 'LLM Анализ']
            values = [
                t['avg_asr'] if t['avg_asr'] is not None else 0,
                t['avg_llm'] if t['avg_llm'] is not None else 0
            ]

            # Процентное соотношение
            percentages = [v/total * 100 if total > 0 else 0 for v in values]

            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig_pie.update_layout(title_text="Распределение времени по этапам", separators=", ")

            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(fig_pie, width='stretch')
            with c2:
                def f_num(n):
                    return f"{n:,.2f}".replace(",", " ").replace(".", ",")

                def f_pct(n):
                    return f"{n:,.1f}".replace(",", " ").replace(".", ",")

                st.write("**Среднее время по этапам (сек):**")
                st.write(f"- ASR: {f_num(values[0])} сек ({f_pct(percentages[0])}%)")
                st.write(f"- LLM: {f_num(values[1])} сек ({f_pct(percentages[1])}%)")
                st.write(f"**Итого в среднем на 1 файл:** {f_num(total)} сек")
        else:
            st.write("Недостаточно данных для анализа этапов обработки")
