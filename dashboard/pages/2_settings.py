import streamlit as st
import pandas as pd
import sys
import os
import subprocess
from datetime import datetime, timedelta

# Добавляем путь к src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from db_utils import (
    get_system_running_status, set_system_running_status,
    get_all_prompts, upsert_prompt, delete_prompt, set_default_prompt,
    get_pg_connection, get_all_phones, update_phone_use, sync_phones_from_external_db,
    get_system_setting, set_system_setting
)
from config import PG_CONFIG

if not st.session_state.get("password_correct", False):
    st.error("Пожалуйста, авторизуйтесь на главной странице.")
    st.stop()

st.title("Настройки и Управление")

# --- Раздел 1: Состояние системы ---
st.header("Состояние системы")
is_running = get_system_running_status()

col_status, col_btn = st.columns([3, 1])
if is_running:
    col_status.success("Система запущена и обрабатывает файлы.")
    if col_btn.button("Остановить"):
        set_system_running_status(False)
        st.rerun()
else:
    col_status.warning("Система находится в режиме ожидания (ОСТАНОВЛЕНА).")
    if col_btn.button("Запустить"):
        set_system_running_status(True)
        st.rerun()

# --- Раздел 2: Управление промптами ---
st.header("Управление промптами")

# Инициализация состояния для редактора
if "show_editor" not in st.session_state:
    st.session_state.show_editor = False
if "editing_prompt" not in st.session_state:
    st.session_state.editing_prompt = None

prompts = get_all_prompts()

if st.session_state.show_editor:
    st.subheader("Редактирование промпта")
    prompt = st.session_state.editing_prompt
    p_id = prompt['id'] if prompt else None

    with st.form("edit_prompt_form"):
        name = st.text_input("Название", value=prompt['name'] if prompt else "")
        text = st.text_area("Текст промпта", value=prompt['prompt_text'] if prompt else "", height=300)

        col_f1, col_f2 = st.columns(2)
        save_btn = col_f1.form_submit_button("Сохранить")
        cancel_btn = col_f2.form_submit_button("Отмена")

        if save_btn:
            if name and text:
                upsert_prompt(name, text, is_default=prompt['is_default'] if prompt else False, prompt_id=p_id)
                st.success("Сохранено!")
                st.session_state.show_editor = False
                st.session_state.editing_prompt = None
                st.rerun()
            else:
                st.error("Название и текст не могут быть пустыми.")

        if cancel_btn:
            st.session_state.show_editor = False
            st.session_state.editing_prompt = None
            st.rerun()
else:
    if prompts:
        display_data = []
        for p in prompts:
            name_display = f"**{p['name']}**" if p['is_default'] else p['name']
            display_data.append({
                "ID": p['id'],
                "Создан": p['created_at'].strftime("%Y-%m-%d %H:%M") if p['created_at'] else "",
                "Название": name_display,
                "_raw": p
            })

        df_display = pd.DataFrame(display_data)

        st.dataframe(
            df_display[["Создан", "Название"]],
            use_container_width=True,
            hide_index=True,
            column_config={
            "Название": st.column_config.TextColumn("Название", help="Жирным выделен промпт по умолчанию")
            }
        )

        prompt_options = {p['id']: p['name'] for p in prompts}
        selected_prompt_id = st.selectbox("Выберите промпт для действий", options=list(prompt_options.keys()), format_func=lambda x: prompt_options[x])

        selected_prompt = next((p for p in prompts if p['id'] == selected_prompt_id), None)

        col1, col2, col3 = st.columns(3)

        if col1.button("Добавить новый", use_container_width=True):
            st.session_state.show_editor = True
            st.session_state.editing_prompt = None
            st.rerun()

        if col2.button("Изменить выбранный", use_container_width=True, disabled=selected_prompt is None):
            st.session_state.show_editor = True
            st.session_state.editing_prompt = selected_prompt
            st.rerun()

        if col3.button("Сделать по умолчанию", use_container_width=True, disabled=selected_prompt is None):
            set_default_prompt(selected_prompt['id'])
            st.success(f"Промпт '{selected_prompt['name']}' установлен по умолчанию.")
            st.rerun()

        if selected_prompt:
            if st.button("Удалить выбранный промпт", type="primary"):
                delete_prompt(selected_prompt['id'])
                st.success("Удалено!")
                st.rerun()
    else:
        st.info("Промпты не найдены.")
        if st.button("Добавить первый промпт"):
            st.session_state.show_editor = True
            st.session_state.editing_prompt = None
            st.rerun()

# --- Раздел 3: Ручной запуск ---
st.header("Ручной запуск")
with st.form("manual_run_form"):
    date_start = st.date_input("Дата начала", datetime.now() - timedelta(days=1))
    date_end = st.date_input("Дата конца", datetime.now())

    prompt_options = {p['id']: p['name'] for p in prompts} if prompts else {}
    selected_manual_prompt = st.selectbox("Промпт", options=list(prompt_options.keys()), format_func=lambda x: prompt_options[x])

    submit_manual = st.form_submit_button("Запустить повторную обработку")

    if submit_manual:
        if selected_manual_prompt:
            script_path = os.path.join(SRC_DIR, "manual_run.py")
            cmd = [
                "python3", script_path,
                date_start.strftime("%Y-%m-%d"),
                date_end.strftime("%Y-%m-%d"),
                "--prompt_id", str(selected_manual_prompt),
                "--ignore-stop-flag"
            ]

            # Передаем окружение с PYTHONPATH
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{PROJECT_ROOT}:{SRC_DIR}"

            # Не используем PIPE, чтобы избежать зависаний при переполнении буфера
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, cwd=PROJECT_ROOT)
            st.info(f"Процесс ручного запуска инициирован (PID: {process.pid}).")
        else:
            st.error("Выберите промпт для запуска.")

# --- Раздел 4: Телефонный справочник ---
st.header("Телефонный справочник")

skip_local = get_system_setting('skip_local_calls', 'false').lower() == 'true'
if st.checkbox("Пропускать анализ локальных звонков", value=skip_local):
    if not skip_local:
        set_system_setting('skip_local_calls', 'true')
        st.rerun()
else:
    if skip_local:
        set_system_setting('skip_local_calls', 'false')
        st.rerun()

phones_list = get_all_phones()
if phones_list:
    df_phones = pd.DataFrame(phones_list)

    edited_df = st.data_editor(
        df_phones,
        column_config={
            "number": st.column_config.TextColumn("Номер", disabled=True),
            "name": st.column_config.TextColumn("Имя", disabled=True),
            "use": st.column_config.CheckboxColumn("Использовать в анализе")
        },
        hide_index=True,
        use_container_width=True
    )

    col_p1, col_p2 = st.columns(2)
    if col_p1.button("Сохранить изменения", use_container_width=True):
        # Проверяем что изменилось
        for i, row in edited_df.iterrows():
            orig_row = df_phones.iloc[i]
            if row['use'] != orig_row['use']:
                update_phone_use(row['number'], row['use'])
        st.success("Изменения сохранены!")
        st.rerun()

    if col_p2.button("Синхронизировать список номеров", use_container_width=True):
        with st.spinner("Синхронизация..."):
            sync_phones_from_external_db()
        st.success("Список номеров синхронизирован!")
        st.rerun()
else:
    st.info("Список номеров пуст.")
    if st.button("Синхронизировать список номеров"):
        with st.spinner("Синхронизация..."):
            sync_phones_from_external_db()
        st.success("Список номеров синхронизирован!")
        st.rerun()

if st.button("Перейти к Аналитике"):
    st.switch_page("pages/1_analytics.py")
