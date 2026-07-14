import streamlit as st
import pandas as pd
import sys
import os
import json
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
    get_system_setting, set_system_setting,
    get_all_tasks, add_task, delete_task,
    get_value_mappings, set_value_mappings, update_evaluations_value,
    get_field_synonyms, set_field_synonyms
)
from llm_analysis import check_prompt, validate_chatml_template
from config import PG_CONFIG

@st.cache_data(ttl=60)
def get_cached_synonyms(prompt_id):
    return get_field_synonyms(prompt_id)

if not st.session_state.get("password_correct", False):
    st.error("Пожалуйста, авторизуйтесь на главной странице.")
    st.stop()

st.title("Настройки и Управление")

# --- Раздел 1: Состояние системы и выбор модели ---
st.header("Состояние системы")
is_running = get_system_running_status()

# Выбор активной модели
active_model = get_system_setting('active_model', 'q4_k_m')
col_status, col_btn, col_model = st.columns([2, 1, 1])

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

new_model = col_model.selectbox("Активная модель LLM", options=["q4_k_m", "q8_0"], index=0 if active_model == 'q4_k_m' else 1)
if new_model != active_model:
    set_system_setting('active_model', new_model)
    st.success(f"Модель изменена на {new_model}!")
    st.rerun()

# --- Раздел 2: Управление промптами ---
st.header("Управление промптами")

# Инициализация состояния для редактора
if "show_editor" not in st.session_state:
    st.session_state.show_editor = False
if "editing_prompt" not in st.session_state:
    st.session_state.editing_prompt = None
if "test_transcript" not in st.session_state:
    st.session_state.test_transcript = ""
if "check_result" not in st.session_state:
    st.session_state.check_result = None

prompts = get_all_prompts()

if st.session_state.show_editor:
    st.subheader("Редактирование промпта")
    prompt = st.session_state.editing_prompt
    p_id = prompt['id'] if prompt else None

    st.info("""
    **Инструкция по составлению промпта в формате ChatML:**
    * Промпт должен состоять из блоков `<|im_start|>system ... <|im_end|>` и `<|im_start|>user ... <|im_end|>`.
    * Обязательно добавьте плейсхолдер `{transcript}` в месте, куда будет вставляться транскрипт звонка.
    * Вы можете добавить плейсхолдер `{json_schema}` в месте, куда должна вставляться сгенерированная Pydantic JSON-схема. Если он отсутствует, схема будет автоматически добавлена в начало системного блока.
    """)

    name = st.text_input("Название", value=prompt['name'] if prompt else "")

    col_text1, col_text2 = st.columns(2)
    text = col_text1.text_area("Текст промпта (в формате ChatML)", value=prompt['prompt_text'] if prompt else "", height=300)
    st.session_state.test_transcript = col_text2.text_area("Тестовый транскрипт", value=st.session_state.test_transcript, height=300)

    # Редактор Pydantic схемы (вложенные разделы)
    st.write("### Структура JSON-ответа (Pydantic Схема)")
    schema_tab = st.selectbox(
        "Выберите набор анализируемых данных для редактирования:",
        options=["Основные", "Чек-лист", "Метрики"],
        key=f"schema_tab_select_{p_id if p_id else 'new'}"
    )

    # Извлечение текущей схемы из prompt
    raw_schema = prompt['schema_json'] if prompt and 'schema_json' in prompt else '{}'
    if not raw_schema:
        raw_schema = '{}'
    if isinstance(raw_schema, str):
        try:
            schema_dict = json.loads(raw_schema)
        except:
            schema_dict = {}
    else:
        schema_dict = raw_schema

    if not isinstance(schema_dict, dict):
        schema_dict = {}

    # Убеждаемся, что все 3 ключа существуют
    if 'main' not in schema_dict or not schema_dict['main']:
        schema_dict['main'] = [
            {"key": "politeness_score", "type": "num", "description": "Оценка вежливости оператора от 0 до 10"},
            {"key": "client_sentiment", "type": "str", "description": "Настроение клиента: positive, neutral, negative или conflict"},
            {"key": "call_purpose", "type": "str", "description": "Цель звонка: appointment, consultation, complaint, cancel_appointment или other"},
            {"key": "call_summary", "type": "str", "description": "Краткое содержание диалога (1-2 предложения)"}
        ]
    if 'checklist' not in schema_dict:
        schema_dict['checklist'] = []
    if 'metrics' not in schema_dict:
        schema_dict['metrics'] = []

    # Храним текущие списки во временной структуре в session_state, чтобы сохранять изменения между переключениями вкладок
    session_schema_key = f"temp_schema_dict_{p_id if p_id else 'new'}"
    if session_schema_key not in st.session_state:
        st.session_state[session_schema_key] = schema_dict

    current_schema = st.session_state[session_schema_key]

    if schema_tab == "Основные":
        st.write("ℹ️ *Для обязательных основных параметров ключи и типы фиксированы. Вы можете редактировать их текстовые описания.*")
        df_main = pd.DataFrame(current_schema['main'])
        edited_main_df = st.data_editor(
            df_main,
            column_config={
                "key": st.column_config.TextColumn("Имя ключа JSON", disabled=True),
                "type": st.column_config.TextColumn("Тип данных", disabled=True),
                "description": st.column_config.TextColumn("Описание значения", required=True)
            },
            num_rows="fixed",
            key=f"main_editor_{p_id if p_id else 'new'}",
            hide_index=True,
            width='stretch'
        )
        current_schema['main'] = edited_main_df.to_dict(orient="records")

    elif schema_tab == "Чек-лист":
        st.write("ℹ️ *Для чек-листа все параметры имеют логический тип (bool). Вы можете добавлять и удалять пункты чек-листа.*")
        df_chk = pd.DataFrame(current_schema['checklist'])
        if df_chk.empty or "key" not in df_chk.columns:
            df_chk = pd.DataFrame(columns=["key", "type", "description"])
        # Убеждаемся что поле type всегда заполнено bool
        df_chk['type'] = 'bool'

        edited_chk_df = st.data_editor(
            df_chk,
            column_config={
                "key": st.column_config.TextColumn("Пункт чек-листа (латиницей, например, greeting)", required=True),
                "type": st.column_config.TextColumn("Тип данных", disabled=True),
                "description": st.column_config.TextColumn("Описание пункта чек-листа", required=True)
            },
            num_rows="dynamic",
            key=f"checklist_editor_{p_id if p_id else 'new'}",
            hide_index=True,
            width='stretch'
        )

        chk_records = []
        for _, row in edited_chk_df.iterrows():
            if pd.notna(row['key']) and str(row['key']).strip():
                chk_records.append({
                    "key": str(row['key']).strip(),
                    "type": "bool",
                    "description": str(row['description']).strip() if pd.notna(row['description']) else ""
                })
        current_schema['checklist'] = chk_records

    elif schema_tab == "Метрики":
        st.write("ℹ️ *Здесь настраиваются дополнительные числовые или строковые метрики звонка.*")
        df_met = pd.DataFrame(current_schema['metrics'])
        if df_met.empty or "key" not in df_met.columns:
            df_met = pd.DataFrame(columns=["key", "type", "description"])

        edited_met_df = st.data_editor(
            df_met,
            column_config={
                "key": st.column_config.TextColumn("Имя ключа метрики (латиницей, например, hold_time_sec)", required=True),
                "type": st.column_config.SelectboxColumn("Тип данных", options=["num", "str", "bool", "list", "dict"], required=True),
                "description": st.column_config.TextColumn("Описание метрики", required=True)
            },
            num_rows="dynamic",
            key=f"metrics_editor_{p_id if p_id else 'new'}",
            hide_index=True,
            width='stretch'
        )

        met_records = []
        for _, row in edited_met_df.iterrows():
            if pd.notna(row['key']) and str(row['key']).strip():
                met_records.append({
                    "key": str(row['key']).strip(),
                    "type": str(row['type']).strip() if pd.notna(row['type']) else "num",
                    "description": str(row['description']).strip() if pd.notna(row['description']) else ""
                })
        current_schema['metrics'] = met_records

    col_f1, col_f2, col_f3, col_f4 = st.columns([1, 1, 1, 1])
    save_btn = col_f1.button("Сохранить", width='stretch')
    cancel_btn = col_f2.button("Отмена", width='stretch')
    check_btn = col_f3.button("Проверить", width='stretch')

    if save_btn:
        if name and text:
            # Валидация ChatML структуры
            is_valid, err_msg = validate_chatml_template(text)
            if not is_valid:
                st.error(f"Ошибка валидации промпта: {err_msg}")
            else:
                upsert_prompt(name, text, is_default=prompt['is_default'] if prompt else False, prompt_id=p_id, schema_json=current_schema)
                st.success("Сохранено!")
                if session_schema_key in st.session_state:
                    del st.session_state[session_schema_key]
                st.session_state.show_editor = False
                st.session_state.editing_prompt = None
                st.session_state.check_result = None
                st.rerun()
        else:
            st.error("Название и текст не могут быть пустыми.")

    if cancel_btn:
        if session_schema_key in st.session_state:
            del st.session_state[session_schema_key]
        st.session_state.show_editor = False
        st.session_state.editing_prompt = None
        st.session_state.check_result = None
        st.rerun()

    if check_btn:
        if text:
            # Валидация ChatML структуры
            is_valid, err_msg = validate_chatml_template(text)
            if not is_valid:
                st.error(f"Ошибка валидации промпта: {err_msg}")
            else:
                with st.status("Выполняется анализ...", expanded=True) as status:
                    try:
                        full_response = st.write_stream(check_prompt(text, st.session_state.test_transcript, stream=True, schema_fields=current_schema))
                        st.session_state.check_result = full_response
                        status.update(label="Анализ завершен!", state="complete", expanded=False)
                        st.rerun()
                    except Exception as e:
                        status.update(label="Ошибка!", state="error")
                        st.error(f"Ошибка при проверке: {e}")
        else:
            st.error("Текст промпта не может быть пустым.")

    if st.session_state.check_result:
        st.subheader("Результат проверки")
        st.text_area("Ответ LLM", value=st.session_state.check_result, height=200, disabled=True)
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
            width='stretch',
            hide_index=True,
            column_config={
            "Название": st.column_config.TextColumn("Название", help="Жирным выделен промпт по умолчанию")
            }
        )

        prompt_options = {p['id']: p['name'] for p in prompts}
        selected_prompt_id = st.selectbox("Выберите промпт для действий", options=list(prompt_options.keys()), format_func=lambda x: prompt_options[x])

        selected_prompt = next((p for p in prompts if p['id'] == selected_prompt_id), None)

        col1, col2, col3 = st.columns(3)

        if col1.button("Добавить новый", width='stretch'):
            st.session_state.show_editor = True
            st.session_state.editing_prompt = None
            st.session_state.check_result = None
            st.rerun()

        if col2.button("Изменить выбранный", width='stretch', disabled=selected_prompt is None):
            st.session_state.show_editor = True
            st.session_state.editing_prompt = selected_prompt
            st.session_state.check_result = None
            st.rerun()

        if col3.button("Сделать по умолчанию", width='stretch', disabled=selected_prompt is None):
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
            st.session_state.check_result = None
            st.rerun()

# --- Раздел 3: Задания на анализ (вместо Ручного запуска) ---
st.header("Задания на анализ")

# Форма добавления задания
with st.expander("Добавить новое задание"):
    with st.form("add_task_form"):
        today = datetime.now().date()
        date_range = st.date_input("Период анализа", (today - timedelta(days=7), today))

        prompt_options = {p['id']: p['name'] for p in prompts} if prompts else {}
        task_prompt_id = st.selectbox("Промпт", options=list(prompt_options.keys()), format_func=lambda x: prompt_options[x])

        analyze_all = st.checkbox("Анализировать все звонки (игнорировать фильтр по телефонам)", value=False)

        submit_task = st.form_submit_button("Добавить задание")

        if submit_task:
            if not (isinstance(date_range, tuple) and len(date_range) == 2):
                st.error("Выберите диапазон дат.")
            elif task_prompt_id:
                add_task(task_prompt_id, date_range[0], date_range[1], analyze_all)
                st.success("Задание добавлено!")
                st.rerun()

# Список заданий
tasks = get_all_tasks()
if tasks:
    def get_status_icon(status):
        if status == 'planned': return "🆕"
        if status == 'processing': return "⏳"
        if status == 'completed': return "✅"
        return status

    task_data = []
    for t in tasks:
        task_data.append({
            "ID": t['id'],
            "Дата создания": t['created_at'].strftime("%Y-%m-%d %H:%M"),
            "Промпт": t['prompt_name'],
            "Период": f"{t['start_date']} - {t['end_date']}",
            "ASR": get_status_icon(t['asr_status']),
            "LLM": get_status_icon(t['llm_status']),
            "Все": "✅" if t['analyze_all'] else "❌"
        })

    st.dataframe(pd.DataFrame(task_data), hide_index=True, width='stretch')

    task_to_delete = st.selectbox("Удалить задание (ID)", options=[t['id'] for t in tasks])
    if st.button("Удалить выбранное задание"):
        delete_task(task_to_delete)
        st.success("Задание удалено!")
        st.rerun()
else:
    st.info("Нет активных или запланированных заданий.")

# --- Раздел 4: Редактирование значений (Настроение и Цели) ---
st.header("Редактирование значений (Настроение и Цели)")

all_mappings = get_value_mappings()

if prompts:
    # Выбор промпта для редактирования значений
    prompt_options_map = {p['id']: p['name'] for p in prompts}
    mapping_prompt_id = st.selectbox(
        "Выберите промпт для настройки значений",
        options=list(prompt_options_map.keys()),
        format_func=lambda x: prompt_options_map[x],
        key="mapping_prompt_select"
    )

    # Находим или создаем запись для этого промпта
    current_mapping = next((m for m in all_mappings if m['prompt_id'] == mapping_prompt_id), None)
    if not current_mapping:
        current_mapping = {
            "prompt_id": mapping_prompt_id,
            "call_purpose": [],
            "client_sentiment": []
        }
        all_mappings.append(current_mapping)

    def manage_mapping_list(label, key_name):
        st.subheader(label)
        items = current_mapping.get(key_name, [])

        # Превращаем [{key: val}, ...] в [{"key": k, "label": v}, ...] для удобства st.data_editor
        display_items = []
        for item in items:
            for k, v in item.items():
                display_items.append({"key": k, "label": v})

        if not display_items:
            df_items = pd.DataFrame(columns=["key", "label"])
        else:
            df_items = pd.DataFrame(display_items)

        edited_items = st.data_editor(
            df_items,
            column_config={
                "key": st.column_config.TextColumn("Техническое значение (в JSON)"),
                "label": st.column_config.TextColumn("Представление для отчетов")
            },
            num_rows="dynamic",
            key=f"editor_{key_name}_{mapping_prompt_id}",
            hide_index=True,
            width='stretch'
        )

        if st.button(f"Сохранить {label}", key=f"save_{key_name}"):
            new_list = []
            for _, row in edited_items.iterrows():
                if pd.notna(row['key']) and str(row['key']).strip():
                    new_list.append({str(row['key']).strip(): str(row['label']).strip()})
            current_mapping[key_name] = new_list
            set_value_mappings(all_mappings)
            st.success("Сохранено!")
            st.rerun()

        return items

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        purpose_items = manage_mapping_list("Цели звонка (call_purpose)", "call_purpose")
    with col_m2:
        sentiment_items = manage_mapping_list("Настроение клиента (client_sentiment)", "client_sentiment")

    # --- Подраздел: Поиск и замена ---
    st.subheader("Поиск и замена несоответствующих значений")
    if st.button("Проверить соответствие"):
        # Получаем все уникальные значения из БД для этого промпта
        with get_pg_connection() as conn:
            cur = conn.cursor()

            # call_purpose
            cur.execute("SELECT DISTINCT call_purpose FROM evaluations WHERE prompt_id = %s AND call_purpose IS NOT NULL", (mapping_prompt_id,))
            db_purposes = [r[0] for r in cur.fetchall()]
            allowed_purposes = [list(item.keys())[0] for item in current_mapping['call_purpose']]
            invalid_purposes = [v for v in db_purposes if v not in allowed_purposes]

            # client_sentiment
            cur.execute("SELECT DISTINCT client_sentiment FROM evaluations WHERE prompt_id = %s AND client_sentiment IS NOT NULL", (mapping_prompt_id,))
            db_sentiments = [r[0] for r in cur.fetchall()]
            allowed_sentiments = [list(item.keys())[0] for item in current_mapping['client_sentiment']]
            invalid_sentiments = [v for v in db_sentiments if v not in allowed_sentiments]

            st.session_state.invalid_values = {
                "call_purpose": invalid_purposes,
                "client_sentiment": invalid_sentiments,
                "allowed_purposes": allowed_purposes,
                "allowed_sentiments": allowed_sentiments
            }

    if "invalid_values" in st.session_state:
        iv = st.session_state.invalid_values
        has_invalid = False

        # Вспомогательная функция для отображения меток в выпадающем списке
        def get_label_map(key_name):
            items = current_mapping.get(key_name, [])
            l_map = {}
            for item in items:
                for k, v in item.items():
                    l_map[k] = f"{v} ({k})"
            return l_map

        if iv["call_purpose"]:
            has_invalid = True
            st.write("### Некорректные цели звонка")
            replacements_purpose = {}
            l_map_purp = get_label_map("call_purpose")

            for val in iv["call_purpose"]:
                c1, c2 = st.columns(2)
                c1.write(f"`{val}`")
                replacements_purpose[val] = c2.selectbox(
                    f"Заменить {val} на:",
                    options=iv["allowed_purposes"],
                    format_func=lambda x: l_map_purp.get(x, x),
                    key=f"repl_purp_{val}"
                )
            if st.button("Применить замены для целей"):
                for old_v, new_v in replacements_purpose.items():
                    update_evaluations_value(mapping_prompt_id, 'call_purpose', old_v, new_v)
                st.success("Значения целей обновлены!")
                st.session_state.invalid_values["call_purpose"] = []
                st.rerun()

        if iv["client_sentiment"]:
            has_invalid = True
            st.write("### Некорректные значения настроения")
            replacements_sentiment = {}
            l_map_sent = get_label_map("client_sentiment")

            for val in iv["client_sentiment"]:
                c1, c2 = st.columns(2)
                c1.write(f"`{val}`")
                replacements_sentiment[val] = c2.selectbox(
                    f"Заменить {val} на:",
                    options=iv["allowed_sentiments"],
                    format_func=lambda x: l_map_sent.get(x, x),
                    key=f"repl_sent_{val}"
                )
            if st.button("Применить замены для настроения"):
                for old_v, new_v in replacements_sentiment.items():
                    update_evaluations_value(mapping_prompt_id, 'client_sentiment', old_v, new_v)
                st.success("Значения настроения обновлены!")
                st.session_state.invalid_values["client_sentiment"] = []
                st.rerun()

        if not has_invalid:
            st.success("Все значения соответствуют настройкам!")

    # --- Подраздел: Синонимы полей ---
    st.header("Синонимы полей (названия колонок и показателей)")

    # Получаем динамические ключи и колонки из БД
    with get_pg_connection() as conn:
        cur = conn.cursor()

        # 1. Получаем реальные колонки из таблиц calls и evaluations
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name IN ('calls', 'evaluations')
            AND table_schema = 'public'
            AND column_name NOT IN ('linkedid', 'prompt_id', 'file_path', 'processing_status', 'processing_duration', 'created_at')
        """)
        db_columns = [r[0] for r in cur.fetchall()]

        # Добавляем вычисляемые поля, которые используются в отчетах
        virtual_columns = ["operator_name", "client_number"]

        # 2. Получаем динамические ключи из JSON для этого промпта
        cur.execute("""
            SELECT checklist_json, metrics_json
            FROM evaluations
            WHERE prompt_id = %s
            LIMIT 50
        """, (mapping_prompt_id,))
        rows = cur.fetchall()

        json_keys = set()
        for r in rows:
            if r[0]: json_keys.update([f"checklist.{k}" for k in r[0].keys()])
            if r[1]: json_keys.update([f"metrics.{k}" for k in r[1].keys()])

        all_tech_names = sorted(list(set(db_columns + virtual_columns + list(json_keys))))

        # Получаем текущие синонимы из БД
        db_synonyms = get_cached_synonyms(mapping_prompt_id)

        # Подготавливаем данные для таблицы
        synonyms_data = []
        for tn in all_tech_names:
            synonyms_data.append({
                "technical_name": tn,
                "synonym": db_synonyms.get(tn, "")
            })

        df_synonyms = pd.DataFrame(synonyms_data)

        st.write(f"Настройте синонимы для промпта: **{prompt_options_map[mapping_prompt_id]}**")
        edited_synonyms = st.data_editor(
            df_synonyms,
            column_config={
                "technical_name": st.column_config.TextColumn("Техническое имя", disabled=True),
                "synonym": st.column_config.TextColumn("Синоним (человекочитаемое имя)")
            },
            hide_index=True,
            width='stretch',
            key=f"synonyms_editor_{mapping_prompt_id}"
        )

        if st.button("Сохранить синонимы"):
            new_synonyms = {}
            for _, row in edited_synonyms.iterrows():
                if pd.notna(row['synonym']) and str(row['synonym']).strip():
                    new_synonyms[row['technical_name']] = str(row['synonym']).strip()
            set_field_synonyms(mapping_prompt_id, new_synonyms)
            st.cache_data.clear() # Сбрасываем кэш, чтобы увидеть изменения
            st.success("Синонимы сохранены!")
            st.rerun()

else:
    st.info("Добавьте хотя бы один промпт для настройки значений.")

# --- Раздел 5: Телефонный справочник ---
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
        width='stretch'
    )

    col_p1, col_p2 = st.columns(2)
    if col_p1.button("Сохранить изменения", width='stretch'):
        # Проверяем что изменилось
        for i, row in edited_df.iterrows():
            orig_row = df_phones.iloc[i]
            if row['use'] != orig_row['use']:
                update_phone_use(row['number'], row['use'])
        st.success("Изменения сохранены!")
        st.rerun()

    if col_p2.button("Синхронизировать список номеров", width='stretch'):
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
