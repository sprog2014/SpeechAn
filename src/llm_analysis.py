import json
import logging
import time
import re
from typing import List, Dict, Any, Union
from pydantic import create_model, Field
from models import get_locked_llm

logger = logging.getLogger(__name__)

# Дефолтный промпт в формате ChatML
DEFAULT_CHATML_PROMPT = """<|im_start|>system
Ты — эксперт по контролю качества в медицинском колл-центре.
Твоя задача — проанализировать диалог между оператором и клиентом.
Результат анализа ты обязан выдать СТРОГО в формате JSON, соответствующем следующей схеме:
{json_schema}

Не добавляй никакого вступительного или заключительного текста, только один JSON объект.
<|im_end|>
<|im_start|>user
Проанализируй следующий диалог:
---
{transcript}
---
Верни JSON-объект с результатами анализа.
<|im_end|>"""

def build_dynamic_model(schema_data: Union[List[Dict[str, Any]], Dict[str, Any]]):
    """
    Динамически конструирует вложенную Pydantic модель на основе структуры из БД.
    Поддерживает как старый плоский формат (list), так и новый вложенный (dict).
    """
    type_mapping = {
        'str': str,
        'bool': bool,
        'num': float,
        'list': List[Any],
        'dict': Dict[str, Any],
        'enum': str
    }

    # Если это словарь с вложенной структурой
    if isinstance(schema_data, dict) and any(k in schema_data for k in ['main', 'checklist', 'metrics']):
        main_list = schema_data.get('main', [])
        checklist_list = schema_data.get('checklist', [])
        metrics_list = schema_data.get('metrics', [])

        # 1. Сборка Чек-листа (только булевые значения)
        chk_fields = {}
        for f in checklist_list:
            key = f.get('key')
            if key:
                chk_fields[key] = (bool, Field(description=f.get('description', '')))
        ChecklistModel = create_model('Checklist', **chk_fields)

        # 2. Сборка Метрик
        met_fields = {}
        for f in metrics_list:
            key = f.get('key')
            if key:
                t_str = f.get('type', 'str')
                py_type = type_mapping.get(t_str, Any)
                met_fields[key] = (py_type, Field(description=f.get('description', '')))
        MetricsModel = create_model('Metrics', **met_fields)

        # 3. Сборка Главной модели
        main_fields = {}
        for f in main_list:
            key = f.get('key')
            if key:
                t_str = f.get('type', 'str')
                py_type = type_mapping.get(t_str, Any)
                main_fields[key] = (py_type, Field(description=f.get('description', '')))

        # Добавляем вложенные модели
        main_fields['checklist'] = (ChecklistModel, Field(description="Чек-лист соответствия требованиям разговора"))
        main_fields['metrics'] = (MetricsModel, Field(description="Количественные метрики и параметры разговора"))

        return create_model('TextAnalysis', **main_fields)

    # Старый плоский формат (list) - обратная совместимость
    fields = {}
    schema_list = schema_data if isinstance(schema_data, list) else []
    for field in schema_list:
        key = field.get('key')
        if not key:
            continue
        t_str = field.get('type', 'str')
        desc = field.get('description', '')

        py_type = type_mapping.get(t_str, Any)
        fields[key] = (py_type, Field(description=desc))

    return create_model('TextAnalysis', **fields)

def validate_chatml_template(prompt_text: str) -> tuple[bool, str]:
    """
    Проверяет корректность ChatML структуры в тексте промпта.
    Должен содержать как минимум блоки system и user, а также плейсхолдер {transcript}.
    """
    if not prompt_text:
        return False, "Промпт пустой"

    if "{transcript}" not in prompt_text:
        return False, "Промпт должен содержать плейсхолдер {transcript} для вставки транскрипта диалога."

    # Проверим наличие тегов ChatML
    required_tags = ["<|im_start|>system", "<|im_start|>user"]
    for tag in required_tags:
        if tag not in prompt_text:
            return False, f"Промпт должен использовать формат ChatML и содержать тег '{tag}'"

    # Проверим закрывающие теги
    if prompt_text.count("<|im_start|>") != prompt_text.count("<|im_end|>"):
        return False, "Количество открывающих тегов <|im_start|> не совпадает с количеством закрывающих <|im_end|>"

    return True, ""

def parse_chatml_to_messages(chatml_text: str) -> list[dict]:
    """
    Парсит ChatML шаблон на список сообщений: [{"role": "system", "content": "..."}]
    """
    pattern = r"<\|im_start\|>(\w+)\s+(.*?)(?:<\|im_end\|>|$)"
    matches = re.findall(pattern, chatml_text, re.DOTALL)

    messages = []
    for role, content in matches:
        messages.append({
            "role": role.strip(),
            "content": content.strip()
        })
    return messages

def extract_json_with_pydantic(text: str, model_class) -> dict:
    """Извлекает первый валидный JSON объект из текста и валидирует его Pydantic-моделью."""
    if not text:
        return None

    bracket_count = 0
    start_pos = -1
    for i, char in enumerate(text):
        if char == '{':
            if bracket_count == 0:
                start_pos = i
            bracket_count += 1
        elif char == '}':
            if bracket_count > 0:
                bracket_count -= 1
                if bracket_count == 0 and start_pos != -1:
                    potential_json = text[start_pos:i+1]
                    try:
                        # Валидируем через Pydantic
                        parsed = model_class.model_validate_json(potential_json)
                        return parsed.model_dump()
                    except Exception as ve:
                        logger.warning(f"Pydantic validation failed: {ve}. Trying standard json.loads...")
                        try:
                            return json.loads(potential_json)
                        except json.JSONDecodeError:
                            pass
                        start_pos = -1
                        continue
    return None

def analyze_transcript(transcript_text: str, prompt_template: str = None, schema_fields: Union[List[Dict[str, Any]], Dict[str, Any]] = None) -> dict:
    start_time = time.time()
    logger.info("Sending transcript to LLM for analysis via Chat Completion...")

    llm = get_locked_llm()

    # 1. Построение Pydantic-модели и схемы
    if not schema_fields:
        # Default fallback nested fields
        schema_fields = {
            "main": [
                {"key": "politeness_score", "type": "num", "description": "Оценка вежливости оператора от 0 до 10"},
                {"key": "client_sentiment", "type": "str", "description": "Настроение клиента: positive, neutral, negative или conflict"},
                {"key": "call_purpose", "type": "str", "description": "Цель звонка: appointment, consultation, complaint, cancel_appointment или other"},
                {"key": "call_summary", "type": "str", "description": "Краткое содержание диалога (1-2 предложения)"}
            ],
            "checklist": [
                {"key": "greeting", "type": "bool", "description": "Приветствие"},
                {"key": "introduced_himself", "type": "bool", "description": "Представился"},
                {"key": "identified_need", "type": "bool", "description": "Выявил потребность"},
                {"key": "informed_price", "type": "bool", "description": "Сообщил стоимость"},
                {"key": "agreed_datetime", "type": "bool", "description": "Согласовал дату/время"},
                {"key": "handled_objection", "type": "bool", "description": "Отработал возражение"},
                {"key": "farewell", "type": "bool", "description": "Прощание"}
            ],
            "metrics": [
                {"key": "interruptions_count", "type": "num", "description": "Количество перебиваний"},
                {"key": "hold_time_sec", "type": "num", "description": "Время удержания в секундах"},
                {"key": "medication_mentioned", "type": "bool", "description": "Упоминание лекарств"}
            ]
        }

    ModelClass = build_dynamic_model(schema_fields)
    schema_json_str = json.dumps(ModelClass.model_json_schema(), indent=2, ensure_ascii=False)

    # 2. Подготовка промпта
    if not prompt_template:
        prompt_template = DEFAULT_CHATML_PROMPT

    # Подставляем схему, если есть плейсхолдер {json_schema}
    if "{json_schema}" in prompt_template:
        prompt_template = prompt_template.replace("{json_schema}", schema_json_str)
    else:
        # Если плейсхолдера нет, принудительно добавим в системный блок
        if "<|im_start|>system" in prompt_template:
            prompt_template = prompt_template.replace(
                "<|im_start|>system",
                f"<|im_start|>system\nОжидаемая JSON схема ответа:\n{schema_json_str}\n"
            )

    # Подставляем диалог
    user_message = prompt_template.replace("{transcript}", transcript_text)

    # 3. Парсинг ChatML в сообщения
    messages = parse_chatml_to_messages(user_message)
    if not messages:
        messages = [
            {"role": "system", "content": f"Ожидаемая JSON схема:\n{schema_json_str}"},
            {"role": "user", "content": user_message}
        ]

    try:
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=2048,
            temperature=0.1,
            response_format={
                "type": "json_object",
                "schema": ModelClass.model_json_schema()
            }
        )
        response = output['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error during LLM generation: {e}")
        raise

    duration = time.time() - start_time
    logger.info(f"LLM analysis finished in {duration:.2f}s")
    logger.debug(f"Raw LLM response: {response}")

    result = extract_json_with_pydantic(response, ModelClass)

    if result is None:
        logger.error(f"Failed to extract JSON from LLM response. Raw output:\n{response}")
        raise ValueError("Valid JSON matching Pydantic model not found in LLM response")

    logger.info(f"LLM analysis successful. Keys found: {list(result.keys())}")
    return result

def check_prompt(prompt_template: str, transcript_text: str, stream: bool = False, schema_fields: Union[List[Dict[str, Any]], Dict[str, Any]] = None):
    """Отправляет промпт на проверку и возвращает сырой текст или стример."""
    start_time = time.time()
    logger.info(f"Checking prompt via LLM (stream={stream})...")

    llm = get_locked_llm()

    # Построение схемы Pydantic
    if not schema_fields:
        schema_fields = {
            "main": [
                {"key": "politeness_score", "type": "num", "description": "Оценка вежливости оператора от 0 до 10"},
                {"key": "client_sentiment", "type": "str", "description": "Настроение клиента: positive, neutral, negative или conflict"},
                {"key": "call_purpose", "type": "str", "description": "Цель звонка: appointment, consultation, complaint, cancel_appointment или other"},
                {"key": "call_summary", "type": "str", "description": "Краткое содержание диалога (1-2 предложения)"}
            ],
            "checklist": [
                {"key": "greeting", "type": "bool", "description": "Приветствие"},
                {"key": "introduced_himself", "type": "bool", "description": "Представился"},
                {"key": "identified_need", "type": "bool", "description": "Выявил потребность"},
                {"key": "informed_price", "type": "bool", "description": "Сообщил стоимость"},
                {"key": "agreed_datetime", "type": "bool", "description": "Согласовал дату/время"},
                {"key": "handled_objection", "type": "bool", "description": "Отработал возражение"},
                {"key": "farewell", "type": "bool", "description": "Прощание"}
            ],
            "metrics": [
                {"key": "interruptions_count", "type": "num", "description": "Количество перебиваний"},
                {"key": "hold_time_sec", "type": "num", "description": "Время удержания в секундах"},
                {"key": "medication_mentioned", "type": "bool", "description": "Упоминание лекарств"}
            ]
        }
    ModelClass = build_dynamic_model(schema_fields)
    schema_json_str = json.dumps(ModelClass.model_json_schema(), indent=2, ensure_ascii=False)

    # Подставляем схему
    if "{json_schema}" in prompt_template:
        prompt_template = prompt_template.replace("{json_schema}", schema_json_str)
    else:
        if "<|im_start|>system" in prompt_template:
            prompt_template = prompt_template.replace(
                "<|im_start|>system",
                f"<|im_start|>system\nОжидаемая JSON схема ответа:\n{schema_json_str}\n"
            )

    # Подставляем диалог
    user_message = prompt_template.replace("{transcript}", transcript_text)

    # Разбираем сообщения
    messages = parse_chatml_to_messages(user_message)
    if not messages:
        messages = [
            {"role": "user", "content": user_message}
        ]

    try:
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=2048,
            temperature=0.1,
            stream=stream,
            response_format={
                "type": "json_object",
                "schema": ModelClass.model_json_schema()
            }
        )
        if stream:
            return output

        response = output['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error during LLM prompt check: {e}")
        raise

    duration = time.time() - start_time
    logger.info(f"LLM check finished in {duration:.2f}s")
    return response
