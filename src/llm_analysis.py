import json
import logging
import time
from models import get_locked_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — эксперт по контролю качества в медицинском колл-центре.
Твоя задача — проанализировать диалог между оператором и клиентом.
Результат анализа ты обязан выдать СТРОГО в формате JSON.
Не добавляй никакого вступительного или заключительного текста, только один JSON объект.

Формат JSON:
{{
  "politeness_score": число от 0 до 10,
  "client_sentiment": "positive", "neutral", "negative" или "conflict",
  "call_purpose": "appointment", "consultation", "complaint", "cancel_appointment" или "other",
  "call_summary": "краткое содержание 1-2 предложения",
  "checklist": {{
    "greeting": true/false,
    "introduced_himself": true/false,
    "identified_need": true/false,
    "informed_price": true/false,
    "agreed_datetime": true/false,
    "handled_objection": true/false,
    "farewell": true/false
  }},
  "metrics": {{
    "interruptions_count": число,
    "hold_time_sec": число,
    "medication_mentioned": true/false
  }}
}}"""

USER_PROMPT_TEMPLATE = """Проанализируй следующий диалог:
---
{transcript}
---
Верни JSON-объект с результатами анализа."""

def extract_json(text: str) -> dict:
    """Извлекает первый валидный JSON объект из текста."""
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
                        return json.loads(potential_json)
                    except json.JSONDecodeError:
                        start_pos = -1
                        continue
    return None

def analyze_transcript(transcript_text: str, prompt_template: str = None) -> dict:
    start_time = time.time()
    logger.info("Sending transcript to LLM for analysis via Chat Completion...")

    llm = get_locked_llm()

    # Если prompt_template передан, используем его как USER_PROMPT.
    # Если нет - используем наш стандартный шаблон.
    if prompt_template:
        user_message = prompt_template.replace("{transcript}", transcript_text)
    else:
        user_message = USER_PROMPT_TEMPLATE.format(transcript=transcript_text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    try:
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=1000,
            temperature=0.1
        )
        response = output['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error during LLM generation: {e}")
        raise

    duration = time.time() - start_time
    logger.info(f"LLM analysis finished in {duration:.2f}s")
    logger.debug(f"Raw LLM response: {response}")

    result = extract_json(response)

    if result is None:
        logger.error(f"Failed to extract JSON from LLM response. Raw output:\n{response}")
        raise ValueError("Valid JSON not found in LLM response")

    logger.info(f"LLM score: {result.get('politeness_score')}, sentiment: {result.get('client_sentiment')}")
    return result

def check_prompt(prompt_template: str, transcript_text: str) -> str:
    """Отправляет промпт на проверку без системного промпта и возвращает сырой текст."""
    start_time = time.time()
    logger.info("Checking prompt via LLM...")

    llm = get_locked_llm()

    user_message = prompt_template.replace("{transcript}", transcript_text)

    messages = [
        {"role": "user", "content": user_message}
    ]

    try:
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=1000,
            temperature=0.1
        )
        response = output['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error during LLM prompt check: {e}")
        raise

    duration = time.time() - start_time
    logger.info(f"LLM check finished in {duration:.2f}s")
    return response
