import json
import logging
import time
import re
from models import get_locked_llm

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Ты — эксперт по контролю качества в медицинском колл-центре. Проанализируй диалог оператора и клиента и верни **только** JSON без лишних слов.
Поля:
- politeness_score: число от 0 до 10
- client_sentiment: "positive", "neutral", "negative", "conflict"
- call_purpose: "appointment", "consultation", "complaint", "cancel_appointment", "other"
- call_summary: краткое содержание 1-2 предложения
- checklist: объект с ключами: greeting, introduced_himself, identified_need, informed_price, agreed_datetime, handled_objection, farewell. Каждое поле true/false.
- metrics: объект с дополнительной информацией, например, interruptions_count, hold_time_sec, medication_mentioned (true/false)

Диалог:
{transcript}
"""

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
                        # Если этот блок не распарсился, продолжаем искать дальше
                        start_pos = -1
                        continue
    return None

def analyze_transcript(transcript_text: str, prompt_template: str = None) -> dict:
    start_time = time.time()
    logger.info("Sending transcript to LLM for analysis...")

    llm = get_locked_llm()
    template = prompt_template if prompt_template else PROMPT_TEMPLATE

    # Уточняем инструкцию в конце промпта для надежности
    prompt = template.format(transcript=transcript_text)
    if "JSON" in prompt:
         prompt += "\nВерни СТРОГО только JSON-объект."

    output = llm.create_completion(prompt, max_tokens=1000, temperature=0.1)
    response = output['choices'][0]['text']

    duration = time.time() - start_time
    logger.info(f"LLM analysis finished in {duration:.2f}s")
    logger.debug(f"Raw LLM response: {response}")

    result = extract_json(response)

    if result is None:
        logger.error(f"Failed to extract JSON from LLM response. Raw output:\n{response}")
        raise ValueError("Valid JSON not found in LLM response")

    logger.info(f"LLM score: {result.get('politeness_score')}, sentiment: {result.get('client_sentiment')}")
    return result
