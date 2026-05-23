import json
import logging
import time
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

def analyze_transcript(transcript_text: str) -> dict:
    start_time = time.time()
    logger.info("Sending transcript to LLM for analysis...")

    llm = get_locked_llm()
    prompt = PROMPT_TEMPLATE.format(transcript=transcript_text)
    output = llm.create_completion(prompt, max_tokens=500, temperature=0.1)
    response = output['choices'][0]['text']

    duration = time.time() - start_time
    logger.info(f"LLM analysis finished in {duration:.2f}s")

    # Извлекаем JSON
    start = response.find('{')
    end = response.rfind('}') + 1
    if start == -1 or end <= start:
        logger.error(f"Failed to find JSON in LLM response: {response}")
        raise ValueError("JSON not found in LLM response")

    result = json.loads(response[start:end])
    logger.info(f"LLM score: {result.get('politeness_score')}, sentiment: {result.get('client_sentiment')}")
    return result
