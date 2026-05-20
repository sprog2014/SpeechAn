import json
from models import load_llm

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

def analyze_transcript(transcript_text):
    llm = load_llm()
    prompt = PROMPT_TEMPLATE.format(transcript=transcript_text)
    output = llm.create_completion(prompt, max_tokens=500, temperature=0.1)
    response_text = output['choices'][0]['text']
    # Находим JSON в ответе
    start = response_text.find('{')
    end = response_text.rfind('}') + 1
    if start == -1 or end <= start:
        raise ValueError("JSON not found in LLM response")
    json_str = response_text[start:end]
    result = json.loads(json_str)
    return result