import logging
import time
import torch
import torchaudio
from models import get_asr_model, get_vad_model

logger = logging.getLogger(__name__)

def transcribe_audio(waveform: torch.Tensor, sample_rate: int = 16000):
    """
    Распознает аудио тензор (один канал) и возвращает:
    - text (str): Распознанный текст
    - diction_score (float): Четкость речи от 0.0 до 100.0
    - speed_wpm (int): Темп речи (слов в минуту)
    """
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)

    # Длительность фрагмента в секундах
    duration_sec = waveform.shape[-1] / sample_rate

    model = get_asr_model()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    waveform = waveform.to(device).to(dtype).unsqueeze(0)
    length = torch.full([1], waveform.shape[-1], device=device)

    with torch.no_grad():
        # 1. Получаем акустические эмбеддинги (логиты)
        encoded, encoded_len = model.forward(waveform, length)

        # 2. Вычисляем уверенность модели (Confidence Score)
        # Переводим логиты в вероятности (0.0 - 1.0) по оси алфавита
        probs = torch.softmax(encoded, dim=-1)
        # Берем максимальную вероятность (выбор модели) для каждого временного фрейма
        max_probs, _ = torch.max(probs, dim=-1)

        # Считаем среднюю уверенность только по значащим фреймам
        valid_probs = max_probs[0, :encoded_len[0]]
        mask = valid_probs > 0.01  # отсекаем абсолютную тишину

        if mask.sum() > 0:
            mean_prob = valid_probs[mask].mean().item()
            diction_score = round(mean_prob * 100, 1)
        else:
            diction_score = 0.0

        # 3. Декодирование текста
        decoded_list = model._decode(encoded, encoded_len, length, word_timestamps=False)
        text = decoded_list[0][0].strip() if decoded_list else ""

    # 4. Расчет темпа речи (Слов в минуту)
    speed_wpm = 0
    if text and duration_sec > 0:
        words_count = len(text.split())
        speed_wpm = int((words_count / duration_sec) * 60)

    return text, diction_score, speed_wpm

def get_speech_segments(waveform: torch.Tensor, sample_rate: int = 16000):
    """
    Использует Silero VAD для обнаружения фрагментов речи в канале.
    Возвращает список (start_sec, end_sec).
    """
    model, utils = get_vad_model()
    (get_speech_timestamps, _, _, _, _) = utils

    # Silero VAD ожидает тензор [batch, samples] или [samples]
    # И определенные частоты дискретизации (8000 или 16000)
    with torch.no_grad():
        speech_timestamps = get_speech_timestamps(waveform, model, sampling_rate=sample_rate)

    segments = []
    for ts in speech_timestamps:
        start_sec = ts['start'] / sample_rate
        end_sec = ts['end'] / sample_rate
        segments.append((start_sec, end_sec))

    # Склеиваем слишком короткие или близкие сегменты и ограничиваем по 25с
    refined_segments = []
    for s, e in segments:
        duration = e - s
        if duration < 0.2: # Игнорируем шумы короче 200мс
            continue

        # Если сегмент слишком длинный, режем его на куски по 20с
        while duration > 25.0:
            refined_segments.append((s, s + 20.0))
            s += 20.0
            duration -= 20.0

        refined_segments.append((s, s + duration))

    return refined_segments

def transcribe_with_vad(waveform_left: torch.Tensor, waveform_right: torch.Tensor, sample_rate: int = 16000):
    """
    Анализирует оба канала, выделяет фрагменты и распознает их.
    Возвращает список (start, end, channel, text)
    """
    logger.info("Starting VAD-based segmentation...")

    left_segs = get_speech_segments(waveform_left, sample_rate)
    right_segs = get_speech_segments(waveform_right, sample_rate)

    all_events = []

    for s, e in left_segs:
        chunk = waveform_left[int(s * sample_rate):int(e * sample_rate)]
        text, dict_score, speed = transcribe_audio(chunk, sample_rate)
        if text:
            all_events.append({
                'start': s, 'end': e, 'channel': 'operator', 'text': text,
                'diction_score': dict_score, 'speed_wpm': speed
            })

    for s, e in right_segs:
        chunk = waveform_right[int(s * sample_rate):int(e * sample_rate)]
        text, dict_score, speed = transcribe_audio(chunk, sample_rate)
        if text:
            all_events.append({
                'start': s, 'end': e, 'channel': 'client', 'text': text,
                'diction_score': dict_score, 'speed_wpm': speed
            })

    # Сортируем по времени начала
    all_events.sort(key=lambda x: x['start'])

    # Форматируем для возврата (теперь включая метрики)
    return [(ev['start'], ev['end'], ev['channel'], ev['text'], ev['diction_score'], ev['speed_wpm']) for ev in all_events]
