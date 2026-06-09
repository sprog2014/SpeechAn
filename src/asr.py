import logging
import time
import torch
import torchaudio
from models import get_asr_model, get_vad_model

logger = logging.getLogger(__name__)

def transcribe_audio(waveform: torch.Tensor, sample_rate: int = 16000):
    """
    Распознает аудио тензор (один канал).
    Теперь эта функция вызывается для заранее нарезанных фрагментов.
    """
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)

    model = get_asr_model()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    waveform = waveform.to(device).to(dtype).unsqueeze(0)
    length = torch.full([1], waveform.shape[-1], device=device)

    with torch.no_grad():
        encoded, encoded_len = model.forward(waveform, length)
        decoded_list = model._decode(encoded, encoded_len, length, word_timestamps=False)
        if decoded_list:
            return decoded_list[0][0].strip()
    return ""

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
        text = transcribe_audio(chunk, sample_rate)
        if text:
            all_events.append({'start': s, 'end': e, 'channel': 'operator', 'text': text})

    for s, e in right_segs:
        chunk = waveform_right[int(s * sample_rate):int(e * sample_rate)]
        text = transcribe_audio(chunk, sample_rate)
        if text:
            all_events.append({'start': s, 'end': e, 'channel': 'client', 'text': text})

    # Сортируем по времени начала
    all_events.sort(key=lambda x: x['start'])

    # Форматируем для возврата
    return [(ev['start'], ev['end'], ev['channel'], ev['text']) for ev in all_events]
