import logging
import time
import torch
import torchaudio
from models import get_asr_model

logger = logging.getLogger(__name__)

def transcribe_audio(audio_path: str, language: str = "ru"):
    """
    Транскрибирует аудиофайл с использованием GigaAM v3_e2e_ctc и кастомной нарезки.
    Возвращает список сегментов: [(start, end, text), ...]
    """
    start_time = time.time()
    logger.info(f"Starting GigaAM transcription for {audio_path}")

    model = get_asr_model()

    # 1. Загружаем аудио (GigaAM ожидает 16кГц)
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Если аудио многоканальное, усредняем до моно (для отдельного канала это не должно быть проблемой)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

    except Exception as e:
        logger.error(f"Failed to load audio for transcription: {e}")
        return []

    chunk_sec = 20
    overlap_sec = 2

    total_samples = waveform.shape[1]
    chunk_samples = chunk_sec * sample_rate
    overlap_samples = overlap_sec * sample_rate
    step_samples = chunk_samples - overlap_samples

    result_segments = []

    current_start_sample = 0

    while current_start_sample < total_samples:
        current_end_sample = min(current_start_sample + chunk_samples, total_samples)
        chunk = waveform[:, current_start_sample:current_end_sample]

        # Смещение в секундах для текущего куска
        offset_sec = current_start_sample / sample_rate

        with torch.no_grad():
            # word_timestamps=True позволяет получить временные метки слов/фраз
            # В e2e версии это дает более точные результаты
            asr_res = model.transcribe(chunk, word_timestamps=True)

            # asr_res обычно объект с полем words или segments в зависимости от версии gigaam
            # По ссылке из описания: result.words содержит start, end, text
            if hasattr(asr_res, 'words') and asr_res.words:
                # Группируем слова в фразы или просто берем как есть
                # Для простоты и соответствия "по фразам", будем считать одно распознавание куска как набор фраз
                # Но GigaAM v3 e2e может вернуть уже нормализованный текст.

                # Если есть words, мы можем восстановить таймстампы относительно начала файла
                for word in asr_res.words:
                    result_segments.append((
                        offset_sec + word.start,
                        offset_sec + word.end,
                        word.text.strip()
                    ))
            elif isinstance(asr_res, str):
                # Если вернулась просто строка (без word_timestamps или если модель так работает)
                if asr_res.strip():
                    result_segments.append((
                        offset_sec,
                        current_end_sample / sample_rate,
                        asr_res.strip()
                    ))

        if current_end_sample == total_samples:
            break
        current_start_sample += step_samples

    # Склеивание сегментов, которые перекрываются (overlap)
    # Это упрощенная версия: просто берем всё. В идеале нужно удалять дубликаты в зонах перекрытия.
    # Но так как мы используем word_timestamps, мы можем просто отфильтровать те, что начинаются во второй половине overlap

    final_segments = []
    last_end = 0

    # Сортируем на всякий случай
    result_segments.sort(key=lambda x: x[0])

    for start, end, text in result_segments:
        # Если слово начинается раньше, чем закончилось предыдущее распознанное (с учетом перекрытия)
        # Мы можем использовать простую эвристику: если мы в зоне перекрытия (последние overlap_sec куска)
        # И это слово уже было в предыдущем куске, то пропускаем.
        # Но проще всего: если start >= last_end, то берем.

        if start >= last_end - (overlap_sec / 2):
            final_segments.append((start, end, text))
            last_end = end

    duration = time.time() - start_time
    logger.info(f"GigaAM transcription finished: {len(final_segments)} segments found in {duration:.2f}s")
    return final_segments
