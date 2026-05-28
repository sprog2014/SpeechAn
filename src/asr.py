import logging
import time
import torch
import torchaudio
from models import get_asr_model

logger = logging.getLogger(__name__)

def transcribe_audio(waveform: torch.Tensor, sample_rate: int = 16000, chunk_sec: int = 20, overlap_sec: int = 2):
    """
    Распознает аудио тензор с использованием GigaAM v3_e2e_ctc и кастомной нарезки.
    Возвращает список сегментов: [(start, end, text), ...]
    """
    start_time_perf = time.time()

    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0) # К mono если вдруг стерео

    model = get_asr_model()

    total_samples = waveform.shape[0]
    chunk_samples = chunk_sec * sample_rate
    overlap_samples = overlap_sec * sample_rate
    step_samples = chunk_samples - overlap_samples

    result = []
    start_samp = 0

    while start_samp < total_samples:
        end_samp = min(start_samp + chunk_samples, total_samples)
        chunk = waveform[start_samp:end_samp]

        # Смещение текущего куска в секундах
        offset_sec = start_samp / sample_rate

        # GigaAM ожидает [1, samples]
        chunk_input = chunk.unsqueeze(0)

        with torch.no_grad():
            # word_timestamps=True возвращает объект с полем words: [word, start, end]
            # Важно: GigaAM возвращает таймстампы относительно начала переданного куска
            transcription_result = model.transcribe(chunk_input, word_timestamps=True)

            if hasattr(transcription_result, 'words') and transcription_result.words:
                # Группируем слова в один сегмент для этого куска (или можно по предложениям)
                # Для простоты и совместимости с текущей логикой worker.py,
                # соберем весь текст куска и определим его общие границы

                chunk_text = " ".join([w.text for w in transcription_result.words])
                if chunk_text.strip():
                    actual_start = transcription_result.words[0].start + offset_sec
                    actual_end = transcription_result.words[-1].end + offset_sec
                    result.append((actual_start, actual_end, chunk_text.strip()))
            elif isinstance(transcription_result, str) and transcription_result.strip():
                # Если word_timestamps почему-то не сработал или не поддерживается как ожидалось
                result.append((offset_sec, end_samp / sample_rate, transcription_result.strip()))

        if end_samp == total_samples:
            break
        start_samp += step_samples

    # Умный мерж overlap-зон (простое удаление дубликатов по времени или тексту может быть сложным)
    # В данном случае, так как мы берем "весь текст куска", overlap может привести к повторам.
    # Для первой итерации оставим как есть, но в идеале нужно учитывать overlap_sec.

    duration = time.time() - start_time_perf
    logger.info(f"Transcription finished: {len(result)} segments found in {duration:.2f}s")
    return result
