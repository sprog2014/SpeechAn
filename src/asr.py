import logging
import time
import threading
import torch
import torchaudio
from models import get_asr_model

logger = logging.getLogger(__name__)

# Блокировка для инференса ASR (внутри одного процесса)
asr_inference_lock = threading.Lock()

def transcribe_audio(waveform, sample_rate, language: str = "ru"):
    """
    Транскрибирует аудио-тензор с использованием GigaAM v3_e2e_ctc и кастомной нарезки.
    """
    start_time = time.time()

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    model = get_asr_model()

    chunk_sec = 20
    overlap_sec = 2

    total_samples = waveform.shape[1]
    chunk_samples = chunk_sec * sample_rate
    overlap_samples = overlap_sec * sample_rate
    step_samples = chunk_samples - overlap_samples

    all_words = []
    current_start_sample = 0

    while current_start_sample < total_samples:
        current_end_sample = min(current_start_sample + chunk_samples, total_samples)
        chunk = waveform[:, current_start_sample:current_end_sample]
        offset_sec = current_start_sample / sample_rate

        with asr_inference_lock:
            with torch.no_grad():
                # v3_e2e_ctc возвращает результат, который может быть объектом или словарем
                asr_res = model.transcribe(chunk, word_timestamps=True)

        words = []
        # Пытаемся извлечь список слов из разных форматов ответа
        if hasattr(asr_res, 'words'):
            words = asr_res.words
        elif isinstance(asr_res, dict) and 'words' in asr_res:
            words = asr_res['words']
        elif hasattr(asr_res, 'segments'):
            # На случай если в этой версии используются сегменты вместо слов
            words = asr_res.segments

        if words:
            for word in words:
                try:
                    if hasattr(word, 'start'):
                        w_start, w_end, w_text = word.start, word.end, word.text
                    elif isinstance(word, dict):
                        w_start, w_end, w_text = word['start'], word['end'], word['text']
                    elif isinstance(word, (list, tuple)) and len(word) >= 3:
                        w_start, w_end, w_text = word[0], word[1], word[2]
                    else:
                        continue

                    word_start_abs = offset_sec + w_start
                    word_end_abs = offset_sec + w_end

                    # Дедупликация: берем слово, если оно начинается до начала следующего шага
                    next_step_start_sec = (current_start_sample + step_samples) / sample_rate
                    if current_end_sample == total_samples or word_start_abs < next_step_start_sec:
                        all_words.append((word_start_abs, word_end_abs, str(w_text).strip()))
                except Exception as e:
                    logger.debug(f"Failed to parse word/segment: {e}")
                    continue
        elif isinstance(asr_res, str) and asr_res.strip():
            # Fallback для plain text
            all_words.append((offset_sec, current_end_sample / sample_rate, asr_res.strip()))

        if current_end_sample == total_samples:
            break
        current_start_sample += step_samples

    duration = time.time() - start_time
    logger.info(f"GigaAM transcription finished: {len(all_words)} segments found in {duration:.2f}s")
    return all_words
