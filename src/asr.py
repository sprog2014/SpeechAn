import logging
import time
import torch
import torchaudio
from models import get_asr_model

logger = logging.getLogger(__name__)

# Убираем asr_inference_lock для параллельной обработки каналов.
# Модели на основе Torch и Conformer обычно поддерживают конкурентный вызов
# или Torch сам управляет очередями на CPU.

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

        with torch.no_grad():
            # word_timestamps=True
            asr_res = model.transcribe(chunk, word_timestamps=True)

        words = []
        if hasattr(asr_res, 'words'):
            words = asr_res.words
        elif isinstance(asr_res, dict) and 'words' in asr_res:
            words = asr_res['words']

        if words:
            for word in words:
                if hasattr(word, 'start'):
                    w_start, w_end, w_text = word.start, word.end, word.text
                else:
                    # Поддержка словарей
                    try:
                        w_start, w_end, w_text = word['start'], word['end'], word['text']
                    except (TypeError, KeyError):
                        # В некоторых версиях может быть кортеж или объект другого типа
                        continue

                word_start_abs = offset_sec + w_start
                word_end_abs = offset_sec + w_end

                next_step_start_sec = (current_start_sample + step_samples) / sample_rate
                if current_end_sample == total_samples or word_start_abs < next_step_start_sec:
                    all_words.append((word_start_abs, word_end_abs, w_text.strip()))
        elif isinstance(asr_res, str) and asr_res.strip():
            all_words.append((offset_sec, current_end_sample / sample_rate, asr_res.strip()))

        if current_end_sample == total_samples:
            break
        current_start_sample += step_samples

    duration = time.time() - start_time
    logger.info(f"GigaAM transcription finished: {len(all_words)} segments found in {duration:.2f}s")
    return all_words
