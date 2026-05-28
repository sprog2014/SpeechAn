import logging
import time
import torch
from models import get_asr_model

logger = logging.getLogger(__name__)

def transcribe_audio(waveform: torch.Tensor, sample_rate: int = 16000, chunk_sec: int = 20, overlap_sec: int = 2):
    """
    Распознает аудио тензор с использованием GigaAM v3_e2e_ctc и кастомной нарезки.
    Возвращает список сегментов: [(start, end, text), ...]
    """
    start_time_perf = time.time()

    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0) # К mono

    model = get_asr_model()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    total_samples = waveform.shape[0]
    chunk_samples = chunk_sec * sample_rate
    overlap_samples = overlap_sec * sample_rate
    step_samples = chunk_samples - overlap_samples

    all_words = []
    start_samp = 0

    # Мы будем использовать порог для фильтрации слов в зонах перекрытия
    # Чтобы не было дублей, берем слова из текущего куска только если они
    # находятся во "второй половине" перекрытия (или после него).
    # Для самого первого куска берем всё.

    while start_samp < total_samples:
        end_samp = min(start_samp + chunk_samples, total_samples)
        chunk = waveform[start_samp:end_samp]

        offset_sec = start_samp / sample_rate

        # Подготовка тензора (имитируем prepare_wav но без загрузки с диска)
        chunk_input = chunk.to(device).to(dtype).unsqueeze(0)
        length = torch.full([1], chunk_input.shape[-1], device=device)

        with torch.no_grad():
            # Выполняем форвард и декодирование напрямую, так как transcribe() хочет файл
            encoded, encoded_len = model.forward(chunk_input, length)
            # _decode возвращает List[Tuple[text, List[Word]]]
            decoded_list = model._decode(encoded, encoded_len, length, word_timestamps=True)

            if decoded_list and decoded_list[0][1]:
                words = decoded_list[0][1]
                for w in words:
                    abs_start = w.start + offset_sec
                    abs_end = w.end + offset_sec

                    # Фильтр дубликатов:
                    # Если это не первый кусок, игнорируем слова, которые начинаются в первой половине overlap
                    if start_samp > 0:
                        if w.start < (overlap_sec / 2):
                            continue

                    # Если это не последний кусок, игнорируем слова, которые начинаются во второй половине overlap
                    if end_samp < total_samples:
                        if w.start >= (chunk_sec - overlap_sec / 2):
                            continue

                    all_words.append((abs_start, abs_end, w.text))

        if end_samp == total_samples:
            break
        start_samp += step_samples

    # Группируем слова в предложения/сегменты для читаемости
    # (Например, по 10 слов или по паузам > 1 сек)
    result = []
    if all_words:
        current_seg_words = []
        seg_start = all_words[0][0]

        for i, (w_start, w_end, w_text) in enumerate(all_words):
            if current_seg_words and (w_start - all_words[i-1][1] > 1.0 or len(current_seg_words) >= 15):
                # Закрываем сегмент
                result.append((seg_start, all_words[i-1][1], " ".join(current_seg_words)))
                current_seg_words = [w_text]
                seg_start = w_start
            else:
                current_seg_words.append(w_text)

        if current_seg_words:
            result.append((seg_start, all_words[-1][1], " ".join(current_seg_words)))

    duration = time.time() - start_time_perf
    logger.info(f"Transcription finished: {len(result)} segments found in {duration:.2f}s")
    return result
