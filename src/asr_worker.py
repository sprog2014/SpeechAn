import os
import logging
import torch
import torchaudio
import time
import numpy as np
from denoiser import pretrained
from db_utils import (
    fetch_call_metadata, upsert_call, insert_transcript, get_pg_connection,
    check_transcript_exists, set_call_status, get_call_status
)
from asr import transcribe_with_vad
from logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

_denoiser_model = None

def get_denoiser_model():
    global _denoiser_model
    if _denoiser_model is None:
        logger.info("Initializing Facebook Denoiser model (dns64)...")
        _denoiser_model = pretrained.dns64().cpu()
        _denoiser_model.eval()
    return _denoiser_model

def denoise_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """
    Подавление шума с помощью Facebook Denoiser (Demucs).
    Ожидает 16кГц на входе.
    """
    if waveform.shape[-1] == 0:
        return waveform

    model = get_denoiser_model()

    # Denoiser ожидает [batch, channels, samples]
    is_1d = False
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0) # [1, samples]
        is_1d = True

    try:
        with torch.no_grad():
            # На вход подаем [batch, channels, samples] -> [1, channels, samples]
            denoised = model(waveform.unsqueeze(0))[0] # Берем первый элемент батча

        if is_1d:
            denoised = denoised.squeeze(0)
        return denoised
    except Exception as e:
        logger.error(f"Error during Facebook Denoiser processing: {e}")
        return waveform

def normalize_waveform_rms(waveform: torch.Tensor, target_rms: float = 0.05, max_gain: float = 8.0) -> tuple[torch.Tensor, float]:
    """
    RMS-нормализация для телефонных записей.
     target_rms = 0.05 — оптимальный уровень средней громкости для Silero VAD и GigaAM.
     max_gain = 8.0 (около 18 дБ) — защита от чрезмерного усиления абсолютной тишины.
    """
    # 1. Считаем среднеквадратичную мощность (RMS) сигнала
    rms = torch.sqrt(torch.mean(waveform ** 2)).item()

    # Защита от деления на ноль (если файл пустой или там абсолютная тишина)
    if rms > 1e-5:
        # Рассчитываем необходимый коэффициент усиления
        gain = target_rms / rms

        # Ограничиваем максимальное усиление, чтобы не выкрутить тихий шум в начале записи
        if gain > max_gain:
            gain = max_gain

        # Умножаем сигнал на коэффициент
        normalized = waveform * gain

        # Защитный жесткий лимитер: если после RMS-усиления отдельные пики
        # вышли за пределы допустимого диапазона, аккуратно срезаем их на уровне 0.95
        normalized = torch.clamp(normalized, -0.95, 0.95)

        return normalized, gain

    return waveform, 1.0

def process_asr(file_path: str):
    base = os.path.basename(file_path)
    linkedid = os.path.splitext(base)[0]
    logger.info(f"[{linkedid}] --- ASR started --- (Path: {file_path})")

    start_time = time.time()
    pg_conn = None
    old_status = None

    try:
        with get_pg_connection() as conn:
            pg_conn = conn
            old_status = get_call_status(linkedid, conn=pg_conn)

            # 1. Check if transcript already exists
            if check_transcript_exists(linkedid, conn=pg_conn):
                logger.info(f"[{linkedid}] Transcript already exists. Skipping.")
                return True

            # 2. Metadata from MySQL
            try:
                metadata = fetch_call_metadata(linkedid)
            except Exception as e:
                logger.error(f"[{linkedid}] Failed to fetch metadata: {e}")
                return False

            # 3. Upsert call record (this sets status to 'processing')
            upsert_call(metadata, file_path, conn=pg_conn)

            # 4. Transcribe
            if not os.path.exists(file_path):
                logger.error(f"[{linkedid}] Audio file not found")
                return False

            waveform, sr = torchaudio.load(file_path)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
                sr = 16000

            if waveform.shape[0] != 2:
                left_waveform = waveform[0]
                right_waveform = torch.zeros_like(left_waveform)
            else:
                left_waveform = waveform[0]
                right_waveform = waveform[1]

            # Очистка шума
            logger.info(f"[{linkedid}] Denoising...")
            left_waveform = denoise_waveform(left_waveform)
            right_waveform = denoise_waveform(right_waveform)

            # RMS-нормализация
            logger.info(f"[{linkedid}] Normalizing (RMS)...")
            left_waveform, _ = normalize_waveform_rms(left_waveform)
            right_waveform, _ = normalize_waveform_rms(right_waveform)

            if metadata['direction'] == 'incoming':
                combined_segments = transcribe_with_vad(left_waveform, right_waveform, sr)
            else:
                combined_segments = transcribe_with_vad(right_waveform, left_waveform, sr)

            if not combined_segments:
                logger.info(f"[{linkedid}] No speech detected.")
                set_call_status(linkedid, 'empty', conn=pg_conn)
                duration = time.time() - start_time
                from db_utils import insert_processing_stats
                insert_processing_stats(linkedid, duration, 0, duration, conn=pg_conn)
                return True

            for start, end, channel, text, dict_score, speed in combined_segments:
                insert_transcript(linkedid, channel, start, end, text, diction=dict_score, wpm=speed, conn=pg_conn)

            duration = time.time() - start_time
            from db_utils import insert_processing_stats
            insert_processing_stats(linkedid, duration, 0, duration, conn=pg_conn)
            set_call_status(linkedid, 'transcribed', conn=pg_conn)
            logger.info(f"[{linkedid}] ASR finished in {duration:.2f}s")
            return True

    except Exception as e:
        logger.exception(f"[{linkedid}] ASR failed: {e}")
        if pg_conn:
            try:
                new_status = 'stop' if old_status == 'error' else 'error'
                set_call_status(linkedid, new_status, conn=pg_conn)
            except:
                pass
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python asr_worker.py <path_to_mp3>")
        sys.exit(1)
    process_asr(sys.argv[1])
