import os
import logging
import torch
import torchaudio
import time
import numpy as np
import ctypes
from db_utils import (
    fetch_call_metadata, upsert_call, insert_transcript, get_pg_connection,
    check_transcript_exists, set_call_status, get_call_status
)
from asr import transcribe_with_vad
from logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class RNNoiseDirect:
    def __init__(self):
        # Пытаемся найти библиотеку в установленном пакете pyrnnoise
        try:
            # Не используем import pyrnnoise здесь, так как он может потянуть сломанный audiolab
            # Просто ищем путь к пакету
            import importlib.util
            spec = importlib.util.find_spec("pyrnnoise")
            if spec and spec.origin:
                lib_path = os.path.join(os.path.dirname(spec.origin), 'librnnoise.so')
                self.lib = ctypes.CDLL(lib_path)
            else:
                raise ImportError("pyrnnoise not found")
        except Exception as e:
            logger.error(f"Failed to load RNNoise library: {e}")
            self.lib = None
            return

        self.lib.rnnoise_create.restype = ctypes.c_void_p
        self.lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        self.lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        self.lib.rnnoise_process_frame.restype = ctypes.c_float

        self.st = self.lib.rnnoise_create(None)

    def __del__(self):
        if hasattr(self, 'lib') and self.lib and self.st:
            self.lib.rnnoise_destroy(self.st)

    def process(self, wav: np.ndarray):
        # wav: float32, 1D, 48000Hz
        if self.lib is None:
            return wav

        frame_size = 480
        num_frames = len(wav) // frame_size
        output = np.zeros_like(wav)

        for i in range(num_frames):
            frame = wav[i*frame_size : (i+1)*frame_size].astype(np.float32)
            in_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            out_ptr = output[i*frame_size : (i+1)*frame_size].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            self.lib.rnnoise_process_frame(self.st, out_ptr, in_ptr)

        return output

def denoise_waveform(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """
    Подавление шума с помощью RNNoise.
    Требует 48кГц, поэтому выполняем ресэмплинг внутри.
    """
    if waveform.shape[-1] == 0:
        return waveform

    # 1. Ресэмплинг в 48кГц
    target_sr = 48000
    resampler_to_48 = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
    wav_48 = resampler_to_48(waveform)

    # Если стерео, обрабатываем каналы по отдельности
    is_1d = False
    if wav_48.ndim == 1:
        wav_48 = wav_48.unsqueeze(0)
        is_1d = True

    num_channels = wav_48.shape[0]
    denoised_channels = []

    try:
        for c in range(num_channels):
            denoiser = RNNoiseDirect()
            channel_data = wav_48[c].numpy()
            denoised_channel = denoiser.process(channel_data)
            denoised_channels.append(torch.from_numpy(denoised_channel))

        denoised_wav_48 = torch.stack(denoised_channels)
    except Exception as e:
        logger.error(f"Error during RNNoise processing: {e}")
        return waveform

    # 2. Ресэмплинг обратно
    resampler_from_48 = torchaudio.transforms.Resample(orig_freq=target_sr, new_freq=sample_rate)
    output = resampler_from_48(denoised_wav_48)

    if is_1d:
        output = output.squeeze(0)
    return output

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
            left_waveform = denoise_waveform(left_waveform, sr)
            right_waveform = denoise_waveform(right_waveform, sr)

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
