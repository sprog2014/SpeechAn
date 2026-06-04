import os
import logging
import torch
import torchaudio
import time
from db_utils import (
    fetch_call_metadata, upsert_call, insert_transcript, get_pg_connection,
    check_transcript_exists, set_call_status
)
from asr import transcribe_with_vad
from logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def normalize_waveform(waveform: torch.Tensor, target_peak: float = 0.9, max_gain: float = 10.0):
    peak = torch.max(torch.abs(waveform)).item()
    if peak > 1e-5:
        gain = target_peak / peak
        if gain > max_gain:
            gain = max_gain
        return waveform * gain, gain
    return waveform, 1.0

def process_asr(file_path: str):
    base = os.path.basename(file_path)
    linkedid = os.path.splitext(base)[0]
    logger.info(f"[{linkedid}] --- ASR started --- (Path: {file_path})")

    start_time = time.time()
    pg_conn = None

    try:
        with get_pg_connection() as conn:
            pg_conn = conn

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

            left_waveform, _ = normalize_waveform(left_waveform)
            right_waveform, _ = normalize_waveform(right_waveform)

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

            for start, end, channel, text in combined_segments:
                insert_transcript(linkedid, channel, start, end, text, conn=pg_conn)

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
                set_call_status(linkedid, 'error', conn=pg_conn)
            except:
                pass
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python asr_worker.py <path_to_mp3>")
        sys.exit(1)
    process_asr(sys.argv[1])
