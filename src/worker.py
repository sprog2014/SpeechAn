import os
import logging
import numpy as np
import librosa
import time
import concurrent.futures
import torch
import torchaudio
from db_utils import (
    fetch_call_metadata, upsert_call, set_call_done, set_call_error,
    insert_transcript, insert_evaluation, get_pg_connection,
    get_default_prompt, get_prompt_by_id, check_transcript_exists, check_evaluation_exists,
    set_processing_duration, check_phone_usage, get_system_setting, is_phone_registered
)
from asr import transcribe_audio
from emotion import predict_emotions_full
from llm_analysis import analyze_transcript

logger = logging.getLogger(__name__)

def process_channel_asr(channel_name, waveform, sample_rate, linkedid):
    """Функция для транскрибации одного канала (в памяти)"""
    logger.info(f"[{linkedid}] Starting ASR for channel: {channel_name}")
    try:
        segments = transcribe_audio(waveform, sample_rate)

        results = []
        with get_pg_connection() as conn:
            for start, end, text in segments:
                insert_transcript(linkedid, channel_name, start, end, text, conn=conn)
                results.append((start, f"{channel_name.capitalize()}: {text}"))
        return results
    except Exception as e:
        logger.error(f"[{linkedid}] Error in ASR for channel {channel_name}: {e}")
        return []

def process_file(file_path: str, prompt_id: int = None, force: bool = False):
    base = os.path.basename(file_path)
    linkedid = os.path.splitext(base)[0]
    logger.info(f"[{linkedid}] --- Processing started --- (Path: {file_path})")

    start_total = time.time()
    pg_conn = None

    try:
        # 1. Получаем промпт
        with get_pg_connection() as conn:
            pg_conn = conn
            if prompt_id:
                logger.debug(f"[{linkedid}] Fetching prompt by ID: {prompt_id}")
                prompt_data = get_prompt_by_id(prompt_id, conn=pg_conn)
            else:
                logger.debug(f"[{linkedid}] Fetching default prompt")
                prompt_data = get_default_prompt(conn=pg_conn)

            if not prompt_data:
                logger.error(f"[{linkedid}] ABORT: Prompt not found (id={prompt_id})")
                return

            current_prompt_id = prompt_data['id']
            current_prompt_text = prompt_data['prompt_text']
            logger.info(f"[{linkedid}] Using prompt ID: {current_prompt_id}")

            # 2. Проверяем, есть ли уже результат для этого промпта
            if not force and check_evaluation_exists(linkedid, current_prompt_id, conn=pg_conn):
                logger.info(f"[{linkedid}] SKIP: Evaluation for prompt_id={current_prompt_id} already exists. Use --force to overwrite.")
                return

            # 3. Метаданные из MySQL
            logger.debug(f"[{linkedid}] Fetching metadata from MySQL")
            try:
                metadata = fetch_call_metadata(linkedid)
            except Exception as e:
                logger.error(f"[{linkedid}] ABORT: Failed to fetch metadata from MySQL: {e}")
                set_call_error(linkedid, conn=pg_conn)
                return

            # 3.5 Проверка фильтров по номеру телефона
            src_num = metadata.get('src')
            dst_num = metadata.get('answeredext')

            if get_system_setting('skip_local_calls', 'false', conn=pg_conn).lower() == 'true':
                if is_phone_registered(src_num, conn=pg_conn) and is_phone_registered(dst_num, conn=pg_conn):
                    logger.info(f"[{linkedid}] SKIP: Local call between {src_num} and {dst_num} skipped.")
                    return

            allowed_src = check_phone_usage(src_num, conn=pg_conn)
            allowed_dst = check_phone_usage(dst_num, conn=pg_conn)

            if not allowed_src and not allowed_dst:
                logger.info(f"[{linkedid}] SKIP: Numbers {src_num} and {dst_num} are not enabled for analysis.")
                return

            # 4. Запись в calls
            logger.debug(f"[{linkedid}] Upserting call record to PostgreSQL")
            upsert_call(metadata, file_path, conn=pg_conn)

            # 5. Загрузка аудио в память
            if not os.path.exists(file_path):
                logger.error(f"[{linkedid}] ABORT: Audio file not found at {file_path}")
                set_call_error(linkedid, conn=pg_conn)
                return

            logger.info(f"[{linkedid}] Loading audio and splitting channels")
            t0 = time.time()
            try:
                # Используем torchaudio для загрузки сразу в тензоры
                waveform, sr = torchaudio.load(file_path)
            except Exception as e:
                logger.error(f"[{linkedid}] ABORT: torchaudio failed to load file: {e}")
                set_call_error(linkedid, conn=pg_conn)
                return

            if waveform.shape[0] != 2:
                logger.warning(f"[{linkedid}] Audio is not stereo, shape: {waveform.shape}. Using as mixed.")
                left_waveform = waveform
                right_waveform = torch.zeros_like(left_waveform)
                waveform_mixed = waveform[0]
            else:
                left_waveform = waveform[0:1] # [1, samples]
                right_waveform = waveform[1:2]
                waveform_mixed = torch.mean(waveform, dim=0) # [samples]

            logger.debug(f"[{linkedid}] Audio loaded in {time.time()-t0:.2f}s")

            # 6. Анализ эмоций (в памяти)
            logger.info(f"[{linkedid}] Analyzing full call emotions...")
            speech_emotions = predict_emotions_full(waveform_mixed.numpy(), sr)

            # 7. Транскрибация (в памяти)
            transcript_exists = check_transcript_exists(linkedid, conn=pg_conn)
            full_dialogue = ""

            if not transcript_exists:
                # Транскрибируем каналы параллельно в памяти
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_asr_left = executor.submit(process_channel_asr, "operator", left_waveform, sr, linkedid)
                    future_asr_right = executor.submit(process_channel_asr, "client", right_waveform, sr, linkedid)

                    left_results = future_asr_left.result()
                    right_results = future_asr_right.result()

                all_segments = left_results + right_results
                all_segments.sort(key=lambda x: x[0])
                full_dialogue = "\n".join([s[1] for s in all_segments])
            else:
                logger.info(f"[{linkedid}] Transcript already exists in DB. Skipping ASR.")
                cur = pg_conn.cursor()
                cur.execute("SELECT channel, text, start_time FROM transcripts WHERE linkedid = %s ORDER BY start_time", (linkedid,))
                rows = cur.fetchall()
                full_dialogue = "\n".join([f"{r[0].capitalize()}: {r[1]}" for r in rows])

            # 8. LLM анализ
            if full_dialogue.strip():
                logger.info(f"[{linkedid}] Starting LLM analysis")
                eval_result = analyze_transcript(full_dialogue, prompt_template=current_prompt_text)
                insert_evaluation(linkedid, current_prompt_id, eval_result, speech_emotions=speech_emotions, conn=pg_conn)
                logger.info(f"[{linkedid}] LLM analysis completed and saved")
            else:
                logger.warning(f"[{linkedid}] Empty transcript, skipping LLM analysis")

            set_call_done(linkedid, conn=pg_conn)
            duration_total = time.time() - start_total
            set_processing_duration(linkedid, duration_total, conn=pg_conn)
            logger.info(f"[{linkedid}] --- SUCCESS! Total time: {duration_total:.2f}s ---")

    except Exception as e:
        logger.exception(f"[{linkedid}] CRITICAL: Failed during processing: {e}")
        try:
            if pg_conn:
                set_call_error(linkedid, conn=pg_conn)
        except:
            pass

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
    if len(sys.argv) != 2:
        print("Usage: python worker.py <path_to_mp3>")
        sys.exit(1)
    process_file(sys.argv[1])
