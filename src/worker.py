import os
import logging
import numpy as np
import librosa
import time
from db_utils import (
    fetch_call_metadata, upsert_call, set_call_done, set_call_error,
    insert_transcript, insert_emotion, insert_evaluation, get_pg_connection,
    get_default_prompt, get_prompt_by_id, check_transcript_exists, check_evaluation_exists
)
from asr import transcribe_audio
from emotion import predict_emotion
from llm_analysis import analyze_transcript
import tempfile
import soundfile as sf

logger = logging.getLogger(__name__)

def process_file(file_path: str, prompt_id: int = None):
    base = os.path.basename(file_path)
    linkedid = os.path.splitext(base)[0]
    logger.info(f"[{linkedid}] --- Processing started ---")

    start_total = time.time()
    pg_conn = None

    try:
        # 1. Получаем промпт
        with get_pg_connection() as conn:
            pg_conn = conn
            if prompt_id:
                prompt_data = get_prompt_by_id(prompt_id, conn=pg_conn)
            else:
                prompt_data = get_default_prompt(conn=pg_conn)

            if not prompt_data:
                logger.error(f"[{linkedid}] Prompt not found (id={prompt_id})")
                return

            current_prompt_id = prompt_data['id']
            current_prompt_text = prompt_data['prompt_text']

            # 2. Проверяем, есть ли уже результат для этого промпта
            if check_evaluation_exists(linkedid, current_prompt_id, conn=pg_conn):
                logger.info(f"[{linkedid}] Evaluation for prompt_id={current_prompt_id} already exists. Skipping.")
                return

            # 3. Метаданные из MySQL
            logger.debug(f"[{linkedid}] Fetching metadata from MySQL")
            metadata = fetch_call_metadata(linkedid)

            # 4. Запись в calls
            logger.debug(f"[{linkedid}] Upserting call record to PostgreSQL")
            upsert_call(metadata, file_path, conn=pg_conn)

            # 5. Проверяем наличие транскрипции
            transcript_exists = check_transcript_exists(linkedid, conn=pg_conn)
            full_dialogue = ""

            if not transcript_exists:
                # 6. Загрузка и разделение каналов
                logger.info(f"[{linkedid}] Loading audio and splitting channels")
                t0 = time.time()
                y, sr = librosa.load(file_path, sr=16000, mono=False)

                if y.ndim != 2 or y.shape[0] != 2:
                    logger.warning(f"[{linkedid}] Audio is not stereo, shape: {y.shape}. Processing as mono.")
                    left_y = y if y.ndim == 1 else y[0]
                    right_y = np.zeros_like(left_y)
                else:
                    left_y = y[0]
                    right_y = y[1]
                logger.debug(f"[{linkedid}] Audio loaded in {time.time()-t0:.2f}s")

                with tempfile.TemporaryDirectory() as tmpdir:
                    left_path = os.path.join(tmpdir, "left.wav")
                    right_path = os.path.join(tmpdir, "right.wav")

                    sf.write(left_path, left_y, sr)
                    sf.write(right_path, right_y, sr)

                    # 7. Транскрибация
                    logger.info(f"[{linkedid}] Transcribing operator channel")
                    left_segments = transcribe_audio(left_path)
                    logger.info(f"[{linkedid}] Transcribing client channel")
                    right_segments = transcribe_audio(right_path)

                    # 8. Эмоции + сохранение
                    def process_segments(segments, channel, audio_data, sample_rate, call_linkedid, db_conn):
                        logger.info(f"[{call_linkedid}] Analyzing emotions for {channel} ({len(segments)} segments)")
                        transcript_texts = []
                        for start, end, text in segments:
                            start_samp = int(start * sample_rate)
                            end_samp = int(end * sample_rate)
                            chunk = audio_data[start_samp:end_samp]

                            if len(chunk) == 0:
                                continue

                            emotion, conf = predict_emotion(chunk, sample_rate)
                            tid = insert_transcript(call_linkedid, channel, start, end, text, conn=db_conn)
                            insert_emotion(tid, emotion, conf, conn=db_conn)
                            transcript_texts.append(f"{channel}: [{start:.2f}-{end:.2f}] {text} (эмоция: {emotion})")
                        return transcript_texts

                    process_segments(left_segments, "operator", left_y, sr, linkedid, pg_conn)
                    process_segments(right_segments, "client", right_y, sr, linkedid, pg_conn)

                    all_segments = []
                    for start, end, text in left_segments:
                        all_segments.append((start, f"Operator: {text}"))
                    for start, end, text in right_segments:
                        all_segments.append((start, f"Client: {text}"))

                    all_segments.sort(key=lambda x: x[0])
                    full_dialogue = "\n".join([s[1] for s in all_segments])
            else:
                logger.info(f"[{linkedid}] Transcript already exists. Loading from DB.")
                cur = pg_conn.cursor()
                cur.execute("SELECT channel, text, start_time FROM transcripts WHERE linkedid = %s ORDER BY start_time", (linkedid,))
                rows = cur.fetchall()
                full_dialogue = "\n".join([f"{r[0].capitalize()}: {r[1]}" for r in rows])

            if full_dialogue.strip():
                logger.info(f"[{linkedid}] Starting LLM analysis with prompt_id={current_prompt_id}")
                eval_result = analyze_transcript(full_dialogue, prompt_template=current_prompt_text)
                insert_evaluation(linkedid, current_prompt_id, eval_result, conn=pg_conn)
            else:
                logger.warning(f"[{linkedid}] Empty transcript, skipping LLM analysis")

            set_call_done(linkedid, conn=pg_conn)
            duration_total = time.time() - start_total
            logger.info(f"[{linkedid}] --- Success! Total time: {duration_total:.2f}s ---")

    except Exception as e:
        logger.exception(f"[{linkedid}] Failed during processing: {e}")
        try:
            set_call_error(linkedid)
        except:
            pass

if __name__ == "__main__":
    import sys
    # Настройка логирования для прямого запуска воркера
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
    if len(sys.argv) != 2:
        print("Usage: python worker.py <path_to_mp3>")
        sys.exit(1)
    process_file(sys.argv[1])
