import os
import logging
import numpy as np
import librosa
from db_utils import (
    fetch_call_metadata, upsert_call, set_call_done, set_call_error,
    insert_transcript, insert_emotion, insert_evaluation, get_pg_connection
)
from asr import transcribe_audio
from emotion import predict_emotion
from llm_analysis import analyze_transcript
import tempfile
import soundfile as sf

def process_file(file_path: str):
    base = os.path.basename(file_path)
    linkedid = os.path.splitext(base)[0]
    logging.info(f"[{linkedid}] Starting processing")

    # Переменная для хранения соединения, чтобы переиспользовать его внутри воркера
    pg_conn = None

    try:
        # 1. Метаданные из MySQL
        metadata = fetch_call_metadata(linkedid)

        # Получаем соединение из пула для всех последующих операций в этом воркере
        with get_pg_connection() as conn:
            pg_conn = conn

            # 2. Запись в calls
            upsert_call(metadata, file_path, conn=pg_conn)

            # 3. Загрузка и разделение каналов
            y, sr = librosa.load(file_path, sr=16000, mono=False)

            if y.ndim != 2 or y.shape[0] != 2:
                # Если запись моно, попробуем обработать как один канал, но по логике должно быть стерео
                logging.warning(f"[{linkedid}] Audio is not stereo, shape: {y.shape}. Processing as mono.")
                left_y = y if y.ndim == 1 else y[0]
                right_y = np.zeros_like(left_y) # Пустой правый канал
            else:
                left_y = y[0]
                right_y = y[1]

            with tempfile.TemporaryDirectory() as tmpdir:
                left_path = os.path.join(tmpdir, "left.wav")
                right_path = os.path.join(tmpdir, "right.wav")

                sf.write(left_path, left_y, sr)
                sf.write(right_path, right_y, sr)

                # 4. Транскрибация
                left_segments = transcribe_audio(left_path)
                right_segments = transcribe_audio(right_path)

                # 5. Эмоции + сохранение
                def process_segments(segments, channel, audio_data, sample_rate, call_linkedid, db_conn):
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

                # 6. LLM анализ
                all_segments = []
                for start, end, text in left_segments:
                    all_segments.append((start, f"Operator: {text}"))
                for start, end, text in right_segments:
                    all_segments.append((start, f"Client: {text}"))

                all_segments.sort(key=lambda x: x[0])
                full_dialogue = "\n".join([s[1] for s in all_segments])

                if full_dialogue.strip():
                    eval_result = analyze_transcript(full_dialogue)
                    insert_evaluation(linkedid, eval_result, conn=pg_conn)
                else:
                    logging.warning(f"[{linkedid}] Empty transcript, skipping LLM analysis")

                set_call_done(linkedid, conn=pg_conn)
                logging.info(f"[{linkedid}] Success")

    except Exception as e:
        logging.exception(f"[{linkedid}] Failed: {e}")
        try:
            set_call_error(linkedid)
        except:
            pass

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python worker.py <path_to_mp3>")
        sys.exit(1)
    process_file(sys.argv[1])
