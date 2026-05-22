import os
import logging
import tempfile
import librosa
from pydub import AudioSegment
from db_utils import (
    fetch_call_metadata, upsert_call, set_call_done, set_call_error,
    insert_transcript, insert_emotion, insert_evaluation
)
from asr import transcribe_audio
from emotion import predict_emotion
from llm_analysis import analyze_transcript

def process_file(file_path: str):
    base = os.path.basename(file_path)
    linkedid = os.path.splitext(base)[0]
    logging.info(f"[{linkedid}] Starting processing")

    try:
        # 1. Метаданные из MySQL
        metadata = fetch_call_metadata(linkedid)
    except Exception as e:
        logging.error(f"[{linkedid}] MySQL error: {e}")
        set_call_error(linkedid)
        return

    # 2. Запись в calls
    upsert_call(metadata, file_path)

    try:
        # 3. Разделение каналов
        audio = AudioSegment.from_file(file_path, format="mp3")
        if audio.channels != 2:
            raise ValueError("Audio must be stereo")
        left_ch = audio.split_to_mono()[0]
        right_ch = audio.split_to_mono()[1]

        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = os.path.join(tmpdir, "left.wav")
            right_path = os.path.join(tmpdir, "right.wav")
            left_ch.export(left_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])
            right_ch.export(right_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])

            # 4. Транскрибация
            left_segments = transcribe_audio(left_path)
            right_segments = transcribe_audio(right_path)

            # 5. Эмоции + сохранение
            def process_segments(segments, channel, audio_path, call_linkedid):
                transcript_texts = []
                y, sr = librosa.load(audio_path, sr=16000)
                for start, end, text in segments:
                    start_samp = int(start * sr)
                    end_samp = int(end * sr)
                    chunk = y[start_samp:end_samp]
                    emotion, conf = predict_emotion(chunk, sr)
                    tid = insert_transcript(call_linkedid, channel, start, end, text)
                    insert_emotion(tid, emotion, conf)
                    transcript_texts.append(f"{channel}: [{start:.2f}-{end:.2f}] {text} (эмоция: {emotion})")
                return transcript_texts

            op_texts = process_segments(left_segments, "operator", left_path, linkedid)
            cl_texts = process_segments(right_segments, "client", right_path, linkedid)

            # 6. LLM анализ
            full_dialogue = "\n".join(op_texts + cl_texts)
            eval_result = analyze_transcript(full_dialogue)
            insert_evaluation(linkedid, eval_result)

            set_call_done(linkedid)
            logging.info(f"[{linkedid}] Success")

    except Exception as e:
        logging.exception(f"[{linkedid}] Failed: {e}")
        set_call_error(linkedid)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python worker.py <path_to_mp3>")
        sys.exit(1)
    process_file(sys.argv[1])
