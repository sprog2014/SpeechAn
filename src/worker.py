import os
import sys
import logging
from datetime import datetime
from pydub import AudioSegment
import tempfile
import librosa

from db_utils import (
    fetch_call_metadata, upsert_call, set_call_done, set_call_error,
    insert_transcript, insert_emotion, insert_evaluation
)
from asr import transcribe_audio
from emotion import predict_emotion
from llm_analysis import analyze_transcript

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def process_file(file_path):
    base = os.path.basename(file_path)
    linkedid = os.path.splitext(base)[0]
    logging.info(f"Processing {linkedid}")

    # 1. Получить метаданные из MySQL
    try:
        metadata = fetch_call_metadata(linkedid)
    except Exception as e:
        logging.error(f"Metadata error: {e}")
        set_call_error(linkedid)
        return

    # 2. Вставить / обновить запись calls
    upsert_call(metadata, file_path)

    try:
        # 3. Разделение каналов
        audio = AudioSegment.from_file(file_path, format="mp3")
        if audio.channels != 2:
            raise ValueError("Audio must be stereo (2 channels)")
        left_channel = audio.split_to_mono()[0]
        right_channel = audio.split_to_mono()[1]

        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = os.path.join(tmpdir, "left.wav")
            right_path = os.path.join(tmpdir, "right.wav")
            left_channel.export(left_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])
            right_channel.export(right_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])

            # 4. Транскрибация
            left_segments = transcribe_audio(left_path)
            right_segments = transcribe_audio(right_path)

            # 5. Обработка сегментов: эмоции + сохранение в БД
            def process_segments(segments, channel, audio_path, call_linkedid):
                transcript_texts = []
                y, sr = librosa.load(audio_path, sr=16000)
                for start, end, text in segments:
                    # Вырезаем аудио фрагмент
                    start_sample = int(start * sr)
                    end_sample = int(end * sr)
                    chunk = y[start_sample:end_sample]
                    emotion, conf = predict_emotion(chunk, sr)
                    # Сохраняем в БД
                    t_id = insert_transcript(call_linkedid, channel, start, end, text)
                    insert_emotion(t_id, emotion, conf)
                    transcript_texts.append(f"{channel}: [{start:.2f}-{end:.2f}] {text} (эмоция: {emotion})")
                return transcript_texts

            left_texts = process_segments(left_segments, "operator", left_path, linkedid)
            right_texts = process_segments(right_segments, "client", right_path, linkedid)

            # 6. Сборка полного диалога для LLM
            full_dialogue = "\n".join(left_texts + right_texts)  # лучше отсортировать по времени, но для простоты так

            # 7. LLM анализ
            eval_result = analyze_transcript(full_dialogue)
            insert_evaluation(linkedid, eval_result)

            set_call_done(linkedid)
            logging.info(f"Success {linkedid}")

    except Exception as e:
        logging.exception(f"Processing failed for {linkedid}")
        set_call_error(linkedid)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python worker.py <path_to_mp3>")
        sys.exit(1)
    process_file(sys.argv[1])