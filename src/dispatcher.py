import os
import time
import logging
import concurrent.futures
from config import RECORDS_ROOT, NUM_WORKERS
from db_utils import get_pg_connection
from worker import process_file  # функция-воркер

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def scan_files():
    with get_pg_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT linkedid FROM calls WHERE processing_status IN ('done','processing')")
        processed = set(row[0] for row in cur.fetchall())

    files_to_process = []
    for root, dirs, files in os.walk(RECORDS_ROOT):
        for f in files:
            if f.lower().endswith('.mp3'):
                linkedid = os.path.splitext(f)[0]
                if linkedid not in processed:
                    files_to_process.append(os.path.join(root, f))
    return files_to_process

def main():
    # Инициализация глобальных моделей при старте (можно принудительно загрузить)
    from models import get_whisper, get_emotion_model, get_llm
    get_whisper()
    get_emotion_model()
    get_llm()
    logging.info("Models loaded. Starting dispatcher with %d workers.", NUM_WORKERS)

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        while True:
            files = scan_files()
            if files:
                logging.info(f"Found {len(files)} new files. Submitting...")
                futures = [executor.submit(process_file, f) for f in files]
                # Ожидаем завершения перед следующим сканированием, чтобы не накапливать задачи
                concurrent.futures.wait(futures)
            else:
                logging.info("No new files. Sleeping 60 seconds.")
                time.sleep(60)

if __name__ == "__main__":
    main()
