import os
import time
import logging
import concurrent.futures
from config import RECORDS_ROOT, NUM_WORKERS
from db_utils import get_pg_connection
from worker import process_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Множество для отслеживания файлов, которые сейчас в обработке
processing_now = set()

def scan_files():
    try:
        with get_pg_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT linkedid FROM calls WHERE processing_status IN ('done','processing','error')")
            processed = set(row[0] for row in cur.fetchall())
    except Exception as e:
        logging.error(f"Error scanning DB: {e}")
        return []

    files_to_process = []
    for root, dirs, files in os.walk(RECORDS_ROOT):
        for f in files:
            if f.lower().endswith('.mp3'):
                linkedid = os.path.splitext(f)[0]
                if linkedid not in processed and linkedid not in processing_now:
                    files_to_process.append(os.path.join(root, f))
    return files_to_process

def task_done_callback(future):
    linkedid = future.linkedid
    processing_now.discard(linkedid)
    try:
        future.result()
        logging.info(f"[{linkedid}] Task completed")
    except Exception as e:
        logging.error(f"[{linkedid}] Task generated an exception: {e}")

def main():
    # Инициализация глобальных моделей при старте
    from models import get_whisper, get_emotion_model, get_llm
    logging.info("Pre-loading models...")
    get_whisper()
    get_emotion_model()
    get_llm()
    logging.info("Models loaded. Starting dispatcher with %d workers.", NUM_WORKERS)

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        while True:
            # Проверяем, есть ли свободные слоты в экзекуторе
            # (Косвенно через processing_now)
            if len(processing_now) < NUM_WORKERS:
                files = scan_files()
                if files:
                    logging.info(f"Found {len(files)} new files. Submitting up to {NUM_WORKERS - len(processing_now)} tasks.")
                    for f_path in files:
                        if len(processing_now) >= NUM_WORKERS:
                            break

                        linkedid = os.path.splitext(os.path.basename(f_path))[0]
                        processing_now.add(linkedid)

                        future = executor.submit(process_file, f_path)
                        future.linkedid = linkedid
                        future.add_done_callback(task_done_callback)
                else:
                    if not processing_now:
                        logging.info("No new files and no tasks running. Sleeping 30 seconds.")
                        time.sleep(30)
                    else:
                        time.sleep(5) # Ждем освобождения слотов
            else:
                # Пул заполнен, просто ждем немного
                time.sleep(5)

if __name__ == "__main__":
    main()
