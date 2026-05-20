import os
import time
import logging
import concurrent.futures
from config import RECORDS_ROOT, NUM_WORKERS
from db_utils import get_pg_connection
from worker import process_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def scan_files():
    """Возвращает список полных путей к mp3-файлам, отсутствующим в базе"""
    with get_pg_connection() as conn:
        cur = conn.cursor()
        # Получаем все уже обработанные или обрабатываемые linkedid
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
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        while True:
            files = scan_files()
            if files:
                logging.info(f"Found {len(files)} new files. Dispatching...")
                futures = [executor.submit(process_file, f) for f in files]
                # Опционально: дожидаться выполнения перед следующей итерацией
                concurrent.futures.wait(futures)
            else:
                logging.info("No new files. Sleeping 60s.")
                time.sleep(60)

if __name__ == "__main__":
    main()