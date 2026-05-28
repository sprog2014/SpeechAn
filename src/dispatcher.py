import os
import time
import logging
import concurrent.futures
from datetime import datetime, timedelta
from config import RECORDS_ROOT, NUM_WORKERS
from db_utils import get_pg_connection, get_system_running_status
from worker import process_file

# Настройка логирования диспетчера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
)
logger = logging.getLogger("dispatcher")

# Множество для отслеживания файлов, которые сейчас в обработке
processing_now = set()

def scan_files():
    logger.debug("Scanning database for processed files...")
    try:
        with get_pg_connection() as conn:
            # Получаем ID дефолтного промпта
            cur = conn.cursor()
            cur.execute("SELECT id FROM prompts WHERE is_default = TRUE LIMIT 1")
            row = cur.fetchone()
            if not row:
                logger.error("No default prompt found in DB")
                return []
            default_prompt_id = row[0]

            # Ищем те, что уже имеют оценку с этим дефолтным промптом
            # ИЛИ уже были обработаны/в процессе/с ошибкой (согласно статусу в calls)
            # ИЛИ пропущены
            cur.execute("""
                SELECT linkedid FROM evaluations WHERE prompt_id = %s
                UNION
                SELECT linkedid FROM calls WHERE processing_status IN ('processing', 'done', 'error', 'skipped')
            """, (default_prompt_id,))
            processed = set(row[0] for row in cur.fetchall())
    except Exception as e:
        logger.error(f"Error scanning DB: {e}")
        return []

    logger.debug(f"Scanning directory: {RECORDS_ROOT}")
    files_to_process = []

    # Ограничиваем сканирование двумя последними днями
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    target_days = [today, yesterday]

    current_time = time.time()

    for day in target_days:
        day_path = os.path.join(RECORDS_ROOT, day.strftime("%Y/%m/%d"))
        if not os.path.exists(day_path):
            continue

        for root, dirs, files in os.walk(day_path):
            for f in files:
                if f.lower().endswith('.mp3'):
                    f_path = os.path.join(root, f)
                    try:
                        # Проверяем время последнего изменения файла
                        mtime = os.path.getmtime(f_path)
                        # Если файл изменен менее минуты назад, пропускаем его (ждем завершения записи/копирования)
                        if current_time - mtime < 60:
                            logger.debug(f"File {f} is too new, skipping for now.")
                            continue

                        linkedid = os.path.splitext(f)[0]
                        if linkedid not in processed and linkedid not in processing_now:
                            files_to_process.append(f_path)
                    except Exception as e:
                        logger.error(f"Error checking file {f_path}: {e}")

    return files_to_process

def task_done_callback(future):
    linkedid = future.linkedid
    processing_now.discard(linkedid)
    try:
        future.result()
        logger.info(f"[{linkedid}] Task execution finished")
    except Exception as e:
        logger.error(f"[{linkedid}] Task generated an exception: {e}")

def main():
    # Инициализация глобальных моделей при старте
    from models import get_whisper, get_emotion_model, get_llm
    logger.info("Starting system initialization...")
    logger.info("Pre-loading models into memory...")
    get_whisper()
    get_emotion_model()
    get_llm()
    logger.info("All models loaded. Starting main loop.")
    logger.info(f"Configuration: NUM_WORKERS={NUM_WORKERS}, RECORDS_ROOT={RECORDS_ROOT}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        while True:
            # Проверка статуса запуска системы
            if not get_system_running_status():
                logger.info("System is in WAITING mode (is_running=false). Sleeping 10s.")
                time.sleep(10)
                continue

            current_active = len(processing_now)
            if current_active < NUM_WORKERS:
                files = scan_files()
                if files:
                    to_submit = NUM_WORKERS - current_active
                    logger.info(f"Found {len(files)} new files. Submitting up to {to_submit} tasks. (Active: {current_active})")
                    for f_path in files:
                        if len(processing_now) >= NUM_WORKERS:
                            break

                        linkedid = os.path.splitext(os.path.basename(f_path))[0]
                        processing_now.add(linkedid)

                        logger.info(f"[{linkedid}] Submitting task for file: {f_path}")
                        future = executor.submit(process_file, f_path)
                        future.linkedid = linkedid
                        future.add_done_callback(task_done_callback)
                else:
                    if not processing_now:
                        logger.info("No new files to process and no active tasks. Sleeping 10s.")
                        time.sleep(10)
                    else:
                        logger.debug("No new files found. Waiting for active tasks...")
                        time.sleep(5)
            else:
                logger.debug(f"Worker pool is full ({current_active} active). Waiting...")
                time.sleep(10)

if __name__ == "__main__":
    main()
