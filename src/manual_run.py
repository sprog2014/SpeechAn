import os
import argparse
import logging
import concurrent.futures
from datetime import datetime, timedelta
from config import RECORDS_ROOT, NUM_WORKERS
from worker import process_file
from models import get_whisper, get_emotion_model, get_llm
from db_utils import get_system_running_status

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
)
logger = logging.getLogger("manual_run")

def get_date_range(start_str, end_str=None):
    start_date = datetime.strptime(start_str, "%Y-%m-%d")
    if end_str:
        end_date = datetime.strptime(end_str, "%Y-%m-%d")
    else:
        end_date = start_date

    date_list = []
    curr = start_date
    while curr <= end_date:
        date_list.append(curr.strftime("%Y/%m/%d"))
        curr += timedelta(days=1)
    return date_list

def scan_files_in_dates(date_paths):
    files_to_process = []
    for date_path in date_paths:
        full_path = os.path.join(RECORDS_ROOT, date_path)
        if os.path.exists(full_path):
            logger.info(f"Scanning directory: {full_path}")
            for root, dirs, files in os.walk(full_path):
                for f in files:
                    if f.lower().endswith('.mp3'):
                        files_to_process.append(os.path.join(root, f))
        else:
            logger.warning(f"Path does not exist: {full_path}")
    return files_to_process

def main():
    parser = argparse.ArgumentParser(description="Manual Call Analysis Processing")
    parser.add_argument("dates", nargs='+', help="Date (YYYY-MM-DD) or range (Start End)")
    parser.add_argument("--prompt_id", type=int, help="Optional prompt ID to use for analysis")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of parallel workers")

    args = parser.parse_args()

    if len(args.dates) == 1:
        date_paths = get_date_range(args.dates[0])
    elif len(args.dates) >= 2:
        date_paths = get_date_range(args.dates[0], args.dates[1])
    else:
        parser.print_help()
        return

    logger.info(f"Target date paths: {date_paths}")
    files = scan_files_in_dates(date_paths)
    logger.info(f"Found {len(files)} files to process.")

    if not files:
        logger.info("Nothing to do.")
        return

    # Pre-load models
    logger.info("Initializing models...")
    get_whisper()
    get_emotion_model()
    get_llm()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for f_path in files:
            # Проверка флага остановки перед запуском очередного файла
            if not get_system_running_status():
                logger.info("System stop flag detected. Stopping manual run.")
                break

            linkedid = os.path.splitext(os.path.basename(f_path))[0]
            logger.info(f"[{linkedid}] Submitting manual task")
            futures.append(executor.submit(process_file, f_path, prompt_id=args.prompt_id))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Task failed: {e}")

if __name__ == "__main__":
    main()
