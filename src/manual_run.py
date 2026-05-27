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
    parser.add_argument("--force", action="store_true", help="Force re-processing even if evaluation exists")
    parser.add_argument("--ignore-stop-flag", action="store_true", help="Ignore system stop flag in system_settings")

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

    # Check system running status
    is_running = get_system_running_status()
    if not is_running:
        if args.ignore_stop_flag:
            logger.warning("System is OSTOPPED in settings, but --ignore-stop-flag is set. Proceeding...")
        else:
            logger.error("System is STOPPED in settings. Manual run aborted. Use --ignore-stop-flag to bypass.")
            return

    # Pre-load models
    logger.info("Initializing models...")
    try:
        get_whisper()
        get_emotion_model()
        get_llm()
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        return

    logger.info(f"Starting processing with {args.workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for f_path in files:
            # Проверка флага остановки перед запуском очередного файла (если не игнорируем)
            if not args.ignore_stop_flag and not get_system_running_status():
                logger.info("System stop flag detected. Stopping manual run.")
                break

            linkedid = os.path.splitext(os.path.basename(f_path))[0]
            logger.info(f"[{linkedid}] Submitting manual task (force={args.force})")
            futures.append(executor.submit(process_file, f_path, prompt_id=args.prompt_id, force=args.force))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Task failed: {e}")

if __name__ == "__main__":
    main()
