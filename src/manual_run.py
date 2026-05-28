import os
import argparse
import logging
import concurrent.futures
from datetime import datetime, timedelta
from config import RECORDS_ROOT, NUM_WORKERS
from worker import process_file
from models import get_asr_model, get_emotion_model, get_llm
from db_utils import get_system_running_status

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
)
logger = logging.getLogger("manual_run")

def get_date_range(start_str, end_str=None):
    try:
        start_date = datetime.strptime(start_str, "%Y-%m-%d")
        if end_str:
            end_date = datetime.strptime(end_str, "%Y-%m-%d")
        else:
            end_date = start_date
    except ValueError as e:
        logger.error(f"Invalid date format. Expected YYYY-MM-DD. Error: {e}")
        raise

    date_list = []
    curr = start_date
    while curr <= end_date:
        date_list.append(curr.strftime("%Y/%m/%d"))
        curr += timedelta(days=1)
    return date_list

def scan_files_in_dates(date_paths):
    files_to_process = []
    logger.info(f"Records root directory: {RECORDS_ROOT}")

    if not os.path.exists(RECORDS_ROOT):
        logger.error(f"RECORDS_ROOT does not exist: {RECORDS_ROOT}")
        return files_to_process

    for date_path in date_paths:
        full_path = os.path.join(RECORDS_ROOT, date_path)
        logger.info(f"Checking path: {full_path}")
        if os.path.exists(full_path):
            found_in_dir = 0
            for root, dirs, files in os.walk(full_path):
                for f in files:
                    if f.lower().endswith('.mp3'):
                        abs_path = os.path.join(root, f)
                        files_to_process.append(abs_path)
                        found_in_dir += 1
            logger.info(f"Found {found_in_dir} .mp3 files in {full_path}")
        else:
            logger.warning(f"Path DOES NOT EXIST: {full_path}")
    return files_to_process

def main():
    parser = argparse.ArgumentParser(description="Manual Call Analysis Processing")
    parser.add_argument("dates", nargs='+', help="Date (YYYY-MM-DD) or range (Start End)")
    parser.add_argument("--prompt_id", type=int, help="Optional prompt ID to use for analysis")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", help="Force re-processing even if evaluation exists")
    parser.add_argument("--ignore-stop-flag", action="store_true", help="Ignore system stop flag in system_settings")

    args = parser.parse_args()

    logger.info("=== Starting manual_run process ===")

    try:
        if len(args.dates) == 1:
            date_paths = get_date_range(args.dates[0])
        elif len(args.dates) >= 2:
            date_paths = get_date_range(args.dates[0], args.dates[1])
        else:
            parser.print_help()
            return
    except Exception:
        return

    logger.info(f"Target date paths: {date_paths}")
    files = scan_files_in_dates(date_paths)
    logger.info(f"Total files found across all dates: {len(files)}")

    if not files:
        logger.info("No files found to process. Exiting.")
        return

    # Check system running status
    logger.info("Checking system operational status...")
    try:
        is_running = get_system_running_status()
        logger.info(f"System 'is_running' status from DB: {is_running}")
    except Exception as e:
        logger.error(f"Could not fetch system status from DB: {e}")
        is_running = False

    if not is_running:
        if args.ignore_stop_flag:
            logger.warning("System is marked as STOPPED in database, but --ignore-stop-flag is present. Proceeding...")
        else:
            logger.error("System is marked as STOPPED in database. Manual run will not start tasks. Use --ignore-stop-flag to bypass.")
            return

    # Pre-load models
    logger.info("Starting model initialization...")
    try:
        t_load_start = datetime.now()
        get_asr_model()
        get_emotion_model()
        get_llm()
        t_load_end = datetime.now()
        logger.info(f"Models initialized successfully in {t_load_end - t_load_start}")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize models: {e}")
        return

    logger.info(f"Submitting {len(files)} tasks to ThreadPoolExecutor with {args.workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for f_path in files:
            # Check flag periodically
            if not args.ignore_stop_flag:
                try:
                    if not get_system_running_status():
                        logger.warning("System stop flag detected in DB during task submission. Aborting further submissions.")
                        break
                except:
                    pass

            linkedid = os.path.splitext(os.path.basename(f_path))[0]
            logger.info(f"[{linkedid}] Submitting for processing (Path: {f_path}, Force: {args.force})")
            futures.append(executor.submit(process_file, f_path, prompt_id=args.prompt_id, force=args.force))

        logger.info(f"All {len(futures)} tasks submitted. Waiting for completion...")

        done_count = 0
        error_count = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
                done_count += 1
            except Exception as e:
                logger.error(f"A task failed with an unhandled exception: {e}")
                error_count += 1

            if (done_count + error_count) % 5 == 0 or (done_count + error_count) == len(futures):
                logger.info(f"Progress: {done_count + error_count}/{len(futures)} completed ({done_count} success, {error_count} failed)")

    logger.info("=== manual_run process finished ===")

if __name__ == "__main__":
    main()
