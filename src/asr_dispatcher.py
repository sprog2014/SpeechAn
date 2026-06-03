import os
import time
import logging
import multiprocessing
import concurrent.futures
from datetime import datetime, timedelta
from config import RECORDS_ROOT, NUM_ASR_WORKERS
from db_utils import get_pg_connection, get_system_running_status, get_active_tasks, update_task_status
from asr_worker import process_asr
from logging_utils import setup_logging

setup_logging()
logger = logging.getLogger("asr_dispatcher")

processing_now = set()

def scan_current_files():
    try:
        with get_pg_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT linkedid FROM transcripts")
            processed = set(row[0] for row in cur.fetchall())

            cur.execute("SELECT linkedid FROM calls WHERE processing_status = 'error'")
            errors = set(row[0] for row in cur.fetchall())
    except Exception as e:
        logger.error(f"Error scanning DB: {e}")
        return []

    files_to_process = []
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    current_time = time.time()

    for day in [today, yesterday]:
        day_path = os.path.join(RECORDS_ROOT, day.strftime("%Y/%m/%d"))
        if not os.path.exists(day_path): continue
        for root, _, files in os.walk(day_path):
            for f in files:
                if f.lower().endswith('.mp3'):
                    f_path = os.path.join(root, f)
                    linkedid = os.path.splitext(f)[0]
                    if linkedid not in processed and linkedid not in processing_now and linkedid not in errors:
                        if current_time - os.path.getmtime(f_path) > 60:
                            files_to_process.append(f_path)
    return files_to_process

def get_task_files(task):
    start_date = task['start_date']
    end_date = task['end_date']
    files_to_process = []

    try:
        with get_pg_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT linkedid FROM transcripts")
            processed = set(row[0] for row in cur.fetchall())
    except Exception as e:
        logger.error(f"Error scanning DB for task: {e}")
        return []

    curr = start_date
    while curr <= end_date:
        day_path = os.path.join(RECORDS_ROOT, curr.strftime("%Y/%m/%d"))
        if os.path.exists(day_path):
            for root, _, files in os.walk(day_path):
                for f in files:
                    if f.lower().endswith('.mp3'):
                        linkedid = os.path.splitext(f)[0]
                        if linkedid not in processed and linkedid not in processing_now:
                            files_to_process.append(os.path.join(root, f))
        curr += timedelta(days=1)
    return files_to_process

def task_done_callback(future):
    processing_now.discard(future.linkedid)

def main():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass

    while True:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_ASR_WORKERS, max_tasks_per_child=10) as executor:
                while True:
                    if not get_system_running_status():
                        time.sleep(5)
                        continue

                    if len(processing_now) < NUM_ASR_WORKERS:
                        # 1. Check current (today/yesterday)
                        files = scan_current_files()
                        available_files = [f for f in files if os.path.splitext(os.path.basename(f))[0] not in processing_now]
                        if available_files:
                            f_path = available_files[0]
                            linkedid = os.path.splitext(os.path.basename(f_path))[0]
                            processing_now.add(linkedid)
                            future = executor.submit(process_asr, f_path)
                            future.linkedid = linkedid
                            future.add_done_callback(task_done_callback)
                            continue

                        # 2. Check tasks
                        tasks = get_active_tasks()
                        task_found = False
                        for task in tasks:
                            if task['asr_status'] == 'completed': continue

                            if task['asr_status'] == 'planned':
                                update_task_status(task['id'], asr_status='processing')

                            task_files = get_task_files(task)
                            available_task_files = [f for f in task_files if os.path.splitext(os.path.basename(f))[0] not in processing_now]
                            if available_task_files:
                                f_path = available_task_files[0]
                                linkedid = os.path.splitext(os.path.basename(f_path))[0]
                                processing_now.add(linkedid)
                                future = executor.submit(process_asr, f_path)
                                future.linkedid = linkedid
                                future.add_done_callback(task_done_callback)
                                task_found = True
                                break
                            else:
                                # ASR task is completed only if all files for the period are processed
                                # and the end_date is in the past (to avoid closing task for today prematurely)
                                # or if we are sure no more files will come.
                                # Simplified: if it's not today, we can complete it.
                                if task['end_date'] < datetime.now().date():
                                    update_task_status(task['id'], asr_status='completed')

                        if not task_found and not processing_now:
                            time.sleep(5)
                    else:
                        time.sleep(1)
        except Exception as e:
            logger.exception(f"Error in asr_dispatcher: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
