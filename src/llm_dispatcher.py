import os
import time
import logging
import multiprocessing
import concurrent.futures
from datetime import datetime, timedelta
from config import NUM_LLM_WORKERS
from db_utils import get_pg_connection, get_system_running_status, get_active_tasks, update_task_status, get_default_prompt
from llm_worker import process_llm
from logging_utils import setup_logging

setup_logging()
logger = logging.getLogger("llm_dispatcher")

processing_now = set()

def get_today_yesterday_calls(prompt_id):
    try:
        with get_pg_connection() as conn:
            cur = conn.cursor()
            # Find calls from today/yesterday that are 'transcribed' or 'done' but no evaluation for the default prompt
            # We explicitly exclude 'empty' as there is no transcript to analyze.
            cur.execute("""
                SELECT c.linkedid
                FROM calls c
                LEFT JOIN evaluations e ON c.linkedid = e.linkedid AND e.prompt_id = %s
                WHERE c.calldate >= CURRENT_DATE - INTERVAL '1 day'
                AND c.processing_status IN ('transcribed', 'done')
                AND e.linkedid IS NULL
            """, (prompt_id,))
            return [row[0] for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Error fetching today/yesterday calls: {e}")
        return []

def get_task_calls(task):
    try:
        with get_pg_connection() as conn:
            cur = conn.cursor()
            statuses = ['transcribed', 'done']
            if task.get('analyze_all'):
                statuses.append('skipped')

            cur.execute("""
                SELECT c.linkedid
                FROM calls c
                LEFT JOIN evaluations e ON c.linkedid = e.linkedid AND e.prompt_id = %s
                WHERE c.calldate::date >= %s AND c.calldate::date <= %s
                AND c.processing_status = ANY(%s)
                AND e.linkedid IS NULL
            """, (task['prompt_id'], task['start_date'], task['end_date'], statuses))
            return [row[0] for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Error fetching task calls: {e}")
        return []

def task_done_callback(future):
    processing_now.discard(future.linkedid)

def main():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass

    while True:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_LLM_WORKERS, max_tasks_per_child=10) as executor:
                while True:
                    if not get_system_running_status():
                        time.sleep(5)
                        continue

                    if len(processing_now) < NUM_LLM_WORKERS:
                        # 0. Get default prompt
                        default_prompt = get_default_prompt()
                        if not default_prompt:
                            logger.error("No default prompt found")
                            time.sleep(10)
                            continue

                        # 1. Check current (today/yesterday)
                        calls = get_today_yesterday_calls(default_prompt['id'])
                        available_calls = [cid for cid in calls if cid not in processing_now]
                        if available_calls:
                            linkedid = available_calls[0]
                            processing_now.add(linkedid)
                            future = executor.submit(process_llm, linkedid, default_prompt['id'], False)
                            future.linkedid = linkedid
                            future.add_done_callback(task_done_callback)
                            continue

                        # 2. Check tasks
                        tasks = get_active_tasks()
                        task_found = False
                        for task in tasks:
                            if task['llm_status'] == 'completed': continue

                            if task['llm_status'] == 'planned':
                                update_task_status(task['id'], llm_status='processing')

                            task_calls = get_task_calls(task)
                            available_task_calls = [cid for cid in task_calls if cid not in processing_now]
                            if available_task_calls:
                                linkedid = available_task_calls[0]
                                processing_now.add(linkedid)
                                future = executor.submit(process_llm, linkedid, task['prompt_id'], task['analyze_all'])
                                future.linkedid = linkedid
                                future.add_done_callback(task_done_callback)
                                task_found = True
                                break
                            else:
                                # LLM task is completed only if ASR part is completed AND no more calls left to process
                                if task['asr_status'] == 'completed':
                                    update_task_status(task['id'], llm_status='completed')

                        if not task_found and not processing_now:
                            time.sleep(5)
                    else:
                        time.sleep(1)
        except Exception as e:
            logger.exception(f"Error in llm_dispatcher: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
