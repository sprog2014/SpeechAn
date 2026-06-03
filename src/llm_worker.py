import os
import logging
import time
from db_utils import (
    get_pg_connection, get_prompt_by_id, check_evaluation_exists,
    insert_evaluation, set_call_done, set_call_error, set_processing_duration,
    insert_processing_stats, check_phone_usage, get_system_setting,
    is_phone_registered, set_call_status, format_dialogue, get_call_transcript
)
from llm_analysis import analyze_transcript
from logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def process_llm(linkedid: str, prompt_id: int, analyze_all: bool = False):
    logger.info(f"[{linkedid}] --- LLM started --- (Prompt ID: {prompt_id})")

    start_time = time.time()
    pg_conn = None

    try:
        with get_pg_connection() as conn:
            pg_conn = conn

            # 1. Check if evaluation already exists
            if check_evaluation_exists(linkedid, prompt_id, conn=pg_conn):
                logger.info(f"[{linkedid}] Evaluation exists. Skipping.")
                return True

            # 2. Get prompt
            prompt_data = get_prompt_by_id(prompt_id, conn=pg_conn)
            if not prompt_data:
                logger.error(f"[{linkedid}] Prompt not found")
                return False

            # Set status to processing
            set_call_status(linkedid, 'processing', conn=pg_conn)

            # 3. Phone filtering (if not analyze_all)
            if not analyze_all:
                cur = pg_conn.cursor()
                cur.execute("SELECT src, answeredext FROM calls WHERE linkedid = %s", (linkedid,))
                row = cur.fetchone()
                if row:
                    src, dst = row
                    allowed_src = check_phone_usage(src, conn=pg_conn)
                    allowed_dst = check_phone_usage(dst, conn=pg_conn)
                    if not allowed_src and not allowed_dst:
                        logger.info(f"[{linkedid}] Numbers not enabled. Skipping LLM.")
                        set_call_status(linkedid, 'skipped', conn=pg_conn)
                        return True

            # 4. Get transcript
            transcript_rows = get_call_transcript(linkedid, conn=pg_conn)
            if not transcript_rows:
                logger.warning(f"[{linkedid}] No transcript found")
                return False

            full_dialogue = format_dialogue(transcript_rows)
            if not full_dialogue.strip():
                logger.warning(f"[{linkedid}] Empty dialogue")
                return True

            # 5. LLM Analysis
            eval_result = analyze_transcript(full_dialogue, prompt_template=prompt_data['prompt_text'])
            insert_evaluation(linkedid, prompt_id, eval_result, conn=pg_conn)

            duration = time.time() - start_time
            set_call_done(linkedid, conn=pg_conn)
            set_processing_duration(linkedid, duration, conn=pg_conn)

            # Try to get asr_duration from processing_stats if it was already inserted by ASR (though currently asr_worker doesn't insert it)
            # Or just update the llm_duration in processing_stats
            cur = pg_conn.cursor()
            cur.execute("SELECT asr_duration FROM processing_stats WHERE linkedid = %s", (linkedid,))
            row = cur.fetchone()
            asr_dur = row[0] if row else 0

            insert_processing_stats(linkedid, asr_dur, duration, asr_dur + duration, conn=pg_conn)

            logger.info(f"[{linkedid}] LLM finished in {duration:.2f}s")
            return True

    except Exception as e:
        logger.exception(f"[{linkedid}] LLM failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python llm_worker.py <linkedid> <prompt_id> [analyze_all]")
        sys.exit(1)
    analyze_all = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
    process_llm(sys.argv[1], int(sys.argv[2]), analyze_all)
