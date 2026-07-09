import sys
import os
import logging

# Добавляем путь к src, чтобы импортировать db_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from db_utils import get_pg_connection, get_rating_from_mysql
from logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def backfill_ratings():
    logger.info("Starting backfill of ratings from MySQL to PostgreSQL evaluations table...")

    try:
        with get_pg_connection() as pg_conn:
            cur = pg_conn.cursor()

            # Получаем все уникальные linkedid из таблицы evaluations
            cur.execute("SELECT DISTINCT linkedid FROM evaluations")
            rows = cur.fetchall()

            if not rows:
                logger.info("No evaluations found to backfill.")
                return

            total = len(rows)
            logger.info(f"Found {total} unique linkedid(s) to process.")

            updated_count = 0
            for index, (linkedid,) in enumerate(rows):
                rating = get_rating_from_mysql(linkedid)

                # Обновляем все записи в evaluations для этого linkedid
                cur.execute("""
                    UPDATE evaluations
                    SET rating = %s
                    WHERE linkedid = %s
                """, (rating, linkedid))

                updated_count += 1
                if updated_count % 100 == 0:
                    pg_conn.commit()
                    logger.info(f"Processed {updated_count}/{total}...")

            pg_conn.commit()
            logger.info(f"Backfill completed successfully. Total linkedid processed: {updated_count}")

    except Exception as e:
        logger.exception(f"Error during backfill: {e}")

if __name__ == "__main__":
    backfill_ratings()
