import os
from pathlib import Path
import multiprocessing

# Пытаемся загрузить .env из /opt/calls/.env или из текущей папки
try:
    from dotenv import load_dotenv
    # Сначала проверяем глобальный путь
    global_env = Path("/opt/calls/.env")
    if global_env.exists():
        load_dotenv(dotenv_path=global_env)
    else:
        load_dotenv()
except ImportError:
    pass

PG_CONFIG = {
    "host": os.getenv("PG_HOST"),
    "port": int(os.getenv("PG_PORT", 5432)),
    "dbname": os.getenv("PG_DB"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD")
}

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "database": os.getenv("MYSQL_DB"),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD")
}

RECORDS_ROOT = os.getenv("RECORDS_ROOT", "/mnt/rec")

# Динамическая настройка на основе ядер процессора
# Цель: занять 100% процессора.
# 80 ядер / 8 потоков на модель = 10 воркеров.
cpu_count = multiprocessing.cpu_count()
default_workers = max(1, cpu_count // 8)
default_threads = 8

NUM_WORKERS = int(os.getenv("NUM_WORKERS", default_workers))
OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS", str(default_threads))
LLM_POOL_SIZE = int(os.getenv("LLM_POOL_SIZE", NUM_WORKERS))

WEB_USER = os.getenv("WEB_USER", "admin")
WEB_PASSWORD = os.getenv("WEB_PASSWORD", "admin")
