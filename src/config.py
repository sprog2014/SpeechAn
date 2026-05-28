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

# Динамическая настройка ресурсов
# 80 ядер / 8 потоков = 10 воркеров.
# При 10 воркерах каждый использует по 1 модели Llama (5ГБ) = 50ГБ.
# Остается 14ГБ на ASR, Эмоции и ОС. Это предел, но должно работать.
cpu_count = multiprocessing.cpu_count()
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 10))
OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS", "8")

WEB_USER = os.getenv("WEB_USER", "admin")
WEB_PASSWORD = os.getenv("WEB_PASSWORD", "admin")
