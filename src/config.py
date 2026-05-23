import os
from dotenv import load_dotenv

load_dotenv()

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

RECORDS_ROOT = os.getenv("RECORDS_ROOT", "/data/calls")
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 10))
OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS", "8")