#!/bin/bash
set -e

# Настройки
BASE_DIR="/opt/calls"
REPO_URL="https://github.com/sprog2014/SpeechAn.git"
DB_NAME="call_analysis"
DB_USER="analyzer"
DB_PASS="]=[-p0o"

echo "=== Speech Analytics Global Setup Script ==="

# 1. Системные зависимости
echo "[1/7] Installing system dependencies..."
sudo apt update
sudo apt install -y git ffmpeg default-mysql-client postgresql postgresql-contrib build-essential software-properties-common wget curl

# 2. Python 3.12
echo "[2/7] Installing Python 3.12..."
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-dev python3-pip

# 3. Структура папок
echo "[3/7] Creating directory structure..."
sudo mkdir -p $BASE_DIR/models
sudo mkdir -p /mnt/rec
sudo chown -R $USER:$USER $BASE_DIR
sudo chown -R $USER:$USER /mnt/rec

# 4. PostgreSQL
echo "[4/7] Setting up PostgreSQL..."
sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';" || echo "User already exists"
sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;" || echo "Database already exists"

# 5. Клонирование проекта
echo "[5/7] Cloning project into $BASE_DIR/SpeechAn..."
cd $BASE_DIR
if [ -d "SpeechAn" ]; then
    echo "SpeechAn directory already exists, pulling updates..."
    cd SpeechAn && git pull && cd ..
else
    git clone $REPO_URL SpeechAn
fi

# Запуск init.sql
echo "Running database initialization..."
PGPASSWORD=$DB_PASS psql -h localhost -U $DB_USER -d $DB_NAME -f $BASE_DIR/SpeechAn/db/init.sql

# 6. Глобальная установка зависимостей Python
echo "[6/7] Installing Python dependencies globally..."
sudo python3.12 -m pip install -r $BASE_DIR/SpeechAn/requirements.txt --break-system-packages --ignore-installed

# 7. Загрузка модели и создание конфига
echo "[7/7] Setup assets and config..."
# Предварительная загрузка и конвертация модели Qwen2.5-7B-Instruct в OpenVINO INT8
if [ ! -d "$BASE_DIR/models/qwen2.5-7b-instruct-ov" ]; then
    echo "Downloading and converting Qwen2.5 model to OpenVINO INT8..."
    # Используем временный скрипт для экспорта, так как зависимости уже установлены глобально
    sudo python3.12 -c "
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer
model_id = 'Qwen/Qwen2.5-7B-Instruct'
save_path = '$BASE_DIR/models/qwen2.5-7b-instruct-ov'
model = OVModelForCausalLM.from_pretrained(model_id, export=True, quantization_config={'bits': 8}, device='CPU')
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
"
fi

if [ ! -f "$BASE_DIR/.env" ]; then
    cat <<EOF > $BASE_DIR/.env
PG_HOST=localhost
PG_PORT=5432
PG_DB=$DB_NAME
PG_USER=$DB_USER
PG_PASSWORD=$DB_PASS

MYSQL_HOST=172.16.1.7
MYSQL_PORT=3306
MYSQL_DB=pbxanalytics
MYSQL_USER=umc
MYSQL_PASSWORD=umc2pbx

RECORDS_ROOT=/mnt/rec
NUM_WORKERS=8
OMP_NUM_THREADS=8
LLM_MODEL_PATH=$BASE_DIR/models/qwen2.5-7b-instruct-ov

WEB_USER=admin
WEB_PASSWORD=admin
EOF
    echo ".env created in $BASE_DIR/.env"
fi

echo "=== Setup Complete! ==="
