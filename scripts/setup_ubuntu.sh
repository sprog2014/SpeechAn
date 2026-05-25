#!/bin/bash
set -e

# Настройки
BASE_DIR="/opt/calls"
REPO_URL="https://github.com/user/SpeechAn.git"
DB_NAME="call_analysis"
DB_USER="analyzer"
DB_PASS="secure_password"
LLAMA_MODEL_URL="https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf/resolve/main/model-q4_K.gguf"

echo "=== Speech Analytics Global Setup Script ==="

# 1. Системные зависимости
echo "[1/7] Installing system dependencies..."
sudo apt update
sudo apt install -y git ffmpeg default-mysql-client postgresql postgresql-contrib build-essential software-properties-common wget curl

# 2. Python 3.14
echo "[2/7] Installing Python 3.14..."
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.14

# Установка pip для 3.14
curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.14

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
sudo python3.14 -m pip install -r $BASE_DIR/SpeechAn/requirements.txt

# 7. Загрузка модели и создание конфига
echo "[7/7] Setup assets and config..."
if [ ! -f "$BASE_DIR/models/model-q4_K.gguf" ]; then
    wget -O $BASE_DIR/models/model-q4_K.gguf $LLAMA_MODEL_URL
fi

if [ ! -f "$BASE_DIR/.env" ]; then
    cat <<EOF > $BASE_DIR/.env
PG_HOST=localhost
PG_PORT=5432
PG_DB=$DB_NAME
PG_USER=$DB_USER
PG_PASSWORD=$DB_PASS

MYSQL_HOST=your_mysql_host
MYSQL_PORT=3306
MYSQL_DB=asterisk
MYSQL_USER=readonly
MYSQL_PASSWORD=readonly_password

RECORDS_ROOT=/mnt/rec
NUM_WORKERS=10
OMP_NUM_THREADS=8
LLM_MODEL_PATH=$BASE_DIR/models/model-q4_K.gguf
EOF
    echo ".env created in $BASE_DIR/.env"
fi

echo "=== Setup Complete! ==="
