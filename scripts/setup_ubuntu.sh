#!/bin/bash
set -e

# Настройки
BASE_DIR="/opt/calls"
REPO_URL="https://github.com/user/SpeechAn.git" # Замените на ваш URL
DB_NAME="call_analysis"
DB_USER="analyzer"
DB_PASS="secure_password"
LLAMA_MODEL_URL="https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf/resolve/main/model-q4_K.gguf"

echo "=== Speech Analytics Setup Script ==="

# 1. Обновление и установка системных зависимостей
echo "[1/8] Installing system dependencies..."
sudo apt update
sudo apt install -y git ffmpeg default-mysql-client postgresql postgresql-contrib build-essential software-properties-common wget curl

# 2. Установка Python 3.14 (через deadsnakes PPA)
echo "[2/8] Installing Python 3.14..."
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.14 python3.14-venv python3.14-dev

# 3. Создание структуры папок
echo "[3/8] Creating directory structure..."
sudo mkdir -p $BASE_DIR/models
sudo mkdir -p $BASE_DIR/data/calls
sudo chown -R $USER:$USER $BASE_DIR

# 4. Настройка PostgreSQL
echo "[4/8] Setting up PostgreSQL..."
sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';" || echo "User already exists"
sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;" || echo "Database already exists"

# 5. Клонирование проекта
echo "[5/8] Cloning project..."
cd $BASE_DIR
if [ -d "SpeechAn" ]; then
    echo "SpeechAn directory already exists, skipping clone."
else
    git clone $REPO_URL SpeechAn
fi
cd SpeechAn

# Запуск init.sql
echo "Running database initialization..."
PGPASSWORD=$DB_PASS psql -h localhost -U $DB_USER -d $DB_NAME -f db/init.sql

# 6. Настройка виртуального окружения
echo "[6/8] Setting up virtual environment..."
python3.14 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 7. Загрузка модели LLM
echo "[7/8] Downloading Llama model..."
if [ -f "$BASE_DIR/models/model-q4_K.gguf" ]; then
    echo "Model already exists, skipping download."
else
    wget -O $BASE_DIR/models/model-q4_K.gguf $LLAMA_MODEL_URL
fi

# 8. Создание .env файла
echo "[8/8] Creating .env file..."
if [ -f ".env" ]; then
    echo ".env already exists, keeping it."
else
    cat <<EOF > .env
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

RECORDS_ROOT=$BASE_DIR/data/calls
NUM_WORKERS=10
OMP_NUM_THREADS=8
LLM_MODEL_PATH=$BASE_DIR/models/model-q4_K.gguf
EOF
    echo ".env created. Please edit it to add MySQL credentials."
fi

echo "=== Setup Complete! ==="
echo "You can now run the dispatcher using: scripts/run_dispatcher.sh"
