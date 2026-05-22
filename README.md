# Call Analysis Pipeline

Автоматическая система анализа телефонных разговоров колл-центра медицинской клиники.

## Возможности
- Транскрибация русской речи (faster-whisper)
- Распознавание эмоций в голосе (GigaAMEmo)
- Смысловой анализ диалога: цель звонка, вежливость, чек-лист (Saiga LLM)
- Хранение результатов в PostgreSQL
- Дашборд на Streamlit

## Требования
- Linux-сервер с CPU (рекомендуется 16+ ядер, 32+ ГБ ОЗУ)
- Python 3.11+
- ffmpeg
- PostgreSQL 15+
- MySQL (исходная БД с метаданными)

## Установка и запуск

1. Клонировать репозиторий:
 ```git clone <repo_url> | cd call_analysis```

2. Установить зависимости:
 ```python -m venv venv | source venv/bin/activate | pip install -r requirements.txt```

3. Установить ffmpeg (если отсутствует):
 ```sudo apt-get install ffmpeg```

4. Создать ```.env``` на основе ```.env.example```, заполнить параметры подключения.

5. Инициализировать БД PostgreSQL:
 ```psql -h ... -U postgres -f db/init.sql```

6. Загрузить необходимые модели (см. раздел «Загрузка моделей»).

7. Запустить диспетчер (он же запускает пул воркеров):
 - В Linux:
   ```bash scripts/run_dispatcher.sh```
 - В Windows:
   ```scripts\run_dispatcher.bat```

8. Для визуализации запустить дашборд:
 - В Linux:
   ```streamlit run dashboard/dashboard.py --server.port 8501```
 - В Windows:
   ```scripts\run_dashboard.bat```

## Загрузка моделей

Система использует три модели.
Путь для хранения моделей по умолчанию – каталог ```models/``` в корне проекта.
Для изменения пути используйте переменные окружения в ```.env```.

### 1. faster-whisper large-v3-turbo
- Загружается автоматически при первом запуске из Hugging Face Hub.
- Кэшируется в ```~/.cache/huggingface/hub```.
- Никаких дополнительных действий не требуется.
- Если нужен оффлайн-режим, предварительно скачайте модель:
 ```python -c "from faster_whisper import WhisperModel; WhisperModel('Systran/faster-whisper-large-v3-turbo', device='cpu', compute_type='int8', download_root='./models')"```
 и установите переменную окружения ```FASTER_WHISPER_MODEL_PATH=./models```.

### 2. GigaAMEmo (Сбер)
- Загружается автоматически через библиотеку transformers при первом вызове.
- Кэш также в ```~/.cache/huggingface/hub```.
- Для оффлайн-режима скачайте вручную:
 ```huggingface-cli download SberDevices/GigaAMEmo --local-dir ./models/GigaAMEmo```
 и укажите путь в коде (по умолчанию ищется в кэше, но можно задать ```local_files_only=True```).

### 3. Saiga/Llama3 8B (GGUF)
- Не загружается автоматически. Необходимо скачать файл GGUF вручную.
- Рекомендуемая квантованная версия: ```q4_K``` (баланс качество/память).
- Скачайте файл ```model-q4_K.gguf``` с Hugging Face:
 ```wget https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf/resolve/main/model-q4_K.gguf -P models/```
- Путь к файлу задаётся в ```.env``` переменной ```LLM_MODEL_PATH``` (по умолчанию ```models/model-q4_K.gguf```).

## Конфигурация
Все настройки – в ```.env```. Основные переменные:
- ```RECORDS_ROOT``` – корень архива mp3 с иерархией год/месяц/день
- ```NUM_WORKERS``` – количество параллельных процессов-воркеров
- ```OMP_NUM_THREADS``` – потоки на один воркер
- ```LLM_MODEL_PATH``` – путь к GGUF-модели Saiga

## Структура БД
- ```calls``` – метаданные звонка (из MySQL) + путь к файлу, статус
- ```transcripts``` – расшифровка фраз с каналами и временем
- ```speech_emotions``` – эмоциональная окраска каждой фразы
- ```evaluations``` – итоговые оценки LLM (вежливость, цель, чек-лист)

## Примечание
- Левый канал mp3 считается оператором, правый – клиентом.
- Для работы измените SQL-запрос метаданных под вашу таблицу ```analytics```.
