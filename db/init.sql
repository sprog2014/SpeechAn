CREATE DATABASE call_analysis;
\c call_analysis

DROP TABLE IF EXISTS processing_stats;
DROP TABLE IF EXISTS evaluations;
DROP TABLE IF EXISTS transcripts;
DROP TABLE IF EXISTS tasks;
DROP TABLE IF EXISTS calls;
DROP TABLE IF EXISTS phones;
DROP TABLE IF EXISTS prompts;
DROP TABLE IF EXISTS system_settings;

CREATE TABLE calls (
    linkedid        VARCHAR(32) PRIMARY KEY,
    calldate        TIMESTAMP NOT NULL,
    src             VARCHAR(40),
    answeredext     VARCHAR(40),
    direction       VARCHAR(20),
    duration        INT,
    billsec         INT,
    fromtrunksrc    VARCHAR(80),
    moduleparams    VARCHAR(200),
    incomingtrunk   VARCHAR(80),
    file_path       TEXT NOT NULL,
    processing_status TEXT NOT NULL DEFAULT 'new'
        CHECK (processing_status IN ('new','processing','transcribed','done','error','skipped','empty','stop')),
    processing_duration REAL,
    created_at      TIMESTAMP DEFAULT now()
);

CREATE TABLE transcripts (
    id          BIGSERIAL PRIMARY KEY,
    linkedid    VARCHAR(32) NOT NULL REFERENCES calls(linkedid) ON DELETE CASCADE,
    channel     TEXT NOT NULL CHECK (channel IN ('operator','client')),
    start_time  NUMERIC(8,3),
    end_time    NUMERIC(8,3),
    text        TEXT NOT NULL,
    language    TEXT DEFAULT 'ru',
    diction     NUMERIC(5,2),
    wpm         INT
);
CREATE INDEX idx_transcripts_linkedid ON transcripts(linkedid);


CREATE TABLE prompts (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    prompt_text     TEXT NOT NULL,
    is_default      BOOLEAN DEFAULT FALSE,
    schema_json     JSONB DEFAULT '{}',
    created_at      TIMESTAMP DEFAULT now()
);

INSERT INTO prompts (id, name, prompt_text, is_default, schema_json) VALUES (
    1,
    'Default Medical Call Analysis',
    '<|im_start|>system
Ты — эксперт по контролю качества в медицинском колл-центре.
Твоя задача — проанализировать диалог между оператором и клиентом.
Результат анализа ты обязан выдать СТРОГО в формате JSON, соответствующем следующей схеме:
{json_schema}

Не добавляй никакого вступительного или заключительного текста, только один JSON объект.
<|im_end|>
<|im_start|>user
Проанализируй следующий диалог:
---
{transcript}
---
Верни JSON-объект с результатами анализа.
<|im_end|>',
    TRUE,
    '{"main": [{"key": "politeness_score", "type": "num", "description": "Оценка вежливости оператора от 0 до 10"}, {"key": "client_sentiment", "type": "str", "description": "Настроение клиента: positive, neutral, negative или conflict"}, {"key": "call_purpose", "type": "str", "description": "Цель звонка: appointment, consultation, complaint, cancel_appointment или other"}, {"key": "call_summary", "type": "str", "description": "Краткое содержание диалога (1-2 предложения)"}], "checklist": [{"key": "greeting", "type": "bool", "description": "Приветствие"}, {"key": "introduced_himself", "type": "bool", "description": "Представился"}, {"key": "identified_need", "type": "bool", "description": "Выявил потребность"}, {"key": "informed_price", "type": "bool", "description": "Сообщил стоимость"}, {"key": "agreed_datetime", "type": "bool", "description": "Согласовал дату/время"}, {"key": "handled_objection", "type": "bool", "description": "Отработал возражение"}, {"key": "farewell", "type": "bool", "description": "Прощание"}], "metrics": [{"key": "interruptions_count", "type": "num", "description": "Количество перебиваний"}, {"key": "hold_time_sec", "type": "num", "description": "Время удержания в секундах"}, {"key": "medication_mentioned", "type": "bool", "description": "Упоминание лекарств"}]}'
);

CREATE TABLE evaluations (
    linkedid          VARCHAR(32) NOT NULL REFERENCES calls(linkedid) ON DELETE CASCADE,
    prompt_id         INT NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
    politeness_score  REAL,
    client_sentiment  TEXT,
    call_purpose      TEXT,
    call_summary      TEXT,
    checklist_json    JSONB DEFAULT '{}',
    metrics_json      JSONB DEFAULT '{}',
    rating            SMALLINT DEFAULT 0,
    created_at        TIMESTAMP DEFAULT now(),
    PRIMARY KEY (linkedid, prompt_id)
);
CREATE INDEX idx_evals_politeness ON evaluations(politeness_score);
CREATE INDEX idx_evals_purpose ON evaluations(call_purpose);
CREATE INDEX idx_evals_checklist_gin ON evaluations USING GIN (checklist_json);
CREATE INDEX idx_evals_metrics_gin ON evaluations USING GIN (metrics_json);
CREATE INDEX idx_evals_prompt_id ON evaluations(prompt_id);
CREATE INDEX idx_calls_calldate ON calls(calldate);

CREATE TABLE system_settings (
    key VARCHAR(50) PRIMARY KEY,
    value TEXT
);

INSERT INTO system_settings (key, value) VALUES ('is_running', 'true');
INSERT INTO system_settings (key, value) VALUES ('skip_local_calls', 'false');

CREATE TABLE phones (
    number VARCHAR(40) PRIMARY KEY,
    name   VARCHAR(200),
    use    BOOLEAN DEFAULT TRUE
);

CREATE TABLE processing_stats (
    linkedid        VARCHAR(32) PRIMARY KEY REFERENCES calls(linkedid) ON DELETE CASCADE,
    asr_duration    REAL,
    llm_duration    REAL,
    total_duration  REAL,
    created_at      TIMESTAMP DEFAULT now()
);
CREATE INDEX idx_stats_created_at ON processing_stats(created_at);

CREATE TABLE tasks (
    id              SERIAL PRIMARY KEY,
    prompt_id       INT NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    analyze_all     BOOLEAN DEFAULT FALSE,
    asr_status      VARCHAR(20) DEFAULT 'planned' CHECK (asr_status IN ('planned','processing','completed')),
    llm_status      VARCHAR(20) DEFAULT 'planned' CHECK (llm_status IN ('planned','processing','completed')),
    created_at      TIMESTAMP DEFAULT now()
);

CREATE TABLE reports (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(200) NOT NULL UNIQUE,
    settings        JSONB NOT NULL,
    created_at      TIMESTAMP DEFAULT now()
);

CREATE TABLE field_synonyms (
    prompt_id INT NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
    technical_name VARCHAR(100) NOT NULL,
    synonym VARCHAR(255) NOT NULL,
    PRIMARY KEY (prompt_id, technical_name)
);

-- Pre-populate synonyms for the default prompt
INSERT INTO field_synonyms (prompt_id, technical_name, synonym) VALUES
(1, 'calldate', 'Дата и время'),
(1, 'direction', 'Направление'),
(1, 'duration', 'Длительность (общая)'),
(1, 'billsec', 'Длительность (разговор)'),
(1, 'politeness_score', 'Вежливость'),
(1, 'client_sentiment', 'Настроение'),
(1, 'call_purpose', 'Цель звонка'),
(1, 'operator_name', 'Имя оператора'),
(1, 'client_number', 'Номер клиента');
