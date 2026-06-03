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
        CHECK (processing_status IN ('new','processing','transcribed','done','error','skipped')),
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
    language    TEXT DEFAULT 'ru'
);
CREATE INDEX idx_transcripts_linkedid ON transcripts(linkedid);


CREATE TABLE prompts (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    prompt_text     TEXT NOT NULL,
    is_default      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT now()
);

INSERT INTO prompts (name, prompt_text, is_default) VALUES (
    'Default Medical Call Analysis',
    'Ты — эксперт по контролю качества в медицинском колл-центре.
Твоя задача — проанализировать диалог между оператором и клиентом.
Результат анализа ты обязан выдать СТРОГО в формате JSON.
Не добавляй никакого вступительного или заключительного текста, только один JSON объект.

Формат JSON:
{{
  "politeness_score": число от 0 до 10,
  "client_sentiment": "positive", "neutral", "negative" или "conflict",
  "call_purpose": "appointment", "consultation", "complaint", "cancel_appointment" или "other",
  "call_summary": "краткое содержание 1-2 предложения",
  "checklist": {{
    "greeting": true/false,
    "introduced_himself": true/false,
    "identified_need": true/false,
    "informed_price": true/false,
    "agreed_datetime": true/false,
    "handled_objection": true/false,
    "farewell": true/false
  }},
  "metrics": {{
    "interruptions_count": число,
    "hold_time_sec": число,
    "medication_mentioned": true/false
  }}
}}

Диалог:
{transcript}
',
    TRUE
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
    created_at        TIMESTAMP DEFAULT now(),
    PRIMARY KEY (linkedid, prompt_id)
);
CREATE INDEX idx_evals_politeness ON evaluations(politeness_score);
CREATE INDEX idx_evals_purpose ON evaluations(call_purpose);
CREATE INDEX idx_evals_checklist_gin ON evaluations USING GIN (checklist_json);
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
