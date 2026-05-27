CREATE DATABASE call_analysis;
\c call_analysis

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
        CHECK (processing_status IN ('new','processing','done','error')),
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

CREATE TABLE speech_emotions (
    id              BIGSERIAL PRIMARY KEY,
    transcript_id   BIGINT REFERENCES transcripts(id) ON DELETE CASCADE,
    emotion         TEXT NOT NULL,
    confidence      REAL CHECK (confidence >= 0 AND confidence <= 1)
);
CREATE INDEX idx_emotions_transcript ON speech_emotions(transcript_id);

CREATE TABLE prompts (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    prompt_text     TEXT NOT NULL,
    is_default      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT now()
);

INSERT INTO prompts (name, prompt_text, is_default) VALUES (
    'Default Medical Call Analysis',
    'Ты — эксперт по контролю качества в медицинском колл-центре. Проанализируй диалог оператора и клиента и верни **только** JSON без лишних слов.
Поля:
- politeness_score: число от 0 до 10
- client_sentiment: "positive", "neutral", "negative", "conflict"
- call_purpose: "appointment", "consultation", "complaint", "cancel_appointment", "other"
- call_summary: краткое содержание 1-2 предложения
- checklist: объект с ключами: greeting, introduced_himself, identified_need, informed_price, agreed_datetime, handled_objection, farewell. Каждое поле true/false.
- metrics: объект с дополнительной информацией, например, interruptions_count, hold_time_sec, medication_mentioned (true/false)

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
