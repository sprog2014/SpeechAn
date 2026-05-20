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

CREATE TABLE evaluations (
    linkedid          VARCHAR(32) PRIMARY KEY REFERENCES calls(linkedid) ON DELETE CASCADE,
    politeness_score  REAL,
    client_sentiment  TEXT,
    call_purpose      TEXT,
    call_summary      TEXT,
    checklist_json    JSONB DEFAULT '{}',
    metrics_json      JSONB DEFAULT '{}',
    created_at        TIMESTAMP DEFAULT now()
);
CREATE INDEX idx_evals_politeness ON evaluations(politeness_score);
CREATE INDEX idx_evals_purpose ON evaluations(call_purpose);
CREATE INDEX idx_evals_checklist_gin ON evaluations USING GIN (checklist_json);