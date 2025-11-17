PRAGMA foreign_keys = ON;

-- 1) Core table: one row per WhatsApp message
-- Directional conversation id: dm:<sender>-><receiver>
CREATE TABLE IF NOT EXISTS messages (
  msg_id            TEXT PRIMARY KEY,
  sender_user_id    TEXT NOT NULL,
  receiver_user_id  TEXT NOT NULL,

  conv_id           TEXT GENERATED ALWAYS AS (
    'dm:' || sender_user_id || '->' || receiver_user_id
  ) STORED,

  sent_at           TEXT NOT NULL,   -- ISO 8601 timestamp
  text              TEXT NOT NULL,   -- cleaned text used for training and retrieval
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_messages_conv_time
  ON messages (conv_id, sent_at);

CREATE INDEX IF NOT EXISTS idx_messages_sender_time
  ON messages (sender_user_id, sent_at);

CREATE INDEX IF NOT EXISTS idx_messages_receiver_time
  ON messages (receiver_user_id, sent_at);

/* -----------------------------------------------------------
   OPTIONAL: Full-text search (FTS5) for fast keyword retrieval
   Uncomment if your SQLite build has FTS5 enabled.
------------------------------------------------------------*/

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  msg_id UNINDEXED,
  conv_id UNINDEXED,
  sender_user_id UNINDEXED,
  receiver_user_id UNINDEXED,
  text,
  content=''
);

-- Keep FTS index in sync with the base table
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts (msg_id, conv_id, sender_user_id, receiver_user_id, text)
  VALUES (new.msg_id, new.conv_id, new.sender_user_id, new.receiver_user_id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  DELETE FROM messages_fts WHERE msg_id = old.msg_id;
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE OF text ON messages BEGIN
  UPDATE messages_fts SET text = new.text WHERE msg_id = old.msg_id;
END;

-- Example keyword query:
-- SELECT msg_id, text FROM messages_fts
-- WHERE messages_fts MATCH 'meeting OR tonight'
-- AND conv_id = 'dm:u_ran->u_dana'
-- LIMIT 50;

/*
INSERT INTO messages (msg_id, sender_user_id, receiver_user_id, sent_at, text)
VALUES
  ('m1', 'u_ran',  'u_dana', '2025-11-04T10:00:00Z', 'מגיע בשבע?'),
  ('m2', 'u_dana', 'u_ran',  '2025-11-04T10:01:05Z', 'כן, מתאים לי');

  SELECT sender_user_id, text
FROM messages
WHERE conv_id = 'dm:u_ran->u_dana'
ORDER BY sent_at DESC
LIMIT 20;
*/