// Simple Node service for whatsapp-web.js with REST endpoints for Django
//
// Endpoints (Django -> Node):
//   POST /node/session           -> create/ensure a session { session_hint }
//   GET  /node/session/:id/qr    -> { qr: <dataURL or null> }
//   GET  /node/session/:id/status -> { state, session_id }
//
// Events (Node -> Django, session state):
//   On authenticated/ready/disconnected it POSTs to:
//     ${DJANGO_CALLBACK_URL}${sessionId}/state
//   with header X-Node-Secret
//
// New in this version:
//   When a session becomes 'ready', Node also:
//   - Scrapes unread messages
//   - Sends them to Django in batches:
//       POST /ingest/unread/<session_id>/batch
//   - Sends final job state:
//       POST /ingest/scrape/<session_id>/state

const express = require('express');
const axios = require('axios');
const qrcode = require('qrcode');
const { Client, LocalAuth } = require('whatsapp-web.js');
const path = require('path');
const { randomUUID } = require('crypto');

const PORT = process.env.PORT || 3001;

// For session state callbacks (already used in your project)
const DJANGO_CALLBACK_URL =
  process.env.DJANGO_CALLBACK_URL ||
  'http://localhost:8000/ingest/session/';

// Shared secret – must match Django settings.NODE_SHARED_SECRET
const DJANGO_SECRET = process.env.DJANGO_SECRET || 'mysecret';

// For unread scraping callbacks
// Final URLs will be:
//   POST `${DJANGO_INGEST_BASE}unread/<session_id>/batch`
//   POST `${DJANGO_INGEST_BASE}scrape/<session_id>/state`
const DJANGO_INGEST_BASE =
  process.env.DJANGO_INGEST_BASE ||
  'http://localhost:8000/ingest/';

const app = express();
app.use(express.json());

// In-memory session registry (ok for a small project)
//
// sessions: Map<string, {
//   client: Client,
//   state: 'pending'|'authenticated'|'ready'|'disconnected',
//   qrDataUrl: string|null,
//   initialized: boolean,
//   scrapingJobId?: string
// }>
const sessions = new Map();

// ───────────────────── helper: notify Django about session state ─────────────────────

async function notifyDjango(sessionId, state, error = '') {
  try {
    await axios.post(
      `${DJANGO_CALLBACK_URL}${sessionId}/state`,
      { state, error },
      {
        headers: {
          'Content-Type': 'application/json',
          'X-Node-Secret': DJANGO_SECRET,
        },
        timeout: 5000,
      }
    );
  } catch (e) {
    // For a student project we just log; Django will pick up state next refresh anyway.
    console.log(`[notifyDjango] failed: ${e.message}`);
  }
}

// ───────────────────── helper: post unread messages in batch ─────────────────────
async function postUnreadBatch(sessionId, messages) {
  if (!messages || messages.length === 0) return;

  try {
    await axios.post(
      `${DJANGO_INGEST_BASE}unread/${sessionId}/batch`,
      { messages },
      {
        headers: {
          'Content-Type': 'application/json',
          'X-Node-Secret': DJANGO_SECRET,
        },
        timeout: 10000,
      }
    );
  } catch (e) {
    console.log(
      `[${sessionId}] postUnreadBatch failed: ${e.message}`
    );
  }
}

async function scrapeUnreadForSession(sessionId, client, record) {
  console.log(`[${sessionId}] starting unread scrape`);

  const THREE_MONTHS_SECONDS = 90 * 24 * 60 * 60;
  const nowSeconds = Math.floor(Date.now() / 1000);
  const cutoffTs = nowSeconds - THREE_MONTHS_SECONDS;

  try {
    const chats = await client.getChats();

    for (const chat of chats) {
      const chatIdStr =
        chat.id && chat.id._serialized ? chat.id._serialized : String(chat.id || '');
      if (!chatIdStr.endsWith('@c.us')) continue;

      if (!chat.unreadCount || chat.unreadCount <= 0) continue;

      let messages = [];
      try {
        messages = await chat.fetchMessages({ limit: chat.unreadCount });
      } catch (e) {
        console.log(`[${sessionId}] failed to fetch messages for ${chatIdStr}: ${e.message}`);
        continue;
      }

      const filtered = messages.filter((m) => {
        if (m.isStatus) return false;
        const ts = m.timestamp ? m.timestamp : nowSeconds;
        return ts >= cutoffTs;
      });

      if (filtered.length === 0) continue;

      // Build payload with async contact name resolution
      const batchPayload = [];
      for (const m of filtered) {
        let displayName = m.from;
        try {
          const contact = await m.getContact();
          displayName =
            contact.pushname ||
            contact.name ||
            contact.shortName ||
            contact.number ||
            m.from;
        } catch (e) {
          console.log(`[${sessionId}] failed contact lookup for ${m.from}: ${e.message}`);
        }

        batchPayload.push({
          wa_msg_id: m.id && m.id._serialized ? m.id._serialized : m.id,
          chat_id: chatIdStr,
          sender: displayName,   
          body: m.body,
          msg_ts: m.timestamp ? m.timestamp : nowSeconds,
        });
      }

      // Send in chunks
      const chunkSize = 50;
      for (let i = 0; i < batchPayload.length; i += chunkSize) {
        await postUnreadBatch(sessionId, batchPayload.slice(i, i + chunkSize));
      }
    }

    console.log(`[${sessionId}] unread scrape done (private chats, last 3 months only)`);
  } catch (err) {
    console.log(`[${sessionId}] scrapeUnread error: ${err.message}`);
  }
}


// ───────────────────── create or get WA client for a session ─────────────────────

async function ensureSession(sessionId) {
  if (sessions.has(sessionId)) {
    return sessions.get(sessionId);
  }

  const client = new Client({
    authStrategy: new LocalAuth({
      // Each session gets a stable folder under ./sessions
      dataPath: path.join(__dirname, 'sessions'),
      clientId: sessionId, // keeps auth for this logical session
    }),
    puppeteer: {
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu',
      ],
    },
  });

  const record = {
    client,
    state: 'pending',
    qrDataUrl: null,
    initialized: false,
    scrapingJobId: null,
  };

  sessions.set(sessionId, record);

  // Wire events

  client.on('qr', async (qr) => {
    try {
      record.qrDataUrl = await qrcode.toDataURL(qr);
      record.state = 'pending'; // still waiting on auth
      console.log(`[${sessionId}] QR updated`);
      // No need to notify Django on QR updates; Django pulls /qr when rendering
    } catch (e) {
      console.log(`[${sessionId}] QR encode error: ${e.message}`);
    }
  });

  client.on('authenticated', async () => {
    record.state = 'authenticated';
    record.qrDataUrl = null; // QR not needed anymore
    console.log(`[${sessionId}] authenticated`);
    await notifyDjango(sessionId, 'authenticated');
  });

  client.on('ready', async () => {
    record.state = 'ready';
    record.qrDataUrl = null;
    console.log(`[${sessionId}] ready`);

    // 1) Notify Django that the session is ready (login wait page will redirect to /unread/)
    await notifyDjango(sessionId, 'ready');

    // 2) Start unread scraping job for this session
    //    (Django will see ScrapeJob + UnreadSession rows and show /unread/ page)
    try {
      await scrapeUnreadForSession(sessionId, client, record);
    } catch (e) {
      console.log(
        `[${sessionId}] failed to start scrapeUnreadForSession: ${e.message}`
      );
    }
  });

  client.on('disconnected', async (reason) => {
    record.state = 'disconnected';
    record.qrDataUrl = null;
    console.log(`[${sessionId}] disconnected: ${reason}`);
    await notifyDjango(sessionId, 'disconnected', String(reason || ''));

    try {
      client.destroy();
    } catch (_) {
      // ignore
    }

    sessions.delete(sessionId);
  });

  // Initialize the client (launches Chromium)
  client.initialize();
  record.initialized = true;

  return record;
}

// ───────────────────── Django → Node endpoints ─────────────────────

// Create/ensure session
app.post('/node/session', async (req, res) => {
  try {
    // Django sends { session_hint: "<uuid>" }
    const hint = (req.body && req.body.session_hint) || null;

    // For a single-user project we can just use the hint as the session id,
    // falling back to a fixed default if hint is missing.
    const sessionId = hint || 'default-session';

    const rec = await ensureSession(sessionId);

    return res.json({ session_id: sessionId, state: rec.state });
  } catch (e) {
    console.log(`[POST /node/session] error: ${e.message}`);
    return res.status(500).json({ error: 'internal_error' });
  }
});

// Get current QR
app.get('/node/session/:id/qr', (req, res) => {
  const sessionId = req.params.id;
  const rec = sessions.get(sessionId);

  if (!rec) {
    return res.status(404).json({ error: 'session_not_found' });
  }

  // If already authenticated/ready, there is no QR to show
  if (rec.state === 'authenticated' || rec.state === 'ready') {
    return res.json({ qr: null });
  }

  return res.json({ qr: rec.qrDataUrl || null });
});

// Check session status (mostly for debugging)
app.get('/node/session/:id/status', (req, res) => {
  const sessionId = req.params.id;
  const rec = sessions.get(sessionId);

  if (!rec) {
    return res.status(404).json({ error: 'session_not_found' });
  }

  return res.json({ session_id: sessionId, state: rec.state });
});

// Get messages for a specific chat
app.get('/node/session/:id/chat/:chatId/messages', async (req, res) => {
  const sessionId = req.params.id;
  const chatId = req.params.chatId;

  const rec = sessions.get(sessionId);
  if (!rec) {
    return res.status(404).json({ error: 'session_not_found' });
  }

  const client = rec.client;
  try {
    const chat = await client.getChatById(chatId);
    // You can tune this: last 30 messages, for example
    const msgs = await chat.fetchMessages({ limit: 30 });

    const nowSeconds = Math.floor(Date.now() / 1000);
    const messagesPayload = [];

    for (const m of msgs) {
      let displayName = m.from;
      try {
        const contact = await m.getContact();
        displayName =
          contact.pushname ||
          contact.name ||
          contact.shortName ||
          contact.number ||
          m.from;
      } catch (e) {
        // ignore, fall back to m.from
      }

      messagesPayload.push({
        wa_msg_id: m.id && m.id._serialized ? m.id._serialized : m.id,
        sender: displayName,
        body: m.body,
        from_me: m.fromMe || false,
        msg_ts: m.timestamp ? m.timestamp : nowSeconds,
      });
    }

    return res.json({ chat_id: chatId, messages: messagesPayload });
  } catch (e) {
    console.log(`[${sessionId}] get chat messages error for ${chatId}: ${e.message}`);
    return res.status(500).json({ error: 'fetch_failed' });
  }
});

// Send a message to a specific chat
app.post('/node/session/:id/chat/:chatId/send', async (req, res) => {
  const sessionId = req.params.id;
  const chatId = req.params.chatId;
  const body = (req.body && req.body.body) || '';

  if (!body) {
    return res.status(400).json({ error: 'empty_body' });
  }

  const rec = sessions.get(sessionId);
  if (!rec) {
    return res.status(404).json({ error: 'session_not_found' });
  }

  const client = rec.client;
  try {
    const chat = await client.getChatById(chatId);
    await chat.sendMessage(body);
    console.log(`[${sessionId}] sent message to ${chatId}: ${body}`);
    return res.json({ ok: true });
  } catch (e) {
    console.log(`[${sessionId}] send message error for ${chatId}: ${e.message}`);
    return res.status(500).json({ error: 'send_failed' });
  }
});


// ───────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`Node server on http://localhost:${PORT}`);
});
