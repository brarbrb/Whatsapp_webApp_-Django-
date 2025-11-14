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
const DJANGO_SECRET = process.env.DJANGO_SECRET || '';

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

// ───────────────────── helpers: unread scraping callbacks ─────────────────────

// Send one batch of unread messages to Django
async function postUnreadBatch(sessionId, jobId, messages) {
  if (!messages || messages.length === 0) return;

  try {
    await axios.post(
      `${DJANGO_INGEST_BASE}unread/${sessionId}/batch`,
      {
        job_id: jobId,
        messages,
      },
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
      `[${sessionId}] postUnreadBatch failed for job ${jobId}: ${e.message}`
    );
    // it's ok to fail occasionally; Django side is idempotent
  }
}

// Send final scrape job state to Django
async function postScrapeState(sessionId, payload) {
  try {
    await axios.post(
      `${DJANGO_INGEST_BASE}scrape/${sessionId}/state`,
      payload,
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
      `[${sessionId}] postScrapeState failed for job ${payload.job_id}: ${e.message}`
    );
  }
}

// ───────────────────── core: scrape unread chats/messages ─────────────────────

// This runs once per "ready" event (or more; Django handles idempotency).
async function scrapeUnreadForSession(sessionId, client, record) {
  // If you really want to guard against double-start, uncomment:
  // if (record.scrapingJobId) {
  //   console.log(`[${sessionId}] scrape already running with job ${record.scrapingJobId}, skipping`);
  //   return;
  // }

  const jobId = randomUUID();
  record.scrapingJobId = jobId;

  console.log(`[${sessionId}] starting unread scrape job ${jobId}`);

  let numChats = 0;
  let numMsgs = 0;

  try {
    const chats = await client.getChats();

    for (const chat of chats) {
      // Skip archives / groups / statuses if you want; for now we only check unreadCount
      if (!chat.unreadCount || chat.unreadCount <= 0) continue;

      numChats += 1;

      const unreadLimit = chat.unreadCount;

      let messages = [];
      try {
        // whatsapp-web.js returns the last N messages, including unread ones.
        messages = await chat.fetchMessages({ limit: unreadLimit });
      } catch (e) {
        console.log(
          `[${sessionId}] failed to fetch messages for chat ${chat.id._serialized}: ${e.message}`
        );
        continue;
      }

      // Filter to messages we care about:
      // - not status messages
      // - not from me (optional; depends on your UX)
      const filtered = messages.filter((m) => {
        if (m.isStatus) return false;
        // you can also check m.ack or m.id.fromMe; basic version:
        return true;
      });

      if (filtered.length === 0) continue;

      const batchPayload = filtered.map((m) => ({
        wa_msg_id: m.id && m.id._serialized ? m.id._serialized : m.id,
        chat_id: chat.id && chat.id._serialized ? chat.id._serialized : chat.id,
        sender: m.from,
        body: m.body,
        // timestamp: seconds since epoch (what you'll parse in Django)
        msg_ts: m.timestamp ? m.timestamp : Math.floor(Date.now() / 1000),
      }));

      numMsgs += batchPayload.length;

      // Send in smaller chunks so we don't exceed body size
      const chunkSize = 50;
      for (let i = 0; i < batchPayload.length; i += chunkSize) {
        const chunk = batchPayload.slice(i, i + chunkSize);
        await postUnreadBatch(sessionId, jobId, chunk);
      }
    }

    // All done: inform Django
    await postScrapeState(sessionId, {
      job_id: jobId,
      status: 'done',
      num_chats: numChats,
      num_msgs: numMsgs,
      error: '',
    });

    console.log(
      `[${sessionId}] unread scrape job ${jobId} done: chats=${numChats}, msgs=${numMsgs}`
    );
  } catch (err) {
    console.log(
      `[${sessionId}] scrapeUnread error in job ${jobId}: ${err.message}`
    );

    await postScrapeState(sessionId, {
      job_id: jobId,
      status: 'error',
      num_chats: numChats,
      num_msgs: numMsgs,
      error: String(err.message || err),
    });
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

// ───────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`Node server on http://localhost:${PORT}`);
});
