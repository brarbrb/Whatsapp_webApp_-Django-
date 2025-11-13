// Simple Node service for whatsapp-web.js with REST endpoints for Django
// Endpoints:
//  POST /node/session              -> create/ensure a session { session_id, state }
//  GET  /node/session/:id/qr       -> { qr: <dataURL or null> }
//  GET  /node/session/:id/status   -> { state, session_id }
//
// On events (authenticated/ready/disconnected) it POSTs to:
//  ${DJANGO_CALLBACK_URL}${session_id}/state  with header X-Node-Secret

const express = require('express');
const axios = require('axios');
const qrcode = require('qrcode');
const { Client, LocalAuth } = require('whatsapp-web.js');
const path = require('path');

const PORT = process.env.PORT || 3001;
const DJANGO_CALLBACK_URL = process.env.DJANGO_CALLBACK_URL || 'http://localhost:8000/ingest/session/';
const DJANGO_SECRET = process.env.DJANGO_SECRET || '';

const app = express();
app.use(express.json());

// In-memory session registry (ok for a small project)
const sessions = new Map(); // sessionId -> { client, state, qrDataUrl, initialized }

// helper: notify Django about state changes
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

// create or get an existing WA client for a session
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

  const record = { client, state: 'pending', qrDataUrl: null, initialized: false };
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
    await notifyDjango(sessionId, 'ready');
  });

  client.on('disconnected', async (reason) => {
    record.state = 'disconnected';
    record.qrDataUrl = null;
    console.log(`[${sessionId}] disconnected: ${reason}`);
    await notifyDjango(sessionId, 'disconnected', String(reason || ''));
    try {
      client.destroy();
    } catch (_) {}
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
    // Django can send { session_hint: "<uuid>" } or anything unique
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
  if (!rec) return res.status(404).json({ error: 'session_not_found' });

  // If already authenticated/ready, there is no QR to show
  if (rec.state === 'authenticated' || rec.state === 'ready') {
    return res.json({ qr: null });
  }
  return res.json({ qr: rec.qrDataUrl || null });
});

// Check session status
app.get('/node/session/:id/status', (req, res) => {
  const sessionId = req.params.id;
  const rec = sessions.get(sessionId);
  if (!rec) return res.status(404).json({ error: 'session_not_found' });
  return res.json({ session_id: sessionId, state: rec.state });
});

// ───────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`Node server on http://localhost:${PORT}`);
});
