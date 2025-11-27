const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const WebSocket = require('ws');

const PORT = process.env.CHATBOT_PORT || 4000;
const ORIGINS = (process.env.ALLOWED_ORIGINS || 'http://localhost:3000,http://127.0.0.1:3000').split(',');

const app = express();
app.use(cors({ origin: ORIGINS }));

const server = app.listen(PORT, () => {
  console.log(`[chatbot-server] Listening on http://localhost:${PORT}`);
});

const wss = new WebSocket.Server({ server });
let child = null;
let buffer = '';

function startCli() {
  const cmd = 'npx';
  const args = (process.env.CHATBOT_NPX_ARGS || 'akshit-sharma-cli --chatbot').split(' ');
  child = spawn(cmd, args, { shell: true, env: process.env });

  console.log('[chatbot-server] Spawned CLI with pid', child.pid);

  child.stdout.on('data', (data) => {
    const text = data.toString();
    buffer += text;
    broadcast(text);
  });
  child.stderr.on('data', (data) => {
    const text = data.toString();
    buffer += text;
    broadcast(text);
  });
  child.on('exit', (code) => {
    console.log('[chatbot-server] CLI exited with code', code);
    broadcast(`\n[chatbot-server] CLI exited (${code}). Restarting…\n`);
    setTimeout(startCli, 1000);
  });
}

function broadcast(text) {
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(text);
    }
  });
}

wss.on('connection', (ws) => {
  console.log('[chatbot-server] WebSocket client connected');
  if (buffer) ws.send(buffer);
  ws.on('message', (msg) => {
    const line = msg.toString();
    if (child && child.stdin && child.stdin.writable) {
      child.stdin.write(line + '\n');
    }
  });
});

app.get('/status', (req, res) => {
  res.json({ running: !!(child && child.pid), pid: child ? child.pid : null });
});

process.on('SIGINT', () => {
  console.log('[chatbot-server] SIGINT received, shutting down...');
  try { child && child.kill(); } catch { }
  server.close(() => process.exit(0));
});

startCli();

module.exports = { app, server };