// Persistent CDP daemon: opens ONE websocket to Edge (single remote-debugging consent),
// then serves commands over local HTTP so we never trigger the permission dialog again.
import http from "node:http";
import { readFileSync, writeFileSync } from "node:fs";

const USER_DATA = `${process.env.LOCALAPPDATA}\\Microsoft\\Edge\\User Data`;

function wsUrl() {
  const raw = readFileSync(`${USER_DATA}\\DevToolsActivePort`, "utf8").trim().split(/\r?\n/);
  return `ws://127.0.0.1:${raw[0].trim()}${raw[1].trim()}`;
}

let ws = null;
let seq = 0;
const pending = new Map();
let attached = new Map(); // targetId -> sessionId

function rawSend(obj) {
  ws.send(JSON.stringify(obj));
}

function call(method, params = {}, sessionId) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`timeout: ${method}`));
    }, 30000);
    pending.set(id, { resolve, reject, timer });
    const msg = { id, method, params };
    if (sessionId) msg.sessionId = sessionId;
    rawSend(msg);
  });
}

function connectOnce() {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(wsUrl());
    socket.addEventListener("open", () => resolve(socket), { once: true });
    socket.addEventListener("error", () => reject(new Error("connect failed")), { once: true });
  });
}

async function ensure() {
  if (ws && ws.readyState === 1) return ws;
  let lastErr;
  for (let i = 0; i < 120; i++) {
    try {
      ws = await connectOnce();
      ws.addEventListener("message", onMessage);
      ws.addEventListener("close", () => { ws = null; attached.clear(); });
      console.log("CDP connected");
      return ws;
    } catch (e) {
      lastErr = e;
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
  throw lastErr;
}

function onMessage(ev) {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) {
    const { resolve, reject, timer } = pending.get(msg.id);
    clearTimeout(timer);
    pending.delete(msg.id);
    msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
  }
}

async function sessionFor(targetId) {
  if (attached.has(targetId)) return attached.get(targetId);
  const { sessionId } = await call("Target.attachToTarget", { targetId, flatten: true });
  await call("Page.enable", {}, sessionId);
  await call("Runtime.enable", {}, sessionId);
  attached.set(targetId, sessionId);
  return sessionId;
}

async function pages() {
  const { targetInfos } = await call("Target.getTargets");
  return targetInfos.filter((t) => t.type === "page");
}

async function findPage(match) {
  const list = await pages();
  if (!match) return list[0];
  return list.find((p) => p.url.includes(match)) || null;
}

function send(res, code, obj) {
  const body = JSON.stringify(obj, null, 2);
  res.writeHead(code, { "Content-Type": "application/json; charset=utf-8" });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve) => {
    let data = "";
    req.on("data", (c) => (data += c));
    req.on("end", () => {
      try { resolve(data ? JSON.parse(data) : {}); } catch { resolve({}); }
    });
  });
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://127.0.0.1");
    if (url.pathname === "/ping") return send(res, 200, { ok: true, connected: !!ws && ws.readyState === 1 });

    if (url.pathname === "/pages") {
      await ensure();
      return send(res, 200, { pages: await pages() });
    }

    if (url.pathname === "/open" && req.method === "POST") {
      const { url: target } = await readBody(req);
      await ensure();
      const { targetId } = await call("Target.createTarget", { url: target || "about:blank" });
      await call("Target.activateTarget", { targetId });
      await sessionFor(targetId);
      return send(res, 200, { targetId });
    }

    if (url.pathname === "/activate" && req.method === "POST") {
      const { targetId } = await readBody(req);
      await ensure();
      await call("Target.activateTarget", { targetId });
      return send(res, 200, { ok: true });
    }

    if (url.pathname === "/eval" && req.method === "POST") {
      const { match, targetId, expression } = await readBody(req);
      await ensure();
      const page = targetId ? { targetId } : await findPage(match);
      if (!page) return send(res, 404, { error: "no matching page" });
      const sid = await sessionFor(page.targetId);
      const r = await call("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true }, sid);
      return send(res, 200, { result: r.result ? r.result.value : null, exception: r.exceptionDetails || null });
    }

    if (url.pathname === "/shot" && req.method === "POST") {
      const { match, targetId, file } = await readBody(req);
      await ensure();
      const page = targetId ? { targetId } : await findPage(match);
      if (!page) return send(res, 404, { error: "no matching page" });
      const sid = await sessionFor(page.targetId);
      const { data } = await call("Page.captureScreenshot", { format: "png" }, sid);
      writeFileSync(file, Buffer.from(data, "base64"));
      return send(res, 200, { saved: file });
    }

    if (url.pathname === "/navigate" && req.method === "POST") {
      const { match, targetId, url: target } = await readBody(req);
      await ensure();
      const page = targetId ? { targetId } : await findPage(match);
      if (!page) return send(res, 404, { error: "no matching page" });
      const sid = await sessionFor(page.targetId);
      await call("Page.navigate", { url: target }, sid);
      return send(res, 200, { ok: true });
    }

    send(res, 404, { error: "not found" });
  } catch (e) {
    send(res, 500, { error: String(e && e.message ? e.message : e) });
  }
});

const PORT = Number(process.env.CDP_DAEMON_PORT || 9400);
server.listen(PORT, "127.0.0.1", () => console.log("daemon on 127.0.0.1:" + PORT));

// heartbeat keeps the single socket alive
setInterval(() => {
  if (ws && ws.readyState === 1) call("Browser.getVersion").catch(() => {});
}, 20000);
