"use strict";

const $ = (id) => document.getElementById(id);
// sidebar / chat
const newChatBtn = $("newChatBtn");
const threadList = $("threadList");
const chatTitle = $("chatTitle");
const stateChip = $("stateChip");
const transcript = $("transcript");
const emptyHint = $("emptyHint");
const artifacts = $("artifacts");
const startUrlEl = $("startUrl");
const toggleBtn = $("toggleBtn");
const stopBtn = $("stopBtn");
const taskEl = $("task");
const startBtn = $("startBtn");
// preview
const goUrl = $("goUrl");
const goBtn = $("goBtn");
const modeBadge = $("modeBadge");
const screenWrap = $("screenWrap");
const screen = $("screen");
const screenEmpty = $("screenEmpty");
const viewOnly = $("viewOnly");

let activeThread = "";
let threads = [];
let historyTurns = [];
let lastStatus = null;
let aiEnabled = true;
let lastTerminal = true;  // was the session terminal (idle/done) on the previous poll

function newThreadId() {
  return "chat-" + Date.now() + "-" + Math.floor(Math.random() * 1e4);
}

async function api(path, body, method) {
  const opt = { method: method || "POST", headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opt.body = JSON.stringify(body);
  const r = await fetch(path, opt);
  return r.json().catch(() => ({}));
}

// ---- threads sidebar ---------------------------------------------------------
async function loadThreads() {
  try {
    const r = await (await fetch("/api/threads")).json();
    threads = r.threads || [];
  } catch (_) { threads = []; }
  renderThreads();
}

function renderThreads() {
  threadList.innerHTML = "";
  const known = threads.some((t) => t.thread_id === activeThread);
  const list = known ? threads.slice() : [{ thread_id: activeThread, title: "New chat", count: 0, _new: true }, ...threads];
  for (const t of list) {
    const row = document.createElement("div");
    row.className = "thread-row" + (t.thread_id === activeThread ? " active" : "");
    const title = document.createElement("div");
    title.className = "thread-title";
    title.textContent = t.title || "New chat";
    title.title = t.title || "";
    title.onclick = () => switchThread(t.thread_id);
    row.appendChild(title);
    if (!t._new) {
      const del = document.createElement("button");
      del.className = "thread-del";
      del.textContent = "🗑";
      del.title = "Delete chat";
      del.onclick = async (e) => {
        e.stopPropagation();
        if (!confirm("Delete this chat and its memory?")) return;
        await api("/api/thread?id=" + encodeURIComponent(t.thread_id), undefined, "DELETE");
        if (t.thread_id === activeThread) newChat();
        else loadThreads();
      };
      row.appendChild(del);
    }
    threadList.appendChild(row);
  }
}

function newChat() {
  activeThread = newThreadId();
  localStorage.setItem("ba_activeThread", activeThread);
  historyTurns = [];
  renderThreads();
  renderTranscript();
  taskEl.focus();
}
newChatBtn.onclick = newChat;

async function switchThread(id) {
  if (id === activeThread) return;
  activeThread = id;
  localStorage.setItem("ba_activeThread", activeThread);
  await loadHistory(id);
  renderThreads();
}

async function loadHistory(id) {
  try {
    const r = await (await fetch("/api/thread/history?id=" + encodeURIComponent(id))).json();
    historyTurns = r.turns || [];
  } catch (_) { historyTurns = []; }
  renderTranscript();
}

// ---- transcript (chat bubbles) ----------------------------------------------
const KIND_LABEL = {
  think: "💭", action: "▶", result: "✓", manual: "✋",
  done: "★", error: "✕", info: "·", file: "📄",
};

function bubble(role, html) {
  const b = document.createElement("div");
  b.className = "bubble " + role;
  b.innerHTML = html;
  return b;
}
function esc(s) {
  return (s || "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function liveSteps(logs) {
  const wrap = document.createElement("div");
  wrap.className = "steps";
  for (const l of logs) {
    if (!["think", "action", "result", "manual", "error", "file", "done"].includes(l.kind)) continue;
    const d = document.createElement("div");
    d.className = "step " + l.kind;
    d.textContent = (KIND_LABEL[l.kind] || "·") + " " + l.text;
    wrap.appendChild(d);
  }
  return wrap;
}

function renderTranscript() {
  const live = lastStatus && lastStatus.thread_id === activeThread &&
    (lastStatus.state === "running" || lastStatus.state === "paused") && lastStatus.task;

  const firstTitle = historyTurns.length ? historyTurns[0].task : (live ? lastStatus.task : "New chat");
  chatTitle.textContent = (firstTitle || "New chat").slice(0, 80);

  transcript.innerHTML = "";
  if (!historyTurns.length && !live) {
    transcript.appendChild(emptyHint);
    emptyHint.style.display = "block";
    return;
  }
  emptyHint.style.display = "none";

  for (const t of historyTurns) {
    transcript.appendChild(bubble("user", esc(t.task)));
    transcript.appendChild(bubble("ai", esc(t.result)));
  }

  if (live) {
    transcript.appendChild(bubble("user", esc(lastStatus.task)));
    const ai = bubble("ai working", "");
    const head = document.createElement("div");
    head.className = "working-head";
    head.textContent = lastStatus.state === "paused" ? "⏸ paused — your turn in the preview" : "▶ working…";
    ai.appendChild(head);
    ai.appendChild(liveSteps(lastStatus.logs || []));
    transcript.appendChild(ai);
  }
  transcript.scrollTop = transcript.scrollHeight;
}

// ---- artifacts (export / screenshots for the active run) ---------------------
function renderArtifacts(s) {
  const parts = [];
  if (s.export) {
    parts.push(`<a class="dl-btn" href="${s.export.url}" download="${esc(s.export.filename)}">⬇ ${esc(s.export.filename)} (${s.export.rows} rows)</a>`);
  } else if ((s.data_rows || 0) > 0) {
    parts.push(`<span class="muted">${s.data_rows} rows collected</span>
      <button class="mini-btn" onclick="doExport('csv')">⬇ CSV</button>
      <button class="mini-btn" onclick="doExport('xlsx')">⬇ Excel</button>`);
  }
  let shotsHtml = "";
  for (const sh of (s.shots || [])) {
    shotsHtml += `<a class="shot-thumb" href="${sh.url}" download="${esc(sh.filename)}" title="${esc(sh.filename)}"><img src="${sh.url}" alt=""></a>`;
  }
  if (shotsHtml) parts.push(`<div class="shots-grid">${shotsHtml}</div>`);

  if (parts.length) { artifacts.hidden = false; artifacts.innerHTML = parts.join(" "); }
  else { artifacts.hidden = true; artifacts.innerHTML = ""; }
}

// ---- actions -----------------------------------------------------------------
startBtn.onclick = async () => {
  const task = taskEl.value.trim();
  if (!task) { taskEl.focus(); return; }
  startBtn.disabled = true;
  const res = await api("/api/start", {
    task,
    start_url: startUrlEl.value.trim() || null,
    thread_id: activeThread,
  });
  if (!res.ok) { alert("Failed to start: " + (res.error || "unknown")); startBtn.disabled = false; }
  else { taskEl.value = ""; autoGrow(); }
};
taskEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); startBtn.onclick(); }
});
function autoGrow() { taskEl.style.height = "auto"; taskEl.style.height = Math.min(taskEl.scrollHeight, 160) + "px"; }
taskEl.addEventListener("input", autoGrow);

stopBtn.onclick = () => api("/api/stop");
toggleBtn.onclick = () => api(aiEnabled ? "/api/pause" : "/api/resume");

window.doExport = async function (fmt) {
  const res = await api("/api/export?fmt=" + fmt);
  if (!res.ok) alert("Export failed: " + (res.error || "no data"));
  else poll();
};

goBtn.onclick = async () => {
  const url = goUrl.value.trim();
  if (!url) return;
  const res = await api("/api/goto", { url });
  if (!res.ok) alert("Go failed: " + (res.error || "unknown"));
};
goUrl.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.stopPropagation(); goBtn.onclick(); } });

// ---- live interactive browser (WebSocket stream + input forwarding) ----------
let ws = null, frameW = 1280, frameH = 800, interactive = false, wsConnecting = false;

function connectWS() {
  if (wsConnecting) return;
  wsConnecting = true;
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(proto + "//" + location.host + "/ws/screen");
  ws.onopen = () => { wsConnecting = false; };
  ws.onmessage = (ev) => {
    let m; try { m = JSON.parse(ev.data); } catch (_) { return; }
    if (m.t !== "frame") return;
    frameW = m.w; frameH = m.h; interactive = !!m.interactive;
    screen.src = "data:image/jpeg;base64," + m.img;
    screen.style.display = "block"; screenEmpty.style.display = "none";
    paintMode();
  };
  ws.onclose = () => { wsConnecting = false; setTimeout(connectWS, 1000); };
  ws.onerror = () => {};
}
function wsSend(o) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }

function paintMode() {
  if (interactive) {
    modeBadge.textContent = "interactive"; modeBadge.className = "mode-badge live";
    viewOnly.hidden = true; screen.style.cursor = "crosshair";
  } else {
    modeBadge.textContent = "view-only (AI)"; modeBadge.className = "mode-badge view";
    viewOnly.hidden = false; screen.style.cursor = "default";
  }
}
function pageCoords(e) {
  const r = screen.getBoundingClientRect();
  const nw = screen.naturalWidth || frameW, nh = screen.naturalHeight || frameH;
  const x = (e.clientX - r.left) / r.width * nw, y = (e.clientY - r.top) / r.height * nh;
  return { x: Math.max(0, Math.min(x, nw)), y: Math.max(0, Math.min(y, nh)) };
}
screen.addEventListener("click", (e) => { if (!interactive) return; const p = pageCoords(e); wsSend({ t: "click", x: p.x, y: p.y, button: "left" }); screenWrap.focus({ preventScroll: true }); });
screen.addEventListener("dblclick", (e) => { if (!interactive) return; const p = pageCoords(e); wsSend({ t: "click", x: p.x, y: p.y, button: "left", clicks: 2 }); });
screen.addEventListener("contextmenu", (e) => { e.preventDefault(); if (!interactive) return; const p = pageCoords(e); wsSend({ t: "click", x: p.x, y: p.y, button: "right" }); });
let scrollDX = 0, scrollDY = 0, scrollTimer = null;
function flushScroll() {
  scrollTimer = null;
  if (scrollDX || scrollDY) { wsSend({ t: "scroll", dx: scrollDX, dy: scrollDY }); scrollDX = scrollDY = 0; }
}
screen.addEventListener("wheel", (e) => {
  if (!interactive) return; e.preventDefault();
  let dx = e.deltaX, dy = e.deltaY;
  if (e.deltaMode === 1) { dx *= 16; dy *= 16; } else if (e.deltaMode === 2) { dx *= 800; dy *= 800; }
  // Coalesce rapid wheel events into one message per frame so we don't flood the
  // server (and the page scrolls in fewer, bigger, smoother steps).
  scrollDX += dx; scrollDY += dy;
  if (!scrollTimer) scrollTimer = setTimeout(flushScroll, 30);
}, { passive: false });
const SPECIAL_KEYS = { Enter: 1, Backspace: 1, Escape: 1, Delete: 1, Home: 1, End: 1, PageUp: 1, PageDown: 1, ArrowUp: 1, ArrowDown: 1, ArrowLeft: 1, ArrowRight: 1 };
screenWrap.addEventListener("keydown", (e) => {
  if (!interactive) return;
  const a = document.activeElement;
  if (a === goUrl || a === taskEl || a === startUrlEl) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (SPECIAL_KEYS[e.key]) { wsSend({ t: "key", key: e.key }); e.preventDefault(); }
  else if (e.key.length === 1) { wsSend({ t: "text", text: e.key }); e.preventDefault(); }
});

// ---- status polling ----------------------------------------------------------
function applyState(s) {
  lastStatus = s;
  aiEnabled = s.ai_enabled;
  stateChip.textContent = s.state;
  stateChip.className = "state-chip " + s.state;

  const active = s.state === "running" || s.state === "paused";
  startBtn.disabled = active;
  stopBtn.disabled = !active;
  toggleBtn.disabled = !active;
  const manualMode = active && !aiEnabled;
  toggleBtn.textContent = manualMode ? "▶ Resume AI" : "✋ Take Over";
  toggleBtn.classList.toggle("manual", manualMode);

  if (document.activeElement !== goUrl && !goUrl.value) goUrl.placeholder = s.url || "Go to URL…  (https://…)";

  renderArtifacts(s);

  // When a run for the active thread finishes, refresh the transcript + sidebar
  // so the new turn shows as a saved bubble and the chat title updates.
  const terminal = !(s.state === "running" || s.state === "paused");
  if (s.thread_id === activeThread && terminal && !lastTerminal) {
    loadHistory(activeThread);
    loadThreads();
  } else if (s.thread_id === activeThread) {
    renderTranscript();  // live update of working steps
  }
  lastTerminal = terminal;
}

async function poll() {
  try { applyState(await (await fetch("/api/status")).json()); } catch (_) {}
}

// ---- init --------------------------------------------------------------------
activeThread = localStorage.getItem("ba_activeThread") || newThreadId();
localStorage.setItem("ba_activeThread", activeThread);
loadThreads();
loadHistory(activeThread);
setInterval(poll, 1000);
poll();
connectWS();
autoGrow();
