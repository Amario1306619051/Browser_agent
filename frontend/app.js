"use strict";

const $ = (id) => document.getElementById(id);
const taskEl = $("task");
const startUrlEl = $("startUrl");
const threadId = $("threadId");
const threadBadge = $("threadBadge");
const startBtn = $("startBtn");
const stopBtn = $("stopBtn");
const toggleBtn = $("toggleBtn");
const stateChip = $("stateChip");
const modeDot = $("modeDot");
const logEl = $("log");
const stepCount = $("stepCount");
const resultCard = $("resultCard");
const resultText = $("resultText");
const exportCard = $("exportCard");
const exportInfo = $("exportInfo");
const exportBtns = $("exportBtns");
const exportCsvBtn = $("exportCsvBtn");
const exportXlsxBtn = $("exportXlsxBtn");
const downloadLink = $("downloadLink");
const captureBtn = $("captureBtn");
const shotsCard = $("shotsCard");
const shotsGrid = $("shotsGrid");
// interactive browser preview
const goUrl = $("goUrl");
const goBtn = $("goBtn");
const modeBadge = $("modeBadge");
const screenWrap = $("screenWrap");
const screen = $("screen");
const screenEmpty = $("screenEmpty");
const viewOnly = $("viewOnly");

let aiEnabled = true;
let renderedLogTs = 0;

async function api(path, body) {
  const opt = { method: "POST", headers: { "Content-Type": "application/json" } };
  if (body) opt.body = JSON.stringify(body);
  const r = await fetch(path, body !== undefined ? opt : { method: "POST" });
  return r.json().catch(() => ({}));
}

startBtn.onclick = async () => {
  const task = taskEl.value.trim();
  if (!task) { taskEl.focus(); return; }
  startBtn.disabled = true;
  const res = await api("/api/start", {
    task,
    start_url: startUrlEl.value.trim() || null,
    thread_id: threadId.value.trim() || null,
  });
  if (!res.ok) {
    alert("Failed to start: " + (res.error || "unknown"));
    startBtn.disabled = false;
  }
};

// ---- thread memory indicator -------------------------------------------------
function setThreadBadge(count) {
  if (count === null) { threadBadge.textContent = "no memory"; threadBadge.className = "thread-badge none"; }
  else if (count === 0) { threadBadge.textContent = "new thread"; threadBadge.className = "thread-badge new"; }
  else { threadBadge.textContent = "🧠 " + count + " remembered"; threadBadge.className = "thread-badge has"; }
}
async function checkThread() {
  const id = threadId.value.trim();
  if (!id) { setThreadBadge(null); return; }
  try {
    const r = await (await fetch("/api/thread?id=" + encodeURIComponent(id))).json();
    setThreadBadge(r.count);
  } catch (_) {}
}
let threadTimer = null;
threadId.addEventListener("input", () => {
  localStorage.setItem("ba_threadId", threadId.value);
  clearTimeout(threadTimer);
  threadTimer = setTimeout(checkThread, 400);
});
threadId.value = localStorage.getItem("ba_threadId") || "";

stopBtn.onclick = () => api("/api/stop");

toggleBtn.onclick = () => api(aiEnabled ? "/api/pause" : "/api/resume");

async function doExport(fmt) {
  const res = await api("/api/export?fmt=" + fmt);
  if (!res.ok) alert("Export failed: " + (res.error || "no data"));
  else poll();
}
exportCsvBtn.onclick = () => doExport("csv");
exportXlsxBtn.onclick = () => doExport("xlsx");

captureBtn.onclick = async () => {
  const res = await api("/api/capture");
  if (!res.ok) alert("Capture failed: " + (res.error || "no browser running"));
  else poll();
};

goBtn.onclick = async () => {
  const url = goUrl.value.trim();
  if (!url) return;
  const res = await api("/api/goto", { url });
  if (!res.ok) alert("Go failed: " + (res.error || "unknown"));
};
goUrl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.stopPropagation(); goBtn.onclick(); }
});

// ---- live interactive browser (WebSocket stream + input forwarding) ----------
let ws = null;
let frameW = 1280, frameH = 800;
let interactive = false;
let wsConnecting = false;

function connectWS() {
  if (wsConnecting) return;  // guard against double-connect / reconnect storms
  wsConnecting = true;
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(proto + "//" + location.host + "/ws/screen");
  ws.onopen = () => { wsConnecting = false; };
  ws.onmessage = (ev) => {
    let m;
    try { m = JSON.parse(ev.data); } catch (_) { return; }
    if (m.t !== "frame") return;
    frameW = m.w; frameH = m.h;
    interactive = !!m.interactive;
    screen.src = "data:image/jpeg;base64," + m.img;
    screen.style.display = "block";
    screenEmpty.style.display = "none";
    paintMode();
  };
  ws.onclose = () => { wsConnecting = false; setTimeout(connectWS, 1000); };  // auto-reconnect
  ws.onerror = () => {};  // onclose handles the reconnect
}
function wsSend(o) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }

function paintMode() {
  if (interactive) {
    modeBadge.textContent = "interactive";
    modeBadge.className = "mode-badge live";
    viewOnly.hidden = true;
    screen.style.cursor = "crosshair";
  } else {
    modeBadge.textContent = "view-only (AI)";
    modeBadge.className = "mode-badge view";
    viewOnly.hidden = false;
    screen.style.cursor = "default";
  }
}

function pageCoords(e) {
  const r = screen.getBoundingClientRect();
  const nw = screen.naturalWidth || frameW;
  const nh = screen.naturalHeight || frameH;
  const x = (e.clientX - r.left) / r.width * nw;
  const y = (e.clientY - r.top) / r.height * nh;
  return { x: Math.max(0, Math.min(x, nw)), y: Math.max(0, Math.min(y, nh)) };
}

screen.addEventListener("click", (e) => {
  if (!interactive) return;
  const p = pageCoords(e);
  wsSend({ t: "click", x: p.x, y: p.y, button: "left" });
  screenWrap.focus({ preventScroll: true });
});
screen.addEventListener("dblclick", (e) => {
  if (!interactive) return;
  const p = pageCoords(e);
  wsSend({ t: "click", x: p.x, y: p.y, button: "left", clicks: 2 });
});
screen.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  if (!interactive) return;
  const p = pageCoords(e);
  wsSend({ t: "click", x: p.x, y: p.y, button: "right" });
});
screen.addEventListener("wheel", (e) => {
  if (!interactive) return;
  e.preventDefault();
  // Normalize delta units (lines/pages → pixels) so scroll feels the same everywhere.
  let dx = e.deltaX, dy = e.deltaY;
  if (e.deltaMode === 1) { dx *= 16; dy *= 16; }
  else if (e.deltaMode === 2) { dx *= 800; dy *= 800; }
  wsSend({ t: "scroll", dx: dx, dy: dy });
}, { passive: false });

const SPECIAL_KEYS = {
  Enter: 1, Backspace: 1, Escape: 1, Delete: 1, Home: 1, End: 1,
  PageUp: 1, PageDown: 1, ArrowUp: 1, ArrowDown: 1, ArrowLeft: 1, ArrowRight: 1,
};
screenWrap.addEventListener("keydown", (e) => {
  if (!interactive) return;
  // Never steal keys from the text fields (Tab stays native too).
  const a = document.activeElement;
  if (a === goUrl || a === taskEl || a === startUrlEl) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;  // don't hijack shortcuts
  if (SPECIAL_KEYS[e.key]) { wsSend({ t: "key", key: e.key }); e.preventDefault(); }
  else if (e.key.length === 1) { wsSend({ t: "text", text: e.key }); e.preventDefault(); }
});

// ---- status polling (logs / result / export / shots / toggle) ----------------
const KIND_LABEL = {
  think: "💭", action: "▶", result: "✓", manual: "✋",
  done: "★", error: "✕", info: "·", file: "📄",
};

function renderLogs(logs) {
  const newest = logs.length ? logs[logs.length - 1].ts : 0;
  if (newest === renderedLogTs) return;
  renderedLogTs = newest;
  logEl.innerHTML = "";
  for (const l of logs) {
    const div = document.createElement("div");
    div.className = "log-line " + l.kind;
    const k = document.createElement("span");
    k.className = "k";
    k.textContent = KIND_LABEL[l.kind] || "·";
    div.appendChild(k);
    div.appendChild(document.createTextNode(l.text));
    logEl.appendChild(div);
  }
  logEl.scrollTop = logEl.scrollHeight;
}

let renderedShots = 0;
function renderShots(shots) {
  shots = shots || [];
  if (shots.length === renderedShots) return;
  renderedShots = shots.length;
  shotsCard.hidden = shots.length === 0;
  shotsGrid.innerHTML = "";
  for (const s of shots) {
    const a = document.createElement("a");
    a.href = s.url; a.download = s.filename; a.title = s.filename; a.className = "shot-thumb";
    const img = document.createElement("img");
    img.src = s.url; img.alt = s.filename;
    a.appendChild(img);
    shotsGrid.appendChild(a);
  }
}

function applyState(s) {
  aiEnabled = s.ai_enabled;
  stateChip.textContent = s.state;
  stateChip.className = "state-chip " + s.state;

  const active = s.state === "running" || s.state === "paused";
  startBtn.disabled = active;
  stopBtn.disabled = !active;
  toggleBtn.disabled = !active;

  if (aiEnabled) {
    toggleBtn.textContent = "✋ Take Over (Manual)";
    toggleBtn.classList.remove("manual");
    modeDot.className = "mode-dot ai";
  } else {
    toggleBtn.textContent = "▶ Resume AI";
    toggleBtn.classList.add("manual");
    modeDot.className = "mode-dot manual";
  }

  stepCount.textContent = active || s.step ? `(${s.step}/${s.max_steps})` : "";
  if (document.activeElement !== goUrl && !goUrl.value) goUrl.placeholder = s.url || "Go to URL…  (https://…)";

  // Keep the memory badge live for the active thread (it grows as tasks finish).
  if (s.thread_id && s.thread_id === threadId.value.trim()) setThreadBadge(s.thread_count);

  if (s.state === "done" && s.result) {
    resultCard.hidden = false;
    resultText.textContent = s.result;
  } else {
    resultCard.hidden = true;
  }

  const hasData = (s.data_rows || 0) > 0;
  if (hasData || s.export) {
    exportCard.hidden = false;
    exportBtns.style.display = hasData ? "flex" : "none";
    if (s.export) {
      downloadLink.hidden = false;
      downloadLink.href = s.export.url;
      downloadLink.textContent = "⬇ Download " + s.export.filename;
      downloadLink.setAttribute("download", s.export.filename);
      exportInfo.textContent = `${s.data_rows || s.export.rows} rows collected · ${s.export.columns.join(", ")}`;
    } else {
      downloadLink.hidden = true;
      exportInfo.textContent = `${s.data_rows} rows collected — export to download`;
    }
  } else {
    exportCard.hidden = true;
  }

  renderShots(s.shots || []);
  renderLogs(s.logs || []);
}

async function poll() {
  try {
    const s = await (await fetch("/api/status")).json();
    applyState(s);
  } catch (_) {}
}

setInterval(poll, 1000);
poll();
checkThread();
connectWS();
