"use strict";

const $ = (id) => document.getElementById(id);
const taskEl = $("task");
const startUrlEl = $("startUrl");
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
const urlBar = $("urlBar");
const captureBtn = $("captureBtn");
const shotsCard = $("shotsCard");
const shotsGrid = $("shotsGrid");
const shot = $("shot");
const shotEmpty = $("shotEmpty");

let lastState = "idle";
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
  });
  if (!res.ok) {
    alert("Failed to start: " + (res.error || "unknown"));
    startBtn.disabled = false;
  }
};

stopBtn.onclick = () => api("/api/stop");

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

let renderedShots = 0;
function renderShots(shots) {
  shots = shots || [];
  if (shots.length === renderedShots) return;  // only re-render on change
  renderedShots = shots.length;
  shotsCard.hidden = shots.length === 0;
  shotsGrid.innerHTML = "";
  for (const s of shots) {
    const a = document.createElement("a");
    a.href = s.url;
    a.download = s.filename;
    a.title = s.filename;
    a.className = "shot-thumb";
    const img = document.createElement("img");
    img.src = s.url;
    img.alt = s.filename;
    a.appendChild(img);
    shotsGrid.appendChild(a);
  }
}

toggleBtn.onclick = async () => {
  // aiEnabled true  -> we want to PAUSE (take over)
  // aiEnabled false -> we want to RESUME (hand back to AI)
  await api(aiEnabled ? "/api/pause" : "/api/resume");
};

const KIND_LABEL = {
  think: "💭", action: "▶", result: "✓", manual: "✋",
  done: "★", error: "✕", info: "·", file: "📄",
};

function renderLogs(logs) {
  // Re-render only when there's something new (cheap: compare last ts).
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

function applyState(s) {
  lastState = s.state;
  aiEnabled = s.ai_enabled;

  stateChip.textContent = s.state;
  stateChip.className = "state-chip " + s.state;

  const active = s.state === "running" || s.state === "paused";
  startBtn.disabled = active;
  stopBtn.disabled = !active;
  toggleBtn.disabled = !active;

  // Takeover toggle reflects current mode
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
  urlBar.textContent = s.url || "";

  if (s.state === "done" && s.result) {
    resultCard.hidden = false;
    resultText.textContent = s.result;
  } else {
    resultCard.hidden = true;
  }

  // Data / export card: manual export buttons appear as soon as rows are
  // collected (works even after stop/pause); a Download button appears once a
  // file is written.
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

// Live preview: swap the JPEG with a cache-busting query.
function refreshShot() {
  if (lastState === "idle") return;
  const img = new Image();
  img.onload = () => {
    shot.src = img.src;
    shot.style.display = "block";
    shotEmpty.style.display = "none";
  };
  img.onerror = () => {};  // 204 / mid-navigation — let the Image be GC'd
  img.src = "/api/screenshot?t=" + Date.now();
}

setInterval(poll, 1000);
setInterval(refreshShot, 1500);
poll();
