import {
  env,
  AutoTokenizer,
  AutoModelForSequenceClassification,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const MODEL_ID = "finetuned-int8";
const MAX_LENGTH = 256;
const HISTORY_KEY = "ipc_prediction_history_v2";
const HISTORY_LIMIT = 30;
const TOP_P_THRESHOLD = 0.95;

// DOM
const titleInput = document.getElementById("title");
const abstractInput = document.getElementById("abstract");
const predictBtn = document.getElementById("predict-btn");
const demoBtn = document.getElementById("demo-btn");
const clearHistoryBtn = document.getElementById("clear-history-btn");
const loadModelBtn = document.getElementById("load-model-btn");
const useApiBtn = document.getElementById("use-api-btn");

const statusEl = document.getElementById("status");
const statusModelEl = document.getElementById("status-model");
const runtimeModeEl = document.getElementById("runtime-mode");
const runtimeModeBadgeEl = document.getElementById("runtime-mode-badge");
const runtimeNoteEl = document.getElementById("runtime-note");

const resultEl = document.getElementById("result");
const resultEmptyEl = document.getElementById("result-empty");
const top1El = document.getElementById("top1");
const topPListEl = document.getElementById("top-p-list");
const topPScrollEl = document.getElementById("top-p-scroll");
const topPMoreBtn = document.getElementById("top-p-more-btn");

const historyEmptyEl = document.getElementById("history-empty");
const historyWrapEl = document.getElementById("history-wrap");
const historyBodyEl = document.getElementById("history-body");

const loaderPhase = document.getElementById("loader-phase");
const loaderProgressFill = document.getElementById("loader-progress-fill");
const loaderPercent = document.getElementById("loader-percent");
const loaderLoaded = document.getElementById("loader-loaded");
const loaderTotal = document.getElementById("loader-total");
const loaderFile = document.getElementById("loader-file");

// Transformers.js env
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "/static/classifier/models/";
env.logLevel = 20;

let tokenizerPromise = null;
let modelPromise = null;
let modelReady = false;
let modelLoading = false;
let runtimeMode = "api"; // default

// ---------- UI helpers ----------
function setStatus(text) {
  statusEl.textContent = text;
}

function setModelStatus(text) {
  statusModelEl.textContent = text;
}

function setRuntimeMode(mode) {
  runtimeMode = mode;

  const label = mode === "local" ? "Browser model" : "Server API";
  runtimeModeEl.textContent = label;
  runtimeModeBadgeEl.textContent = label;

  if (mode === "local") {
    runtimeNoteEl.textContent = "Browser model is active. Predictions now run locally in this tab.";
    setStatus("Ready. Current mode: Browser model.");

    if (modelReady) {
      loadModelBtn.disabled = false;
      loadModelBtn.textContent = "Use browser model";
    }
  } else {
    runtimeNoteEl.textContent = modelReady
      ? "Server API is active. Browser model is still downloaded and can be re-enabled."
      : "Server API is active. Users with slow connection can skip the browser download.";

    setStatus("Ready. Current mode: Server API.");

    if (modelReady) {
      loadModelBtn.disabled = false;
      loadModelBtn.textContent = "Use browser model";
    }
  }
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 MB";
  const mb = bytes / (1024 * 1024);
  if (mb < 1024) return `${mb.toFixed(1)} MB`;
  return `${(mb / 1024).toFixed(2)} GB`;
}

function clampPct(x) {
  if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(100, x));
}

function updateLoaderProgress(pct, loaded = null, total = null, file = null, phase = null) {
  const safePct = clampPct(pct);
  loaderProgressFill.style.width = `${safePct}%`;
  loaderPercent.textContent = `${Math.round(safePct)}%`;

  if (loaded !== null) loaderLoaded.textContent = formatBytes(loaded);
  if (total !== null) loaderTotal.textContent = formatBytes(total);
  if (file) loaderFile.textContent = file;
  if (phase) loaderPhase.textContent = phase;
}

function updateLoaderUI(info) {
  if (!info || typeof info !== "object") return;

  if (info.status === "progress_total") {
    updateLoaderProgress(
      info.progress ?? 0,
      info.loaded ?? 0,
      info.total ?? 0,
      info.file ?? loaderFile.textContent,
      info.name ? `Loading ${info.name}` : null
    );
  } else if (info.status === "progress") {
    updateLoaderProgress(
      info.progress ?? 0,
      info.loaded ?? 0,
      info.total ?? 0,
      info.file ?? null,
      loaderPhase.textContent
    );
  } else if (info.status === "initiate") {
    if (info.file) loaderFile.textContent = info.file;
    loaderPhase.textContent = "Preparing files...";
  } else if (info.status === "done") {
    if (info.file) loaderFile.textContent = `${info.file} loaded`;
  } else if (info.status === "ready") {
    updateLoaderProgress(100, null, null, "All assets loaded.", "Browser model is ready");
  }
}

// ---------- custom cursor ----------
const cursor = document.getElementById("cursor");
const trail = document.getElementById("cursor-trail");

if (cursor && trail && window.matchMedia("(pointer:fine)").matches) {
  let targetX = window.innerWidth / 2;
  let targetY = window.innerHeight / 2;
  let trailX = targetX;
  let trailY = targetY;

  cursor.style.left = `${targetX}px`;
  cursor.style.top = `${targetY}px`;
  trail.style.left = `${trailX}px`;
  trail.style.top = `${trailY}px`;

  document.addEventListener("pointermove", (e) => {
    targetX = e.clientX;
    targetY = e.clientY;
    cursor.style.left = `${targetX}px`;
    cursor.style.top = `${targetY}px`;
  }, { passive: true });

  function animateTrail() {
    trailX += (targetX - trailX) * 0.28; // faster than before
    trailY += (targetY - trailY) * 0.28;
    trail.style.left = `${trailX}px`;
    trail.style.top = `${trailY}px`;
    requestAnimationFrame(animateTrail);
  }
  requestAnimationFrame(animateTrail);

  document.querySelectorAll("a, button, input, textarea").forEach((el) => {
    el.addEventListener("mouseenter", () => cursor.classList.add("hovering"));
    el.addEventListener("mouseleave", () => cursor.classList.remove("hovering"));
  });
}

// ---------- scroll progress + reveal ----------
const progressBar = document.getElementById("progress-bar");

window.addEventListener("scroll", () => {
  const total = document.body.scrollHeight - window.innerHeight;
  const pct = total > 0 ? (window.scrollY / total) * 100 : 0;
  progressBar.style.width = `${pct}%`;
}, { passive: true });

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) entry.target.classList.add("visible");
  });
}, { threshold: 0.12 });

document.querySelectorAll(".reveal").forEach((el) => observer.observe(el));

// ---------- starfield ----------
const canvas = document.getElementById("starfield");
const ctx = canvas.getContext("2d", { alpha: true });

let stars = [];
let shootingStars = [];
let W = 0;
let H = 0;
let pageScrollY = 0;
let starfieldActive = false;
let starfieldRAF = null;
let lastFrameTs = 0;

const STAR_DIRECTION = { x: 0.045, y: 0.18 };
const STAR_FPS = 24;
const NUM_STARS = window.innerWidth <= 768 ? 90 : 180;

function resizeCanvas() {
  W = canvas.width = window.innerWidth;
  H = canvas.height = window.innerHeight;
  stars = Array.from({ length: NUM_STARS }, () => ({
    x: Math.random() * W,
    y: Math.random() * H,
    r: Math.random() * 1.5 + 0.05,
    a: Math.random() * 0.8 + 0.06,
    speed: Math.random() * 0.022 + 0.003,
  }));
}
resizeCanvas();

window.addEventListener("resize", resizeCanvas, { passive: true });
window.addEventListener("scroll", () => {
  pageScrollY = window.scrollY;
}, { passive: true });

function wrapStar(s) {
  if (s.x > W + 4) s.x = -4;
  if (s.x < -4) s.x = W + 4;
  if (s.y > H + 4) s.y = -4;
  if (s.y < -4) s.y = H + 4;
}

function spawnShootingStar() {
  if (!starfieldActive) return;

  shootingStars.push({
    x: Math.random() * W * 0.75,
    y: Math.random() * H * 0.35,
    length: Math.random() * 110 + 70,
    vx: Math.random() * 5 + 5.5,
    vy: Math.random() * 2 + 2.2,
    opacity: 1,
    fade: Math.random() * 0.014 + 0.008,
  });

  setTimeout(spawnShootingStar, Math.random() * 7000 + 4000);
}

function drawShootingStars() {
  shootingStars = shootingStars.filter((star) => star.opacity > 0 && star.x < W + 20 && star.y < H + 20);

  shootingStars.forEach((star) => {
    star.x += star.vx;
    star.y += star.vy;
    star.opacity -= star.fade;

    const mag = Math.hypot(star.vx, star.vy) || 1;
    const tailX = (star.vx / mag) * star.length;
    const tailY = (star.vy / mag) * star.length;

    const grad = ctx.createLinearGradient(star.x, star.y, star.x - tailX, star.y - tailY);
    grad.addColorStop(0, `rgba(255,255,255,${Math.min(star.opacity, 1)})`);
    grad.addColorStop(0.7, "rgba(255,255,255,0.18)");
    grad.addColorStop(1, "rgba(255,255,255,0)");

    ctx.save();
    ctx.strokeStyle = grad;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(star.x, star.y);
    ctx.lineTo(star.x - tailX, star.y - tailY);
    ctx.stroke();
    ctx.restore();
  });
}

function renderStarfield(ts = 0) {
  if (!starfieldActive) return;

  const frameInterval = 1000 / STAR_FPS;
  if (ts - lastFrameTs < frameInterval) {
    starfieldRAF = requestAnimationFrame(renderStarfield);
    return;
  }
  lastFrameTs = ts;

  ctx.clearRect(0, 0, W, H);

  for (const s of stars) {
    s.x += STAR_DIRECTION.x * s.speed;
    s.y += STAR_DIRECTION.y * s.speed;
    wrapStar(s);

    const px = ((s.x + pageScrollY * 0.03) % W + W) % W;
    const py = ((s.y + pageScrollY * 0.015) % H + H) % H;

    const grd = ctx.createRadialGradient(px, py, 0, px, py, s.r * 2.2);
    grd.addColorStop(0, `rgba(200,240,255,${s.a})`);
    grd.addColorStop(1, "transparent");

    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(px, py, s.r * 2.2, 0, Math.PI * 2);
    ctx.fill();
  }

  drawShootingStars();
  starfieldRAF = requestAnimationFrame(renderStarfield);
}

function startStarfield() {
  if (starfieldActive) return;
  starfieldActive = true;
  lastFrameTs = 0;
  spawnShootingStar();
  starfieldRAF = requestAnimationFrame(renderStarfield);
}

function stopStarfield() {
  starfieldActive = false;
  if (starfieldRAF !== null) {
    cancelAnimationFrame(starfieldRAF);
    starfieldRAF = null;
  }
}

document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopStarfield();
  else startStarfield();
});

startStarfield();

// ---------- local model ----------
function modelProgressCallback(info) {
  updateLoaderUI(info);
}

async function loadLocalModel() {
  if (!tokenizerPromise) {
    tokenizerPromise = AutoTokenizer.from_pretrained(MODEL_ID, {
      local_files_only: true,
      progress_callback: modelProgressCallback,
    });
  }

  if (!modelPromise) {
    modelPromise = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, {
      local_files_only: true,
      subfolder: "onnx",
      progress_callback: modelProgressCallback,
    });
  }

  const [tokenizer, model] = await Promise.all([tokenizerPromise, modelPromise]);
  return { tokenizer, model };
}

async function ensureLocalModelLoaded() {
  if (modelReady) {
    setRuntimeMode("local");
    return;
  }

  if (modelLoading) return;

  modelLoading = true;
  loadModelBtn.disabled = true;
  loadModelBtn.textContent = "Downloading...";
  loaderPhase.textContent = "Starting browser model download...";
  loaderFile.textContent = "Requesting tokenizer and ONNX weights...";

  try {
    await loadLocalModel();
    modelReady = true;
    modelLoading = false;

    setModelStatus("Downloaded and ready");
    setRuntimeMode("local");
    updateLoaderProgress(100, null, null, "All files loaded successfully.", "Browser model is ready");

    loadModelBtn.disabled = false;
    loadModelBtn.textContent = "Use browser model";
  } catch (err) {
    console.error(err);
    modelLoading = false;

    setModelStatus("Download failed");
    loaderPhase.textContent = "Download failed";
    loaderFile.textContent = err.message;
    loadModelBtn.disabled = false;
    loadModelBtn.textContent = "Download browser model";
    setStatus(`Failed to load browser model: ${err.message}`);
    throw err;
  }
}

// ---------- prediction helpers ----------
function softmax(values) {
  const maxVal = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function sortPredictions(predictions) {
  return [...predictions].sort((a, b) => b.prob - a.prob);
}

function computeTopP(predictions, threshold = TOP_P_THRESHOLD) {
  const out = [];
  let cumulative = 0;

  for (const pred of predictions) {
    out.push(pred);
    cumulative += pred.prob;
    if (cumulative >= threshold) break;
  }

  return out;
}

function updateTopPMoreButton() {
  if (!topPScrollEl || !topPMoreBtn) return;

  const hasOverflow = topPScrollEl.scrollHeight > topPScrollEl.clientHeight + 6;
  const nearBottom =
    topPScrollEl.scrollTop + topPScrollEl.clientHeight >= topPScrollEl.scrollHeight - 8;

  topPMoreBtn.classList.toggle("hidden", !hasOverflow || nearBottom);
}

function renderPredictions(predictions) {
  const sorted = sortPredictions(predictions);
  const top1 = sorted[0];
  const topP = computeTopP(sorted, TOP_P_THRESHOLD);

  top1El.textContent = `${top1.label} (${(top1.prob * 100).toFixed(2)}%)`;

  topPListEl.innerHTML = "";
  topP.forEach((pred, idx) => {
    const li = document.createElement("li");
    li.className = "top-p-row";

    const head = document.createElement("div");
    head.className = "top-p-row-head";

    const rank = document.createElement("span");
    rank.className = "top-p-rank";
    rank.textContent = `#${idx + 1}`;

    const label = document.createElement("span");
    label.className = "top-p-label";
    label.textContent = pred.label;

    const value = document.createElement("span");
    value.className = "top-p-value";
    value.textContent = `${(pred.prob * 100).toFixed(2)}%`;

    head.appendChild(rank);
    head.appendChild(label);
    head.appendChild(value);

    const track = document.createElement("div");
    track.className = "top-p-bar-track";

    const fill = document.createElement("div");
    fill.className = "top-p-bar-fill";
    fill.style.width = `${(pred.prob * 100).toFixed(2)}%`;

    track.appendChild(fill);
    li.appendChild(head);
    li.appendChild(track);

    topPListEl.appendChild(li);
  });

  resultEmptyEl.classList.add("hidden");
  resultEl.classList.remove("hidden");

  requestAnimationFrame(updateTopPMoreButton);

  return {
    top1,
    top5: sorted.slice(0, 5),
    topP,
  };
}

// ---------- API ----------
async function predictViaAPI(title, abstract) {
  const response = await fetch("/api/predict/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ title, abstract }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "API request failed");
  }

  return data.predictions;
}

async function predictViaLocal(title, abstract) {
  const { tokenizer, model } = await loadLocalModel();

  const inputs = await tokenizer(title, {
    text_pair: abstract,
    truncation: true,
    max_length: MAX_LENGTH,
    padding: true,
    return_tensor: true,
  });

  const outputs = await model(inputs);
  const logits = Array.from(outputs.logits.data);
  const probs = softmax(logits);

  const id2label = model.config.id2label || {};
  const predictions = probs.map((prob, idx) => ({
    label: id2label[String(idx)] ?? id2label[idx] ?? `class_${idx}`,
    prob,
  }));

  return sortPredictions(predictions);
}

// ---------- history ----------
function getHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function setHistory(items) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
}

function saveHistoryEntry(entry) {
  const items = getHistory();
  items.unshift(entry);
  const trimmed = items.slice(0, HISTORY_LIMIT);
  setHistory(trimmed);
  renderHistory();
}

function formatTime(iso) {
  const d = new Date(iso);
  return d.toLocaleString();
}

function renderHistory() {
  const items = getHistory();
  historyBodyEl.innerHTML = "";

  if (!items.length) {
    historyEmptyEl.classList.remove("hidden");
    historyWrapEl.classList.add("hidden");
    return;
  }

  historyEmptyEl.classList.add("hidden");
  historyWrapEl.classList.remove("hidden");

  for (const item of items) {
    const tr = document.createElement("tr");

    const tdTime = document.createElement("td");
    tdTime.className = "history-time";
    tdTime.textContent = formatTime(item.created_at);

    const tdTitle = document.createElement("td");
    tdTitle.className = "history-title";
    tdTitle.textContent = item.title;

    const tdTop1 = document.createElement("td");
    tdTop1.className = "history-top1";
    tdTop1.textContent = item.top1;

    const tdTop5 = document.createElement("td");
    const wrap = document.createElement("div");
    wrap.className = "history-top5";

    for (const pred of item.top5) {
      const pill = document.createElement("span");
      pill.className = "history-pill";
      pill.textContent = `${pred.label} ${pred.prob}`;
      wrap.appendChild(pill);
    }

    tdTop5.appendChild(wrap);

    tr.appendChild(tdTime);
    tr.appendChild(tdTitle);
    tr.appendChild(tdTop1);
    tr.appendChild(tdTop5);

    historyBodyEl.appendChild(tr);
  }
}

// ---------- main prediction ----------
async function predict() {
  const title = titleInput.value.trim();
  const abstract = abstractInput.value.trim();

  if (!title || !abstract) {
    setStatus("Please fill both the title and the abstract.");
    return;
  }

  predictBtn.disabled = true;
  setStatus(`Running inference via ${runtimeMode === "local" ? "browser model" : "server API"}...`);

  try {
    const predictions =
      runtimeMode === "local" && modelReady
        ? await predictViaLocal(title, abstract)
        : await predictViaAPI(title, abstract);

    const rendered = renderPredictions(predictions);

    saveHistoryEntry({
      created_at: new Date().toISOString(),
      title,
      top1: `${rendered.top1.label} (${(rendered.top1.prob * 100).toFixed(2)}%)`,
      top5: rendered.top5.map((x) => ({
        label: x.label,
        prob: `${(x.prob * 100).toFixed(2)}%`,
      })),
    });

    setStatus(`Done. Used ${runtimeMode === "local" && modelReady ? "browser model" : "server API"}.`);
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`);
  } finally {
    predictBtn.disabled = false;
  }
}

// ---------- demo ----------
function fillDemoExample() {
  titleInput.value = "Attention Is All You Need";
  abstractInput.value =
    "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.";
}

// ---------- events ----------
loadModelBtn.addEventListener("click", async () => {
  try {
    if (modelReady) {
      setRuntimeMode("local");
      return;
    }

    await ensureLocalModelLoaded();
  } catch (_) {
    // handled above
  }
});

useApiBtn.addEventListener("click", () => {
  setRuntimeMode("api");
});

predictBtn.addEventListener("click", predict);
demoBtn.addEventListener("click", fillDemoExample);

clearHistoryBtn.addEventListener("click", () => {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
});

topPScrollEl.addEventListener("scroll", updateTopPMoreButton, { passive: true });

topPMoreBtn.addEventListener("click", () => {
  topPScrollEl.scrollBy({
    top: 180,
    behavior: "smooth",
  });
});

// ---------- init ----------
renderHistory();
setRuntimeMode("api");