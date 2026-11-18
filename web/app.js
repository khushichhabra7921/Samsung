const get = (id) => document.getElementById(id);
const video = get("video");
const seek = get("seek");
const upload = get("upload-button");
const input = get("file-input");
const notice = get("notice");
const sourceView = get("source-view");
const highlightView = get("highlight-view");
const motionCanvas = get("motion");
const handCanvas = get("hand-overlay");
const handToggle = get("hands-toggle");
const player = video.parentElement;
const demoSource = {
  name: "Kitchen prep",
  source: "assets/kitchen-demo.mp4",
  highlight: "assets/kitchen-highlight.mp4",
  report: "assets/kitchen-demo.json",
  trace: "assets/kitchen-trace.json",
};

const handEdges = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const sleep = (delay) => new Promise((resolve) => setTimeout(resolve, delay));

let engine;
let current;
let view = "source";
let activeClip;
let uploading = false;

function timestamp(ms, precise = true) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const minutes = String(Math.floor(total / 60)).padStart(2, "0");
  const seconds = String(total % 60).padStart(2, "0");
  const hundredths = String(Math.floor(Math.max(0, ms) % 1000 / 10)).padStart(2, "0");
  return `${minutes}:${seconds}${precise ? `.${hundredths}` : ""}`;
}

function duration(ms) {
  return `${(ms / 1000).toFixed(2)} s`;
}

function showNotice(message = "", isError = false) {
  notice.textContent = message;
  notice.dataset.error = String(isError);
  notice.hidden = !message;
}

async function json(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const message = typeof error.detail === "string" ? error.detail : "The local engine could not complete the request.";
    throw new Error(message);
  }
  return response.json();
}

async function loadResult(source) {
  const [report, trace] = await Promise.all([json(source.report), json(source.trace)]);
  return { ...source, result: report, timeline: trace };
}

function download(id, url, filename) {
  const link = get(id);
  link.setAttribute("aria-disabled", String(!url));
  if (url) {
    link.href = url;
    link.download = filename;
  } else {
    link.removeAttribute("href");
  }
}

function playVideo() {
  video.play().catch((error) => {
    if (error.name === "NotSupportedError") showNotice("Your browser cannot play this MP4 codec.", true);
  });
}

function setView(mode) {
  if (!current) return;
  const url = mode === "highlights" ? current.highlight : current.source;
  if (!url) return;
  activeClip = undefined;
  view = mode;
  sourceView.setAttribute("aria-pressed", String(mode === "source"));
  highlightView.setAttribute("aria-pressed", String(mode === "highlights"));
  video.setAttribute("aria-label", mode === "source" ? "First-person source video" : "Selected highlights");
  if (video.src !== new URL(url, location.href).href) {
    video.pause();
    video.src = url;
  }
  if (!current.id && mode === "source") video.poster = "assets/kitchen-start.webp";
  else video.removeAttribute("poster");
  updatePlayback();
}

function sourceTime() {
  let ms = video.currentTime * 1000;
  if (view === "source" || !current) return ms;
  for (const clip of current.result.highlights) {
    const encodedDuration = Math.floor((clip.end_ms - clip.start_ms) * 30 / 1000) * 1000 / 30;
    if (ms < encodedDuration) return clip.start_ms + ms;
    ms -= encodedDuration;
  }
  return current.result.highlights.at(-1)?.end_ms ?? 0;
}

function updatePlayback() {
  const report = current?.result;
  if (!report) return;
  const time = Math.min(sourceTime(), report.duration_ms);
  if (activeClip && view === "source" && time >= activeClip.end_ms) {
    activeClip = undefined;
    video.pause();
  }
  seek.value = String(Math.round(time));
  seek.setAttribute("aria-valuetext", timestamp(time));
  get("source-time").textContent = timestamp(time);
  get("cursor").style.left = `calc(${100 * time / report.duration_ms}% - 1px)`;
  for (const [index, button] of Array.from(get("highlight-list").querySelectorAll("button")).entries()) {
    const clip = report.highlights[index];
    button.setAttribute("aria-current", String(time >= clip.start_ms && time < clip.end_ms));
  }
  drawHands();
}

function seekSource(ms, clip) {
  if (view !== "source") setView("source");
  activeClip = clip;
  video.currentTime = ms / 1000;
  updatePlayback();
}

function drawMotion() {
  if (!current) return;
  const { width, height } = motionCanvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  motionCanvas.width = Math.max(1, width * dpr);
  motionCanvas.height = Math.max(1, height * dpr);
  const context = motionCanvas.getContext("2d");
  context.scale(dpr, dpr);
  const times = current.timeline.motion;
  const length = current.result.duration_ms;
  const step = width / Math.max(1, times.length);
  context.fillStyle = "#7b91b7";
  for (const [time, score] of times) {
    const level = 2 + score * (height - 7);
    context.fillRect(time * width / length, (height - level) / 2, Math.max(1, step - 2), level);
  }
}

function nearestFrame(trace, time) {
  let low = 0;
  let high = trace.length - 1;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (trace[middle].time_ms < time) low = middle + 1;
    else high = middle;
  }
  const before = trace[Math.max(0, low - 1)];
  const after = trace[low];
  if (!after) return undefined;
  const frame = Math.abs(after.time_ms - time) < Math.abs(before.time_ms - time) ? after : before;
  return Math.abs(frame.time_ms - time) < 210 ? frame : undefined;
}

function drawTracks(canvas, frame, sourceWidth, sourceHeight, fit) {
  const { width, height } = canvas.getBoundingClientRect();
  if (!width || !height) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
    canvas.width = width * dpr;
    canvas.height = height * dpr;
  }
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  if (!frame || !sourceWidth || !sourceHeight) return;
  const scale = fit === "cover" ? Math.max(width / sourceWidth, height / sourceHeight)
    : Math.min(width / sourceWidth, height / sourceHeight);
  const renderedWidth = sourceWidth * scale;
  const renderedHeight = sourceHeight * scale;
  const dx = (width - renderedWidth) / 2;
  const dy = (height - renderedHeight) / 2;
  for (const hand of frame.hands) {
    const color = hand.side === "left" ? "#aec7ff" : "#a4f1ed";
    const points = hand.points.map(([x, y]) => [x * renderedWidth + dx, y * renderedHeight + dy]);
    context.strokeStyle = color;
    context.lineWidth = 1.4;
    context.shadowColor = "#080d16";
    context.shadowBlur = 3;
    context.beginPath();
    for (const [from, to] of handEdges) {
      context.moveTo(...points[from]);
      context.lineTo(...points[to]);
    }
    context.stroke();
    context.fillStyle = color;
    for (const point of points) {
      context.beginPath();
      context.arc(...point, 2.05, 0, 2 * Math.PI);
      context.fill();
    }
    if (fit === "cover") {
      const xs = points.map((point) => point[0]);
      const ys = points.map((point) => point[1]);
      const left = Math.max(6, Math.min(...xs) - 8);
      const top = Math.max(18, Math.min(...ys) - 8);
      const boxWidth = Math.max(...xs) - left + 8;
      const boxHeight = Math.max(...ys) - top + 8;
      context.lineWidth = 0.85;
      context.strokeRect(left, top, boxWidth, boxHeight);
      const text = `${hand.side.toUpperCase()} · ID ${String(hand.id).padStart(2, "0")}`;
      context.font = "600 9px ui-monospace, monospace";
      context.fillStyle = "#101827";
      context.fillRect(left, top - 14, context.measureText(text).width + 9, 14);
      context.fillStyle = color;
      context.fillText(text, left + 4, top - 4);
    }
  }
}

function drawHands() {
  const enabled = current && handToggle.checked && view === "source" && !get("hand-control").hidden;
  handCanvas.hidden = !enabled;
  if (!enabled) return;
  const frame = nearestFrame(current.timeline.hands, video.currentTime * 1000);
  drawTracks(handCanvas, frame, video.videoWidth, video.videoHeight, "contain");
}

function clipButton(clip, index) {
  const row = document.createElement("li");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "highlight-item";
  button.innerHTML =
    '<span class="clip-play" aria-hidden="true"><svg class="icon"><use href="#i-play"></use></svg></span>' +
    '<span class="clip-text"><strong></strong><small></small></span><span class="clip-length"></span>';
  const name = clip.labels.map(({ verb, noun }) => `${verb} ${noun}`).join(" · ");
  button.querySelector("strong").textContent = name || (current.result.highlights.length === 1 ? "Motion highlight" : `Highlight ${index + 1}`);
  button.querySelector("small").textContent = `${timestamp(clip.start_ms)} – ${timestamp(clip.end_ms)}`;
  button.querySelector(".clip-length").textContent = duration(clip.end_ms - clip.start_ms);
  button.addEventListener("click", () => {
    seekSource(clip.start_ms, clip);
    playVideo();
  });
  row.append(button);
  return row;
}

function display(result) {
  current = result;
  const report = result.result;
  const hasHands = result.timeline.hands.some((frame) => frame.hands.length);
  get("source-name").textContent = result.name;
  get("source-type").textContent = result.id ? "Uploaded footage" : "Demo footage";
  get("mode-chip").textContent = report.mode === "semantic" ? "Semantic" : "CPU preview";
  get("result-provenance").textContent = result.id ? "Local analysis · Temporary files" : "Sample analysis · Team 404NotFound";
  get("remove-video").hidden = !result.id;
  get("hand-control").hidden = !hasHands;
  handToggle.checked = hasHands && !result.id;
  highlightView.disabled = !result.highlight;
  seek.max = String(report.duration_ms);
  get("source-end").textContent = timestamp(report.duration_ms);
  get("selected-duration").textContent = duration(report.highlight_duration_ms);
  get("max-duration").textContent = `/ ${duration(report.budget_ms)}`;
  get("budget-meter").value = report.budget_ms ? report.highlight_duration_ms / report.budget_ms : 0;
  get("short-note").hidden = report.status === "complete";
  get("short-note").textContent = report.highlights.length
    ? "Short clip preview. A 4-second interval exceeds this clip’s 15% cap."
    : "The source is too short to select a highlight within the cap.";
  get("interval-count").textContent = String(report.highlights.length);
  get("highlight-list").replaceChildren(...report.highlights.map(clipButton));
  if (!report.highlights.length) {
    const empty = document.createElement("li");
    empty.className = "empty-list";
    empty.textContent = "No selected intervals";
    get("highlight-list").append(empty);
  }

  get("intervals").replaceChildren(...report.highlights.map((clip) => {
    const interval = document.createElement("span");
    interval.className = "timeline-range";
    interval.style.left = `${100 * clip.start_ms / report.duration_ms}%`;
    interval.style.width = `${100 * (clip.end_ms - clip.start_ms) / report.duration_ms}%`;
    return interval;
  }));
  get("action-panel").hidden = report.mode !== "semantic";
  get("action-count").textContent = `(${report.actions.length})`;
  get("video-label").hidden = !report.video_label;
  if (report.video_label) {
    const { verb, noun } = report.video_label;
    get("video-label").textContent = `Video label · ${verb} ${noun}`;
  }
  get("action-list").replaceChildren(...report.actions.map((action) => {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = `${action.label.verb} ${action.label.noun}`;
    const contact = action.evidence.contact == null ? "contact unavailable" : `contact ${Math.round(action.evidence.contact * 100)}%`;
    item.append(title, `${timestamp(action.start_ms)} · ${action.verified ? "Verified" : "Unverified"} · score ${action.confidence.toFixed(2)} · ${contact}`);
    return item;
  }));
  download("json-download", result.report, `${report.video_id}.report.json`);
  download("video-download", result.highlight, `${report.video_id}.highlights.mp4`);
  setView("source");
  drawMotion();
  showNotice(result.export_error || "", Boolean(result.export_error));
}

function setupOpening() {
  const background = get("opening-video");
  const toggle = get("opening-toggle");
  const motion = matchMedia("(prefers-reduced-motion: reduce)");
  let wantsPlayback = !motion.matches;
  let visible = false;

  function syncPlayback() {
    if (!visible || document.hidden || !wantsPlayback) {
      background.pause();
      return;
    }
    background.play().catch((error) => {
      if (error.name !== "AbortError") wantsPlayback = false;
    });
  }

  toggle.hidden = false;
  toggle.addEventListener("click", () => {
    wantsPlayback = background.paused;
    syncPlayback();
  });
  background.addEventListener("playing", () => { toggle.textContent = "Pause background"; });
  background.addEventListener("pause", () => { toggle.textContent = "Play background"; });
  background.addEventListener("error", () => { toggle.hidden = true; });
  motion.addEventListener("change", () => {
    wantsPlayback = !motion.matches;
    syncPlayback();
  });
  document.addEventListener("visibilitychange", syncPlayback);
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    syncPlayback();
  }, { threshold: 0.15 }).observe(get("top"));
}

function drawDemoSignal(canvas, { result, timeline }) {
  const { width, height } = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(width * dpr));
  canvas.height = Math.max(1, Math.round(height * dpr));
  const context = canvas.getContext("2d");
  context.scale(dpr, dpr);
  const x = (time) => time * width / result.duration_ms;
  const baseline = height * 0.86;

  context.strokeStyle = "#293b57";
  context.lineWidth = 1;
  context.beginPath();
  for (let row = 0; row <= 4; row++) {
    const y = 10 + row * (baseline - 10) / 4;
    context.moveTo(0, y);
    context.lineTo(width, y);
  }
  for (let time = 0; time < result.duration_ms; time += 2000) {
    context.moveTo(x(time), 10);
    context.lineTo(x(time), baseline);
  }
  context.stroke();

  for (const { start_ms, end_ms } of result.highlights) {
    const left = x(start_ms);
    const selectedWidth = x(end_ms) - left;
    context.fillStyle = "#769fff26";
    context.fillRect(left, 10, selectedWidth, baseline - 10);
    context.fillStyle = "#8aabf9";
    context.fillRect(left, baseline, selectedWidth, 2);
  }

  const bars = timeline.motion;
  const barWidth = Math.max(1, width / Math.max(1, bars.length) * 0.46);
  const shading = context.createLinearGradient(0, 10, 0, baseline);
  shading.addColorStop(0, "#c3d7ff");
  shading.addColorStop(1, "#4c6ca35e");
  context.fillStyle = shading;
  for (const [time, score] of bars) {
    const level = 2 + score * (baseline - 16);
    context.fillRect(x(time), baseline - level, barWidth, level);
  }
}

setupOpening();
const demo = loadResult(demoSource);
demo.then((result) => {
  get("hero-duration").textContent = timestamp(result.result.duration_ms, false);
  get("signal-preview").hidden = false;
  const signal = get("overview-signal");
  const resizeSignal = () => drawDemoSignal(signal, result);
  new ResizeObserver(resizeSignal).observe(signal);
  resizeSignal();

  const canvas = get("sample-landmarks");
  const frame = nearestFrame(result.timeline.hands, Number(canvas.dataset.time));
  if (frame?.hands.length) {
    canvas.hidden = false;
    const poster = get("sample-poster");
    const resize = () => drawTracks(canvas, frame, poster.naturalWidth, poster.naturalHeight, "cover");
    new ResizeObserver(resize).observe(canvas);
    poster.addEventListener("load", resize, { once: true });
    resize();
  }
}).catch(() => {});

async function openDemo(play = false) {
  try {
    display(await demo);
    if (play) {
      video.currentTime = 0;
      playVideo();
    }
  } catch {
    showNotice("The bundled sample couldn’t be loaded.", true);
  }
}

for (const link of document.querySelectorAll("[data-sample-play]")) {
  link.addEventListener("click", () => openDemo(true));
}
sourceView.addEventListener("click", () => setView("source"));
highlightView.addEventListener("click", () => setView("highlights"));
seek.addEventListener("input", () => seekSource(Number(seek.value)));
handToggle.addEventListener("change", drawHands);
video.addEventListener("timeupdate", updatePlayback);
video.addEventListener("seeked", updatePlayback);
video.addEventListener("loadedmetadata", updatePlayback);
video.addEventListener("error", () => showNotice("Your browser couldn’t play the video. Try an H.264 MP4.", true));
if ("requestVideoFrameCallback" in video) {
  const onFrame = () => {
    updatePlayback();
    video.requestVideoFrameCallback(onFrame);
  };
  video.requestVideoFrameCallback(onFrame);
}
new ResizeObserver(() => { drawMotion(); drawHands(); }).observe(player);

function setProgress(stage, fraction) {
  get("job-progress").hidden = false;
  get("job-stage").textContent = stage;
  get("job-percent").textContent = fraction == null ? "" : `${Math.round(fraction * 100)}%`;
  if (fraction == null) get("job-meter").removeAttribute("value");
  else get("job-meter").value = fraction;
}

function storeJob(id) {
  try {
    if (id) sessionStorage.setItem("handscribe-local-job", id);
    else sessionStorage.removeItem("handscribe-local-job");
  } catch {
    // The current analysis still works if browser storage is disabled.
  }
}

async function watchJob(id) {
  let job = await json(`api/jobs/${id}`);
  while (job.state === "running" || job.state === "uploading") {
    setProgress(job.stage, job.fraction);
    await sleep(750);
    job = await json(`api/jobs/${id}`);
  }
  if (job.state !== "complete") throw new Error(job.error || "The video analysis failed.");
  display(await loadResult(job));
}

function setBusy(busy) {
  uploading = busy;
  upload.disabled = busy || !engine;
  get("remove-video").disabled = busy;
  if (!busy) get("job-progress").hidden = true;
}

upload.addEventListener("click", () => input.click());
input.addEventListener("change", async () => {
  const [file] = input.files;
  input.value = "";
  if (!file || uploading || !engine) return;
  if (!file.name.toLowerCase().endsWith(".mp4")) {
    showNotice("Choose a first-person MP4 file.", true);
    return;
  }
  if (file.size > engine.max_upload_bytes || file.size === 0) {
    const limit = Math.round(engine.max_upload_bytes / (1024 * 1024));
    showNotice(`Choose a non-empty MP4 no larger than ${limit} MiB.`, true);
    return;
  }
  setBusy(true);
  showNotice();
  setProgress("Sending MP4 to the local engine", null);
  try {
    const job = await json(`api/jobs?name=${encodeURIComponent(file.name)}`, {
      method: "POST", headers: { "Content-Type": "video/mp4" }, body: file,
    });
    storeJob(job.id);
    await watchJob(job.id);
  } catch (error) {
    storeJob();
    showNotice(error.message, true);
  } finally {
    setBusy(false);
  }
});

get("remove-video").addEventListener("click", async () => {
  if (!current?.id || uploading) return;
  const id = current.id;
  const response = await fetch(`api/jobs/${id}`, { method: "DELETE" }).catch(() => null);
  if (response?.ok || response?.status === 404) {
    storeJob();
    await openDemo();
    showNotice("Uploaded video and analysis removed.");
  } else {
    showNotice("The local engine couldn’t remove the video.", true);
  }
});

async function connect() {
  await openDemo();
  const connection = get("connection");
  try {
    engine = await json("api/health");
    connection.textContent = engine.mode === "semantic" ? "Local engine · Semantic mode" : "Local engine · CPU preview";
    connection.classList.add("ready");
    setBusy(false);
  } catch {
    connection.textContent = "Sample mode · Start the local engine to upload";
    return;
  }
  let id;
  try { id = sessionStorage.getItem("handscribe-local-job"); } catch { return; }
  if (!id || !/^[a-f0-9]{32}$/.test(id)) return;
  setBusy(true);
  try {
    await watchJob(id);
  } catch (error) {
    storeJob();
    showNotice(error.message, true);
  } finally {
    setBusy(false);
  }
}

connect();
