const palette = {
  cyan: "#31d7c4",
  blue: "#6aa7ff",
  amber: "#f2bd5a",
  pink: "#ef7ba7",
  muted: "#8e99aa",
  grid: "rgba(142, 153, 170, 0.12)",
};

const state = {
  runs: [],
  selectedRun: null,
  payload: null,
  charts: {},
  source: "api",
  timer: null,
};

const elements = {
  runSelect: document.querySelector("#run-select"),
  smoothing: document.querySelector("#smoothing"),
  smoothingValue: document.querySelector("#smoothing-value"),
  autoRefresh: document.querySelector("#auto-refresh"),
  refreshButton: document.querySelector("#refresh-button"),
  status: document.querySelector("#status-pill"),
  updatedAt: document.querySelector("#updated-at"),
  progressBar: document.querySelector("#progress-bar"),
  progressLabel: document.querySelector("#progress-label"),
  empty: document.querySelector("#empty-state"),
  error: document.querySelector("#error-message"),
};

Chart.defaults.color = palette.muted;
Chart.defaults.borderColor = palette.grid;
Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatTokens(value) {
  if (!value) return "0";
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return String(value);
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
}

function movingAverage(values, windowSize) {
  if (windowSize <= 1) return values.slice();
  const result = [];
  let sum = 0;
  for (let index = 0; index < values.length; index += 1) {
    sum += values[index];
    if (index >= windowSize) sum -= values[index - windowSize];
    result.push(sum / Math.min(index + 1, windowSize));
  }
  return result;
}

function chartOptions(yTitle, secondAxisTitle = null) {
  const scales = {
    x: {
      type: "linear",
      title: { display: true, text: "Tokens seen (millions)" },
      grid: { color: palette.grid },
    },
    y: {
      title: { display: true, text: yTitle },
      grid: { color: palette.grid },
    },
  };
  if (secondAxisTitle) {
    scales.y1 = {
      position: "right",
      title: { display: true, text: secondAxisTitle },
      grid: { drawOnChartArea: false },
    };
  }
  return {
    responsive: true,
    maintainAspectRatio: false,
    parsing: false,
    animation: false,
    interaction: { mode: "nearest", intersect: false },
    plugins: {
      legend: { labels: { usePointStyle: true, boxWidth: 8 } },
      decimation: { enabled: true, algorithm: "lttb", samples: 1000 },
    },
    scales,
  };
}

function replaceChart(name, canvasId, data, options) {
  if (state.charts[name]) state.charts[name].destroy();
  state.charts[name] = new Chart(document.querySelector(`#${canvasId}`), {
    type: "line",
    data,
    options,
  });
}

async function fetchJson(url) {
  const response = await fetch(`${url}${url.includes("?") ? "&" : "?"}t=${Date.now()}`, {
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function loadRuns() {
  const previous = state.selectedRun;
  try {
    const payload = await fetchJson("/api/runs");
    state.runs = payload.runs || [];
    state.source = "api";
  } catch (_) {
    const payload = await fetchJson("./data/runs.json");
    state.runs = payload.runs || [];
    state.source = "static";
  }

  elements.runSelect.innerHTML = "";
  state.runs.forEach((run) => {
    const option = document.createElement("option");
    option.value = run.name;
    option.textContent = `${run.name} · ${run.status || "running"}`;
    elements.runSelect.append(option);
  });
  state.selectedRun = state.runs.some((run) => run.name === previous)
    ? previous
    : state.runs.at(-1)?.name || null;
  if (state.selectedRun) elements.runSelect.value = state.selectedRun;
}

async function loadSelectedRun() {
  if (!state.selectedRun) {
    state.payload = null;
    render();
    return;
  }
  const run = state.runs.find((item) => item.name === state.selectedRun);
  try {
    state.payload = state.source === "api"
      ? await fetchJson(`/api/run/${encodeURIComponent(state.selectedRun)}`)
      : await fetchJson(`./data/${run.data_file}`);
    hideError();
    render();
  } catch (error) {
    showError(`Could not load ${state.selectedRun}: ${error.message}`);
  }
}

function renderCards(train, validation, summary, config) {
  const lastTrain = train.at(-1) || {};
  const generalValidation = validation.filter((record) =>
    String(record.split || "").startsWith("general")
  );
  const lastValidation = generalValidation.at(-1) || {};
  const tokensSeen = summary.tokens_seen ?? lastTrain.tokens_seen ?? 0;
  const targetTokens = summary.target_tokens ?? config.training?.target_tokens ?? 0;
  const throughput = lastTrain.tokens_per_second;
  const eta = throughput > 0 ? (targetTokens - tokensSeen) / throughput : NaN;
  const progress = targetTokens > 0 ? Math.min(tokensSeen / targetTokens, 1) : 0;

  document.querySelector("#card-step").textContent = formatNumber(summary.step ?? lastTrain.step, 0);
  document.querySelector("#card-loss").textContent = formatNumber(lastTrain.loss, 4);
  document.querySelector("#card-val-loss").textContent = formatNumber(lastValidation.loss, 4);
  document.querySelector("#card-perplexity").textContent = formatNumber(lastValidation.perplexity, 2);
  document.querySelector("#card-throughput").textContent = throughput
    ? `${formatNumber(throughput, 0)} tok/s`
    : "—";
  document.querySelector("#card-eta").textContent = formatDuration(eta);
  elements.progressBar.style.width = `${progress * 100}%`;
  elements.progressLabel.textContent = `${formatTokens(tokensSeen)} / ${formatTokens(targetTokens)} (${(progress * 100).toFixed(1)}%)`;
}

function renderCharts(train, validation) {
  const windowSize = Number(elements.smoothing.value);
  const trainLoss = train.map((record) => Number(record.loss));
  const smoothed = movingAverage(trainLoss, windowSize);
  const trainPoints = train.map((record) => ({ x: record.tokens_seen / 1e6, y: record.loss }));
  const smoothPoints = train.map((record, index) => ({ x: record.tokens_seen / 1e6, y: smoothed[index] }));
  const general = validation
    .filter((record) => String(record.split || "").startsWith("general"))
    .map((record) => ({ x: record.tokens_seen / 1e6, y: record.loss }));
  const structured = validation
    .filter((record) => String(record.split || "").startsWith("structured"))
    .map((record) => ({ x: record.tokens_seen / 1e6, y: record.loss }));

  replaceChart("loss", "loss-chart", {
    datasets: [
      { label: "Train loss", data: trainPoints, borderColor: "rgba(106, 167, 255, 0.28)", borderWidth: 1, pointRadius: 0 },
      { label: `Train MA(${windowSize})`, data: smoothPoints, borderColor: palette.blue, borderWidth: 2, pointRadius: 0 },
      { label: "Validation", data: general, borderColor: palette.cyan, backgroundColor: palette.cyan, borderWidth: 2, pointRadius: 3 },
      { label: "Structured validation", data: structured, borderColor: palette.pink, backgroundColor: palette.pink, borderWidth: 2, pointRadius: 3 },
    ],
  }, chartOptions("Cross-entropy loss"));

  replaceChart("optimization", "optimization-chart", {
    datasets: [
      {
        label: "Learning rate",
        data: train.map((record) => ({ x: record.tokens_seen / 1e6, y: record.learning_rate })),
        borderColor: palette.cyan,
        pointRadius: 0,
        yAxisID: "y",
      },
      {
        label: "Gradient norm",
        data: train.map((record) => ({ x: record.tokens_seen / 1e6, y: record.gradient_norm })),
        borderColor: palette.amber,
        pointRadius: 0,
        yAxisID: "y1",
      },
    ],
  }, chartOptions("Learning rate", "Gradient norm"));

  replaceChart("throughput", "throughput-chart", {
    datasets: [
      {
        label: "Tokens / second",
        data: train.map((record) => ({ x: record.tokens_seen / 1e6, y: record.tokens_per_second })),
        borderColor: palette.blue,
        pointRadius: 0,
        yAxisID: "y",
      },
      {
        label: "Step seconds",
        data: train.map((record) => ({ x: record.tokens_seen / 1e6, y: record.step_seconds })),
        borderColor: palette.pink,
        pointRadius: 0,
        yAxisID: "y1",
      },
    ],
  }, chartOptions("Tokens / second", "Seconds / step"));

  replaceChart("memory", "memory-chart", {
    datasets: [
      {
        label: "Allocated GiB",
        data: train.map((record) => ({ x: record.tokens_seen / 1e6, y: record.gpu_memory_allocated_gib })),
        borderColor: palette.cyan,
        backgroundColor: "rgba(49, 215, 196, 0.12)",
        fill: true,
        pointRadius: 0,
      },
      {
        label: "Reserved GiB",
        data: train.map((record) => ({ x: record.tokens_seen / 1e6, y: record.gpu_memory_reserved_gib })),
        borderColor: palette.blue,
        pointRadius: 0,
      },
    ],
  }, chartOptions("GPU memory (GiB)"));
}

function render() {
  if (!state.payload) {
    elements.empty.classList.remove("hidden");
    return;
  }
  const { metrics = [], summary = {}, config = {} } = state.payload;
  const train = metrics.filter((record) => record.type === "train");
  const validation = metrics.filter((record) => record.type === "validation");
  elements.empty.classList.toggle("hidden", train.length > 0 || validation.length > 0);

  const status = summary.status || state.runs.find((run) => run.name === state.selectedRun)?.status || "running";
  elements.status.textContent = status;
  elements.status.className = `pill ${status}`;
  elements.updatedAt.textContent = summary.updated_at
    ? `Updated ${new Date(summary.updated_at).toLocaleString()}`
    : state.source === "api" ? "Live local data" : "Static exported data";
  renderCards(train, validation, summary, config);
  renderCharts(train, validation);
}

function showError(message) {
  elements.error.textContent = message;
  elements.error.classList.remove("hidden");
}

function hideError() {
  elements.error.classList.add("hidden");
}

async function refresh() {
  try {
    await loadRuns();
    await loadSelectedRun();
  } catch (error) {
    showError(`Dashboard refresh failed: ${error.message}`);
  }
}

function configureTimer() {
  if (state.timer) clearInterval(state.timer);
  state.timer = elements.autoRefresh.checked ? setInterval(refresh, 10000) : null;
}

elements.runSelect.addEventListener("change", async (event) => {
  state.selectedRun = event.target.value;
  await loadSelectedRun();
});
elements.smoothing.addEventListener("input", () => {
  elements.smoothingValue.textContent = elements.smoothing.value;
  render();
});
elements.refreshButton.addEventListener("click", refresh);
elements.autoRefresh.addEventListener("change", configureTimer);

configureTimer();
refresh();

