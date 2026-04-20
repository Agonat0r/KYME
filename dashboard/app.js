/**
 * app.js — KYMA dashboard
 *
 * this file does everything on the frontend:
 *   - websocket connection to the python server
 *   - live 8-channel EMG waveform renderer (canvas 2d)
 *   - 3D robotic hand digital twin (three.js)
 *   - theme system (dark / retro terminal / mario NES)
 *   - training UI (hold-to-record, classifier picker, fit button)
 *   - manual servo sliders + quick gesture buttons
 *
 * the whole thing runs at ~60fps with no framework, just vanilla js.
 * i kept it in one file because there's no build step — just refresh.
 */

'use strict';

// =============================================================================
// THEME SYSTEM
//
// instead of trying to override :root with body.mario (which doesn't work
// because :root has higher specificity), we just write the vars directly
// onto document.documentElement.style with JS. simple and bulletproof.
// =============================================================================

const THEME_VARS = {
  dark: {
    '--bg':'#e7e2d9','--surface':'rgba(247, 242, 235, 0.74)','--border':'rgba(46, 47, 56, 0.10)',
    '--accent':'#586fda','--green':'#4f8f69','--yellow':'#b28327',
    '--red':'#c15c70','--text':'#252831','--text-dim':'#727683',
    '--radius':'18px','--font':"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  },
  retro: {
    '--bg':'#000000','--surface':'#0a0a0a','--border':'#004400',
    '--accent':'#00ff00','--green':'#00ff00','--yellow':'#ffff00',
    '--red':'#ff0000','--text':'#00ff00','--text-dim':'#008800',
    '--radius':'0px','--font':"'Press Start 2P', monospace",
  },
  mario: {
    '--bg':'#6b8cff','--surface':'#c84c0c','--border':'#000000',
    '--accent':'#fca044','--green':'#00a800','--yellow':'#fca044',
    '--red':'#e40712','--text':'#fcfcfc','--text-dim':'#e0d0c0',
    '--radius':'0px','--font':"'Press Start 2P', monospace",
  },
  gameboy: {
    '--bg':'#9bbc0f','--surface':'#8bac0f','--border':'#0f380f',
    '--accent':'#306230','--green':'#0f380f','--yellow':'#306230',
    '--red':'#0f380f','--text':'#0f380f','--text-dim':'#306230',
    '--radius':'0px','--font':"'Press Start 2P', monospace",
  },
  cyberpunk: {
    '--bg':'#0a0010','--surface':'#120020','--border':'#ff2d95',
    '--accent':'#00ffff','--green':'#00ff88','--yellow':'#ffee00',
    '--red':'#ff2d95','--text':'#e0d0ff','--text-dim':'#8866aa',
    '--radius':'2px','--font':"'Press Start 2P', monospace",
  },
};

// canvas-specific colors (can't use css vars in canvas api)
const THEME_CANVAS = {
  dark:      { bg:'#f1ece5', grid:'rgba(55,59,70,0.09)', alt:'rgba(88,111,218,0.030)' },
  retro:     { bg:'#000000', grid:'rgba(0,255,0,0.12)',     alt:'rgba(0,255,0,0.04)' },
  mario:     { bg:'#000040', grid:'rgba(252,160,68,0.15)',  alt:'rgba(252,160,68,0.06)' },
  gameboy:   { bg:'#9bbc0f', grid:'rgba(15,56,15,0.2)',     alt:'rgba(15,56,15,0.08)' },
  cyberpunk: { bg:'#0a0010', grid:'rgba(255,45,149,0.12)',  alt:'rgba(0,255,255,0.04)' },
};

// per-channel waveform colors — different palette for each theme
const CH_COLORS = {
  dark:      ['#586fda','#4f8f69','#c15c70','#b28327','#8664d6','#3f8f93','#c87a47','#6372b8'],
  retro:     ['#00ff00','#00cc00','#33ff33','#00ff66','#66ff66','#00ff99','#99ff99','#00ffcc'],
  mario:     ['#e40712','#00a800','#fca044','#049cd8','#ff6b6b','#43b047','#fcb514','#ffffff'],
  gameboy:   ['#0f380f','#306230','#0f380f','#306230','#0f380f','#306230','#0f380f','#306230'],
  cyberpunk: ['#ff2d95','#00ffff','#ff2d95','#00ffff','#ff2d95','#00ffff','#ff2d95','#00ffff'],
};

const THEME_SCENE_BG = {
  dark: 0xe9e4db,
  retro: 0x000a00,
  mario: 0x000040,
  gameboy: 0x0f380f,
  cyberpunk: 0x0a0010,
};

let currentTheme = localStorage.getItem('emg-theme') || 'dark';

function applyTheme(name) {
  currentTheme = name;
  const vars = THEME_VARS[name];
  if (!vars) return;

  // slam the vars directly onto :root so everything picks them up
  const root = document.documentElement;
  for (const [prop, val] of Object.entries(vars)) {
    root.style.setProperty(prop, val);
  }

  // body class drives the structural overrides (box-shadow, border-width, etc)
  document.body.className = name === 'dark' ? '' : name;
  document.body.style.fontSize = name === 'dark' ? '14px' : '10px';
  localStorage.setItem('emg-theme', name);

  const sel = document.getElementById('theme-select');
  if (sel) sel.value = name;

  // tell the 3d arm about the theme change
  if (window._arm3d) window._arm3d.setTheme(name);
}

function chColors() { return CH_COLORS[currentTheme] || CH_COLORS.dark; }


// =============================================================================
// CONFIG + STATE
// =============================================================================

const API = location.origin;
const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
const DISPLAY_SAMPLES = 500;  // how many samples to show in the rolling waveform
const N_CH = 8;               // cyton has 8 channels

// everything mutable lives here so it's easy to find
const S = {
  ws: null,
  sysState: 'idle',
  streaming: false,
  trained: false,
  recSession: false,
  recGesture: null,     // which gesture is currently being recorded (or null)
  gestures: [],         // list of gesture names from the server config
  trainCounts: {},      // gesture -> window count
  lastGesture: '',      // last predicted gesture (for bounce animation)
  lastRenderedGesture: '',
  lastPrediction: null,
  lastAngles: new Array(8).fill(90),  // last servo angles sent to 3d arm

  // ring buffers for the 8 EMG channels
  emg: Array.from({length:N_CH}, () => new Float32Array(DISPLAY_SAMPLES)),
  emgHead: 0,
  emgTotal: 0,                             // total samples received (for buffer-fill tracking)
  frames: 0,
  fpsTime: performance.now(),
  rms: new Array(N_CH).fill(0),
  rmsSmooth: new Float32Array(N_CH),       // smoothed RMS for mute hysteresis
  chMuted: new Array(N_CH).fill(true),     // mute state with hysteresis
  emgPeakSmooth: new Float32Array(N_CH),   // smoothed peak amplitude per channel

  // fatigue tracking — we compare current RMS to the peak RMS seen so far
  peakRms: new Array(N_CH).fill(0),
  fatigue: 1.0,  // 1.0 = fresh, 0.0 = exhausted

  // gesture timeline — last 60 prediction results as [gestureIdx, confidence]
  timeline: [],
  timelineMax: 60,

  // latency / performance tracking
  predCount: 0,
  predCountTime: performance.now(),
  predRate: 0,
  wsLatency: 0,
  lastPingTime: 0,
  lastSignalAtClient: 0,
  streamSource: 'hardware',
  streamDetails: {},
  playbackSessionId: '',
  sampleRate: 250,
  decoderMode: 'LDA',
  signalProfileKey: 'emg',
  signalProfileName: 'Signal',
  signalDescription: 'Select a biosignal profile.',
  signalSupportLevel: 'profile',
  signalUnits: 'uV',
  signalFullScale: 200.0,
  signalMetricLabel: 'Channel Activity',
  signalMetricScale: 200.0,
  muteFloor: 0.5,
  supportsTraining: true,
  supportsArmGestures: true,
  calibrationStage: 'idle',
  calibrationProtocol: null,
  availableProfiles: [],
  lslInputs: [],
  xdfStreams: [],
  subjects: [],
  sessions: [],
  selectedSessionIds: new Set(),
  datasets: [],
  experiments: [],
  selectedDatasetId: '',
  exportMeta: null,
  protocolTemplates: [],
  protocolStepIndex: 0,
  protocolRunId: '',
  eegExperiments: [],
  selectedEegExperiment: '',
  dashboardWorkspace: localStorage.getItem('kyma-dashboard-workspace') || 'live',
  activeViz: 'hand',
  eegBrain: {
    available: false,
    surface_available: false,
    note: '',
    surface_note: '',
    dominant_band: '',
    topomap_url: '',
    sensors_url: '',
    surface_url: '',
    loading: false,
    lastRefresh: 0,
  },
  review: {
    paused: false,
    snapshot: null,
    timelineSnapshot: null,
    predictionSnapshot: null,
    eegBrainSnapshot: null,
    selection: null,
    markers: [],
    dragging: false,
    dragOriginX: 0,
    lastStats: null,
  },
  channelEnabled: new Array(N_CH).fill(true),
  channelLabels: Array.from({length:N_CH}, (_, i) => `CH${i + 1}`),
  lsl: {
    available: false,
    active: false,
    include_markers: true,
    stream_name: '',
    marker_stream_name: '',
    last_error: '',
  },
  osc: {
    available: false,
    active: false,
    host: '',
    port: 9000,
    prefix: '/kyma',
    mirror_events: true,
    last_error: '',
  },

  // proportional control — maps EMG channels directly to joints (no classifier needed)
  diagnostics: {
    spectrum: { freq_hz: [], mag_db: [], segment_ms: 0 },
    noise: { hum_50_db: 0, hum_60_db: 0, drift_db: 0, clip_pct: 0, crest_factor: 0 },
    timing: {
      process_last_ms: 0,
      process_avg_ms: 0,
      process_max_ms: 0,
      interval_last_ms: 0,
      interval_avg_ms: 0,
      interval_jitter_ms: 0,
      dropped_windows: 0,
      window_count: 0,
      expected_interval_ms: 0,
      signal_age_ms: 0,
    },
    active_filter: null,
  },
  safety: {
    enabled: true,
    stream_timeout_ms: 1500,
    auto_estop_on_stale: true,
    signal_age_ms: 0,
    stale: false,
    trip_count: 0,
    last_trip_reason: '',
  },
  filterLab: {
    available: false,
    methods: [],
    responses: [],
    apply_modes: [],
    exports: [],
    filters: [],
    records: {},
    active_filter_id: '',
    active_filter: null,
    selected_filter_id: '',
    selected_filter: null,
    preview: null,
    last_error: '',
  },
  workshop: {
    available: false,
    last_error: '',
    views: ['fft', 'psd', 'spectrogram', 'autocorrelation', 'histogram', 'envelope', 'correlation', 'laplace'],
    view: 'fft',
    result: null,
    selectionMeta: null,
    loading: false,
    lastRequest: null,
  },
  proportional: false,
  propGain: 3,
  propDeadZone: 15,  // µV dead zone (BrainFlow reports µV directly)
  propRestRms: new Float32Array(N_CH),  // baseline RMS captured at enable
  propCalibrated: false,
  // channel→joint mappings: each entry is {ch, joint, dir} where dir=1 (flex) or -1 (extend)
  propMap: [
    { ch: 0, joint: 2, dir: 1 },
  ],
};


// =============================================================================
// DOM SHORTCUTS
// =============================================================================

const $ = id => document.getElementById(id);
const canvas = $('emg-canvas');
const ctx = canvas.getContext('2d');
const TOUR_STORAGE_KEY = 'kyma-tour-seen-v1';


function captureReviewSnapshot() {
  S.review.snapshot = {
    emg: S.emg.map(buf => Float32Array.from(buf)),
    head: S.emgHead,
    total: S.emgTotal,
    rms: Array.isArray(S.rms) ? S.rms.slice() : [],
    sampleRate: Number(S.sampleRate || 250),
    capturedAt: Date.now(),
  };
}

function clonePredictionPayload(payload) {
  if (!payload) return null;
  return {
    ...payload,
    confidence: Number(payload.confidence || 0),
    label: payload.label || payload.gesture || '--',
    gesture: payload.gesture || payload.label || '--',
    summary: payload.summary || '',
  };
}

function renderPredictionPanel(payload, { animate = false } = {}) {
  const el = $('pred-gesture');
  if (!el) return;

  const data = clonePredictionPayload(payload) || {
    label: '--',
    gesture: '--',
    confidence: 0,
    summary: '',
  };
  const label = data.label || '--';
  const confidence = Math.max(0, Math.min(1, Number(data.confidence || 0)));
  const summary = data.summary || `${S.signalProfileName} decoder active`;

  if (animate && label !== '--' && label !== S.lastRenderedGesture) {
    el.classList.remove('bounce');
    void el.offsetWidth;
    el.classList.add('bounce');
  }
  S.lastRenderedGesture = label;

  el.textContent = label;
  $('pred-confidence').textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
  $('pred-summary').textContent = summary;
  const bar = $('conf-bar');
  if (bar) {
    bar.style.width = `${confidence * 100}%`;
    bar.style.background = confidence > 0.8 ? 'var(--green)'
      : confidence > 0.55 ? 'var(--yellow)' : 'var(--red)';
  }
}

function syncPredictionPanel() {
  const payload = S.review.paused && S.review.predictionSnapshot
    ? S.review.predictionSnapshot
    : S.lastPrediction;
  renderPredictionPanel(payload, { animate: false });
}

function clearReviewSelection() {
  S.review.selection = null;
  S.review.lastStats = null;
}

function resetReviewState({ clearMarkers = false } = {}) {
  S.review.paused = false;
  S.review.snapshot = null;
  S.review.timelineSnapshot = null;
  S.review.predictionSnapshot = null;
  S.review.eegBrainSnapshot = null;
  S.review.dragging = false;
  if (clearMarkers) S.review.markers = [];
  clearReviewSelection();
  syncReviewUI();
  syncWorkshopUI();
}

function getReviewRenderState() {
  const paused = !!S.review.paused && !!S.review.snapshot;
  const source = paused ? S.review.snapshot : {
    emg: S.emg,
    head: S.emgHead,
    total: S.emgTotal,
    sampleRate: Number(S.sampleRate || 250),
  };
  const filled = Math.min(Number(source.total || 0), DISPLAY_SAMPLES);
  const baseAbs = Math.max(0, Number(source.total || 0) - filled);
  return {
    paused,
    emg: source.emg,
    head: Number(source.head || 0),
    total: Number(source.total || 0),
    filled,
    baseAbs,
    sampleRate: Number(source.sampleRate || S.sampleRate || 250),
  };
}

function getReviewLayout(state, width) {
  const filled = Math.max(state?.filled || 0, 0);
  const drawStart = filled < DISPLAY_SAMPLES
    ? Math.floor(width * (1 - filled / DISPLAY_SAMPLES))
    : 0;
  const visibleWidth = Math.max(1, width - drawStart);
  return { drawStart, visibleWidth };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function bufferIndexForAbsSample(state, absSample) {
  const rel = Math.round(absSample - state.baseAbs);
  const oldest = (state.head - state.filled + DISPLAY_SAMPLES) % DISPLAY_SAMPLES;
  return (oldest + rel + DISPLAY_SAMPLES) % DISPLAY_SAMPLES;
}

function sampleFromCanvasX(x, state, width) {
  const { drawStart, visibleWidth } = getReviewLayout(state, width);
  if (!state.filled) return state.baseAbs;
  const ratio = clamp((x - drawStart) / visibleWidth, 0, 0.999999);
  return state.baseAbs + Math.floor(ratio * state.filled);
}

function canvasXFromSample(absSample, state, width) {
  const { drawStart, visibleWidth } = getReviewLayout(state, width);
  const rel = clamp(absSample - state.baseAbs, 0, Math.max(state.filled - 1, 0));
  const ratio = state.filled > 1 ? rel / (state.filled - 1) : 0;
  return drawStart + ratio * visibleWidth;
}

function getSelectionRange(selection) {
  if (!selection) return null;
  return {
    start: Math.min(selection.startSample, selection.endSample),
    end: Math.max(selection.startSample, selection.endSample),
  };
}

function computeSelectionStats(selection, state) {
  const range = getSelectionRange(selection);
  if (!range || !state?.filled || range.end < range.start) return null;

  const visibleChannels = Array.from({ length: N_CH }, (_, i) => i).filter(i => isChannelVisible(i));
  const channels = visibleChannels.length ? visibleChannels : Array.from({ length: N_CH }, (_, i) => i);
  const count = Math.max(1, range.end - range.start + 1);
  let total = 0;
  let totalSquares = 0;
  let min = Infinity;
  let max = -Infinity;
  let focusChannel = channels[0] ?? 0;
  let focusRms = -Infinity;

  channels.forEach(ch => {
    let sumSquares = 0;
    for (let abs = range.start; abs <= range.end; abs++) {
      const idx = bufferIndexForAbsSample(state, abs);
      const value = Number(state.emg[ch]?.[idx] || 0);
      total += value;
      totalSquares += value * value;
      sumSquares += value * value;
      if (value < min) min = value;
      if (value > max) max = value;
    }
    const channelRms = Math.sqrt(sumSquares / count);
    if (channelRms > focusRms) {
      focusRms = channelRms;
      focusChannel = ch;
    }
  });

  const sampleRate = Math.max(Number(state.sampleRate || 250), 1);
  const totalPoints = count * Math.max(channels.length, 1);
  return {
    startSample: range.start,
    endSample: range.end,
    samples: count,
    durationMs: (count / sampleRate) * 1000,
    mean: total / totalPoints,
    rms: Math.sqrt(totalSquares / totalPoints),
    min,
    max,
    peakToPeak: max - min,
    focusChannel,
    focusLabel: S.channelLabels[focusChannel] || `CH${focusChannel + 1}`,
    focusRms,
  };
}

function pushReviewMarker(marker) {
  if (!marker || !marker.event) return;
  const item = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    event: String(marker.event),
    note: String(marker.note || ''),
    selection: marker.selection || null,
    createdAt: marker.createdAt || Date.now(),
    sampleIndex: Number.isFinite(marker.sampleIndex) ? Number(marker.sampleIndex) : null,
  };
  S.review.markers.unshift(item);
  if (S.review.markers.length > 18) S.review.markers.length = 18;
  syncReviewUI();
}


function normalizeChannelMask(mask) {
  const next = Array.isArray(mask) ? mask.slice(0, N_CH).map(v => v !== false) : [];
  while (next.length < N_CH) next.push(true);
  return next;
}

function loadChannelMask() {
  let store = {};
  try {
    store = JSON.parse(localStorage.getItem('kyma-channel-mask-v2') || '{}') || {};
  } catch {}
  S.channelEnabled = normalizeChannelMask(store[S.signalProfileKey]);
}

function saveChannelMask() {
  let store = {};
  try {
    store = JSON.parse(localStorage.getItem('kyma-channel-mask-v2') || '{}') || {};
  } catch {}
  store[S.signalProfileKey] = normalizeChannelMask(S.channelEnabled);
  localStorage.setItem('kyma-channel-mask-v2', JSON.stringify(store));
}

function isChannelVisible(index) {
  return !!S.channelEnabled[index];
}

function setChannelMask(mask) {
  S.channelEnabled = normalizeChannelMask(mask);
  saveChannelMask();
  buildLegend();
  buildRmsBars();
  buildQualityGrid();
  updateRmsBars(S.rms || []);
  updateQualityGrid(S.rms || []);
  if (S.review.selection) {
    S.review.lastStats = computeSelectionStats(S.review.selection, getReviewRenderState());
  }
  syncReviewUI();
}

function setAllChannelVisibility(value) {
  setChannelMask(new Array(N_CH).fill(!!value));
}

function toggleChannelVisibility(index) {
  const next = S.channelEnabled.slice();
  next[index] = !next[index];
  setChannelMask(next);
}


// =============================================================================
// BOOT — kicks everything off
// =============================================================================

async function boot() {
  // theme first so the page doesn't flash white
  applyTheme(currentTheme);
  $('theme-select').onchange = e => applyTheme(e.target.value);

  // grab server state before building the UI
  await loadStatus();
  await loadConfig();
  await loadFilterLabStatus();
  await refreshLSLStatus();
  await refreshOSCStatus();
  await loadLSLInputs();
  await scanPorts();
  buildLegend();
  buildRmsBars();
  buildServos();
  buildQualityGrid();
  updateFilterFieldVisibility();
  renderSpectrum();
  refreshFilterLabUI();
  initProportionalUI();
  bindButtons();
  bindReviewCanvas();
  connectWS();
  resizeCanvas();
  window.onresize = resizeCanvas;
  renderLoop();
  await loadSessions();
  await loadSubjects();
  await loadDatasets();
  await loadExperiments();

  // fire up the 3d arm once three.js is loaded
  init3DArm();

  // fire up the Babylon hand render (loads dashboard/assets/hand.glb)
  HandView.init();

  switchWorkspace(S.dashboardWorkspace);
  switchViz(S.activeViz);
  syncReviewUI();

  // model selector
  const modelSel = $('model-select');
  if (modelSel) {
    modelSel.onchange = () => {
      if (window._arm3d) window._arm3d.setModel(modelSel.value);
    };
  }

  // update performance stats once per second
  setInterval(updatePerformance, 1000);
  if (!localStorage.getItem(TOUR_STORAGE_KEY)) {
    setTimeout(() => toast('Click Tour for a guided walkthrough'), 1400);
  }
}


// =============================================================================
// API HELPERS
//
// thin wrappers around fetch so we don't repeat ourselves everywhere.
// errors get thrown so the calling code can toast() them.
// =============================================================================

async function api(method, path, body) {
  const opts = { method, headers:{'Content-Type':'application/json'} };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({detail:res.statusText}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}
const get  = p => api('GET', p);
const post = (p,b) => api('POST', p, b);

// ── Port scanning + mode detection ──────────────────────────────────────────

async function scanPorts() {
  try {
    const res = await get('/api/ports');
    const ports = res.ports || [];
    const cytonSel = $('cyton-port');
    const arduinoSel = $('arduino-port');

    // Rebuild Cyton port options
    cytonSel.innerHTML = '';
    if (ports.length === 0) {
      cytonSel.innerHTML = '<option value="">No ports found</option>';
    } else {
      ports.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.device;
        opt.textContent = p.device + ' - ' + p.description;
        cytonSel.appendChild(opt);
      });
    }

    // Rebuild Arduino port options
    arduinoSel.innerHTML = '<option value="">None</option>';
    ports.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.device;
      opt.textContent = p.device + ' - ' + p.description;
      arduinoSel.appendChild(opt);
    });

    toast('Found ' + ports.length + ' port(s)');
  } catch (e) {
    toast('Port scan failed: ' + e.message, 'red');
  }
}

async function loadStatus() {
  try {
    const s = await get('/api/status');
    S.gestures   = s.gestures || [];
    S.trained    = s.model_trained;
    S.streaming  = s.stream_running;
    S.streamSource = s.stream_source || 'hardware';
    S.streamDetails = s.stream_details || {};
    S.playbackSessionId = s.playback_session_id || '';
    S.decoderMode = s.decoder_mode || S.decoderMode;
    S.recSession = s.is_recording;
    applyLSLStatus(s.lsl || {});
    applyOSCStatus(s.osc || {});
    applySignalProfile(s.signal_profile || {});
    applyProtocolTemplates(s.protocol_templates || []);
    applyEEGExperiments(s.eeg_experiments || []);
    applyCalibrationState(s.calibration || {});
    applyEEGBrainView(s.eeg_brain || {});
    applyDiagnostics(s.diagnostics || {});
    applySafety(s.safety || {});
    applyFilterLabStatus(s.filter_lab || {});
    applyWorkshopStatus(s.workshop || {});
    S.lastPrediction = clonePredictionPayload(s.last_prediction);
    setSysState(s.state);
    syncStreamModeUI();
    syncProfileUI();
    syncProtocolUI();
    syncEEGExperimentUI();
    syncEEGMarkerHelperUI();
    syncLSLMarkerTesterUI();
    syncCalibrationUI();
    syncLSLUI();
    syncOSCUI();
    syncReviewUI();
    syncPredictionPanel();
    syncWorkshopUI();
    refreshFilterLabUI();
    buildGestureList();
    buildQuickGestures();
    buildGestureMappingUI();
    refreshTrainSummary();
  } catch { /* server might not be up yet, that's fine */ }
}

async function loadConfig() {
  try {
    const c = await get('/api/config');
    S.sampleRate = Number(c.sample_rate || S.sampleRate || 250);
    S.decoderMode = c.decoder_mode || S.decoderMode;
    applySignalProfile(c.signal_profile || {});
    applyProtocolTemplates(c.protocol_templates || []);
    applyEEGExperiments(c.eeg_experiments || []);
    applyCalibrationState({ protocol: c.calibration_protocol || null });
    applyEEGBrainView(c.eeg_brain || {});
    applyOSCStatus(c.osc || S.osc);
    applyDiagnostics(c.diagnostics || {});
    applySafety(c.safety || {});
    applyFilterLabStatus(c.filter_lab || {});
    applyWorkshopStatus(c.workshop || {});
    S.gestures = c.class_labels || c.gestures || S.gestures;
    S.availableProfiles = c.available_profiles || [];
    S.channelLabels = (c.channel_labels || []).slice(0, N_CH);
    while (S.channelLabels.length < N_CH) S.channelLabels.push(`CH${S.channelLabels.length + 1}`);
    primeFilterDefaultsFromProfile();
    syncProfileUI();
    syncProtocolUI();
    syncEEGExperimentUI();
    syncEEGMarkerHelperUI();
    syncLSLMarkerTesterUI();
    syncWorkshopUI();
    buildLegend();
    buildRmsBars();
    buildGestureList();
    buildQuickGestures();
    buildGestureMappingUI();
    syncCalibrationUI();
    syncReviewUI();
    refreshFilterLabUI();
  } catch { /* config fetch can fail during startup */ }
}

function applySignalProfile(profile) {
  if (!profile || !profile.key) return;
  S.signalProfileKey = profile.key;
  S.signalProfileName = profile.display_name || profile.key.toUpperCase();
  S.signalDescription = profile.description || `${S.signalProfileName} profile ready.`;
  S.signalSupportLevel = profile.support_level || 'profile';
  S.signalUnits = profile.units || 'a.u.';
  S.signalFullScale = profile.display_full_scale || 1.0;
  S.signalMetricLabel = profile.metric_label || 'Channel Activity';
  S.signalMetricScale = profile.metric_full_scale || 1.0;
  S.muteFloor = profile.mute_floor || 0.01;
  S.supportsTraining = !!profile.training_supported;
  S.supportsArmGestures = !!profile.robotic_arm_supported;
  S.workshop.result = null;
  S.workshop.selectionMeta = null;
  S.workshop.lastRequest = null;
  loadChannelMask();
  resetReviewState({ clearMarkers: true });
  syncResearchUI();
  primeFilterDefaultsFromProfile();
  syncFilterChainLabel();
}

function applyProtocolTemplates(templates) {
  S.protocolTemplates = Array.isArray(templates) ? templates : [];
  const select = $('protocol-template');
  const current = select?.value || '';
  const valid = S.protocolTemplates.some(template => template.key === current);
  if (valid) return;
  resetProtocolRun();
  const preferred = S.protocolTemplates.find(template => template.key === 'balanced_decoder_run') || S.protocolTemplates[0];
  if (select) select.value = preferred?.key || '';
}

function applyEEGExperiments(items) {
  S.eegExperiments = Array.isArray(items) ? items : [];
  const select = $('eeg-experiment-select');
  const current = select?.value || S.selectedEegExperiment || localStorage.getItem('kyma-eeg-experiment') || '';
  const valid = S.eegExperiments.some(item => item.key === current);
  S.selectedEegExperiment = valid ? current : (S.eegExperiments[0]?.key || '');
  if (S.selectedEegExperiment) {
    localStorage.setItem('kyma-eeg-experiment', S.selectedEegExperiment);
  } else {
    localStorage.removeItem('kyma-eeg-experiment');
  }
}

function applyCalibrationState(payload) {
  if (!payload) return;
  if (payload.stage) S.calibrationStage = payload.stage;
  if (payload.protocol) S.calibrationProtocol = payload.protocol;
}

function applyEEGBrainView(payload) {
  if (!payload) return;
  S.eegBrain = {
    ...S.eegBrain,
    ...payload,
    topomap_url: payload.topomap_url || '',
    sensors_url: payload.sensors_url || '',
    surface_url: payload.surface_url || '',
    surface_available: !!(payload.surface_available || payload.surface_url),
    note: payload.note || payload.reason || S.eegBrain.note || '',
    surface_note: payload.surface_note || payload.surface_reason || S.eegBrain.surface_note || '',
    dominant_band: payload.dominant_band || '',
  };

  const note = $('eeg-brain-note');
  const band = $('eeg-brain-band');
  const topomap = $('eeg-brain-topomap');
  const sensors = $('eeg-brain-sensors');
  const markerLink = $('eeg-brain-marker-link');
  const markerSummary = $('eeg-brain-marker-summary');

  const setMediaSrc = (el, url, options = {}) => {
    if (!el) return;
    const next = String(url || '').trim();
    if (next) {
      const current = String(el.getAttribute('src') || '').trim();
      if (options.once && current) {
        el.style.display = '';
        return;
      }
      if (current === next || el.dataset.pendingSrc === next) {
        el.style.display = '';
        return;
      }
      el.dataset.pendingSrc = next;
      const probe = new Image();
      probe.decoding = 'async';
      probe.onload = () => {
        if (el.dataset.pendingSrc !== next) return;
        el.src = next;
        el.style.display = '';
        delete el.dataset.pendingSrc;
      };
      probe.onerror = () => {
        if (el.dataset.pendingSrc === next) delete el.dataset.pendingSrc;
      };
      probe.src = next;
    } else {
      if (!options.preserve) {
        el.removeAttribute('src');
        el.style.display = 'none';
      }
    }
  };

  if (note) note.textContent = S.eegBrain.note || 'Switch to the EEG profile and start a stream to generate MNE/Nilearn views.';
  if (band) {
    band.textContent = S.eegBrain.dominant_band
      ? `Dominant band: ${String(S.eegBrain.dominant_band).toUpperCase()}`
      : 'Waiting for EEG data.';
  }

  setMediaSrc(topomap, S.eegBrain.available ? S.eegBrain.topomap_url : '', { preserve: true });
  setMediaSrc(sensors, S.eegBrain.available ? S.eegBrain.sensors_url : '', { once: true, preserve: true });

  if (markerLink) {
    if (S.eegBrain.surface_available && S.eegBrain.surface_url) {
      const resolved = new URL(S.eegBrain.surface_url, location.origin).toString();
      markerLink.href = resolved;
      markerLink.dataset.href = resolved;
      markerLink.setAttribute('aria-disabled', 'false');
      markerLink.title = 'Open Nilearn electrode reference in a new tab';
    } else {
      markerLink.href = '#';
      markerLink.dataset.href = '';
      markerLink.setAttribute('aria-disabled', 'true');
      markerLink.title = S.eegBrain.surface_note || 'Nilearn marker reference is not available yet.';
    }
  }
  if (markerSummary) {
    markerSummary.textContent = S.eegBrain.surface_available
      ? 'This opens Nilearn in a separate page. It is a spatial reference only, not source localization.'
      : (S.eegBrain.surface_note || 'Nilearn marker reference is not available yet.');
  }
}

async function refreshEEGBrainView(force = false) {
  if (S.signalProfileKey !== 'eeg') return;
  if (S.review.paused && !force) return;
  if (S.eegBrain.loading) return;
  const now = Date.now();
  if (!force && now - (S.eegBrain.lastRefresh || 0) < 1200) return;

  S.eegBrain.loading = true;
  S.eegBrain.lastRefresh = now;
  try {
    const payload = await get('/api/eeg/brain-view');
    if (S.review.paused && !force) return;
    applyEEGBrainView(payload || {});
  } catch (e) {
    if (S.review.paused && !force) return;
    applyEEGBrainView({
      available: false,
      note: e.message || 'Failed to load EEG brain view.',
      surface_available: false,
      surface_note: '',
      dominant_band: '',
      topomap_url: '',
      sensors_url: '',
      surface_url: '',
    });
  } finally {
    S.eegBrain.loading = false;
  }
}

function workshopViewTitle(key) {
  return ({
    fft: 'Fourier / FFT',
    psd: 'Power Spectral Density',
    spectrogram: 'Spectrogram',
    autocorrelation: 'Autocorrelation',
    histogram: 'Amplitude Histogram',
    envelope: 'Hilbert Envelope',
    correlation: 'Cross-Channel Correlation',
    laplace: 'Laplace Surface',
  })[key] || 'Signal Workshop';
}

function workshopViewNote(key) {
  return ({
    fft: 'FFT shows the chunk’s frequency magnitude relative to its strongest spectral component.',
    psd: 'PSD uses Welch averaging so you can inspect band energy without relying on a single FFT slice.',
    spectrogram: 'Spectrogram shows how the selected chunk’s frequency content changes across time.',
    autocorrelation: 'Autocorrelation helps reveal periodicity, rhythm, and repeat intervals inside the chunk.',
    histogram: 'Histogram shows amplitude distribution and spread. A narrow histogram suggests lower variance.',
    envelope: 'Hilbert envelope compares the raw chunk to its amplitude envelope for burst-like activity.',
    correlation: 'Cross-channel correlation shows which channels rise and fall together in the selected chunk.',
    laplace: 'Laplace view is a numeric magnitude surface over frequency and sigma. It is not system identification.',
  })[key] || 'Selected transform for the current chunk.';
}

function applyWorkshopStatus(payload) {
  S.workshop.available = !!payload?.available;
  S.workshop.last_error = String(payload?.last_error || '');
  const views = Array.isArray(payload?.views) && payload.views.length ? payload.views : S.workshop.views;
  S.workshop.views = views.slice();
  if (!S.workshop.views.includes(S.workshop.view)) {
    S.workshop.view = S.workshop.views[0] || 'fft';
  }
}

function populateWorkshopFocusOptions() {
  const sel = $('workshop-focus-channel');
  if (!sel) return;
  const labels = Array.isArray(S.workshop.result?.channel_labels) && S.workshop.result.channel_labels.length
    ? S.workshop.result.channel_labels
    : S.channelLabels;
  const current = String(sel.value || S.workshop.selectionMeta?.focusChannel || 0);
  sel.innerHTML = '';
  labels.slice(0, N_CH).forEach((label, idx) => {
    const opt = document.createElement('option');
    opt.value = String(idx);
    opt.textContent = label || `CH${idx + 1}`;
    opt.selected = opt.value === current;
    sel.appendChild(opt);
  });
  if (![...sel.options].some(opt => opt.selected) && sel.options.length) {
    sel.value = String(Math.max(0, Math.min(Number(current || 0), sel.options.length - 1)));
  }
}

function workshopSelectionSummary() {
  if (S.workshop.selectionMeta) return S.workshop.selectionMeta;
  if (S.review.lastStats && S.review.snapshot) {
    return {
      sourceLabel: 'Selected review range',
      rangeLabel: `${S.review.lastStats.focusLabel} window`,
      samples: S.review.lastStats.samples,
      durationMs: S.review.lastStats.durationMs,
      focusChannel: S.review.lastStats.focusChannel,
    };
  }
  if (S.review.snapshot) {
    return {
      sourceLabel: 'Entire frozen window',
      rangeLabel: 'Full paused display',
      samples: Math.min(Number(S.review.snapshot.total || 0), DISPLAY_SAMPLES),
      durationMs: (Math.min(Number(S.review.snapshot.total || 0), DISPLAY_SAMPLES) / Math.max(Number(S.review.snapshot.sampleRate || 250), 1)) * 1000,
      focusChannel: 0,
    };
  }
  return null;
}

function syncWorkshopUI() {
  const sel = $('workshop-view');
  const focusSel = $('workshop-focus-channel');
  const analyzeBtn = $('btn-workshop-analyze');
  const pullBtn = $('btn-workshop-from-review');
  const refreshBtn = $('btn-workshop-refresh');
  const status = $('workshop-status');
  const selection = workshopSelectionSummary();

  if (sel) {
    const current = S.workshop.view || 'fft';
    if (!sel.querySelector(`option[value="${current}"]`)) {
      sel.innerHTML = '';
      S.workshop.views.forEach(view => {
        const opt = document.createElement('option');
        opt.value = view;
        opt.textContent = workshopViewTitle(view);
        sel.appendChild(opt);
      });
    }
    sel.value = current;
  }

  populateWorkshopFocusOptions();

  if ($('workshop-view-title')) $('workshop-view-title').textContent = workshopViewTitle(S.workshop.view);
  if ($('workshop-view-note')) $('workshop-view-note').textContent = workshopViewNote(S.workshop.view);

  if ($('workshop-selection-source')) $('workshop-selection-source').textContent = selection?.sourceLabel || 'Review freeze';
  if ($('workshop-selection-range')) $('workshop-selection-range').textContent = selection?.rangeLabel || 'No chunk selected';
  if ($('workshop-selection-samples')) $('workshop-selection-samples').textContent = selection ? `${selection.samples}` : '--';
  if ($('workshop-selection-duration')) $('workshop-selection-duration').textContent = selection ? `${Number(selection.durationMs || 0).toFixed(1)} ms` : '-- ms';
  if ($('workshop-selection-note')) {
    $('workshop-selection-note').textContent = selection
      ? 'This chunk comes from the paused review buffer. Change the focus channel or analysis view, then analyze it.'
      : 'Pause the live display, drag a chunk on the waveform, then send it here for deeper analysis.';
  }

  if (analyzeBtn) analyzeBtn.disabled = !S.workshop.available || S.workshop.loading || !S.review.snapshot;
  if (pullBtn) pullBtn.disabled = !S.review.snapshot;
  if (refreshBtn) refreshBtn.disabled = !S.workshop.available || S.workshop.loading || !S.workshop.lastRequest;
  if (focusSel) focusSel.disabled = !S.workshop.available || S.workshop.loading;
  if (sel) sel.disabled = !S.workshop.available;

  const summary = S.workshop.result?.summary || null;
  if ($('workshop-summary-profile')) $('workshop-summary-profile').textContent = S.workshop.result?.profile?.toUpperCase?.() || S.signalProfileName || '--';
  if ($('workshop-summary-focus')) $('workshop-summary-focus').textContent = summary?.focus_label || '--';
  if ($('workshop-summary-dominant')) $('workshop-summary-dominant').textContent = summary ? `${Number(summary.dominant_frequency_hz || 0).toFixed(2)} Hz` : '--';
  if ($('workshop-summary-centroid')) $('workshop-summary-centroid').textContent = summary ? `${Number(summary.spectral_centroid_hz || 0).toFixed(2)} Hz` : '--';
  if ($('workshop-summary-rms')) $('workshop-summary-rms').textContent = summary ? `${Number(summary.rms || 0).toFixed(5)} ${S.signalUnits}` : '--';
  if ($('workshop-summary-zc')) $('workshop-summary-zc').textContent = summary ? `${summary.zero_crossings}` : '--';
  if ($('workshop-summary-note')) {
    $('workshop-summary-note').textContent = summary
      ? `${summary.samples} samples across ${summary.channels} channel(s) at ${Number(S.workshop.result?.sample_rate || 0).toFixed(0)} Hz.`
      : 'Run an analysis to fill the chunk metrics and transform outputs.';
  }
  if ($('workshop-metric-mean')) $('workshop-metric-mean').textContent = summary ? `${Number(summary.mean || 0).toFixed(5)} ${S.signalUnits}` : '--';
  if ($('workshop-metric-ptp')) $('workshop-metric-ptp').textContent = summary ? `${Number(summary.peak_to_peak || 0).toFixed(5)} ${S.signalUnits}` : '--';
  if ($('workshop-metric-derivative')) $('workshop-metric-derivative').textContent = summary ? `${Number(summary.derivative_rms || 0).toFixed(5)}` : '--';

  const bandList = $('workshop-band-list');
  if (bandList) {
    bandList.innerHTML = '';
    const bands = S.workshop.result?.psd?.bands || null;
    if (!bands || !Object.keys(bands).length) {
      const empty = document.createElement('div');
      empty.className = 'setup-copy';
      empty.textContent = 'Run an analysis to populate band metrics.';
      bandList.appendChild(empty);
    } else {
      Object.entries(bands).forEach(([name, value]) => {
        const row = document.createElement('div');
        row.className = 'workshop-list-item';
        row.innerHTML = `<strong>${name.toUpperCase()}</strong><div class="workshop-note" style="margin-top:4px">${Number(value || 0).toFixed(6)}</div>`;
        bandList.appendChild(row);
      });
    }
  }

  const details = $('workshop-detail-list');
  if (details) {
    details.innerHTML = '';
    if (!summary) {
      const empty = document.createElement('div');
      empty.className = 'setup-copy';
      empty.textContent = 'No analysis has been run yet.';
      details.appendChild(empty);
    } else {
      const rows = [
        ['Sample Rate', `${Number(S.workshop.result?.sample_rate || 0).toFixed(0)} Hz`],
        ['Selection Label', S.workshop.result?.selection_label || 'selected_chunk'],
        ['Min / Max', `${Number(summary.min || 0).toFixed(5)} / ${Number(summary.max || 0).toFixed(5)} ${S.signalUnits}`],
        ['Std Dev', `${Number(summary.std || 0).toFixed(5)} ${S.signalUnits}`],
        ['Area', `${Number(summary.area || 0).toFixed(6)}`],
      ];
      rows.forEach(([label, value]) => {
        const row = document.createElement('div');
        row.className = 'workshop-list-item';
        row.innerHTML = `<strong>${label}</strong><div class="workshop-note" style="margin-top:4px">${value}</div>`;
        details.appendChild(row);
      });
    }
  }

  if (status) {
    if (!S.workshop.available) {
      status.textContent = S.workshop.last_error || 'Signal workshop is unavailable on this runtime.';
    } else if (S.workshop.loading) {
      status.textContent = 'Analyzing the selected chunk on the server.';
    } else if (S.workshop.result) {
      status.textContent = `Ready: ${workshopViewTitle(S.workshop.view)} for ${S.workshop.result.channel_labels?.length || 0} channel(s).`;
    } else if (S.review.snapshot) {
      status.textContent = 'Paused review chunk is ready. Analyze it here or send a new range from the waveform.';
    } else {
      status.textContent = 'Waiting for a review chunk.';
    }
  }

  renderWorkshopView();
}

function buildWorkshopRequestFromReview() {
  const state = getReviewRenderState();
  if (!state.paused || !state.filled) {
    throw new Error('Pause the display first so the workshop can use a frozen chunk.');
  }

  const range = getSelectionRange(S.review.selection) || {
    start: state.baseAbs,
    end: state.baseAbs + state.filled - 1,
  };
  const start = Math.max(state.baseAbs, range.start);
  const end = Math.min(state.baseAbs + state.filled - 1, range.end);
  if (end < start) {
    throw new Error('The selected chunk is empty.');
  }

  const sampleRate = Math.max(Number(state.sampleRate || S.sampleRate || 250), 1);
  const channelLabels = S.channelLabels.slice(0, N_CH);
  while (channelLabels.length < N_CH) channelLabels.push(`CH${channelLabels.length + 1}`);
  const channels = channelLabels.map((_, ch) => {
    const row = [];
    for (let abs = start; abs <= end; abs++) {
      const idx = bufferIndexForAbsSample(state, abs);
      row.push(Number(state.emg[ch]?.[idx] || 0));
    }
    return row;
  });
  const stats = S.review.lastStats;
  const focusChannel = Math.max(
    0,
    Math.min(
      Number($('workshop-focus-channel')?.value || stats?.focusChannel || 0),
      channelLabels.length - 1,
    ),
  );
  const selectionLabel = (S.review.selection
    ? (($('review-marker-event')?.value || '').trim() || 'selected_chunk')
    : 'frozen_window');

  const meta = {
    sourceLabel: S.review.selection ? 'Selected review range' : 'Entire frozen window',
    rangeLabel: `${(start / sampleRate).toFixed(3)}s -> ${(end / sampleRate).toFixed(3)}s`,
    samples: end - start + 1,
    durationMs: ((end - start + 1) / sampleRate) * 1000,
    focusChannel,
    selectionLabel,
  };
  S.workshop.selectionMeta = meta;
  populateWorkshopFocusOptions();

  return {
    profile: S.signalProfileKey,
    sample_rate: sampleRate,
    channel_labels: channelLabels,
    channels,
    focus_channel: focusChannel,
    selection_label: selectionLabel,
    selection_start_s: start / sampleRate,
    selection_end_s: end / sampleRate,
  };
}

function workshopCanvasBase() {
  return THEME_CANVAS[currentTheme] || THEME_CANVAS.dark;
}

function drawWorkshopPlaceholder(message) {
  const cv = $('workshop-main-canvas');
  if (!cv) return;
  const gx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  const theme = workshopCanvasBase();
  gx.clearRect(0, 0, W, H);
  gx.fillStyle = theme.bg;
  gx.fillRect(0, 0, W, H);
  gx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-dim') || '#727683';
  gx.font = '12px Segoe UI';
  gx.fillText(message, 18, H / 2);
}

function drawWorkshopLineChart({ x = [], series = [], xLabel = '', yLabel = '' }) {
  const cv = $('workshop-main-canvas');
  if (!cv) return;
  const gx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  const theme = workshopCanvasBase();
  const pad = { l: 52, r: 18, t: 14, b: 34 };
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  gx.clearRect(0, 0, W, H);
  gx.fillStyle = theme.bg;
  gx.fillRect(0, 0, W, H);

  const allY = series.flatMap(item => item.data || []);
  if (!x.length || !allY.length) {
    drawWorkshopPlaceholder('No transform data for the selected chunk.');
    return;
  }
  let minY = Math.min(...allY);
  let maxY = Math.max(...allY);
  if (!Number.isFinite(minY) || !Number.isFinite(maxY) || minY === maxY) {
    minY -= 1;
    maxY += 1;
  }
  const minX = Math.min(...x);
  const maxX = Math.max(...x);
  const xSpan = Math.max(maxX - minX, 1e-9);
  const ySpan = Math.max(maxY - minY, 1e-9);

  gx.strokeStyle = theme.grid;
  gx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (plotH / 4) * i;
    gx.beginPath();
    gx.moveTo(pad.l, y);
    gx.lineTo(W - pad.r, y);
    gx.stroke();
  }
  for (let i = 0; i <= 5; i++) {
    const xPos = pad.l + (plotW / 5) * i;
    gx.beginPath();
    gx.moveTo(xPos, pad.t);
    gx.lineTo(xPos, H - pad.b);
    gx.stroke();
  }

  gx.strokeStyle = 'rgba(40,44,52,0.25)';
  gx.beginPath();
  gx.moveTo(pad.l, pad.t);
  gx.lineTo(pad.l, H - pad.b);
  gx.lineTo(W - pad.r, H - pad.b);
  gx.stroke();

  series.forEach((item, idx) => {
    const color = item.color || chColors()[idx % chColors().length];
    gx.strokeStyle = color;
    gx.lineWidth = item.width || 2;
    gx.beginPath();
    item.data.forEach((yVal, i) => {
      const xVal = x[Math.min(i, x.length - 1)];
      const xPx = pad.l + ((xVal - minX) / xSpan) * plotW;
      const yPx = H - pad.b - ((yVal - minY) / ySpan) * plotH;
      if (i === 0) gx.moveTo(xPx, yPx);
      else gx.lineTo(xPx, yPx);
    });
    gx.stroke();
  });

  gx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-dim') || '#727683';
  gx.font = '10px Segoe UI';
  gx.fillText(`${minX.toFixed(2)}`, pad.l, H - 10);
  gx.fillText(`${maxX.toFixed(2)} ${xLabel}`.trim(), W - pad.r - 74, H - 10);
  gx.save();
  gx.translate(12, H / 2);
  gx.rotate(-Math.PI / 2);
  gx.fillText(yLabel, 0, 0);
  gx.restore();
}

function drawWorkshopHeatmap({ x = [], y = [], grid = [], xLabel = '', yLabel = '' }) {
  const cv = $('workshop-main-canvas');
  if (!cv) return;
  const gx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  const theme = workshopCanvasBase();
  const pad = { l: 54, r: 20, t: 14, b: 36 };
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  gx.clearRect(0, 0, W, H);
  gx.fillStyle = theme.bg;
  gx.fillRect(0, 0, W, H);

  if (!x.length || !y.length || !grid.length) {
    drawWorkshopPlaceholder('No heatmap data for the selected chunk.');
    return;
  }

  const flat = grid.flat();
  const minVal = Math.min(...flat);
  const maxVal = Math.max(...flat);
  const span = Math.max(maxVal - minVal, 1e-9);
  const cellW = plotW / Math.max(x.length, 1);
  const cellH = plotH / Math.max(y.length, 1);

  for (let yi = 0; yi < y.length; yi++) {
    for (let xi = 0; xi < x.length; xi++) {
      const value = Number(grid[yi]?.[xi] || 0);
      const norm = (value - minVal) / span;
      const hue = 220 - norm * 170;
      const light = 92 - norm * 46;
      gx.fillStyle = `hsl(${hue} 74% ${light}%)`;
      gx.fillRect(pad.l + xi * cellW, pad.t + yi * cellH, cellW + 0.6, cellH + 0.6);
    }
  }

  gx.strokeStyle = 'rgba(40,44,52,0.25)';
  gx.strokeRect(pad.l, pad.t, plotW, plotH);
  gx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-dim') || '#727683';
  gx.font = '10px Segoe UI';
  gx.fillText(`${x[0].toFixed(2)}`, pad.l, H - 10);
  gx.fillText(`${x[x.length - 1].toFixed(2)} ${xLabel}`.trim(), W - pad.r - 74, H - 10);
  gx.fillText(`${y[0].toFixed(2)} ${yLabel}`.trim(), 8, pad.t + 10);
  gx.fillText(`${y[y.length - 1].toFixed(2)}`, 8, pad.t + plotH);
}

function drawWorkshopMatrix({ labels = [], matrix = [] }) {
  const cv = $('workshop-main-canvas');
  if (!cv) return;
  const gx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  const theme = workshopCanvasBase();
  const pad = { l: 72, r: 24, t: 24, b: 54 };
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  gx.clearRect(0, 0, W, H);
  gx.fillStyle = theme.bg;
  gx.fillRect(0, 0, W, H);

  if (!labels.length || !matrix.length) {
    drawWorkshopPlaceholder('No matrix data for the selected chunk.');
    return;
  }

  const size = labels.length;
  const cellW = plotW / Math.max(size, 1);
  const cellH = plotH / Math.max(size, 1);
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const value = Math.max(-1, Math.min(1, Number(matrix[row]?.[col] || 0)));
      const norm = (value + 1) / 2;
      const hue = 220 - norm * 170;
      const light = 92 - norm * 48;
      gx.fillStyle = `hsl(${hue} 74% ${light}%)`;
      gx.fillRect(pad.l + col * cellW, pad.t + row * cellH, cellW - 1, cellH - 1);
      gx.fillStyle = value > 0.55 ? '#ffffff' : '#223042';
      gx.font = '10px Segoe UI';
      gx.fillText(value.toFixed(2), pad.l + col * cellW + 6, pad.t + row * cellH + cellH / 2 + 3);
    }
  }
  gx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-dim') || '#727683';
  gx.font = '10px Segoe UI';
  labels.forEach((label, idx) => {
    gx.fillText(label, pad.l + idx * cellW + 4, H - 16);
    gx.fillText(label, 10, pad.t + idx * cellH + cellH / 2 + 3);
  });
}

function renderWorkshopView() {
  const result = S.workshop.result;
  if (!result) {
    drawWorkshopPlaceholder('Pause the display, select a chunk, then analyze it here.');
    return;
  }
  const view = S.workshop.view || 'fft';
  if (view === 'fft') {
    drawWorkshopLineChart({
      x: result.fft?.freq_hz || [],
      series: [{ data: result.fft?.mag_db || [], color: chColors()[0] }],
      xLabel: 'Hz',
      yLabel: 'dB',
    });
  } else if (view === 'psd') {
    drawWorkshopLineChart({
      x: result.psd?.freq_hz || [],
      series: [{ data: result.psd?.psd_db || [], color: chColors()[1] || chColors()[0] }],
      xLabel: 'Hz',
      yLabel: 'PSD dB',
    });
  } else if (view === 'spectrogram') {
    drawWorkshopHeatmap({
      x: result.spectrogram?.time_s || [],
      y: result.spectrogram?.freq_hz || [],
      grid: result.spectrogram?.mag_db_grid || [],
      xLabel: 's',
      yLabel: 'Hz',
    });
  } else if (view === 'autocorrelation') {
    drawWorkshopLineChart({
      x: result.autocorrelation?.lags_ms || [],
      series: [{ data: result.autocorrelation?.values || [], color: chColors()[2] || chColors()[0] }],
      xLabel: 'ms',
      yLabel: 'corr',
    });
  } else if (view === 'histogram') {
    drawWorkshopLineChart({
      x: result.histogram?.bin_centers || [],
      series: [{ data: result.histogram?.counts || [], color: chColors()[3] || chColors()[0] }],
      xLabel: S.signalUnits,
      yLabel: 'count',
    });
  } else if (view === 'envelope') {
    drawWorkshopLineChart({
      x: result.envelope?.time_ms || [],
      series: [
        { data: result.envelope?.signal || [], color: chColors()[0], width: 1.5 },
        { data: result.envelope?.envelope || [], color: chColors()[4] || '#d18b3f', width: 2.2 },
      ],
      xLabel: 'ms',
      yLabel: S.signalUnits,
    });
  } else if (view === 'correlation') {
    drawWorkshopMatrix({
      labels: result.correlation?.labels || [],
      matrix: result.correlation?.matrix || [],
    });
  } else if (view === 'laplace') {
    drawWorkshopHeatmap({
      x: result.laplace?.freq_hz || [],
      y: result.laplace?.sigma || [],
      grid: result.laplace?.mag_db_grid || [],
      xLabel: 'Hz',
      yLabel: 'sigma',
    });
  } else {
    drawWorkshopPlaceholder('Unsupported workshop view.');
  }
}

async function analyzeWorkshopSelection({ switchToTab = false } = {}) {
  if (!S.workshop.available) {
    toast(S.workshop.last_error || 'Signal workshop is unavailable.', 'red');
    return;
  }
  if (switchToTab && typeof window.switchTab === 'function') window.switchTab('workshop');
  let body;
  try {
    body = buildWorkshopRequestFromReview();
  } catch (e) {
    toast(e.message, 'yellow');
    syncWorkshopUI();
    return;
  }

  const view = $('workshop-view')?.value || S.workshop.view || 'fft';
  S.workshop.view = view;
  S.workshop.loading = true;
  S.workshop.lastRequest = body;
  syncWorkshopUI();

  try {
    const out = await post('/api/workshop/analyze', body);
    S.workshop.result = out || null;
    S.workshop.last_error = '';
    toast(`${workshopViewTitle(S.workshop.view)} ready`);
  } catch (e) {
    S.workshop.last_error = e.message || 'Workshop analysis failed.';
    toast(S.workshop.last_error, 'red');
  } finally {
    S.workshop.loading = false;
    syncWorkshopUI();
  }
}

async function loadSubjects() {
  try {
    const list = await get('/api/subjects');
    S.subjects = Array.isArray(list) ? list : [];
  } catch {
    S.subjects = [];
  }
  syncSubjectRegistryUI();
}

function findSubjectRecord(subjectId) {
  const wanted = String(subjectId || '').trim().toLowerCase();
  if (!wanted) return null;
  return S.subjects.find(subject => String(subject.subject_id || '').trim().toLowerCase() === wanted) || null;
}

function populateSubjectRegistryForm(record) {
  $('subject-registry-id').value = record?.subject_id || '';
  $('subject-registry-name').value = record?.display_name || '';
  $('subject-registry-cohort').value = record?.cohort || '';
  $('subject-registry-handedness').value = record?.handedness || '';
  $('subject-registry-notes').value = record?.notes || '';
}

function applySubjectToSession(record) {
  if (!record) return;
  const subjectInput = $('session-subject');
  if (subjectInput) subjectInput.value = record.subject_id || '';
}

function syncSubjectRegistryUI() {
  const select = $('subject-registry-select');
  const status = $('subject-registry-status');
  const currentSubjectId = $('session-subject')?.value?.trim() || '';
  const selectedId = select?.value || currentSubjectId || '';
  const active = findSubjectRecord(selectedId);

  if (select) {
    select.innerHTML = '';
    if (!S.subjects.length) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No subjects saved';
      select.appendChild(opt);
    } else {
      const blank = document.createElement('option');
      blank.value = '';
      blank.textContent = 'Select subject';
      select.appendChild(blank);
      S.subjects.forEach(subject => {
        const opt = document.createElement('option');
        opt.value = subject.subject_id || '';
        const name = subject.display_name ? ` - ${subject.display_name}` : '';
        const cohort = subject.cohort ? ` (${subject.cohort})` : '';
        opt.textContent = `${subject.subject_id || ''}${name}${cohort}`;
        opt.selected = subject.subject_id === (active?.subject_id || '');
        select.appendChild(opt);
      });
    }
  }

  if (active) {
    populateSubjectRegistryForm(active);
    if (status) {
      const bits = [];
      if (active.display_name) bits.push(active.display_name);
      if (active.cohort) bits.push(active.cohort);
      if (active.handedness) bits.push(active.handedness);
      if (active.session_count != null) bits.push(`${active.session_count} sessions`);
      status.textContent = bits.join(' | ') || 'Saved subject loaded.';
    }
  } else {
    if (currentSubjectId && !findSubjectRecord(currentSubjectId)) {
      populateSubjectRegistryForm({ subject_id: currentSubjectId });
      if (status) status.textContent = 'Unsaved subject ID. Save it to use the registry and subject-holdout workflows.';
    } else if (!currentSubjectId && !select?.value) {
      populateSubjectRegistryForm(null);
      if (status) status.textContent = 'Saved subjects feed leave-one-subject-out evaluation and session tagging.';
    }
  }
}

function syncCalibrationUI() {
  const summary = $('calibration-summary');
  if (!summary) return;

  const protocol = S.calibrationProtocol;
  if (!protocol) {
    summary.textContent = 'Calibration protocol information is loading.';
    return;
  }

  summary.innerHTML = '';

  const meta = document.createElement('div');
  meta.className = 'setup-meta';
  const title = document.createElement('span');
  title.textContent = protocol.title || `${S.signalProfileName} calibration`;
  const stage = document.createElement('span');
  stage.textContent = (S.calibrationStage || 'idle').replace(/_/g, ' ');
  meta.append(title, stage);

  const copy = document.createElement('div');
  copy.className = 'setup-copy';
  copy.textContent = protocol.summary || `${S.signalProfileName} calibration protocol ready.`;

  summary.append(meta, copy);

  if (Array.isArray(protocol.instructions) && protocol.instructions.length) {
    const steps = document.createElement('div');
    steps.className = 'setup-copy';
    steps.style.marginTop = '4px';
    steps.textContent = protocol.instructions.join(' | ');
    summary.appendChild(steps);
  }
}

function getSelectedSource() {
  return $('stream-source')?.value || localStorage.getItem('kyma-stream-source') || S.streamSource || 'hardware';
}

function setSelectedSource(source) {
  const value = source || 'hardware';
  const sel = $('stream-source');
  if (sel) sel.value = value;
  localStorage.setItem('kyma-stream-source', value);
}

function setSelectedPlaybackSession(sessionId) {
  const value = sessionId || '';
  const sel = $('playback-session');
  if (sel) sel.value = value;
  if (value) localStorage.setItem('kyma-playback-session', value);
  else localStorage.removeItem('kyma-playback-session');
}

function populatePlaybackSessionOptions() {
  const sel = $('playback-session');
  if (!sel) return;

  const previous = sel.value || localStorage.getItem('kyma-playback-session') || S.playbackSessionId || '';
  const playable = (S.sessions || []).filter(s => s.playable);
  sel.innerHTML = '';

  if (!playable.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No playable sessions found';
    sel.appendChild(opt);
    return;
  }

  playable.forEach(session => {
    const opt = document.createElement('option');
    opt.value = session.session_id;
    const label = session.label ? `${session.label} - ${session.session_id}` : session.session_id;
    const profile = session.signal_profile_name || session.signal_profile || 'Signal';
    opt.textContent = `${label} (${profile})`;
    opt.selected = session.session_id === previous;
    sel.appendChild(opt);
  });

  if (!sel.value && playable[0]) {
    sel.value = playable[0].session_id;
  }
}

function getSelectedProtocolTemplate() {
  const key = $('protocol-template')?.value || '';
  return S.protocolTemplates.find(template => template.key === key) || null;
}

function buildProtocolPlan(template) {
  if (!template) return [];
  const labels = Array.isArray(template.labels) ? template.labels.filter(Boolean) : [];
  const repetitions = Math.max(1, Number(template.repetitions || 1));
  const plan = [];
  let trialIndex = 1;
  for (let rep = 1; rep <= repetitions; rep += 1) {
    labels.forEach(label => {
      plan.push({
        label,
        trial_index: trialIndex,
        repetition_index: rep,
      });
      trialIndex += 1;
    });
  }
  return plan;
}

function generateProtocolRunId(template) {
  const subject = ($('session-subject')?.value || 'anon').trim() || 'anon';
  const condition = ($('session-condition')?.value || 'baseline').trim() || 'baseline';
  const stem = `${subject}_${condition}_${template?.key || S.signalProfileKey}_${new Date().toISOString().replace(/[-:TZ.]/g, '').slice(0, 14)}`;
  return stem.replace(/[^A-Za-z0-9._-]+/g, '_').slice(0, 60);
}

function ensureProtocolRunId(template) {
  const input = $('protocol-run-id');
  const current = input?.value?.trim() || S.protocolRunId || '';
  if (current) {
    S.protocolRunId = current;
    return current;
  }
  const next = generateProtocolRunId(template);
  S.protocolRunId = next;
  if (input) input.value = next;
  return next;
}

function resetProtocolRun() {
  S.protocolStepIndex = 0;
  S.protocolRunId = '';
  const input = $('protocol-run-id');
  if (input) input.value = '';
}

function buildSessionStartPayload(override = {}) {
  const template = getSelectedProtocolTemplate();
  return {
    label: override.label ?? ($('session-label')?.value?.trim() || ''),
    subject_id: $('session-subject')?.value?.trim() || '',
    condition: $('session-condition')?.value?.trim() || '',
    notes: $('session-notes')?.value?.trim() || '',
    protocol_key: override.protocol_key ?? (template?.key || ''),
    protocol_title: override.protocol_title ?? (template?.title || ''),
    session_group_id: override.session_group_id ?? ($('protocol-run-id')?.value?.trim() || ''),
    trial_index: override.trial_index ?? null,
    repetition_index: override.repetition_index ?? null,
  };
}

function applyDiagnostics(payload) {
  if (!payload) return;
  S.diagnostics = {
    ...S.diagnostics,
    ...payload,
    spectrum: {
      ...(S.diagnostics.spectrum || {}),
      ...((payload && payload.spectrum) || {}),
    },
    noise: {
      ...(S.diagnostics.noise || {}),
      ...((payload && payload.noise) || {}),
    },
    timing: {
      ...(S.diagnostics.timing || {}),
      ...((payload && payload.timing) || {}),
    },
  };
  renderSpectrum();
  syncFilterChainLabel();
  if ($('bench-report')?.classList.contains('active')) refreshBenchReportUI();
}

function applySafety(payload) {
  if (!payload) return;
  S.safety = { ...S.safety, ...payload };
  if ($('bench-report')?.classList.contains('active')) refreshBenchReportUI();
}

function applyFilterLabStatus(payload) {
  if (!payload) return;
  S.filterLab = {
    ...S.filterLab,
    ...payload,
    filters: Array.isArray(payload.filters) ? payload.filters : (S.filterLab.filters || []),
    records: S.filterLab.records || {},
  };
  if (!S.filterLab.selected_filter_id) {
    S.filterLab.selected_filter_id = S.filterLab.active_filter_id || S.filterLab.filters[0]?.id || '';
  } else if (!S.filterLab.filters.some(item => item.id === S.filterLab.selected_filter_id)) {
    S.filterLab.selected_filter_id = S.filterLab.active_filter_id || S.filterLab.filters[0]?.id || '';
  }
  syncFilterChainLabel();
  if ($('block-editor')?.classList.contains('active')) {
    renderCanvas();
  }
}

function getFilterSummaryById(filterId) {
  const wanted = String(filterId || '');
  if (!wanted) return null;
  return (S.filterLab.filters || []).find(item => String(item.id || '') === wanted) || null;
}

function getCachedFilterRecord(filterId) {
  const wanted = String(filterId || '');
  if (!wanted) return null;
  return (S.filterLab.records || {})[wanted] || null;
}

async function ensureFilterRecord(filterId) {
  const wanted = String(filterId || '');
  if (!wanted) return null;
  const cached = getCachedFilterRecord(wanted);
  if (cached) return cached;
  try {
    const item = await get(`/api/filterlab/${wanted}`);
    S.filterLab.records = { ...(S.filterLab.records || {}), [wanted]: item };
    return item;
  } catch {
    return null;
  }
}

async function loadFilterLabStatus() {
  try {
    const status = await get('/api/filterlab/status');
    applyFilterLabStatus(status || {});
    if (S.filterLab.selected_filter_id) {
      await loadFilterRecord(S.filterLab.selected_filter_id, false);
    } else {
      S.filterLab.selected_filter = null;
      refreshFilterLabUI();
    }
  } catch {}
}

function syncProtocolUI() {
  const select = $('protocol-template');
  const summary = $('protocol-summary');
  const runInput = $('protocol-run-id');
  const nextBtn = $('btn-protocol-next');
  const template = getSelectedProtocolTemplate() || (S.protocolTemplates[0] || null);

  if (select) {
    const current = select.value || template?.key || '';
    select.innerHTML = '';
    if (!S.protocolTemplates.length) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No templates loaded';
      select.appendChild(opt);
    } else {
      S.protocolTemplates.forEach(item => {
        const opt = document.createElement('option');
        opt.value = item.key;
        opt.textContent = item.title || item.key;
        opt.selected = item.key === current;
        select.appendChild(opt);
      });
      if (!select.value && template) select.value = template.key;
    }
  }

  if (runInput && S.protocolRunId && !runInput.value) {
    runInput.value = S.protocolRunId;
  }

  const activeTemplate = getSelectedProtocolTemplate();
  const plan = buildProtocolPlan(activeTemplate);
  const nextStep = plan[S.protocolStepIndex] || null;

  if (summary) {
    if (!activeTemplate) {
      summary.textContent = 'Load a signal profile to see the recommended protocol templates.';
    } else {
      const labels = (activeTemplate.labels || []).join(', ') || 'no labels';
      const nextText = nextStep
        ? `Next: ${nextStep.label} (trial ${nextStep.trial_index}, rep ${nextStep.repetition_index})`
        : 'Run complete. Reset the run or choose another template.';
      const duration = Number(activeTemplate.estimated_duration_s || 0).toFixed(1);
      summary.textContent = `${activeTemplate.summary} Labels: ${labels}. ${activeTemplate.repetitions} repetition(s), ~${duration}s total. ${nextText}`;
    }
  }

  if (nextBtn) {
    nextBtn.disabled = !activeTemplate || !plan.length || S.recSession || !S.streaming || S.streamSource === 'playback';
    nextBtn.textContent = nextStep ? `Start ${nextStep.label}` : 'Run Complete';
  }
}

function getSelectedEEGExperiment() {
  const key = $('eeg-experiment-select')?.value || S.selectedEegExperiment || '';
  return S.eegExperiments.find(item => item.key === key) || null;
}

function syncEEGExperimentUI() {
  const select = $('eeg-experiment-select');
  const summary = $('eeg-experiment-summary');
  const applyBtn = $('btn-eeg-experiment-apply');
  const active = getSelectedEEGExperiment() || (S.eegExperiments[0] || null);

  if (select) {
    const current = select.value || active?.key || S.selectedEegExperiment || '';
    select.innerHTML = '';
    if (!S.eegExperiments.length) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No EEG presets loaded';
      select.appendChild(opt);
    } else {
      S.eegExperiments.forEach(item => {
        const opt = document.createElement('option');
        opt.value = item.key;
        opt.textContent = item.title || item.key;
        opt.selected = item.key === current;
        select.appendChild(opt);
      });
      if (!select.value && active) select.value = active.key;
      S.selectedEegExperiment = select.value || active?.key || '';
    }
    select.disabled = S.signalProfileKey !== 'eeg';
  }

  if (summary) {
    if (S.signalProfileKey !== 'eeg') {
      summary.textContent = 'These presets target EEG research runs. Switch Signal Type to EEG to use them.';
    } else if (!active) {
      summary.textContent = 'No EEG experiment presets are loaded.';
    } else {
      const exportText = Array.isArray(active.recommended_export) && active.recommended_export.length
        ? active.recommended_export.join(' + ').toUpperCase()
        : 'session export';
      const steps = Array.isArray(active.instructions) && active.instructions.length
        ? ` ${active.instructions.join(' | ')}`
        : '';
      summary.textContent = `${active.summary} Mode: ${(active.mode || 'record_only').replace(/_/g, ' ')}. Source: ${active.recommended_source || 'hardware'}. Export: ${exportText}. This prepares record/export metadata and does not change KYMA's live EEG class labels.${steps}`;
    }
  }

  if (applyBtn) {
    applyBtn.disabled = !active || S.signalProfileKey !== 'eeg';
    applyBtn.textContent = S.signalProfileKey === 'eeg' ? 'Apply To Session' : 'EEG Only';
  }
}

function syncEEGMarkerHelperUI() {
  const summary = $('eeg-marker-helper-summary');
  const strategy = $('eeg-marker-strategy');
  const markers = $('eeg-marker-list');
  const blocks = $('eeg-block-structure');
  const active = getSelectedEEGExperiment() || (S.eegExperiments[0] || null);

  if (!summary || !strategy || !markers || !blocks) return;

  if (S.signalProfileKey !== 'eeg') {
    summary.textContent = 'These marker hints are only relevant for EEG research presets.';
    strategy.textContent = '--';
    markers.textContent = '--';
    blocks.textContent = '--';
    return;
  }

  if (!active) {
    summary.textContent = 'Choose an EEG research preset first.';
    strategy.textContent = '--';
    markers.textContent = '--';
    blocks.textContent = '--';
    return;
  }

  summary.textContent = `${active.title}: use these names and block notes in the external stimulus presenter so KYMA recordings export with a consistent event structure.`;
  strategy.textContent = String(active.marker_strategy || 'external_lsl_markers').replace(/_/g, ' ');
  markers.textContent = Array.isArray(active.marker_names) && active.marker_names.length
    ? active.marker_names.join(' | ')
    : 'No marker names defined.';
  blocks.textContent = Array.isArray(active.block_structure) && active.block_structure.length
    ? active.block_structure.join(' | ')
    : 'No block structure guidance defined.';
}

function syncLSLMarkerTesterUI() {
  const select = $('lsl-marker-select');
  const eventInput = $('lsl-marker-event');
  const payloadInput = $('lsl-marker-payload');
  const status = $('lsl-marker-status');
  const sendBtn = $('btn-send-lsl-marker');
  const active = getSelectedEEGExperiment() || (S.eegExperiments[0] || null);
  const markerNames = Array.isArray(active?.marker_names) ? active.marker_names : [];

  if (!select || !eventInput || !payloadInput || !status || !sendBtn) return;

  const previous = select.value || localStorage.getItem('kyma-lsl-marker-name') || '';
  select.innerHTML = '';
  if (!markerNames.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No preset markers loaded';
    select.appendChild(opt);
  } else {
    markerNames.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      opt.selected = name === previous;
      select.appendChild(opt);
    });
    if (!select.value) select.value = markerNames[0];
  }

  if (!eventInput.value && select.value) {
    eventInput.value = select.value;
  }
  if (!payloadInput.value.trim()) {
    const seed = {
      subject_id: $('session-subject')?.value?.trim() || '',
      condition: $('session-condition')?.value?.trim() || '',
      profile: S.signalProfileKey || '',
    };
    payloadInput.value = JSON.stringify(seed, null, 2);
  }

  const lslReady = !!(S.lsl?.active && S.lsl?.include_markers);
  if (S.signalProfileKey !== 'eeg') {
    status.textContent = 'This tester is aimed at EEG research runs. Switch Signal Type to EEG to use preset markers.';
  } else if (!lslReady) {
    status.textContent = 'Start LSL output with markers enabled in Control, then send a test marker from here.';
  } else {
    status.textContent = `Sending to ${S.lsl.marker_stream_name || 'KYMA marker stream'}. Use this to verify your external recorder sees the expected marker names.`;
  }

  sendBtn.disabled = !lslReady;
}

function syncReviewUI() {
  const pauseBtn = $('btn-review-pause');
  const clearBtn = $('btn-review-clear-selection');
  const status = $('review-status');
  const chip = $('scope-status-chip');
  const list = $('review-marker-list');
  const markerBtn = $('btn-review-marker');
  const workshopBtn = $('btn-review-workshop');
  const container = $('canvas-container');
  const timelineLabel = $('timeline-label');
  const stats = S.review.lastStats;
  const paused = !!S.review.paused;

  if (pauseBtn) {
    pauseBtn.textContent = paused ? 'Resume Review' : 'Freeze Review';
    pauseBtn.disabled = !S.streaming && !S.review.snapshot;
  }
  if (clearBtn) clearBtn.disabled = !S.review.selection;
  if (markerBtn) {
    markerBtn.disabled = !S.streaming;
    markerBtn.textContent = S.review.lastStats ? 'Save Range Marker' : 'Save Marker';
  }
  if (workshopBtn) workshopBtn.disabled = !S.review.snapshot;
  if (container) container.classList.toggle('review-paused', paused);
  if (timelineLabel) timelineLabel.textContent = paused ? 'Frozen decoded output snapshot' : 'Last 60 decoded outputs';
  if (chip) {
    chip.textContent = paused ? 'Paused' : 'Live';
    chip.classList.toggle('paused', paused);
  }

  if ($('review-selection-span')) $('review-selection-span').textContent = stats ? `${stats.samples} samples` : 'None';
  if ($('review-selection-duration')) $('review-selection-duration').textContent = stats ? `${stats.durationMs.toFixed(1)} ms` : '-- ms';
  if ($('review-selection-rms')) $('review-selection-rms').textContent = stats ? `${stats.rms.toFixed(4)} ${S.signalUnits}` : '--';
  if ($('review-selection-ptp')) $('review-selection-ptp').textContent = stats ? `${stats.peakToPeak.toFixed(4)} ${S.signalUnits}` : '--';
  if ($('review-selection-mean')) $('review-selection-mean').textContent = stats ? `${stats.mean.toFixed(4)} ${S.signalUnits}` : '--';
  if ($('review-selection-focus')) {
    $('review-selection-focus').textContent = stats
      ? `${stats.focusLabel} (${stats.focusRms.toFixed(4)} ${S.signalUnits})`
      : '--';
  }

  if (status) {
    if (!S.streaming && !S.review.snapshot) {
      status.textContent = 'Start a stream first. Then pause the trace, drag a region, and save custom markers.';
    } else if (paused && stats) {
      status.textContent = 'Review freeze is active. The selected region is frozen, measured locally, and ready for markers or workshop analysis.';
    } else if (paused) {
      status.textContent = 'Review freeze is active. Drag across the waveform to inspect a region, or send the full frozen window to the workshop.';
    } else {
      status.textContent = 'Live display is running. Pause the trace, then drag across the waveform to inspect a region.';
    }
  }

  if (list) {
    list.innerHTML = '';
    if (!S.review.markers.length) {
      const empty = document.createElement('div');
      empty.className = 'setup-copy';
      empty.textContent = 'No custom markers saved yet.';
      list.appendChild(empty);
    } else {
      S.review.markers.forEach(item => {
        const row = document.createElement('div');
        row.className = 'review-marker-item';

        const head = document.createElement('div');
        head.className = 'review-marker-head';

        const name = document.createElement('span');
        name.className = 'review-marker-name';
        name.textContent = item.event;

        const range = document.createElement('span');
        range.className = 'review-marker-range';
        range.textContent = item.selection ? 'range' : 'point';

        const note = document.createElement('div');
        note.className = 'review-marker-note';
        const selectionText = item.selection
          ? ` ${Number(item.selection.start_s || 0).toFixed(3)}s -> ${Number(item.selection.end_s || 0).toFixed(3)}s`
          : '';
        note.textContent = `${item.note || 'No note.'}${selectionText}`.trim();

        head.append(name, range);
        row.append(head, note);
        list.appendChild(row);
      });
    }
  }
}

function setSelectedLSLInput(sourceIdOrName) {
  const value = sourceIdOrName || '';
  const sel = $('lsl-input-stream');
  if (sel) sel.value = value;
  if (value) localStorage.setItem('kyma-lsl-input', value);
  else localStorage.removeItem('kyma-lsl-input');
}

function getSelectedLSLInput() {
  return $('lsl-input-stream')?.value || localStorage.getItem('kyma-lsl-input') || '';
}

function populateLSLInputOptions() {
  const sel = $('lsl-input-stream');
  if (!sel) return;

  const previous = getSelectedLSLInput();
  sel.innerHTML = '';

  if (!S.lslInputs.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No external LSL streams found';
    sel.appendChild(opt);
    return;
  }

  S.lslInputs.forEach(stream => {
    const key = stream.source_id || stream.uid || stream.name;
    const opt = document.createElement('option');
    opt.value = key;
    const rate = stream.sample_rate ? `${Number(stream.sample_rate).toFixed(0)} Hz` : 'irregular';
    opt.textContent = `${stream.name || 'Stream'} (${stream.type || 'signal'}, ${stream.n_channels || 0} ch, ${rate})`;
    opt.selected = key === previous;
    sel.appendChild(opt);
  });

  if (!sel.value && sel.options.length) {
    sel.value = sel.options[0].value;
  }
}

function syncLSLInputUI(message) {
  const wrap = $('lsl-input-status');
  const sel = $('lsl-input-stream');
  const btn = $('btn-refresh-lsl-inputs');
  if (!wrap || !sel || !btn) return;

  const activeKey = getSelectedLSLInput();
  const active = S.lslInputs.find(stream =>
    (stream.source_id || stream.uid || stream.name) === activeKey
  );
  const meta = document.createElement('div');
  meta.className = 'setup-meta';
  const state = document.createElement('span');
  state.textContent = S.lslInputs.length ? `${S.lslInputs.length} stream${S.lslInputs.length === 1 ? '' : 's'}` : 'No streams';
  const source = document.createElement('span');
  source.textContent = active?.type || 'LSL input';
  meta.append(state, source);

  const copy = document.createElement('div');
  copy.className = 'setup-copy';
  if (message) {
    copy.textContent = message;
  } else if (active) {
    const rate = active.sample_rate ? `${Number(active.sample_rate).toFixed(0)} Hz` : 'irregular rate';
    copy.textContent = `${active.name || 'Stream'} | ${active.n_channels || 0} ch | ${rate}${active.source_id ? ` | ${active.source_id}` : ''}`;
  } else {
    copy.textContent = 'Select an external numeric LSL stream to use as the live source.';
  }

  wrap.innerHTML = '';
  wrap.append(meta, copy);
  sel.disabled = S.streaming;
  btn.disabled = false;
}

async function loadLSLInputs() {
  try {
    const payload = await get('/api/lsl/inputs');
    S.lslInputs = Array.isArray(payload.streams) ? payload.streams : [];
    populateLSLInputOptions();
    setSelectedLSLInput(getSelectedLSLInput() || $('lsl-input-stream')?.value || '');
    syncLSLInputUI(payload.last_error || '');
  } catch (e) {
    S.lslInputs = [];
    populateLSLInputOptions();
    syncLSLInputUI(e.message || 'LSL input scan failed.');
  }
}

function populateXDFStreamOptions(streams) {
  const sel = $('xdf-stream');
  if (!sel) return;

  S.xdfStreams = Array.isArray(streams) ? streams : [];
  sel.innerHTML = '';

  if (!S.xdfStreams.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No numeric streams found';
    sel.appendChild(opt);
    return;
  }

  S.xdfStreams.forEach(stream => {
    const key = stream.stream_id || stream.name;
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = `${stream.name || 'Stream'} (${stream.type || 'signal'}, ${stream.channel_count || 0} ch, ${stream.sample_rate || '?'} Hz)`;
    sel.appendChild(opt);
  });
}

function applyLSLStatus(status) {
  if (!status) return;
  S.lsl = {
    available: !!status.available,
    active: !!status.active,
    include_markers: status.include_markers !== false,
    stream_name: status.stream_name || '',
    marker_stream_name: status.marker_stream_name || '',
    last_error: status.last_error || '',
  };
}

function syncLSLUI() {
  const wrap = $('lsl-status');
  const btn = $('btn-lsl');
  const nameInput = $('lsl-name');
  const markerChk = $('lsl-markers');
  if (!wrap || !btn || !nameInput || !markerChk) return;

  const storedName = localStorage.getItem('kyma-lsl-name') || '';
  const storedMarkers = localStorage.getItem('kyma-lsl-markers');

  if (S.lsl.active && S.lsl.stream_name) {
    nameInput.value = S.lsl.stream_name;
  } else if (!nameInput.value && storedName) {
    nameInput.value = storedName;
  }
  if (!storedName && !nameInput.value) {
    nameInput.value = `KYMA_${S.signalProfileName.toUpperCase()}`;
  }
  if (S.lsl.active) {
    markerChk.checked = S.lsl.include_markers !== false;
  } else if (storedMarkers !== null) {
    markerChk.checked = storedMarkers === '1';
  } else {
    markerChk.checked = true;
  }

  const meta = document.createElement('div');
  meta.className = 'setup-meta';
  const state = document.createElement('span');
  state.textContent = S.lsl.active ? 'Active' : (S.lsl.available ? 'Ready' : 'Unavailable');
  const mode = document.createElement('span');
  mode.textContent = S.lsl.active && S.lsl.stream_name ? S.lsl.stream_name : 'LSL';
  meta.append(state, mode);

  const copy = document.createElement('div');
  copy.className = 'setup-copy';
  if (!S.lsl.available) {
    copy.textContent = S.lsl.last_error || 'pylsl is not installed. Install requirements-research.txt to enable LSL.';
  } else if (S.lsl.active) {
    const markerCopy = S.lsl.marker_stream_name ? ` | Markers: ${S.lsl.marker_stream_name}` : '';
    copy.textContent = `Signal: ${S.lsl.stream_name}${markerCopy}`;
  } else {
    copy.textContent = 'Publish live biosignal samples and decoded markers over Lab Streaming Layer.';
  }

  wrap.innerHTML = '';
  wrap.append(meta, copy);
  btn.textContent = S.lsl.active ? 'Stop LSL' : 'Start LSL';
  btn.className = S.lsl.active ? 'btn danger' : 'btn';
  btn.disabled = !S.lsl.available;
  nameInput.disabled = S.lsl.active || !S.lsl.available;
  markerChk.disabled = S.lsl.active || !S.lsl.available;
  syncLSLMarkerTesterUI();
}

async function refreshLSLStatus() {
  try {
    const status = await get('/api/lsl/status');
    applyLSLStatus(status);
  } catch (e) {
    applyLSLStatus({
      available: false,
      active: false,
      include_markers: true,
      last_error: e.message,
    });
  }
  syncLSLUI();
}

function applyOSCStatus(status) {
  if (!status) return;
  S.osc = {
    available: !!status.available,
    active: !!status.active,
    host: status.host || '',
    port: Number(status.port || 9000),
    prefix: status.prefix || '/kyma',
    mirror_events: status.mirror_events !== false,
    last_error: status.last_error || '',
  };
}

function syncOSCUI() {
  const wrap = $('osc-status');
  const btn = $('btn-osc');
  const hostInput = $('osc-host');
  const portInput = $('osc-port');
  const prefixInput = $('osc-prefix');
  const eventChk = $('osc-events');
  if (!wrap || !btn || !hostInput || !portInput || !prefixInput || !eventChk) return;

  const storedHost = localStorage.getItem('kyma-osc-host') || '127.0.0.1';
  const storedPort = localStorage.getItem('kyma-osc-port') || '9000';
  const storedPrefix = localStorage.getItem('kyma-osc-prefix') || '/kyma';
  const storedEvents = localStorage.getItem('kyma-osc-events');

  hostInput.value = S.osc.active ? (S.osc.host || storedHost) : (hostInput.value || S.osc.host || storedHost);
  portInput.value = S.osc.active ? String(S.osc.port || storedPort) : (portInput.value || String(S.osc.port || storedPort));
  prefixInput.value = S.osc.active ? (S.osc.prefix || storedPrefix) : (prefixInput.value || S.osc.prefix || storedPrefix);
  if (S.osc.active) eventChk.checked = S.osc.mirror_events !== false;
  else if (storedEvents !== null) eventChk.checked = storedEvents === '1';
  else eventChk.checked = true;

  const meta = document.createElement('div');
  meta.className = 'setup-meta';
  const state = document.createElement('span');
  state.textContent = S.osc.active ? 'Active' : (S.osc.available ? 'Ready' : 'Unavailable');
  const target = document.createElement('span');
  target.textContent = S.osc.active ? `${S.osc.host}:${S.osc.port}` : 'OSC';
  meta.append(state, target);

  const copy = document.createElement('div');
  copy.className = 'setup-copy';
  if (!S.osc.available) {
    copy.textContent = S.osc.last_error || 'python-osc is not installed. Install requirements.txt to enable OSC.';
  } else if (S.osc.active) {
    copy.textContent = `Prefix: ${S.osc.prefix} | Event mirroring ${S.osc.mirror_events ? 'on' : 'off'}`;
  } else {
    copy.textContent = 'Mirror decoded labels, state changes, and control commands to an OSC target.';
  }

  wrap.innerHTML = '';
  wrap.append(meta, copy);
  btn.textContent = S.osc.active ? 'Stop OSC' : 'Start OSC';
  btn.className = S.osc.active ? 'btn danger' : 'btn';
  btn.disabled = !S.osc.available;
  hostInput.disabled = S.osc.active || !S.osc.available;
  portInput.disabled = S.osc.active || !S.osc.available;
  prefixInput.disabled = S.osc.active || !S.osc.available;
  eventChk.disabled = S.osc.active || !S.osc.available;
}

async function refreshOSCStatus() {
  try {
    const status = await get('/api/osc/status');
    applyOSCStatus(status);
  } catch (e) {
    applyOSCStatus({
      available: false,
      active: false,
      host: '',
      port: 9000,
      prefix: '/kyma',
      mirror_events: true,
      last_error: e.message,
    });
  }
  syncOSCUI();
}

function getWorkspaceDescription(name) {
  if (name === 'record') {
    return 'Session metadata, protocol runs, and subject-tagged recording workflows.';
  }
  if (name === 'train') {
    return 'Model fitting, dataset creation, and offline experiment evaluation.';
  }
  if (name === 'control') {
    return 'Actuation, output transports, and live control surfaces.';
  }
  return 'Live monitoring, quality review, and profile-specific visualization.';
}

function syncWorkspaceUI() {
  const current = S.dashboardWorkspace || 'live';
  document.querySelectorAll('.workspace-btn').forEach(btn => {
    btn.classList.toggle('active', btn.id === `workspace-${current}`);
  });
  const summary = $('workspace-summary');
  if (summary) summary.textContent = getWorkspaceDescription(current);

  document.querySelectorAll('[data-workspaces]').forEach(el => {
    const allowed = String(el.dataset.workspaces || '')
      .split(',')
      .map(s => s.trim())
      .filter(Boolean)
      .includes(current);
    let visible = allowed;
    if (visible && el.dataset.requiresArm === '1' && !S.supportsArmGestures) visible = false;
    if (visible && el.dataset.requiresTraining === '1' && !S.supportsTraining) visible = false;
    el.classList.toggle('workspace-hidden', !visible);
  });
}

window.switchWorkspace = function(name) {
  const wanted = ['live', 'record', 'train', 'control'].includes(name) ? name : 'live';
  S.dashboardWorkspace = wanted;
  localStorage.setItem('kyma-dashboard-workspace', wanted);
  syncWorkspaceUI();
};

function getTourFactory() {
  return window.driver?.js?.driver || null;
}

function setCardExpanded(target, expanded = true) {
  const card = typeof target === 'string' ? document.querySelector(target) : target;
  if (!card) return false;
  card.classList.toggle('collapsed', !expanded);
  return true;
}

function prepareTourView({
  tab = 'dashboard',
  workspace = 'live',
  expand = [],
  viz = null,
} = {}) {
  if (tab && typeof window.switchTab === 'function') window.switchTab(tab);
  if (workspace && typeof window.switchWorkspace === 'function') window.switchWorkspace(workspace);
  if (viz && typeof window.switchViz === 'function') window.switchViz(viz);
  expand.forEach(selector => setCardExpanded(selector, true));
}

function queueTourNext(tour, prep) {
  const advance = () => {
    window.requestAnimationFrame(() => {
      window.setTimeout(() => {
        tour.refresh();
        tour.moveNext();
      }, 180);
    });
  };
  try {
    const result = typeof prep === 'function' ? prep() : null;
    if (result && typeof result.then === 'function') {
      result.finally(advance);
      return;
    }
  } catch {}
  advance();
}

function onTourShow(prep) {
  return () => {
    if (typeof prep !== 'function') return;
    prep();
  };
}

function createQuickTour() {
  const driverFactory = getTourFactory();
  if (!driverFactory) return null;

  const tour = driverFactory({
    allowClose: true,
    animate: true,
    overlayClickBehavior: 'close',
    showProgress: true,
    showButtons: ['next', 'close'],
    smoothScroll: true,
    popoverClass: 'kyma-tour',
    nextBtnText: 'Next',
    doneBtnText: 'Done',
    onDestroyed: () => {
      $('btn-tour')?.classList.remove('active');
    },
  });

  const advance = prep => () => queueTourNext(tour, prep);
  const liveViz = S.signalProfileKey === 'eeg' ? 'eeg' : 'hand';

  tour.setSteps([
    {
      popover: {
        title: 'KYMA Quick Start',
        description: 'This walkthrough covers Live, Record, Train, and Control so a new user can go from signal setup to outputs without guessing through the UI.',
        nextBtnText: 'Start Tour',
      },
      onNextClick: advance(() => prepareTourView({
        tab: 'dashboard',
        workspace: 'live',
        viz: liveViz,
        expand: ['#card-signal-type', '#card-stream', '#card-decoded-output', '#card-channel-activity'],
      })),
    },
    {
      element: '#workspace-bar',
      onHighlightStarted: onTourShow(() => prepareTourView({
        tab: 'dashboard',
        workspace: 'live',
        viz: liveViz,
      })),
      popover: {
        title: 'Workspaces',
        description: 'The dashboard is split by job: Live for monitoring, Record for structured capture, Train for offline models, and Control for outputs and actuation.',
      },
      onNextClick: advance(() => prepareTourView({
        workspace: 'live',
        expand: ['#card-signal-type'],
      })),
    },
    {
      element: '#card-signal-type',
      onHighlightStarted: onTourShow(() => prepareTourView({
        workspace: 'live',
        expand: ['#card-signal-type'],
      })),
      popover: {
        title: 'Signal Type',
        description: 'Choose the biosignal profile first. This changes labels, filters, training support, and whether arm-specific controls are relevant.',
      },
      onNextClick: advance(() => prepareTourView({
        workspace: 'live',
        expand: ['#card-stream'],
      })),
    },
    {
      element: '#card-stream',
      onHighlightStarted: onTourShow(() => prepareTourView({
        workspace: 'live',
        expand: ['#card-stream'],
      })),
      popover: {
        title: 'Stream Setup',
        description: 'Pick the data source here. Synthetic is the fastest test path, Playback reuses saved sessions, LSL connects to external streams, and hardware uses the live device stack.',
      },
      onNextClick: advance(() => prepareTourView({
        workspace: 'live',
        expand: ['#card-decoded-output', '#card-channel-activity'],
      })),
    },
    {
      element: '#channel-legend',
      onHighlightStarted: onTourShow(() => prepareTourView({
        workspace: 'live',
      })),
      popover: {
        title: 'Channel Checkboxes',
        description: 'Each checkbox hides a channel from the live display only. Recording, training, and decoding still use the full stream.',
      },
      onNextClick: advance(() => prepareTourView({
        workspace: 'live',
        expand: ['#card-review-markers'],
      })),
    },
    {
      element: '#card-review-markers',
      onHighlightStarted: onTourShow(() => prepareTourView({
        workspace: 'live',
        expand: ['#card-review-markers'],
      })),
      popover: {
        title: 'Review And Markers',
        description: 'Pause the live trace here, drag across a frozen region to inspect it, then save custom point or range markers into the session and marker stream path.',
      },
      onNextClick: advance(() => prepareTourView({
        workspace: 'live',
        expand: ['#card-decoded-output'],
      })),
    },
    {
      element: '#card-decoded-output',
      onHighlightStarted: onTourShow(() => prepareTourView({
        workspace: 'live',
        expand: ['#card-decoded-output'],
      })),
      popover: {
        title: 'Decoded Output',
        description: 'This is the live state estimate. Confidence and summary update as windows arrive from the active profile.',
      },
      onNextClick: advance(() => prepareTourView({
        workspace: 'record',
        expand: ['#card-session-metadata', '#card-protocol-template'],
      })),
    },
    {
      element: '#session-label',
      onHighlightStarted: onTourShow(() => prepareTourView({
        workspace: 'record',
        expand: ['#card-session-metadata'],
      })),
      popover: {
        title: 'Record Sessions',
        description: 'Start here when capturing a new session. Label, subject, condition, and notes become part of the saved research metadata.',
      },
      onNextClick: advance(() => prepareTourView({
        workspace: 'record',
        expand: ['#card-protocol-template'],
      })),
    },
    {
      element: '#protocol-template',
      onHighlightStarted: onTourShow(() => prepareTourView({
        workspace: 'record',
        expand: ['#card-protocol-template'],
      })),
      popover: {
        title: 'Protocol Runs',
        description: 'Use protocol templates for structured trial sequences. KYMA can advance labels and run ids so the session set stays consistent.',
      },
      onNextClick: advance(() => prepareTourView({
        workspace: 'record',
        expand: ['#card-eeg-experiments'],
      })),
    },
    {
      element: '#card-eeg-experiments',
      onHighlightStarted: onTourShow(() => prepareTourView({
        workspace: 'record',
        expand: ['#card-eeg-experiments'],
      })),
      popover: {
        title: 'EEG Research Presets',
        description: 'When EEG is active, this card loads recording presets for research paradigms such as oddball, SSVEP, and N170. They prepare session metadata and export workflow notes, but they do not replace KYMA’s live band-state decoder.',
      },
      onNextClick: advance(() => prepareTourView({
        workspace: 'record',
        expand: ['#card-eeg-markers'],
      })),
    },
    {
      element: '#card-eeg-markers',
      onHighlightStarted: onTourShow(() => prepareTourView({
        workspace: 'record',
        expand: ['#card-eeg-markers'],
      })),
      popover: {
        title: 'Stimulus and Markers',
        description: 'This helper shows the expected marker names and suggested block structure for the selected EEG preset. Use it as the reference for your external stimulus presenter and LSL marker stream.',
      },
      onNextClick: advance(() => prepareTourView({
        workspace: 'record',
        expand: ['#card-lsl-marker-tester'],
      })),
    },
    {
      element: '#card-lsl-marker-tester',
      onHighlightStarted: onTourShow(() => prepareTourView({
        workspace: 'record',
        expand: ['#card-lsl-marker-tester'],
      })),
      popover: {
        title: 'LSL Marker Tester',
        description: 'After you start LSL output in Control, use this card to send test markers and verify your recorder or stimulus stack sees the exact event names you expect.',
      },
      onNextClick: advance(() => prepareTourView({
        workspace: 'train',
        expand: ['#card-training', '#card-research'],
      })),
    },
    {
      element: '#clf-select',
      onHighlightStarted: onTourShow(() => prepareTourView({
        workspace: 'train',
        expand: ['#card-training'],
      })),
      popover: {
        title: 'Train Models',
        description: 'Choose a classifier here after collecting labeled data. This is the supervised training path, separate from the always-on live analyzers.',
      },
      onNextClick: advance(() => prepareTourView({
        workspace: 'train',
        expand: ['#card-research'],
      })),
    },
    {
      element: '#dataset-name',
      onHighlightStarted: onTourShow(() => prepareTourView({
        workspace: 'train',
        expand: ['#card-research'],
      })),
      popover: {
        title: 'Datasets and Experiments',
        description: 'Create a dataset from selected sessions here, then run the offline experiment controls below it for holdout or LOSO validation.',
      },
      onNextClick: advance(() => prepareTourView({
        workspace: 'control',
        expand: ['#card-lsl', '#card-osc'],
      })),
    },
    {
      element: '#card-lsl',
      onHighlightStarted: onTourShow(() => prepareTourView({
        workspace: 'control',
        expand: ['#card-lsl'],
      })),
      popover: {
        title: 'LSL Output',
        description: 'Publish live biosignal samples and markers over Lab Streaming Layer so external research tools can subscribe in real time.',
      },
      onNextClick: advance(() => prepareTourView({
        workspace: 'control',
        expand: ['#card-osc'],
      })),
    },
    {
      element: '#card-osc',
      onHighlightStarted: onTourShow(() => prepareTourView({
        workspace: 'control',
        expand: ['#card-osc'],
      })),
      popover: {
        title: 'OSC Output',
        description: 'Send decoded labels, state changes, and control commands to OSC targets like TouchDesigner, Max, audio software, or other control systems.',
      },
      onNextClick: advance(() => prepareTourView({
        tab: 'dashboard',
        workspace: 'control',
      })),
    },
    {
      element: '#btn-estop',
      onHighlightStarted: onTourShow(() => prepareTourView({
        tab: 'dashboard',
        workspace: 'control',
      })),
      popover: {
        title: 'Safety',
        description: 'E-STOP is always visible. Use it any time you need to halt actuation immediately.',
      },
      onNextClick: advance(() => prepareTourView({
        tab: 'dashboard',
        workspace: 'live',
      })),
    },
    {
      element: '#tab-blocks',
      onHighlightStarted: onTourShow(() => prepareTourView({
        tab: 'dashboard',
        workspace: 'live',
      })),
      popover: {
        title: 'Other Features',
        description: 'Blocks gives you visual automation, Filter Lab handles reusable DSP design, Signal Workshop analyzes frozen chunks in depth, and Arduino Guide documents wiring, firmware, and protocol details. Each tab has its own guided tour.',
      },
    },
  ]);

  return tour;
}

window.startQuickTour = function() {
  const driverFactory = getTourFactory();
  if (!driverFactory) {
    toast('Tour library is unavailable', 'red');
    return;
  }
  if (document.body.classList.contains('driver-active')) {
    return;
  }
  localStorage.setItem(TOUR_STORAGE_KEY, '1');
  $('btn-tour')?.classList.add('active');
  prepareTourView({
    tab: 'dashboard',
    workspace: 'live',
    viz: S.signalProfileKey === 'eeg' ? 'eeg' : 'hand',
    expand: ['#card-signal-type', '#card-stream', '#card-decoded-output', '#card-channel-activity'],
  });
  const tour = createQuickTour();
  if (!tour) {
    $('btn-tour')?.classList.remove('active');
    toast('Tour failed to initialize', 'red');
    return;
  }
  tour.drive();
};

function createBlocksTour() {
  const driverFactory = getTourFactory();
  if (!driverFactory) return null;

  const tour = driverFactory({
    allowClose: true,
    animate: true,
    overlayClickBehavior: 'close',
    showProgress: true,
    showButtons: ['next', 'close'],
    smoothScroll: true,
    popoverClass: 'kyma-tour',
    nextBtnText: 'Next',
    doneBtnText: 'Done',
    onDestroyed: () => {
      $('btn-tour')?.classList.remove('active');
      $('btn-blocks-tour')?.classList.remove('active');
      window.closeCodeModal?.();
    },
  });

  const advance = prep => () => queueTourNext(tour, prep);

  tour.setSteps([
    {
      popover: {
        title: 'Blocks Tour',
        description: 'This walkthrough shows how to load a simple signal-driven example, inspect the graph, and generate Arduino code from it.',
        nextBtnText: 'Start Tour',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'blocks' })),
    },
    {
      element: '#workspace-toolbar',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'blocks' })),
      popover: {
        title: 'Blocks Toolbar',
        description: 'Create, rename, delete, export, and run programs here. The new example and tutorial actions also live in this toolbar.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'blocks' })),
    },
    {
      element: '#btn-block-example',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'blocks' })),
      popover: {
        title: 'Load Signal Example',
        description: 'This creates a simple exportable starter: if channel 1 metric crosses a threshold, set pin 13 HIGH, otherwise LOW.',
      },
      onNextClick: advance(() => {
        prepareTourView({ tab: 'blocks' });
        window.loadSignalExampleProgram?.();
      }),
    },
    {
      element: '#program-select',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'blocks' })),
      popover: {
        title: 'Active Program',
        description: 'The example program becomes the active graph here. You can keep it as a template or duplicate it into your own workflows.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'blocks' })),
    },
    {
      element: '.palette-block[data-type="if_rms"]',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'blocks' })),
      popover: {
        title: 'Exportable Signal Logic',
        description: 'Use If Metric > when you want something that can turn into standalone Arduino code. It maps cleanly to an analog threshold in the generated sketch.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'blocks' })),
    },
    {
      element: '#node-canvas-container',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'blocks' })),
      popover: {
        title: 'Graph Workspace',
        description: 'This is the node graph. The starter example should now be visible: Start -> If Metric > -> Digital Write HIGH/LOW.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'blocks' })),
    },
    {
      element: '#btn-block-export',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'blocks' })),
      popover: {
        title: 'Generate Code',
        description: 'Export .ino turns the current graph into Arduino C++. The signal-threshold example is designed to produce a usable sketch instead of placeholders only.',
      },
      onNextClick: advance(() => {
        prepareTourView({ tab: 'blocks' });
        window.exportArduinoCode?.();
      }),
    },
    {
      element: '#code-modal-inner',
      onHighlightStarted: onTourShow(() => {
        prepareTourView({ tab: 'blocks' });
        if (!$('code-modal')?.classList.contains('active')) window.exportArduinoCode?.();
      }),
      popover: {
        title: 'Generated Arduino Code',
        description: 'Review the generated sketch here, then copy or download it. This is the full code path from a simple signal block graph to a `.ino` file.',
      },
      onNextClick: advance(() => {
        window.closeCodeModal?.();
        prepareTourView({ tab: 'blocks' });
      }),
    },
    {
      element: '.palette-block[data-type="saved_filter"]',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'blocks' })),
      popover: {
        title: 'Saved Filter Block',
        description: 'Use this when you want a block program to switch the active KYMA host filter during live execution. The `.ino` export keeps it as a documented placeholder because the filter itself lives in the KYMA/filter-export path, not on the Cyton.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'blocks' })),
    },
    {
      element: '#card-block-mapping',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'blocks' })),
      popover: {
        title: 'Decoded Label Mapping',
        description: 'This is the live KYMA path. Map decoded labels to programs when you want the server to trigger scripts from predictions instead of exporting standalone firmware.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'blocks' })),
    },
    {
      element: '#card-block-help',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'blocks' })),
      popover: {
        title: 'When To Use Which Blocks',
        description: 'Metric-threshold blocks are best for exportable firmware. Label-driven blocks are best when KYMA stays in the loop and executes programs from live decoded outputs.',
      },
    },
  ]);

  return tour;
}

window.startBlocksTour = function() {
  const driverFactory = getTourFactory();
  if (!driverFactory) {
    toast('Tour library is unavailable', 'red');
    return;
  }
  if (document.body.classList.contains('driver-active')) {
    return;
  }
  localStorage.setItem(TOUR_STORAGE_KEY, '1');
  $('btn-tour')?.classList.add('active');
  $('btn-blocks-tour')?.classList.add('active');
  prepareTourView({ tab: 'blocks' });
  const tour = createBlocksTour();
  if (!tour) {
    $('btn-tour')?.classList.remove('active');
    $('btn-blocks-tour')?.classList.remove('active');
    toast('Blocks tour failed to initialize', 'red');
    return;
  }
  tour.drive();
};

function createFilterTour() {
  const driverFactory = getTourFactory();
  if (!driverFactory) return null;

  const tour = driverFactory({
    allowClose: true,
    animate: true,
    overlayClickBehavior: 'close',
    showProgress: true,
    showButtons: ['next', 'close'],
    smoothScroll: true,
    popoverClass: 'kyma-tour',
    nextBtnText: 'Next',
    doneBtnText: 'Done',
    onDestroyed: () => {
      $('btn-tour')?.classList.remove('active');
      $('btn-filter-tour')?.classList.remove('active');
    },
  });

  const advance = prep => () => queueTourNext(tour, prep);

  tour.setSteps([
    {
      popover: {
        title: 'Filter Lab Tour',
        description: 'This walkthrough shows how to design a digital filter, preview its response, save it, activate it in KYMA, and export reusable code.',
        nextBtnText: 'Start Tour',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'filters' })),
    },
    {
      element: '#card-filter-design',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'filters' })),
      popover: {
        title: 'Design Inputs',
        description: 'Pick the response type, method, order, and cutoff frequencies here. Designs are tied to the active signal profile and 250 Hz runtime sample rate.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'filters' })),
    },
    {
      element: '#btn-filter-preview',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'filters' })),
      popover: {
        title: 'Preview The Response',
        description: 'Preview uses the backend design engine to compute second-order sections and the expected frequency response before you save anything.',
      },
      onNextClick: advance(async () => {
        prepareTourView({ tab: 'filters' });
        await previewFilterDesign();
      }),
    },
    {
      element: '#filter-response-canvas',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'filters' })),
      popover: {
        title: 'Frequency Response',
        description: 'This graph is the magnitude response of the current preview or saved filter. Use it to verify your passband, stopband, and overall shaping before activation.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'filters' })),
    },
    {
      element: '#filter-polezero-canvas',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'filters' })),
      popover: {
        title: 'Pole / Zero Check',
        description: 'This panel shows the discrete-time pole and zero placement. An EE will care about this because stability and filter shape are immediately visible against the unit circle.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'filters' })),
    },
    {
      element: '#card-filter-quant',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'filters' })),
      popover: {
        title: 'Fixed-Point Estimate',
        description: 'These estimates show whether direct Q1.15 or Q1.31 would overflow, and what signed 16-bit or 32-bit format is safer if you need to move coefficients into embedded code.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'filters' })),
    },
    {
      element: '#card-filter-saved',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'filters' })),
      popover: {
        title: 'Saved Filters',
        description: 'Save validated designs here so they can be reused later. Activation is per biosignal profile, so you can keep different custom chains for EMG, EEG, ECG, and others.',
      },
      onNextClick: advance(async () => {
        prepareTourView({ tab: 'filters' });
        if (!S.filterLab.filters.length && S.filterLab.preview) {
          await saveFilterDesign();
        }
      }),
    },
    {
      element: '#card-filter-active',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'filters' })),
      popover: {
        title: 'Activate In KYMA',
        description: 'Use a saved filter here to insert it into the live KYMA host processing path. Append adds it after the profile defaults; replace bypasses the default BrainFlow stages for the hardware path.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'filters' })),
    },
    {
      element: '#card-filter-export',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'filters' })),
      popover: {
        title: 'Export Code',
        description: 'Export targets now include host code, C++ reuse, Bode CSV, pole-zero JSON, and a fixed-point header. That keeps the design grounded in existing DSP workflows instead of rewriting the math yourself.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'bench' })),
    },
    {
      element: '#card-bench-preview',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'bench' })),
      popover: {
        title: 'Bench Report',
        description: 'This report page compresses the live timing, noise, safety, and filter analysis into one engineering summary you can export or hand to a reviewer.',
      },
    },
  ]);

  return tour;
}

window.startFilterTour = function() {
  const driverFactory = getTourFactory();
  if (!driverFactory) {
    toast('Tour library is unavailable', 'red');
    return;
  }
  if (document.body.classList.contains('driver-active')) {
    return;
  }
  localStorage.setItem(TOUR_STORAGE_KEY, '1');
  $('btn-tour')?.classList.add('active');
  $('btn-filter-tour')?.classList.add('active');
  prepareTourView({ tab: 'filters' });
  const tour = createFilterTour();
  if (!tour) {
    $('btn-tour')?.classList.remove('active');
    $('btn-filter-tour')?.classList.remove('active');
    toast('Filter tour failed to initialize', 'red');
    return;
  }
  tour.drive();
};

function createWorkshopTour() {
  const driverFactory = getTourFactory();
  if (!driverFactory) return null;

  const tour = driverFactory({
    allowClose: true,
    animate: true,
    overlayClickBehavior: 'close',
    showProgress: true,
    showButtons: ['next', 'close'],
    smoothScroll: true,
    popoverClass: 'kyma-tour',
    onDestroyed: () => {
      $('btn-tour')?.classList.remove('active');
      $('btn-workshop-tour')?.classList.remove('active');
    },
  });

  const advance = prep => () => queueTourNext(tour, prep);
  const onTourShow = cb => () => window.requestAnimationFrame(() => cb());

  tour.setSteps([
    {
      element: '#card-workshop-selection',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'workshop' })),
      popover: {
        title: 'Workshop Entry',
        description: 'This tab analyzes frozen chunks from Review. Freeze the live display first, drag a chunk, then send it here or pull the full frozen window.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'workshop' })),
    },
    {
      element: '#btn-workshop-from-review',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'workshop' })),
      popover: {
        title: 'Pull From Review',
        description: 'Use this to import the current frozen review window or selected chunk directly into the workshop.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'workshop' })),
    },
    {
      element: '#btn-workshop-analyze',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'workshop' })),
      popover: {
        title: 'Analyze Selection',
        description: 'Run the selected chunk through FFT, PSD, spectrogram, autocorrelation, histogram, envelope, correlation, and Laplace analysis on the server.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'workshop' })),
    },
    {
      element: '#workshop-main-canvas',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'workshop' })),
      popover: {
        title: 'Transform View',
        description: 'This canvas switches between the active analysis views. Spectrogram and Laplace render as heatmaps; correlation renders as a channel matrix.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'workshop' })),
    },
    {
      element: '#card-workshop-details',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'workshop' })),
      popover: {
        title: 'Numeric Readout',
        description: 'Use the summary, band metrics, and detail list as the quick engineering readout for the selected chunk.',
      },
    },
  ]);

  return tour;
}

window.startWorkshopTour = function() {
  const driverFactory = getTourFactory();
  if (!driverFactory) {
    toast('Tour library is unavailable', 'red');
    return;
  }
  if (document.body.classList.contains('driver-active')) {
    return;
  }
  localStorage.setItem(TOUR_STORAGE_KEY, '1');
  $('btn-tour')?.classList.add('active');
  $('btn-workshop-tour')?.classList.add('active');
  prepareTourView({ tab: 'workshop' });
  const tour = createWorkshopTour();
  if (!tour) {
    $('btn-tour')?.classList.remove('active');
    $('btn-workshop-tour')?.classList.remove('active');
    toast('Workshop tour failed to initialize', 'red');
    return;
  }
  tour.drive();
};

function syncProfileUI() {
  const profileBadge = $('profile-badge');
  const header = $('signal-header-label');
  const metricTitle = $('metric-card-title');
  const handTab = $('viz-tab-hand');
  const armTab = $('viz-tab-arm');
  const eegTab = $('viz-tab-eeg');
  const profileSel = $('profile-select');
  const profileSummary = $('profile-summary');
  const trainStatus = $('train-status');
  const fitBtn = $('btn-fit');
  const clearBtn = $('btn-clear-train');
  const clfSel = $('clf-select');
  const propChk = $('chk-proportional');

  if (profileBadge) profileBadge.textContent = S.signalProfileName;
  if (header) header.textContent = `LIVE ${S.signalProfileName.toUpperCase()} -- ${N_CH} channels`;
  if (metricTitle) metricTitle.textContent = `${S.signalMetricLabel} (${S.signalUnits})`;

  if (profileSel) {
    profileSel.disabled = S.streaming;
    profileSel.innerHTML = '';
    const profiles = S.availableProfiles.length ? S.availableProfiles : [{ key: S.signalProfileKey, display_name: S.signalProfileName }];
    profiles.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.key;
      opt.textContent = p.display_name;
      opt.title = p.support_level || 'profile';
      opt.selected = p.key === S.signalProfileKey;
      profileSel.appendChild(opt);
    });
  }

  if (profileSummary) {
    const active = S.availableProfiles.find(p => p.key === S.signalProfileKey);
    const support = (S.signalSupportLevel || active?.support_level || 'profile').replace(/_/g, ' ');
    const note = S.signalDescription || active?.support_notes || `${S.signalProfileName} profile ready.`;
    profileSummary.innerHTML = '';

    const meta = document.createElement('div');
    meta.className = 'setup-meta';
    const units = document.createElement('span');
    units.textContent = S.signalUnits;
    const level = document.createElement('span');
    level.textContent = support;
    meta.append(units, level);

    const copy = document.createElement('div');
    copy.className = 'setup-copy';
    copy.textContent = note;

    profileSummary.append(meta, copy);
  }

  if (fitBtn) fitBtn.disabled = !S.supportsTraining;
  if (clearBtn) clearBtn.disabled = !S.supportsTraining;
  if (clfSel) clfSel.disabled = !S.supportsTraining;
  if (propChk) {
    if (!S.supportsArmGestures) {
      propChk.checked = false;
      S.proportional = false;
    }
    propChk.disabled = !S.supportsArmGestures;
  }

  if (trainStatus) {
    if (!S.supportsTraining) {
      trainStatus.textContent = `${S.signalProfileName} currently uses the live analyzer path in this build.`;
    } else if (trainStatus.textContent.includes('currently uses the live analyzer path')) {
      trainStatus.textContent = '';
    }
  }

  if (handTab) handTab.style.display = S.signalProfileKey === 'eeg' ? 'none' : '';
  if (armTab) armTab.style.display = S.supportsArmGestures ? '' : 'none';
  if (eegTab) {
    eegTab.style.display = S.signalProfileKey === 'eeg' ? '' : 'none';
  }
  if (S.signalProfileKey === 'eeg' && S.activeViz !== 'eeg') {
    switchViz('eeg');
  } else if (S.signalProfileKey !== 'eeg' && S.activeViz === 'eeg') {
    switchViz('hand');
  }
  if (S.signalProfileKey === 'eeg') {
    refreshEEGBrainView(true);
  }

  syncStreamModeUI();
  syncCalibrationUI();
  syncEEGExperimentUI();
  syncEEGMarkerHelperUI();
  syncLSLMarkerTesterUI();
  syncLSLUI();
  syncOSCUI();
  syncWorkspaceUI();
}

function syncStreamModeUI() {
  const sourceSel = $('stream-source');
  const cytonSel = $('cyton-port');
  const scanBtn = $('btn-refresh-ports');
  const streamBtn = $('btn-stream');
  const hint = $('stream-mode-hint');
  const sourceBadge = $('source-badge');
  const profileSel = $('profile-select');
  const hardwareSetup = $('hardware-setup');
  const lslInputSetup = $('lsl-input-setup');
  const playbackSetup = $('playback-setup');
  const playbackSel = $('playback-session');
  const playbackRateSel = $('playback-rate');
  const lslInputSel = $('lsl-input-stream');
  const recordBtn = $('btn-record-session');
  const calibrateBtn = $('btn-calibrate');
  if (!sourceSel || !cytonSel || !scanBtn || !hint) return;

  if (S.streaming && sourceSel.value !== S.streamSource) {
    sourceSel.value = S.streamSource;
  }
  if (S.streaming && playbackSel && S.playbackSessionId) {
    playbackSel.value = S.playbackSessionId;
  }

  const selectedSource = getSelectedSource();
  const isSynthetic = selectedSource === 'synthetic';
  const isPlayback = selectedSource === 'playback';
  const isLSL = selectedSource === 'lsl';
  const activeProfile = S.availableProfiles.find(p => p.key === S.signalProfileKey);
  const requiresSynthetic = !!(activeProfile && !activeProfile.hardware_supported);
  const hasPlaybackSession = !!(playbackSel && playbackSel.value);
  const hasLSLStream = !!getSelectedLSLInput();

  sourceSel.disabled = S.streaming;
  cytonSel.disabled = isSynthetic || isPlayback || isLSL || requiresSynthetic || S.streaming;
  scanBtn.disabled = cytonSel.disabled;
  if (profileSel) profileSel.disabled = S.streaming;
  if (hardwareSetup) hardwareSetup.style.opacity = (!isSynthetic && !isPlayback && !isLSL && !requiresSynthetic) ? '1' : '0.55';
  if (lslInputSetup) lslInputSetup.style.display = isLSL ? 'block' : 'none';
  if (playbackSetup) playbackSetup.style.display = isPlayback ? 'block' : 'none';
  if (playbackSel) playbackSel.disabled = S.streaming || !isPlayback;
  if (playbackRateSel) playbackRateSel.disabled = S.streaming || !isPlayback;
  if (lslInputSel) lslInputSel.disabled = S.streaming || !isLSL;

  if (streamBtn) {
    streamBtn.disabled = S.streaming
      ? false
      : (
          (selectedSource === 'hardware' && requiresSynthetic)
          || (isPlayback && !hasPlaybackSession)
          || (isLSL && !hasLSLStream)
        );
  }

  if (recordBtn) {
    recordBtn.disabled = !S.streaming || S.streamSource === 'playback';
    recordBtn.textContent = S.recSession ? 'Stop Rec' : 'Rec Session';
    recordBtn.className = S.recSession ? 'btn danger' : 'btn';
  }
  if (calibrateBtn) calibrateBtn.disabled = !S.streaming || S.streamSource === 'playback';

  if (sourceBadge) {
    const shownSource = S.streaming ? S.streamSource : selectedSource;
    sourceBadge.textContent = shownSource;
    sourceBadge.className = shownSource;
  }

  if (S.streaming) {
    if (S.streamSource === 'playback') {
      const sid = S.streamDetails?.session_id || S.playbackSessionId || 'session';
      const rate = Number(S.streamDetails?.playback_rate || 1).toFixed(2);
      hint.textContent = `Playback source: ${sid} at ${rate}x. Stop stream to change.`;
    } else if (S.streamSource === 'lsl') {
      const name = S.streamDetails?.name || S.streamDetails?.stream_name || 'LSL stream';
      hint.textContent = `LSL source: ${name}. Stop stream to change.`;
    } else {
      hint.textContent = `Source: ${S.streamSource}. Stop stream to change.`;
    }
  } else if (isSynthetic) {
    hint.textContent = `Synthetic ${S.signalProfileName} stream. No hardware required.`;
  } else if (isLSL && !hasLSLStream) {
    hint.textContent = 'Select an external LSL signal stream first.';
  } else if (isLSL) {
    const active = S.lslInputs.find(stream => (stream.source_id || stream.uid || stream.name) === getSelectedLSLInput());
    const rate = active?.sample_rate ? `${Number(active.sample_rate).toFixed(0)} Hz` : 'irregular rate';
    hint.textContent = `${active?.name || 'LSL stream'} ready (${rate}).`;
  } else if (isPlayback && !hasPlaybackSession) {
    hint.textContent = 'Select a recorded session for playback.';
  } else if (isPlayback) {
    const sid = playbackSel?.value || 'session';
    const rate = Number(playbackRateSel?.value || 1).toFixed(2);
    hint.textContent = `Playback ${sid} at ${rate}x. Profile will follow the recorded session.`;
  } else if (requiresSynthetic) {
    hint.textContent = `${S.signalProfileName} has no live hardware path in this build. Use synthetic or playback.`;
  } else {
    hint.textContent = `Hardware ${S.signalProfileName} stream ready.`;
  }
  syncProtocolUI();
}


// =============================================================================
// WEBSOCKET
//
// auto-reconnects every 3 seconds if it drops.
// the server sends: emg (waveform data), prediction, state, calibration, ping
// =============================================================================

function connectWS() {
  S.ws = new WebSocket(WS_URL);
  S.ws.onopen = () => { $('ws-dot').className = 'ok'; toast('Connected'); };
  S.ws.onclose = () => { $('ws-dot').className = ''; setTimeout(connectWS, 3000); };
  S.ws.onerror = () => S.ws.close();
  S.ws.onmessage = e => {
    const m = JSON.parse(e.data);
    switch (m.type) {
      case 'emg':         onEmg(m.data); break;
      case 'prediction':  onPrediction(m.data); break;
      case 'diagnostics': applyDiagnostics(m.data); break;
      case 'safety':      applySafety(m.data); break;
      case 'review_marker':
        pushReviewMarker({
          ...(m.data || {}),
          createdAt: Date.now(),
          sampleIndex: Number.isFinite(m.data?.selection?.end_sample) ? Number(m.data.selection.end_sample) : Math.max(0, S.emgTotal - 1),
        });
        break;
      case 'state':       setSysState(m.data.state); break;
      case 'calibration': onCalibration(m.data); break;
      case 'ping':
        // measure round-trip latency from server ping timestamp
        if (m.timestamp) S.wsLatency = Math.round(Date.now() - m.timestamp * 1000);
        S.ws.send('{"type":"pong"}');
        break;
    }
  };
}


// =============================================================================
// MESSAGE HANDLERS
// =============================================================================

/**
 * EMG data comes in as the new increment only (not the full window).
 * we just append each sample to the ring buffer and it scrolls naturally.
 */
function onEmg(d) {
  const ch = d.channels;
  const n = ch[0]?.length ?? 0;
  S.lastSignalAtClient = performance.now();
  for (let c = 0; c < Math.min(ch.length, N_CH); c++) {
    for (let i = 0; i < n; i++) {
      S.emg[c][(S.emgHead + i) % DISPLAY_SAMPLES] = ch[c][i];
    }
  }
  S.emgHead = (S.emgHead + n) % DISPLAY_SAMPLES;
  S.emgTotal += n;
    if (d.rms) {
      S.rms = d.rms;
      updateRmsBars(d.rms);
      updateFatigue(d.rms);
      updateQualityGrid(d.quality || d.rms);
      if (S.proportional) applyProportional(d.rms);
    }
    if (S.signalProfileKey === 'eeg' && S.activeViz === 'eeg' && !S.review.paused) {
      refreshEEGBrainView(false);
    }
    S.frames++;
  }

// =============================================================================
// PROPORTIONAL CONTROL
//
// Maps raw EMG RMS directly to joint angles — no classifier, no training.
// Each mapping: channel RMS → joint angle (flex or extend direction).
// Competing channels on the same joint cancel out (agonist/antagonist).
// =============================================================================

function applyProportional(rms) {
  if (!window._arm3d) return;

  const gain = S.propGain;
  const DEAD_ZONE = S.propDeadZone;    // configurable dead zone
  const BASELINE_RATE = 0.02; // slow adaptation rate for rest baseline
  const SMOOTH_RATE = 0.15;   // output smoothing (0=frozen, 1=instant)

  // Adapt baseline: if all mapped channels are quiet, slowly update rest baseline
  const allQuiet = S.propMap.every(m => {
    const above = (rms[m.ch] || 0) - (S.propRestRms[m.ch] || 0);
    return above < DEAD_ZONE * 3; // "quiet" = less than 3x dead zone above baseline
  });
  if (allQuiet) {
    for (const m of S.propMap) {
      // slowly drift baseline toward current RMS
      S.propRestRms[m.ch] += ((rms[m.ch] || 0) - S.propRestRms[m.ch]) * BASELINE_RATE;
    }
  }

  // Accumulate per-joint contributions
  const joints = {};

  for (const m of S.propMap) {
    const raw = rms[m.ch] || 0;
    // subtract rest baseline (noise floor)
    const active = Math.max(0, raw - (S.propRestRms[m.ch] || 0));
    // dead zone: ignore small noise fluctuations
    const gated = active < DEAD_ZONE ? 0 : (active - DEAD_ZONE);
    // scale: 200µV full-scale at gain=1 (BrainFlow RMS is already in µV)
    const norm = Math.min(1, (gated * gain) / 200);

    if (!joints[m.joint]) joints[m.joint] = 0;
    joints[m.joint] += norm * m.dir;  // +1 flex, -1 extend
  }

  // Initialize smooth state if needed
  if (!S._propSmooth) S._propSmooth = {};

  // Convert to servo angle: 0 = -90°, 90 = neutral, 180 = +90°
  for (const [joint, val] of Object.entries(joints)) {
    // clamp to [-1, 1]
    const clamped = Math.max(-1, Math.min(1, val));
    // map: 0 contribution → 90 (neutral), full flex → 180, full extend → 0
    const rawAngle = 90 + clamped * 90;

    // smooth the output to prevent jitter
    const prev = S._propSmooth[joint] ?? 90;
    const angle = prev + (rawAngle - prev) * SMOOTH_RATE;
    S._propSmooth[joint] = angle;

    window._arm3d.setJoint(parseInt(joint), angle);

    // also update the manual slider to reflect
    const sv = document.getElementById(`sv-${joint}`);
    const lbl = document.getElementById(`sv-v-${joint}`);
    if (sv) { sv.value = Math.round(angle); }
    if (lbl) { lbl.textContent = Math.round(angle); }
  }
}

const JOINT_NAMES = ['Shoulder Rot','Shoulder Pitch','Elbow','Wrist Pitch','Wrist Roll',
                     'Thumb','Index','Middle','Ring','Pinky','Forearm Rot','Grip'];

function initProportionalUI() {
  const chk = $('chk-proportional');
  const gainSlider = $('prop-gain');
  const gainVal = $('prop-gain-val');
  const dzSlider = $('prop-deadzone');
  const dzVal = $('prop-dz-val');
  const mappingsDiv = $('prop-mappings');
  const addBtn = $('prop-add-btn');

  // load saved mappings from localStorage
  try {
    const saved = localStorage.getItem('propMap');
    if (saved) S.propMap = JSON.parse(saved);
    const savedGain = localStorage.getItem('propGain');
    if (savedGain) { S.propGain = parseInt(savedGain); gainSlider.value = S.propGain; gainVal.textContent = `${S.propGain}x`; }
    const savedDz = localStorage.getItem('propDeadZone');
    if (savedDz) { S.propDeadZone = parseFloat(savedDz); dzSlider.value = Math.round(S.propDeadZone); dzVal.textContent = `${Math.round(S.propDeadZone)}µV`; }
  } catch(e) {}

  function saveMappings() {
    localStorage.setItem('propMap', JSON.stringify(S.propMap));
  }

  function renderMappings() {
    if (S.propMap.length === 0) {
      mappingsDiv.innerHTML = '<div style="color:var(--text-dim);padding:4px 0">No mappings — add one below</div>';
      return;
    }
    mappingsDiv.innerHTML = S.propMap.map((m, i) => {
      const chName = `CH${m.ch + 1}`;
      const jName = JOINT_NAMES[m.joint] || `J${m.joint}`;
      const dirLabel = m.dir > 0 ? '+' : '−';
      const color = m.dir > 0 ? 'var(--green)' : 'var(--red)';
      return `<div style="display:flex;align-items:center;gap:4px;padding:2px 0;border-bottom:1px solid var(--border)">
        <span style="color:var(--cyan);font-weight:bold;min-width:28px">${chName}</span>
        <span style="color:var(--text-dim)">→</span>
        <span style="flex:1;color:var(--text)">${jName}</span>
        <span style="color:${color};font-weight:bold;min-width:16px;text-align:center">${dirLabel}</span>
        <button onclick="window._removePropMap(${i})" style="font-size:9px;padding:1px 5px;background:var(--red);color:#fff;border:none;border-radius:2px;cursor:pointer;opacity:0.7" title="Remove">✕</button>
      </div>`;
    }).join('');
  }
  renderMappings();

  window._removePropMap = (idx) => {
    S.propMap.splice(idx, 1);
    saveMappings();
    renderMappings();
  };

  addBtn.addEventListener('click', () => {
    const ch = parseInt($('prop-add-ch').value);
    const joint = parseInt($('prop-add-joint').value);
    const dir = parseInt($('prop-add-dir').value);
    // check for duplicate
    const dup = S.propMap.find(m => m.ch === ch && m.joint === joint && m.dir === dir);
    if (dup) { toast('Mapping already exists', 'yellow'); return; }
    S.propMap.push({ ch, joint, dir });
    saveMappings();
    renderMappings();
    toast(`CH${ch+1} → ${JOINT_NAMES[joint]} (${dir > 0 ? 'flex' : 'extend'})`);
  });

  chk.addEventListener('change', () => {
    S.proportional = chk.checked;
    if (chk.checked) {
      // capture current RMS as rest baseline
      S.propRestRms = new Float32Array(S.rms);
      S.propCalibrated = true;
      S._propSmooth = {};  // reset output smoothing
      toast('Proportional ON — keep arm relaxed for 1s baseline');
      // set arm to neutral
      if (window._arm3d) {
        for (const m of S.propMap) window._arm3d.setJoint(m.joint, 90);
      }
    } else {
      toast('Proportional OFF');
      if (window._arm3d) window._arm3d.setGesture('rest');
      HandView.setGesture('rest');
    }
  });

  gainSlider.addEventListener('input', () => {
    S.propGain = parseInt(gainSlider.value);
    gainVal.textContent = `${gainSlider.value}x`;
    localStorage.setItem('propGain', String(S.propGain));
  });

  dzSlider.addEventListener('input', () => {
    S.propDeadZone = parseInt(dzSlider.value);
    dzVal.textContent = `${dzSlider.value}µV`;
    localStorage.setItem('propDeadZone', String(S.propDeadZone));
  });
}

// =============================================================================
// HAND VIEW — Babylon.js rigged glb renderer (dashboard/assets/hand.glb).
//
// The model has 20 generic bones (`Bone`, `Bone.001`..`Bone.019`), so we can't
// look them up by semantic names. Instead we walk the skeleton:
//   1. The root `Bone` has 5 children = the 5 finger roots.
//   2. The chain with 3 bones is the THUMB; the four 4-bone chains are the
//      index/middle/ring/pinky. We order them by world-space X, then flip if
//      needed so the finger nearest the thumb is treated as the index.
// Poses are expressed as per-phalanx curl angles (degrees). Curl axis is
// detected at load time by probing which local axis produces the largest
// displacement when rotated (models vary: some are X, some Z).
// =============================================================================

// Angles in degrees. thumb = [proximal, distal]; fingers = [proximal, middle, distal].
const HAND_POSES = {
  rest:  { thumb:[12, 18], fingers:[[16,22,20],[16,22,20],[16,22,20],[18,24,22]] },
  open:  { thumb:[ 0,  0], fingers:[[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0]] },
  close: { thumb:[40, 55], fingers:[[80,90,78],[80,90,78],[80,90,78],[80,90,78]] },
  pinch: { thumb:[30, 50], fingers:[[35,55,55],[75,85,72],[78,88,75],[80,90,78]] },
  point: { thumb:[20, 28], fingers:[[ 0, 0, 0],[82,90,78],[82,90,78],[82,90,78]] },
  wave:  { thumb:[ 8,  8], fingers:[[ 6, 6, 6],[ 6, 6, 6],[ 6, 6, 6],[ 6, 6, 6]] },
  lift:  { thumb:[ 8,  8], fingers:[[ 8,10,10],[ 8,10,10],[ 8,10,10],[10,12,12]] },
};

const HandView = {
  _engine: null, _scene: null,
  _thumb: [],      // array of TransformNodes: [metacarpal, proximal, distal]
  _fingers: [],    // array of [index, middle, ring, pinky], each 4 TransformNodes
  _restRot: new Map(),  // node.uniqueId -> initial local rotation quaternion
  _curlSign: 1,
  _curlAxis: 'x',
  _last: null,
  _queued: null,
  _booting: false,

  async init() {
    if (this._engine || this._booting) return;
    if (typeof BABYLON === 'undefined') {
      console.warn('[HandView] Babylon not loaded yet; retrying in 300ms');
      setTimeout(() => this.init(), 300);
      return;
    }
    const canvas = document.getElementById('hand-canvas');
    if (!canvas) return;
    this._booting = true;

    const engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true });
    const scene  = new BABYLON.Scene(engine);
    scene.clearColor = new BABYLON.Color4(0.06, 0.08, 0.10, 1);
    this._engine = engine;
    this._scene  = scene;

    // arc-rotate camera — target + radius get fitted to the model after import
    const camera = new BABYLON.ArcRotateCamera(
      'handCam', -Math.PI / 2, Math.PI / 2.4, 1,
      BABYLON.Vector3.Zero(), scene,
    );
    camera.attachControl(canvas, true);
    camera.wheelDeltaPercentage = 0.02;
    camera.minZ = 0.001;
    this._camera = camera;

    // two lights for soft volumetric shading
    const hemi = new BABYLON.HemisphericLight('hemi', new BABYLON.Vector3(0.2, 1, 0.3), scene);
    hemi.intensity = 0.85;
    const dir = new BABYLON.DirectionalLight('dir', new BABYLON.Vector3(-0.6, -1, -0.4), scene);
    dir.intensity = 0.55;

    try {
      const result = await BABYLON.SceneLoader.ImportMeshAsync(
        '', '/static/assets/', 'hand.glb', scene,
      );

      // Kill every imported animation group so it can't overwrite our pose.
      // (The Blender export contains "Armature.003Action" et al. that auto-play.)
      for (const ag of scene.animationGroups.slice()) {
        try { ag.stop(); ag.dispose(); } catch (_) {}
      }

      // Hide decorative props; keep only skinned hand meshes.
      const PROP_NAMES = /^(cube|cylinder|clouds|sphere|circle|plane|light)/i;
      const handMeshes = [];
      for (const m of result.meshes) {
        if (!m.getBoundingInfo) continue;
        if (m.skeleton) handMeshes.push(m);
        else if (PROP_NAMES.test(m.name || '')) m.setEnabled(false);
      }
      if (handMeshes.length > 1) {
        const keepSkel = handMeshes[0].skeleton;
        for (const m of handMeshes) if (m.skeleton !== keepSkel) m.setEnabled(false);
      }
      const framed = handMeshes.filter(m => m.isEnabled());
      this._handMeshes = framed;
      console.log('[HandView] hand meshes kept:', framed.map(m => m.name),
                  ' (hidden', result.meshes.length - framed.length, 'props/dupes)');

      // Frame camera on bbox of the kept hand meshes.
      let min = null, max = null;
      for (const m of framed) {
        m.refreshBoundingInfo(true);
        const bi = m.getBoundingInfo();
        const bmin = bi.boundingBox.minimumWorld;
        const bmax = bi.boundingBox.maximumWorld;
        if (!isFinite(bmin.x) || !isFinite(bmax.x)) continue;
        if (!min) { min = bmin.clone(); max = bmax.clone(); continue; }
        min = BABYLON.Vector3.Minimize(min, bmin);
        max = BABYLON.Vector3.Maximize(max, bmax);
      }
      if (min && max) {
        const center = min.add(max).scale(0.5);
        const size = Math.max(max.subtract(min).length(), 0.01);
        camera.setTarget(center);
        camera.radius = size * 0.75;
        camera.lowerRadiusLimit = size * 0.2;
        camera.upperRadiusLimit = size * 8;
        camera.panningSensibility = 2000 / size;
        console.log('[HandView] bbox size:', size.toFixed(3),
                    ' center:', center.x.toFixed(2), center.y.toFixed(2), center.z.toFixed(2));
      }

      // ---- TransformNode-based skeleton walk ----
      // Babylon's glTF loader puts authoritative pose data on TransformNodes
      // (the glTF joint nodes), not on Bones. We operate on TransformNodes
      // directly and let the skin follow.
      const NODE_RX = /^Bone(\.\d+)?$/;
      const joints = scene.transformNodes.filter(n => NODE_RX.test(n.name || ''));
      if (!joints.length) throw new Error('no Bone.* transform nodes in hand.glb');

      // Ensure every joint has a rotationQuaternion (so assignment sticks).
      for (const j of joints) {
        if (!j.rotationQuaternion) {
          j.rotationQuaternion = BABYLON.Quaternion.FromEulerAngles(
            j.rotation?.x || 0, j.rotation?.y || 0, j.rotation?.z || 0,
          );
        }
      }

      // Root = node named exactly "Bone" whose parent is NOT another Bone.*
      const isJoint = (n) => n && NODE_RX.test(n.name || '');
      const root = joints.find(j => j.name === 'Bone' && !isJoint(j.parent))
                  || joints.find(j => !isJoint(j.parent));
      if (!root) throw new Error('root Bone TransformNode not found');

      const childrenOf = (n) => joints.filter(j => j.parent === n);
      const follow = (n) => {
        const chain = [n];
        let cur = n;
        while (true) {
          const kids = childrenOf(cur);
          if (kids.length !== 1) break;
          cur = kids[0]; chain.push(cur);
        }
        return chain;
      };
      const chains = childrenOf(root).map(follow);

      // 3-node chain = thumb; 4-node chains = fingers
      this._thumb = chains.find(c => c.length === 3) || [];
      let fingers = chains.filter(c => c.length === 4);
      fingers.sort((a, b) => a[0].getAbsolutePosition().x - b[0].getAbsolutePosition().x);
      if (this._thumb.length && fingers.length) {
        const thumbX = this._thumb[0].getAbsolutePosition().x;
        const firstX = fingers[0][0].getAbsolutePosition().x;
        const lastX  = fingers[fingers.length - 1][0].getAbsolutePosition().x;
        if (Math.abs(thumbX - lastX) < Math.abs(thumbX - firstX)) fingers.reverse();
      }
      this._fingers = fingers;

      // Snapshot rest-pose rotation for every joint TransformNode.
      this._restRot = new Map();
      for (const j of joints) this._restRot.set(j.uniqueId, j.rotationQuaternion.clone());

      // Probe the index proximal node to learn which local axis curls toward
      // the palm. Force matrix refresh + a render between probes.
      if (this._fingers.length >= 1 && this._fingers[0].length >= 3) {
        const testNode = this._fingers[0][1];
        const tipNode  = this._fingers[0][this._fingers[0].length - 1];
        const restQ    = this._restRot.get(testNode.uniqueId);

        const setRot = (q) => { testNode.rotationQuaternion = q.clone(); testNode.computeWorldMatrix(true); };
        const tipWorld = () => { tipNode.computeWorldMatrix(true); return tipNode.getAbsolutePosition().clone(); };
        const probe = (ax, ay, az) => {
          setRot(restQ.multiply(BABYLON.Quaternion.FromEulerAngles(ax, ay, az)));
          scene.render();
          const p = tipWorld();
          setRot(restQ);
          return p;
        };
        setRot(restQ); scene.render();
        const tipRest = tipWorld();
        const A = 0.8;
        const candidates = [
          { axis: 'x', sign: +1, pos: probe( A, 0, 0) },
          { axis: 'x', sign: -1, pos: probe(-A, 0, 0) },
          { axis: 'y', sign: +1, pos: probe( 0, A, 0) },
          { axis: 'y', sign: -1, pos: probe( 0,-A, 0) },
          { axis: 'z', sign: +1, pos: probe( 0, 0, A) },
          { axis: 'z', sign: -1, pos: probe( 0, 0,-A) },
        ];
        const handCenter = (min && max) ? min.add(max).scale(0.5) : BABYLON.Vector3.Zero();
        let best = candidates[0], bestScore = -Infinity;
        for (const c of candidates) {
          const disp = c.pos.subtract(tipRest);
          const toCenter = handCenter.subtract(tipRest);
          const dl = disp.length();
          if (dl < 1e-6) continue;
          const tcl = Math.max(toCenter.length(), 1e-6);
          const dot = (disp.x*toCenter.x + disp.y*toCenter.y + disp.z*toCenter.z) / (dl * tcl);
          const score = dl * (0.5 + dot);
          if (score > bestScore) { bestScore = score; best = c; }
        }
        this._curlAxis = best.axis;
        this._curlSign = best.sign;
        console.log('[HandView] probe: best axis=', best.axis, 'sign=', best.sign,
                    'displacement=', bestScore.toFixed(4));
      }

      console.log('[HandView] loaded. thumb:', this._thumb.map(n => n.name),
                  ' fingers:', this._fingers.map(f => f.map(n => n.name)),
                  ' axis:', this._curlAxis, ' sign:', this._curlSign);

      document.getElementById('hand-loading')?.remove();
      engine.runRenderLoop(() => scene.render());
      window.addEventListener('resize', () => engine.resize());

      if (this._queued) { const g = this._queued; this._queued = null; this.setGesture(g); }
      else this.setGesture('rest');
    } catch (err) {
      console.error('[HandView] failed to load hand.glb:', err);
      const el = document.getElementById('hand-loading');
      if (el) { el.textContent = 'hand.glb failed to load — see console'; el.style.color = 'var(--red)'; }
    } finally {
      this._booting = false;
    }
  },

  _curl(node, degrees) {
    if (!node) return;
    const restQ = this._restRot.get(node.uniqueId);
    if (!restQ) return;
    const rad = degrees * Math.PI / 180 * this._curlSign;
    const rot = BABYLON.Quaternion.FromEulerAngles(
      this._curlAxis === 'x' ? rad : 0,
      this._curlAxis === 'y' ? rad : 0,
      this._curlAxis === 'z' ? rad : 0,
    );
    node.rotationQuaternion = restQ.multiply(rot);
  },

  setGesture(name) {
    if (!this._scene || !this._fingers.length) { this._queued = name; this.init(); return; }
    const pose = HAND_POSES[name] || HAND_POSES.rest;
    this._last = name;

    if (this._thumb.length >= 3) {
      this._curl(this._thumb[1], pose.thumb[0]);
      this._curl(this._thumb[2], pose.thumb[1]);
    }
    for (let i = 0; i < this._fingers.length && i < 4; i++) {
      const arr = pose.fingers[i];
      if (!arr || this._fingers[i].length < 4) continue;
      this._curl(this._fingers[i][1], arr[0]);
      this._curl(this._fingers[i][2], arr[1]);
      this._curl(this._fingers[i][3], arr[2]);
    }
  },

  // Debug helpers (open DevTools and call e.g. `HandView.cycle()` to see each
  // gesture land for a second). Also `HandView.setAxis('x', -1)` to override
  // the auto-detected curl axis if the heuristic picked wrong.
  cycle(delayMs = 1000) {
    const seq = ['open','close','pinch','point','rest'];
    let i = 0;
    const step = () => {
      console.log('[HandView] ->', seq[i]);
      this.setGesture(seq[i]);
      i = (i + 1) % seq.length;
    };
    step();
    return setInterval(step, delayMs);
  },
  setAxis(axis, sign) {
    this._curlAxis = axis;
    this._curlSign = sign || 1;
    console.log('[HandView] axis=', axis, ' sign=', this._curlSign);
    const prev = this._last;
    this._last = null;
    this.setGesture(prev || 'close');
  },
};
window.HandView = HandView;
// DevTools helper — run `_HVdebug()` in the console if gestures don't move
// anything. Prints what the view thinks it's controlling + tries a max curl.
window._HVdebug = function() {
  const hv = window.HandView;
  console.log('scene?',       !!hv._scene);
  console.log('joints thumb', hv._thumb.map(n => n.name));
  console.log('joints fingers', hv._fingers.map(f => f.map(n => n.name)));
  console.log('axis/sign',   hv._curlAxis, hv._curlSign);
  if (hv._fingers[0] && hv._fingers[0][1]) {
    const n = hv._fingers[0][1];
    console.log('index proximal node:', n.name, 'rotQ:', n.rotationQuaternion);
  }
  hv.setGesture('close');
  console.log('applied "close" — if hand did not curl, axis guess is wrong; try HandView.setAxis("x",-1), "y",±1, or "z",±1');
  return 'ok';
};

// Toggle between the SVG hand and the Three.js arm viewports.
// Default = hand (faster, no GPU, gesture-focused).
window.switchViz = function(which) {
  const sections = {
    hand: document.getElementById('hand-section'),
    arm: document.getElementById('arm-section'),
    eeg: document.getElementById('eeg-brain-section'),
  };
  if (!sections.hand || !sections.arm || !sections.eeg) return;
  S.activeViz = sections[which] ? which : 'hand';
  Object.entries(sections).forEach(([key, el]) => {
    if (!el) return;
    const active = key === S.activeViz;
    el.style.display = active ? (key === 'hand' ? 'flex' : 'block') : 'none';
  });
  const armLabel = $('arm-gesture-label');
  if (armLabel) armLabel.style.display = S.activeViz === 'arm' && S.supportsArmGestures ? 'block' : 'none';
  document.querySelectorAll('.viz-tab').forEach(b => {
    b.classList.toggle('active', b.dataset.viz === S.activeViz);
  });
  // nudge three.js to resize when revealed
  if (S.activeViz === 'arm' && window._arm3d && window._arm3d.resize) {
    setTimeout(() => window._arm3d.resize(), 50);
  }
  if (S.activeViz === 'eeg' && !S.review.paused) {
    refreshEEGBrainView(true);
  }
};

/**
 * prediction comes in ~20 times/sec with the decoded label + confidence.
 * we bounce the label when the output changes and update any linked outputs.
 */
function onPrediction(d) {
  // skip classifier-driven arm updates when proportional control is active
  if (S.proportional) return;

  const g = d.label || d.gesture || '--';
  const summary = d.summary || '';
  const payload = clonePredictionPayload({ ...d, label: g, gesture: g, summary });
  S.lastPrediction = payload;

  // bounce animation when the decoded label actually changes
  if (g !== S.lastGesture && g !== '--') {
    S.lastGesture = g;

    // update hand render + 3d arm only for arm-capable profiles
    if (S.supportsArmGestures) {
      HandView.setGesture(g);
      if (window._arm3d) window._arm3d.setGesture(g);
      $('arm-gesture-name').textContent = g.toUpperCase();
    }

    // optional sound beep on decoded-label change
    if ($('chk-sound').checked) playBeep(g);

    // run any block program mapped to this decoded label
    checkGestureProgramMapping(g);
  }

  if (!S.review.paused) {
    renderPredictionPanel(payload, { animate: true });
  }

  // track prediction rate and add to timeline
  S.predCount++;
  const gIdx = S.gestures.indexOf(g);
  S.timeline.push({g: gIdx >= 0 ? gIdx : 0, c: d.confidence});
  if (S.timeline.length > S.timelineMax) S.timeline.shift();
  drawTimeline();
}

/** calibration steps get appended to the log box one at a time */
function onCalibration(d) {
  applyCalibrationState(d);
  syncCalibrationUI();
  const log = $('cal-log');
  const line = document.createElement('div');
  line.textContent = `[${d.stage||''}] ${d.message}`;
  if (d.stage === 'complete') line.style.color = 'var(--green)';
  if (d.stage === 'failed')   line.style.color = 'var(--red)';
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}


// =============================================================================
// SYSTEM STATE
// controls which buttons are enabled/disabled, what the badge says, etc
// =============================================================================

function setSysState(s) {
  S.sysState = s;
  const badge = $('state-badge');
  badge.textContent = s;
  badge.className = s;

  const btn = $('btn-stream');
  if (['streaming','training','calibrating'].includes(s)) {
    S.streaming = true;
    btn.textContent = 'Stop Stream';
    btn.className = 'btn danger';
  } else if (s === 'idle') {
    S.streaming = false;
    S.lastSignalAtClient = 0;
    btn.textContent = 'Start Stream';
    btn.className = 'btn primary';
  }
  $('btn-calibrate').disabled = !S.streaming;
  $('btn-fit').disabled = s === 'estop' || !S.supportsTraining;
  $('btn-clear-train').disabled = !S.supportsTraining;
  $('clf-select').disabled = !S.supportsTraining;
  syncStreamModeUI();
  syncReviewUI();
}

function toggleReviewPause(force) {
  const next = typeof force === 'boolean' ? force : !S.review.paused;
  if (next) {
    captureReviewSnapshot();
    S.review.timelineSnapshot = S.timeline.map(item => ({ ...item }));
    S.review.predictionSnapshot = clonePredictionPayload(S.lastPrediction);
    S.review.eegBrainSnapshot = { ...S.eegBrain };
    S.review.paused = true;
  } else {
    S.review.paused = false;
    S.review.snapshot = null;
    S.review.timelineSnapshot = null;
    S.review.predictionSnapshot = null;
    S.review.eegBrainSnapshot = null;
    clearReviewSelection();
  }
  syncReviewUI();
  syncPredictionPanel();
  drawTimeline();
  syncWorkshopUI();
  if (!next && S.activeViz === 'eeg') refreshEEGBrainView(true);
}

async function saveReviewMarker() {
  if (!S.streaming) {
    toast('Start the stream before saving a marker', 'red');
    return;
  }

  const event = ($('review-marker-event')?.value || '').trim();
  const note = ($('review-marker-note')?.value || '').trim();
  if (!event) {
    toast('Enter a marker event first', 'red');
    return;
  }

  const state = getReviewRenderState();
  const stats = S.review.lastStats;
  const selection = stats ? {
    start_s: (stats.startSample - state.baseAbs) / Math.max(state.sampleRate, 1),
    end_s: (stats.endSample - state.baseAbs) / Math.max(state.sampleRate, 1),
    start_sample: stats.startSample,
    end_sample: stats.endSample,
  } : null;

  const body = {
    event,
    note,
    selection_start_s: selection?.start_s,
    selection_end_s: selection?.end_s,
    selection_start_sample: selection?.start_sample,
    selection_end_sample: selection?.end_sample,
    metrics: stats ? {
      duration_ms: Number(stats.durationMs.toFixed(3)),
      rms: Number(stats.rms.toFixed(6)),
      mean: Number(stats.mean.toFixed(6)),
      peak_to_peak: Number(stats.peakToPeak.toFixed(6)),
      min: Number(stats.min.toFixed(6)),
      max: Number(stats.max.toFixed(6)),
      focus_channel: stats.focusLabel,
      focus_rms: Number(stats.focusRms.toFixed(6)),
    } : {},
  };

  try {
    const out = await post('/api/review/marker', body);
    if (!S.ws || S.ws.readyState !== WebSocket.OPEN) {
      pushReviewMarker({
        ...(out.marker || {}),
        createdAt: Date.now(),
        sampleIndex: selection ? selection.end_sample : Math.max(0, state.total - 1),
      });
    }
    toast(selection ? `Range marker saved: ${event}` : `Marker saved: ${event}`);
  } catch (e) {
    toast(e.message, 'red');
  }
}

function bindReviewCanvas() {
  const container = $('canvas-container');
  if (!container || container.dataset.reviewBound === '1') return;
  container.dataset.reviewBound = '1';

  container.addEventListener('pointerdown', e => {
    if (!S.review.paused || e.button !== 0) return;
    const state = getReviewRenderState();
    if (!state.filled) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const sample = sampleFromCanvasX(x, state, canvas.width || rect.width || 1);
    S.review.dragging = true;
    S.review.dragOriginX = x;
    S.review.selection = { startSample: sample, endSample: sample };
    S.review.lastStats = computeSelectionStats(S.review.selection, state);
    syncReviewUI();
    syncWorkshopUI();
  });

  window.addEventListener('pointermove', e => {
    if (!S.review.dragging || !S.review.paused) return;
    const rect = canvas.getBoundingClientRect();
    const x = clamp(e.clientX - rect.left, 0, rect.width);
    const state = getReviewRenderState();
    const sample = sampleFromCanvasX(x, state, canvas.width || rect.width || 1);
    if (!S.review.selection) return;
    S.review.selection.endSample = sample;
    S.review.lastStats = computeSelectionStats(S.review.selection, state);
    syncReviewUI();
    syncWorkshopUI();
  });

  window.addEventListener('pointerup', () => {
    if (!S.review.dragging) return;
    S.review.dragging = false;
    const state = getReviewRenderState();
    S.review.lastStats = computeSelectionStats(S.review.selection, state);
    syncReviewUI();
    syncWorkshopUI();
  });
}


// =============================================================================
// BUTTON WIRING
//
// every button gets its click handler here. i like having them all in one
// place so you don't have to hunt through the HTML for onclick handlers.
// =============================================================================

function bindButtons() {
  const sourceSel = $('stream-source');
  const playbackSel = $('playback-session');
  const playbackRateSel = $('playback-rate');
  const profileSel = $('profile-select');
  const sessionSubjectInput = $('session-subject');
  const lslInputSel = $('lsl-input-stream');
  const lslName = $('lsl-name');
  const lslMarkers = $('lsl-markers');
  const oscHost = $('osc-host');
  const oscPort = $('osc-port');
  const oscPrefix = $('osc-prefix');
  const oscEvents = $('osc-events');
  const tourBtn = $('btn-tour');
  const markerLink = $('eeg-brain-marker-link');
  if (tourBtn) {
    tourBtn.onclick = () => {
      if ($('block-editor')?.classList.contains('active')) window.startBlocksTour();
      else if ($('signal-workshop')?.classList.contains('active')) window.startWorkshopTour();
      else if ($('filter-lab')?.classList.contains('active') || $('bench-report')?.classList.contains('active')) window.startFilterTour();
      else window.startQuickTour();
    };
  }
  if (markerLink && markerLink.dataset.bound !== '1') {
    markerLink.dataset.bound = '1';
    markerLink.onclick = e => {
      if (markerLink.getAttribute('aria-disabled') === 'true') {
        e.preventDefault();
        toast(markerLink.title || 'Marker reference is not available yet.', 'yellow');
        return;
      }
      const href = markerLink.dataset.href || markerLink.href;
      if (!href || href === '#') {
        e.preventDefault();
        toast('Marker reference is not ready yet.', 'yellow');
        return;
      }
      e.preventDefault();
      window.open(href, '_blank', 'noopener,noreferrer');
    };
  }
  const blocksTourBtn = $('btn-blocks-tour');
  if (blocksTourBtn) {
    blocksTourBtn.onclick = () => window.startBlocksTour();
  }
  const filterTourBtn = $('btn-filter-tour');
  if (filterTourBtn) {
    filterTourBtn.onclick = () => window.startFilterTour();
  }
  const workshopTourBtn = $('btn-workshop-tour');
  if (workshopTourBtn) {
    workshopTourBtn.onclick = () => window.startWorkshopTour();
  }
  const blockExampleBtn = $('btn-block-example');
  if (blockExampleBtn) {
    blockExampleBtn.onclick = () => window.loadSignalExampleProgram();
  }
  $('btn-filter-preview')?.addEventListener('click', previewFilterDesign);
  $('btn-filter-save')?.addEventListener('click', saveFilterDesign);
  $('btn-filter-activate')?.addEventListener('click', activateSelectedFilter);
  $('btn-filter-clear')?.addEventListener('click', clearActiveFilter);
  $('btn-filter-export')?.addEventListener('click', openFilterExportModal);
  $('btn-bench-refresh')?.addEventListener('click', refreshBenchReportUI);
  $('btn-bench-export')?.addEventListener('click', openBenchReportModal);
  $('btn-workshop-analyze')?.addEventListener('click', () => analyzeWorkshopSelection());
  $('btn-workshop-from-review')?.addEventListener('click', () => analyzeWorkshopSelection({ switchToTab: true }));
  $('btn-workshop-refresh')?.addEventListener('click', () => analyzeWorkshopSelection());
  const workshopViewSel = $('workshop-view');
  if (workshopViewSel) {
    const storedWorkshopView = localStorage.getItem('kyma-workshop-view');
    if (storedWorkshopView) {
      S.workshop.view = storedWorkshopView;
      workshopViewSel.value = storedWorkshopView;
    }
    workshopViewSel.addEventListener('change', () => {
      S.workshop.view = workshopViewSel.value || 'fft';
      localStorage.setItem('kyma-workshop-view', S.workshop.view);
      syncWorkshopUI();
    });
  }
  $('workshop-focus-channel')?.addEventListener('change', () => {
    if (S.workshop.selectionMeta) {
      S.workshop.selectionMeta.focusChannel = Number($('workshop-focus-channel')?.value || 0);
    }
  });
  $('filter-export-target')?.addEventListener('change', renderFilterExportPreview);
  $('filter-response-type')?.addEventListener('change', updateFilterFieldVisibility);
  $('filter-method')?.addEventListener('change', updateFilterFieldVisibility);
  if (sourceSel) {
    const initialSource = localStorage.getItem('kyma-stream-source') || S.streamSource || 'hardware';
    setSelectedSource(initialSource);
    sourceSel.onchange = async () => {
      localStorage.setItem('kyma-stream-source', sourceSel.value);
      if (sourceSel.value === 'lsl') {
        await loadLSLInputs();
      }
      syncStreamModeUI();
    };
    syncStreamModeUI();
  }
  if (playbackSel) {
    playbackSel.onchange = () => {
      setSelectedPlaybackSession(playbackSel.value);
      syncStreamModeUI();
    };
  }
  if (playbackRateSel) {
    const storedRate = String(S.streamDetails?.playback_rate || localStorage.getItem('kyma-playback-rate') || '1');
    playbackRateSel.value = storedRate;
    playbackRateSel.onchange = () => {
      localStorage.setItem('kyma-playback-rate', playbackRateSel.value);
      syncStreamModeUI();
    };
  }
  if (lslInputSel) {
    lslInputSel.onchange = () => {
      setSelectedLSLInput(lslInputSel.value);
      syncLSLInputUI();
      syncStreamModeUI();
    };
  }
  if (lslName) {
    const storedName = localStorage.getItem('kyma-lsl-name') || '';
    if (storedName) lslName.value = storedName;
    lslName.onchange = () => localStorage.setItem('kyma-lsl-name', lslName.value.trim());
  }
  if (lslMarkers) {
    const storedMarkers = localStorage.getItem('kyma-lsl-markers');
    if (storedMarkers !== null) lslMarkers.checked = storedMarkers === '1';
    lslMarkers.onchange = () => localStorage.setItem('kyma-lsl-markers', lslMarkers.checked ? '1' : '0');
  }
  if (oscHost) {
    const storedHost = localStorage.getItem('kyma-osc-host') || '127.0.0.1';
    if (!oscHost.value) oscHost.value = storedHost;
    oscHost.onchange = () => localStorage.setItem('kyma-osc-host', oscHost.value.trim() || '127.0.0.1');
  }
  if (oscPort) {
    const storedPort = localStorage.getItem('kyma-osc-port') || '9000';
    if (!oscPort.value) oscPort.value = storedPort;
    oscPort.onchange = () => localStorage.setItem('kyma-osc-port', String(oscPort.value || '9000'));
  }
  if (oscPrefix) {
    const storedPrefix = localStorage.getItem('kyma-osc-prefix') || '/kyma';
    if (!oscPrefix.value) oscPrefix.value = storedPrefix;
    oscPrefix.onchange = () => localStorage.setItem('kyma-osc-prefix', oscPrefix.value.trim() || '/kyma');
  }
  if (oscEvents) {
    const storedEvents = localStorage.getItem('kyma-osc-events');
    if (storedEvents !== null) oscEvents.checked = storedEvents === '1';
    oscEvents.onchange = () => localStorage.setItem('kyma-osc-events', oscEvents.checked ? '1' : '0');
  }
  const protocolSel = $('protocol-template');
  if (protocolSel) {
    protocolSel.onchange = () => {
      resetProtocolRun();
      syncProtocolUI();
    };
  }
  const protocolRunInput = $('protocol-run-id');
  if (protocolRunInput) {
    protocolRunInput.onchange = () => {
      S.protocolRunId = protocolRunInput.value.trim();
      syncProtocolUI();
    };
  }
  const eegExperimentSel = $('eeg-experiment-select');
  if (eegExperimentSel) {
    eegExperimentSel.onchange = () => {
      S.selectedEegExperiment = eegExperimentSel.value || '';
      if (S.selectedEegExperiment) localStorage.setItem('kyma-eeg-experiment', S.selectedEegExperiment);
      else localStorage.removeItem('kyma-eeg-experiment');
      syncEEGExperimentUI();
      syncEEGMarkerHelperUI();
      syncLSLMarkerTesterUI();
    };
  }
  const eegExperimentApplyBtn = $('btn-eeg-experiment-apply');
  if (eegExperimentApplyBtn) {
    eegExperimentApplyBtn.onclick = async () => {
      const preset = getSelectedEEGExperiment();
      if (!preset) {
        toast('Select an EEG experiment preset first', 'red');
        return;
      }
      if (S.signalProfileKey !== 'eeg') {
        toast('Switch Signal Type to EEG first', 'yellow');
        return;
      }
      if ($('session-condition')) $('session-condition').value = preset.session_condition || preset.key;
      if ($('session-notes')) $('session-notes').value = `${preset.title}: ${preset.summary}`;
      resetProtocolRun();
      if (!S.streaming && preset.recommended_source) {
        setSelectedSource(preset.recommended_source);
        if (preset.recommended_source === 'lsl') await loadLSLInputs();
      }
      syncStreamModeUI();
      syncProtocolUI();
      syncEEGExperimentUI();
      syncEEGMarkerHelperUI();
      syncLSLMarkerTesterUI();
      toast(`EEG preset applied: ${preset.title}`);
    };
  }
  const lslMarkerSelect = $('lsl-marker-select');
  if (lslMarkerSelect) {
    const storedMarkerName = localStorage.getItem('kyma-lsl-marker-name');
    if (storedMarkerName && !lslMarkerSelect.value) lslMarkerSelect.value = storedMarkerName;
    lslMarkerSelect.onchange = () => {
      localStorage.setItem('kyma-lsl-marker-name', lslMarkerSelect.value || '');
      if ($('lsl-marker-event') && lslMarkerSelect.value) $('lsl-marker-event').value = lslMarkerSelect.value;
    };
  }
  const lslMarkerEvent = $('lsl-marker-event');
  if (lslMarkerEvent) {
    const storedMarkerEvent = localStorage.getItem('kyma-lsl-marker-event');
    if (storedMarkerEvent && !lslMarkerEvent.value) lslMarkerEvent.value = storedMarkerEvent;
    lslMarkerEvent.onchange = () => {
      if (lslMarkerEvent.value.trim()) localStorage.setItem('kyma-lsl-marker-event', lslMarkerEvent.value.trim());
      else localStorage.removeItem('kyma-lsl-marker-event');
    };
  }
  const lslMarkerPayload = $('lsl-marker-payload');
  if (lslMarkerPayload) {
    const storedMarkerPayload = localStorage.getItem('kyma-lsl-marker-payload');
    if (storedMarkerPayload && !lslMarkerPayload.value.trim()) lslMarkerPayload.value = storedMarkerPayload;
    lslMarkerPayload.onchange = () => localStorage.setItem('kyma-lsl-marker-payload', lslMarkerPayload.value);
  }
  const lslMarkerSendBtn = $('btn-send-lsl-marker');
  if (lslMarkerSendBtn) {
    lslMarkerSendBtn.onclick = async () => {
      const eventName = ($('lsl-marker-event')?.value || '').trim();
      if (!eventName) {
        toast('Enter a marker event first', 'red');
        return;
      }
      let payload = {};
      const raw = ($('lsl-marker-payload')?.value || '').trim();
      if (raw) {
        try {
          payload = JSON.parse(raw);
        } catch (e) {
          toast('Payload JSON is invalid', 'red');
          return;
        }
      }
      payload = {
        subject_id: $('session-subject')?.value?.trim() || '',
        condition: $('session-condition')?.value?.trim() || '',
        ...payload,
      };
      try {
        const r = await post('/api/lsl/marker', { event: eventName, payload });
        if ($('lsl-marker-status')) {
          $('lsl-marker-status').textContent = `Sent "${r.event}" to ${r.marker_stream_name || 'LSL markers'} at ${new Date().toLocaleTimeString()}.`;
        }
        toast(`Marker sent: ${r.event}`);
      } catch (e) {
        toast(e.message, 'red');
      }
    };
  }
  const subjectSelect = $('subject-registry-select');
  if (subjectSelect) {
    subjectSelect.onchange = () => {
      const record = findSubjectRecord(subjectSelect.value);
      if (record) populateSubjectRegistryForm(record);
      syncSubjectRegistryUI();
    };
  }
  const subjectApplyBtn = $('btn-subject-apply');
  if (subjectApplyBtn) {
    subjectApplyBtn.onclick = () => {
      const subjectId = $('subject-registry-id')?.value?.trim() || $('subject-registry-select')?.value || '';
      const record = findSubjectRecord(subjectId) || (subjectId ? { subject_id: subjectId } : null);
      if (!record) {
        toast('Choose or enter a subject first', 'red');
        return;
      }
      applySubjectToSession(record);
      syncSubjectRegistryUI();
      toast(`Session subject set: ${record.subject_id}`);
    };
  }
  const subjectNewBtn = $('btn-subject-new');
  if (subjectNewBtn) {
    subjectNewBtn.onclick = () => {
      if ($('subject-registry-select')) $('subject-registry-select').value = '';
      populateSubjectRegistryForm(null);
      $('subject-registry-status').textContent = 'New subject draft. Save it to add it to the registry.';
    };
  }
  const subjectSaveBtn = $('btn-subject-save');
  if (subjectSaveBtn) {
    subjectSaveBtn.onclick = async () => {
      const payload = {
        subject_id: $('subject-registry-id')?.value?.trim() || '',
        display_name: $('subject-registry-name')?.value?.trim() || '',
        cohort: $('subject-registry-cohort')?.value?.trim() || '',
        handedness: $('subject-registry-handedness')?.value || '',
        notes: $('subject-registry-notes')?.value?.trim() || '',
      };
      if (!payload.subject_id) {
        toast('Subject ID is required', 'red');
        return;
      }
      try {
        subjectSaveBtn.disabled = true;
        const out = await post('/api/subjects', payload);
        applySubjectToSession(out.subject || payload);
        await loadSubjects();
        if ($('subject-registry-select')) $('subject-registry-select').value = out.subject?.subject_id || payload.subject_id;
        syncSubjectRegistryUI();
        toast(`Subject saved: ${out.subject?.subject_id || payload.subject_id}`);
      } catch (e) {
        toast(e.message, 'red');
      } finally {
        subjectSaveBtn.disabled = false;
      }
    };
  }
  if (sessionSubjectInput) {
    sessionSubjectInput.onchange = () => {
      syncSubjectRegistryUI();
    };
  }

  if (profileSel) {
    profileSel.onchange = async () => {
      try {
        const r = await post('/api/profile', { profile: profileSel.value });
        applySignalProfile(r.signal_profile || {});
        await loadConfig();
        await loadFilterLabStatus();
        await loadDatasets();
        await loadExperiments();
        syncStreamModeUI();
        await refreshLSLStatus();
        toast(`Profile set: ${S.signalProfileName}`);
      } catch (e) {
        toast(e.message, 'red');
        await loadConfig();
        await loadFilterLabStatus();
      }
    };
  }

  // --- emergency stop: kills servo power immediately ---
  $('btn-estop').onclick = async () => {
    try { await post('/api/estop'); toast('E-STOP activated', 'red'); }
    catch (e) { toast(e.message, 'red'); }
  };

  // --- home: returns all servos to 90 degrees ---
  $('btn-home').onclick = async () => {
    try {
      await post('/api/home');
      toast('Homing...');
      HandView.setGesture('rest');
      if (window._arm3d) window._arm3d.setGesture('rest');
    } catch (e) { toast(e.message, 'red'); }
  };

  // --- scan serial ports ---
  $('btn-refresh-ports').onclick = () => scanPorts();
  const refreshLslInputsBtn = $('btn-refresh-lsl-inputs');
  if (refreshLslInputsBtn) refreshLslInputsBtn.onclick = () => loadLSLInputs();

  // --- start/stop the brainflow stream ---
  $('btn-stream').onclick = async () => {
    try {
      if (S.streaming) {
        const r = await post('/api/stream/stop');
        S.streamDetails = {};
        resetReviewState({ clearMarkers: true });
        if (r.saved_to) {
          S.recSession = false;
          await loadSessions();
          toast(`Saved: ${r.saved_to}`);
        }
        await refreshLSLStatus();
      } else {
        const source = getSelectedSource();
        const body = { source };
        const cPort = $('cyton-port').value;
        if (source === 'hardware' && cPort) body.cyton_port = cPort;
        if (source === 'lsl') {
          const active = S.lslInputs.find(stream => (stream.source_id || stream.uid || stream.name) === getSelectedLSLInput());
          if (!active) {
            toast('Select an external LSL stream first', 'red');
            return;
          }
          body.lsl_stream_name = active.name;
          body.lsl_source_id = active.source_id || null;
        }
        if (source === 'playback') {
          const sessionId = $('playback-session')?.value || '';
          if (!sessionId) {
            toast('Select a recorded session first', 'red');
            return;
          }
          body.playback_session_id = sessionId;
          body.playback_rate = Number($('playback-rate')?.value || '1');
        }
        const aPort = $('arduino-port').value;
        if (aPort) body.arduino_port = aPort;
        const r = await post('/api/stream/start', body);
        S.streamSource = r.stream_source || source;
        S.streamDetails = r.stream_details || {};
        S.playbackSessionId = r.playback_session_id || body.playback_session_id || '';
        resetReviewState({ clearMarkers: true });
        applySignalProfile(r.signal_profile || {});
        syncProfileUI();
        await refreshLSLStatus();
        if (S.streamSource === 'synthetic') {
          toast(`Synthetic ${S.signalProfileName} stream started`);
        } else if (S.streamSource === 'playback') {
          toast(`Playback started: ${S.playbackSessionId}`);
        } else if (S.streamSource === 'lsl') {
          toast(`LSL input started: ${S.streamDetails?.name || S.streamDetails?.stream_name || 'stream'}`);
        } else {
          toast(`${S.signalProfileName} hardware stream started`);
        }
      }
    } catch (e) { toast(e.message, 'red'); }
  };

  // --- run the 3-stage calibration routine ---
  $('btn-calibrate').onclick = async () => {
    $('cal-log').innerHTML = '';
    try {
      const r = await post('/api/calibrate');
      applyCalibrationState({ stage: 'calibrating', protocol: r.protocol || null });
      syncCalibrationUI();
      toast('Calibration started');
    }
    catch (e) { toast(e.message, 'red'); }
  };

  // --- fit the selected classifier on all recorded data ---
  // LDA takes ~1 second, TCN takes 30-60 seconds, Mamba takes 60-120 seconds
  $('btn-fit').onclick = async () => {
    if (!S.supportsTraining) {
      toast(`${S.signalProfileName} training is not implemented yet`, 'yellow');
      return;
    }
    const clf = $('clf-select').value;
    try {
      toast(`Training ${clf}... (LDA ~1s, TCN ~30s, Mamba ~60s)`);
      const r = await post(`/api/train/fit?classifier=${clf}`);
      if (r.success) {
        S.trained = true;
        const acc = r.val_accuracy != null ? (r.val_accuracy*100).toFixed(1) : '?';
        toast(`${clf} done - acc=${acc}% params=${r.n_params||'?'}`);
        $('train-status').textContent = `${clf} | acc ${acc}% | ${r.n_params||'?'} params`;
      } else {
        toast(r.error, 'red');
      }
    } catch (e) { toast(e.message, 'red'); }
  };

  // --- clear training data (double-click confirm so you don't lose work) ---
  let armed = false, armedTimer;
  $('btn-clear-train').onclick = async () => {
    if (!S.supportsTraining) {
      toast(`${S.signalProfileName} does not use the training path in this build`, 'yellow');
      return;
    }
    if (!armed) {
      // first click: arm it. button turns red and says "Confirm?"
      armed = true;
      $('btn-clear-train').textContent = 'Confirm?';
      $('btn-clear-train').style.background = 'var(--red)';
      $('btn-clear-train').style.color = '#fff';
      armedTimer = setTimeout(() => {
        armed = false;
        $('btn-clear-train').textContent = 'Clear';
        $('btn-clear-train').style.background = '';
        $('btn-clear-train').style.color = '';
      }, 3000);
      return;
    }
    // second click within 3s: actually clear
    clearTimeout(armedTimer);
    armed = false;
    $('btn-clear-train').textContent = 'Clear';
    $('btn-clear-train').style.background = '';
    $('btn-clear-train').style.color = '';
    try {
      await post('/api/train/clear');
      S.trainCounts = {};
      S.gestures.forEach(g => setGestureCount(g, 0));
      $('train-status').textContent = '';
      toast('Training data cleared');
    } catch (e) { toast(e.message, 'red'); }
  };

  // --- wipe the emg canvas buffer (visual only, doesn't affect server) ---
  $('btn-clear-stream').onclick = () => {
    for (let c = 0; c < N_CH; c++) S.emg[c].fill(0);
    S.emgHead = 0;
    S.emgTotal = 0;
    resetReviewState({ clearMarkers: true });
    toast('Stream cleared');
  };

  const reviewPauseBtn = $('btn-review-pause');
  if (reviewPauseBtn) {
    reviewPauseBtn.onclick = () => toggleReviewPause();
  }
  const reviewClearBtn = $('btn-review-clear-selection');
  if (reviewClearBtn) {
    reviewClearBtn.onclick = () => {
      clearReviewSelection();
      syncReviewUI();
      syncWorkshopUI();
    };
  }
  const reviewMarkerBtn = $('btn-review-marker');
  if (reviewMarkerBtn) {
    reviewMarkerBtn.onclick = () => saveReviewMarker();
  }
  const reviewWorkshopBtn = $('btn-review-workshop');
  if (reviewWorkshopBtn) {
    reviewWorkshopBtn.onclick = () => analyzeWorkshopSelection({ switchToTab: true });
  }
  const reviewMarkerEvent = $('review-marker-event');
  if (reviewMarkerEvent) {
    reviewMarkerEvent.onchange = () => {
      const next = reviewMarkerEvent.value.trim();
      if (next) localStorage.setItem('kyma-review-marker-event', next);
      else localStorage.removeItem('kyma-review-marker-event');
    };
    const storedReviewEvent = localStorage.getItem('kyma-review-marker-event');
    if (storedReviewEvent && !reviewMarkerEvent.value) reviewMarkerEvent.value = storedReviewEvent;
  }

  const startSessionRecording = async (override = {}) => {
    if (!S.streaming) {
      toast('Start the stream before recording a session', 'red');
      return;
    }
    const payload = buildSessionStartPayload(override);
    const r = await post('/api/session/start', payload);
    S.recSession = true;
    $('btn-record-session').textContent = 'Stop Rec';
    $('btn-record-session').className = 'btn danger';
    syncProtocolUI();
    await loadSubjects();
    toast(`Recording: ${r.session_id}`);
  };

  // --- toggle session recording (saves raw signal to disk) ---
  $('btn-record-session').onclick = async () => {
    try {
      if (S.recSession) {
        const r = await post('/api/session/stop');
        S.recSession = false;
        $('btn-record-session').textContent = 'Rec Session';
        $('btn-record-session').className = 'btn';
        syncProtocolUI();
        toast(`Saved: ${r.saved_to}`);
        await loadSessions();
      } else {
        await startSessionRecording();
      }
    } catch (e) { toast(e.message, 'red'); }
  };

  const protocolNextBtn = $('btn-protocol-next');
  if (protocolNextBtn) {
    protocolNextBtn.onclick = async () => {
      try {
        if (S.recSession) {
          toast('Stop the active session before starting the next protocol trial', 'red');
          return;
        }
        const template = getSelectedProtocolTemplate();
        if (!template) {
          toast('Select a protocol template first', 'red');
          return;
        }
        const plan = buildProtocolPlan(template);
        const nextStep = plan[S.protocolStepIndex];
        if (!nextStep) {
          toast('This protocol run is complete. Reset the run to start again.', 'yellow');
          return;
        }
        const runId = ensureProtocolRunId(template);
        if ($('session-label')) $('session-label').value = nextStep.label;
        await startSessionRecording({
          label: nextStep.label,
          protocol_key: template.key,
          protocol_title: template.title || template.key,
          session_group_id: runId,
          trial_index: nextStep.trial_index,
          repetition_index: nextStep.repetition_index,
        });
        S.protocolStepIndex += 1;
        syncProtocolUI();
      } catch (e) {
        toast(e.message, 'red');
      }
    };
  }

  const protocolResetBtn = $('btn-protocol-reset');
  if (protocolResetBtn) {
    protocolResetBtn.onclick = () => {
      resetProtocolRun();
      syncProtocolUI();
      toast('Protocol run reset');
    };
  }

  const refreshResearch = async () => {
    await loadSessions();
    await loadSubjects();
    await loadDatasets();
    await loadExperiments();
    toast('Research lists refreshed');
  };
  $('btn-sessions-refresh').onclick = refreshResearch;
  const researchRefreshBtn = $('btn-research-refresh');
  if (researchRefreshBtn) researchRefreshBtn.onclick = refreshResearch;
  const datasetSel = $('dataset-select');
  if (datasetSel) {
    datasetSel.onchange = () => {
      setSelectedDataset(datasetSel.value);
      syncResearchUI();
    };
  }
  const experimentSplitSel = $('experiment-split');
  if (experimentSplitSel) {
    experimentSplitSel.onchange = () => {
      syncResearchUI();
    };
  }
  const experimentHoldoutInput = $('experiment-holdout');
  if (experimentHoldoutInput) {
    experimentHoldoutInput.onchange = () => {
      syncResearchUI();
    };
  }
  const experimentGapInput = $('experiment-gap');
  if (experimentGapInput) {
    experimentGapInput.onchange = () => {
      syncResearchUI();
    };
  }
  const createDatasetBtn = $('btn-dataset-create');
  if (createDatasetBtn) {
    createDatasetBtn.onclick = async () => {
      const sessionIds = [...S.selectedSessionIds];
      if (!sessionIds.length) {
        toast('Select one or more sessions first', 'red');
        return;
      }
      try {
        createDatasetBtn.disabled = true;
        const body = {
          name: $('dataset-name')?.value?.trim() || '',
          session_ids: sessionIds,
        };
        const out = await post('/api/datasets', body);
        await loadDatasets();
        setSelectedDataset(out.dataset?.dataset_id || S.selectedDatasetId);
        if ($('dataset-name')) $('dataset-name').value = '';
        syncResearchUI();
        toast(`Dataset created: ${out.dataset?.name || out.dataset?.dataset_id || 'dataset'}`);
      } catch (e) {
        toast(e.message, 'red');
      } finally {
        createDatasetBtn.disabled = false;
      }
    };
  }
  const runExperimentBtn = $('btn-experiment-run');
  if (runExperimentBtn) {
    runExperimentBtn.onclick = async () => {
      if (!S.selectedDatasetId) {
        toast('Choose a dataset first', 'red');
        return;
      }
      const classifier = $('experiment-clf')?.value || 'LDA';
      const splitStrategy = $('experiment-split')?.value || 'temporal_holdout';
      const holdoutFraction = Math.min(0.5, Math.max(0.1, Number($('experiment-holdout')?.value || 0.25)));
      const holdoutGap = Math.min(2.0, Math.max(0.0, Number($('experiment-gap')?.value || 0.2)));
      const notes = $('experiment-notes')?.value?.trim() || '';
      const status = $('experiment-status');
      try {
        runExperimentBtn.disabled = true;
        if (status) status.textContent = `Running ${classifier} with ${humanizeExperimentSplit(splitStrategy)}...`;
        const out = await post('/api/experiments/run', {
          dataset_id: S.selectedDatasetId,
          classifier,
          notes,
          split_strategy: splitStrategy,
          holdout_fraction: holdoutFraction,
          holdout_gap_s: holdoutGap,
        });
        await loadExperiments();
        const result = out.report?.result || {};
        const acc = result.val_accuracy != null ? `${(Number(result.val_accuracy) * 100).toFixed(1)}%` : 'n/a';
        const splitName = humanizeExperimentSplit(out.report?.split?.strategy || splitStrategy);
        if ($('experiment-notes')) $('experiment-notes').value = '';
        if (status) {
          status.textContent = out.report?.status === 'completed'
            ? `Experiment complete: ${classifier}, ${splitName}, ${acc}`
            : (out.report?.error || 'Experiment failed');
        }
        toast(
          out.report?.status === 'completed'
            ? `${classifier} experiment complete (${splitName}, ${acc})`
            : (out.report?.error || 'Experiment failed'),
          out.report?.status === 'completed' ? '' : 'red',
        );
      } catch (e) {
        if (status) status.textContent = e.message || 'Experiment failed';
        toast(e.message, 'red');
      } finally {
        runExperimentBtn.disabled = false;
        syncResearchUI();
      }
    };
  }
  const xdfInspectBtn = $('btn-xdf-inspect');
  if (xdfInspectBtn) {
    xdfInspectBtn.onclick = async () => {
      const path = $('xdf-path')?.value?.trim() || '';
      const status = $('xdf-status');
      if (!path) {
        toast('Enter an XDF file path first', 'red');
        return;
      }
      try {
        xdfInspectBtn.disabled = true;
        if (status) status.textContent = 'Inspecting XDF...';
        const out = await post('/api/xdf/inspect', { path });
        populateXDFStreamOptions(out.streams || []);
        if (status) {
          status.textContent = (out.streams || []).length
            ? `Found ${(out.streams || []).length} stream(s). Select a numeric signal stream to import.`
            : 'No numeric streams found in the XDF file.';
        }
        toast(`XDF inspected: ${(out.streams || []).length} stream(s)`);
      } catch (e) {
        if (status) status.textContent = e.message || 'XDF inspect failed.';
        toast(e.message || 'XDF inspect failed', 'red');
      } finally {
        xdfInspectBtn.disabled = false;
      }
    };
  }
  const xdfImportBtn = $('btn-xdf-import');
  if (xdfImportBtn) {
    xdfImportBtn.onclick = async () => {
      const path = $('xdf-path')?.value?.trim() || '';
      const selected = $('xdf-stream')?.value || '';
      const chosen = S.xdfStreams.find(stream => (stream.stream_id || stream.name) === selected);
      const status = $('xdf-status');
      if (!path) {
        toast('Enter an XDF file path first', 'red');
        return;
      }
      try {
        xdfImportBtn.disabled = true;
        if (status) status.textContent = 'Importing XDF into sessions...';
        const body = {
          path,
          stream_id: chosen?.stream_id || null,
          stream_name: chosen?.name || null,
        };
        const out = await post('/api/xdf/import', body);
        if (status) status.textContent = `Imported ${out.session_id} (${out.n_samples} samples).`;
        await loadSessions();
        await loadDatasets();
        toast(`XDF imported: ${out.session_id}`);
      } catch (e) {
        if (status) status.textContent = e.message || 'XDF import failed.';
        toast(e.message || 'XDF import failed', 'red');
      } finally {
        xdfImportBtn.disabled = false;
      }
    };
  }
  $('btn-lsl').onclick = async () => {
    try {
      if (S.lsl.active) {
        await post('/api/lsl/stop');
        await refreshLSLStatus();
        toast('LSL stopped');
      } else {
        const body = {
          stream_name: $('lsl-name')?.value?.trim() || null,
          include_markers: $('lsl-markers')?.checked !== false,
        };
        const r = await post('/api/lsl/start', body);
        applyLSLStatus(r.lsl || {});
        syncLSLUI();
        toast(`LSL active: ${S.lsl.stream_name}`);
      }
    } catch (e) { toast(e.message, 'red'); }
  };

  $('btn-osc').onclick = async () => {
    try {
      if (S.osc.active) {
        await post('/api/osc/stop');
        await refreshOSCStatus();
        toast('OSC stopped');
      } else {
        localStorage.setItem('kyma-osc-host', $('osc-host')?.value?.trim() || '127.0.0.1');
        localStorage.setItem('kyma-osc-port', String($('osc-port')?.value || 9000));
        localStorage.setItem('kyma-osc-prefix', $('osc-prefix')?.value?.trim() || '/kyma');
        localStorage.setItem('kyma-osc-events', $('osc-events')?.checked ? '1' : '0');
        const body = {
          host: $('osc-host')?.value?.trim() || '127.0.0.1',
          port: Number($('osc-port')?.value || 9000),
          prefix: $('osc-prefix')?.value?.trim() || '/kyma',
          mirror_events: $('osc-events')?.checked !== false,
        };
        const r = await post('/api/osc/start', body);
        applyOSCStatus(r.osc || {});
        syncOSCUI();
        toast(`OSC active: ${S.osc.host}:${S.osc.port}`);
      }
    } catch (e) { toast(e.message, 'red'); }
  };
}


// =============================================================================
// GESTURE TRAINING
//
// you hold down the "Hold" button next to each gesture name while making
// that pose with your hand. the server records EMG windows the whole time.
// do this for each gesture, then hit "Train Model".
// =============================================================================

function buildGestureList() {
  const c = $('gesture-list');
  c.innerHTML = '';
  if (!S.supportsTraining) {
    c.innerHTML = `<div style="font-size:11px;color:var(--text-dim);line-height:1.4">${S.signalProfileName} currently uses the live analyzer path in this build.</div>`;
    return;
  }
  if (!S.gestures.length) {
    c.innerHTML = `<div style="font-size:11px;color:var(--text-dim);line-height:1.4">No trainable labels are defined for ${S.signalProfileName}.</div>`;
    return;
  }
  S.gestures.forEach(g => {
    const row = document.createElement('div');
    row.className = 'gesture-row';
    row.innerHTML = `
      <span class="name">${g}</span>
      <span class="count" id="gc-${g}">0 win</span>
      <button class="btn-rec" id="gr-${g}"
        onmousedown="recStart('${g}')" onmouseup="recStop()"
        ontouchstart="recStart('${g}')" ontouchend="recStop()">Hold</button>`;
    c.appendChild(row);
  });
}

// these are global so the inline event handlers can reach them
window.recStart = async function(g) {
  if (!S.streaming) { toast('Start stream first', 'red'); return; }
  if (!S.supportsTraining) { toast(`${S.signalProfileName} training is not implemented yet`, 'yellow'); return; }
  try {
    await post('/api/train/start', {gesture:g, duration_s:10});
    S.recGesture = g;
    $(`gr-${g}`).classList.add('active');
    $(`gr-${g}`).textContent = 'REC';
    toast(`Recording "${g}"...`);
    // poll the window count while they're holding the button
    S._pollId = setInterval(refreshTrainSummary, 500);
  } catch (e) { toast(e.message, 'red'); }
};

window.recStop = async function() {
  if (!S.recGesture) return;
  const g = S.recGesture;
  clearInterval(S._pollId);
  try {
    const r = await post('/api/train/stop');
    S.trainCounts = r.summary?.per_gesture || S.trainCounts;
    setGestureCount(g, S.trainCounts[g] || 0);
    $(`gr-${g}`).classList.remove('active');
    $(`gr-${g}`).textContent = 'Hold';
    S.recGesture = null;
    toast(`"${g}" -- ${S.trainCounts[g]||0} windows`);
  } catch (e) { toast(e.message, 'red'); }
};

function setGestureCount(g, n) {
  const el = $(`gc-${g}`);
  if (el) el.textContent = `${n} win`;
}

async function refreshTrainSummary() {
  try {
    const s = await get('/api/train/summary');
    S.trainCounts = s.per_gesture || {};
    S.trained = s.is_trained;
    Object.entries(S.trainCounts).forEach(([g,n]) => setGestureCount(g, n));
  } catch {}
}


// =============================================================================
// QUICK GESTURE BUTTONS (manually trigger a named pose on the arm)
// =============================================================================

function buildQuickGestures() {
  const c = $('quick-gestures');
  c.innerHTML = '';

  if (!S.supportsArmGestures) {
    c.innerHTML = `<div style="font-size:11px;color:var(--text-dim);line-height:1.4">${S.signalProfileName} does not expose robotic-arm shortcut poses.</div>`;
    return;
  }

  // built-in gestures
  S.gestures.forEach(g => {
    const b = document.createElement('button');
    b.className = 'btn'; b.textContent = g;
    b.style.width = 'auto'; b.style.flex = '1';
    b.onclick = async () => {
      try {
        await post(`/api/gesture/${g}`);
        toast(`Arm gesture: ${g}`);
        HandView.setGesture(g);
        if (window._arm3d) window._arm3d.setGesture(g);
        $('arm-gesture-name').textContent = g.toUpperCase();
      } catch (e) { toast(e.message, 'red'); }
    };
    c.appendChild(b);
  });

  // also add 3D-only poses (wave, lift)
  ['wave','lift'].forEach(g => {
    const b = document.createElement('button');
    b.className = 'btn'; b.textContent = g;
    b.style.width = 'auto'; b.style.flex = '1'; b.style.borderColor = 'var(--accent)';
    b.onclick = () => {
      HandView.setGesture(g);
      if (window._arm3d) window._arm3d.setGesture(g);
      $('arm-gesture-name').textContent = g.toUpperCase();
      toast(`3D Pose: ${g}`);
    };
    c.appendChild(b);
  });

  // add block programs as runnable scripts
  if (S.blockPrograms && S.blockPrograms.length) {
    const sep = document.createElement('div');
    sep.style.cssText = 'width:100%;font-size:9px;color:var(--text-dim);margin-top:4px;text-transform:uppercase;letter-spacing:.5px';
    sep.textContent = '— Block Scripts —';
    c.appendChild(sep);

    S.blockPrograms.forEach(prog => {
      const b = document.createElement('button');
      b.className = 'btn'; b.textContent = '▶ ' + prog.name;
      b.style.cssText = 'width:auto;flex:1;border-color:var(--yellow);color:var(--yellow);font-size:10px';
      b.onclick = () => {
        if (S.executingProgram) { toast('Already running', 'red'); return; }
        S.executingProgram = true;
        S.executionAbort = false;
        const startNode = Object.values(prog.nodes || {}).find(n => n.type === 'start');
        if (!startNode) { toast('No Start block', 'red'); S.executingProgram = false; return; }
        toast(`Running: ${prog.name}`);
        followFlow(prog, startNode.id, 'flow_out')
          .then(() => { S.executingProgram = false; toast('Done: ' + prog.name); })
          .catch(e => { S.executingProgram = false; toast('Error: ' + e.message, 'red'); });
      };
      c.appendChild(b);
    });
  }
}


// =============================================================================
// RMS BARS (per-channel signal strength indicator)
// =============================================================================

function buildRmsBars() {
  const c = $('rms-bars');
  c.innerHTML = '';
  const cols = chColors();
  for (let i = 0; i < N_CH; i++) {
    const label = S.channelLabels[i] || `CH${i + 1}`;
    c.innerHTML += `<div class="rms-row ${isChannelVisible(i) ? '' : 'row-hidden'}" id="rrow-${i}">
      <span class="rms-label" title="${label}">${label.slice(0, 6)}</span>
      <div class="rms-bar-bg"><div class="rms-bar" id="rb-${i}" style="background:${cols[i]}"></div></div>
      <span class="rms-val" id="rv-${i}">0.000</span>
    </div>`;
  }
}

function updateRmsBars(rms) {
  rms.forEach((v,i) => {
    const bar = $(`rb-${i}`);
    const val = $(`rv-${i}`);
    const row = $(`rrow-${i}`);
    if (row) row.classList.toggle('row-hidden', !isChannelVisible(i));
    if (bar) bar.style.width = `${isChannelVisible(i) ? Math.min(v / Math.max(S.signalMetricScale, 1e-6) * 100, 100) : 0}%`;
    if (val) val.textContent = isChannelVisible(i) ? v.toFixed(4) : 'OFF';
  });
}


// =============================================================================
// CHANNEL LEGEND (the little colored dots at the bottom of the waveform)
// =============================================================================

function buildLegend() {
  const c = $('channel-legend');
  c.innerHTML = '';
  const cols = chColors();
  for (let i = 0; i < N_CH; i++) {
    const label = S.channelLabels[i] || `CH${i + 1}`;
    const row = document.createElement('label');
    row.className = `ch-label ${isChannelVisible(i) ? '' : 'off'}`;
    row.title = `${label} (${isChannelVisible(i) ? 'visible' : 'hidden'})`;

    const toggle = document.createElement('input');
    toggle.type = 'checkbox';
    toggle.className = 'ch-toggle';
    toggle.checked = isChannelVisible(i);
    toggle.onchange = () => toggleChannelVisibility(i);

    const dot = document.createElement('span');
    dot.className = 'ch-dot';
    dot.style.background = cols[i];

    const text = document.createElement('span');
    text.className = 'ch-name';
    text.textContent = label;

    row.appendChild(toggle);
    row.appendChild(dot);
    row.appendChild(text);
    c.appendChild(row);
  }
}


// =============================================================================
// MANUAL SERVO SLIDERS
//
// each slider sends a move command when you release it.
// it also updates the 3d arm in real time as you drag.
// =============================================================================

function buildServos() {
  const c = $('servo-sliders');
  const names = ['Shoulder Rot','Shoulder Pitch','Elbow','Wrist Pitch','Wrist Roll',
                 'Thumb','Index','Middle','Ring','Pinky','Forearm Rot','Grip'];
  const count = 12;
  for (let i = 0; i < count; i++) {
    const row = document.createElement('div');
    row.className = 'servo-row';
    row.innerHTML = `<label title="${names[i]}">${names[i].substring(0,3)}</label>
      <input type="range" min="0" max="180" value="90" id="sv-${i}"/>
      <span class="angle-val" id="sv-v-${i}">90</span>`;
    c.appendChild(row);

    const slider = row.querySelector('input');
    const label = row.querySelector('.angle-val');

    slider.oninput = () => {
      label.textContent = slider.value;
      if (window._arm3d) window._arm3d.setJoint(i, parseInt(slider.value));
    };

    // send move for hardware joints 0-7
    slider.onchange = async () => {
      if (i < 8) {
        try { await post('/api/move', {joint_id:i, angle:parseInt(slider.value)}); }
        catch (e) { toast(e.message, 'red'); }
      }
    };
  }
}


// =============================================================================
// PIPELINE DIAGRAM (collapsible)
// =============================================================================

window.togglePipeline = function() {
  const sec = $('pipeline-section');
  const tog = $('pipeline-toggle');
  sec.classList.toggle('open');
  tog.textContent = sec.classList.contains('open')
    ? 'PIPELINE DIAGRAM [click to collapse]'
    : 'PIPELINE DIAGRAM [click to expand]';
};


// =============================================================================
// EMG CANVAS RENDERER
//
// draws 8 channels of rolling EMG waveforms onto a single canvas.
// each channel gets its own horizontal strip.
// the ring buffer wraps around so we read starting from emgHead.
// =============================================================================

function resizeCanvas() {
  const c = $('canvas-container');
  canvas.width = c.clientWidth;
  canvas.height = c.clientHeight;

  // also tell the 3d arm to resize if it exists
  if (window._arm3d) window._arm3d.resize();
}

function renderLoop() {
  drawEMG();

  // fps counter - update once per second
  S.frames++;
  const now = performance.now();
  if (now - S.fpsTime >= 1000) {
    $('fps-label').textContent = `${Math.round(S.frames * 1000 / (now - S.fpsTime))} fps`;
    S.frames = 0;
    S.fpsTime = now;
  }

  requestAnimationFrame(renderLoop);
}

function drawEMG() {
  const W = canvas.width;
  const H = canvas.height;
  if (!W || !H) return;

  const t = THEME_CANVAS[currentTheme] || THEME_CANVAS.dark;
  const cols = chColors();
  const retro = currentTheme === 'retro';
  const reviewState = getReviewRenderState();

  // clear + fill background
  ctx.fillStyle = t.bg;
  ctx.fillRect(0, 0, W, H);

  const rowH = H / N_CH;
  const head = reviewState.head;

  // How many samples have actually been written to the buffer?
  // Don't draw the zero-filled portion — that causes square-wave artifacts.
  const filled = reviewState.filled;
  const { drawStart } = getReviewLayout(reviewState, W);

  // ── Adaptive mute: compare each channel's RMS to the BEST channel ──
  // Disconnected channels on Cyton still pick up crosstalk so a fixed
  // threshold doesn't work.  Instead, find the strongest channel and mute
  // anything that's less than 20 % of it.
  const MUTE_ABS_FLOOR = 0.5;   // µV — absolute minimum to even consider
  let maxRms = 0;
  for (let ch = 0; ch < N_CH; ch++) {
    if (!isChannelVisible(ch)) continue;
    const raw = S.rms ? S.rms[ch] : 0;
    S.rmsSmooth[ch] = S.rmsSmooth[ch] * 0.90 + raw * 0.10;
    if (S.rmsSmooth[ch] > maxRms) maxRms = S.rmsSmooth[ch];
  }
  const muteThresh = Math.max(maxRms * 0.20, S.muteFloor || MUTE_ABS_FLOOR);

  for (let ch = 0; ch < N_CH; ch++) {
    const buf = reviewState.emg[ch];
    const mid = rowH * (ch + 0.5);

    // subtle alternating row shading
    if (ch % 2 === 0) {
      ctx.fillStyle = t.alt;
      ctx.fillRect(0, ch * rowH, W, rowH);
    }

    // center line
    ctx.beginPath();
    ctx.strokeStyle = t.grid;
    ctx.lineWidth = 1;
    ctx.moveTo(0, mid); ctx.lineTo(W, mid);
    ctx.stroke();

    // ── Mute with hysteresis ──
    const sr = S.rmsSmooth[ch];
    if (S.chMuted[ch]) {
      if (sr > muteThresh * 1.3) S.chMuted[ch] = false;   // need 30 % above to turn on
    } else {
      if (sr < muteThresh * 0.7) S.chMuted[ch] = true;    // 30 % below to turn off
    }
    const hidden = !isChannelVisible(ch);
    const muted = hidden || S.chMuted[ch];

    // ── Fixed scale — NO auto-scale for noise ──
    // Use a fixed sensitivity that shows real EMG well.
    // Real EMG from Cyton is typically 50-500+ µV.  Noise is 1-20 µV.
    // A fixed scale of (rowH*0.4 / 200µV) means 200 µV fills 80 % of the row.
    // Strong contractions (>200 µV) get clamped at the row edge — that's fine.
    const FULL_SCALE_UV = S.signalFullScale || 200.0;
    const usable = rowH * 0.40;
    const scale = usable / FULL_SCALE_UV;

    // neon glow effect on retro theme (skip if muted)
    if (retro && !muted) { ctx.shadowColor = cols[ch]; ctx.shadowBlur = 6; }

    // the actual waveform
    ctx.beginPath();
    if (muted) {
      ctx.strokeStyle = t.grid;
      ctx.lineWidth = 0.6;
      ctx.globalAlpha = 0.25;
    } else {
      ctx.strokeStyle = cols[ch];
      ctx.lineWidth = retro ? 1.5 : 1.2;
    }
    ctx.lineJoin = 'round';

    // Only draw the portion of the buffer that has real data
    // Map: rightmost pixel = newest sample, scroll left
    for (let x = drawStart; x < W; x++) {
      const si = (head + Math.floor(x * DISPLAY_SAMPLES / W)) % DISPLAY_SAMPLES;
      const y = muted ? mid : mid - buf[si] * scale;
      const clamped = muted ? y : Math.max(ch * rowH + 2, Math.min((ch + 1) * rowH - 2, y));
      if (x === drawStart) ctx.moveTo(x, clamped);
      else ctx.lineTo(x, clamped);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1.0;

    // channel label
    ctx.fillStyle = muted ? t.grid : cols[ch];
    ctx.font = currentTheme === 'dark' ? '11px Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' : '7px "Press Start 2P", monospace';
    const baseLabel = S.channelLabels[ch] || `CH${ch+1}`;
    const label = hidden ? `${baseLabel} (hidden)` : (muted ? `${baseLabel} (off)` : baseLabel);
    ctx.fillText(label, 4, ch * rowH + 12);
  }

  if (filled > 0) {
    const drawMarkerLine = (x, label, color, dashed = false) => {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.85;
      if (dashed) ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
      ctx.setLineDash([]);
      if (label) {
        ctx.fillStyle = color;
        ctx.font = '10px Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.fillText(label, clamp(x + 5, 6, Math.max(6, W - 110)), 14);
      }
      ctx.restore();
    };

    S.review.markers.forEach(marker => {
      const range = marker.selection || null;
      if (range && Number.isFinite(range.start_sample) && Number.isFinite(range.end_sample)) {
        const start = Number(range.start_sample);
        const end = Number(range.end_sample);
        if (end < reviewState.baseAbs || start > reviewState.total) return;
        const x1 = canvasXFromSample(start, reviewState, W);
        const x2 = canvasXFromSample(end, reviewState, W);
        ctx.save();
        ctx.fillStyle = 'rgba(88,111,218,0.10)';
        ctx.fillRect(Math.min(x1, x2), 0, Math.max(2, Math.abs(x2 - x1)), H);
        ctx.restore();
        drawMarkerLine(x1, marker.event, 'rgba(88,111,218,0.82)');
        drawMarkerLine(x2, '', 'rgba(88,111,218,0.62)', true);
      } else if (Number.isFinite(marker.sampleIndex)) {
        const sampleIndex = Number(marker.sampleIndex);
        if (sampleIndex < reviewState.baseAbs || sampleIndex > reviewState.total) return;
        drawMarkerLine(canvasXFromSample(sampleIndex, reviewState, W), marker.event, 'rgba(79,143,105,0.84)');
      }
    });
  }

  if (reviewState.paused && S.review.selection) {
    const range = getSelectionRange(S.review.selection);
    if (range) {
      const x1 = canvasXFromSample(range.start, reviewState, W);
      const x2 = canvasXFromSample(range.end, reviewState, W);
      const left = Math.min(x1, x2);
      const width = Math.max(2, Math.abs(x2 - x1));
      ctx.save();
      ctx.fillStyle = 'rgba(88,111,218,0.12)';
      ctx.fillRect(left, 0, width, H);
      ctx.strokeStyle = 'rgba(88,111,218,0.74)';
      ctx.lineWidth = 1.3;
      ctx.strokeRect(left, 1, width, Math.max(0, H - 2));
      ctx.restore();
    }
  }
}


// =============================================================================
// 3D ROBOTIC ARM (digital twin)
//
// uses three.js to render an articulated hand:
//   - base platform + forearm cylinder
//   - wrist block (3 axes of rotation via J5/J6/J7)
//   - 5 fingers with 2 segments each that curl via J0-J4
//
// the arm smoothly interpolates to target angles so it feels organic,
// not jerky like the real servos. you can orbit/zoom with the mouse.
// =============================================================================

function init3DArm() {
  if (typeof THREE === 'undefined') { console.warn('three.js not loaded'); return; }

  const container = $('arm-container');
  const w = container.clientWidth || 400;
  const h = container.clientHeight || 300;

  // ── Scene ──────────────────────────────────────────────────────────────
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1117);
  scene.fog = new THREE.Fog(0x0d1117, 18, 35);

  const camera = new THREE.PerspectiveCamera(35, w/h, 0.1, 100);
  camera.position.set(8, 6, 10);

  const renderer = new THREE.WebGLRenderer({ antialias:true, alpha:false });
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  container.appendChild(renderer.domElement);

  // make sure canvas receives pointer events
  renderer.domElement.style.touchAction = 'none';

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.enablePan = true;
  controls.enableZoom = true;
  controls.enableRotate = true;
  controls.target.set(0, 3, 0);
  controls.minDistance = 2;
  controls.maxDistance = 30;
  controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
  renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());

  // ── Lights ─────────────────────────────────────────────────────────────
  scene.add(new THREE.AmbientLight(0x404060, 0.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.0);
  sun.position.set(6, 10, 8); sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.camera.near = 1; sun.shadow.camera.far = 25;
  sun.shadow.camera.left = -6; sun.shadow.camera.right = 6;
  sun.shadow.camera.top = 6; sun.shadow.camera.bottom = -6;
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0x58a6ff, 0.4);
  fill.position.set(-4, 5, -3); scene.add(fill);
  const rim = new THREE.PointLight(0xff7b72, 0.3, 15);
  rim.position.set(-3, 4, 5); scene.add(rim);

  // ── Ground ─────────────────────────────────────────────────────────────
  const groundGeo = new THREE.PlaneGeometry(20, 20);
  const groundMat = new THREE.MeshStandardMaterial({color:0x1a1a2e, roughness:0.9});
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI/2; ground.receiveShadow = true;
  scene.add(ground);
  scene.add(new THREE.GridHelper(12, 24, 0x30363d, 0x1a1a2e));

  // ── Shared geometry helpers ────────────────────────────────────────────
  function cyl(rTop, rBot, h, segs, mat) {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(rTop, rBot, h, segs), mat);
    m.castShadow = true; return m;
  }
  function box(w2, h2, d, mat) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w2, h2, d), mat);
    m.castShadow = true; return m;
  }
  function sphere(r, segs, mat) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(r, segs, segs), mat);
    m.castShadow = true; return m;
  }

  // ── Arm container — holds all model geometry ──────────────────────────
  // Pivots are shared across all models for animation
  const armRoot = new THREE.Group();
  scene.add(armRoot);

  // Pivot hierarchy (always the same, geometry fills these)
  const baseGroup = new THREE.Group();         // J0  shoulder rotate (yaw)
  armRoot.add(baseGroup);
  const shoulderPivot = new THREE.Group();     // J1  shoulder pitch housing
  shoulderPivot.position.y = 0.3;
  baseGroup.add(shoulderPivot);
  const upperArmPivot = new THREE.Group();     // J1  upper arm pitch
  upperArmPivot.position.y = 0.85;
  shoulderPivot.add(upperArmPivot);
  const forearmPivot = new THREE.Group();      // J2  elbow BEND  (up/down hinge)
  forearmPivot.position.y = 2.05;
  upperArmPivot.add(forearmPivot);
  const forearmRotPivot = new THREE.Group();   // J10 forearm SPIN disc
  forearmRotPivot.position.y = 0;             // sits right at elbow, geometry offsets itself
  forearmPivot.add(forearmRotPivot);
  const wristPivot = new THREE.Group();        // J3/J4 wrist
  wristPivot.position.y = 1.85;
  forearmRotPivot.add(wristPivot);

  // finger pivots (only used by models that have fingers)
  let fingers = [];

  // Track all model meshes so we can clear them on model switch
  let modelMeshes = [];

  // ── Animation state ────────────────────────────────────────────────────
  const NUM_JOINTS = 12;
  const target = new Float32Array(NUM_JOINTS).fill(90);
  const smooth = new Float32Array(NUM_JOINTS).fill(90);

  const POSES = {
    rest:  [90,90,90,90,90, 90,90,90,90,90, 90,90],
    open:  [90,90,90,90,90, 30,30,30,30,30, 90,30],
    close: [90,90,90,90,90, 150,150,150,150,150, 90,150],
    pinch: [90,90,90,90,90, 150,150,30,30,30, 90,90],
    point: [90,90,90,90,90, 30,150,150,150,150, 90,90],
    wave:  [90,60,90,90,90, 30,30,30,30,30, 90,30],
    lift:  [90,45,60,90,90, 90,90,90,90,90, 90,90],
  };

  // ── Model builders ─────────────────────────────────────────────────────
  // Each returns an array of meshes added, and sets up finger pivots

  function clearModel() {
    // Remove all model meshes from their parents
    modelMeshes.forEach(m => { if (m.parent) m.parent.remove(m); });
    modelMeshes = [];
    // Clear finger pivot children
    fingers.forEach(f => {
      if (f.root && f.root.parent) f.root.parent.remove(f.root);
    });
    fingers = [];
    // Clear all children of pivots (but not sub-pivots)
    const keepGroups = new Set([shoulderPivot, upperArmPivot, forearmPivot, forearmRotPivot, wristPivot]);
    [baseGroup, shoulderPivot, upperArmPivot, forearmPivot, forearmRotPivot, wristPivot].forEach(g => {
      const toRemove = [];
      g.children.forEach(c => { if (!keepGroups.has(c)) toRemove.push(c); });
      toRemove.forEach(c => g.remove(c));
    });
  }

  // ── MODEL 1: Industrial 6-Axis Robot (like KUKA/ABB) ──────────────────
  function buildIndustrial() {
    const bodyMat = new THREE.MeshStandardMaterial({color:0xcc4400, roughness:0.3, metalness:0.7});
    const darkMat = new THREE.MeshStandardMaterial({color:0x2a2a2a, roughness:0.2, metalness:0.9});
    const accentM = new THREE.MeshStandardMaterial({color:0xff6600, roughness:0.4, metalness:0.5});
    const warnMat = new THREE.MeshStandardMaterial({color:0xffcc00, roughness:0.5, metalness:0.3});

    // Heavy round base
    const base1 = cyl(2.0, 2.2, 0.4, 24, darkMat); base1.position.y = 0.2;
    baseGroup.add(base1); modelMeshes.push(base1);
    const base2 = cyl(1.6, 1.8, 0.3, 24, bodyMat); base2.position.y = 0.45;
    baseGroup.add(base2); modelMeshes.push(base2);
    // Warning stripes ring
    const warnRing = cyl(2.05, 2.05, 0.06, 24, warnMat); warnRing.position.y = 0.42;
    baseGroup.add(warnRing); modelMeshes.push(warnRing);

    // Shoulder tower
    shoulderPivot.position.y = 0.6;
    const tower = box(1.0, 1.4, 0.8, bodyMat); tower.position.y = 0.7;
    shoulderPivot.add(tower); modelMeshes.push(tower);
    const towerCap = box(1.1, 0.1, 0.9, darkMat); towerCap.position.y = 1.42;
    shoulderPivot.add(towerCap); modelMeshes.push(towerCap);
    // Shoulder joint disc
    const sDisk = cyl(0.45, 0.45, 0.2, 16, darkMat); sDisk.position.y = 1.5;
    sDisk.rotation.z = Math.PI/2;
    shoulderPivot.add(sDisk); modelMeshes.push(sDisk);

    // Upper arm — thick industrial beam
    upperArmPivot.position.y = 1.5;
    const ua1 = box(0.6, 2.2, 0.5, bodyMat); ua1.position.y = 1.1;
    upperArmPivot.add(ua1); modelMeshes.push(ua1);
    // Hydraulic pistons
    const piston1 = cyl(0.08, 0.08, 1.8, 6, darkMat);
    piston1.position.set(0.35, 0.9, 0.15); upperArmPivot.add(piston1); modelMeshes.push(piston1);
    const piston2 = cyl(0.08, 0.08, 1.8, 6, darkMat);
    piston2.position.set(-0.35, 0.9, 0.15); upperArmPivot.add(piston2); modelMeshes.push(piston2);
    // Piston cylinder sleeves
    const ps1 = cyl(0.12, 0.12, 0.5, 8, accentM);
    ps1.position.set(0.35, 0.2, 0.15); upperArmPivot.add(ps1); modelMeshes.push(ps1);
    const ps2 = cyl(0.12, 0.12, 0.5, 8, accentM);
    ps2.position.set(-0.35, 0.2, 0.15); upperArmPivot.add(ps2); modelMeshes.push(ps2);
    // Brand plate
    const brand = box(0.4, 0.15, 0.02, warnMat);
    brand.position.set(0, 1.5, 0.27); upperArmPivot.add(brand); modelMeshes.push(brand);

    // ── ELBOW HINGE (J2 = bend up/down) ──────────────────────────────────
    forearmPivot.position.y = 2.2;
    // Hinge pin (left-right axis — bending axis)
    const iHingePin = cyl(0.14, 0.14, 1.0, 8, warnMat); iHingePin.rotation.z = Math.PI/2;
    forearmPivot.add(iHingePin); modelMeshes.push(iHingePin);
    // Bracket blocks
    const iBrL = box(0.2, 0.55, 0.22, bodyMat); iBrL.position.set(-0.48, 0, 0); forearmPivot.add(iBrL); modelMeshes.push(iBrL);
    const iBrR = box(0.2, 0.55, 0.22, bodyMat); iBrR.position.set(0.48, 0, 0); forearmPivot.add(iBrR); modelMeshes.push(iBrR);
    const iCL = cyl(0.18, 0.18, 0.06, 8, darkMat); iCL.rotation.z = Math.PI/2; iCL.position.x = -0.6; forearmPivot.add(iCL); modelMeshes.push(iCL);
    const iCR = cyl(0.18, 0.18, 0.06, 8, darkMat); iCR.rotation.z = Math.PI/2; iCR.position.x = 0.6; forearmPivot.add(iCR); modelMeshes.push(iCR);

    // ── SPIN DISC (J10 = forearm twist) ───────────────────────────────────
    const iSpinDisc = cyl(0.48, 0.48, 0.32, 16, darkMat); iSpinDisc.position.y = 0.22; forearmRotPivot.add(iSpinDisc); modelMeshes.push(iSpinDisc);
    const iSpinRing = cyl(0.51, 0.51, 0.05, 16, accentM); iSpinRing.position.y = 0.35; forearmRotPivot.add(iSpinRing); modelMeshes.push(iSpinRing);
    const iCr1 = box(0.75, 0.06, 0.06, warnMat); iCr1.position.y = 0.4; forearmRotPivot.add(iCr1); modelMeshes.push(iCr1);
    const iCr2 = box(0.06, 0.06, 0.75, warnMat); iCr2.position.y = 0.4; forearmRotPivot.add(iCr2); modelMeshes.push(iCr2);

    // Forearm — into forearmRotPivot so it spins with J10
    const fa = box(0.45, 1.8, 0.4, bodyMat); fa.position.y = 1.35; forearmRotPivot.add(fa); modelMeshes.push(fa);
    const faStripe = box(0.47, 0.08, 0.42, warnMat); faStripe.position.y = 0.85; forearmRotPivot.add(faStripe); modelMeshes.push(faStripe);
    const cable = cyl(0.05, 0.05, 1.6, 6, darkMat); cable.position.set(0.28, 1.35, 0); forearmRotPivot.add(cable); modelMeshes.push(cable);

    // Wrist
    wristPivot.position.y = 2.3;
    const w1 = cyl(0.35, 0.3, 0.25, 12, darkMat); wristPivot.add(w1); modelMeshes.push(w1);
    const w2 = cyl(0.28, 0.28, 0.15, 12, accentM); w2.position.y = 0.18; wristPivot.add(w2); modelMeshes.push(w2);
    const grip1 = box(0.08, 0.5, 0.3, darkMat); grip1.position.set(-0.25, 0.5, 0); wristPivot.add(grip1); modelMeshes.push(grip1);
    const grip2 = box(0.08, 0.5, 0.3, darkMat); grip2.position.set(0.25, 0.5, 0); wristPivot.add(grip2); modelMeshes.push(grip2);
    const gripBar = box(0.6, 0.1, 0.25, bodyMat); gripBar.position.y = 0.3; wristPivot.add(gripBar); modelMeshes.push(gripBar);
    const pad1 = box(0.06, 0.2, 0.28, accentM); pad1.position.set(-0.22, 0.65, 0); wristPivot.add(pad1); modelMeshes.push(pad1);
    const pad2 = box(0.06, 0.2, 0.28, accentM); pad2.position.set(0.22, 0.65, 0); wristPivot.add(pad2); modelMeshes.push(pad2);

    const jawL = new THREE.Group(); jawL.position.set(-0.25, 0.28, 0); wristPivot.add(jawL);
    const jawR = new THREE.Group(); jawR.position.set(0.25, 0.28, 0); wristPivot.add(jawR);
    fingers = [
      {meta:jawL, prox:jawL, dist:jawL, thumb:false, root:jawL},
      {meta:jawR, prox:jawR, dist:jawR, thumb:false, root:jawR},
    ];
    for (let i = 2; i < 5; i++) {
      const noop = new THREE.Group();
      fingers.push({meta:noop, prox:noop, dist:noop, thumb:false, root:noop});
    }
  }

  // ── MODEL 2: Humanoid Hand ────────────────────────────────────────────
  function buildHumanoid() {
    const metalDark = new THREE.MeshStandardMaterial({color:0x2a2a3a, roughness:0.3, metalness:0.85});
    const metalMid  = new THREE.MeshStandardMaterial({color:0x4a4a5a, roughness:0.35, metalness:0.8});
    const metalLight= new THREE.MeshStandardMaterial({color:0x6a6a7a, roughness:0.4, metalness:0.75});
    const accentM   = new THREE.MeshStandardMaterial({color:0x58a6ff, roughness:0.4, metalness:0.6});
    const jointM    = new THREE.MeshStandardMaterial({color:0x333340, roughness:0.2, metalness:0.9});
    const thumbM    = new THREE.MeshStandardMaterial({color:0xff7b72, roughness:0.4, metalness:0.6});
    const fingerM   = new THREE.MeshStandardMaterial({color:0x58a6ff, roughness:0.4, metalness:0.6});
    const tipM      = new THREE.MeshStandardMaterial({color:0x79c0ff, roughness:0.5, metalness:0.5});

    shoulderPivot.position.y = 0.3;
    const basePlate = cyl(1.8, 2.0, 0.25, 24, metalDark); basePlate.position.y = 0.125;
    baseGroup.add(basePlate); modelMeshes.push(basePlate);
    const baseRing = cyl(1.85, 1.85, 0.08, 24, accentM); baseRing.position.y = 0.26;
    baseGroup.add(baseRing); modelMeshes.push(baseRing);
    for (let a = 0; a < Math.PI*2; a += Math.PI/4) {
      const bolt = cyl(0.08, 0.08, 0.06, 6, jointM);
      bolt.position.set(Math.cos(a)*1.6, 0.28, Math.sin(a)*1.6);
      baseGroup.add(bolt); modelMeshes.push(bolt);
    }
    const sHouse = box(1.2, 0.8, 1.2, metalMid); sHouse.position.y = 0.4;
    shoulderPivot.add(sHouse); modelMeshes.push(sHouse);
    const sBall = sphere(0.4, 12, jointM); sBall.position.y = 0.85;
    shoulderPivot.add(sBall); modelMeshes.push(sBall);
    upperArmPivot.position.y = 0.85;
    const ua = cyl(0.35, 0.3, 2.0, 10, metalLight); ua.position.y = 1.0;
    upperArmPivot.add(ua); modelMeshes.push(ua);
    const h1 = cyl(0.06, 0.06, 1.6, 6, accentM); h1.position.set(0.25, 1.0, 0.15); upperArmPivot.add(h1); modelMeshes.push(h1);
    const h2 = cyl(0.06, 0.06, 1.6, 6, accentM); h2.position.set(-0.25, 1.0, 0.15); upperArmPivot.add(h2); modelMeshes.push(h2);

    // ── ELBOW HINGE (J2 = bend up/down) — stays in forearmPivot ──────────
    forearmPivot.position.y = 2.05;
    // Hinge pin goes left-right (X axis) — this is what bends
    const hingePin = cyl(0.12, 0.12, 0.9, 8, accentM); hingePin.rotation.z = Math.PI/2;
    forearmPivot.add(hingePin); modelMeshes.push(hingePin);
    // Hinge bracket left
    const hBrL = box(0.18, 0.45, 0.25, metalMid); hBrL.position.set(-0.4, 0, 0);
    forearmPivot.add(hBrL); modelMeshes.push(hBrL);
    // Hinge bracket right
    const hBrR = box(0.18, 0.45, 0.25, metalMid); hBrR.position.set(0.4, 0, 0);
    forearmPivot.add(hBrR); modelMeshes.push(hBrR);
    // Pin caps (visible bolts)
    const capL = cyl(0.15, 0.15, 0.06, 8, jointM); capL.rotation.z = Math.PI/2; capL.position.x = -0.48;
    forearmPivot.add(capL); modelMeshes.push(capL);
    const capR = cyl(0.15, 0.15, 0.06, 8, jointM); capR.rotation.z = Math.PI/2; capR.position.x = 0.48;
    forearmPivot.add(capR); modelMeshes.push(capR);

    // ── SPIN DISC (J10 = forearm twist) — in forearmRotPivot ─────────────
    // Disc spins around Y axis (the arm's length axis)
    const spinDisc = cyl(0.38, 0.38, 0.28, 16, jointM); spinDisc.position.y = 0.22;
    forearmRotPivot.add(spinDisc); modelMeshes.push(spinDisc);
    const spinRing = cyl(0.41, 0.41, 0.05, 16, accentM); spinRing.position.y = 0.33;
    forearmRotPivot.add(spinRing); modelMeshes.push(spinRing);
    // Cross pattern on disc face (shows it's spinning)
    const cr1 = box(0.6, 0.05, 0.05, accentM); cr1.position.y = 0.38; forearmRotPivot.add(cr1); modelMeshes.push(cr1);
    const cr2 = box(0.05, 0.05, 0.6, accentM); cr2.position.y = 0.38; forearmRotPivot.add(cr2); modelMeshes.push(cr2);

    // Forearm — goes up from spin disc
    const fa = cyl(0.28, 0.22, 1.8, 10, metalMid); fa.position.y = 1.3;
    forearmRotPivot.add(fa); modelMeshes.push(fa);

    // Wrist
    wristPivot.position.y = 2.28;
    const wb = sphere(0.2, 10, jointM); wristPivot.add(wb); modelMeshes.push(wb);
    const palm = box(1.8, 0.3, 1.0, metalLight); palm.position.y = 0.25; wristPivot.add(palm); modelMeshes.push(palm);
    const palmTop = box(1.6, 0.08, 0.85, accentM); palmTop.position.y = 0.42; wristPivot.add(palmTop); modelMeshes.push(palmTop);

    const FING = [
      {x:-0.85, z:0.35, l:0.65, mat:thumbM, thumb:true},
      {x:-0.42, z:0.55, l:0.75, mat:fingerM, thumb:false},
      {x: 0.0,  z:0.58, l:0.80, mat:fingerM, thumb:false},
      {x: 0.42, z:0.55, l:0.72, mat:fingerM, thumb:false},
      {x: 0.78, z:0.48, l:0.55, mat:fingerM, thumb:false},
    ];
    FING.forEach(f => {
      const root = new THREE.Group(); root.position.set(f.x, 0.4, f.z * 0.5);
      if (f.thumb) root.rotation.z = 0.3;
      wristPivot.add(root);
      const meta = new THREE.Group(); root.add(meta);
      const mm = box(0.22, f.l*0.3, 0.22, f.mat); mm.position.y = f.l*0.15; meta.add(mm); modelMeshes.push(mm);
      const pj = sphere(0.09, 8, jointM); pj.position.y = f.l*0.3; meta.add(pj); modelMeshes.push(pj);
      const prox = new THREE.Group(); prox.position.y = f.l*0.3; meta.add(prox);
      const pm = box(0.19, f.l*0.35, 0.19, f.mat); pm.position.y = f.l*0.175; prox.add(pm); modelMeshes.push(pm);
      const dj = sphere(0.07, 8, jointM); dj.position.y = f.l*0.35; prox.add(dj); modelMeshes.push(dj);
      const dist = new THREE.Group(); dist.position.y = f.l*0.35; prox.add(dist);
      const dm = box(0.16, f.l*0.28, 0.16, f.mat); dm.position.y = f.l*0.14; dist.add(dm); modelMeshes.push(dm);
      const tip = sphere(0.09, 8, tipM); tip.position.y = f.l*0.28; dist.add(tip); modelMeshes.push(tip);
      fingers.push({meta, prox, dist, thumb:f.thumb, root});
    });
  }

  // ── MODEL 3: Mech Warrior ─────────────────────────────────────────────
  function buildMech() {
    const armorMat = new THREE.MeshStandardMaterial({color:0x3a4a3a, roughness:0.25, metalness:0.85, emissive:new THREE.Color(0x001100), emissiveIntensity:0.05});
    const frameMat = new THREE.MeshStandardMaterial({color:0x1a2a1a, roughness:0.2, metalness:0.95});
    const glowMat  = new THREE.MeshStandardMaterial({color:0xff2200, roughness:0.3, metalness:0.5, emissive:new THREE.Color(0xff0000), emissiveIntensity:0.4});
    const panelMat = new THREE.MeshStandardMaterial({color:0x4a5a4a, roughness:0.3, metalness:0.8});

    shoulderPivot.position.y = 0.5;
    const base1 = box(3.0, 0.5, 3.0, frameMat); base1.position.y = 0.25; baseGroup.add(base1); modelMeshes.push(base1);
    const base2 = box(2.5, 0.3, 2.5, armorMat); base2.position.y = 0.55; baseGroup.add(base2); modelMeshes.push(base2);
    for (let i = 0; i < 4; i++) {
      const vent = box(0.6, 0.04, 0.08, glowMat); vent.position.set((i-1.5)*0.8, 0.72, 1.3);
      baseGroup.add(vent); modelMeshes.push(vent);
    }
    const sBlock = box(1.6, 1.2, 1.4, armorMat); sBlock.position.y = 0.6; shoulderPivot.add(sBlock); modelMeshes.push(sBlock);
    const sGlow = box(0.3, 0.8, 0.02, glowMat); sGlow.position.set(0, 0.6, 0.72); shoulderPivot.add(sGlow); modelMeshes.push(sGlow);
    const sPlate = box(1.8, 0.15, 1.0, panelMat); sPlate.position.set(0, 1.25, 0); shoulderPivot.add(sPlate); modelMeshes.push(sPlate);
    upperArmPivot.position.y = 1.3;
    const ua = box(0.8, 2.4, 0.7, armorMat); ua.position.y = 1.2; upperArmPivot.add(ua); modelMeshes.push(ua);
    const hp1 = cyl(0.1, 0.1, 2.0, 6, frameMat); hp1.position.set(0.5, 1.0, 0.2); upperArmPivot.add(hp1); modelMeshes.push(hp1);
    const hp2 = cyl(0.1, 0.1, 2.0, 6, frameMat); hp2.position.set(-0.5, 1.0, 0.2); upperArmPivot.add(hp2); modelMeshes.push(hp2);
    const uaGlow = box(0.06, 1.8, 0.06, glowMat); uaGlow.position.set(0, 1.2, 0.38); upperArmPivot.add(uaGlow); modelMeshes.push(uaGlow);

    // ── ELBOW HINGE (J2) — heavy armored hinge ────────────────────────────
    forearmPivot.position.y = 2.4;
    const mHingePin = cyl(0.18, 0.18, 1.1, 8, glowMat); mHingePin.rotation.z = Math.PI/2;
    forearmPivot.add(mHingePin); modelMeshes.push(mHingePin);
    const mHBrL = box(0.22, 0.6, 0.3, armorMat); mHBrL.position.set(-0.52, 0, 0); forearmPivot.add(mHBrL); modelMeshes.push(mHBrL);
    const mHBrR = box(0.22, 0.6, 0.3, armorMat); mHBrR.position.set(0.52, 0, 0); forearmPivot.add(mHBrR); modelMeshes.push(mHBrR);
    const mCapL = cyl(0.2, 0.2, 0.06, 8, frameMat); mCapL.rotation.z = Math.PI/2; mCapL.position.x = -0.62; forearmPivot.add(mCapL); modelMeshes.push(mCapL);
    const mCapR = cyl(0.2, 0.2, 0.06, 8, frameMat); mCapR.rotation.z = Math.PI/2; mCapR.position.x = 0.62; forearmPivot.add(mCapR); modelMeshes.push(mCapR);

    // ── SPIN DISC (J10) — glowing rotary actuator ─────────────────────────
    const mSpinDisc = cyl(0.5, 0.5, 0.35, 12, frameMat); mSpinDisc.position.y = 0.25; forearmRotPivot.add(mSpinDisc); modelMeshes.push(mSpinDisc);
    const mSpinGlow = cyl(0.53, 0.53, 0.06, 12, glowMat); mSpinGlow.position.y = 0.38; forearmRotPivot.add(mSpinGlow); modelMeshes.push(mSpinGlow);
    const mCr1 = box(0.8, 0.06, 0.06, glowMat); mCr1.position.y = 0.45; forearmRotPivot.add(mCr1); modelMeshes.push(mCr1);
    const mCr2 = box(0.06, 0.06, 0.8, glowMat); mCr2.position.y = 0.45; forearmRotPivot.add(mCr2); modelMeshes.push(mCr2);

    // Forearm
    const fa = box(0.65, 2.0, 0.6, armorMat); fa.position.y = 1.45; forearmRotPivot.add(fa); modelMeshes.push(fa);
    const rail = box(0.15, 1.6, 0.15, frameMat); rail.position.set(0.45, 1.55, 0); forearmRotPivot.add(rail); modelMeshes.push(rail);
    const railTip = cyl(0.04, 0.08, 0.3, 6, glowMat); railTip.position.set(0.45, 2.45, 0); forearmRotPivot.add(railTip); modelMeshes.push(railTip);

    wristPivot.position.y = 2.5;
    const wj = cyl(0.35, 0.3, 0.25, 10, frameMat); wristPivot.add(wj); modelMeshes.push(wj);
    const prongs = [-0.3, 0, 0.3];
    prongs.forEach((x, i) => {
      const root = new THREE.Group(); root.position.set(x, 0.15, 0); wristPivot.add(root);
      const meta = new THREE.Group(); root.add(meta);
      const seg1 = box(0.12, 0.5, 0.15, armorMat); seg1.position.y = 0.25; meta.add(seg1); modelMeshes.push(seg1);
      const prox = new THREE.Group(); prox.position.y = 0.5; meta.add(prox);
      const seg2 = box(0.1, 0.4, 0.12, panelMat); seg2.position.y = 0.2; prox.add(seg2); modelMeshes.push(seg2);
      const dist = new THREE.Group(); dist.position.y = 0.4; prox.add(dist);
      const tip = cyl(0.02, 0.06, 0.2, 6, glowMat); tip.position.y = 0.1; dist.add(tip); modelMeshes.push(tip);
      fingers.push({meta, prox, dist, thumb: i === 0, root});
    });
    for (let i = 3; i < 5; i++) {
      const noop = new THREE.Group();
      fingers.push({meta:noop, prox:noop, dist:noop, thumb:false, root:noop});
    }
  }

  // ── MODEL 4: Minimal Wire ─────────────────────────────────────────────
  function buildMinimal() {
    const nodeMat = new THREE.MeshStandardMaterial({color:0x00ffff, roughness:0.3, metalness:0.5, emissive:new THREE.Color(0x00ffff), emissiveIntensity:0.3});
    const linkMat = new THREE.MeshStandardMaterial({color:0xcccccc, roughness:0.4, metalness:0.4, wireframe:true});

    shoulderPivot.position.y = 0.2;
    const ring = cyl(1.5, 1.5, 0.05, 32, nodeMat); ring.position.y = 0.05; baseGroup.add(ring); modelMeshes.push(ring);
    const post = cyl(0.08, 0.08, 0.2, 8, linkMat); post.position.y = 0.15; baseGroup.add(post); modelMeshes.push(post);
    const sNode = sphere(0.25, 12, nodeMat); sNode.position.y = 0.4; shoulderPivot.add(sNode); modelMeshes.push(sNode);
    upperArmPivot.position.y = 0.5;
    const ua = cyl(0.06, 0.06, 2.0, 8, linkMat); ua.position.y = 1.0; upperArmPivot.add(ua); modelMeshes.push(ua);

    // ── ELBOW HINGE (J2) — glowing pin node ──────────────────────────────
    forearmPivot.position.y = 2.0;
    const ePin = cyl(0.06, 0.06, 0.7, 8, nodeMat); ePin.rotation.z = Math.PI/2; forearmPivot.add(ePin); modelMeshes.push(ePin);
    const ePinL = sphere(0.09, 8, nodeMat); ePinL.position.x = -0.35; forearmPivot.add(ePinL); modelMeshes.push(ePinL);
    const ePinR = sphere(0.09, 8, nodeMat); ePinR.position.x = 0.35; forearmPivot.add(ePinR); modelMeshes.push(ePinR);

    // ── SPIN DISC (J10) — glowing ring node ──────────────────────────────
    const eDisc = cyl(0.3, 0.3, 0.06, 16, nodeMat); eDisc.position.y = 0.18; forearmRotPivot.add(eDisc); modelMeshes.push(eDisc);
    const eCr1 = box(0.5, 0.04, 0.04, nodeMat); eCr1.position.y = 0.22; forearmRotPivot.add(eCr1); modelMeshes.push(eCr1);
    const eCr2 = box(0.04, 0.04, 0.5, nodeMat); eCr2.position.y = 0.22; forearmRotPivot.add(eCr2); modelMeshes.push(eCr2);

    const fa = cyl(0.05, 0.05, 1.6, 8, linkMat); fa.position.y = 1.0; forearmRotPivot.add(fa); modelMeshes.push(fa);
    wristPivot.position.y = 1.9;
    const wNode = sphere(0.15, 12, nodeMat); wristPivot.add(wNode); modelMeshes.push(wNode);
    const FING = [
      {x:-0.4, z:0.2, l:0.5},{x:-0.2, z:0.3, l:0.6},{x: 0.0, z:0.32,l:0.65},{x: 0.2, z:0.3, l:0.55},{x: 0.4, z:0.2, l:0.4},
    ];
    FING.forEach((f, i) => {
      const root = new THREE.Group(); root.position.set(f.x, 0.1, f.z * 0.5); wristPivot.add(root);
      const meta = new THREE.Group(); root.add(meta);
      const s1 = cyl(0.03, 0.03, f.l*0.4, 6, linkMat); s1.position.y = f.l*0.2; meta.add(s1); modelMeshes.push(s1);
      const n1 = sphere(0.05, 8, nodeMat); n1.position.y = f.l*0.4; meta.add(n1); modelMeshes.push(n1);
      const prox = new THREE.Group(); prox.position.y = f.l*0.4; meta.add(prox);
      const s2 = cyl(0.025, 0.025, f.l*0.35, 6, linkMat); s2.position.y = f.l*0.175; prox.add(s2); modelMeshes.push(s2);
      const dist = new THREE.Group(); dist.position.y = f.l*0.35; prox.add(dist);
      const tip = sphere(0.04, 8, nodeMat); tip.position.y = f.l*0.15; dist.add(tip); modelMeshes.push(tip);
      fingers.push({meta, prox, dist, thumb: i === 0, root});
    });
  }

  // ── MODEL 5: Bionic — sleek prosthetic-style ──────────────────────────
  function buildBionic() {
    const shellMat = new THREE.MeshStandardMaterial({color:0xe8e8e8, roughness:0.15, metalness:0.6});
    const innerMat = new THREE.MeshStandardMaterial({color:0x222222, roughness:0.3, metalness:0.8});
    const blueMat  = new THREE.MeshStandardMaterial({color:0x2196f3, roughness:0.3, metalness:0.5, emissive:new THREE.Color(0x2196f3), emissiveIntensity:0.15});
    const carbonMat= new THREE.MeshStandardMaterial({color:0x1a1a1a, roughness:0.1, metalness:0.95});

    // Sleek base
    shoulderPivot.position.y = 0.3;
    const base = cyl(1.2, 1.5, 0.3, 32, carbonMat); base.position.y = 0.15;
    baseGroup.add(base); modelMeshes.push(base);
    const bGlow = cyl(1.25, 1.25, 0.04, 32, blueMat); bGlow.position.y = 0.32;
    baseGroup.add(bGlow); modelMeshes.push(bGlow);

    // Shoulder — smooth capsule shape
    const s1 = cyl(0.6, 0.5, 0.8, 16, shellMat); s1.position.y = 0.5;
    shoulderPivot.add(s1); modelMeshes.push(s1);
    const sCap = sphere(0.5, 16, shellMat); sCap.position.y = 0.9;
    shoulderPivot.add(sCap); modelMeshes.push(sCap);

    // Upper arm — smooth tapered
    upperArmPivot.position.y = 0.9;
    const ua = cyl(0.3, 0.25, 2.0, 12, shellMat); ua.position.y = 1.0;
    upperArmPivot.add(ua); modelMeshes.push(ua);
    // Carbon fiber detail strip
    const cfStrip = box(0.05, 1.8, 0.02, carbonMat); cfStrip.position.set(0.22, 1.0, 0.18);
    upperArmPivot.add(cfStrip); modelMeshes.push(cfStrip);
    // Blue accent line
    const uaGlow = box(0.02, 1.6, 0.02, blueMat); uaGlow.position.set(-0.22, 1.0, 0.18);
    upperArmPivot.add(uaGlow); modelMeshes.push(uaGlow);

    // ── ELBOW HINGE (J2 = bend up/down) ──────────────────────────────────
    forearmPivot.position.y = 2.05;
    // Hinge pin (left-right, white shell)
    const bHingePin = cyl(0.1, 0.1, 0.82, 12, shellMat); bHingePin.rotation.z = Math.PI/2;
    forearmPivot.add(bHingePin); modelMeshes.push(bHingePin);
    // Blue LED rings on pin ends
    const bPinL = cyl(0.13, 0.13, 0.04, 12, blueMat); bPinL.rotation.z = Math.PI/2; bPinL.position.x = -0.42;
    forearmPivot.add(bPinL); modelMeshes.push(bPinL);
    const bPinR = cyl(0.13, 0.13, 0.04, 12, blueMat); bPinR.rotation.z = Math.PI/2; bPinR.position.x = 0.42;
    forearmPivot.add(bPinR); modelMeshes.push(bPinR);
    // Side shells
    const bSL = box(0.12, 0.46, 0.12, shellMat); bSL.position.set(-0.38, 0, 0); forearmPivot.add(bSL); modelMeshes.push(bSL);
    const bSR = box(0.12, 0.46, 0.12, shellMat); bSR.position.set(0.38, 0, 0); forearmPivot.add(bSR); modelMeshes.push(bSR);

    // ── SPIN DISC (J10 = forearm twist) ───────────────────────────────────
    const bSpinDisc = cyl(0.38, 0.38, 0.28, 20, innerMat); bSpinDisc.position.y = 0.2; forearmRotPivot.add(bSpinDisc); modelMeshes.push(bSpinDisc);
    const bSpinRing = cyl(0.41, 0.41, 0.04, 20, blueMat); bSpinRing.position.y = 0.3; forearmRotPivot.add(bSpinRing); modelMeshes.push(bSpinRing);
    const bCr1 = box(0.62, 0.03, 0.03, blueMat); bCr1.position.y = 0.34; forearmRotPivot.add(bCr1); modelMeshes.push(bCr1);
    const bCr2 = box(0.03, 0.03, 0.62, blueMat); bCr2.position.y = 0.34; forearmRotPivot.add(bCr2); modelMeshes.push(bCr2);

    // Forearm into forearmRotPivot
    const fa = cyl(0.24, 0.2, 1.7, 12, shellMat); fa.position.y = 1.25; forearmRotPivot.add(fa); modelMeshes.push(fa);
    const cfStrip2 = box(0.05, 1.5, 0.02, carbonMat); cfStrip2.position.set(0.22, 1.25, 0.18); forearmRotPivot.add(cfStrip2); modelMeshes.push(cfStrip2);

    // Wrist
    wristPivot.position.y = 2.18;
    const wrist = cyl(0.22, 0.25, 0.2, 12, innerMat); wristPivot.add(wrist); modelMeshes.push(wrist);
    const palm = box(1.4, 0.25, 0.8, shellMat); palm.position.y = 0.25; wristPivot.add(palm); modelMeshes.push(palm);
    const palmGlow = box(0.8, 0.02, 0.5, blueMat); palmGlow.position.y = 0.39; wristPivot.add(palmGlow); modelMeshes.push(palmGlow);

    const FING = [
      {x:-0.65, z:0.3, l:0.55, thumb:true},
      {x:-0.3,  z:0.45, l:0.65, thumb:false},
      {x: 0.0,  z:0.48, l:0.7,  thumb:false},
      {x: 0.3,  z:0.45, l:0.62, thumb:false},
      {x: 0.55, z:0.38, l:0.48, thumb:false},
    ];
    FING.forEach((f, i) => {
      const root = new THREE.Group(); root.position.set(f.x, 0.38, f.z * 0.5);
      if (f.thumb) root.rotation.z = 0.25;
      wristPivot.add(root);
      const meta = new THREE.Group(); root.add(meta);
      const s1 = cyl(0.08, 0.07, f.l*0.35, 8, shellMat); s1.position.y = f.l*0.175; meta.add(s1); modelMeshes.push(s1);
      const j1 = sphere(0.07, 8, innerMat); j1.position.y = f.l*0.35; meta.add(j1); modelMeshes.push(j1);
      const prox = new THREE.Group(); prox.position.y = f.l*0.35; meta.add(prox);
      const s2 = cyl(0.07, 0.06, f.l*0.3, 8, shellMat); s2.position.y = f.l*0.15; prox.add(s2); modelMeshes.push(s2);
      const j2b = sphere(0.06, 8, innerMat); j2b.position.y = f.l*0.3; prox.add(j2b); modelMeshes.push(j2b);
      const dist = new THREE.Group(); dist.position.y = f.l*0.3; prox.add(dist);
      const s3 = cyl(0.06, 0.04, f.l*0.25, 8, shellMat); s3.position.y = f.l*0.125; dist.add(s3); modelMeshes.push(s3);
      const tip = sphere(0.05, 8, blueMat); tip.position.y = f.l*0.25; dist.add(tip); modelMeshes.push(tip);
      fingers.push({meta, prox, dist, thumb:f.thumb, root});
    });
  }

  // ── Build initial model ────────────────────────────────────────────────
  let currentModel = 'industrial';
  function switchModel(name) {
    clearModel();
    currentModel = name;
    switch (name) {
      case 'industrial': buildIndustrial(); break;
      case 'humanoid':   buildHumanoid(); break;
      case 'mech':       buildMech(); break;
      case 'minimal':    buildMinimal(); break;
      case 'bionic':     buildBionic(); break;
      default:           buildIndustrial(); break;
    }
  }
  switchModel('industrial');

  // ── Render loop ────────────────────────────────────────────────────────
  function tick() {
    requestAnimationFrame(tick);

    for (let i = 0; i < NUM_JOINTS; i++) smooth[i] += (target[i] - smooth[i]) * 0.1;

    // J0: shoulder rotation (base turntable Y-axis)
    baseGroup.rotation.y = -(smooth[0] - 90) * Math.PI / 180;

    // J1: shoulder pitch (upper arm forward/back tilt)
    upperArmPivot.rotation.x = -(smooth[1] - 90) * Math.PI / 180;

    // J2: elbow bend
    forearmPivot.rotation.x = -(smooth[2] - 90) * Math.PI / 180;

    // J3: wrist pitch
    wristPivot.rotation.x = -(smooth[3] - 90) * Math.PI / 300;

    // J4: wrist roll
    wristPivot.rotation.z = -(smooth[4] - 90) * Math.PI / 300;

    // J5-J9: individual finger curls
    if (fingers.length >= 5) {
      for (let i = 0; i < Math.min(5, fingers.length); i++) {
        const curl = -(smooth[5 + i] - 90) * (Math.PI / 160);
        const axis = fingers[i].thumb ? 'z' : 'x';
        fingers[i].meta.rotation[axis] = curl * 0.4;
        fingers[i].prox.rotation[axis] = curl * 0.6;
        fingers[i].dist.rotation[axis] = curl * 0.5;
      }
    }

    // J10: forearm spin (rotation disc — twists the forearm around its own axis)
    forearmRotPivot.rotation.y = -(smooth[10] - 90) * Math.PI / 180;

    // J11: grip override (moves all fingers if not at 90)
    if (Math.abs(smooth[11] - 90) > 2 && fingers.length >= 5) {
      for (let i = 0; i < Math.min(5, fingers.length); i++) {
        const curl = -(smooth[11] - 90) * (Math.PI / 160);
        const axis = fingers[i].thumb ? 'z' : 'x';
        fingers[i].meta.rotation[axis] = curl * 0.4;
        fingers[i].prox.rotation[axis] = curl * 0.6;
        fingers[i].dist.rotation[axis] = curl * 0.5;
      }
    }

    // subtle idle animation
    const t = performance.now() * 0.001;
    upperArmPivot.rotation.z = Math.sin(t * 0.5) * 0.005;

    controls.update();
    renderer.render(scene, camera);
  }
  tick();

  // ── Public API ─────────────────────────────────────────────────────────
  window._arm3d = {
    NUM_JOINTS,
    JOINT_NAMES: ['Shoulder Rot','Shoulder Pitch','Elbow','Wrist Pitch','Wrist Roll',
                  'Thumb','Index','Middle','Ring','Pinky','Forearm Rot','Grip'],
    setAngles(a) { for (let i=0;i<NUM_JOINTS;i++) target[i] = a[i] ?? 90; },
    setJoint(id,a) { if (id>=0 && id<NUM_JOINTS) target[id] = a; },
    setGesture(name) {
      const p = POSES[name];
      if (p) for (let i=0;i<NUM_JOINTS;i++) target[i] = p[i] ?? 90;
    },
    getGestures() { return Object.keys(POSES); },
    setModel(name) { switchModel(name); },
    resize() {
      const w2 = container.clientWidth, h2 = container.clientHeight;
      if (!w2 || !h2) return;
      camera.aspect = w2/h2;
      camera.updateProjectionMatrix();
      renderer.setSize(w2, h2);
    },
    setTheme(name) {
      const c = new THREE.Color(THEME_SCENE_BG[name] || THEME_SCENE_BG.dark);
      scene.background = c; scene.fog.color = c;
    },
  };
  window._arm3d.setTheme(currentTheme);
}

// =============================================================================
// SESSIONS (past recordings from disk)
// =============================================================================

function setSelectedDataset(datasetId) {
  S.selectedDatasetId = datasetId || '';
  const sel = $('dataset-select');
  if (sel) sel.value = S.selectedDatasetId;
}

async function loadDatasets() {
  try {
    const list = await get('/api/datasets');
    S.datasets = Array.isArray(list) ? list : [];
  } catch {
    S.datasets = [];
  }

  if (!S.datasets.find(d => d.dataset_id === S.selectedDatasetId)) {
    const preferred = S.datasets.find(d => d.signal_profile === S.signalProfileKey) || S.datasets[0];
    S.selectedDatasetId = preferred?.dataset_id || '';
  }
  syncResearchUI();
}

async function loadExperiments() {
  try {
    const list = await get('/api/experiments');
    S.experiments = Array.isArray(list) ? list : [];
  } catch {
    S.experiments = [];
  }
  syncResearchUI();
}

function humanizeExperimentSplit(strategy) {
  if (strategy === 'random_window') return 'random windows';
  if (strategy === 'leave_one_session_out') return 'leave one session out';
  if (strategy === 'leave_one_subject_out') return 'leave one subject out';
  return 'temporal holdout';
}

function syncResearchUI() {
  const datasetStatus = $('dataset-status');
  const experimentStatus = $('experiment-status');
  const datasetSel = $('dataset-select');
  const datasetList = $('dataset-list');
  const experimentList = $('experiment-list');
  const createBtn = $('btn-dataset-create');
  const runBtn = $('btn-experiment-run');
  const splitSel = $('experiment-split');
  const holdoutInput = $('experiment-holdout');
  const gapInput = $('experiment-gap');

  const liveIds = new Set((S.sessions || []).map(s => s.session_id).filter(Boolean));
  S.selectedSessionIds = new Set([...S.selectedSessionIds].filter(id => liveIds.has(id)));
  const selectedSessions = (S.sessions || []).filter(s => S.selectedSessionIds.has(s.session_id));
  const matchingSessions = selectedSessions.filter(s => String(s.signal_profile || '').toLowerCase() === S.signalProfileKey);
  const mismatchedSessions = selectedSessions.filter(s => String(s.signal_profile || '').toLowerCase() !== S.signalProfileKey);

  if (datasetStatus) {
    if (!selectedSessions.length) {
      datasetStatus.textContent = `Select sessions in the Sessions card to build a ${S.signalProfileName} dataset.`;
    } else if (mismatchedSessions.length) {
      datasetStatus.textContent = `${selectedSessions.length} selected: ${matchingSessions.length} match ${S.signalProfileName}, ${mismatchedSessions.length} use a different profile and will be rejected.`;
    } else {
      datasetStatus.textContent = `${matchingSessions.length} ${S.signalProfileName} session(s) selected. Create a dataset manifest from them.`;
    }
  }

  if (createBtn) createBtn.disabled = !selectedSessions.length;

  if (datasetSel) {
    const previous = S.selectedDatasetId;
    datasetSel.innerHTML = '';
    if (!S.datasets.length) {
      S.selectedDatasetId = '';
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No datasets';
      datasetSel.appendChild(opt);
    } else {
      S.datasets.forEach(d => {
        const opt = document.createElement('option');
        opt.value = d.dataset_id;
        const name = d.name || d.dataset_id;
        const profile = d.signal_profile_name || d.signal_profile || 'Signal';
        opt.textContent = `${name} (${profile})`;
        opt.selected = d.dataset_id === previous;
        datasetSel.appendChild(opt);
      });
      if (!S.datasets.find(d => d.dataset_id === previous)) {
        datasetSel.value = S.datasets[0].dataset_id;
        S.selectedDatasetId = datasetSel.value;
      }
    }
  }

  const activeDataset = S.datasets.find(d => d.dataset_id === S.selectedDatasetId) || null;
  const datasetReady = !!(activeDataset && activeDataset.ready);
  const datasetProfileMismatch = !!(activeDataset && activeDataset.signal_profile !== S.signalProfileKey);
  const splitStrategy = splitSel?.value || 'temporal_holdout';
  const temporalGapEnabled = splitStrategy === 'temporal_holdout';
  const losoEnabled = splitStrategy === 'leave_one_session_out';
  const subjectHoldoutEnabled = splitStrategy === 'leave_one_subject_out';
  const losoReady = !!(activeDataset && activeDataset.ready_for_loso);
  const subjectHoldoutReady = !!(activeDataset && activeDataset.ready_for_subject_holdout);

  if (runBtn) {
    runBtn.disabled = !activeDataset
      || !datasetReady
      || datasetProfileMismatch
      || (losoEnabled && !losoReady)
      || (subjectHoldoutEnabled && !subjectHoldoutReady);
  }
  if (holdoutInput) {
    const nextValue = Math.min(0.5, Math.max(0.1, Number(holdoutInput.value || 0.25)));
    if (Number.isFinite(nextValue)) holdoutInput.value = nextValue.toFixed(2);
    holdoutInput.disabled = losoEnabled || subjectHoldoutEnabled;
  }
  if (gapInput) {
    const nextGap = Math.min(2.0, Math.max(0.0, Number(gapInput.value || 0.2)));
    if (Number.isFinite(nextGap)) gapInput.value = nextGap.toFixed(2);
    gapInput.disabled = !temporalGapEnabled || losoEnabled || subjectHoldoutEnabled;
  }

  if (experimentStatus) {
    if (!activeDataset) {
      experimentStatus.textContent = 'Choose a saved dataset, then run an offline experiment.';
    } else if (datasetProfileMismatch) {
      experimentStatus.textContent = `Switch Signal Type to ${(activeDataset.signal_profile_name || activeDataset.signal_profile || 'that profile')} before running this dataset.`;
    } else if (!activeDataset.ready) {
      experimentStatus.textContent = `Dataset "${activeDataset.name || activeDataset.dataset_id}" is not ready yet. It needs at least 2 labels and 8 estimated windows.`;
    } else if (losoEnabled && !losoReady) {
      const groups = (activeDataset.full_session_groups || []).length || 0;
      experimentStatus.textContent = `Dataset "${activeDataset.name || activeDataset.dataset_id}" is not ready for leave-one-session-out yet. Record at least 2 full protocol runs with the same labels across a session group. Current full runs: ${groups}.`;
    } else if (subjectHoldoutEnabled && !subjectHoldoutReady) {
      const subjects = (activeDataset.full_subjects || []).length || 0;
      experimentStatus.textContent = `Dataset "${activeDataset.name || activeDataset.dataset_id}" is not ready for leave-one-subject-out yet. Record at least 2 subjects whose sessions cover every label. Current full subjects: ${subjects}.`;
    } else {
      const labels = (activeDataset.labels_present || []).join(', ') || 'no labels';
      const gapText = temporalGapEnabled && !losoEnabled && !subjectHoldoutEnabled ? ` with ${(Number(gapInput?.value || 0.2)).toFixed(2)} s gap` : '';
      const losoText = losoEnabled ? ` Full runs: ${(activeDataset.full_session_groups || []).length || 0}.` : '';
      const subjectText = subjectHoldoutEnabled ? ` Full subjects: ${(activeDataset.full_subjects || []).length || 0}.` : '';
      experimentStatus.textContent = `Dataset ready: ${activeDataset.estimated_windows || 0} windows across ${labels}. ${humanizeExperimentSplit(splitStrategy)}${gapText}.${losoText}${subjectText}`;
    }
  }

  if (datasetList) {
    datasetList.innerHTML = '';
    if (!S.datasets.length) {
      datasetList.innerHTML = '<div style="font-size:11px;color:var(--text-dim)">No datasets created yet</div>';
    } else {
      S.datasets.slice(0, 6).forEach(d => {
        const row = document.createElement('div');
        row.className = 'session-item';

        const copy = document.createElement('div');
        copy.className = 'session-copy';
        const sid = document.createElement('div');
        sid.className = 'sid';
        sid.textContent = d.name || d.dataset_id || '?';
        const meta = document.createElement('div');
        meta.className = 'session-meta';
        const bits = [];
        if (d.signal_profile_name || d.signal_profile) bits.push(d.signal_profile_name || d.signal_profile);
        if (d.n_sessions) bits.push(`${d.n_sessions} sessions`);
        if (Array.isArray(d.subjects_present) && d.subjects_present.length) bits.push(`${d.subjects_present.length} subjects`);
        if (Array.isArray(d.session_groups_present) && d.session_groups_present.length) bits.push(`${d.session_groups_present.length} runs`);
        if (d.estimated_windows) bits.push(`${d.estimated_windows} windows`);
        bits.push(d.ready ? 'ready' : 'needs labels');
        if (d.ready_for_loso) bits.push('LOSO ready');
        if (d.ready_for_subject_holdout) bits.push('subject holdout ready');
        meta.textContent = bits.join(' | ');
        copy.append(sid, meta);
        row.appendChild(copy);

        const actions = document.createElement('div');
        actions.className = 'session-actions';
        const useBtn = document.createElement('button');
        useBtn.className = 'btn session-action';
        useBtn.textContent = d.dataset_id === S.selectedDatasetId ? 'Selected' : 'Use';
        useBtn.disabled = d.dataset_id === S.selectedDatasetId;
        useBtn.onclick = () => {
          setSelectedDataset(d.dataset_id || '');
          syncResearchUI();
        };
        actions.appendChild(useBtn);
        row.appendChild(actions);
        datasetList.appendChild(row);
      });
    }
  }

  if (experimentList) {
    experimentList.innerHTML = '';
    if (!S.experiments.length) {
      experimentList.innerHTML = '<div style="font-size:11px;color:var(--text-dim)">No experiments run yet</div>';
    } else {
      S.experiments.slice(0, 6).forEach(exp => {
        const row = document.createElement('div');
        row.className = 'session-item';

        const copy = document.createElement('div');
        copy.className = 'session-copy';
        const sid = document.createElement('div');
        sid.className = 'sid';
        sid.textContent = `${exp.classifier || 'Model'} - ${exp.status || 'unknown'}`;
        const meta = document.createElement('div');
        meta.className = 'session-meta';
        const bits = [];
        if (exp.dataset_name || exp.dataset_id) bits.push(exp.dataset_name || exp.dataset_id);
        if (exp.split_strategy) bits.push(humanizeExperimentSplit(exp.split_strategy));
        if (exp.holdout_subject_id) bits.push(`subject ${exp.holdout_subject_id}`);
        if (exp.holdout_group_id) bits.push(`holdout ${exp.holdout_group_id}`);
        if (exp.val_accuracy != null) bits.push(`${(Number(exp.val_accuracy) * 100).toFixed(1)}%`);
        if (exp.f1_macro != null) bits.push(`F1 ${(Number(exp.f1_macro) * 100).toFixed(1)}%`);
        if (exp.n_windows) bits.push(`${exp.n_windows} windows`);
        if (exp.duration_s != null) bits.push(`${Number(exp.duration_s).toFixed(1)}s`);
        meta.textContent = bits.join(' | ');
        copy.append(sid, meta);
        row.appendChild(copy);
        experimentList.appendChild(row);
      });
    }
  }
}

async function loadSessions() {
  try {
    const list = await get('/api/sessions');
    S.sessions = Array.isArray(list) ? list : [];
    const liveIds = new Set(S.sessions.map(s => s.session_id).filter(Boolean));
    S.selectedSessionIds = new Set([...S.selectedSessionIds].filter(id => liveIds.has(id)));
    populatePlaybackSessionOptions();
    setSelectedPlaybackSession(localStorage.getItem('kyma-playback-session') || S.playbackSessionId || $('playback-session')?.value || '');

    const el = $('session-list');
    el.innerHTML = '';
    if (!S.sessions.length) {
      el.innerHTML = '<div style="font-size:11px;color:var(--text-dim)">No sessions</div>';
      syncStreamModeUI();
      syncResearchUI();
      return;
    }

    S.sessions.slice(0, 10).forEach(s => {
      const row = document.createElement('div');
      row.className = 'session-item';

      if (s.session_id) {
        const pick = document.createElement('input');
        pick.type = 'checkbox';
        pick.checked = S.selectedSessionIds.has(s.session_id);
        pick.style.margin = '4px 8px 0 0';
        pick.onchange = () => {
          if (pick.checked) S.selectedSessionIds.add(s.session_id);
          else S.selectedSessionIds.delete(s.session_id);
          syncResearchUI();
        };
        row.appendChild(pick);
      }

      const copy = document.createElement('div');
      copy.className = 'session-copy';

      const sid = document.createElement('div');
      sid.className = 'sid';
      sid.textContent = s.label ? `${s.label}` : (s.session_id || '?');

      const sub = document.createElement('div');
      sub.className = 'session-sub';
      sub.textContent = s.label ? (s.session_id || '?') : (s.created_at_utc || '');

      const meta = document.createElement('div');
      meta.className = 'session-meta';
      const bits = [];
      if (s.signal_profile_name || s.signal_profile) bits.push(s.signal_profile_name || s.signal_profile);
      if (s.subject_id) bits.push(s.subject_id);
      if (s.condition) bits.push(s.condition);
      if (s.session_group_id) bits.push(`run ${s.session_group_id}`);
      if (s.protocol_title || s.protocol_key) bits.push(s.protocol_title || s.protocol_key);
      if (s.stream_source) bits.push(s.stream_source);
      if (s.duration_s) bits.push(`${Number(s.duration_s).toFixed(1)}s`);
      if (s.n_samples) bits.push(`${s.n_samples} samples`);
      meta.textContent = bits.join(' | ');

      copy.appendChild(sid);
      if (sub.textContent) copy.appendChild(sub);
      copy.appendChild(meta);
      row.appendChild(copy);

      const actions = document.createElement('div');
      actions.className = 'session-actions';

      if (s.playable) {
        const useBtn = document.createElement('button');
        useBtn.className = 'btn session-action';
        useBtn.textContent = 'Use';
        useBtn.onclick = () => {
          setSelectedSource('playback');
          setSelectedPlaybackSession(s.session_id || '');
          S.playbackSessionId = s.session_id || '';
          syncStreamModeUI();
          toast(`Playback selected: ${s.session_id}`);
        };
        actions.appendChild(useBtn);
      }

      if (s.session_id) {
        const exportBtn = document.createElement('button');
        exportBtn.className = 'btn session-action';
        exportBtn.textContent = 'Export';
        exportBtn.onclick = async () => {
          try {
            exportBtn.disabled = true;
            exportBtn.textContent = 'Exporting...';
            const out = await post(`/api/sessions/${encodeURIComponent(s.session_id)}/export/bids`, {});
            const note = out.validated ? 'validated' : 'validation warning';
            toast(`BIDS export saved: ${out.session || s.session_id} (${note})`);
          } catch (err) {
            toast(err.message || 'BIDS export failed', 'red');
          } finally {
            exportBtn.disabled = false;
            exportBtn.textContent = 'Export';
          }
        };
        actions.appendChild(exportBtn);
      }

      if (actions.children.length) {
        row.appendChild(actions);
      }

      el.appendChild(row);
    });
    syncStreamModeUI();
    syncResearchUI();
  } catch {}
}


// =============================================================================
// TOAST (floating notification at bottom right)
// =============================================================================

let _tt;
function toast(msg, color='') {
  const el = $('toast');
  el.textContent = msg;
  el.style.borderColor = color === 'red' ? 'var(--red)' : 'var(--border)';
  el.classList.add('show');
  clearTimeout(_tt);
  _tt = setTimeout(() => el.classList.remove('show'), 3000);
}


// =============================================================================
// FATIGUE MONITOR
//
// simple approach: track peak RMS per channel over the session.
// as muscles tire, RMS drops relative to the peak. when average RMS
// across channels falls below 60% of peak, you're getting fatigued.
// =============================================================================

function updateFatigue(rms) {
  // update peak for each channel (only goes up, never down)
  for (let i = 0; i < N_CH; i++) {
    if (rms[i] > S.peakRms[i]) S.peakRms[i] = rms[i];
  }

  // calculate fatigue as ratio of current to peak
  let ratioSum = 0, active = 0;
  for (let i = 0; i < N_CH; i++) {
    if (S.peakRms[i] > 0.001) {  // only count channels with real signal
      ratioSum += rms[i] / S.peakRms[i];
      active++;
    }
  }

  if (active > 0) {
    // smooth it so it doesn't jump around like crazy
    const raw = ratioSum / active;
    S.fatigue += (raw - S.fatigue) * 0.02;
  }

  // update the bar
  const pct = Math.max(0, Math.min(100, S.fatigue * 100));
  const bar = $('fatigue-bar');
  bar.style.width = `${pct}%`;
  bar.style.background = pct > 70 ? 'var(--green)' : pct > 40 ? 'var(--yellow)' : 'var(--red)';

  const label = pct > 70 ? 'Fresh' : pct > 40 ? 'Getting tired' : 'Fatigued - rest soon';
  $('fatigue-label').textContent = `Endurance: ${pct.toFixed(0)}% -- ${label}`;
}


// =============================================================================
// GESTURE TIMELINE
//
// draws the last 60 predictions as colored blocks on a mini canvas.
// each gesture gets a different color, block height = confidence.
// you can visually see if the classifier is stable or jumping around.
// =============================================================================

function drawTimeline() {
  const cv = $('timeline-canvas');
  if (!cv) return;
  const tx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  const cols = chColors();
  const timeline = S.review.paused && Array.isArray(S.review.timelineSnapshot)
    ? S.review.timelineSnapshot
    : S.timeline;

  tx.clearRect(0, 0, W, H);
  const tlBgs = {dark:'#f6f8fb', retro:'#000', mario:'#000040', gameboy:'#9bbc0f', cyberpunk:'#0a0010'};
  tx.fillStyle = tlBgs[currentTheme] || '#f6f8fb';
  tx.fillRect(0, 0, W, H);

  const n = timeline.length;
  if (!n) return;

  const bw = W / S.timelineMax;  // block width
  for (let i = 0; i < n; i++) {
    const t = timeline[i];
    const x = i * bw;
    const h = t.c * H;  // height = confidence
    tx.fillStyle = cols[t.g % cols.length];
    tx.fillRect(x, H - h, bw - 1, h);
  }
}

function renderSpectrum() {
  const cv = $('spectrum-canvas');
  if (!cv) return;
  const gx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  gx.clearRect(0, 0, W, H);

  const theme = THEME_CANVAS[currentTheme] || THEME_CANVAS.dark;
  gx.fillStyle = theme.bg;
  gx.fillRect(0, 0, W, H);
  gx.strokeStyle = theme.grid;
  gx.lineWidth = 1;
  for (let i = 1; i <= 4; i++) {
    const y = (H / 5) * i;
    gx.beginPath();
    gx.moveTo(0, y);
    gx.lineTo(W, y);
    gx.stroke();
  }

  const spectrum = S.diagnostics.spectrum || {};
  const freq = Array.isArray(spectrum.freq_hz) ? spectrum.freq_hz : [];
  const mag = Array.isArray(spectrum.mag_db) ? spectrum.mag_db : [];
  if (!freq.length || !mag.length) {
    gx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-dim') || '#8b949e';
    gx.font = '11px Segoe UI';
    gx.fillText('Waiting for live spectrum data', 12, H / 2);
  } else {
    const minDb = Math.min(-80, ...mag);
    const maxDb = 2;
    gx.strokeStyle = chColors()[0] || '#58a6ff';
    gx.lineWidth = 2;
    gx.beginPath();
    for (let i = 0; i < mag.length; i++) {
      const x = (i / Math.max(mag.length - 1, 1)) * (W - 1);
      const norm = (mag[i] - minDb) / (maxDb - minDb);
      const y = H - (Math.max(0, Math.min(1, norm)) * (H - 8)) - 4;
      if (i === 0) gx.moveTo(x, y);
      else gx.lineTo(x, y);
    }
    gx.stroke();
    gx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-dim') || '#8b949e';
    gx.font = '10px Segoe UI';
    gx.fillText(`0 Hz`, 6, H - 6);
    gx.fillText(`${Math.round(freq[freq.length - 1] || 0)} Hz`, W - 42, H - 6);
  }

  const noise = S.diagnostics.noise || {};
  $('hum50-val').textContent = `${Number(noise.hum_50_db || 0).toFixed(1)} dB`;
  $('hum60-val').textContent = `${Number(noise.hum_60_db || 0).toFixed(1)} dB`;
  $('drift-val').textContent = `${Number(noise.drift_db || 0).toFixed(1)} dB`;
  $('clip-val').textContent = `${Number(noise.clip_pct || 0).toFixed(2)} %`;
  $('crest-val').textContent = Number(noise.crest_factor || 0).toFixed(2);
}

function syncFilterChainLabel() {
  const el = $('filter-chain-label');
  if (!el) return;
  const baseFilters = Array.isArray((S.availableProfiles.find(p => p.key === S.signalProfileKey) || {}).filters)
    ? (S.availableProfiles.find(p => p.key === S.signalProfileKey) || {}).filters
    : [];
  const base = baseFilters.map(stage => stage.kind).join(' -> ') || 'no base stages';
  const active = S.diagnostics.active_filter || S.filterLab.active_filter || null;
  if (active?.name) {
    const mode = active.apply_mode === 'replace_defaults' ? 'replacing profile stages' : 'appended after profile stages';
    el.textContent = `Profile chain: ${base}. Custom: ${active.name} (${mode}).`;
  } else {
    el.textContent = `Profile chain: ${base}. No custom filter active.`;
  }
}

function primeFilterDefaultsFromProfile() {
  const profile = S.availableProfiles.find(item => item.key === S.signalProfileKey);
  if (!profile || !Array.isArray(profile.filters)) return;
  const primary = profile.filters.find(stage => stage.kind === 'bandpass')
    || profile.filters.find(stage => stage.kind === 'lowpass')
    || profile.filters.find(stage => stage.kind === 'highpass')
    || profile.filters.find(stage => stage.kind === 'bandstop');
  if (!primary) return;

  const responseType = primary.kind === 'lowpass' ? 'lowpass'
    : primary.kind === 'highpass' ? 'highpass'
    : primary.kind === 'bandstop' ? 'bandstop'
    : 'bandpass';

  if ($('filter-response-type')) $('filter-response-type').value = responseType;
  if ($('filter-order')) $('filter-order').value = String(primary.order || 2);
  if (primary.cutoff_hz && $('filter-cutoff-hz')) $('filter-cutoff-hz').value = String(primary.cutoff_hz);
  if (primary.low_hz && $('filter-low-hz')) $('filter-low-hz').value = String(primary.low_hz);
  if (primary.high_hz && $('filter-high-hz')) $('filter-high-hz').value = String(primary.high_hz);
  if ($('filter-name') && !$('filter-name').value.trim()) $('filter-name').placeholder = `${S.signalProfileName} custom filter`;
  updateFilterFieldVisibility();
}


// =============================================================================
// SIGNAL QUALITY GRID
//
// 8 blocks (4x2 grid) showing electrode health for each channel.
// green = good, yellow = weak, red = noisy/saturated, grey = dead
// =============================================================================

function buildQualityGrid() {
  const c = $('quality-grid');
  c.innerHTML = '';
  for (let i = 0; i < N_CH; i++) {
    const label = (S.channelLabels[i] || `CH${i + 1}`).slice(0, 7);
    c.innerHTML += `<div class="q-block dead ${isChannelVisible(i) ? '' : 'row-hidden'}" id="qb-${i}">${label}<br>--</div>`;
  }
}

function updateQualityGrid(values) {
  for (let i = 0; i < N_CH; i++) {
    const el = $(`qb-${i}`);
    if (!el) continue;
    if (!isChannelVisible(i)) {
      el.className = 'q-block dead row-hidden';
      el.innerHTML = `${(S.channelLabels[i] || `CH${i + 1}`).slice(0, 7)}<br>OFF`;
      continue;
    }
    const v = Number(values[i] || 0);
    let cls, label;
    if (v < 0.02) {
      cls = 'dead'; label = 'NO SIG';      // electrode probably fell off
    } else if (v > 0.85) {
      cls = 'bad';  label = 'NOISY';        // saturated or bad contact
    } else if (v < 0.12) {
      cls = 'ok';   label = 'WEAK';         // signal is there but low
    } else {
      cls = 'good'; label = 'OK';           // good muscle signal
    }
    el.className = `q-block ${cls}`;
    el.innerHTML = `CH${i+1}<br>${label}`;
  }
}


// =============================================================================
// PERFORMANCE STATS
//
// updates once per second with: ws latency, predictions per second,
// total training windows in buffer
// =============================================================================

function updatePerformance() {
  // prediction rate
  const now = performance.now();
  const dt = (now - S.predCountTime) / 1000;
  S.predRate = Math.round(S.predCount / dt);
  $('pred-rate').textContent = S.predRate;
  S.predCount = 0;
  S.predCountTime = now;

  // ws latency
  $('latency-val').textContent = S.wsLatency > 0 ? `${S.wsLatency} ms` : '-- ms';

  const timing = S.diagnostics.timing || {};
  const signalAge = S.lastSignalAtClient > 0
    ? Math.max(0, now - S.lastSignalAtClient)
    : Number(timing.signal_age_ms || S.safety.signal_age_ms || 0);
  $('decode-latency-val').textContent = Number(timing.process_last_ms || 0) > 0
    ? `${Number(timing.process_last_ms).toFixed(2)} ms`
    : '-- ms';
  $('window-jitter-val').textContent = `${Number(timing.interval_jitter_ms || 0).toFixed(2)} ms`;
  $('drop-val').textContent = String(timing.dropped_windows || 0);
  $('stale-val').textContent = `${signalAge.toFixed(0)} ms`;
  $('watchdog-val').textContent = S.safety.stale
    ? `STALE (${S.safety.trip_count || 0})`
    : (S.safety.enabled ? 'armed' : 'off');

  // training window count
  let total = 0;
  Object.values(S.trainCounts).forEach(n => total += n);
  $('win-count').textContent = total;
}


// =============================================================================
// FILTER LAB
// =============================================================================

function getCurrentFilterModel() {
  return S.filterLab.preview || S.filterLab.selected_filter || null;
}

function getBaseProfileFilterStages() {
  const profile = S.availableProfiles.find(item => item.key === S.signalProfileKey) || {};
  return Array.isArray(profile.filters) ? profile.filters : [];
}

function getBaseProfileFilterChain() {
  const stages = getBaseProfileFilterStages();
  return stages.map(stage => stage.kind).join(' -> ') || 'no base stages';
}

function formatFilterCutoffSummary(model) {
  if (!model) return 'none';
  const responseType = String(model.response_type || '');
  if (responseType === 'lowpass' || responseType === 'highpass') {
    return `${responseType} @ ${Number(model.cutoff_hz || 0).toFixed(2)} Hz`;
  }
  return `${responseType} ${Number(model.low_hz || 0).toFixed(2)}-${Number(model.high_hz || 0).toFixed(2)} Hz`;
}

function updateFilterFieldVisibility() {
  const responseType = $('filter-response-type')?.value || 'bandpass';
  const method = $('filter-method')?.value || 'butter';
  const singleWrap = $('filter-single-cutoff-wrap');
  const bandWrap = $('filter-band-cutoff-wrap');
  if (singleWrap) singleWrap.style.display = ['lowpass', 'highpass'].includes(responseType) ? 'block' : 'none';
  if (bandWrap) bandWrap.style.display = ['bandpass', 'bandstop'].includes(responseType) ? 'flex' : 'none';
  if ($('filter-rp-db')) $('filter-rp-db').disabled = !['cheby1', 'ellip'].includes(method);
  if ($('filter-rs-db')) $('filter-rs-db').disabled = !['cheby2', 'ellip'].includes(method);
}

function collectFilterSpec() {
  return {
    name: ($('filter-name')?.value || '').trim(),
    profile: S.signalProfileKey,
    method: $('filter-method')?.value || 'butter',
    response_type: $('filter-response-type')?.value || 'bandpass',
    order: Number($('filter-order')?.value || 2),
    sample_rate: Number(S.sampleRate || 250),
    cutoff_hz: Number($('filter-cutoff-hz')?.value || 0) || null,
    low_hz: Number($('filter-low-hz')?.value || 0) || null,
    high_hz: Number($('filter-high-hz')?.value || 0) || null,
    rp_db: Number($('filter-rp-db')?.value || 1),
    rs_db: Number($('filter-rs-db')?.value || 40),
    apply_mode: $('filter-apply-mode')?.value || 'append',
  };
}

function renderFilterResponseCanvas() {
  const cv = $('filter-response-canvas');
  if (!cv) return;
  const gx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  gx.clearRect(0, 0, W, H);
  const theme = THEME_CANVAS[currentTheme] || THEME_CANVAS.dark;
  gx.fillStyle = theme.bg;
  gx.fillRect(0, 0, W, H);
  gx.strokeStyle = theme.grid;
  gx.lineWidth = 1;
  for (let i = 1; i <= 4; i++) {
    const y = (H / 5) * i;
    gx.beginPath();
    gx.moveTo(0, y);
    gx.lineTo(W, y);
    gx.stroke();
  }

  const model = getCurrentFilterModel();
  const response = model?.response || {};
  const freq = Array.isArray(response.freq_hz) ? response.freq_hz : [];
  const mag = Array.isArray(response.mag_db) ? response.mag_db : [];
  if (!freq.length || !mag.length) {
    gx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-dim') || '#8b949e';
    gx.font = '12px Segoe UI';
    gx.fillText('Preview a filter to inspect its frequency response', 14, H / 2);
    return;
  }

  const minDb = Math.min(-100, ...mag);
  const maxDb = Math.max(6, ...mag);
  gx.strokeStyle = chColors()[2] || '#ff7b72';
  gx.lineWidth = 2;
  gx.beginPath();
  for (let i = 0; i < mag.length; i++) {
    const x = (i / Math.max(mag.length - 1, 1)) * (W - 1);
    const norm = (mag[i] - minDb) / Math.max(maxDb - minDb, 1);
    const y = H - (Math.max(0, Math.min(1, norm)) * (H - 12)) - 6;
    if (i === 0) gx.moveTo(x, y);
    else gx.lineTo(x, y);
  }
  gx.stroke();
  gx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-dim') || '#8b949e';
  gx.font = '10px Segoe UI';
  gx.fillText('0 Hz', 8, H - 8);
  gx.fillText(`${Math.round(freq[freq.length - 1] || 0)} Hz`, W - 42, H - 8);
}

function renderFilterPoleZeroCanvas() {
  const cv = $('filter-polezero-canvas');
  if (!cv) return;
  const gx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  gx.clearRect(0, 0, W, H);

  const theme = THEME_CANVAS[currentTheme] || THEME_CANVAS.dark;
  gx.fillStyle = theme.bg;
  gx.fillRect(0, 0, W, H);

  const model = getCurrentFilterModel();
  const response = model?.response || {};
  const zeros = Array.isArray(response.zeros) ? response.zeros : [];
  const poles = Array.isArray(response.poles) ? response.poles : [];
  if (!zeros.length && !poles.length) {
    gx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-dim') || '#8b949e';
    gx.font = '12px Segoe UI';
    gx.fillText('Preview a filter to inspect pole / zero placement', 14, H / 2);
    return;
  }

  const radii = [...zeros, ...poles].map(item => Number(item?.radius || Math.hypot(item?.re || 0, item?.im || 0)));
  const range = Math.max(1.25, Math.ceil((Math.max(...radii, 1) + 0.15) * 10) / 10);
  const margin = 18;
  const radiusPx = Math.min(W, H) * 0.38;
  const cx = W / 2;
  const cy = H / 2;
  const toX = re => cx + (Number(re || 0) / range) * radiusPx;
  const toY = im => cy - (Number(im || 0) / range) * radiusPx;

  gx.strokeStyle = theme.grid;
  gx.lineWidth = 1;
  gx.beginPath();
  gx.moveTo(margin, cy);
  gx.lineTo(W - margin, cy);
  gx.moveTo(cx, margin);
  gx.lineTo(cx, H - margin);
  gx.stroke();

  gx.strokeStyle = chColors()[0] || '#58a6ff';
  gx.beginPath();
  gx.arc(cx, cy, radiusPx / range, 0, Math.PI * 2);
  gx.stroke();

  gx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-dim') || '#8b949e';
  gx.font = '10px Segoe UI';
  gx.fillText('unit circle', cx + (radiusPx / range) + 8, cy - 8);
  gx.fillText(`Re ±${range.toFixed(1)}`, 8, H - 8);

  gx.strokeStyle = chColors()[2] || '#ff7b72';
  gx.fillStyle = 'transparent';
  zeros.forEach(item => {
    const x = toX(item.re);
    const y = toY(item.im);
    gx.beginPath();
    gx.arc(x, y, 5, 0, Math.PI * 2);
    gx.stroke();
  });

  gx.strokeStyle = chColors()[1] || '#3fb950';
  poles.forEach(item => {
    const x = toX(item.re);
    const y = toY(item.im);
    gx.beginPath();
    gx.moveTo(x - 5, y - 5);
    gx.lineTo(x + 5, y + 5);
    gx.moveTo(x + 5, y - 5);
    gx.lineTo(x - 5, y + 5);
    gx.stroke();
  });
}

function formatFixedPointLabel(entry) {
  if (!entry) return '--';
  return `${Number(entry.integer_bits || 0)} int / ${Number(entry.frac_bits || 0)} frac`;
}

function renderFilterQuantization() {
  const model = getCurrentFilterModel();
  const quant = model?.response?.quantization || null;
  $('filter-qmax-coeff').textContent = quant
    ? Number(quant.max_abs_coeff || 0).toFixed(4)
    : '--';
  $('filter-q15-fit').textContent = quant
    ? ((quant.direct_q15?.fits ? 'fits' : 'scale') + ` | ${Number(quant.direct_q15?.overflow_coefficients || 0)} ovf`)
    : '--';
  $('filter-q31-fit').textContent = quant
    ? ((quant.direct_q31?.fits ? 'fits' : 'scale') + ` | ${Number(quant.direct_q31?.overflow_coefficients || 0)} ovf`)
    : '--';
  $('filter-q16fmt').textContent = quant ? formatFixedPointLabel(quant.recommended_s16) : '--';
  $('filter-q32fmt').textContent = quant ? formatFixedPointLabel(quant.recommended_s32) : '--';

  const mem = quant?.memory_bytes || {};
  $('filter-qmem-s16').textContent = quant ? `${Number(mem.int16 || 0)} B` : '--';
  $('filter-qmem-s32').textContent = quant ? `${Number(mem.int32 || 0)} B` : '--';
  $('filter-quant-note').textContent = quant
    ? `Direct Q1.15/Q1.31 checks show whether the raw SOS coefficients fit without scaling. Recommended signed 16/32-bit formats choose integer and fractional bits to fit the current design.`
    : 'Preview or load a filter to estimate fixed-point formats and state memory.';
}

function renderFilterSOS() {
  const model = getCurrentFilterModel();
  const area = $('filter-sos-text');
  if (!area) return;
  if (!model || !Array.isArray(model.sos) || !model.sos.length) {
    area.value = '// No filter designed yet';
    return;
  }
  area.value = model.sos
    .map((row, idx) => `section ${idx + 1}: ${row.map(v => Number(v).toFixed(10)).join(', ')}`)
    .join('\n');
}

function renderFilterExportPreview() {
  const target = $('filter-export-target')?.value || 'kyma_host';
  const model = getCurrentFilterModel();
  const note = $('filter-export-note');
  const area = $('filter-export-code');
  if (!area) return;
  if (!model || !model.exports || !model.exports[target]) {
    area.value = '// Preview or select a saved filter to export code';
    if (note) note.textContent = 'Export the selected or previewed filter as reusable code.';
    return;
  }
  const entry = model.exports[target];
  area.value = entry.code || '';
  if (note) {
    note.textContent = entry.available === false
      ? `${entry.label} is not available for this filter shape.`
      : `${entry.label} export ready${entry.filename ? ` -> ${entry.filename}` : ''}.`;
  }
}

function buildBenchReportText() {
  const model = getCurrentFilterModel();
  const summary = model?.summary || model || {};
  const response = model?.response || {};
  const quant = response.quantization || {};
  const noise = S.diagnostics.noise || {};
  const timing = S.diagnostics.timing || {};
  const activeFilter = S.filterLab.active_filter || S.diagnostics.active_filter || null;
  const sampleRate = Number(S.sampleRate || 250);
  const signalAge = Number(timing.signal_age_ms || S.safety.signal_age_ms || 0);
  const lines = [
    '# KYMA Bench Report',
    '',
    `Generated: ${new Date().toISOString()}`,
    `Profile: ${S.signalProfileName} (${S.signalProfileKey.toUpperCase()})`,
    `Source: ${String(S.streamSource || 'hardware')}`,
    `Sample rate: ${sampleRate.toFixed(2)} Hz`,
    `Stream state: ${S.streaming ? 'running' : 'idle'}`,
    `Decoder mode: ${S.decoderMode || 'n/a'}`,
    '',
    '## Signal Chain',
    `Base profile filters: ${getBaseProfileFilterChain()}`,
    `Active custom filter: ${activeFilter?.name ? `${activeFilter.name} (${activeFilter.apply_mode || 'append'})` : 'none'}`,
    '',
    '## Live Diagnostics',
    `Pipeline latency: ${Number(timing.process_last_ms || 0).toFixed(2)} ms last | ${Number(timing.process_avg_ms || 0).toFixed(2)} ms avg | ${Number(timing.process_max_ms || 0).toFixed(2)} ms max`,
    `Window timing: ${Number(timing.interval_avg_ms || 0).toFixed(2)} ms avg | ${Number(timing.interval_jitter_ms || 0).toFixed(2)} ms jitter | dropped ${Number(timing.dropped_windows || 0)}`,
    `Safety/watchdog: ${S.safety.enabled ? 'armed' : 'off'} | stale=${S.safety.stale ? 'yes' : 'no'} | signal age ${signalAge.toFixed(0)} ms | trips ${Number(S.safety.trip_count || 0)}`,
    `Noise snapshot: hum50 ${Number(noise.hum_50_db || 0).toFixed(1)} dB | hum60 ${Number(noise.hum_60_db || 0).toFixed(1)} dB | drift ${Number(noise.drift_db || 0).toFixed(1)} dB | clip ${Number(noise.clip_pct || 0).toFixed(2)} % | crest ${Number(noise.crest_factor || 0).toFixed(2)}`,
    '',
    '## Selected Filter Under Test',
    model ? `Name: ${summary.name || model.name || 'unnamed filter'}` : 'Name: none selected',
    model ? `Shape: ${summary.method || model.method} ${formatFilterCutoffSummary(model)} | order ${Number(summary.order || model.order || 0)}` : 'Shape: --',
    model ? `Sections: ${Number(response.sections || summary.sections || 0)} | stability: ${response.stable ? 'stable' : 'check poles'} | gain span ${Number(response.min_gain_db || 0).toFixed(2)} to ${Number(response.peak_gain_db || 0).toFixed(2)} dB` : 'Sections: --',
    '',
    '## Pole / Zero Snapshot',
    `Zeros: ${Array.isArray(response.zeros) ? response.zeros.length : 0} | Poles: ${Array.isArray(response.poles) ? response.poles.length : 0}`,
    response.poles?.length
      ? `Pole radii max: ${Math.max(...response.poles.map(item => Number(item.radius || 0))).toFixed(4)}`
      : 'Pole radii max: --',
    '',
    '## Fixed-Point Estimate',
    quant.coefficient_count ? `Coefficient count: ${Number(quant.coefficient_count || 0)} | state count: ${Number(quant.state_count || 0)}` : 'Coefficient count: --',
    quant.coefficient_count ? `Max |coeff|: ${Number(quant.max_abs_coeff || 0).toFixed(6)}` : 'Max |coeff|: --',
    quant.direct_q15 ? `Direct Q1.15: ${quant.direct_q15.fits ? 'fits' : 'needs scaling'} | overflow coeffs ${Number(quant.direct_q15.overflow_coefficients || 0)} | rms err ${Number(quant.direct_q15.rms_error || 0).toExponential(3)} | stable=${quant.direct_q15.stable ? 'yes' : 'no'}` : 'Direct Q1.15: --',
    quant.direct_q31 ? `Direct Q1.31: ${quant.direct_q31.fits ? 'fits' : 'needs scaling'} | overflow coeffs ${Number(quant.direct_q31.overflow_coefficients || 0)} | rms err ${Number(quant.direct_q31.rms_error || 0).toExponential(3)} | stable=${quant.direct_q31.stable ? 'yes' : 'no'}` : 'Direct Q1.31: --',
    quant.recommended_s16 ? `Recommended s16: ${formatFixedPointLabel(quant.recommended_s16)} | LSB ${Number(quant.recommended_s16.lsb || 0).toExponential(3)} | rms err ${Number(quant.recommended_s16.rms_error || 0).toExponential(3)} | stable=${quant.recommended_s16.stable ? 'yes' : 'no'}` : 'Recommended s16: --',
    quant.recommended_s32 ? `Recommended s32: ${formatFixedPointLabel(quant.recommended_s32)} | LSB ${Number(quant.recommended_s32.lsb || 0).toExponential(3)} | rms err ${Number(quant.recommended_s32.rms_error || 0).toExponential(3)} | stable=${quant.recommended_s32.stable ? 'yes' : 'no'}` : 'Recommended s32: --',
    quant.memory_bytes ? `Memory estimate: int16 ${Number(quant.memory_bytes.int16 || 0)} B | int32 ${Number(quant.memory_bytes.int32 || 0)} B | float32 ${Number(quant.memory_bytes.float32 || 0)} B` : 'Memory estimate: --',
    '',
    '## Notes',
    '- Custom filters run on the KYMA host path in this build; the Cyton is the acquisition front-end.',
    '- Arduino/C++ exports are coefficient reuse targets, not proof that the same runtime path is running on-board.',
  ];
  return lines.join('\n');
}

function refreshBenchReportUI() {
  const report = buildBenchReportText();
  const area = $('bench-report-text');
  if (area) area.value = report;

  const activeFilter = S.filterLab.active_filter || S.diagnostics.active_filter || null;
  $('bench-profile').textContent = `${S.signalProfileName} @ ${Number(S.sampleRate || 250).toFixed(0)} Hz`;
  $('bench-source').textContent = String(S.streamSource || 'hardware');
  $('bench-watchdog').textContent = S.safety.stale ? 'stale' : (S.safety.enabled ? 'armed' : 'off');
  $('bench-active-filter').textContent = activeFilter?.name || 'none';

  const model = getCurrentFilterModel();
  const quant = model?.response?.quantization || null;
  $('bench-quant').textContent = quant?.recommended_s16
    ? `${formatFixedPointLabel(quant.recommended_s16)} | ${quant.recommended_s32 ? formatFixedPointLabel(quant.recommended_s32) : '--'}`
    : '--';
  $('bench-status-note').textContent = model?.name
    ? `Report includes "${model.name}" plus the current live timing/noise snapshot.`
    : 'Report uses the current live timing/noise snapshot and active filter chain.';
}

function buildFilterSavedList() {
  const list = $('filter-saved-list');
  if (!list) return;
  const filters = S.filterLab.filters || [];
  if (!filters.length) {
    list.innerHTML = '<div class="filter-note">No saved filters yet.</div>';
    return;
  }
  list.innerHTML = filters.map(item => {
    const active = item.id === S.filterLab.active_filter_id;
    const selected = item.id === S.filterLab.selected_filter_id;
    return `
      <div class="filter-row ${selected ? 'active' : ''}" onclick="window.selectFilterRecord('${item.id}')">
        <div class="filter-row-title">
          <span>${item.name || item.id}</span>
          <span style="font-size:9px;color:${active ? 'var(--green)' : 'var(--text-dim)'}">${active ? 'ACTIVE' : item.profile_key.toUpperCase()}</span>
        </div>
        <div class="filter-row-meta">
          ${item.method} ${item.response_type} | order ${item.order} | ${item.apply_mode}<br>
          ${item.stable ? 'stable' : 'check poles'} | ${item.sections} SOS
        </div>
        <div style="display:flex;justify-content:flex-end;margin-top:6px">
          <button class="btn danger" style="width:auto;padding:2px 8px;font-size:10px" onclick="window.deleteFilterRecord('${item.id}', event)">Delete</button>
        </div>
      </div>
    `;
  }).join('');
}

function refreshFilterLabUI() {
  const model = getCurrentFilterModel();
  const summary = model?.summary || model || null;
  const response = model?.response || {};
  $('btn-filter-preview').disabled = !S.filterLab.available;
  $('btn-filter-save').disabled = !S.filterLab.available;
  $('btn-filter-activate').disabled = !S.filterLab.available || !S.filterLab.selected_filter_id;
  $('btn-filter-clear').disabled = !S.filterLab.available;
  $('filter-profile-note').textContent =
    S.filterLab.available
      ? `Design for ${S.signalProfileName} at ${Number(S.sampleRate || 250).toFixed(0)} Hz. Custom filters run on the KYMA host path for this build.`
      : (S.filterLab.last_error || 'SciPy filter design support is not available in this runtime.');
  $('filter-active-summary').textContent = S.filterLab.active_filter?.name
    ? `${S.filterLab.active_filter.name} | ${S.filterLab.active_filter.method} ${S.filterLab.active_filter.response_type} | ${S.filterLab.active_filter.apply_mode}`
    : 'No custom filter active for this profile.';
  $('filter-peak-gain').textContent = response.peak_gain_db !== undefined
    ? `${Number(response.peak_gain_db).toFixed(2)} dB`
    : '-- dB';
  $('filter-min-gain').textContent = response.min_gain_db !== undefined
    ? `${Number(response.min_gain_db).toFixed(2)} dB`
    : '-- dB';
  $('filter-sections').textContent = response.sections ?? summary?.sections ?? '--';
  $('filter-stability').textContent = response.stable === true ? 'stable' : (model ? 'check' : '--');
  $('filter-toolbar-title').textContent = model?.name || summary?.name
    ? `${summary?.name || model?.name} response preview`
    : 'Filter response preview';
  buildFilterSavedList();
  renderFilterResponseCanvas();
  renderFilterPoleZeroCanvas();
  renderFilterQuantization();
  renderFilterSOS();
  renderFilterExportPreview();
  refreshBenchReportUI();
  updateFilterFieldVisibility();
}

async function previewFilterDesign() {
  try {
    const res = await post('/api/filterlab/design', collectFilterSpec());
    S.filterLab.preview = res.preview || null;
    if (S.filterLab.preview?.id) {
      S.filterLab.records = { ...(S.filterLab.records || {}), [S.filterLab.preview.id]: S.filterLab.preview };
    }
    refreshFilterLabUI();
    toast('Filter preview ready');
  } catch (e) {
    toast(`Filter preview failed: ${e.message}`, 'red');
  }
}

async function loadFilterRecord(filterId, notify = true) {
  if (!filterId) {
    S.filterLab.selected_filter_id = '';
    S.filterLab.selected_filter = null;
    refreshFilterLabUI();
    return;
  }
  try {
    const item = await get(`/api/filterlab/${filterId}`);
    S.filterLab.selected_filter_id = filterId;
    S.filterLab.selected_filter = item;
    S.filterLab.records = { ...(S.filterLab.records || {}), [filterId]: item };
    S.filterLab.preview = null;
    refreshFilterLabUI();
    if (notify) toast(`Loaded ${item.name || filterId}`);
  } catch (e) {
    toast(`Filter load failed: ${e.message}`, 'red');
  }
}

window.selectFilterRecord = function(filterId) {
  loadFilterRecord(filterId);
};

window.deleteFilterRecord = async function(filterId, event) {
  if (event) event.stopPropagation();
  try {
    await post('/api/filterlab/delete', { filter_id: filterId, profile: S.signalProfileKey });
    if (S.filterLab.selected_filter_id === filterId) {
      S.filterLab.selected_filter_id = '';
      S.filterLab.selected_filter = null;
      S.filterLab.preview = null;
    }
    await loadFilterLabStatus();
    toast('Filter deleted');
  } catch (e) {
    toast(`Delete failed: ${e.message}`, 'red');
  }
};

async function saveFilterDesign() {
  try {
    const res = await post('/api/filterlab/save', collectFilterSpec());
    applyFilterLabStatus(res.filter_lab || {});
    S.filterLab.selected_filter_id = res.filter?.id || '';
    S.filterLab.selected_filter = res.filter || null;
    if (res.filter?.id) {
      S.filterLab.records = { ...(S.filterLab.records || {}), [res.filter.id]: res.filter };
    }
    S.filterLab.preview = null;
    if ($('filter-name') && res.filter?.name) $('filter-name').value = res.filter.name;
    refreshFilterLabUI();
    toast('Filter saved');
  } catch (e) {
    toast(`Save failed: ${e.message}`, 'red');
  }
}

async function activateSelectedFilter() {
  if (!S.filterLab.selected_filter_id) {
    toast('Select a saved filter first', 'yellow');
    return;
  }
  try {
    const res = await post('/api/filterlab/activate', {
      filter_id: S.filterLab.selected_filter_id,
      profile: S.signalProfileKey,
    });
    applyFilterLabStatus(res.filter_lab || {});
    refreshFilterLabUI();
    toast('Custom filter activated');
  } catch (e) {
    toast(`Activate failed: ${e.message}`, 'red');
  }
}

async function clearActiveFilter() {
  try {
    const res = await post('/api/filterlab/clear', { profile: S.signalProfileKey });
    applyFilterLabStatus(res.filter_lab || {});
    refreshFilterLabUI();
    toast('Custom filter cleared');
  } catch (e) {
    toast(`Clear failed: ${e.message}`, 'red');
  }
}

function openFilterExportModal() {
  const model = getCurrentFilterModel();
  const target = $('filter-export-target')?.value || 'kyma_host';
  if (!model?.exports?.[target]) {
    toast('No export available for the current selection', 'yellow');
    return;
  }
  const entry = model.exports[target];
  const output = $('code-output');
  if (output) output.value = entry.code || '';
  if ($('code-modal-title')) $('code-modal-title').textContent = entry.label || 'Generated Output';
  S.exportMeta = {
    filename: entry.filename || 'filter_export.txt',
    name: model.name || model.summary?.name || 'filter_export',
  };
  $('code-modal')?.classList.add('active');
}

function openBenchReportModal() {
  refreshBenchReportUI();
  const output = $('code-output');
  if (output) output.value = buildBenchReportText();
  if ($('code-modal-title')) $('code-modal-title').textContent = 'Bench Report Export';
  S.exportMeta = {
    filename: `kyma_bench_report_${S.signalProfileKey}.md`,
    name: `kyma_bench_report_${S.signalProfileKey}`,
  };
  $('code-modal')?.classList.add('active');
}

window.openBenchReportModal = openBenchReportModal;


// =============================================================================
// SOUND FEEDBACK
//
// plays a short beep using Web Audio API when gesture changes.
// each gesture gets a different pitch so you can tell them apart by ear.
// useful when you're looking at the arm, not the screen.
// =============================================================================

let _audioCtx;
function playBeep(gesture) {
  try {
    if (!_audioCtx) _audioCtx = new AudioContext();
    const freqs = { rest:220, open:330, close:440, pinch:550, point:660 };
    let freq = freqs[gesture];
    if (!freq) {
      let hash = 0;
      for (const ch of String(gesture || 'signal')) hash = ((hash << 5) - hash + ch.charCodeAt(0)) | 0;
      freq = 280 + (Math.abs(hash) % 8) * 55;
    }

    const osc = _audioCtx.createOscillator();
    const gain = _audioCtx.createGain();
    osc.connect(gain);
    gain.connect(_audioCtx.destination);

    osc.type = 'square';  // square wave = 8-bit retro sound
    osc.frequency.value = freq;
    gain.gain.value = 0.08;
    gain.gain.exponentialRampToValueAtTime(0.001, _audioCtx.currentTime + 0.15);

    osc.start();
    osc.stop(_audioCtx.currentTime + 0.15);
  } catch {}
}


// =============================================================================
// BLOCK PROGRAMMING EDITOR
//
// a visual drag-and-drop system for building custom Arduino action sequences.
// users compose programs from blocks (servo moves, delays, digital writes,
// loops, conditionals) and map EMG gestures to trigger them.
//
// this makes the EMG system generic — it's not just for a robotic arm,
// it can control anything: LED strips, relays, solenoids, motors, whatever.
// =============================================================================

// Block type definitions with ports for the node-graph editor
const BLOCK_TYPES = {
  // ── Entry point ──
  start:         { color:'#3fb950', label:'START',          category:'control', defaults:{},
    ports:[{id:'flow_out',kind:'flow',dir:'out',label:''}] },
  // ── Action blocks ──
  servo_move:    { color:'#58a6ff', label:'Servo Move',     category:'action', defaults:{ joint_id:0, angle:90 },
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'flow_out',kind:'flow',dir:'out',label:''}] },
  servo_sweep:   { color:'#58c6ff', label:'Servo Sweep',    category:'action', defaults:{ joint_id:0, from:30, to:150, step_ms:20 },
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'flow_out',kind:'flow',dir:'out',label:''}] },
  delay:         { color:'#d29922', label:'Delay',          category:'action', defaults:{ ms:500 },
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'flow_out',kind:'flow',dir:'out',label:''}] },
  digital_write: { color:'#3fb950', label:'Digital Write',  category:'action', defaults:{ pin:2, value:1 },
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'flow_out',kind:'flow',dir:'out',label:''}] },
  analog_write:  { color:'#bc8cff', label:'Analog Write',   category:'action', defaults:{ pin:3, value:128 },
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'flow_out',kind:'flow',dir:'out',label:''}] },
  tone:          { color:'#e6db74', label:'Play Tone',      category:'action', defaults:{ pin:8, freq:440, duration:200 },
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'flow_out',kind:'flow',dir:'out',label:''}] },
  saved_filter:  { color:'#7ee787', label:'Use Saved Filter', category:'action', defaults:{ filter_id:'', export_target:'fixed_point_header' },
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'flow_out',kind:'flow',dir:'out',label:''}] },
  gesture:       { color:'#f85149', label:'Run Arm Gesture', category:'preset', defaults:{ gesture:'open' },
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'flow_out',kind:'flow',dir:'out',label:''}] },
  set_3d:        { color:'#ff79c6', label:'3D Arm Pose',    category:'preset', defaults:{ j0:90,j1:90,j2:90,j3:90,j4:90,j5:90,j6:90,j7:90 },
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'flow_out',kind:'flow',dir:'out',label:''}] },
  log_msg:       { color:'#6272a4', label:'Log Message',    category:'action', defaults:{ msg:'Hello!' },
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'flow_out',kind:'flow',dir:'out',label:''}] },
  // ── Control blocks ──
  loop:          { color:'#ffa657', label:'Loop',           category:'control', defaults:{ count:3 },
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'body',kind:'flow',dir:'out',label:'body'},{id:'flow_out',kind:'flow',dir:'out',label:'done'}] },
  loop_forever:  { color:'#ff8c00', label:'Loop Forever',   category:'control', defaults:{},
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'body',kind:'flow',dir:'out',label:'body'}] },
  if_rms:        { color:'#79c0ff', label:'If Metric >',    category:'control', defaults:{ channel:0, threshold:0.2 },
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'true_out',kind:'flow',dir:'out',label:'true'},{id:'false_out',kind:'flow',dir:'out',label:'false'},{id:'flow_out',kind:'flow',dir:'out',label:'done'}] },
  if_gesture:    { color:'#50e3c2', label:'If Label',       category:'control', defaults:{ gesture:'open' },
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'true_out',kind:'flow',dir:'out',label:'yes'},{id:'false_out',kind:'flow',dir:'out',label:'no'},{id:'flow_out',kind:'flow',dir:'out',label:'done'}] },
  wait_gesture:  { color:'#50fa7b', label:'Wait Label',     category:'control', defaults:{ gesture:'close', timeout_s:10 },
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'flow_out',kind:'flow',dir:'out',label:''}] },
  sequence:      { color:'#8b949e', label:'Sequence',       category:'control', defaults:{},
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'flow_out',kind:'flow',dir:'out',label:''}] },
};

// ── Node-graph state ─────────────────────────────────────────────────────────

S.blockPrograms = JSON.parse(localStorage.getItem('emg-programs-v2') || '[]');
S.gestureMap = JSON.parse(localStorage.getItem('emg-gesture-map') || '{}');
S.activeProgram = S.blockPrograms.length ? S.blockPrograms[0].id : null;
S.executingProgram = false;
S.executionAbort = false;
S.selectedNodeId = null;
S.selectedWireId = null;
S._nodeIdCounter = Date.now();
S._connIdCounter = Date.now() + 100000;

function nextNodeId() { return 'n' + (S._nodeIdCounter++); }
function nextConnId() { return 'c' + (S._connIdCounter++); }

function savePrograms() {
  localStorage.setItem('emg-programs-v2', JSON.stringify(S.blockPrograms));
}
function saveGestureMap() {
  localStorage.setItem('emg-gesture-map', JSON.stringify(S.gestureMap));
}

function buildSignalExampleProgram() {
  const presetId = `signal_threshold_v1_${S.signalProfileKey || 'signal'}`;
  const existing = S.blockPrograms.find(p => p.preset === presetId);
  if (existing) return existing;

  const startNode = { id: nextNodeId(), type:'start', params:{}, x:80, y:180 };
  const ifNode = { id: nextNodeId(), type:'if_rms', params:{ channel:0, threshold:0.2 }, x:320, y:160 };
  const highNode = { id: nextNodeId(), type:'digital_write', params:{ pin:13, value:1 }, x:620, y:100 };
  const lowNode = { id: nextNodeId(), type:'digital_write', params:{ pin:13, value:0 }, x:620, y:240 };

  const prog = {
    id: 'p' + Date.now(),
    name: `${S.signalProfileName} Signal Example`,
    preset: presetId,
    signalProfile: S.signalProfileKey || '',
    nodes: {
      [startNode.id]: startNode,
      [ifNode.id]: ifNode,
      [highNode.id]: highNode,
      [lowNode.id]: lowNode,
    },
    connections: [
      { id: nextConnId(), fromNode: startNode.id, fromPort: 'flow_out', toNode: ifNode.id, toPort: 'flow_in' },
      { id: nextConnId(), fromNode: ifNode.id, fromPort: 'true_out', toNode: highNode.id, toPort: 'flow_in' },
      { id: nextConnId(), fromNode: ifNode.id, fromPort: 'false_out', toNode: lowNode.id, toPort: 'flow_in' },
    ],
    viewOffset: { x: 0, y: 0 },
    viewZoom: 1,
  };
  S.blockPrograms.push(prog);
  return prog;
}

window.loadSignalExampleProgram = function() {
  const prog = buildSignalExampleProgram();
  S.activeProgram = prog.id;
  savePrograms();
  refreshProgramSelect();
  renderCanvas();
  buildGestureMappingUI();
  toast(`Loaded signal example: ${prog.name}`);
};

// ── Program CRUD ─────────────────────────────────────────────────────────────

function getActiveProgram() {
  return S.blockPrograms.find(p => p.id === S.activeProgram) || null;
}
window.getActiveProgram = getActiveProgram;

window.newProgram = function() {
  const name = prompt('Program name:');
  if (!name) return;
  const startNode = { id: nextNodeId(), type:'start', params:{}, x:80, y:200 };
  const prog = { id:'p'+Date.now(), name, nodes:{ [startNode.id]:startNode }, connections:[], viewOffset:{x:0,y:0}, viewZoom:1 };
  S.blockPrograms.push(prog);
  S.activeProgram = prog.id;
  savePrograms();
  refreshProgramSelect();
  renderCanvas();
  toast(`Program "${name}" created`);
};

window.renameProgram = function() {
  const prog = getActiveProgram();
  if (!prog) { toast('Select a program first', 'red'); return; }
  const name = prompt('New name:', prog.name);
  if (!name) return;
  prog.name = name;
  savePrograms();
  refreshProgramSelect();
};

window.deleteProgram = function() {
  const prog = getActiveProgram();
  if (!prog) return;
  if (!confirm(`Delete "${prog.name}"?`)) return;
  S.blockPrograms = S.blockPrograms.filter(p => p.id !== prog.id);
  for (const g of Object.keys(S.gestureMap)) {
    if (S.gestureMap[g] === prog.id) delete S.gestureMap[g];
  }
  S.activeProgram = S.blockPrograms.length ? S.blockPrograms[0].id : null;
  savePrograms();
  saveGestureMap();
  refreshProgramSelect();
  renderCanvas();
  buildGestureMappingUI();
  toast('Program deleted');
};

function refreshProgramSelect() {
  const sel = $('program-select');
  if (!sel) return;
  sel.innerHTML = '';
  if (!S.blockPrograms.length) {
    sel.innerHTML = '<option value="">No programs</option>';
    return;
  }
  S.blockPrograms.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.id; opt.textContent = p.name;
    if (p.id === S.activeProgram) opt.selected = true;
    sel.appendChild(opt);
  });
  sel.onchange = () => {
    S.activeProgram = sel.value;
    renderCanvas();
  };
}

// ── Node creation ────────────────────────────────────────────────────────────

function createNode(type, x, y) {
  const def = BLOCK_TYPES[type];
  if (!def) return null;
  return {
    id: nextNodeId(),
    type,
    params: { ...def.defaults },
    x: x || 200,
    y: y || 200,
  };
}
window.createBlock = createNode; // back-compat

// ── Canvas rendering ────────────────────────────────────────────────────────

function renderCanvas() {
  const container = $('node-canvas-container');
  const canvas = $('node-canvas');
  if (!canvas || !container) return;

  // clear nodes (keep SVG)
  canvas.querySelectorAll('.node-block').forEach(n => n.remove());

  const prog = getActiveProgram();
  if (!prog) {
    const hint = document.createElement('div');
    hint.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:var(--text-dim);font-size:13px;text-align:center';
    hint.textContent = 'Create a program to get started';
    hint.className = 'node-block'; // reuse class for easy cleanup
    canvas.appendChild(hint);
    renderWires();
    return;
  }

  // ensure nodes object
  if (!prog.nodes) prog.nodes = {};
  if (!prog.connections) prog.connections = [];

  // apply pan/zoom
  const zoom = prog.viewZoom || 1;
  const ox = prog.viewOffset?.x || 0;
  const oy = prog.viewOffset?.y || 0;
  canvas.style.transform = `translate(${ox}px,${oy}px) scale(${zoom})`;

  const zi = $('zoom-indicator');
  if (zi) zi.textContent = Math.round(zoom * 100) + '%';

  // render each node
  for (const node of Object.values(prog.nodes)) {
    canvas.appendChild(renderNodeBlock(node, prog));
  }

  renderWires();
}
window.renderWorkspace = renderCanvas; // back-compat

function renderNodeBlock(node, prog) {
  const def = BLOCK_TYPES[node.type];
  if (!def) return document.createElement('div');

  const el = document.createElement('div');
  el.className = 'node-block';
  el.dataset.nodeId = node.id;
  el.style.left = node.x + 'px';
  el.style.top = node.y + 'px';

  if (node.id === S.selectedNodeId) el.classList.add('selected');

  // header
  const header = document.createElement('div');
  header.className = 'node-header';
  header.style.background = def.color;
  header.innerHTML = `<span>${def.label}</span>`;
  if (node.type !== 'start') {
    const del = document.createElement('span');
    del.className = 'node-delete';
    del.innerHTML = '&times;';
    del.onclick = (e) => {
      e.stopPropagation();
      // remove node and its connections
      delete prog.nodes[node.id];
      prog.connections = prog.connections.filter(c => c.fromNode !== node.id && c.toNode !== node.id);
      savePrograms();
      renderCanvas();
    };
    header.appendChild(del);
  }
  el.appendChild(header);

  // input ports
  const inPorts = (def.ports || []).filter(p => p.dir === 'in');
  if (inPorts.length) {
    const portsDiv = document.createElement('div');
    portsDiv.className = 'node-ports';
    inPorts.forEach(p => {
      const port = document.createElement('div');
      port.className = 'node-port port-in';
      port.dataset.portId = p.id;
      port.dataset.nodeId = node.id;
      const dot = document.createElement('span');
      dot.className = 'port-dot';
      // check if connected
      if (prog.connections.some(c => c.toNode === node.id && c.toPort === p.id)) dot.classList.add('connected');
      dot.dataset.portId = p.id;
      dot.dataset.nodeId = node.id;
      dot.dataset.dir = 'in';
      port.appendChild(dot);
      if (p.label) {
        const lbl = document.createElement('span');
        lbl.className = 'port-label';
        lbl.textContent = p.label;
        port.appendChild(lbl);
      }
      portsDiv.appendChild(port);
    });
    el.appendChild(portsDiv);
  }

  // params body
  if (node.type !== 'start') {
    const body = document.createElement('div');
    body.className = 'node-body';
    renderNodeParams(node, body);
    el.appendChild(body);
  }

  // output ports
  const outPorts = (def.ports || []).filter(p => p.dir === 'out');
  if (outPorts.length) {
    const portsDiv = document.createElement('div');
    portsDiv.className = 'node-ports';
    outPorts.forEach(p => {
      const port = document.createElement('div');
      port.className = 'node-port port-out';
      port.dataset.portId = p.id;
      port.dataset.nodeId = node.id;
      const dot = document.createElement('span');
      dot.className = 'port-dot';
      if (prog.connections.some(c => c.fromNode === node.id && c.fromPort === p.id)) dot.classList.add('connected');
      dot.dataset.portId = p.id;
      dot.dataset.nodeId = node.id;
      dot.dataset.dir = 'out';
      if (p.label) {
        const lbl = document.createElement('span');
        lbl.className = 'port-label';
        lbl.textContent = p.label;
        port.appendChild(lbl);
      }
      port.appendChild(dot);
      portsDiv.appendChild(port);
    });
    el.appendChild(portsDiv);
  }

  // click to select
  el.addEventListener('pointerdown', (e) => {
    if (e.target.classList.contains('port-dot')) return; // handled by port drag
    e.stopPropagation();
    S.selectedNodeId = node.id;
    S.selectedWireId = null;
    document.querySelectorAll('.node-block.selected').forEach(b => b.classList.remove('selected'));
    document.querySelectorAll('.wire-path.selected').forEach(w => w.classList.remove('selected'));
    el.classList.add('selected');
  });

  // node dragging (on header)
  initNodeDrag(header, node, el, prog);

  return el;
}

function renderNodeParams(node, container) {
  const p = node.params;
  const h = (label, type, key, extra='') => {
    const val = p[key] ?? '';
    if (type === 'select') return `<label>${label} ${extra}</label>`;
    return `<label>${label} <input type="${type}" value="${val}" data-key="${key}" style="${type==='text'?'width:80px':''}" ${extra}></label>`;
  };
  const jointSel = (key) => {
    let html = `<label>J <select data-key="${key}">`;
    for (let i = 0; i < 8; i++) html += `<option value="${i}" ${p[key]==i?'selected':''}>${i}</option>`;
    return html + '</select></label>';
  };
  const filterSelect = (key) => {
    const filters = S.filterLab.filters || [];
    let html = `<label>Filter <select data-key="${key}" data-role="filter-select">`;
    html += `<option value="" ${!p[key] ? 'selected' : ''}>Clear Active</option>`;
    filters.forEach(item => {
      const activeTag = item.id === S.filterLab.active_filter_id ? ' [ACTIVE]' : '';
      html += `<option value="${item.id}" ${p[key]===item.id?'selected':''}>${(item.name || item.id)}${activeTag}</option>`;
    });
    html += '</select></label>';
    if (!filters.length) {
      html += '<div style="font-size:9px;color:var(--text-dim)">No saved filters for this profile yet.</div>';
    }
    return html;
  };
  const filterExportSelect = (key) => {
    const options = [
      ['fixed_point_header', 'Fixed-Point'],
      ['arduino_filters', 'Arduino-Filters'],
      ['iir1_cpp', 'iir1 C++'],
      ['kyma_host', 'KYMA Host'],
    ];
    let html = `<label>Export <select data-key="${key}">`;
    options.forEach(([value, label]) => {
      html += `<option value="${value}" ${p[key]===value?'selected':''}>${label}</option>`;
    });
    return html + '</select></label>';
  };

  switch (node.type) {
    case 'servo_move':
      container.innerHTML = jointSel('joint_id') + h('Angle','number','angle');
      break;
    case 'servo_sweep':
      container.innerHTML = jointSel('joint_id') + h('From','number','from') + h('To','number','to') + h('ms','number','step_ms');
      break;
    case 'delay':
      container.innerHTML = h('ms','number','ms');
      break;
    case 'digital_write':
      container.innerHTML = h('Pin','number','pin') + h('Val','number','value');
      break;
    case 'analog_write':
      container.innerHTML = h('Pin','number','pin') + h('Val','number','value');
      break;
    case 'tone':
      container.innerHTML = h('Pin','number','pin') + h('Hz','number','freq') + h('ms','number','duration');
      break;
    case 'saved_filter':
      container.innerHTML = filterSelect('filter_id') + filterExportSelect('export_target');
      break;
    case 'gesture': {
      const gestures = ['rest','open','close','pinch','point'];
      let html = '<label>Gesture <select data-key="gesture">';
      gestures.forEach(g => html += `<option value="${g}" ${p.gesture===g?'selected':''}>${g}</option>`);
      container.innerHTML = html + '</select></label>';
      break;
    }
    case 'set_3d':
      container.innerHTML = Array.from({length:8}, (_,i) => h(`J${i}`,'number',`j${i}`)).join('');
      break;
    case 'log_msg':
      container.innerHTML = h('Msg','text','msg');
      break;
    case 'loop':
      container.innerHTML = h('Count','number','count');
      break;
    case 'loop_forever':
      container.innerHTML = '<span style="color:var(--text-dim)">∞ runs until stop</span>';
      break;
    case 'if_rms':
      container.innerHTML = h('Ch','number','channel') + h('>','number','threshold');
      break;
    case 'if_gesture': {
      const gestures = (S.gestures && S.gestures.length) ? S.gestures : ['rest','open','close','pinch','point'];
      let html = '<label>Gesture <select data-key="gesture">';
      gestures.forEach(g => html += `<option value="${g}" ${p.gesture===g?'selected':''}>${g}</option>`);
      container.innerHTML = html + '</select></label>';
      break;
    }
    case 'wait_gesture': {
      const gestures = (S.gestures && S.gestures.length) ? S.gestures : ['rest','open','close','pinch','point'];
      let html = '<label>Wait <select data-key="gesture">';
      gestures.forEach(g => html += `<option value="${g}" ${p.gesture===g?'selected':''}>${g}</option>`);
      container.innerHTML = html + '</select></label>' + h('Timeout(s)','number','timeout_s');
      break;
    }
    default:
      container.innerHTML = '';
  }

  // wire up param changes
  container.querySelectorAll('input, select').forEach(input => {
    const key = input.dataset.key;
    if (!key) return;
    input.addEventListener('change', async () => {
      const val = input.type === 'number' ? parseFloat(input.value) : input.value;
      p[key] = val;
      if (node.type === 'saved_filter' && key === 'filter_id' && val) {
        await ensureFilterRecord(String(val));
      }
      savePrograms();
    });
    // prevent drag when editing
    input.addEventListener('pointerdown', e => e.stopPropagation());
  });
}

// ── Wire rendering (SVG) ────────────────────────────────────────────────────

function renderWires() {
  const svg = document.getElementById('wire-svg');
  if (!svg) return;
  svg.innerHTML = '';

  const prog = getActiveProgram();
  if (!prog || !prog.connections) return;

  prog.connections.forEach(conn => {
    const path = createWirePath(conn, prog);
    if (path) {
      path.dataset.connId = conn.id;
      path.classList.add('wire-path');
      if (conn.id === S.selectedWireId) path.classList.add('selected');
      // click to select wire
      path.style.pointerEvents = 'stroke';
      path.addEventListener('click', (e) => {
        e.stopPropagation();
        S.selectedWireId = conn.id;
        S.selectedNodeId = null;
        document.querySelectorAll('.wire-path.selected').forEach(w => w.classList.remove('selected'));
        document.querySelectorAll('.node-block.selected').forEach(b => b.classList.remove('selected'));
        path.classList.add('selected');
      });
      svg.appendChild(path);
    }
  });
}

function createWirePath(conn, prog) {
  const fromDot = document.querySelector(`.port-dot[data-node-id="${conn.fromNode}"][data-port-id="${conn.fromPort}"][data-dir="out"]`);
  const toDot = document.querySelector(`.port-dot[data-node-id="${conn.toNode}"][data-port-id="${conn.toPort}"][data-dir="in"]`);
  if (!fromDot || !toDot) return null;

  const canvas = $('node-canvas');
  const canvasRect = canvas.getBoundingClientRect();
  const zoom = prog.viewZoom || 1;

  const fromRect = fromDot.getBoundingClientRect();
  const toRect = toDot.getBoundingClientRect();

  const x1 = (fromRect.left + fromRect.width/2 - canvasRect.left) / zoom;
  const y1 = (fromRect.top + fromRect.height/2 - canvasRect.top) / zoom;
  const x2 = (toRect.left + toRect.width/2 - canvasRect.left) / zoom;
  const y2 = (toRect.top + toRect.height/2 - canvasRect.top) / zoom;

  const dx = Math.max(Math.abs(x2 - x1) * 0.5, 40);
  const d = `M ${x1} ${y1} C ${x1+dx} ${y1}, ${x2-dx} ${y2}, ${x2} ${y2}`;

  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', d);
  // invisible wider hit area
  const hitPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  hitPath.setAttribute('d', d);
  hitPath.setAttribute('stroke', 'transparent');
  hitPath.setAttribute('stroke-width', '12');
  hitPath.setAttribute('fill', 'none');
  hitPath.style.pointerEvents = 'stroke';

  return path;
}

function getPortCenter(nodeId, portId, dir) {
  const dot = document.querySelector(`.port-dot[data-node-id="${nodeId}"][data-port-id="${portId}"][data-dir="${dir}"]`);
  if (!dot) return null;
  const canvas = $('node-canvas');
  const canvasRect = canvas.getBoundingClientRect();
  const prog = getActiveProgram();
  const zoom = prog?.viewZoom || 1;
  const r = dot.getBoundingClientRect();
  return {
    x: (r.left + r.width/2 - canvasRect.left) / zoom,
    y: (r.top + r.height/2 - canvasRect.top) / zoom,
  };
}

// ── Node dragging ───────────────────────────────────────────────────────────

function initNodeDrag(header, node, el, prog) {
  header.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    if (e.target.classList.contains('node-delete') || e.target.classList.contains('port-dot')) return;
    e.preventDefault();
    e.stopPropagation();

    const zoom = prog.viewZoom || 1;
    const startX = e.clientX, startY = e.clientY;
    const origX = node.x, origY = node.y;

    const onMove = (ev) => {
      node.x = origX + (ev.clientX - startX) / zoom;
      node.y = origY + (ev.clientY - startY) / zoom;
      el.style.left = node.x + 'px';
      el.style.top = node.y + 'px';
      renderWires();
    };

    const onUp = () => {
      document.removeEventListener('pointermove', onMove);
      document.removeEventListener('pointerup', onUp);
      savePrograms();
    };

    document.addEventListener('pointermove', onMove);
    document.addEventListener('pointerup', onUp);
  });
}

// ── Port connection dragging ────────────────────────────────────────────────

(function initPortDrag() {
  document.addEventListener('pointerdown', (e) => {
    const dot = e.target.closest('.port-dot');
    if (!dot) return;
    e.preventDefault();
    e.stopPropagation();

    const prog = getActiveProgram();
    if (!prog) return;

    const fromNodeId = dot.dataset.nodeId;
    const fromPortId = dot.dataset.portId;
    const fromDir = dot.dataset.dir;

    // only start connections from output ports
    if (fromDir !== 'out') return;

    const svg = document.getElementById('wire-svg');
    const canvas = $('node-canvas');
    const zoom = prog.viewZoom || 1;

    const tempPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    tempPath.classList.add('wire-temp');
    svg.appendChild(tempPath);

    const startPos = getPortCenter(fromNodeId, fromPortId, 'out');
    if (!startPos) { tempPath.remove(); return; }

    const onMove = (ev) => {
      const canvasRect = canvas.getBoundingClientRect();
      const mx = (ev.clientX - canvasRect.left) / zoom;
      const my = (ev.clientY - canvasRect.top) / zoom;
      const dx = Math.max(Math.abs(mx - startPos.x) * 0.5, 40);
      const d = `M ${startPos.x} ${startPos.y} C ${startPos.x+dx} ${startPos.y}, ${mx-dx} ${my}, ${mx} ${my}`;
      tempPath.setAttribute('d', d);

      // highlight potential target port
      document.querySelectorAll('.port-dot').forEach(d => d.style.transform = '');
      const target = document.elementFromPoint(ev.clientX, ev.clientY);
      if (target?.classList.contains('port-dot') && target.dataset.dir === 'in') {
        target.style.transform = 'scale(1.5)';
      }
    };

    const onUp = (ev) => {
      document.removeEventListener('pointermove', onMove);
      document.removeEventListener('pointerup', onUp);
      tempPath.remove();
      document.querySelectorAll('.port-dot').forEach(d => d.style.transform = '');

      // check if dropped on an input port
      const target = document.elementFromPoint(ev.clientX, ev.clientY);
      if (!target?.classList.contains('port-dot') || target.dataset.dir !== 'in') return;

      const toNodeId = target.dataset.nodeId;
      const toPortId = target.dataset.portId;

      // validate: no self-connection, no duplicate, input can only have one connection
      if (toNodeId === fromNodeId) return;
      if (prog.connections.some(c => c.toNode === toNodeId && c.toPort === toPortId)) {
        toast('Port already connected', 'red');
        return;
      }
      if (prog.connections.some(c => c.fromNode === fromNodeId && c.fromPort === fromPortId && c.toNode === toNodeId && c.toPort === toPortId)) return;

      prog.connections.push({
        id: nextConnId(),
        fromNode: fromNodeId, fromPort: fromPortId,
        toNode: toNodeId, toPort: toPortId,
      });
      savePrograms();
      renderCanvas();
      toast('Connected!');
    };

    document.addEventListener('pointermove', onMove);
    document.addEventListener('pointerup', onUp);
  });
})();

// ── Pan & Zoom ──────────────────────────────────────────────────────────────

(function initPanZoom() {
  const container = $('node-canvas-container');
  if (!container) return;

  let isPanning = false;
  let spaceHeld = false;

  // middle-click pan or space+left-click pan
  container.addEventListener('pointerdown', (e) => {
    if (e.target.closest('.node-block')) return;
    if (e.target.closest('.port-dot')) return;

    const prog = getActiveProgram();
    if (!prog) return;

    if (e.button === 1 || (e.button === 0 && spaceHeld) || e.button === 0) {
      e.preventDefault();
      isPanning = true;
      container.classList.add('panning');
      const startX = e.clientX, startY = e.clientY;
      const origOx = prog.viewOffset?.x || 0;
      const origOy = prog.viewOffset?.y || 0;

      const onMove = (ev) => {
        prog.viewOffset = prog.viewOffset || {x:0,y:0};
        prog.viewOffset.x = origOx + (ev.clientX - startX);
        prog.viewOffset.y = origOy + (ev.clientY - startY);
        const canvas = $('node-canvas');
        if (canvas) canvas.style.transform = `translate(${prog.viewOffset.x}px,${prog.viewOffset.y}px) scale(${prog.viewZoom||1})`;
      };

      const onUp = () => {
        isPanning = false;
        container.classList.remove('panning');
        document.removeEventListener('pointermove', onMove);
        document.removeEventListener('pointerup', onUp);
        savePrograms();
        renderWires();
      };

      document.addEventListener('pointermove', onMove);
      document.addEventListener('pointerup', onUp);
    }
  });

  // deselect when clicking empty canvas
  container.addEventListener('click', (e) => {
    if (!e.target.closest('.node-block') && !e.target.closest('.wire-path')) {
      S.selectedNodeId = null;
      S.selectedWireId = null;
      document.querySelectorAll('.node-block.selected').forEach(b => b.classList.remove('selected'));
      document.querySelectorAll('.wire-path.selected').forEach(w => w.classList.remove('selected'));
    }
  });

  // scroll wheel zoom
  container.addEventListener('wheel', (e) => {
    e.preventDefault();
    const prog = getActiveProgram();
    if (!prog) return;

    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    prog.viewZoom = Math.max(0.2, Math.min(3, (prog.viewZoom || 1) * delta));

    const canvas = $('node-canvas');
    if (canvas) canvas.style.transform = `translate(${prog.viewOffset?.x||0}px,${prog.viewOffset?.y||0}px) scale(${prog.viewZoom})`;

    const zi = $('zoom-indicator');
    if (zi) zi.textContent = Math.round(prog.viewZoom * 100) + '%';

    savePrograms();
    // debounce wire re-render
    clearTimeout(container._wireTimer);
    container._wireTimer = setTimeout(renderWires, 100);
  }, { passive: false });

  // space key for pan mode
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !e.target.matches('input,textarea,select')) {
      spaceHeld = true;
      container.style.cursor = 'grab';
    }
    // Delete key to remove selected node or wire
    if ((e.code === 'Delete' || e.code === 'Backspace') && !e.target.matches('input,textarea,select')) {
      const prog = getActiveProgram();
      if (!prog) return;

      if (S.selectedWireId) {
        prog.connections = prog.connections.filter(c => c.id !== S.selectedWireId);
        S.selectedWireId = null;
        savePrograms();
        renderCanvas();
        toast('Wire deleted');
      } else if (S.selectedNodeId) {
        const node = prog.nodes[S.selectedNodeId];
        if (node && node.type !== 'start') {
          delete prog.nodes[S.selectedNodeId];
          prog.connections = prog.connections.filter(c => c.fromNode !== S.selectedNodeId && c.toNode !== S.selectedNodeId);
          S.selectedNodeId = null;
          savePrograms();
          renderCanvas();
          toast('Block deleted');
        }
      }
    }
  });
  document.addEventListener('keyup', (e) => {
    if (e.code === 'Space') {
      spaceHeld = false;
      container.style.cursor = '';
    }
  });
})();

// ── Palette drag to canvas ──────────────────────────────────────────────────

(function initPaletteDragToCanvas() {
  document.querySelectorAll('.palette-block').forEach(el => {
    el.draggable = false;

    // Double-click: instant add at auto position
    el.addEventListener('dblclick', (e) => {
      e.preventDefault();
      const type = el.dataset.type;
      if (!type) return;
      const prog = getActiveProgram();
      if (!prog) { toast('Create a program first', 'red'); return; }
      // find good position
      const nodeCount = Object.keys(prog.nodes).length;
      const x = 100 + (nodeCount % 4) * 200;
      const y = 80 + Math.floor(nodeCount / 4) * 120;
      const node = createNode(type, x, y);
      if (!node) return;
      prog.nodes[node.id] = node;
      savePrograms();
      renderCanvas();
      toast(`Added: ${BLOCK_TYPES[type]?.label || type}`);
    });

    // Pointer drag to canvas
    el.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      const type = el.dataset.type;
      if (!type) return;

      const startX = e.clientX, startY = e.clientY;
      let ghost = null;
      let started = false;

      const onMove = (ev) => {
        if (!started && Math.abs(ev.clientX - startX) + Math.abs(ev.clientY - startY) < 6) return;
        if (!started) {
          started = true;
          ghost = document.createElement('div');
          ghost.className = 'drag-ghost';
          const icon = el.querySelector('.block-icon');
          if (icon) ghost.appendChild(icon.cloneNode(true));
          ghost.appendChild(document.createTextNode(el.textContent.trim()));
          document.body.appendChild(ghost);
        }
        ghost.style.left = (ev.clientX + 12) + 'px';
        ghost.style.top = (ev.clientY + 12) + 'px';

        const container = $('node-canvas-container');
        if (container) {
          const r = container.getBoundingClientRect();
          if (ev.clientX >= r.left && ev.clientX <= r.right && ev.clientY >= r.top && ev.clientY <= r.bottom) {
            container.classList.add('drag-hover');
          } else {
            container.classList.remove('drag-hover');
          }
        }
      };

      const onUp = (ev) => {
        document.removeEventListener('pointermove', onMove);
        document.removeEventListener('pointerup', onUp);
        if (ghost) ghost.remove();
        const container = $('node-canvas-container');
        if (container) container.classList.remove('drag-hover');
        if (!started) return;

        const prog = getActiveProgram();
        if (!prog) return;

        if (!container) return;
        const r = container.getBoundingClientRect();
        if (ev.clientX < r.left || ev.clientX > r.right || ev.clientY < r.top || ev.clientY > r.bottom) return;

        const zoom = prog.viewZoom || 1;
        const ox = prog.viewOffset?.x || 0;
        const oy = prog.viewOffset?.y || 0;
        const x = (ev.clientX - r.left - ox) / zoom;
        const y = (ev.clientY - r.top - oy) / zoom;

        const node = createNode(type, x, y);
        if (!node) return;
        prog.nodes[node.id] = node;
        savePrograms();
        renderCanvas();
        toast(`Added: ${BLOCK_TYPES[type]?.label || type}`);
      };

      document.addEventListener('pointermove', onMove);
      document.addEventListener('pointerup', onUp);
    });
  });
})();

// ── Execution engine (graph-based) ──────────────────────────────────────────

window.runProgram = async function() {
  const prog = getActiveProgram();
  if (!prog) { toast('Select a program first', 'red'); return; }
  if (S.executingProgram) { toast('Already running', 'red'); return; }

  S.executingProgram = true;
  S.executionAbort = false;
  execLog('--- Running: ' + prog.name + ' ---', 'run');

  try {
    const startNode = Object.values(prog.nodes).find(n => n.type === 'start');
    if (!startNode) throw new Error('No Start block');
    await followFlow(prog, startNode.id, 'flow_out');
    execLog('--- Done ---', 'ok');
  } catch (err) {
    execLog('Error: ' + err.message, 'err');
  }
  S.executingProgram = false;
  document.querySelectorAll('.node-block.running').forEach(b => b.classList.remove('running'));
};

window.stopProgram = function() {
  S.executionAbort = true;
  S.executingProgram = false;
  execLog('--- Stopped ---', 'err');
};

async function followFlow(prog, fromNodeId, fromPortId) {
  if (S.executionAbort) throw new Error('Aborted');
  const conn = prog.connections.find(c => c.fromNode === fromNodeId && c.fromPort === fromPortId);
  if (!conn) return;
  const node = prog.nodes[conn.toNode];
  if (!node) return;
  await executeNode(prog, node);
}

async function executeNode(prog, node) {
  if (S.executionAbort) throw new Error('Aborted');

  // highlight
  const el = document.querySelector(`.node-block[data-node-id="${node.id}"]`);
  if (el) el.classList.add('running');

  const p = node.params;
  try {
    switch (node.type) {
      case 'servo_move':
        execLog(`Servo J${p.joint_id} -> ${p.angle}°`);
        await post('/api/move', { joint_id: p.joint_id, angle: p.angle });
        if (window._arm3d) window._arm3d.setJoint(p.joint_id, p.angle);
        break;

      case 'servo_sweep': {
        const from = p.from, to = p.to, step = p.step_ms || 20;
        const dir = from < to ? 1 : -1;
        for (let a = from; dir > 0 ? a <= to : a >= to; a += dir) {
          if (S.executionAbort) throw new Error('Aborted');
          await post('/api/move', { joint_id: p.joint_id, angle: a });
          if (window._arm3d) window._arm3d.setJoint(p.joint_id, a);
          await cancellableDelay(step);
        }
        execLog(`Sweep J${p.joint_id}: ${from}->${to}`);
        break;
      }

      case 'delay':
        execLog(`Delay ${p.ms}ms`);
        await cancellableDelay(p.ms);
        break;

      case 'digital_write':
        execLog(`Digital pin ${p.pin} = ${p.value}`);
        await post('/api/digital_write', { pin: p.pin, value: p.value });
        break;

      case 'analog_write':
        execLog(`Analog pin ${p.pin} = ${p.value}`);
        await post('/api/analog_write', { pin: p.pin, value: p.value });
        break;

      case 'tone':
        execLog(`Tone pin ${p.pin}: ${p.freq}Hz ${p.duration}ms`);
        await post('/api/digital_write', { pin: p.pin, value: 1 });
        await cancellableDelay(p.duration);
        break;

      case 'saved_filter':
        if (p.filter_id) {
          execLog(`Use saved filter: ${getFilterSummaryById(p.filter_id)?.name || p.filter_id}`);
          const res = await post('/api/filterlab/activate', {
            filter_id: p.filter_id,
            profile: S.signalProfileKey,
          });
          applyFilterLabStatus(res.filter_lab || {});
          await ensureFilterRecord(String(p.filter_id));
        } else {
          execLog('Clear active custom filter');
          const res = await post('/api/filterlab/clear', { profile: S.signalProfileKey });
          applyFilterLabStatus(res.filter_lab || {});
        }
        refreshFilterLabUI();
        break;

      case 'gesture':
        execLog(`Gesture: ${p.gesture}`);
        await post('/api/gesture', { gesture: p.gesture });
        HandView.setGesture(p.gesture);
        if (window._arm3d) window._arm3d.setGesture(p.gesture);
        break;

      case 'set_3d':
        if (window._arm3d) {
          const angles = Array.from({length:8}, (_,i) => p[`j${i}`] || 90);
          window._arm3d.setAngles(angles);
          execLog('3D pose set');
        }
        break;

      case 'log_msg':
        execLog(p.msg || '');
        break;

      case 'loop':
        for (let i = 0; i < (p.count || 1); i++) {
          if (S.executionAbort) throw new Error('Aborted');
          execLog(`Loop ${i+1}/${p.count}`);
          await followFlow(prog, node.id, 'body');
        }
        break;

      case 'loop_forever':
        for (let i = 0; !S.executionAbort; i++) {
          execLog(`Loop forever #${i+1}`);
          await followFlow(prog, node.id, 'body');
        }
        break;

      case 'if_rms': {
        const rawMetric = Number(S.rms?.[p.channel] || 0);
        const scale = Math.max(Number(S.signalMetricScale || 1), 1e-6);
        const metricNorm = Math.max(0, Math.min(1, rawMetric / scale));
        if (metricNorm > (p.threshold || 0)) {
          execLog(`Metric ch${p.channel} norm=${metricNorm.toFixed(3)} raw=${rawMetric.toFixed(3)} > ${p.threshold} -> true`);
          await followFlow(prog, node.id, 'true_out');
        } else {
          execLog(`Metric ch${p.channel} norm=${metricNorm.toFixed(3)} raw=${rawMetric.toFixed(3)} <= ${p.threshold} -> false`);
          await followFlow(prog, node.id, 'false_out');
        }
        break;
      }

      case 'if_gesture': {
        const current = S.lastGesture || '';
        if (current === p.gesture) {
          execLog(`Gesture is "${p.gesture}" → yes`);
          await followFlow(prog, node.id, 'true_out');
        } else {
          execLog(`Gesture is "${current}" ≠ "${p.gesture}" → no`);
          await followFlow(prog, node.id, 'false_out');
        }
        break;
      }

      case 'wait_gesture': {
        execLog(`Waiting for gesture "${p.gesture}"...`);
        const timeout = (p.timeout_s || 10) * 1000;
        const start = Date.now();
        while (S.lastGesture !== p.gesture) {
          if (S.executionAbort) throw new Error('Aborted');
          if (Date.now() - start > timeout) { execLog('Timeout', 'err'); break; }
          await cancellableDelay(100);
        }
        break;
      }

      case 'sequence':
        // sequence just passes flow through
        break;
    }
  } finally {
    if (el) el.classList.remove('running');
  }

  // follow the main flow_out (or 'done' port for branching blocks)
  const def = BLOCK_TYPES[node.type];
  const mainOut = (def.ports || []).find(p => p.dir === 'out' && (p.id === 'flow_out'));
  if (mainOut) {
    await followFlow(prog, node.id, mainOut.id);
  }
}

function cancellableDelay(ms) {
  return new Promise(resolve => {
    const check = setInterval(() => { if (S.executionAbort) { clearInterval(check); resolve(); } }, 50);
    setTimeout(() => { clearInterval(check); resolve(); }, ms);
  });
}

function execLog(msg, cls = '') {
  const log = $('exec-log');
  if (!log) return;
  const line = document.createElement('div');
  line.className = 'exec-line' + (cls ? ' ' + cls : '');
  line.textContent = msg;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}

window.clearExecLog = function() {
  const log = $('exec-log');
  if (log) log.innerHTML = 'Ready.';
};

// ── Arduino code generation ─────────────────────────────────────────────────

function generateArduinoCode() {
  const prog = getActiveProgram();
  if (!prog) return '// No program selected';

  const startNode = Object.values(prog.nodes).find(n => n.type === 'start');
  if (!startNode) return '// No Start block found';

  const includes = new Set();
  const globals = [];
  const setupLines = [];
  const loopLines = [];
  const usedServos = new Set();
  const usedPins = new Set();

  // pre-scan all nodes
  for (const node of Object.values(prog.nodes)) {
    if (node.type === 'servo_move' || node.type === 'servo_sweep') usedServos.add(node.params.joint_id);
    if (node.type === 'digital_write' || node.type === 'analog_write') usedPins.add(node.params.pin);
    if (node.type === 'tone') usedPins.add(node.params.pin);
  }

  if (usedServos.size) {
    includes.add('#include <Servo.h>');
    const servoPins = [2,3,4,5,6,7,8,9];
    usedServos.forEach(id => {
      globals.push(`Servo servo_${id};`);
      setupLines.push(`  servo_${id}.attach(${servoPins[id] || (id+2)});`);
    });
  }
  usedPins.forEach(pin => setupLines.push(`  pinMode(${pin}, OUTPUT);`));

  const visited = new Set();

  function walkFlow(fromNodeId, fromPortId, indent) {
    const conn = prog.connections.find(c => c.fromNode === fromNodeId && c.fromPort === fromPortId);
    if (!conn || visited.has(conn.id)) return;
    visited.add(conn.id);
    const node = prog.nodes[conn.toNode];
    if (!node) return;
    genNode(node, indent);
  }

  function genNode(node, indent) {
    const p = node.params;
    const pad = '  '.repeat(indent);

    switch (node.type) {
      case 'servo_move':
        loopLines.push(`${pad}servo_${p.joint_id}.write(${p.angle});`);
        walkFlow(node.id, 'flow_out', indent);
        break;
      case 'servo_sweep':
        loopLines.push(`${pad}for (int a = ${p.from}; a ${p.from<p.to?'<=':'>='}  ${p.to}; a${p.from<p.to?'++':'--'}) {`);
        loopLines.push(`${pad}  servo_${p.joint_id}.write(a);`);
        loopLines.push(`${pad}  delay(${p.step_ms || 20});`);
        loopLines.push(`${pad}}`);
        walkFlow(node.id, 'flow_out', indent);
        break;
      case 'delay':
        loopLines.push(`${pad}delay(${p.ms});`);
        walkFlow(node.id, 'flow_out', indent);
        break;
      case 'digital_write':
        loopLines.push(`${pad}digitalWrite(${p.pin}, ${p.value ? 'HIGH' : 'LOW'});`);
        walkFlow(node.id, 'flow_out', indent);
        break;
      case 'analog_write':
        loopLines.push(`${pad}analogWrite(${p.pin}, ${p.value});`);
        walkFlow(node.id, 'flow_out', indent);
        break;
      case 'tone':
        loopLines.push(`${pad}tone(${p.pin}, ${p.freq}, ${p.duration});`);
        loopLines.push(`${pad}delay(${p.duration});`);
        walkFlow(node.id, 'flow_out', indent);
        break;
      case 'saved_filter': {
        if (!p.filter_id) {
          loopLines.push(`${pad}// Clear active KYMA host filter (runtime-only; not compiled into standalone Arduino code)`);
        } else {
          const filterInfo = getCachedFilterRecord(p.filter_id) || getFilterSummaryById(p.filter_id);
          const filterName = String(filterInfo?.name || p.filter_id).replace(/"/g, '\\"');
          const target = String(p.export_target || 'fixed_point_header');
          const exportInfo = filterInfo?.exports?.[target] || null;
          loopLines.push(`${pad}// Use saved filter "${filterName}" in KYMA host runtime`);
          loopLines.push(`${pad}// Embedded target hint: ${target}${exportInfo?.filename ? ` -> ${exportInfo.filename}` : ''}`);
          loopLines.push(`${pad}// This generated sketch does not auto-apply KYMA host filters on-board.`);
        }
        walkFlow(node.id, 'flow_out', indent);
        break;
      }
      case 'log_msg':
        loopLines.push(`${pad}Serial.println("${(p.msg||'').replace(/"/g,'\\"')}");`);
        walkFlow(node.id, 'flow_out', indent);
        break;
      case 'loop':
        loopLines.push(`${pad}for (int i = 0; i < ${p.count || 1}; i++) {`);
        walkFlow(node.id, 'body', indent + 1);
        loopLines.push(`${pad}}`);
        walkFlow(node.id, 'flow_out', indent);
        break;
      case 'loop_forever':
        loopLines.push(`${pad}while (true) {`);
        walkFlow(node.id, 'body', indent + 1);
        loopLines.push(`${pad}}`);
        break;
      case 'if_rms':
        loopLines.push(`${pad}if (analogRead(A${p.channel}) > ${Math.round((p.threshold||0)*1023)}) {`);
        walkFlow(node.id, 'true_out', indent + 1);
        loopLines.push(`${pad}} else {`);
        walkFlow(node.id, 'false_out', indent + 1);
        loopLines.push(`${pad}}`);
        walkFlow(node.id, 'flow_out', indent);
        break;
      case 'gesture':
      case 'set_3d':
        loopLines.push(`${pad}// ${BLOCK_TYPES[node.type]?.label || node.type} (KYMA server / arm demo only)`);
        walkFlow(node.id, 'flow_out', indent);
        break;
      case 'if_gesture':
        loopLines.push(`${pad}// If label "${p.gesture}" (requires live decoder)`);
        loopLines.push(`${pad}// true branch:`);
        walkFlow(node.id, 'true_out', indent);
        loopLines.push(`${pad}// false branch:`);
        walkFlow(node.id, 'false_out', indent);
        walkFlow(node.id, 'flow_out', indent);
        break;
      case 'wait_gesture':
        loopLines.push(`${pad}// Wait for label "${p.gesture}" (requires live decoder)`);
        loopLines.push(`${pad}delay(${(p.timeout_s||10)*1000}); // placeholder`);
        walkFlow(node.id, 'flow_out', indent);
        break;
      case 'sequence':
        walkFlow(node.id, 'flow_out', indent);
        break;
    }
  }

  walkFlow(startNode.id, 'flow_out', 1);

  const code = [
    '// Auto-generated by KYMA Block Editor',
    '// Program: ' + prog.name,
    '',
    ...[...includes],
    '',
    ...globals,
    '',
    'void setup() {',
    '  Serial.begin(115200);',
    ...setupLines,
    '}',
    '',
    'void loop() {',
    ...(loopLines.length ? loopLines : ['  // No blocks connected to Start']),
    '}',
    '',
  ].join('\n');

  return code;
}

window.exportArduinoCode = function() {
  const code = generateArduinoCode();
  const output = $('code-output');
  if (output) output.value = code;
  if ($('code-modal-title')) $('code-modal-title').textContent = 'Generated Arduino Code';
  const prog = getActiveProgram();
  S.exportMeta = {
    filename: ((prog?.name || 'program').replace(/[^a-zA-Z0-9_]/g, '_') || 'program') + '.ino',
    name: prog?.name || 'program',
  };
  const modal = $('code-modal');
  if (modal) modal.classList.add('active');
};

window.closeCodeModal = function() {
  const modal = $('code-modal');
  if (modal) modal.classList.remove('active');
};

window.copyExportedCode = function() {
  const output = $('code-output');
  if (output) {
    navigator.clipboard.writeText(output.value);
    toast('Code copied to clipboard!');
  }
};

window.downloadExportedCode = function() {
  const code = $('code-output')?.value || '';
  const fallback = (getActiveProgram()?.name || 'program').replace(/[^a-zA-Z0-9_]/g, '_');
  const name = (S.exportMeta?.filename || `${fallback}.ino`).replace(/\.(ino|hpp|py|txt)$/i, '');
  const filename = S.exportMeta?.filename || `${name}.ino`;
  const blob = new Blob([code], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
  toast('Downloaded ' + filename);
};

// ── Gesture mapping UI ──────────────────────────────────────────────────────

function buildGestureMappingUI() {
  const list = $('gesture-mapping-list');
  if (!list) return;
  list.innerHTML = '';

  S.gestures.forEach(g => {
    const row = document.createElement('div');
    row.className = 'gmap-row';
    const opts = S.blockPrograms.map(p =>
      `<option value="${p.id}" ${S.gestureMap[g]===p.id?'selected':''}>${p.name}</option>`
    ).join('');
    row.innerHTML = `
      <span class="gmap-name">${g}</span>
      <select onchange="setGestureMapping('${g}', this.value)">
        <option value="">(none)</option>
        ${opts}
      </select>`;
    list.appendChild(row);
  });
}

window.setGestureMapping = function(gesture, programId) {
  if (programId) S.gestureMap[gesture] = programId;
  else delete S.gestureMap[gesture];
  saveGestureMap();
};

// ── Collapsible cards ────────────────────────────────────────────────────────

function initCollapsibleCards() {
  const storageVersion = 'v3';
  document.querySelectorAll('.card-collapsible').forEach(card => {
    const header = card.querySelector('h3');
    if (!header) return;

    const key = card.dataset.collapseKey
      || header.textContent.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-');
    const storageKey = `kyma-card-collapse:${storageVersion}:${key}`;
    const saved = localStorage.getItem(storageKey);

    if (saved === '1') card.classList.add('collapsed');
    else if (saved === '0') card.classList.remove('collapsed');

    const syncState = () => {
      const expanded = !card.classList.contains('collapsed');
      header.setAttribute('aria-expanded', String(expanded));
      localStorage.setItem(storageKey, expanded ? '0' : '1');
    };

    header.setAttribute('role', 'button');
    header.setAttribute('tabindex', '0');

    if (header.dataset.collapseBound !== '1') {
      const toggle = () => {
        card.classList.toggle('collapsed');
        syncState();
      };
      header.addEventListener('click', toggle);
      header.addEventListener('keydown', ev => {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault();
          toggle();
        }
      });
      header.dataset.collapseBound = '1';
    }

    syncState();
  });
}

initCollapsibleCards();

// ── Tab switching ────────────────────────────────────────────────────────────

window.switchTab = function(tab) {
  const main = $('main');
  const editor = $('block-editor');
  const filterLab = $('filter-lab');
  const workshop = $('signal-workshop');
  const benchReport = $('bench-report');
  const hwdocs = $('hw-docs');
  const tabDash = $('tab-dashboard');
  const tabBlocks = $('tab-blocks');
  const tabFilters = $('tab-filters');
  const tabWorkshop = $('tab-workshop');
  const tabBench = $('tab-bench');
  const tabHwdocs = $('tab-hwdocs');

  // Hide all
  main.classList.add('hidden');
  editor.classList.remove('active');
  filterLab.classList.remove('active');
  workshop.classList.remove('active');
  benchReport.classList.remove('active');
  hwdocs.classList.remove('active');
  tabDash.classList.remove('active');
  tabBlocks.classList.remove('active');
  tabFilters.classList.remove('active');
  tabWorkshop.classList.remove('active');
  tabBench.classList.remove('active');
  tabHwdocs.classList.remove('active');

  if (tab === 'blocks') {
    editor.classList.add('active');
    tabBlocks.classList.add('active');
    refreshProgramSelect();
    renderCanvas();
    buildGestureMappingUI();
  } else if (tab === 'filters') {
    filterLab.classList.add('active');
    tabFilters.classList.add('active');
    refreshFilterLabUI();
  } else if (tab === 'workshop') {
    workshop.classList.add('active');
    tabWorkshop.classList.add('active');
    syncWorkshopUI();
  } else if (tab === 'bench') {
    benchReport.classList.add('active');
    tabBench.classList.add('active');
    refreshBenchReportUI();
  } else if (tab === 'hwdocs') {
    hwdocs.classList.add('active');
    tabHwdocs.classList.add('active');
  } else {
    main.classList.remove('hidden');
    tabDash.classList.add('active');
  }
};

// ── Hook into EMG predictions for gesture-triggered programs ────────────────

function checkGestureProgramMapping(gestureName) {
  const progId = S.gestureMap[gestureName];
  if (progId && !S.executingProgram) {
    const prog = S.blockPrograms.find(p => p.id === progId);
    if (prog) {
      execLog(`[Signal] Label "${gestureName}" -> "${prog.name}"`);
      S.executingProgram = true;
      S.executionAbort = false;
      const startNode = Object.values(prog.nodes).find(n => n.type === 'start');
      if (startNode) {
        followFlow(prog, startNode.id, 'flow_out')
          .then(() => { S.executingProgram = false; })
          .catch(() => { S.executingProgram = false; });
      }
    }
  }
}
// =============================================================================
// RESIZABLE PANELS
// =============================================================================
(function initResizePanels() {
  document.querySelectorAll('.resize-handle-h').forEach(handle => {
    handle.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      handle.classList.add('active');
      handle.setPointerCapture(e.pointerId);
      document.body.classList.add('resizing');

      const kind = handle.dataset.resize;
      const parent = handle.parentElement;

      const onMove = (ev) => {
        const rect = parent.getBoundingClientRect();
        const x = ev.clientX - rect.left;
        const w = rect.width;
        const children = [...parent.children].filter(c => !c.classList.contains('resize-handle-h'));
        const leftPanel = children[0];
        const rightPanel = children[children.length - 1];

        if (kind === 'main-left' || kind === 'blocks-left') {
          const newW = Math.max(140, Math.min(w * 0.4, x));
          leftPanel.style.width = newW + 'px';
          parent.style.gridTemplateColumns =
            newW + 'px 5px 1fr 5px ' + (rightPanel.style.width || (kind === 'main-left' ? '280px' : '260px'));
        } else if (kind === 'main-right' || kind === 'blocks-right') {
          const newW = Math.max(140, Math.min(w * 0.4, w - x));
          rightPanel.style.width = newW + 'px';
          parent.style.gridTemplateColumns =
            (leftPanel.style.width || (kind === 'main-right' ? '225px' : '225px')) + ' 5px 1fr 5px ' + newW + 'px';
        }
      };

      const onUp = () => {
        handle.classList.remove('active');
        document.body.classList.remove('resizing');
        handle.removeEventListener('pointermove', onMove);
        handle.removeEventListener('pointerup', onUp);
        // trigger canvas / 3D resize
        if (typeof resizeCanvas === 'function') resizeCanvas();
        if (window._arm3d) window._arm3d.resize();
      };

      handle.addEventListener('pointermove', onMove);
      handle.addEventListener('pointerup', onUp);
    });
  });

  // Vertical resize between EMG chart and 3D arm
  document.querySelectorAll('.resize-handle-v').forEach(handle => {
    handle.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      handle.classList.add('active');
      handle.setPointerCapture(e.pointerId);
      document.body.classList.add('resizing-v');

      const center = handle.parentElement;
      const chartContainer = $('canvas-container');
      const armSection = $('arm-section');
      if (!chartContainer || !armSection) return;

      const onMove = (ev) => {
        const rect = center.getBoundingClientRect();
        const y = ev.clientY - rect.top;
        // Account for emg-header and channel-legend heights
        const headerH = $('emg-header')?.offsetHeight || 30;
        const legendH = document.getElementById('channel-legend')?.offsetHeight || 25;
        const totalH = rect.height;
        const chartH = Math.max(100, Math.min(totalH - 150, y - headerH));
        const armH = Math.max(100, totalH - chartH - headerH - legendH - 40);
        chartContainer.style.flex = 'none';
        chartContainer.style.height = chartH + 'px';
        armSection.style.flex = 'none';
        armSection.style.height = armH + 'px';
      };

      const onUp = () => {
        handle.classList.remove('active');
        document.body.classList.remove('resizing-v');
        handle.removeEventListener('pointermove', onMove);
        handle.removeEventListener('pointerup', onUp);
        if (typeof resizeCanvas === 'function') resizeCanvas();
        if (window._arm3d) window._arm3d.resize();
      };

      handle.addEventListener('pointermove', onMove);
      handle.addEventListener('pointerup', onUp);
    });
  });
})();


// =============================================================================
// GO
// =============================================================================
boot();
