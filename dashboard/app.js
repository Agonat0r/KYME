/**
 * app.js — KYMA dashboard
 *
 * this file does everything on the frontend:
 *   - websocket connection to the python server
 *   - live 8-channel EMG waveform renderer (canvas 2d)
 *   - 3D robotic hand digital twin (three.js)
 *   - theme system (neutral / dark)
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
// Two built-in themes only: neutral and dark.
// We still write vars onto :root directly so the runtime palette is
// authoritative and doesn't depend on stylesheet ordering.
// =============================================================================

const THEME_VARS = {
  neutral: {
    '--bg':'#ede5d8','--surface':'rgba(255, 252, 248, 0.76)','--surface-strong':'rgba(255,255,255,0.88)',
    '--surface-soft':'rgba(250,246,240,0.52)','--border':'rgba(160,120,80,0.14)',
    '--accent':'#d97756','--green':'#4d8f72','--yellow':'#a87d28',
    '--red':'#b85464','--text':'#2a2018','--text-dim':'#a8907c',
    '--glass-border':'rgba(220,180,148,0.32)','--glass-fill-strong':'rgba(255,255,255,0.62)',
    '--glass-fill-mid':'rgba(250,246,240,0.36)','--glass-fill-soft':'rgba(255,252,248,0.16)',
    '--glass-shadow':'0 20px 56px rgba(100,70,40,0.13), 0 6px 18px rgba(80,50,30,0.08)',
    '--radius':'18px','--font':"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  },
  dark: {
    '--bg':'#0d1118','--surface':'rgba(15, 21, 32, 0.78)','--border':'rgba(133, 148, 173, 0.22)',
    '--accent':'#7aa2ff','--green':'#5bb98c','--yellow':'#d6b35f',
    '--red':'#e0788c','--text':'#edf2ff','--text-dim':'#94a0b8',
    '--glass-border':'rgba(164, 186, 255, 0.16)','--glass-fill-strong':'rgba(18, 24, 35, 0.90)',
    '--glass-fill-mid':'rgba(12, 17, 27, 0.76)','--glass-fill-soft':'rgba(7, 11, 18, 0.56)',
    '--glass-shadow':'0 24px 60px rgba(0, 0, 0, 0.42), 0 10px 24px rgba(0, 0, 0, 0.24)',
    '--radius':'18px','--font':"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  },
};

// canvas-specific colors (can't use css vars in canvas api)
const THEME_CANVAS = {
  neutral: { bg:'#fffefc', grid:'rgba(160,120,80,0.08)', alt:'rgba(217,119,86,0.025)' },
  dark: { bg:'#0a111a', grid:'rgba(148, 160, 184, 0.14)', alt:'rgba(122, 162, 255, 0.05)' },
};

// per-channel waveform colors
const CH_COLORS = {
  neutral: ['#d97756','#4d8f72','#b85464','#a87d28','#6b4cb8','#2e8a8a','#c4783c','#5868b8'],
  dark: ['#7aa2ff','#5bb98c','#e0788c','#d6b35f','#9c89ff','#58c6cf','#f39a64','#88a5df'],
};

const THEME_SCENE_BG = {
  neutral: 0xfaf5ee,
  dark: 0x0b111a,
};

const LEGACY_THEME_CLASSES = ['retro', 'mario', 'gameboy', 'cyberpunk'];
let currentTheme = 'neutral';

function detectGlassEngine() {
  const brands = Array.isArray(navigator.userAgentData?.brands)
    ? navigator.userAgentData.brands.map(entry => entry.brand).join(' ')
    : '';
  const ua = `${brands} ${navigator.userAgent || ''}`;
  const chromium = /(Chrome|Chromium|Edg|OPR)/i.test(ua) && !/Firefox/i.test(ua);
  document.documentElement.dataset.glassEngine = chromium ? 'chromium' : 'fallback';
}

detectGlassEngine();

function initGlassMotion() {
  if (window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches) return;
  const root = document.documentElement;
  let raf = 0;
  const state = { x: 0, y: 0 };
  const target = { x: 0, y: 0 };

  function commit() {
    raf = 0;
    state.x += (target.x - state.x) * 0.14;
    state.y += (target.y - state.y) * 0.14;

    const panX = `${(state.x * 26).toFixed(2)}px`;
    const panY = `${(state.y * 18).toFixed(2)}px`;
    const rotX = `${(-state.y * 7.2).toFixed(2)}deg`;
    const rotY = `${(state.x * 8.8).toFixed(2)}deg`;
    const lightX = `${(50 + state.x * 32).toFixed(2)}%`;
    const lightY = `${(10 + state.y * 22).toFixed(2)}%`;

    root.style.setProperty('--glass-pan-x', panX);
    root.style.setProperty('--glass-pan-y', panY);
    root.style.setProperty('--glass-rot-x', rotX);
    root.style.setProperty('--glass-rot-y', rotY);
    root.style.setProperty('--glass-light-x', lightX);
    root.style.setProperty('--glass-light-y', lightY);

    if (Math.abs(target.x - state.x) > 0.001 || Math.abs(target.y - state.y) > 0.001) {
      raf = requestAnimationFrame(commit);
    }
  }

  function schedule() {
    if (!raf) raf = requestAnimationFrame(commit);
  }

  window.addEventListener('pointermove', event => {
    const w = Math.max(window.innerWidth || 1, 1);
    const h = Math.max(window.innerHeight || 1, 1);
    target.x = ((event.clientX / w) - 0.5) * 2;
    target.y = ((event.clientY / h) - 0.5) * 2;
    schedule();
  }, { passive: true });

  window.addEventListener('pointerleave', () => {
    target.x = 0;
    target.y = 0;
    schedule();
  }, { passive: true });

  window.addEventListener('deviceorientation', event => {
    if (!Number.isFinite(event.gamma) && !Number.isFinite(event.beta)) return;
    target.x = Math.max(-1, Math.min(1, Number(event.gamma || 0) / 14));
    target.y = Math.max(-1, Math.min(1, Number(event.beta || 0) / 18));
    schedule();
  }, { passive: true });
}

initGlassMotion();

function applyTheme(name = currentTheme) {
  currentTheme = Object.prototype.hasOwnProperty.call(THEME_VARS, name) ? name : 'neutral';
  const vars = THEME_VARS[currentTheme];

  // slam the vars directly onto :root so everything picks them up
  const root = document.documentElement;
  for (const [prop, val] of Object.entries(vars)) {
    root.style.setProperty(prop, val);
  }

  root.style.colorScheme = currentTheme === 'dark' ? 'dark' : 'light';
  LEGACY_THEME_CLASSES.forEach(theme => document.body.classList.remove(theme));
  document.body.classList.toggle('dark-mode', currentTheme === 'dark');
  document.body.style.fontSize = '14px';
  try { localStorage.setItem('emg-theme', currentTheme); } catch {}

  const sel = document.getElementById('theme-select');
  if (sel) sel.value = currentTheme;

  // tell the 3d arm about the theme change
  if (window._arm3d) window._arm3d.setTheme(currentTheme);
}

function chColors() { return CH_COLORS[currentTheme] || CH_COLORS.neutral; }


// =============================================================================
// CONFIG + STATE
// =============================================================================

const API = location.origin;
const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
const DISPLAY_SAMPLES = 500;  // how many samples to show in the rolling waveform
const REVIEW_ARCHIVE_SAMPLES = 600000; // frozen review history (~40 min at 250 Hz)
const N_CH = 8;               // cyton has 8 channels
const REVIEW_X_ZOOM_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
const REVIEW_Y_ZOOM_LEVELS = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
const EXTERNAL_SCRIPT_PROMISES = new Map();
const BABYLON_CORE_URL = 'https://cdn.babylonjs.com/babylon.js';
const BABYLON_LOADERS_URL = 'https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js';
const THREE_CORE_URL = 'https://cdn.jsdelivr.net/npm/three@0.149.0/build/three.min.js';

function loadExternalScript(src) {
  if (!src) return Promise.reject(new Error('Missing script URL'));
  if (EXTERNAL_SCRIPT_PROMISES.has(src)) return EXTERNAL_SCRIPT_PROMISES.get(src);
  const promise = new Promise((resolve, reject) => {
    const existing = Array.from(document.scripts || []).find((tag) => tag.src === src);
    if (existing && existing.dataset.loaded === '1') {
      resolve();
      return;
    }
    const tag = existing || document.createElement('script');
    let settled = false;
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      fn(value);
    };
    tag.src = src;
    tag.async = true;
    tag.onload = () => {
      tag.dataset.loaded = '1';
      finish(resolve);
    };
    tag.onerror = () => finish(reject, new Error(`Failed to load script: ${src}`));
    if (!existing) document.head.appendChild(tag);
  });
  EXTERNAL_SCRIPT_PROMISES.set(src, promise);
  return promise;
}

async function ensureBabylonLoaded() {
  if (typeof BABYLON !== 'undefined') return true;
  await loadExternalScript(BABYLON_CORE_URL);
  await loadExternalScript(BABYLON_LOADERS_URL);
  return typeof BABYLON !== 'undefined';
}

async function ensureThreeLoaded() {
  if (typeof THREE !== 'undefined') return true;
  await loadExternalScript(THREE_CORE_URL);
  return typeof THREE !== 'undefined';
}
const AI_REASONING_STEPS = [
  { line: 'Reading live diagnostics and review context.', stage: 'scan' },
  { line: 'Scoring signal quality and artifact pressure.', stage: 'score' },
  { line: 'Drafting operator actions and marker ideas.', stage: 'plan' },
  { line: 'Preparing export and handoff suggestions.', stage: 'route' },
];
const SIGNAL_LOGIC_STORAGE_KEY = 'kyma-signal-logic-rules-v1';
const SIGNAL_LOGIC_ENABLED_KEY = 'kyma-signal-logic-enabled';
const SIGNAL_LOGIC_FEED_MAX = 18;

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
  reviewArchive: Array.from({length:N_CH}, () => new Float32Array(REVIEW_ARCHIVE_SAMPLES)),
  reviewArchiveHead: 0,
  reviewArchiveTotal: 0,
  frames: 0,
  fpsTime: performance.now(),
  rms: new Array(N_CH).fill(0),
  lastQuality: new Array(N_CH).fill(0),
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
  controlConfidenceThreshold: 0.55,
  controlSensitivityTimer: 0,
  supportsTraining: true,
  supportsArmGestures: true,
  lieDetector: {
    status: null,
    baselineTimer: 0,
    baselineRemaining: 0,
    baselineRunning: false,
  },
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
  protocolRunner: {
    active: false,
    phase: 'idle',
    step: null,
    endsAt: 0,
    tickId: 0,
    token: 0,
    runId: '',
    templateKey: '',
  },
  eegExperiments: [],
  selectedEegExperiment: '',
  dashboardWorkspace: localStorage.getItem('kyma-dashboard-workspace') || 'live',
  activeViz: localStorage.getItem('kyma-active-viz') || 'none',
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
    diagnosticsSnapshot: null,
    aiResultSnapshot: null,
    artifactSnapshot: null,
    qualitySnapshot: null,
    fatigueSnapshot: null,
    selection: null,
    markers: [],
    markerTool: false,
    cursors: { a: null, b: null },
    nextCursorSlot: 'a',
    dragging: false,
    draggingCursor: null,
    dragOriginX: 0,
    hoverSample: null,
    hoverChannel: null,
    viewCenterSample: null,
    zoomX: 1,
    zoomY: 1,
    liveAutoScale: false,
    lastAutoScaleAt: 0,
    lastStats: null,
    artifacts: [],
    showOverlays: false,
  },
  guidedLabeling: {
    active: false,
    prompt: '',
    labels: [],
    regions: [],
    activeRegionKey: '',
    activeRegionStartSample: 0,
  },
  liveMl: null,
  liveMlSeenAt: 0,
  signalHealth: {
    warnings: [],
    flatChannels: [],
    duplicateChannels: [],
    activeChannels: [],
    autoHiddenChannels: [],
    updatedAt: 0,
  },
  promptModel: {
    prediction: null,
    server: null,
    seenAt: 0,
    regions: [],
    maxRegions: 96,
  },
  scopeOverlays: {
    envelope: localStorage.getItem('kyma-scope-envelope') !== '0',
    thresholds: localStorage.getItem('kyma-scope-thresholds') !== '0',
  },
  channelEnabled: new Array(N_CH).fill(true),
  autoChannelVisibility: localStorage.getItem('kyma-auto-channel-visibility') !== '0',
  autoChannelLastMask: '',
  autoChannelLastAt: 0,
  manualChannelOverrideUntil: 0,
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
    saving: false,
    exporting: false,
    lastRequest: null,
    lastSaved: null,
  },
  firmware: {
    root_dir: '',
    generated_dir: '',
    arduino_cli: { available: false, path: '', note: '' },
    files: [],
    selectedPath: '',
    selectedKind: '',
    content: '',
    dirty: false,
    loading: false,
    saving: false,
    actionOutput: '',
    fqbn: localStorage.getItem('kyma-firmware-fqbn') || 'arduino:avr:uno',
    port: localStorage.getItem('kyma-firmware-port') || '',
    lastCompile: null,
  },
  ai: {
    available: true,
    configured: false,
    mode: 'heuristic',
    provider: 'local',
    model: '',
    base_url: '',
    backend_label: '',
    credential_source: 'none',
    config_source: 'none',
    config_saved: false,
    key_hint: '',
    storage_path: '',
    loading: false,
    last_error: '',
    result: null,
    activeTab: localStorage.getItem('kyma-ai-tab') || 'summary',
    thinkingStep: 0,
    thinkingTimer: 0,
    thinkingStartedAt: 0,
    spotlight: null,
    regions: [],
    background: false,
    lastSignature: '',
    changedPulseUntil: 0,
    expandUntil: 0,
    lastAutoRunAt: 0,
    autoEveryMs: 7000,
    lensMinimized: localStorage.getItem('kyma-ai-lens-minimized') !== '0',
    lensX: Number(localStorage.getItem('kyma-ai-lens-x')),
    lensY: Number(localStorage.getItem('kyma-ai-lens-y')),
    lensDragging: false,
    lensDragOffsetX: 0,
    lensDragOffsetY: 0,
    lensWidth: Number(localStorage.getItem('kyma-ai-lens-width')),
    lensHeight: Number(localStorage.getItem('kyma-ai-lens-height')),
    lensResizing: false,
    lensResizeStartX: 0,
    lensResizeStartY: 0,
    lensResizeStartW: 0,
    lensResizeStartH: 0,
    localModels: {
      torch_available: false,
      model_count: 0,
      tasks: [],
      models: [],
      last_error: '',
    },
  },
  logic: {
    enabled: localStorage.getItem(SIGNAL_LOGIC_ENABLED_KEY) !== '0',
    rules: [],
    runtime: {},
    feed: [],
    liveSnapshot: null,
    drawerOpen: localStorage.getItem('kyma-signal-logic-drawer') === '1',
  },
  copilot: {
    input: '',
    status: 'Ask for a channel, noise pass, or cleanup filter.',
    steps: [],
    actions: [],
    findings: [],
    focusChannel: -1,
    focusMix: 0,
    focusTarget: 0,
    focusLabel: '',
    lastCommand: '',
  },
  pipelineBuilder: {
    open: false,
    autopilotLoading: false,
    autopilot: null,
    createLoading: false,
    createStage: '',
    loading: false,
    qaLoading: false,
    trainLoading: false,
    acquireLoading: false,
    acquisitionRunLoading: false,
    acquisitionTrainLoading: false,
    labelsLoading: false,
    datasetLoading: false,
    ingestLoading: false,
    compareLoading: false,
    exportLoading: false,
    runtimeLoading: false,
    deployLoading: false,
    embeddingLoading: false,
    throughputLoading: false,
    modelsLoading: false,
    modelActionLoading: false,
    exportsLoading: false,
    exportActionLoading: false,
    projectsLoading: false,
    projectActionLoading: false,
    prompt: '',
    source: '',
    output: localStorage.getItem('kyma-pipeline-output') || 'live_model',
    datasetPath: localStorage.getItem('kyma-pipeline-dataset-path') || '',
    schemaMapping: null,
    recipeDraft: null,
    plan: null,
    qa: null,
    trainResult: null,
    acquisition: null,
    acquisitionRun: null,
    labelSuggestions: null,
    dataset: null,
    ingest: null,
    comparison: null,
    exportResult: null,
    runtime: null,
    deploy: null,
    embedding: null,
    throughput: null,
    modelManager: null,
    exportManager: null,
    projectRegistry: null,
    activeProject: null,
    rebuildJob: null,
    rebuildProgress: [],
    guidedTaskIndex: 0,
    guidedProtocol: null,
    sourceChoiceConfirmed: false,
    appType: '',
    appChoiceConfirmed: false,
    codeWorkbenchOpen: false,
    codeWorkbench: {
      files: [],
      selectedPath: '',
      logs: [],
      previewHTML: '',
      key: '',
      mode: 'code',
      typedPath: '',
      typedChars: 0,
    },
    registryFilters: {
      query: '',
      status: 'all',
      format: 'all',
    },
    error: '',
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
const COMMAND_PALETTE = {
  open: false,
  query: '',
  index: 0,
  items: [],
};


function captureReviewSnapshot() {
  const useArchive = Number(S.reviewArchiveTotal || 0) > 0;
  S.review.snapshot = {
    emg: (useArchive ? S.reviewArchive : S.emg).map(buf => Float32Array.from(buf)),
    head: useArchive ? S.reviewArchiveHead : S.emgHead,
    total: useArchive ? S.reviewArchiveTotal : S.emgTotal,
    rms: Array.isArray(S.rms) ? S.rms.slice() : [],
    sampleRate: Number(S.sampleRate || 250),
    capturedAt: Date.now(),
    capacity: useArchive ? REVIEW_ARCHIVE_SAMPLES : DISPLAY_SAMPLES,
  };
}

function resetSignalBuffers() {
  S.emgHead = 0;
  S.emgTotal = 0;
  S.reviewArchiveHead = 0;
  S.reviewArchiveTotal = 0;
  for (let ch = 0; ch < N_CH; ch += 1) {
    S.emg[ch].fill(0);
    S.reviewArchive[ch].fill(0);
  }
}

function clonePlainData(value) {
  if (value == null) return null;
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return null;
  }
}

function clonePredictionPayload(payload) {
  if (!payload) return null;
  return {
    ...payload,
    arousal: payload.arousal ? clonePlainData(payload.arousal) : null,
    confidence: Number(payload.confidence || 0),
    label: payload.label || payload.gesture || '--',
    gesture: payload.gesture || payload.label || '--',
    summary: payload.summary || '',
  };
}

function renderArousalPanel(arousal) {
  const panel = $('arousal-panel');
  if (!panel) return;
  if (!arousal) {
    panel.style.display = 'block';
    if ($('arousal-level')) $('arousal-level').textContent = '--';
    const bar = $('arousal-bar');
    if (bar) {
      bar.style.width = '0%';
      bar.style.background = 'var(--green)';
    }
    if ($('arousal-detail')) {
      $('arousal-detail').textContent = 'Waiting for ECG or EDA/GSR metrics.';
    }
    if ($('arousal-note')) {
      $('arousal-note').textContent = 'Physiological arousal only. This does not detect truth or deception.';
    }
    syncLieDetectorUI();
    return;
  }
  const score = Math.max(0, Math.min(100, Number(arousal.score || 0)));
  const level = String(arousal.level || 'baseline');
  const drivers = Array.isArray(arousal.drivers) ? arousal.drivers.filter(Boolean) : [];
  panel.style.display = 'block';
  if ($('arousal-level')) $('arousal-level').textContent = `${level} ${score.toFixed(0)}`;
  const bar = $('arousal-bar');
  if (bar) {
    bar.style.width = `${score}%`;
    bar.style.background = score >= 70 ? 'var(--red)'
      : score >= 42 ? 'var(--yellow)'
        : score >= 18 ? 'var(--accent)' : 'var(--green)';
  }
  if ($('arousal-detail')) {
    $('arousal-detail').textContent = drivers.length
      ? drivers.slice(0, 2).join(' | ')
      : 'No arousal driver available yet.';
  }
  if ($('arousal-note')) {
    $('arousal-note').textContent = arousal.disclaimer || 'Physiological arousal only. This does not detect truth or deception.';
  }
  syncLieDetectorUI();
}

function applyLieDetectorStatus(status) {
  S.lieDetector.status = status || null;
  syncLieDetectorUI();
}

function syncLieDetectorUI() {
  const panel = $('lie-panel');
  if (!panel) return;
  const status = S.lieDetector.status || {};
  const result = status.last_result || {};
  const ready = !!status.ready;
  const baseline = Number(status.baseline_count || 0);
  const truth = Number(status.truth_count || 0);
  const lie = Number(status.lie_count || 0);
  const label = result.label || (ready ? 'Prototype: ready' : 'Prototype: training');
  const confidence = result.confidence != null
    ? `${Math.round(Number(result.confidence || 0) * 100)}% confidence`
    : (ready ? 'Ready to score target questions' : 'Needs controls');

  if ($('lie-result-label')) $('lie-result-label').textContent = label;
  if ($('lie-result-confidence')) $('lie-result-confidence').textContent = confidence;
  if ($('lie-training-status')) {
    const baselineText = S.lieDetector.baselineRunning
      ? `baseline ${S.lieDetector.baselineRemaining}s left`
      : `baseline ${baseline}/5`;
    $('lie-training-status').textContent = `${baselineText} | truth ${truth}/2 | lie ${lie}/2`;
  }
  if ($('lie-result-reason')) {
    $('lie-result-reason').textContent = result.reason || status.disclaimer || 'Prototype only: compares this person against their own controls.';
  }
  if ($('btn-lie-baseline')) {
    $('btn-lie-baseline').textContent = S.lieDetector.baselineRunning
      ? `Training ${S.lieDetector.baselineRemaining}s`
      : 'Train Baseline';
    $('btn-lie-baseline').disabled = S.lieDetector.baselineRunning;
  }
  if ($('btn-lie-score')) $('btn-lie-score').disabled = !ready && !status.active_question;
}

async function refreshLieDetectorStatus() {
  try {
    const status = await get('/api/lie_detector/status');
    applyLieDetectorStatus(status);
  } catch {}
}

async function sampleLieDetector(kind) {
  try {
    const status = await post(`/api/lie_detector/sample/${kind}`, {});
    applyLieDetectorStatus(status);
    const labels = { baseline: 'Baseline sample saved', truth: 'Known-truth sample saved', lie: 'Known-false sample saved' };
    toast(labels[kind] || 'Sample saved');
  } catch (e) {
    toast(e.message, 'red');
  }
}

async function trainLieBaseline() {
  if (S.lieDetector.baselineRunning) return;
  S.lieDetector.baselineRunning = true;
  S.lieDetector.baselineRemaining = 30;
  syncLieDetectorUI();
  toast('Baseline training started. Keep the hand still.');
  await sampleLieDetector('baseline');
  S.lieDetector.baselineTimer = setInterval(async () => {
    S.lieDetector.baselineRemaining -= 1;
    if (S.lieDetector.baselineRemaining > 0 && S.lieDetector.baselineRemaining % 3 === 0) {
      await sampleLieDetector('baseline');
    }
    if (S.lieDetector.baselineRemaining <= 0) {
      clearInterval(S.lieDetector.baselineTimer);
      S.lieDetector.baselineTimer = 0;
      S.lieDetector.baselineRunning = false;
      await refreshLieDetectorStatus();
      toast('Baseline training complete');
    }
    syncLieDetectorUI();
  }, 1000);
}

async function resetLieDetector() {
  if (S.lieDetector.baselineTimer) clearInterval(S.lieDetector.baselineTimer);
  S.lieDetector.baselineTimer = 0;
  S.lieDetector.baselineRunning = false;
  S.lieDetector.baselineRemaining = 0;
  try {
    const status = await post('/api/lie_detector/reset', {});
    applyLieDetectorStatus(status);
    toast('Prototype lie detector reset');
  } catch (e) {
    toast(e.message, 'red');
  }
}

async function startLieQuestion() {
  const question = $('lie-question-input')?.value || '';
  try {
    const status = await post('/api/lie_detector/question/start', { question });
    applyLieDetectorStatus(status);
    toast('Question window started');
  } catch (e) {
    toast(e.message, 'red');
  }
}

async function scoreLieQuestion() {
  const answer = $('lie-answer-input')?.value || '';
  try {
    const status = await post('/api/lie_detector/question/score', { answer });
    applyLieDetectorStatus(status);
    toast(`Prototype result: ${status.result?.label || 'Inconclusive'}`);
  } catch (e) {
    toast(e.message, 'red');
  }
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
  renderArousalPanel(data.arousal || null);
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

function clearReviewCursors() {
  S.review.cursors = { a: null, b: null };
  S.review.draggingCursor = null;
  S.review.nextCursorSlot = 'a';
}

function resetReviewState({ clearMarkers = false } = {}) {
  S.review.paused = false;
  S.review.snapshot = null;
  S.review.timelineSnapshot = null;
  S.review.predictionSnapshot = null;
  S.review.eegBrainSnapshot = null;
  S.review.diagnosticsSnapshot = null;
  S.review.aiResultSnapshot = null;
  S.review.artifactSnapshot = null;
  S.review.dragging = false;
  S.review.draggingCursor = null;
  S.review.hoverSample = null;
  S.review.hoverChannel = null;
  S.review.viewCenterSample = null;
  S.review.zoomX = 1;
  S.review.zoomY = 1;
  S.review.liveAutoScale = false;
  S.review.lastAutoScaleAt = 0;
  if (clearMarkers) S.review.markers = [];
  clearReviewCursors();
  clearReviewSelection();
  syncReviewUI();
  syncWorkshopUI();
  syncAICopilotUI();
}

function getReviewRenderState() {
  const paused = !!S.review.paused && !!S.review.snapshot;
  const source = paused ? S.review.snapshot : {
    emg: S.emg,
    head: S.emgHead,
    total: S.emgTotal,
    sampleRate: Number(S.sampleRate || 250),
    capacity: DISPLAY_SAMPLES,
  };
  const capacity = Math.max(1, Number(source.capacity || DISPLAY_SAMPLES));
  const filled = Math.min(Number(source.total || 0), capacity);
  const baseAbs = Math.max(0, Number(source.total || 0) - filled);
  return {
    paused,
    emg: source.emg,
    head: Number(source.head || 0),
    total: Number(source.total || 0),
    capacity,
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

function resolveReviewAnchorSample(state) {
  const stats = S.review.lastStats;
  if (stats) return Math.round((Number(stats.startSample || 0) + Number(stats.endSample || 0)) / 2);
  if (Number.isFinite(S.review.viewCenterSample)) return Number(S.review.viewCenterSample);
  if (Number.isFinite(S.review.hoverSample)) return Number(S.review.hoverSample);
  return state.baseAbs + Math.floor(Math.max(state.filled - 1, 0) / 2);
}

function getReviewViewport(state, width) {
  const filled = Math.max(state?.filled || 0, 0);
  const fullStart = Number(state?.baseAbs || 0);
  const fullEnd = fullStart + Math.max(filled - 1, 0);

  if (!filled) {
    return {
      drawStart: width,
      visibleWidth: 1,
      fullStart,
      fullEnd,
      viewStart: fullStart,
      viewEnd: fullStart,
      viewSamples: 0,
    };
  }

  const zoomX = Math.max(1, Number(S.review.zoomX || 1));
  if (!state.paused) {
    const viewSamples = Math.max(1, Math.min(filled, Math.round(filled / zoomX)));
    const viewEnd = fullEnd;
    const viewStart = Math.max(fullStart, viewEnd - viewSamples + 1);
    return {
      drawStart: 0,
      visibleWidth: Math.max(1, width),
      fullStart,
      fullEnd,
      viewStart,
      viewEnd,
      viewSamples,
    };
  }

  const viewSamples = Math.max(1, Math.min(filled, Math.round(filled / zoomX)));
  const anchor = clamp(Math.round(resolveReviewAnchorSample(state)), fullStart, fullEnd);
  const maxStart = Math.max(fullStart, fullEnd - viewSamples + 1);
  const viewStart = clamp(anchor - Math.floor((viewSamples - 1) / 2), fullStart, maxStart);
  const viewEnd = Math.min(fullEnd, viewStart + viewSamples - 1);
  return {
    drawStart: 0,
    visibleWidth: Math.max(1, width),
    fullStart,
    fullEnd,
    viewStart,
    viewEnd,
    viewSamples,
  };
}

function getPlaybackDisplayFullScale(state = getReviewRenderState()) {
  const base = Math.max(Number(S.signalFullScale || 200), 1e-6);
  if (S.streamSource !== 'playback' || state?.paused || !state?.filled) return base;
  const viewport = getReviewViewport(state, canvas.width || canvas.clientWidth || 1);
  const start = Number(viewport.viewStart || state.baseAbs || 0);
  const end = Number(viewport.viewEnd || start);
  let peak = 0;
  for (let ch = 0; ch < Math.min(N_CH, state.emg?.length || 0); ch += 1) {
    if (S.channelEnabled[ch] === false) continue;
    const buf = state.emg?.[ch];
    if (!buf) continue;
    for (let sample = start; sample <= end; sample += 1) {
      const idx = bufferIndexForAbsSample(state, sample);
      peak = Math.max(peak, Math.abs(Number(buf[idx] || 0)));
    }
  }
  if (!(peak > 0)) return base;
  return clamp(Math.max(peak * 1.15, Number(S.muteFloor || 0.5) * 0.25, 0.05), 0.05, base);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function bufferIndexForAbsSample(state, absSample) {
  const rel = Math.round(absSample - state.baseAbs);
  const capacity = Math.max(1, Number(state?.capacity || DISPLAY_SAMPLES));
  const oldest = (Number(state?.head || 0) - Number(state?.filled || 0) + capacity) % capacity;
  return (oldest + rel + capacity) % capacity;
}

function sampleFromCanvasX(x, state, width) {
  const { drawStart, visibleWidth, viewStart, viewSamples } = getReviewViewport(state, width);
  if (!state.filled) return state.baseAbs;
  const ratio = clamp((x - drawStart) / visibleWidth, 0, 0.999999);
  return viewStart + Math.floor(ratio * Math.max(viewSamples, 1));
}

function canvasXFromSample(absSample, state, width) {
  const { drawStart, visibleWidth, viewStart, viewSamples } = getReviewViewport(state, width);
  const rel = clamp(absSample - viewStart, 0, Math.max(viewSamples - 1, 0));
  const ratio = viewSamples > 1 ? rel / (viewSamples - 1) : 0;
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
    sampleIndex: Number.isFinite(marker.sampleIndex) ? Number(marker.sampleIndex)
      : (Number.isFinite(marker.sample_index) ? Number(marker.sample_index) : null),
  };
  S.review.markers.unshift(item);
  if (S.review.markers.length > 18) S.review.markers.length = 18;
  syncReviewUI();
}

function reviewSampleOffsetMs(sample, state) {
  const sr = Math.max(Number(state?.sampleRate || S.sampleRate || 250), 1);
  return ((Number(sample || 0) - Number(state?.baseAbs || 0)) / sr) * 1000;
}

function reviewPointLabel(sample, state) {
  if (!Number.isFinite(sample) || !state?.filled) return '--';
  return `${reviewSampleOffsetMs(sample, state).toFixed(1)} ms`;
}

function reviewRangeLabel(stats, state) {
  if (!stats) return '--';
  return `${reviewPointLabel(stats.startSample, state)} -> ${reviewPointLabel(stats.endSample, state)}`;
}

function pinFrozenReviewStats(state = getReviewRenderState()) {
  if (!state?.paused || !state?.filled) return null;
  if (S.review.selection) {
    S.review.lastStats = computeSelectionStats(S.review.selection, state);
    return S.review.lastStats;
  }
  const viewport = getReviewViewport(state, canvas.width || 1);
  S.review.lastStats = computeSelectionStats(
    {
      startSample: Number(viewport.viewStart || 0),
      endSample: Number(viewport.viewEnd || 0),
    },
    state,
  );
  return S.review.lastStats;
}

function renderFatigueValue(value) {
  const pct = Math.max(0, Math.min(100, Number(value || 0) * 100));
  const bar = $('fatigue-bar');
  if (bar) {
    bar.style.width = `${pct}%`;
    bar.style.background = pct > 70 ? 'var(--green)' : pct > 40 ? 'var(--yellow)' : 'var(--red)';
  }
  const label = pct > 70 ? 'Fresh' : pct > 40 ? 'Getting tired' : 'Fatigued - rest soon';
  if ($('fatigue-label')) $('fatigue-label').textContent = `Endurance: ${pct.toFixed(0)}% -- ${label}`;
}

function syncInspectorTelemetry() {
  if (S.review.paused) {
    updateRmsBars(S.review.snapshot?.rms || S.rms || []);
    updateQualityGrid(S.review.qualitySnapshot || S.lastQuality || []);
    renderFatigueValue(Number(S.review.fatigueSnapshot ?? S.fatigue ?? 1));
    return;
  }
  updateRmsBars(S.rms || []);
  updateQualityGrid(S.lastQuality || S.rms || []);
  renderFatigueValue(Number(S.fatigue || 1));
}

function formatAxisDuration(ms) {
  const durationMs = Math.max(Number(ms || 0), 0);
  if (durationMs >= 1000) {
    const seconds = durationMs / 1000;
    return seconds >= 10 ? `${seconds.toFixed(1)} s` : `${seconds.toFixed(2)} s`;
  }
  return `${durationMs.toFixed(0)} ms`;
}

function formatAxisAmplitude(value, units) {
  const amp = Math.max(Number(value || 0), 0);
  const decimals = amp >= 100 ? 0 : (amp >= 10 ? 1 : (amp >= 1 ? 2 : (amp >= 0.1 ? 3 : 4)));
  return `±${amp.toFixed(decimals)}`;
}

function formatSignalValue(value) {
  const numeric = Number(value || 0);
  const abs = Math.abs(numeric);
  const decimals = abs >= 100 ? 1 : (abs >= 10 ? 2 : 3);
  return `${numeric.toFixed(decimals)} ${S.signalUnits || ''}`.trim();
}

function getReviewChannelAtCanvasY(y, height) {
  const rows = getChannelDrawLayout(height);
  const safeY = clamp(Number(y || 0), 0, Math.max(Number(height || 0) - 1, 0));
  const row = rows.find(item => safeY >= Number(item.top || 0) && safeY < Number(item.top || 0) + Number(item.height || 0));
  return Number.isFinite(row?.index) ? Number(row.index) : clamp(Math.floor((safeY / Math.max(Number(height || 1), 1)) * N_CH), 0, N_CH - 1);
}

function getReviewSampleValue(state, sample, channel = getReviewFocusChannelIndex(state)) {
  if (!state?.filled || !Number.isFinite(sample)) return null;
  const viewport = getReviewViewport(state, canvas.width || canvas.clientWidth || 1);
  const safeChannel = clamp(Number(channel || 0), 0, N_CH - 1);
  const clampedSample = clamp(
    Math.round(Number(sample || 0)),
    Number(viewport.fullStart || 0),
    Number(viewport.fullEnd || 0),
  );
  const idx = bufferIndexForAbsSample(state, clampedSample);
  const value = Number(state.emg?.[safeChannel]?.[idx] || 0);
  return {
    sample: clampedSample,
    channel: safeChannel,
    channelLabel: channelDisplayLabel(safeChannel),
    value,
    valueLabel: formatSignalValue(value),
    timeLabel: reviewPointLabel(clampedSample, state),
  };
}

function getReviewHoverMetrics(state = getReviewRenderState()) {
  if (!state?.filled || !Number.isFinite(S.review.hoverSample)) return null;
  const channel = Number.isFinite(S.review.hoverChannel)
    ? Number(S.review.hoverChannel)
    : getReviewFocusChannelIndex(state);
  return getReviewSampleValue(state, Number(S.review.hoverSample), channel);
}

function clampReviewCenterSample(sample, state, width = canvas.width || canvas.clientWidth || 1) {
  const viewport = getReviewViewport(state, width);
  const span = Math.max(1, Number(viewport.viewSamples || 1));
  const halfBefore = Math.floor(Math.max(span - 1, 0) / 2);
  const halfAfter = Math.ceil(Math.max(span - 1, 0) / 2);
  const minCenter = Number(viewport.fullStart || 0) + halfBefore;
  const maxCenter = Number(viewport.fullEnd || 0) - halfAfter;
  if (maxCenter < minCenter) {
    return Math.round((Number(viewport.fullStart || 0) + Number(viewport.fullEnd || 0)) / 2);
  }
  return clamp(Math.round(Number(sample || 0)), minCenter, maxCenter);
}

function channelDisplayLabel(index) {
  const safe = clamp(Number(index || 0), 0, N_CH - 1);
  return String(S.channelLabels[safe] || `CH${safe + 1}`);
}

function setCopilotChannelFocus(index, { label = '', active = true } = {}) {
  if (!active || !Number.isFinite(index) || Number(index) < 0 || Number(index) >= N_CH) {
    S.copilot.focusChannel = -1;
    S.copilot.focusTarget = 0;
    S.copilot.focusLabel = '';
    syncScopeCopilotUI();
    return;
  }
  S.copilot.focusChannel = Number(index);
  S.copilot.focusTarget = 1;
  S.copilot.focusLabel = label || channelDisplayLabel(index);
  syncScopeCopilotUI();
}

function stepCopilotChannelFocus() {
  const target = Math.max(0, Math.min(1, Number(S.copilot.focusTarget || 0)));
  const current = Number(S.copilot.focusMix || 0);
  const next = current + (target - current) * 0.18;
  S.copilot.focusMix = Math.abs(next - target) < 0.01 ? target : next;
  if (S.copilot.focusMix < 0.02 && target === 0) {
    S.copilot.focusMix = 0;
    if (S.copilot.focusChannel < 0) S.copilot.focusLabel = '';
  }
}

function getChannelDrawLayout(totalHeight) {
  const focusIndex = Number(S.copilot.focusChannel || -1);
  const mix = Math.max(0, Math.min(1, Number(S.copilot.focusMix || 0)));
  const visible = Array.from({ length: N_CH }, (_, idx) => isChannelVisible(idx));
  const hasVisible = visible.some(Boolean);
  const weights = Array.from({ length: N_CH }, (_, idx) => {
    if (hasVisible && !visible[idx]) return 0;
    if (focusIndex < 0 || mix <= 0.001) return 1;
    return idx === focusIndex ? 1 + mix * 2.3 : Math.max(0.52, 1 - mix * 0.42);
  });
  const weightSum = weights.reduce((sum, value) => sum + value, 0) || N_CH;
  let top = 0;
  return weights.map((weight, idx) => {
    const height = (Number(totalHeight || 0) * weight) / weightSum;
    const row = {
      index: idx,
      top,
      height,
      mid: top + height * 0.5,
      focused: idx === focusIndex && mix > 0.03,
      dimmed: focusIndex >= 0 && idx !== focusIndex && mix > 0.03,
    };
    top += height;
    return row;
  });
}

function syncScopeCopilotUI() {
  const input = $('scope-copilot-input');
  const status = $('scope-copilot-status');
  const trace = $('scope-copilot-trace');
  const actions = $('scope-copilot-actions');
  const findings = $('scope-copilot-findings');
  const focusChip = $('signal-focus-chip');
  const focusLabel = $('signal-focus-label');

  if (input && input !== document.activeElement) input.value = String(S.copilot.input || '');
  if (status) status.textContent = String(S.copilot.status || 'Copilot ready.');

  if (trace) {
    trace.innerHTML = '';
    (Array.isArray(S.copilot.steps) ? S.copilot.steps : []).slice(0, 4).forEach(step => {
      const pill = document.createElement('span');
      pill.className = 'scope-copilot-step';
      pill.textContent = String(step);
      trace.appendChild(pill);
    });
  }

  if (actions) {
    actions.innerHTML = '';
    (Array.isArray(S.copilot.actions) ? S.copilot.actions : []).slice(0, 5).forEach((action, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `scope-copilot-action ${String(action?.tone || '')}`.trim();
      btn.textContent = String(action?.label || `Action ${idx + 1}`);
      btn.dataset.copilotActionIndex = String(idx);
      actions.appendChild(btn);
    });
  }

  if (findings) {
    findings.innerHTML = '';
    (Array.isArray(S.copilot.findings) ? S.copilot.findings : []).slice(0, 4).forEach((finding, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'scope-copilot-finding';
      btn.dataset.copilotFindingIndex = String(idx);
      btn.innerHTML = `
        <strong>${escapeHTML(String(finding?.label || `Region ${idx + 1}`))}</strong>
        <span>${escapeHTML(String(finding?.rangeLabel || 'Visible span'))}</span>
        <em>${escapeHTML(String(finding?.recommendationLabel || 'Inspect region'))}</em>
      `;
      findings.appendChild(btn);
    });
  }

  if (focusChip && focusLabel) {
    const visible = S.copilot.focusChannel >= 0 && Number(S.copilot.focusTarget || 0) > 0;
    focusChip.classList.toggle('visible', visible);
    focusLabel.textContent = visible
      ? `${S.copilot.focusLabel || channelDisplayLabel(S.copilot.focusChannel)} highlighted`
      : 'All channels';
  }
}

function escapeHTML(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function signalLogicSources() {
  return [
    { key: 'focus_rms', label: 'Focus RMS', type: 'number' },
    { key: 'rms', label: 'Window RMS', type: 'number' },
    { key: 'peak_to_peak', label: 'Peak To Peak', type: 'number' },
    { key: 'dominant_hz', label: 'Dominant Hz', type: 'number' },
    { key: 'visible_span_ms', label: 'Visible Span ms', type: 'number' },
    { key: 'focus_channel', label: 'Focus Channel', type: 'text' },
    { key: 'clip_pct', label: 'Clip %', type: 'number' },
    { key: 'hum_db', label: 'Hum dB', type: 'number' },
    { key: 'drift_db', label: 'Drift dB', type: 'number' },
    { key: 'issue_count', label: 'Issue Count', type: 'number' },
    { key: 'baseline_delta', label: 'Baseline Delta', type: 'number' },
    { key: 'prediction_label', label: 'Prediction Label', type: 'text' },
    { key: 'prediction_confidence', label: 'Prediction Confidence', type: 'number' },
    { key: 'arousal_score', label: 'Arousal Score', type: 'number' },
    { key: 'arousal_level', label: 'Arousal Level', type: 'text' },
    { key: 'artifact_label', label: 'Artifact Label', type: 'text' },
    { key: 'artifact_confidence', label: 'Artifact Confidence', type: 'number' },
    { key: 'qa_score', label: 'AI QA Score', type: 'number' },
    { key: 'signal_age_ms', label: 'Signal Age ms', type: 'number' },
    { key: 'streaming', label: 'Streaming', type: 'number' },
  ];
}

function signalLogicActions() {
  return [
    { key: 'toast', label: 'Toast', placeholder: 'Strong contraction detected' },
    { key: 'save_marker', label: 'Save Marker', placeholder: 'contraction_onset' },
    { key: 'freeze_focus', label: 'Freeze + Focus', placeholder: '' },
    { key: 'spotlight_artifact', label: 'Spotlight Artifact', placeholder: '' },
    { key: 'run_ai_scan', label: 'Scan', placeholder: '' },
    { key: 'gesture', label: 'Send Gesture', placeholder: 'fist' },
    { key: 'move_joint', label: 'Move Joint', placeholder: '2:120' },
    { key: 'digital_write', label: 'Digital Write', placeholder: '13:1' },
    { key: 'osc_message', label: 'OSC Message', placeholder: '/kyma/custom={"state":1}' },
    { key: 'serial_write', label: 'Serial Write', placeholder: 'LED ON\\n' },
    { key: 'webhook', label: 'Webhook', placeholder: 'https://host/path|{"event":"signal"}' },
  ];
}

function signalLogicOperatorsForType(type) {
  if (type === 'text') {
    return [
      { key: 'equals', label: 'equals' },
      { key: 'contains', label: 'contains' },
      { key: 'starts_with', label: 'starts with' },
      { key: 'changes_to', label: 'changes to' },
      { key: 'not_equals', label: 'not equals' },
    ];
  }
  return [
    { key: '>', label: '>' },
    { key: '>=', label: '>=' },
    { key: '<', label: '<' },
    { key: '<=', label: '<=' },
    { key: '==', label: '==' },
    { key: '!=', label: '!=' },
    { key: 'between', label: 'between' },
    { key: 'crosses_above', label: 'crosses above' },
    { key: 'crosses_below', label: 'crosses below' },
    { key: 'rises_by', label: 'rises by' },
    { key: 'falls_by', label: 'falls by' },
  ];
}

function getSignalLogicSourceMeta(key) {
  return signalLogicSources().find(item => item.key === key) || signalLogicSources()[0];
}

function getSignalLogicActionMeta(key) {
  return signalLogicActions().find(item => item.key === key) || signalLogicActions()[0];
}

function signalLogicDefaultRule(partial = {}) {
  return {
    id: partial.id || `logic-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    enabled: partial.enabled !== false,
    source: partial.source || 'focus_rms',
    operator: partial.operator || '>',
    value: partial.value ?? '10',
    hold_ms: Number.isFinite(Number(partial.hold_ms)) ? Number(partial.hold_ms) : 250,
    cooldown_ms: Number.isFinite(Number(partial.cooldown_ms)) ? Number(partial.cooldown_ms) : 1200,
    action: partial.action || 'save_marker',
    payload: partial.payload ?? '',
  };
}

function normalizeSignalLogicRule(rule = {}) {
  const next = signalLogicDefaultRule(rule);
  const sourceMeta = getSignalLogicSourceMeta(next.source);
  if (!signalLogicOperatorsForType(sourceMeta.type).some(item => item.key === next.operator)) {
    next.operator = signalLogicOperatorsForType(sourceMeta.type)[0]?.key || '>';
  }
  return next;
}

function signalLogicTemplates(profileKey = S.signalProfileKey) {
  const generic = [
    {
      label: 'Clip risk -> freeze and focus',
      rule: {
        source: 'clip_pct',
        operator: '>',
        value: '0.3',
        hold_ms: 150,
        cooldown_ms: 2500,
        action: 'freeze_focus',
        payload: '',
      },
    },
    {
      label: 'Artifact flagged -> AI scan',
      rule: {
        source: 'artifact_label',
        operator: 'not_equals',
        value: 'clean',
        hold_ms: 0,
        cooldown_ms: 4000,
        action: 'run_ai_scan',
        payload: '',
      },
    },
  ];
  const profile = String(profileKey || '').toLowerCase();
  if (profile === 'emg') {
    return [
      {
        label: 'Strong contraction -> marker',
        rule: {
          source: 'focus_rms',
          operator: '>',
          value: '12',
          hold_ms: 200,
          cooldown_ms: 1500,
          action: 'save_marker',
          payload: 'emg_contraction',
        },
      },
      {
        label: 'Prediction confidence -> gesture',
        rule: {
          source: 'prediction_confidence',
          operator: '>',
          value: '0.85',
          hold_ms: 150,
          cooldown_ms: 1200,
          action: 'gesture',
          payload: 'fist',
        },
      },
      ...generic,
    ];
  }
  if (profile === 'eeg') {
    return [
      {
        label: 'Alpha dominant -> marker',
        rule: {
          source: 'dominant_hz',
          operator: '>',
          value: '8',
          hold_ms: 300,
          cooldown_ms: 2200,
          action: 'save_marker',
          payload: 'eeg_band_shift',
        },
      },
      ...generic,
    ];
  }
  if (profile === 'eog') {
    return [
      {
        label: 'Blink burst artifact -> marker',
        rule: {
          source: 'artifact_label',
          operator: 'contains',
          value: 'artifact',
          hold_ms: 0,
          cooldown_ms: 1800,
          action: 'save_marker',
          payload: 'eog_event',
        },
      },
      ...generic,
    ];
  }
  if (profile === 'eog') {
    return [
      {
        label: 'Blink artifact -> marker',
        rule: {
          source: 'artifact_label',
          operator: 'contains',
          value: 'blink',
          hold_ms: 0,
          cooldown_ms: 1800,
          action: 'save_marker',
          payload: 'blink_window',
        },
      },
      ...generic,
    ];
  }
  return generic;
}

function persistSignalLogicConfig() {
  const serializable = (S.logic.rules || []).map(rule => ({
    id: rule.id,
    enabled: rule.enabled !== false,
    source: rule.source,
    operator: rule.operator,
    value: rule.value,
    hold_ms: Number(rule.hold_ms || 0),
    cooldown_ms: Number(rule.cooldown_ms || 0),
    action: rule.action,
    payload: rule.payload || '',
  }));
  localStorage.setItem(SIGNAL_LOGIC_STORAGE_KEY, JSON.stringify(serializable));
  localStorage.setItem(SIGNAL_LOGIC_ENABLED_KEY, S.logic.enabled ? '1' : '0');
}

function loadSignalLogicConfig() {
  let parsed = [];
  try {
    parsed = JSON.parse(localStorage.getItem(SIGNAL_LOGIC_STORAGE_KEY) || '[]');
  } catch {
    parsed = [];
  }
  S.logic.rules = Array.isArray(parsed) ? parsed.map(normalizeSignalLogicRule) : [];
  S.logic.enabled = localStorage.getItem(SIGNAL_LOGIC_ENABLED_KEY) !== '0';
}

function ensureSignalLogicRuntime(ruleId) {
  if (!S.logic.runtime[ruleId]) {
    S.logic.runtime[ruleId] = {
      activeSince: 0,
      lastFiredAt: 0,
      lastValue: null,
      busy: false,
      hotUntil: 0,
      lastMessage: '',
    };
  }
  return S.logic.runtime[ruleId];
}

function cleanupSignalLogicRuntime() {
  const keep = new Set((S.logic.rules || []).map(rule => rule.id));
  Object.keys(S.logic.runtime || {}).forEach(key => {
    if (!keep.has(key)) delete S.logic.runtime[key];
  });
}

function formatSignalLogicValue(key, value) {
  if (value == null || value === '') return '--';
  if (key === 'focus_rms' || key === 'rms' || key === 'peak_to_peak') return formatSignalValue(value);
  if (key === 'dominant_hz') return `${Number(value).toFixed(Number(value) >= 10 ? 1 : 2)} Hz`;
  if (key === 'visible_span_ms') return `${Number(value).toFixed(0)} ms`;
  if (key === 'hum_db' || key === 'drift_db') return `${Number(value).toFixed(1)} dB`;
  if (key === 'clip_pct') return `${Number(value).toFixed(2)}%`;
  if (key === 'baseline_delta') return `${Number(value).toFixed(2)} ${S.signalUnits || 'a.u.'}`;
  if (key === 'prediction_confidence' || key === 'artifact_confidence') return `${(Number(value) * 100).toFixed(0)}%`;
  if (key === 'qa_score') return `${Number(value).toFixed(0)}`;
  if (key === 'signal_age_ms') return `${Number(value).toFixed(0)} ms`;
  if (key === 'streaming') return Number(value) ? 'On' : 'Off';
  return String(value);
}

function readAIOverallScore(result = S.ai.result) {
  const raw = result?.qa_score;
  if (raw && typeof raw === 'object') return Number(raw.overall || 0);
  return Number(raw || 0);
}

function readAITopIssue(result = S.ai.result) {
  const raw = String(result?.artifact_summary?.top_issue || '').trim();
  if (!raw || raw === 'clean_window') return 'No issue';
  return raw.replace(/_/g, ' ');
}

function readAINextAction(result = S.ai.result) {
  const action = result?.top_action || (Array.isArray(result?.next_actions) ? result.next_actions[0] : null);
  return String(action?.title || action?.label || '').trim() || 'No action';
}

function buildSignalLogicSnapshot() {
  const state = getReviewRenderState();
  const result = state.paused ? (S.review.aiResultSnapshot || S.ai.result) : S.ai.result;
  const diagnostics = state.paused ? (S.review.diagnosticsSnapshot || S.diagnostics) : S.diagnostics;
  const metrics = aiStripMeasurements(state, result);
  const baseline = Number(S.propRestRms?.[getReviewFocusChannelIndex(state)] || 0);
  const artifacts = Array.isArray(state.paused ? S.review.artifactSnapshot : S.review.artifacts)
    ? (state.paused ? S.review.artifactSnapshot : S.review.artifacts)
    : [];
  const issueCount = artifacts.length;
  if (!state?.filled) {
    return {
      profile: S.signalProfileName || 'Signal',
      focus_rms: 0,
      rms: 0,
      peak_to_peak: 0,
      dominant_hz: 0,
      visible_span_ms: Number(metrics.visibleSpanMs || 0),
      focus_channel: String(metrics.focusLabel || S.channelLabels[0] || 'CH1'),
      clip_pct: Number(diagnostics?.noise?.clip_pct || 0),
      hum_db: Math.max(Number(diagnostics?.noise?.hum_50_db || 0), Number(diagnostics?.noise?.hum_60_db || 0)),
      drift_db: Number(diagnostics?.noise?.drift_db || 0),
      issue_count: issueCount,
      baseline_delta: 0 - baseline,
      prediction_label: String(S.lastPrediction?.label || ''),
      prediction_confidence: Number(S.lastPrediction?.confidence || 0),
      arousal_score: Number(S.lastPrediction?.arousal?.score || 0),
      arousal_level: String(S.lastPrediction?.arousal?.level || ''),
      artifact_label: 'clean',
      artifact_confidence: 0,
      qa_score: readAIOverallScore(result),
      signal_age_ms: Number(diagnostics?.timing?.signal_age_ms || 0),
      streaming: S.streaming ? 1 : 0,
    };
  }
  const prediction = (S.review.paused && S.review.predictionSnapshot)
    ? S.review.predictionSnapshot
    : S.lastPrediction;
  const artifactHead = result?.local_model_insights?.artifact_classifier || null;
  const artifact = artifacts.length ? artifacts[0] : null;
  const topIssue = String(result?.artifact_summary?.top_issue || '').trim();
  let artifactLabel = artifactHead?.label || artifact?.kind || topIssue || 'clean';
  if (artifactLabel === 'clean_window') artifactLabel = 'clean';
  artifactLabel = normalizeArtifactKind(artifactLabel);
  return {
    profile: S.signalProfileName || 'Signal',
    focus_rms: Number(metrics.focusRms || 0),
    rms: Number(metrics.rms || 0),
    peak_to_peak: Number(metrics.peakToPeak || 0),
    dominant_hz: Number(metrics.dominantHz || 0),
    visible_span_ms: Number(metrics.visibleSpanMs || 0),
    focus_channel: String(metrics.focusLabel || S.channelLabels[getReviewFocusChannelIndex(state)] || 'CH1'),
    clip_pct: Number(diagnostics?.noise?.clip_pct || 0),
    hum_db: Math.max(Number(diagnostics?.noise?.hum_50_db || 0), Number(diagnostics?.noise?.hum_60_db || 0)),
    drift_db: Number(diagnostics?.noise?.drift_db || 0),
    issue_count: issueCount,
    baseline_delta: Number(metrics.focusRms || 0) - baseline,
    prediction_label: String(prediction?.label || ''),
    prediction_confidence: Number(prediction?.confidence || 0),
    arousal_score: Number(prediction?.arousal?.score || 0),
    arousal_level: String(prediction?.arousal?.level || ''),
    artifact_label: artifactLabel,
    artifact_confidence: Number(artifactHead?.score || artifactHead?.confidence || 0),
    qa_score: readAIOverallScore(result),
    signal_age_ms: Number(diagnostics?.timing?.signal_age_ms || 0),
    streaming: S.streaming ? 1 : 0,
  };
}

function syncSignalLogicLiveReadout(snapshot = S.logic.liveSnapshot || buildSignalLogicSnapshot()) {
  S.logic.liveSnapshot = snapshot;
  if ($('logic-live-profile')) $('logic-live-profile').textContent = `${snapshot.profile || S.signalProfileName} ${S.logic.enabled ? '· runtime on' : '· runtime off'}`;
  if ($('logic-live-focus-rms')) $('logic-live-focus-rms').textContent = formatSignalLogicValue('focus_rms', snapshot.focus_rms);
  if ($('logic-live-dom')) $('logic-live-dom').textContent = formatSignalLogicValue('dominant_hz', snapshot.dominant_hz);
  if ($('logic-live-pred')) {
    const pred = snapshot.prediction_label
      ? `${snapshot.prediction_label} · ${formatSignalLogicValue('prediction_confidence', snapshot.prediction_confidence)}`
      : 'No decoded label';
    $('logic-live-pred').textContent = pred;
  }
  if ($('logic-live-artifact')) {
    const art = snapshot.artifact_label && snapshot.artifact_label !== 'clean'
      ? `${artifactStyle(normalizeArtifactKind(snapshot.artifact_label)).label} · ${formatSignalLogicValue('artifact_confidence', snapshot.artifact_confidence)}`
      : 'Clean / none';
    $('logic-live-artifact').textContent = art;
  }
  if ($('logic-live-qa')) $('logic-live-qa').textContent = snapshot.qa_score ? `${formatSignalLogicValue('qa_score', snapshot.qa_score)} / 100` : '--';
  if ($('logic-trust-score')) $('logic-trust-score').textContent = snapshot.qa_score ? `${formatSignalLogicValue('qa_score', snapshot.qa_score)} / 100` : '--';
  if ($('logic-trust-issue')) $('logic-trust-issue').textContent = readAITopIssue();
  if ($('logic-trust-next')) $('logic-trust-next').textContent = readAINextAction();
  renderSignalLogicSourceBrowser(snapshot);
  syncSignalStudioDrawer(snapshot);
}

function signalLogicRuleSummary(rule) {
  const source = getSignalLogicSourceMeta(rule.source);
  const action = getSignalLogicActionMeta(rule.action);
  const target = String(rule.value || '').trim();
  const comparison = target ? `${source.label} ${rule.operator.replace(/_/g, ' ')} ${target}` : `${source.label} ${rule.operator.replace(/_/g, ' ')}`;
  return `${comparison} -> ${action.label}${rule.payload ? ` (${rule.payload})` : ''}`;
}

function renderSignalLogicRules() {
  const list = $('logic-rule-list');
  if (!list) return;
  cleanupSignalLogicRuntime();
  if (!S.logic.rules.length) {
    list.innerHTML = '<div class="setup-copy">No signal rules yet. Add one or load a starter.</div>';
    return;
  }
  list.innerHTML = '';
  S.logic.rules.forEach((rule, index) => {
    const sourceMeta = getSignalLogicSourceMeta(rule.source);
    const actionMeta = getSignalLogicActionMeta(rule.action);
    const operators = signalLogicOperatorsForType(sourceMeta.type)
      .map(op => `<option value="${escapeHTML(op.key)}" ${op.key === rule.operator ? 'selected' : ''}>${escapeHTML(op.label)}</option>`)
      .join('');
    const sources = signalLogicSources()
      .map(item => `<option value="${escapeHTML(item.key)}" ${item.key === rule.source ? 'selected' : ''}>${escapeHTML(item.label)}</option>`)
      .join('');
    const actions = signalLogicActions()
      .map(item => `<option value="${escapeHTML(item.key)}" ${item.key === rule.action ? 'selected' : ''}>${escapeHTML(item.label)}</option>`)
      .join('');
    const runtime = ensureSignalLogicRuntime(rule.id);
    const cooldownRemaining = Math.max(0, Number(rule.cooldown_ms || 0) - (Date.now() - Number(runtime.lastFiredAt || 0)));
    const active = Date.now() < Number(runtime.hotUntil || 0);
    const statusText = !rule.enabled
      ? 'Disabled'
      : active
        ? 'Fired'
        : cooldownRemaining > 0
          ? 'Cooldown'
          : runtime.activeSince
            ? 'Armed'
            : 'Idle';
    const statusClass = active ? 'hot' : (cooldownRemaining > 0 ? 'cooldown' : '');
    const card = document.createElement('div');
    card.className = `logic-rule-card ${active ? 'active' : ''}`;
    card.dataset.logicRuleId = rule.id;
    card.innerHTML = `
      <div class="logic-rule-head">
        <div class="logic-rule-head-left">
          <label class="logic-rule-toggle">
            <input type="checkbox" data-logic-field="enabled" ${rule.enabled ? 'checked' : ''}>
            <span>Rule ${index + 1}</span>
          </label>
          <span class="logic-rule-state ${statusClass}" data-logic-role="state">${escapeHTML(statusText)}</span>
        </div>
        <button class="btn logic-rule-delete" type="button" data-logic-action="delete">Delete</button>
      </div>
      <div class="logic-rule-grid">
        <div class="logic-rule-field">
          <label class="setup-label">Signal</label>
          <select class="setup-input" data-logic-field="source">${sources}</select>
        </div>
        <div class="logic-rule-field">
          <label class="setup-label">Condition</label>
          <select class="setup-input" data-logic-field="operator">${operators}</select>
        </div>
        <div class="logic-rule-field">
          <label class="setup-label">Value</label>
          <input class="setup-input" data-logic-field="value" type="text" value="${escapeHTML(rule.value)}" placeholder="threshold or label">
        </div>
        <div class="logic-rule-field">
          <label class="setup-label">Action</label>
          <select class="setup-input" data-logic-field="action">${actions}</select>
        </div>
        <div class="logic-rule-field">
          <label class="setup-label">Hold ms</label>
          <input class="setup-input" data-logic-field="hold_ms" type="number" min="0" step="50" value="${Number(rule.hold_ms || 0)}">
        </div>
        <div class="logic-rule-field">
          <label class="setup-label">Cooldown ms</label>
          <input class="setup-input" data-logic-field="cooldown_ms" type="number" min="0" step="100" value="${Number(rule.cooldown_ms || 0)}">
        </div>
        <div class="logic-rule-field full">
          <label class="setup-label">Payload / Target</label>
          <input class="setup-input" data-logic-field="payload" type="text" value="${escapeHTML(rule.payload)}" placeholder="${escapeHTML(actionMeta.placeholder || '')}">
        </div>
      </div>
      <div class="logic-rule-note" data-logic-role="note">${escapeHTML(signalLogicRuleSummary(rule))}${runtime.lastMessage ? ` | ${escapeHTML(runtime.lastMessage)}` : ''}</div>`;
    list.appendChild(card);
  });
}

function syncSignalLogicRuleStates() {
  document.querySelectorAll('[data-logic-rule-id]').forEach(row => {
    const rule = (S.logic.rules || []).find(item => item.id === row.dataset.logicRuleId);
    if (!rule) return;
    const runtime = ensureSignalLogicRuntime(rule.id);
    const cooldownRemaining = Math.max(0, Number(rule.cooldown_ms || 0) - (Date.now() - Number(runtime.lastFiredAt || 0)));
    const active = Date.now() < Number(runtime.hotUntil || 0);
    const statusText = !rule.enabled
      ? 'Disabled'
      : active
        ? 'Fired'
        : cooldownRemaining > 0
          ? 'Cooldown'
          : runtime.activeSince
            ? 'Armed'
            : 'Idle';
    const stateEl = row.querySelector('[data-logic-role="state"]');
    if (stateEl) {
      stateEl.textContent = statusText;
      stateEl.className = `logic-rule-state ${active ? 'hot' : (cooldownRemaining > 0 ? 'cooldown' : '')}`;
    }
    row.classList.toggle('active', active);
    const noteEl = row.querySelector('[data-logic-role="note"]');
    if (noteEl) {
      noteEl.textContent = `${signalLogicRuleSummary(rule)}${runtime.lastMessage ? ` | ${runtime.lastMessage}` : ''}`;
    }
  });
}

function renderSignalLogicFeed() {
  const feed = $('logic-runtime-feed');
  if (!feed) return;
  if (!S.logic.feed.length) {
    feed.innerHTML = '<div class="setup-copy">No logic events yet.</div>';
    return;
  }
  feed.innerHTML = '';
  S.logic.feed.forEach(item => {
    const row = document.createElement('div');
    row.className = 'logic-feed-item';
    if (Number.isFinite(item.startSample) && Number.isFinite(item.endSample)) {
      row.dataset.logicFeedStart = String(item.startSample);
      row.dataset.logicFeedEnd = String(item.endSample);
      row.style.cursor = 'pointer';
    }
    const when = Number.isFinite(item.startSample)
      ? reviewPointLabel(item.startSample, getReviewRenderState())
      : new Date(item.at || Date.now()).toLocaleTimeString();
    row.innerHTML = `
      <div class="logic-feed-head">
        <strong>${escapeHTML(item.title || 'Logic Event')}</strong>
        <span>${escapeHTML(when)}</span>
      </div>
      <div class="logic-feed-copy">${escapeHTML(item.detail || '')}</div>`;
    feed.appendChild(row);
  });
}

function focusSignalLogicFeedRange(startSample, endSample) {
  if (!Number.isFinite(startSample) || !Number.isFinite(endSample)) return;
  spotlightReviewRange(Number(startSample), Number(endSample), 'Logic Replay Focus');
}

function populateSignalLogicTemplateOptions() {
  const select = $('logic-template');
  if (!select) return;
  const previous = select.value || '';
  const templates = signalLogicTemplates();
  select.innerHTML = '<option value="">Choose a starter rule</option>';
  templates.forEach((item, idx) => {
    const opt = document.createElement('option');
    opt.value = String(idx);
    opt.textContent = item.label;
    select.appendChild(opt);
  });
  if (previous && templates[Number(previous)]) select.value = previous;
}

function syncSignalLogicSummary() {
  const summary = $('signal-logic-summary');
  const toggle = $('btn-logic-toggle');
  const logicChip = $('logic-command-chip');
  if (summary) {
    const activeCount = (S.logic.rules || []).filter(rule => rule.enabled !== false).length;
    summary.textContent = S.logic.enabled
      ? `${activeCount} live rule${activeCount === 1 ? '' : 's'} armed. Build from live vars, trust, artifacts, or replay tests.`
      : 'Runtime is off. You can still edit rules, inspect vars, and run replay tests.';
  }
  if (toggle) {
    toggle.textContent = S.logic.enabled ? 'Runtime On' : 'Runtime Off';
    toggle.className = S.logic.enabled ? 'btn success' : 'btn';
  }
  if (logicChip) {
    const activeCount = (S.logic.rules || []).filter(rule => rule.enabled !== false).length;
    logicChip.textContent = S.logic.enabled ? `Logic ${activeCount}` : 'Logic Off';
    logicChip.className = S.logic.enabled ? 'scope-status-chip' : 'scope-status-chip muted';
  }
}

function renderSignalLogicSourceBrowser(snapshot = S.logic.liveSnapshot || buildSignalLogicSnapshot()) {
  const list = $('logic-source-list');
  const query = String($('logic-source-filter')?.value || '').trim().toLowerCase();
  if (!list) return;
  const items = signalLogicSources().filter((item) => {
    if (!query) return true;
    return `${item.label} ${item.key} ${item.type}`.toLowerCase().includes(query);
  });
  if (!items.length) {
    list.innerHTML = '<div class="setup-copy">No variables match this filter.</div>';
    return;
  }
  list.innerHTML = '';
  items.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'logic-source-item';
    row.dataset.logicSourceKey = item.key;
    row.innerHTML = `
      <div class="logic-source-copy">
        <strong>${escapeHTML(item.label)}</strong>
        <span>${escapeHTML(formatSignalLogicValue(item.key, snapshot[item.key]))}</span>
        <em>${escapeHTML(item.key)} · ${escapeHTML(item.type)}</em>
      </div>
      <div class="logic-source-actions">
        <button class="btn logic-source-btn" type="button" data-logic-source-action="use">Use</button>
        <button class="btn logic-source-btn" type="button" data-logic-source-action="copy">Copy</button>
      </div>
    `;
    list.appendChild(row);
  });
}

function syncSignalStudioDrawer(snapshot = S.logic.liveSnapshot || buildSignalLogicSnapshot()) {
  const drawer = $('scope-studio-drawer');
  const source = $('scope-studio-source');
  const runtimeMeta = $('scope-studio-runtime-meta');
  const variableList = $('scope-variable-list');
  const debuggerList = $('scope-logic-debugger');
  const toggle = $('btn-studio-toggle');
  if (drawer) drawer.classList.toggle('is-hidden', !S.logic.drawerOpen);
  if (toggle) toggle.textContent = S.logic.drawerOpen ? 'Hide Studio' : 'Studio';
  if (source) {
    const mode = S.review.paused ? 'Frozen window' : (S.streamSource === 'playback' ? 'Playback stream' : 'Live window');
    source.textContent = mode;
  }
  if (runtimeMeta) {
    const activeCount = (S.logic.rules || []).filter(rule => rule.enabled !== false).length;
    runtimeMeta.textContent = activeCount ? `${activeCount} armed · replay ready` : 'No rules';
  }
  if (variableList) {
    const sources = signalLogicSources();
    variableList.innerHTML = '';
    sources.forEach(item => {
      const row = document.createElement('div');
      row.className = 'scope-studio-item';
      row.innerHTML = `<strong>${escapeHTML(item.label)}</strong><span>${escapeHTML(formatSignalLogicValue(item.key, snapshot[item.key]))}</span><em>${escapeHTML(item.type)}</em>`;
      variableList.appendChild(row);
    });
  }
  if (debuggerList) {
    if (!(S.logic.rules || []).length) {
      debuggerList.innerHTML = '<div class="setup-copy">No rules armed yet. Open Signal Logic in the left rail to add one.</div>';
    } else {
      debuggerList.innerHTML = '';
      (S.logic.rules || []).forEach(rule => {
        const runtime = ensureSignalLogicRuntime(rule.id);
        const row = document.createElement('div');
        const active = Date.now() < Number(runtime.hotUntil || 0);
        const cooldownRemaining = Math.max(0, Number(rule.cooldown_ms || 0) - (Date.now() - Number(runtime.lastFiredAt || 0)));
        const stateLabel = !rule.enabled
          ? 'disabled'
          : active
            ? 'fired'
            : cooldownRemaining > 0
              ? `cooldown ${Math.ceil(cooldownRemaining / 100) / 10}s`
              : runtime.activeSince
                ? 'armed'
                : 'idle';
        const currentValue = formatSignalLogicValue(rule.source, snapshot[rule.source]);
        row.className = 'scope-studio-item';
        row.innerHTML = `<strong>${escapeHTML(getSignalLogicSourceMeta(rule.source).label)}</strong><span>${escapeHTML(signalLogicRuleSummary(rule))} · now ${escapeHTML(currentValue)}</span><em>${escapeHTML(stateLabel)}</em>`;
        debuggerList.appendChild(row);
      });
      (S.logic.feed || []).slice(0, 4).forEach(item => {
        const row = document.createElement('div');
        row.className = 'scope-studio-item';
        if (Number.isFinite(item.startSample) && Number.isFinite(item.endSample)) {
          row.style.cursor = 'pointer';
          row.title = 'Click to focus this replay hit on the signal';
          row.onclick = () => {
            focusSignalLogicFeedRange(item.startSample, item.endSample);
            toast('Focused logic replay hit');
          };
        }
        row.innerHTML = `<strong>${escapeHTML(item.title)}</strong><span>${escapeHTML(item.detail || '')}</span><em>${escapeHTML(new Date(item.at || Date.now()).toLocaleTimeString())}</em>`;
        debuggerList.appendChild(row);
      });
    }
  }
}

function pipelineListHTML(items = []) {
  const safeItems = Array.isArray(items) ? items : [];
  if (!safeItems.length) return '<li>No items yet.</li>';
  return safeItems.map(item => `<li>${escapeHTML(String(item || ''))}</li>`).join('');
}

function formatPipelineFilter(stage = {}) {
  const kind = String(stage.kind || 'filter').replace(/_/g, ' ');
  const order = Number(stage.order || 0);
  const low = stage.low_hz == null ? NaN : Number(stage.low_hz);
  const high = stage.high_hz == null ? NaN : Number(stage.high_hz);
  const cutoff = stage.cutoff_hz == null ? NaN : Number(stage.cutoff_hz);
  const hz = value => Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (Number.isFinite(low) && Number.isFinite(high)) {
    return `${kind} ${hz(low)} to ${hz(high)} Hz${order ? `, order ${order}` : ''}`;
  }
  if (Number.isFinite(cutoff)) {
    return `${kind} ${hz(cutoff)} Hz${order ? `, order ${order}` : ''}`;
  }
  return `${kind}${order ? `, order ${order}` : ''}`;
}

function renderPipelineQAHTML(qa) {
  if (!qa) return '';
  const artifacts = Array.isArray(qa.artifacts) ? qa.artifacts : [];
  const channels = Array.isArray(qa.channels) ? qa.channels : [];
  const score = Number(qa.score || 0);
  const tone = qa.ready_for_training ? 'good' : (score >= 60 ? 'warn' : 'bad');
  const artifactText = artifacts.length
    ? artifacts.map(item => `${item.kind || 'artifact'}: ${item.detail || 'flagged'}`)
    : ['No artifact threshold crossed.'];
  const channelRows = channels.slice(0, 8).map(ch => `
    <div class="pipeline-result-row">
      <strong>${escapeHTML(ch.label || `CH${Number(ch.index || 0) + 1}`)}</strong>
      <span>${escapeHTML(`${Number(ch.rms || 0).toFixed(2)} ${S.signalUnits || ''} rms | q ${Number(ch.quality || 0).toFixed(2)}`)}</span>
    </div>
  `).join('');
  return `
    <div class="pipeline-section">
      <strong>Signal QA</strong>
      <div class="pipeline-pill-row">
        <span class="pipeline-pill ${tone}">${escapeHTML(`${score.toFixed(1)} / 100`)}</span>
        <span class="pipeline-pill ${qa.ready_for_training ? 'good' : 'warn'}">${qa.ready_for_training ? 'ready to train' : 'needs cleanup'}</span>
        <span class="pipeline-pill">${escapeHTML(`${qa.channel_count || channels.length || N_CH} ch`)}</span>
      </div>
      <ul class="pipeline-list">${pipelineListHTML(artifactText)}</ul>
    </div>
    <div class="pipeline-section">
      <strong>Channel QA</strong>
      ${channelRows || '<p class="pipeline-summary">No channel QA returned.</p>'}
    </div>
  `;
}

function renderPipelineTrainHTML(trainResult) {
  if (!trainResult) return '';
  const trained = !!trainResult.trained;
  const ready = trainResult.ready !== false;
  const req = trainResult.requirements || {};
  const result = trainResult.result || {};
  const summary = trainResult.summary || {};
  const perGesture = summary.per_gesture || summary.per_label || result.per_label || {};
  const counts = Object.keys(perGesture).map(label => `${label}: ${perGesture[label]} windows`);
  const labelQuality = result.per_label_quality || {};
  const qualityRows = Object.keys(labelQuality).map(label => {
    const item = labelQuality[label] || {};
    const recall = Number(item.recall || 0);
    const tone = recall >= 0.75 ? 'good' : (recall >= 0.5 ? 'warn' : 'bad');
    const confused = item.top_confusion ? ` | confused as ${item.top_confusion.label} x${item.top_confusion.count}` : '';
    return `
      <div class="pipeline-result-row pipeline-recipe-row">
        <strong>${escapeHTML(label)}</strong>
        <span class="${tone}">${escapeHTML(`P ${(Number(item.precision || 0) * 100).toFixed(0)} | R ${(recall * 100).toFixed(0)} | F1 ${(Number(item.f1 || 0) * 100).toFixed(0)} | n ${item.support || 0}${confused}`)}</span>
      </div>
    `;
  }).join('');
  const confusionLabels = Array.isArray(result.confusion_labels) ? result.confusion_labels : [];
  const confusion = Array.isArray(result.confusion_matrix) ? result.confusion_matrix : [];
  const confusionHTML = confusionLabels.length && confusion.length ? `
    <div class="pipeline-confusion" style="--cols:${confusionLabels.length + 1}">
      <span></span>
      ${confusionLabels.map(label => `<strong>${escapeHTML(label)}</strong>`).join('')}
      ${confusion.map((row, i) => `
        <strong>${escapeHTML(confusionLabels[i] || `L${i + 1}`)}</strong>
        ${confusionLabels.map((_, j) => {
          const value = Number(row?.[j] || 0);
          const diag = i === j;
          return `<span class="${diag ? 'diag' : (value ? 'miss' : '')}">${value}</span>`;
        }).join('')}
      `).join('')}
    </div>
  ` : '';
  const warnings = Array.isArray(result.quality_warnings) ? result.quality_warnings : [];
  const lines = trained
    ? [
        `${trainResult.classifier || result.classifier || 'LDA'} trained on ${result.n_windows || summary.total_windows || 0} windows.`,
        result.cv_balanced_accuracy == null ? 'Cross-validation unavailable for this split.' : `Balanced CV accuracy ${(Number(result.cv_balanced_accuracy) * 100).toFixed(1)}%.`,
        result.macro_f1 == null ? 'Macro F1 unavailable.' : `Macro F1 ${(Number(result.macro_f1) * 100).toFixed(1)}%.`,
        ready ? 'Validation gate passed for live preview.' : `Validation gate failed${result.readiness_threshold ? `; needs ${(Number(result.readiness_threshold) * 100).toFixed(1)}%+` : ''}.`,
        trainResult.model_path ? `Saved ${trainResult.model_path}` : 'Model saved by active pipeline.',
        ...(ready ? [] : (Array.isArray(trainResult.next_actions) ? trainResult.next_actions.slice(0, 2) : [])),
      ]
    : [
        `Need ${req.minimum_windows_total || 10}+ windows across ${req.minimum_labeled_classes || 2}+ labels.`,
        `Current: ${req.current_windows_total || 0} windows across ${req.current_labeled_classes || 0} labels.`,
        ...(Array.isArray(trainResult.next_actions) ? trainResult.next_actions : []),
      ];
  return `
    <div class="pipeline-section">
      <strong>Model Training</strong>
      <div class="pipeline-pill-row">
        <span class="pipeline-pill ${trained && ready ? 'good' : 'warn'}">${trained ? (ready ? 'trained' : 'needs data') : 'not ready'}</span>
        <span class="pipeline-pill">${escapeHTML(trainResult.classifier || 'LDA')}</span>
        ${trained && result.quality_score != null ? `<span class="pipeline-pill ${Number(result.quality_score) >= 70 ? 'good' : (Number(result.quality_score) >= 50 ? 'warn' : 'bad')}">quality ${Number(result.quality_score).toFixed(1)}</span>` : ''}
      </div>
      <ul class="pipeline-list">${pipelineListHTML(lines)}</ul>
      ${warnings.length ? `<ul class="pipeline-list">${pipelineListHTML(warnings)}</ul>` : ''}
      ${counts.length ? `<ul class="pipeline-list">${pipelineListHTML(counts)}</ul>` : ''}
      ${qualityRows ? `<div class="pipeline-model-quality">${qualityRows}</div>` : ''}
      ${confusionHTML}
    </div>
  `;
}

function pipelinePlainGoal() {
  const prompt = String(S.pipelineBuilder.prompt || '').toLowerCase();
  const profile = String(S.signalProfileName || 'biosignal');
  if (pipelinePromptRequestsFrequencyApp()) return `${profile} frequency band visualizer`;
  if (prompt.includes('fatigue')) return `${profile} fatigue detector`;
  if (prompt.includes('blink')) return `${profile} blink detector`;
  if (prompt.includes('gesture') || prompt.includes('open') || prompt.includes('close') || prompt.includes('pinch')) return `${profile} gesture model`;
  if (prompt.includes('attention') || prompt.includes('focus')) return `${profile} attention model`;
  if (prompt.includes('artifact') || prompt.includes('noise') || prompt.includes('clean')) return `${profile} signal-quality model`;
  if (prompt.includes('control')) return `${profile} control model`;
  return `${profile} model`;
}

function pipelinePromptRequestsFrequencyApp() {
  const prompt = String(S.pipelineBuilder.prompt || '').toLowerCase();
  return /\b(frequency|frequencies|freq|bands?|bandpower|fft|spectrum|spectral|psd|power\s+bands?)\b/.test(prompt);
}

function pipelineBuildNeedsTraining() {
  return !pipelinePromptRequestsFrequencyApp();
}

function pipelineAutopilotNowText(status, phase, taskSteps, currentTask) {
  if (S.pipelineBuilder.autopilotLoading) {
    const stage = String(S.pipelineBuilder.createStage || '').trim();
    if (stage) return `${stage}...`;
    return `Thinking through the ${pipelinePlainGoal()}...`;
  }
  const currentLabel = String(currentTask?.label || currentTask?.name || '').trim();
  const firstTask = taskSteps[0] || {};
  const firstLabel = String(firstTask.label || firstTask.name || '').trim();
  if (status === 'waiting_setup') {
    return `Put the electrodes on, start the stream, then KYMA will walk you through the tasks.`;
  }
  if (status === 'waiting_task') {
    return currentLabel
      ? `Prepare for: ${currentLabel}. Press start when you are ready.`
      : (firstLabel ? `Prepare for: ${firstLabel}. Press start when you are ready.` : `Prepare for the next recording task.`);
  }
  if (status === 'needs_data') {
    return `That recording was not clean enough yet. Fix the setup and retry the guided tasks.`;
  }
  if (status === 'failed') {
    return `KYMA stopped before the model was ready. Check the message below and retry.`;
  }
  if (status === 'choose_setup') {
    return `Choose the data source and the app KYMA should build.`;
  }
  if (phase === 'tasks') {
    return currentLabel
      ? `Do this now: ${currentLabel}.`
      : (firstLabel ? `Get ready for the first task: ${firstLabel}.` : `Follow the current task prompt.`);
  }
  if (phase === 'done') return `The ${pipelinePlainGoal()} is ready to review and export.`;
  return `KYMA is turning the prompt into a guided ${pipelinePlainGoal()} workflow.`;
}

function pipelineAppTypeLabel(type = S.pipelineBuilder.appType) {
  const map = {
    dashboard: 'Live dashboard',
    alert: 'Alert or API trigger',
    control: 'Control app',
    research: 'Research report',
  };
  return map[String(type || '')] || 'App not selected';
}

function renderPipelineBuildChooserHTML() {
  const source = String(S.pipelineBuilder.source || '');
  const appType = String(S.pipelineBuilder.appType || '');
  const sourceCard = (value, title, copy) => `
    <button class="pipeline-choice-card ${source === value ? 'active' : ''}" type="button" onclick="window.choosePipelineBuildSource('${value}')">
      <strong>${escapeHTML(title)}</strong>
      <span>${escapeHTML(copy)}</span>
    </button>
  `;
  const appCard = (value, title, copy) => `
    <button class="pipeline-choice-card ${appType === value ? 'active' : ''}" type="button" onclick="window.choosePipelineAppType('${value}')">
      <strong>${escapeHTML(title)}</strong>
      <span>${escapeHTML(copy)}</span>
    </button>
  `;
  const ready = ['live', 'synthetic', 'dataset'].includes(source) && !!appType && (source !== 'dataset' || !!String(S.pipelineBuilder.datasetPath || '').trim());
  return `
    <div class="pipeline-task-prep">
      <strong>Choose the data source</strong>
      <span>KYMA will not use synthetic data unless you choose Synthetic demo here.</span>
      <div class="pipeline-choice-grid">
        ${sourceCard('live', 'Live electrodes', 'Use connected biosignal hardware and guide recording one task at a time.')}
        ${sourceCard('synthetic', 'Synthetic demo', 'Use simulated biosignals only for demos or UI tests.')}
        ${sourceCard('dataset', 'Dataset import', 'Train from an existing file or session folder.')}
      </div>
      ${source === 'dataset' ? `<input id="pipeline-choice-dataset-path" class="pipeline-choice-input" type="text" spellcheck="false" value="${escapeHTML(S.pipelineBuilder.datasetPath || '')}" placeholder="Dataset path, CSV, XDF, EDF/BDF, NPZ, or session folder">` : ''}
    </div>
    <div class="pipeline-task-prep">
      <strong>Choose the app to generate</strong>
      <span>The coding workspace opens after KYMA trains and checks the model quality.</span>
      <div class="pipeline-choice-grid">
        ${appCard('dashboard', 'Live dashboard', 'Show predictions, confidence, signal state, and task status.')}
        ${appCard('alert', 'Alert/API trigger', 'Send events when the signal condition is detected.')}
        ${appCard('control', 'Control app', 'Turn decoded intent into keyboard, game, or device commands.')}
        ${appCard('research', 'Research report', 'Create a model card, QA summary, and export package.')}
      </div>
      <div class="pipeline-action-row">
        <button class="scope-command-btn primary" type="button" onclick="window.beginPipelineBuildWithChoices()" ${ready ? '' : 'disabled'}>Continue</button>
        <span class="pipeline-meta">${escapeHTML(source ? `${source.replace(/_/g, ' ')} | ${pipelineAppTypeLabel(appType)}` : 'Choose source and app type')}</span>
      </div>
    </div>
  `;
}

function renderPipelineTrainingQualityHTML() {
  const train = S.pipelineBuilder.trainResult || {};
  const result = train.result || train || {};
  if (!train.trained && !result.trained) return '';
  if (train.analysis_app) {
    const quality = Number(result.quality_score ?? train.quality_score);
    return `
      <div class="pipeline-task-prep">
        <strong>Signal analysis app is ready</strong>
        <span>This app uses live EMG samples directly, so no gesture classifier is required.</span>
        <ul class="pipeline-list">${pipelineListHTML([
          Number.isFinite(quality) ? `Signal QA score: ${quality.toFixed(1)}/100` : 'Signal QA completed',
          'Coding handoff: ready',
          'Runtime: WebSocket EMG sample stream',
        ])}</ul>
      </div>
    `;
  }
  const quality = Number(result.quality_score ?? train.quality_score);
  const val = Number(result.val_accuracy ?? train.val_accuracy);
  const acc = Number(result.accuracy ?? train.accuracy ?? train.train_accuracy);
  const ready = train.ready !== false && result.ready !== false && (Number.isFinite(quality) ? quality >= 55 : true);
  const warnings = (result.quality_warnings || train.quality_warnings || train.next_actions || []).filter(Boolean).slice(0, 3);
  const lines = [
    Number.isFinite(quality) ? `Quality score: ${quality.toFixed(1)}/100` : '',
    Number.isFinite(val) ? `Validation accuracy: ${Math.round(val * 100)}%` : '',
    Number.isFinite(acc) ? `Training accuracy: ${Math.round(acc * 100)}%` : '',
    `Coding handoff: ${ready ? 'ready' : 'waiting for clearer examples'}`,
  ].filter(Boolean);
  return `
    <div class="pipeline-task-prep">
      <strong>${ready ? 'Training looks usable' : 'Training needs another pass'}</strong>
      <span>${escapeHTML(ready ? 'KYMA can generate the app code now.' : (warnings[0] || 'Collect cleaner examples before generating the app.'))}</span>
      <ul class="pipeline-list">${pipelineListHTML(lines)}</ul>
      ${warnings.length ? `<ul class="pipeline-list">${pipelineListHTML(warnings)}</ul>` : ''}
    </div>
  `;
}

function renderPipelineAutopilotHTML(autopilot) {
  if (!autopilot) return '';
  const steps = Array.isArray(autopilot.steps) ? autopilot.steps : [];
  const status = String(autopilot.status || 'ready');
  const tone = status === 'ready' ? 'good' : (status === 'failed' ? 'bad' : (status === 'needs_data' ? 'warn' : ''));
  const completed = steps.filter(step => String(step?.status || '') === 'done').length;
  const phase = status === 'ready'
    ? 'done'
    : (status === 'choose_setup' || ['waiting_setup', 'needs_data', 'failed'].includes(status) ? 'setup' : (['collecting', 'waiting_task'].includes(status) ? 'tasks' : 'review'));
  const taskSteps = pipelineBuildNeedsTraining()
    ? (S.pipelineBuilder.acquisition?.steps || [])
      .filter(step => String(step?.action || '') === 'record_label')
      .slice(0, 8)
    : [];
  const run = S.pipelineBuilder.acquisitionRun || {};
  const currentTask = run.current_step || {};
  const guidedIndex = clamp(Number(S.pipelineBuilder.guidedTaskIndex || 0), 0, Math.max(taskSteps.length - 1, 0));
  const taskCardHTML = (step, index, prominent = false) => {
    const label = String(step.label || step.name || step.id || `Task ${index + 1}`);
    const duration = Number(step.duration_s || 0);
    const target = Number(step.target_windows || 0);
    const isCurrent = String(currentTask.label || currentTask.id || '') === String(step.label || step.id || '');
    return `
      <div class="pipeline-task-card ${isCurrent || prominent ? 'current' : ''}">
        <strong>${escapeHTML(`${index + 1}. ${label}`)}</strong>
        <span>${escapeHTML(`${duration || '--'} seconds${target ? ` | target ${target} clean windows` : ''}${isCurrent ? ' | current task' : ''}`)}</span>
      </div>
    `;
  };
  const primaryTaskIndex = taskSteps.findIndex(step => String(currentTask.label || currentTask.id || '') === String(step.label || step.id || ''));
  const displayTaskIndex = primaryTaskIndex >= 0 ? primaryTaskIndex : guidedIndex;
  const primaryTask = taskSteps[displayTaskIndex] ? taskCardHTML(taskSteps[displayTaskIndex], displayTaskIndex, true) : '';
  const upcomingTasks = taskSteps.slice(displayTaskIndex + 1).map((step, index) => taskCardHTML(step, displayTaskIndex + index + 1, false)).join('');
  const profileName = S.signalProfileName || 'signal';
  const sourceName = autopilot.mode || S.pipelineBuilder.source || S.streamSource || 'source';
  const channelCount = Math.max(0, (S.channelLabels || []).length || N_CH);
  const rmsValues = Array.isArray(S.rms) ? S.rms : [];
  const activeChannels = rmsValues.filter(v => Number(v || 0) > 0.001).length;
  const plotContext = activeChannels
    ? `${activeChannels}/${channelCount} ${profileName} channels are moving in the plot.`
    : `${channelCount} ${profileName} channels are visible in the plot.`;
  const healthWarnings = Array.isArray(S.signalHealth?.warnings) ? S.signalHealth.warnings.slice(0, 1) : [];
  const nowText = pipelineAutopilotNowText(status, phase, taskSteps, currentTask);
  const helperText = phase === 'setup'
    ? 'Attach the electrodes, confirm the channel bars are moving, then begin the guided tasks.'
    : (phase === 'tasks'
      ? 'Do only the current task. KYMA accepts clean windows and rejects noisy ones automatically.'
      : (phase === 'done'
        ? (pipelineBuildNeedsTraining() ? 'The model package is ready for review.' : 'The app code is ready for review.')
        : 'KYMA is checking the model and export readiness.'));
  const stepChip = (name, active, done) => `<div class="pipeline-step-chip ${active ? 'active' : ''} ${done ? 'done' : ''}">${escapeHTML(name)}</div>`;
  const lines = steps.map(step => {
    const stepStatus = String(step?.status || 'pending');
    const label = String(step?.label || step?.id || 'Step');
    const detail = String(step?.detail || '');
    const marker = stepStatus === 'done' ? 'done' : (stepStatus === 'running' ? 'running' : (stepStatus === 'failed' ? 'failed' : (stepStatus === 'needs_data' ? 'needs data' : 'pending')));
    return `${marker}: ${label}${detail ? ` - ${detail}` : ''}`;
  });
  return `
    <div class="pipeline-section pipeline-autopilot-card">
      <div class="pipeline-autopilot-top">
        <div class="pipeline-autopilot-kicker">
          <strong>Autopilot</strong>
          <div class="pipeline-pill-row">
            <span class="pipeline-pill ${tone}">${escapeHTML(status.replace(/_/g, ' '))}</span>
            ${autopilot.mode ? `<span class="pipeline-pill">${escapeHTML(String(autopilot.mode).replace(/_/g, ' '))}</span>` : ''}
          </div>
        </div>
        <div class="pipeline-now ${S.pipelineBuilder.autopilotLoading ? 'thinking' : ''}">${escapeHTML(nowText)}</div>
        <p class="pipeline-helper">${escapeHTML(`${plotContext} ${healthWarnings.length ? `${healthWarnings[0]} ` : ''}${helperText}`)}</p>
      </div>
      <div class="pipeline-step-strip">
        ${stepChip('1. Setup', phase === 'setup', ['tasks', 'review', 'done'].includes(phase))}
        ${stepChip('2. Tasks', phase === 'tasks', ['review', 'done'].includes(phase))}
        ${stepChip('3. Model', ['review', 'done'].includes(phase), phase === 'done')}
      </div>
      <div class="pipeline-focus">
        ${status === 'choose_setup' ? renderPipelineBuildChooserHTML() : ''}
        ${renderPipelineTrainingQualityHTML()}
        ${primaryTask ? `
          <strong>Task Queue</strong>
          <div class="pipeline-task-grid">${primaryTask}</div>
          ${upcomingTasks ? `
            <details class="pipeline-secondary-details">
              <summary>Show upcoming tasks</summary>
              <div class="pipeline-task-grid">${upcomingTasks}</div>
            </details>
          ` : ''}
          ${run.accepted_windows != null ? `<p class="pipeline-summary">${escapeHTML(`${run.accepted_windows || 0} accepted | ${run.rejected_windows || 0} rejected${run.last_reject_reason ? ` | last issue: ${run.last_reject_reason}` : ''}`)}</p>` : ''}
        ` : ''}
        ${autopilot.action === 'setup' ? `
          <div class="pipeline-action-row">
            <button class="scope-command-btn primary" type="button" onclick="window.startPipelineAutopilotStream()">Start Stream</button>
            <button class="scope-command-btn primary" type="button" onclick="window.continuePipelineAutopilotLive()">I Am Ready - Start Tasks</button>
          </div>
        ` : ''}
        ${autopilot.action === 'task_ready' ? `
          <div class="pipeline-task-prep">
            <strong>${escapeHTML(`Prepare for task ${displayTaskIndex + 1} of ${taskSteps.length || 1}`)}</strong>
            <span>${escapeHTML(taskSteps[displayTaskIndex]?.details?.[0] || `Get ready for ${taskSteps[displayTaskIndex]?.label || 'the next task'}. Press start when your electrodes and posture are ready.`)}</span>
            <div class="pipeline-action-row">
              <button class="scope-command-btn primary" type="button" onclick="window.startPipelineAutopilotTask()">Start This Task</button>
              <button class="scope-command-btn" type="button" onclick="window.runPipelineQA()">Check Signal First</button>
            </div>
          </div>
        ` : ''}
        ${autopilot.action === 'retry' ? `
          <div class="pipeline-action-row">
            <button class="scope-command-btn primary" type="button" onclick="window.continuePipelineAutopilotLive({ retry: true })">Retry Tasks</button>
            <button class="scope-command-btn" type="button" onclick="window.runPipelineQA()">Recheck Signal</button>
          </div>
        ` : ''}
        <details class="pipeline-secondary-details">
          <summary>Show technical details</summary>
          ${autopilot.summary && autopilot.summary !== nowText ? `<p class="pipeline-summary">${escapeHTML(autopilot.summary)}</p>` : ''}
          ${lines.length ? `<ul class="pipeline-list">${pipelineListHTML(lines)}</ul>` : ''}
        </details>
      </div>
    </div>
  `;
}

function renderPipelineModelManagerHTML(manager) {
  if (!manager) return '';
  const active = manager.active || S.promptModel?.server || {};
  const models = Array.isArray(manager.models) ? manager.models : [];
  const activePath = String(active.model_path || '');
  const activeLabel = active.loaded
    ? `${active.labels?.length || 0} labels | ${active.classifier || 'model'}`
    : 'No active prompt model loaded';
  const modelRows = models.slice(0, 12).map((model, index) => {
    const path = String(model.model_path || '');
    const isActive = !!model.active || (activePath && path === activePath);
    const labels = Array.isArray(model.labels) ? model.labels : [];
    const score = model.quality_score == null ? null : Number(model.quality_score);
    const cv = model.cv_balanced_accuracy == null ? null : Number(model.cv_balanced_accuracy) * 100;
    const qualityText = score == null ? (cv == null ? 'quality n/a' : `CV ${cv.toFixed(1)}%`) : `quality ${score.toFixed(1)}`;
    const tone = score == null ? '' : (score >= 70 ? 'good' : (score >= 50 ? 'warn' : 'bad'));
    const labelText = labels.slice(0, 5).join(', ') || 'no labels';
    const encodedPath = encodeURIComponent(path);
    return `
      <div class="pipeline-result-row">
        <strong>${escapeHTML(isActive ? 'Active model' : `Model ${index + 1}`)}</strong>
        <span class="${tone}">${escapeHTML(`${qualityText} | ${model.n_windows || 0} windows | ${labels.length} labels`)}</span>
      </div>
      <div class="pipeline-result-row">
        <strong>${escapeHTML(labelText)}</strong>
        <span>${escapeHTML(model.modified_at || model.created_at || '')}</span>
      </div>
      <div class="pipeline-pill-row">
        <span class="pipeline-pill ${isActive ? 'good' : ''}">${escapeHTML(isActive ? 'loaded' : (model.ready === false ? 'review' : 'saved'))}</span>
        <button class="scope-command-btn" type="button" onclick="window.loadPipelinePromptModel(decodeURIComponent('${escapeHTML(encodedPath)}'))" ${isActive ? 'disabled' : ''}>Load</button>
        <button class="scope-command-btn" type="button" onclick="window.deletePipelinePromptModel(decodeURIComponent('${escapeHTML(encodedPath)}'))">Delete</button>
      </div>
    `;
  }).join('');
  return `
    <div class="pipeline-section">
      <strong>Saved Models</strong>
      <div class="pipeline-pill-row">
        <span class="pipeline-pill ${active.loaded ? 'good' : 'warn'}">${escapeHTML(active.loaded ? 'active' : 'none loaded')}</span>
        <span class="pipeline-pill">${escapeHTML(activeLabel)}</span>
        <span class="pipeline-pill">${escapeHTML(`${models.length} saved`)}</span>
      </div>
      ${active.model_path ? `<p class="pipeline-summary">${escapeHTML(active.model_path)}</p>` : ''}
      ${modelRows || '<p class="pipeline-summary">No saved prompt models for the active signal profile yet.</p>'}
    </div>
  `;
}

function renderPipelineAcquisitionHTML(acquisition) {
  if (!acquisition) return '';
  const run = S.pipelineBuilder.acquisitionRun || {};
  const current = run.current_step || {};
  const perLabel = run.per_label || {};
  const counts = Object.keys(perLabel).map(label => `${label}: ${perLabel[label]} accepted`);
  const steps = Array.isArray(acquisition.steps) ? acquisition.steps : [];
  const rows = steps.slice(0, 10).map(step => `
    <div class="pipeline-result-row pipeline-recipe-row">
      <strong>${escapeHTML(step.label || step.id || 'Step')}</strong>
      <span>${escapeHTML(`${step.duration_s || 0}s | ${String(step.action || '').replace(/_/g, ' ')}`)}</span>
    </div>
  `).join('');
  return `
    <div class="pipeline-section">
      <strong>Guided Acquisition</strong>
      <div class="pipeline-pill-row">
        <span class="pipeline-pill good">${escapeHTML(acquisition.mode || 'guided')}</span>
        <span class="pipeline-pill">${escapeHTML(`${acquisition.estimated_minutes || 0} min`)}</span>
        ${run.id ? `<span class="pipeline-pill ${run.running ? 'good' : 'warn'}">${escapeHTML(run.running ? 'running' : 'stopped')}</span>` : ''}
      </div>
      ${run.id ? `
        <p class="pipeline-summary">${escapeHTML(`Current: ${current.label || 'complete'} | ${run.accepted_windows || 0} accepted | ${run.rejected_windows || 0} rejected | ${run.step_remaining_s ?? '--'}s left`)}</p>
      ` : ''}
      ${rows || '<p class="pipeline-summary">No acquisition steps generated.</p>'}
      ${counts.length ? `<ul class="pipeline-list">${pipelineListHTML(counts)}</ul>` : ''}
      <ul class="pipeline-list">${pipelineListHTML(acquisition.automation || [])}</ul>
    </div>
  `;
}

function renderPipelineLabelSuggestionsHTML(labels) {
  if (!labels) return '';
  const suggestions = Array.isArray(labels.suggestions) ? labels.suggestions : [];
  const rows = suggestions.map(item => `
    <div class="pipeline-result-row">
      <strong>${escapeHTML(item.label || 'label')}</strong>
      <span>${escapeHTML(`${item.kind || 'suggestion'} | ${Number(item.confidence || 0).toFixed(2)} | ${item.channel || 'all'}`)}</span>
    </div>
  `).join('');
  const reasons = suggestions.map(item => `${item.label || 'label'}: ${item.reason || ''}`);
  return `
    <div class="pipeline-section">
      <strong>Auto-Label Suggestions</strong>
      ${rows || '<p class="pipeline-summary">No label suggestions yet.</p>'}
      <ul class="pipeline-list">${pipelineListHTML(reasons.concat(labels.policy || []))}</ul>
    </div>
  `;
}

function getPipelineSchemaColumns(dataset = S.pipelineBuilder.dataset) {
  const files = Array.isArray(dataset?.files) ? dataset.files : [];
  const file = files.find(item => Array.isArray(item?.schema?.columns) && item.schema.columns.length);
  return Array.isArray(file?.schema?.columns) ? file.schema.columns : [];
}

function ensurePipelineSchemaMapping(dataset = S.pipelineBuilder.dataset) {
  const columns = getPipelineSchemaColumns(dataset);
  if (!columns.length) return null;
  const existing = S.pipelineBuilder.schemaMapping || {};
  const roles = { ...(existing.roles || {}) };
  columns.forEach(col => {
    const name = String(col.name || '');
    if (name && !roles[name]) roles[name] = String(col.role || 'ignore');
  });
  const mapping = { roles };
  mapping.channels = columns
    .filter(col => roles[String(col.name || '')] === 'channel')
    .map(col => String(col.name || ''))
    .filter(Boolean)
    .slice(0, N_CH);
  mapping.label = columns.find(col => roles[String(col.name || '')] === 'label')?.name || '';
  mapping.time = columns.find(col => roles[String(col.name || '')] === 'time')?.name || '';
  S.pipelineBuilder.schemaMapping = mapping;
  return mapping;
}

function buildPipelineSchemaMappingPayload() {
  const mapping = ensurePipelineSchemaMapping();
  return mapping || {};
}

function renderPipelineSchemaMapperHTML(dataset) {
  const columns = getPipelineSchemaColumns(dataset);
  if (!columns.length) return '';
  const mapping = ensurePipelineSchemaMapping(dataset) || { roles: {} };
  const options = ['ignore', 'time', 'channel', 'label', 'event', 'metadata'];
  const rows = columns.slice(0, 16).map(col => {
    const name = String(col.name || '');
    const role = String((mapping.roles || {})[name] || col.role || 'ignore');
    const opts = options.map(opt => `<option value="${opt}" ${opt === role ? 'selected' : ''}>${opt}</option>`).join('');
    const detail = `${Math.round(Number(col.numeric_coverage || 0) * 100)}% numeric | ${col.unique_values || 0} unique`;
    return `
      <div class="pipeline-result-row">
        <strong>${escapeHTML(name)}</strong>
        <span>${escapeHTML(detail)}</span>
        <select class="pipeline-mini-select pipeline-schema-role" data-column="${escapeHTML(name)}">${opts}</select>
      </div>
    `;
  }).join('');
  const detected = dataset?.files?.find(item => item?.schema)?.schema?.detected || {};
  const summary = [
    `${(mapping.channels || []).length} channel column(s)`,
    mapping.label ? `label: ${mapping.label}` : 'label not selected',
    mapping.time ? `time: ${mapping.time}` : 'time not selected',
  ];
  return `
    <div class="pipeline-section">
      <strong>Schema Mapper</strong>
      <div class="pipeline-pill-row">
        <span class="pipeline-pill ${(mapping.channels || []).length ? 'good' : 'warn'}">${escapeHTML(`${(mapping.channels || []).length}/${N_CH} channels`)}</span>
        <span class="pipeline-pill ${mapping.label ? 'good' : 'warn'}">${escapeHTML(mapping.label ? 'label mapped' : 'label needed')}</span>
        ${detected.channels?.length ? `<span class="pipeline-pill">${escapeHTML('auto mapped')}</span>` : ''}
      </div>
      ${rows}
      <ul class="pipeline-list">${pipelineListHTML(summary)}</ul>
    </div>
  `;
}

function applyPipelineRecipeDraftToPlan() {
  const plan = S.pipelineBuilder.plan;
  const draft = S.pipelineBuilder.recipeDraft || {};
  if (!plan) return plan;
  const next = { ...plan };
  const windowing = { ...(next.windowing || {}) };
  if (Number.isFinite(Number(draft.window_size_ms))) windowing.window_size_ms = Number(draft.window_size_ms);
  if (Number.isFinite(Number(draft.window_step_ms))) windowing.window_step_ms = Number(draft.window_step_ms);
  const sampleRate = Number(windowing.sample_rate_hz || S.sampleRate || 250);
  if (Number.isFinite(sampleRate) && sampleRate > 0) {
    if (Number.isFinite(Number(windowing.window_size_ms))) windowing.window_samples = Math.max(1, Math.round(Number(windowing.window_size_ms) * sampleRate / 1000));
    if (Number.isFinite(Number(windowing.window_step_ms))) windowing.step_samples = Math.max(1, Math.round(Number(windowing.window_step_ms) * sampleRate / 1000));
  }
  next.windowing = windowing;
  if (draft.model_id) {
    next.selected_model_id = String(draft.model_id);
  }
  if (draft.export_target) {
    next.selected_export_target = String(draft.export_target);
  }
  if (Array.isArray(draft.filters)) {
    next.filters = draft.filters
      .filter(stage => stage && stage.kind && stage.kind !== 'none')
      .map(stage => ({
        kind: String(stage.kind || 'bandpass'),
        low_hz: Number(stage.low_hz || 0) || null,
        high_hz: Number(stage.high_hz || 0) || null,
        cutoff_hz: Number(stage.cutoff_hz || 0) || null,
        order: Math.max(1, Math.round(Number(stage.order || 2))),
      }));
  }
  if (Array.isArray(draft.labels)) {
    next.label_protocol = {
      ...(next.label_protocol || {}),
      labels: draft.labels
        .filter(item => String(item.name || '').trim())
        .map(item => ({
          name: String(item.name || '').trim(),
          target_windows: Math.max(1, Math.round(Number(item.target_windows || 25))),
          min_seconds: Math.max(1, Math.round(Number(item.min_seconds || 8))),
          cue: String(item.cue || `Collect clean examples for ${item.name || 'label'}.`),
        })),
    };
    next.task = {
      ...(next.task || {}),
      labels: draft.labels.map(item => String(item.name || '').trim()).filter(Boolean),
    };
  }
  S.pipelineBuilder.plan = next;
  return next;
}

function ensurePipelineRecipeDraft(plan = S.pipelineBuilder.plan) {
  if (!plan) return null;
  const windowing = plan.windowing || {};
  const existing = S.pipelineBuilder.recipeDraft || {};
  const candidates = Array.isArray(plan.model_candidates) ? plan.model_candidates : [];
  const filters = Array.isArray(existing.filters)
    ? existing.filters
    : (Array.isArray(plan.filters) ? plan.filters : []).map(stage => ({
      kind: String(stage.kind || 'bandpass'),
      low_hz: stage.low_hz ?? '',
      high_hz: stage.high_hz ?? '',
      cutoff_hz: stage.cutoff_hz ?? '',
      order: stage.order ?? 2,
    }));
  const sourceLabels = Array.isArray((plan.label_protocol || {}).labels)
    ? (plan.label_protocol || {}).labels
    : [];
  const labels = Array.isArray(existing.labels)
    ? existing.labels
    : sourceLabels.map(item => ({
      name: String(item.name || ''),
      target_windows: Number(item.target_windows || 25),
      min_seconds: Number(item.min_seconds || 8),
      cue: String(item.cue || ''),
    }));
  S.pipelineBuilder.recipeDraft = {
    window_size_ms: existing.window_size_ms ?? windowing.window_size_ms ?? 200,
    window_step_ms: existing.window_step_ms ?? windowing.window_step_ms ?? 50,
    model_id: existing.model_id || candidates[0]?.id || '',
    export_target: existing.export_target || 'browser_onnx',
    filters,
    labels,
  };
  return S.pipelineBuilder.recipeDraft;
}

function validatePipelineRecipe(plan, draft) {
  const warnings = [];
  const sampleRate = Number((plan?.windowing || {}).sample_rate_hz || S.sampleRate || 250);
  const nyquist = sampleRate / 2;
  const windowMs = Number(draft.window_size_ms || 0);
  const stepMs = Number(draft.window_step_ms || 0);
  if (!windowMs || windowMs < 20) warnings.push('Window must be at least 20 ms.');
  if (!stepMs || stepMs < 5) warnings.push('Step must be at least 5 ms.');
  if (stepMs > windowMs) warnings.push('Step should not exceed the window size.');
  (draft.filters || []).forEach((stage, idx) => {
    const kind = String(stage.kind || '');
    const low = Number(stage.low_hz || 0);
    const high = Number(stage.high_hz || 0);
    const cutoff = Number(stage.cutoff_hz || 0);
    if (['bandpass', 'bandstop'].includes(kind) && (!low || !high || low >= high)) warnings.push(`Filter ${idx + 1} needs low < high.`);
    if (['bandpass', 'bandstop'].includes(kind) && high >= nyquist) warnings.push(`Filter ${idx + 1} exceeds Nyquist ${nyquist.toFixed(0)} Hz.`);
    if (['highpass', 'lowpass'].includes(kind) && (!cutoff || cutoff >= nyquist)) warnings.push(`Filter ${idx + 1} needs cutoff below ${nyquist.toFixed(0)} Hz.`);
  });
  if (!(draft.labels || []).filter(item => String(item.name || '').trim()).length) warnings.push('At least one label is required.');
  if (!draft.model_id) warnings.push('Choose a model candidate.');
  if (String(draft.export_target || '').includes('browser') && !S.pipelineBuilder.trainResult?.trained && !S.promptModel?.server?.loaded) {
    warnings.push('Browser export needs a trained prompt model.');
  }
  return warnings;
}

function renderPipelineRecipeEditorHTML(plan) {
  if (!plan) return '';
  const draft = ensurePipelineRecipeDraft(plan) || {};
  const candidates = Array.isArray(plan.model_candidates) ? plan.model_candidates : [];
  const modelOptions = candidates.length
    ? candidates.map(item => `<option value="${escapeHTML(item.id || item.name || '')}" ${(draft.model_id === (item.id || item.name)) ? 'selected' : ''}>${escapeHTML(item.name || item.id || 'model')}</option>`).join('')
    : '<option value="">Baseline</option>';
  const exportOptions = [
    ['browser_onnx', 'Browser + ONNX'],
    ['report_only', 'Report only'],
    ['dataset_manifest', 'Dataset manifest'],
    ['software_control', 'Software control'],
    ['firmware_handoff', 'Firmware handoff'],
  ].map(([value, label]) => `<option value="${value}" ${draft.export_target === value ? 'selected' : ''}>${label}</option>`).join('');
  const filterKinds = ['bandpass', 'bandstop', 'highpass', 'lowpass'];
  const filterRows = (draft.filters || []).map((stage, idx) => {
    const kindOptions = filterKinds.map(kind => `<option value="${kind}" ${stage.kind === kind ? 'selected' : ''}>${kind}</option>`).join('');
    return `
      <div class="pipeline-result-row">
        <strong>Filter ${idx + 1}</strong>
        <select class="pipeline-mini-select pipeline-recipe-filter" data-filter-index="${idx}" data-filter-field="kind">${kindOptions}</select>
        <input class="pipeline-mini-input pipeline-recipe-filter" data-filter-index="${idx}" data-filter-field="low_hz" type="number" placeholder="low" value="${escapeHTML(stage.low_hz ?? '')}">
        <input class="pipeline-mini-input pipeline-recipe-filter" data-filter-index="${idx}" data-filter-field="high_hz" type="number" placeholder="high" value="${escapeHTML(stage.high_hz ?? '')}">
        <input class="pipeline-mini-input pipeline-recipe-filter" data-filter-index="${idx}" data-filter-field="cutoff_hz" type="number" placeholder="cutoff" value="${escapeHTML(stage.cutoff_hz ?? '')}">
        <button class="scope-command-btn" type="button" data-recipe-action="remove-filter" data-filter-index="${idx}">Remove</button>
      </div>
    `;
  }).join('');
  const labelRows = (draft.labels || []).map((item, idx) => `
    <div class="pipeline-result-row">
      <input class="pipeline-mini-input pipeline-recipe-label" data-label-index="${idx}" data-label-field="name" type="text" value="${escapeHTML(item.name || '')}" placeholder="label">
      <input class="pipeline-mini-input pipeline-recipe-label" data-label-index="${idx}" data-label-field="target_windows" type="number" min="1" value="${escapeHTML(item.target_windows || 25)}">
      <input class="pipeline-mini-input pipeline-recipe-label" data-label-index="${idx}" data-label-field="min_seconds" type="number" min="1" value="${escapeHTML(item.min_seconds || 8)}">
      <button class="scope-command-btn" type="button" data-recipe-action="remove-label" data-label-index="${idx}">Remove</button>
    </div>
  `).join('');
  const warnings = validatePipelineRecipe(plan, draft);
  const labelCount = (draft.labels || []).filter(item => String(item.name || '').trim()).length;
  const summary = [
    `Run: ${draft.window_size_ms} ms windows every ${draft.window_step_ms} ms.`,
    `Train: ${candidates.find(item => (item.id || item.name) === draft.model_id)?.name || draft.model_id || 'baseline model'} on ${labelCount} label(s).`,
    `Filters: ${(draft.filters || []).length || 0} stage(s).`,
    `Export: ${String(draft.export_target || 'browser_onnx').replace(/_/g, ' ')}.`,
  ];
  return `
    <details class="pipeline-section pipeline-recipe-details">
      <summary>Pipeline Recipe</summary>
      <div class="pipeline-recipe-grid">
        <label class="pipeline-field">
          <span>Window ms</span>
          <input class="pipeline-mini-input pipeline-recipe-input" data-recipe-field="window_size_ms" type="number" min="20" max="5000" step="10" value="${escapeHTML(draft.window_size_ms)}">
        </label>
        <label class="pipeline-field">
          <span>Step ms</span>
          <input class="pipeline-mini-input pipeline-recipe-input" data-recipe-field="window_step_ms" type="number" min="5" max="5000" step="5" value="${escapeHTML(draft.window_step_ms)}">
        </label>
        <label class="pipeline-field">
          <span>Model</span>
          <select class="pipeline-mini-select pipeline-recipe-input" data-recipe-field="model_id">${modelOptions}</select>
        </label>
        <label class="pipeline-field">
          <span>Export</span>
          <select class="pipeline-mini-select pipeline-recipe-input" data-recipe-field="export_target">${exportOptions}</select>
        </label>
      </div>
      <div class="pipeline-pill-row">
        <span class="pipeline-pill good">${escapeHTML(`${draft.window_size_ms} ms`)}</span>
        <span class="pipeline-pill">${escapeHTML(`${draft.window_step_ms} ms step`)}</span>
        <span class="pipeline-pill">${escapeHTML(String(draft.export_target || 'browser_onnx').replace(/_/g, ' '))}</span>
        <span class="pipeline-pill ${warnings.length ? 'warn' : 'good'}">${escapeHTML(warnings.length ? `${warnings.length} warning${warnings.length === 1 ? '' : 's'}` : 'valid')}</span>
      </div>
      <div class="pipeline-section">
        <strong>Filters</strong>
        ${filterRows || '<p class="pipeline-summary">No filters configured.</p>'}
        <button class="scope-command-btn" type="button" data-recipe-action="add-filter">Add Filter</button>
      </div>
      <div class="pipeline-section">
        <strong>Labels</strong>
        ${labelRows || '<p class="pipeline-summary">No labels configured.</p>'}
        <button class="scope-command-btn" type="button" data-recipe-action="add-label">Add Label</button>
      </div>
      <div class="pipeline-section">
        <strong>Run Summary</strong>
        <ul class="pipeline-list">${pipelineListHTML(summary.concat(warnings))}</ul>
      </div>
    </details>
  `;
}

function renderPipelineDatasetHTML(dataset) {
  const ingest = S.pipelineBuilder.ingest || null;
  if (!dataset && !ingest) return '';
  const files = Array.isArray(dataset?.files) ? dataset.files : [];
  const readiness = dataset?.readiness || null;
  const readinessScore = Number(readiness?.score || 0);
  const readinessTone = readiness?.ready ? 'good' : (readinessScore >= 50 ? 'warn' : 'bad');
  const rows = files.slice(0, 8).map(file => `
    <div class="pipeline-result-row">
      <strong>${escapeHTML(file.name || 'file')}</strong>
      <span class="${file.trainable ? 'good' : (file.supported ? 'warn' : 'bad')}">${escapeHTML(`${file.extension || ''} | ${file.trainable ? `${file.estimated_windows || 0} trainable windows` : (file.supported ? 'supported' : 'review')} | ${file.bytes || 0} B`)}</span>
    </div>
  `).join('');
  const readinessCounts = readiness?.labels || {};
  const readinessRows = Object.keys(readinessCounts).slice(0, 8).map(label => `
    <div class="pipeline-result-row">
      <strong>${escapeHTML(label)}</strong>
      <span class="${Number(readinessCounts[label] || 0) >= 5 ? 'good' : 'warn'}">${escapeHTML(`${readinessCounts[label]} estimated windows`)}</span>
    </div>
  `).join('');
  const ingestCounts = ingest?.per_label || {};
  const ingestRows = Object.keys(ingestCounts).map(label => `
    <div class="pipeline-result-row">
      <strong>${escapeHTML(label)}</strong>
      <span class="good">${escapeHTML(`${ingestCounts[label]} windows`)}</span>
    </div>
  `).join('');
  return `
    <div class="pipeline-section">
      <strong>Dataset Import Scan</strong>
      <div class="pipeline-pill-row">
        ${dataset ? `<span class="pipeline-pill">${escapeHTML(dataset.mode || 'path')}</span>` : ''}
        ${dataset ? `<span class="pipeline-pill ${dataset.supported_count ? 'good' : 'warn'}">${escapeHTML(`${dataset.supported_count || 0}/${dataset.file_count || 0} supported`)}</span>` : ''}
        ${readiness ? `<span class="pipeline-pill ${readinessTone}">${escapeHTML(`${readinessScore.toFixed(0)}/100 readiness`)}</span>` : ''}
        ${readiness ? `<span class="pipeline-pill ${readiness.ready ? 'good' : 'warn'}">${escapeHTML(readiness.ready ? 'trainable' : 'needs data')}</span>` : ''}
        ${ingest ? `<span class="pipeline-pill ${ingest.ready ? 'good' : 'warn'}">${escapeHTML(`${ingest.added_windows || 0} ingested`)}</span>` : ''}
      </div>
      <p class="pipeline-summary">${escapeHTML(ingest?.run_dir || dataset?.path || '')}</p>
      ${rows || (dataset ? '<p class="pipeline-summary">No importable files found.</p>' : '')}
      ${renderPipelineSchemaMapperHTML(dataset)}
      ${readinessRows ? `<div class="pipeline-model-quality">${readinessRows}</div>` : ''}
      ${ingestRows ? `<div class="pipeline-model-quality">${ingestRows}</div>` : ''}
      ${readiness?.gaps?.length ? `<ul class="pipeline-list">${pipelineListHTML(readiness.gaps)}</ul>` : ''}
      ${ingest?.warnings?.length ? `<ul class="pipeline-list">${pipelineListHTML(ingest.warnings)}</ul>` : ''}
      <ul class="pipeline-list">${pipelineListHTML(dataset?.pipeline || [])}</ul>
    </div>
  `;
}

function renderPipelineComparisonHTML(comparison) {
  if (!comparison) return '';
  const rows = (Array.isArray(comparison.comparisons) ? comparison.comparisons : []).map(item => `
    <div class="pipeline-result-row">
      <strong>${escapeHTML(item.name || 'Model')}</strong>
      <span>${escapeHTML(`${item.status || 'candidate'} | ${Number(item.score || 0).toFixed(0)}/100`)}</span>
    </div>
  `).join('');
  const failures = (Array.isArray(comparison.failure_cases) ? comparison.failure_cases : [])
    .map(item => `${item.type || 'case'}: ${item.detail || ''}`);
  return `
    <div class="pipeline-section">
      <strong>Model Comparison</strong>
      ${rows || '<p class="pipeline-summary">No model comparison yet.</p>'}
      <ul class="pipeline-list">${pipelineListHTML(failures.length ? failures : ['No failure cases logged yet.'])}</ul>
    </div>
  `;
}

function renderPipelineThroughputHTML(throughput) {
  if (!throughput) return '';
  const results = Array.isArray(throughput.results) ? throughput.results : [];
  const rows = results.map(item => `
    <div class="pipeline-result-row">
      <strong>${escapeHTML(item.name || 'codec')}</strong>
      <span>${escapeHTML(`${Number(item.ratio || 0).toFixed(3)}x | ${Number(item.roundtrip_ms || 0).toFixed(2)} ms | err ${Number(item.max_abs_error || 0).toFixed(4)}`)}</span>
    </div>
  `).join('');
  const recommended = throughput.recommended || {};
  return `
    <div class="pipeline-section">
      <strong>Speed & Compression</strong>
      <div class="pipeline-pill-row">
        <span class="pipeline-pill good">${escapeHTML(recommended.name || 'benchmark')}</span>
        <span class="pipeline-pill">${escapeHTML(`${throughput.raw_bytes || 0} raw B`)}</span>
      </div>
      ${rows || '<p class="pipeline-summary">No throughput benchmark yet.</p>'}
      <ul class="pipeline-list">${pipelineListHTML(throughput.architecture || [])}</ul>
    </div>
  `;
}

function renderPipelineRuntimeHTML(runtime) {
  const deploy = S.pipelineBuilder.deploy || null;
  if (!runtime && !deploy) return '';
  runtime = runtime || deploy?.runtime || {};
  const ready = !!runtime.ready;
  const browserCount = Number(runtime.browser_model_count || 0);
  const onnxCount = Number(runtime.onnx_model_count || 0);
  const deployRows = deploy ? [
    `Deploy smoke: ${deploy.status || (deploy.ready ? 'passed' : 'review')}`,
    `Sample: ${(deploy.sample_shape || []).join(' x ') || 'n/a'}`,
    ...(Array.isArray(deploy.browser_results) ? deploy.browser_results.slice(0, 2).map(item => `Browser ${item.label || 'model'} ${(Number(item.confidence || 0) * 100).toFixed(1)}%`) : []),
    ...(Array.isArray(deploy.onnx_results) ? deploy.onnx_results.slice(0, 2).map(item => `ONNX ${item.ok ? 'validated' : 'review'} ${item.output_shapes ? JSON.stringify(item.output_shapes) : ''}`) : []),
    ...(Array.isArray(deploy.errors) ? deploy.errors.slice(0, 3) : []),
  ] : [];
  const lines = [
    `Backend: ${runtime.recommended_backend || 'onnxruntime-web/wasm'}`,
    `WebGPU: ${runtime.client_webgpu_available ? 'available' : 'not available'}`,
    `ONNX models: ${onnxCount}`,
    `Browser linear models: ${browserCount}`,
    `Status: ${String(runtime.status || '').replace(/_/g, ' ')}`,
    runtime.export_dir ? `Runtime assets: ${runtime.export_dir}` : '',
    runtime.onnx_error ? `ONNX note: ${runtime.onnx_error}` : '',
  ];
  return `
    <div class="pipeline-section">
      <strong>Browser Runtime</strong>
      <div class="pipeline-pill-row">
        <span class="pipeline-pill ${ready ? 'good' : 'warn'}">${ready ? 'ready' : 'not ready'}</span>
        ${deploy ? `<span class="pipeline-pill ${deploy.ready ? 'good' : 'warn'}">${escapeHTML(deploy.ready ? 'smoke passed' : 'smoke review')}</span>` : ''}
        <span class="pipeline-pill ${browserCount ? 'good' : 'warn'}">${escapeHTML(`${browserCount} browser`)}</span>
        <span class="pipeline-pill ${onnxCount ? 'good' : ''}">${escapeHTML(`${onnxCount} ONNX`)}</span>
        <span class="pipeline-pill">${escapeHTML(runtime.recommended_backend || 'runtime')}</span>
      </div>
      <ul class="pipeline-list">${pipelineListHTML(lines.filter(Boolean).concat(deployRows).concat(runtime.next_actions || []))}</ul>
    </div>
  `;
}

function renderPipelineEmbeddingHTML(embedding) {
  if (!embedding) return '';
  const readiness = embedding.readiness || {};
  const latest = readiness.latest_embedding || {};
  const score = Number(readiness.score || 0);
  const tone = readiness.ready ? 'good' : (score >= 50 ? 'warn' : 'bad');
  const lines = [
    `${readiness.embedding_count || 0} embedding points`,
    latest.name ? `${latest.name}: ${latest.embedding_dim || 0}-d, norm ${Number(latest.embedding_norm || 0).toFixed(2)}` : 'No latest embedding yet.',
    `${readiness.total_windows || 0} labeled windows across ${readiness.label_count || 0} labels`,
  ];
  return `
    <div class="pipeline-section">
      <strong>Dataset Readiness</strong>
      <div class="pipeline-pill-row">
        <span class="pipeline-pill ${tone}">${score.toFixed(1)} / 100</span>
        <span class="pipeline-pill ${readiness.ready ? 'good' : 'warn'}">${readiness.ready ? 'export ready' : 'needs data'}</span>
      </div>
      <ul class="pipeline-list">${pipelineListHTML(lines.concat(readiness.gaps || []))}</ul>
    </div>
  `;
}

function renderPipelineExportHTML(exportResult) {
  if (!exportResult) return '';
  const files = exportResult.files || {};
  const names = Object.keys(files).map(name => `${name}: ${files[name]}`);
  const browserFiles = names.filter(name => name.includes('.browser.json') || name.includes('browser_runtime.js'));
  const standalone = files['index.html'] || '';
  return `
    <div class="pipeline-section">
      <strong>Export Package</strong>
      <p class="pipeline-summary">${escapeHTML(exportResult.export_dir || 'No export directory.')}</p>
      ${browserFiles.length || standalone ? `<div class="pipeline-pill-row">
        ${browserFiles.length ? `<span class="pipeline-pill good">${escapeHTML(`${browserFiles.length} browser runtime files`)}</span>` : ''}
        ${standalone ? '<span class="pipeline-pill good">standalone self-test</span>' : ''}
      </div>` : ''}
      <ul class="pipeline-list">${pipelineListHTML(names)}</ul>
    </div>
  `;
}

function pipelineCodeWorkbenchContext() {
  const train = S.pipelineBuilder.trainResult || {};
  const modelPath = String(train.model_path || train.result?.model_path || S.promptModel?.server?.model_path || '').replaceAll('\\', '/');
  const exportDir = String(S.pipelineBuilder.exportResult?.export_dir || '').replaceAll('\\', '/');
  const promptGoal = pipelinePlainGoal();
  const appName = promptGoal.replace(/[^a-z0-9]+/gi, '-').replace(/^-|-$/g, '').toLowerCase() || 'kyma-app';
  const appType = String(S.pipelineBuilder.appType || 'dashboard');
  return { train, modelPath, exportDir, promptGoal, appName, appType };
}

let pipelineCodeTypingTimer = null;

function generatedKymaClientJS() {
  return `export function connectKyma({ onSamples, onPrediction, onStatus, onLog } = {}) {\n  const ws = new WebSocket('ws://127.0.0.1:8007/ws');\n  ws.onopen = () => onStatus?.('connected');\n  ws.onclose = () => onStatus?.('disconnected');\n  ws.onerror = () => onStatus?.('error');\n  ws.onmessage = (event) => {\n    const msg = JSON.parse(event.data);\n    if (msg.type === 'emg') {\n      onSamples?.(msg.data);\n    }\n    if (msg.type === 'prompt_prediction') {\n      onPrediction?.(msg.data);\n    }\n    if (msg.type === 'ml_insights') {\n      onLog?.('ML insight received');\n    }\n    if (msg.type === 'ping') {\n      ws.send(JSON.stringify({ type: 'pong' }));\n    }\n  };\n  return ws;\n}\n`;
}

function generatedPredictionAppHTML(title, appCopy, appType) {
  return `<!doctype html>\n<html>\n<head>\n  <meta charset="utf-8">\n  <title>${title}</title>\n  <style>\n    body { margin:0; font-family:Inter,Segoe UI,sans-serif; background:#f6f4ef; color:#20242c; }\n    main { max-width:760px; margin:0 auto; padding:28px; }\n    .status { display:inline-flex; padding:6px 10px; border-radius:999px; background:#e8ebff; color:#4256bf; font-weight:800; font-size:12px; }\n    .prediction { margin-top:18px; padding:18px; border:1px solid #d9dde8; border-radius:14px; background:white; }\n    .label { font-size:38px; font-weight:850; }\n    .confidence { color:#667085; font-weight:700; }\n    pre { padding:12px; border-radius:10px; background:#111827; color:#e5edff; overflow:auto; }\n  </style>\n</head>\n<body>\n  <main>\n    <span id="status" class="status">connecting</span>\n    <h1>${title}</h1>\n    <p>${appCopy}</p>\n    <section class="prediction">\n      <div id="label" class="label">waiting</div>\n      <div id="confidence" class="confidence">confidence --</div>\n    </section>\n    <pre id="log">Booting KYMA client...</pre>\n  </main>\n  <script type="module">\n    import { connectKyma } from './kyma-client.js';\n    const status = document.getElementById('status');\n    const label = document.getElementById('label');\n    const confidence = document.getElementById('confidence');\n    const log = document.getElementById('log');\n    const appType = '${appType}';\n    connectKyma({\n      onStatus: value => status.textContent = value,\n      onLog: value => log.textContent += '\\n' + value,\n      onPrediction: prediction => {\n        const detected = prediction.label || 'unknown';\n        label.textContent = appType === 'alert' ? 'trigger: ' + detected : detected;\n        confidence.textContent = 'confidence ' + Math.round((prediction.confidence || 0) * 100) + '%';\n        log.textContent = JSON.stringify({ appType, prediction }, null, 2);\n      },\n    });\n  </script>\n</body>\n</html>\n`;
}

function generatedFrequencyBandAppHTML(title, appCopy) {
  return `<!doctype html>\n<html>\n<head>\n  <meta charset="utf-8">\n  <title>${title}</title>\n  <style>\n    :root { color-scheme: light; }\n    body { margin:0; font-family:Inter,Segoe UI,sans-serif; background:#f3f0ea; color:#20242c; }\n    main { max-width:1040px; margin:0 auto; padding:26px; }\n    header { display:flex; align-items:flex-start; justify-content:space-between; gap:18px; margin-bottom:18px; }\n    h1 { margin:0 0 7px; font-size:30px; letter-spacing:0; }\n    p { margin:0; color:#667085; font-weight:650; line-height:1.45; }\n    .status { display:inline-flex; padding:7px 11px; border-radius:999px; background:#e8ebff; color:#4256bf; font-weight:850; font-size:12px; white-space:nowrap; }\n    .layout { display:grid; grid-template-columns:minmax(0, 1.15fr) minmax(280px, .85fr); gap:16px; }\n    section { border:1px solid #d9dde8; border-radius:12px; background:white; padding:14px; }\n    canvas { width:100%; height:310px; display:block; border-radius:10px; background:#111827; }\n    .bands { display:grid; gap:10px; }\n    .band { display:grid; grid-template-columns:92px 1fr 62px; align-items:center; gap:10px; font-size:12px; font-weight:800; }\n    .bar { height:16px; border-radius:999px; background:#eef1f6; overflow:hidden; }\n    .fill { height:100%; width:0%; border-radius:999px; background:#586fda; transition:width .12s linear; }\n    .band:nth-child(2) .fill { background:#4f8f69; }\n    .band:nth-child(3) .fill { background:#b28327; }\n    .band:nth-child(4) .fill { background:#c15c70; }\n    .band:nth-child(5) .fill { background:#8664d6; }\n    .meta { margin-top:12px; color:#667085; font-size:12px; font-weight:700; line-height:1.45; }\n    .readout { margin-top:10px; padding:11px; border-radius:10px; background:#f7f8fb; color:#344054; font-size:12px; line-height:1.45; white-space:pre-wrap; }\n    @media (max-width:820px) { .layout { grid-template-columns:1fr; } header { flex-direction:column; } }\n  </style>\n</head>\n<body>\n  <main>\n    <header>\n      <div>\n        <h1>${title}</h1>\n        <p>${appCopy}</p>\n      </div>\n      <span id="status" class="status">connecting</span>\n    </header>\n    <div class="layout">\n      <section>\n        <canvas id="spectrum" width="900" height="360"></canvas>\n        <div id="meta" class="meta">Waiting for EMG samples from KYMA.</div>\n      </section>\n      <section>\n        <div class="bands" id="bands"></div>\n        <div id="readout" class="readout">Band power will update from live EMG chunks.</div>\n      </section>\n    </div>\n  </main>\n  <script type="module">\n    import { connectKyma } from './kyma-client.js';\n    const status = document.getElementById('status');\n    const meta = document.getElementById('meta');\n    const readout = document.getElementById('readout');\n    const canvas = document.getElementById('spectrum');\n    const ctx = canvas.getContext('2d');\n    const sampleRate = 250;\n    const fftSize = 256;\n    const buffer = Array.from({ length: 8 }, () => []);\n    const bands = [\n      { key:'motion', label:'0-20 Hz', low:0, high:20 },\n      { key:'low', label:'20-45 Hz', low:20, high:45 },\n      { key:'mid', label:'45-75 Hz', low:45, high:75 },\n      { key:'high', label:'75-120 Hz', low:75, high:120 },\n      { key:'line', label:'58-62 Hz', low:58, high:62 },\n    ];\n    const bandEls = new Map();\n    document.getElementById('bands').innerHTML = bands.map(b => '<div class="band"><span>' + b.label + '</span><div class="bar"><div class="fill" data-band="' + b.key + '"></div></div><span data-value="' + b.key + '">0</span></div>').join('');\n    bands.forEach(b => bandEls.set(b.key, {\n      fill: document.querySelector('[data-band="' + b.key + '"]'),\n      value: document.querySelector('[data-value="' + b.key + '"]'),\n    }));\n    function dftPower(samples) {\n      const mean = samples.reduce((a, b) => a + b, 0) / Math.max(samples.length, 1);\n      const powers = [];\n      for (let k = 0; k <= fftSize / 2; k += 1) {\n        let re = 0;\n        let im = 0;\n        for (let n = 0; n < fftSize; n += 1) {\n          const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (fftSize - 1));\n          const x = ((samples[n] || 0) - mean) * w;\n          const angle = (2 * Math.PI * k * n) / fftSize;\n          re += x * Math.cos(angle);\n          im -= x * Math.sin(angle);\n        }\n        powers.push({ hz: k * sampleRate / fftSize, power: Math.sqrt(re * re + im * im) / fftSize });\n      }\n      return powers;\n    }\n    function bandPower(spectrum, low, high) {\n      const bins = spectrum.filter(p => p.hz >= low && p.hz < high);\n      return bins.reduce((sum, p) => sum + p.power, 0) / Math.max(bins.length, 1);\n    }\n    function drawSpectrum(spectrum) {\n      ctx.clearRect(0, 0, canvas.width, canvas.height);\n      ctx.fillStyle = '#111827';\n      ctx.fillRect(0, 0, canvas.width, canvas.height);\n      const max = Math.max(...spectrum.map(p => p.power), 1e-6);\n      ctx.strokeStyle = '#8ea2ff';\n      ctx.lineWidth = 2;\n      ctx.beginPath();\n      spectrum.forEach((p, i) => {\n        const x = (p.hz / 125) * canvas.width;\n        const y = canvas.height - (p.power / max) * (canvas.height - 26) - 12;\n        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);\n      });\n      ctx.stroke();\n      ctx.fillStyle = '#d8e1ff';\n      ctx.font = '12px Segoe UI';\n      [20,45,75,120].forEach(hz => {\n        const x = (hz / 125) * canvas.width;\n        ctx.fillRect(x, 0, 1, canvas.height);\n        ctx.fillText(hz + ' Hz', x + 5, 18);\n      });\n    }\n    function updateFromSamples(payload) {\n      const channels = payload.channels || [];\n      channels.slice(0, 8).forEach((values, ch) => {\n        buffer[ch].push(...values.map(Number));\n        if (buffer[ch].length > fftSize) buffer[ch].splice(0, buffer[ch].length - fftSize);\n      });\n      const channel = buffer.findIndex(v => v.length >= fftSize);\n      if (channel < 0) return;\n      const samples = buffer[channel].slice(-fftSize);\n      const spectrum = dftPower(samples);\n      const values = bands.map(b => ({ ...b, value: bandPower(spectrum, b.low, b.high) }));\n      const max = Math.max(...values.map(v => v.value), 1e-6);\n      values.forEach(v => {\n        const pct = Math.min(100, (v.value / max) * 100);\n        bandEls.get(v.key).fill.style.width = pct.toFixed(1) + '%';\n        bandEls.get(v.key).value.textContent = v.value.toFixed(2);\n      });\n      drawSpectrum(spectrum);\n      meta.textContent = 'Analyzing channel ' + (channel + 1) + ' at ' + sampleRate + ' Hz with ' + fftSize + ' samples.';\n      readout.textContent = values.map(v => v.label + ': ' + v.value.toFixed(3)).join('\\n');\n    }\n    connectKyma({\n      onStatus: value => status.textContent = value,\n      onSamples: updateFromSamples,\n      onLog: value => readout.textContent = value,\n    });\n  </script>\n</body>\n</html>\n`;
}

function generatedPipelineCodeFiles() {
  const { modelPath, exportDir, promptGoal, appName, appType } = pipelineCodeWorkbenchContext();
  const title = promptGoal.replace(/\b\w/g, ch => ch.toUpperCase());
  const wantsFrequency = pipelinePromptRequestsFrequencyApp();
  const appLabel = pipelineAppTypeLabel(appType);
  const appCopy = wantsFrequency
    ? 'This app splits the live EMG stream into motion, low, mid, high, and line-noise frequency bands.'
    : (appType === 'alert'
    ? 'This app turns KYMA predictions into clear trigger events for another product.'
    : (appType === 'control'
      ? 'This app maps KYMA predictions into control-ready actions.'
      : (appType === 'research'
        ? 'This app presents model quality, status, and export context for research review.'
        : 'This app shows live predictions, confidence, and stream status.')));
  const indexHTML = wantsFrequency ? generatedFrequencyBandAppHTML(title, appCopy) : generatedPredictionAppHTML(title, appCopy, appType);
  return [
    {
      path: `apps/${appName}/README.md`,
      language: 'markdown',
      content: `# ${title}\n\nGenerated KYMA ${appLabel.toLowerCase()} for a trained biosignal model.\n\n## Run checks\n\n1. Keep KYMA running at http://127.0.0.1:8007.\n2. Open src/index.html in a browser or use the preview panel.\n3. Verify that prompt_prediction events appear after the live stream starts.\n\nModel path: ${modelPath || 'not exported yet'}\nExport directory: ${exportDir || 'not exported yet'}\n`,
    },
    {
      path: `apps/${appName}/src/kyma-client.js`,
      language: 'javascript',
      content: generatedKymaClientJS(),
    },
    {
      path: `apps/${appName}/src/index.html`,
      language: 'html',
      content: indexHTML,
    },
    {
      path: `apps/${appName}/tests/live-smoke.spec.js`,
      language: 'javascript',
      content: `import { test, expect } from '@playwright/test';\n\ntest('KYMA server exposes status', async ({ request }) => {\n  const response = await request.get('http://127.0.0.1:8007/api/status');\n  expect(response.ok()).toBeTruthy();\n  const status = await response.json();\n  expect(status.signal_profile.key).toBeTruthy();\n});\n\ntest('KYMA QA endpoint responds', async ({ request }) => {\n  const response = await request.post('http://127.0.0.1:8007/api/pipeline/qa', { data: { plan: {} } });\n  expect([200, 400]).toContain(response.status());\n});\n`,
    },
  ];
}

function ensurePipelineCodeWorkbench({ force = false } = {}) {
  const wb = S.pipelineBuilder.codeWorkbench || {};
  const { modelPath, exportDir, appName, appType } = pipelineCodeWorkbenchContext();
  const key = `${appName}|${appType}|${modelPath}|${exportDir}`;
  if (force || wb.key !== key || !Array.isArray(wb.files) || !wb.files.length) {
    const files = generatedPipelineCodeFiles();
    S.pipelineBuilder.codeWorkbench = {
      files,
      selectedPath: files[1]?.path || files[0]?.path || '',
      logs: [
        'Writing app files from the trained KYMA pipeline.',
        'Use Preview after the code is ready.',
      ],
      previewHTML: '',
      key,
      mode: 'code',
      typedPath: files[1]?.path || files[0]?.path || '',
      typedChars: 0,
    };
  }
  return S.pipelineBuilder.codeWorkbench;
}

function selectedPipelineCodeFile() {
  const wb = S.pipelineBuilder.codeWorkbenchOpen ? ensurePipelineCodeWorkbench() : (S.pipelineBuilder.codeWorkbench || {});
  return wb.files.find(file => file.path === wb.selectedPath) || wb.files[0] || null;
}

function startPipelineCodeWorkbench({ force = false } = {}) {
  S.pipelineBuilder.codeWorkbenchOpen = true;
  const wb = ensurePipelineCodeWorkbench({ force });
  wb.mode = 'code';
  wb.typedPath = wb.selectedPath || wb.files[0]?.path || '';
  wb.typedChars = force ? 0 : Math.min(Number(wb.typedChars || 0), selectedPipelineCodeFile()?.content?.length || 0);
  S.pipelineBuilder.codeWorkbench = wb;
  if (pipelineCodeTypingTimer) clearInterval(pipelineCodeTypingTimer);
  pipelineCodeTypingTimer = setInterval(() => {
    const current = S.pipelineBuilder.codeWorkbench || {};
    if (!S.pipelineBuilder.codeWorkbenchOpen) {
      clearInterval(pipelineCodeTypingTimer);
      pipelineCodeTypingTimer = null;
      return;
    }
    const file = selectedPipelineCodeFile();
    const full = String(file?.content || '');
    const next = Math.min(full.length, Number(current.typedChars || 0) + Math.max(16, Math.ceil(full.length / 80)));
    current.typedPath = file?.path || current.typedPath || '';
    current.typedChars = next;
    S.pipelineBuilder.codeWorkbench = current;
    syncPipelineBuilderUI();
    if (next >= full.length) {
      clearInterval(pipelineCodeTypingTimer);
      pipelineCodeTypingTimer = null;
      addPipelineCodeLog(`Finished writing ${file?.path || 'selected file'}.`);
      syncPipelineBuilderUI();
    }
  }, 34);
  syncPipelineBuilderUI();
}

function addPipelineCodeLog(line) {
  const wb = ensurePipelineCodeWorkbench();
  wb.logs = [String(line || '')].concat(Array.isArray(wb.logs) ? wb.logs : []).slice(0, 12);
  S.pipelineBuilder.codeWorkbench = wb;
}

function runPipelineCodePreview() {
  const wb = ensurePipelineCodeWorkbench();
  const htmlFile = wb.files.find(file => file.path.endsWith('/src/index.html')) || selectedPipelineCodeFile();
  const client = wb.files.find(file => file.path.endsWith('/src/kyma-client.js'))?.content || '';
  const bundledClient = client.replace('export function connectKyma', 'function connectKyma');
  wb.previewHTML = (htmlFile?.content || '<!doctype html><body><main>No preview file.</main></body>')
    .replace("import { connectKyma } from './kyma-client.js';", bundledClient);
  wb.mode = 'preview';
  S.pipelineBuilder.codeWorkbench = wb;
  addPipelineCodeLog(`Preview rendered from ${htmlFile?.path || 'selected file'}.`);
  syncPipelineBuilderUI();
}

async function runPipelineCodeSmoke() {
  const wb = ensurePipelineCodeWorkbench();
  const paths = wb.files.map(file => file.path);
  const client = wb.files.find(file => file.path.endsWith('kyma-client.js'))?.content || '';
  const html = wb.files.find(file => file.path.endsWith('index.html'))?.content || '';
  const missing = [
    paths.some(path => path.endsWith('README.md')) ? '' : 'README.md missing',
    client.includes('prompt_prediction') ? '' : 'client does not listen for prompt_prediction',
    html.includes('connectKyma') ? '' : 'preview HTML is not wired to connectKyma',
  ].filter(Boolean);
  try {
    const status = await get('/api/status');
    addPipelineCodeLog(`Server OK: ${status.signal_profile?.display_name || status.signal_profile?.key || 'signal'} profile.`);
  } catch (e) {
    missing.push('KYMA status endpoint is not reachable');
  }
  if (missing.length) {
    addPipelineCodeLog(`Smoke failed: ${missing.join('; ')}`);
    toast('Code smoke needs review', 'yellow');
  } else {
    addPipelineCodeLog('Smoke passed: generated files and KYMA server are ready.');
    toast('Code smoke passed');
  }
  syncPipelineBuilderUI();
}

function downloadSelectedPipelineCodeFile() {
  const file = selectedPipelineCodeFile();
  if (!file) return;
  const blob = new Blob([file.content || ''], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = file.path.split('/').pop() || 'kyma-file.txt';
  a.click();
  URL.revokeObjectURL(url);
  addPipelineCodeLog(`Downloaded ${file.path}.`);
  syncPipelineBuilderUI();
}

function pipelineCodeCommand(kind) {
  if (kind === 'qa') return 'Invoke-RestMethod http://127.0.0.1:8007/api/pipeline/qa -Method Post -ContentType "application/json" -Body "{\\"plan\\":{}}" | ConvertTo-Json -Depth 6';
  return 'Invoke-RestMethod http://127.0.0.1:8007/api/status | ConvertTo-Json -Depth 4';
}

async function copyPipelineCodeCommand(kind) {
  const command = pipelineCodeCommand(kind);
  try {
    await navigator.clipboard.writeText(command);
    addPipelineCodeLog(`Copied ${kind === 'qa' ? 'QA' : 'status'} command.`);
    toast('Command copied');
  } catch {
    addPipelineCodeLog(command);
    toast('Command added to logs', 'yellow');
  }
  syncPipelineBuilderUI();
}

function renderPipelineCodeWorkbenchHTML() {
  if (!S.pipelineBuilder.codeWorkbenchOpen) return '';
  const train = S.pipelineBuilder.trainResult || {};
  if (!train?.trained && !S.pipelineBuilder.exportResult && !S.promptModel?.server?.loaded) return '';
  const wb = ensurePipelineCodeWorkbench();
  const selected = selectedPipelineCodeFile();
  const previewHTML = wb.previewHTML || '<!doctype html><body style="font-family:Segoe UI,sans-serif;padding:18px"><strong>Preview not running.</strong><p>Click Preview to render the generated app.</p></body>';
  const statusCmd = 'Invoke-RestMethod http://127.0.0.1:8007/api/status | ConvertTo-Json -Depth 4';
  const qaCmd = 'Invoke-RestMethod http://127.0.0.1:8007/api/pipeline/qa -Method Post -ContentType "application/json" -Body "{\\"plan\\":{}}" | ConvertTo-Json -Depth 6';
  const fileTabs = wb.files.map(file => `
    <button class="pipeline-file-tab ${file.path === selected?.path ? 'active' : ''}" type="button" data-code-file="${escapeHTML(file.path)}">${escapeHTML(file.path)}</button>
  `).join('');
  const logs = (Array.isArray(wb.logs) ? wb.logs : []).join('\n');
  const fullContent = String(selected?.content || '');
  const typedContent = selected?.path === wb.typedPath ? fullContent.slice(0, Math.max(0, Number(wb.typedChars || 0))) : fullContent;
  const typing = typedContent.length < fullContent.length;
  const mode = wb.mode === 'preview' ? 'preview' : 'code';
  return `
    <div class="pipeline-section">
      <strong>Generated App</strong>
      <div class="pipeline-code-panel">
        <div class="pipeline-code-steps">
          <div class="pipeline-pill-row">
            <span class="pipeline-pill ${typing ? 'warn' : 'good'}">${typing ? 'writing' : 'ready'}</span>
            <span class="pipeline-pill">${escapeHTML(String(wb.files.length))} files</span>
          </div>
          <div class="pipeline-code-toolbar">
            <button class="scope-command-btn" type="button" data-code-action="regenerate">Regenerate</button>
            <button class="scope-command-btn ${mode === 'code' ? 'primary' : ''}" type="button" data-code-action="show-code">Code</button>
            <button class="scope-command-btn ${mode === 'preview' ? 'primary' : ''}" type="button" data-code-action="run-preview">Preview</button>
            <button class="scope-command-btn" type="button" data-code-action="run-smoke">Run Smoke</button>
          </div>
          <div class="pipeline-file-list">${fileTabs}</div>
          <div class="pipeline-code-log">${escapeHTML(logs || `Status command:\n${statusCmd}\n\nQA command:\n${qaCmd}`)}</div>
        </div>
        <div class="pipeline-code-editor pipeline-code-view ${mode === 'code' ? 'active' : ''}">
          <strong>${escapeHTML(selected?.path || 'No file selected')}</strong>
          <textarea class="pipeline-code-textarea" spellcheck="false" data-code-editor="${escapeHTML(selected?.path || '')}">${escapeHTML(typedContent)}</textarea>
          <div class="pipeline-code-toolbar">
            <button class="scope-command-btn" type="button" data-code-action="download">Download File</button>
          </div>
        </div>
        <div class="pipeline-code-preview pipeline-code-view ${mode === 'preview' ? 'active' : ''}">
          <strong>Preview</strong>
          <iframe title="KYMA generated app preview" sandbox="allow-scripts allow-same-origin" srcdoc="${escapeHTML(previewHTML)}"></iframe>
        </div>
      </div>
    </div>
  `;
}

function pipelineExportDemoUrl(path) {
  return `/api/pipeline/exports/demo?path=${encodeURIComponent(String(path || ''))}`;
}

function pipelineExportDownloadUrl(path) {
  return `/api/pipeline/exports/download?path=${encodeURIComponent(String(path || ''))}`;
}

function renderPipelineExportManagerHTML(manager) {
  if (!manager) return '';
  const exports = Array.isArray(manager.exports) ? manager.exports : [];
  const verified = manager.lastVerify || null;
  const deploy = verified?.deploy || null;
  const rows = exports.slice(0, 8).map((item, idx) => {
    const path = String(item.path || '');
    const encodedPath = encodeURIComponent(path);
    const modelPath = String(item.model_path || '');
    const encodedModelPath = encodeURIComponent(modelPath);
    const status = item.error ? 'error' : (item.ready ? 'ready' : 'review');
    const detail = [
      `${item.browser_count || 0} browser`,
      `${item.onnx_count || 0} ONNX`,
      item.has_index ? 'demo' : 'no demo',
      item.modified_at || '',
    ].filter(Boolean).join(' | ');
    return `
      <div class="pipeline-result-row">
        <strong>${escapeHTML(item.name || `Package ${idx + 1}`)}</strong>
        <span class="${item.ready ? 'good' : (item.error ? 'bad' : 'warn')}">${escapeHTML(`${status} | ${detail}`)}</span>
      </div>
      <div class="pipeline-pill-row">
        ${item.has_index ? `<button class="scope-command-btn" type="button" onclick="window.openPipelineExportDemo(decodeURIComponent('${escapeHTML(encodedPath)}'))">Open Demo</button>` : ''}
        <button class="scope-command-btn" type="button" onclick="window.verifyPipelineExportPackage(decodeURIComponent('${escapeHTML(encodedPath)}'))">Verify</button>
        ${modelPath ? `<button class="scope-command-btn" type="button" onclick="window.loadPipelinePromptModel(decodeURIComponent('${escapeHTML(encodedModelPath)}'))">Load Model</button>` : ''}
        <button class="scope-command-btn" type="button" onclick="window.copyPipelineExportPath(decodeURIComponent('${escapeHTML(encodedPath)}'))">Copy Path</button>
        <button class="scope-command-btn" type="button" onclick="window.deletePipelineExportPackage(decodeURIComponent('${escapeHTML(encodedPath)}'))">Delete</button>
      </div>
    `;
  }).join('');
  const latest = exports[0] || null;
  const checklist = latest ? [
    `${latest.has_manifest ? 'Manifest ready' : 'Manifest missing'}`,
    `${latest.has_report ? 'Report ready' : 'Report missing'}`,
    `${latest.has_index ? 'Standalone demo ready' : 'Standalone demo missing'}`,
    `${latest.browser_count || 0} browser runtime model(s)`,
    `${latest.onnx_count || 0} ONNX model(s)`,
  ] : ['No export packages found yet.'];
  const verifyLines = deploy ? [
    `Deploy smoke: ${deploy.status || (deploy.ready ? 'passed' : 'review')}`,
    `Sample: ${(deploy.sample_shape || []).join(' x ') || 'n/a'}`,
    ...(Array.isArray(deploy.browser_results) ? deploy.browser_results.slice(0, 2).map(item => `Browser ${item.label || 'model'} ${(Number(item.confidence || 0) * 100).toFixed(1)}%`) : []),
    ...(Array.isArray(deploy.onnx_results) ? deploy.onnx_results.slice(0, 2).map(item => `ONNX ${item.ok ? 'validated' : 'review'} ${item.output_shapes ? JSON.stringify(item.output_shapes) : ''}`) : []),
    ...(Array.isArray(deploy.errors) ? deploy.errors.slice(0, 3) : []),
  ] : [];
  return `
    <div class="pipeline-section">
      <strong>Export Packages</strong>
      <div class="pipeline-pill-row">
        <span class="pipeline-pill ${exports.length ? 'good' : 'warn'}">${escapeHTML(`${exports.length} packages`)}</span>
        ${latest ? `<span class="pipeline-pill ${latest.ready ? 'good' : 'warn'}">${escapeHTML(latest.ready ? 'latest ready' : 'latest review')}</span>` : ''}
        ${deploy ? `<span class="pipeline-pill ${deploy.ready ? 'good' : 'warn'}">${escapeHTML(deploy.ready ? 'smoke passed' : 'smoke review')}</span>` : ''}
        ${manager.export_dir ? `<span class="pipeline-pill">${escapeHTML('registry')}</span>` : ''}
      </div>
      ${latest ? `<p class="pipeline-summary">${escapeHTML(latest.path || '')}</p>` : ''}
      ${rows || '<p class="pipeline-summary">No export packages created yet.</p>'}
      ${verifyLines.length ? `<ul class="pipeline-list">${pipelineListHTML(verifyLines)}</ul>` : ''}
      <ul class="pipeline-list">${pipelineListHTML(checklist)}</ul>
    </div>
  `;
}

function buildPipelineProjectSnapshot() {
  syncPipelineBuilderInputsFromForm();
  const plan = applyPipelineRecipeDraftToPlan() || S.pipelineBuilder.plan || {};
  const recipe = ensurePipelineRecipeDraft(plan) || S.pipelineBuilder.recipeDraft || {};
  return {
    name: plan.title || S.pipelineBuilder.prompt || 'Biosignal project',
    project_id: S.pipelineBuilder.activeProject?.project_id || '',
    profile: S.signalProfileKey || S.signalProfileName || 'biosignal',
    prompt: S.pipelineBuilder.prompt || '',
    source: S.pipelineBuilder.source || 'live',
    output: S.pipelineBuilder.output || 'live_model',
    dataset_path: S.pipelineBuilder.datasetPath || '',
    schema_mapping: S.pipelineBuilder.schemaMapping || {},
    recipe,
    plan,
    qa: S.pipelineBuilder.qa || {},
    train_result: S.pipelineBuilder.trainResult || {},
    acquisition: S.pipelineBuilder.acquisition || {},
    acquisition_run: S.pipelineBuilder.acquisitionRun || {},
    labels: S.pipelineBuilder.labelSuggestions || {},
    dataset: S.pipelineBuilder.dataset || {},
    ingest: S.pipelineBuilder.ingest || {},
    comparison: S.pipelineBuilder.comparison || {},
    runtime: S.pipelineBuilder.runtime || {},
    deploy: S.pipelineBuilder.deploy || {},
    embedding: S.pipelineBuilder.embedding || {},
    throughput: S.pipelineBuilder.throughput || {},
    export_result: S.pipelineBuilder.exportResult || {},
    model_manager: S.pipelineBuilder.modelManager || {},
    export_manager: S.pipelineBuilder.exportManager || {},
    saved_from: {
      stream_source: S.streamSource || '',
      sample_rate_hz: S.sampleRate || 0,
      channel_labels: (S.channelLabels || []).slice(0, N_CH),
      browser: navigator.userAgent || '',
    },
  };
}

function restorePipelineProjectSnapshot(snapshot = {}) {
  S.pipelineBuilder.prompt = String(snapshot.prompt || '');
  S.pipelineBuilder.source = String(snapshot.source || 'live');
  S.pipelineBuilder.output = String(snapshot.output || 'live_model');
  S.pipelineBuilder.datasetPath = String(snapshot.dataset_path || '');
  S.pipelineBuilder.schemaMapping = snapshot.schema_mapping || null;
  S.pipelineBuilder.recipeDraft = snapshot.recipe || snapshot.recipeDraft || null;
  S.pipelineBuilder.plan = snapshot.plan || null;
  S.pipelineBuilder.qa = snapshot.qa || null;
  S.pipelineBuilder.trainResult = snapshot.train_result || null;
  S.pipelineBuilder.acquisition = snapshot.acquisition || null;
  S.pipelineBuilder.acquisitionRun = snapshot.acquisition_run || null;
  S.pipelineBuilder.labelSuggestions = snapshot.labels || null;
  S.pipelineBuilder.dataset = snapshot.dataset || null;
  S.pipelineBuilder.ingest = snapshot.ingest || null;
  S.pipelineBuilder.comparison = snapshot.comparison || null;
  S.pipelineBuilder.runtime = snapshot.runtime || null;
  S.pipelineBuilder.deploy = snapshot.deploy || null;
  S.pipelineBuilder.embedding = snapshot.embedding || null;
  S.pipelineBuilder.throughput = snapshot.throughput || null;
  S.pipelineBuilder.exportResult = snapshot.export_result || null;
  S.pipelineBuilder.modelManager = snapshot.model_manager || S.pipelineBuilder.modelManager;
  S.pipelineBuilder.exportManager = snapshot.export_manager || S.pipelineBuilder.exportManager;
  localStorage.setItem('kyma-pipeline-source', S.pipelineBuilder.source);
  localStorage.setItem('kyma-pipeline-output', S.pipelineBuilder.output);
  localStorage.setItem('kyma-pipeline-dataset-path', S.pipelineBuilder.datasetPath);
  if (S.pipelineBuilder.plan && S.pipelineBuilder.recipeDraft) applyPipelineRecipeDraftToPlan();
}

function setPipelineRebuildProgress(label, status = 'running') {
  const now = new Date().toLocaleTimeString();
  const list = Array.isArray(S.pipelineBuilder.rebuildProgress) ? S.pipelineBuilder.rebuildProgress : [];
  const next = [...list, { label: String(label || ''), status: String(status || 'running'), at: now }].slice(-12);
  S.pipelineBuilder.rebuildProgress = next;
  S.pipelineBuilder.createStage = label;
  syncPipelineBuilderUI();
}

function syncPipelineRebuildJob(job) {
  if (!job) return;
  S.pipelineBuilder.rebuildJob = job;
  S.pipelineBuilder.rebuildProgress = Array.isArray(job.steps)
    ? job.steps.map(item => ({
      label: item.detail ? `${item.step}: ${item.detail}` : item.step,
      status: item.status || job.status || 'running',
      at: item.at || '',
    })).slice(-12)
    : [];
  S.pipelineBuilder.createStage = job.current_step || job.status || S.pipelineBuilder.createStage;
}

function pipelineRegistryFilterText(value) {
  return String(value || '').trim().toLowerCase();
}

function pipelineProjectSearchText(project = {}) {
  return [
    project.project_id,
    project.name,
    project.profile,
    project.source,
    project.output,
    project.dataset_path,
    ...(Array.isArray(project.labels) ? project.labels : []),
  ].join(' ').toLowerCase();
}

function pipelineJobSearchText(job = {}) {
  return [
    job.job_id,
    job.project_id,
    job.status,
    job.current_step,
    job.error,
    job.result?.report_path,
    job.result?.project?.dataset_path,
  ].join(' ').toLowerCase();
}

function pipelineProjectMatchesFilters(project, filters) {
  const query = pipelineRegistryFilterText(filters.query);
  const status = String(filters.status || 'all');
  const format = String(filters.format || 'all');
  if (query && !pipelineProjectSearchText(project).includes(query)) return false;
  if (status !== 'all') {
    const projectStatus = project.ready ? 'ready' : (project.trained ? 'review' : 'draft');
    if (projectStatus !== status) return false;
  }
  if (format !== 'all') {
    const haystack = pipelineProjectSearchText(project);
    if (!haystack.includes(format.toLowerCase())) return false;
  }
  return true;
}

function pipelineJobMatchesFilters(job, filters) {
  const query = pipelineRegistryFilterText(filters.query);
  const status = String(filters.status || 'all');
  const format = String(filters.format || 'all');
  if (query && !pipelineJobSearchText(job).includes(query)) return false;
  if (status !== 'all' && String(job.status || '') !== status) return false;
  if (format !== 'all') {
    const haystack = pipelineJobSearchText(job);
    if (!haystack.includes(format.toLowerCase())) return false;
  }
  return true;
}

function openPipelineProjectReport(projectId) {
  if (!projectId) return;
  window.open(`/api/pipeline/projects/report/${encodeURIComponent(String(projectId || ''))}`, '_blank', 'noopener');
}

function openPipelineRebuildReport(jobId) {
  if (!jobId) return;
  window.open(`/api/pipeline/projects/rebuild/${encodeURIComponent(String(jobId || ''))}/report`, '_blank', 'noopener');
}

function renderPipelineProjectRegistryHTML(registry) {
  if (!registry) return '';
  const filters = S.pipelineBuilder.registryFilters || { query: '', status: 'all', format: 'all' };
  const allProjects = Array.isArray(registry.projects) ? registry.projects : [];
  const allJobs = Array.isArray(registry.rebuild_jobs) ? registry.rebuild_jobs : [];
  const projects = allProjects.filter(project => pipelineProjectMatchesFilters(project, filters));
  const jobs = allJobs.filter(job => pipelineJobMatchesFilters(job, filters));
  const active = S.pipelineBuilder.activeProject || null;
  const rows = projects.slice(0, 8).map((project, idx) => {
    const id = String(project.project_id || '');
    const encodedId = encodeURIComponent(id);
    const labels = Array.isArray(project.labels) ? project.labels.length : 0;
    const acc = Number(project.cv_balanced_accuracy);
    const score = Number.isFinite(acc) ? `${(acc * 100).toFixed(0)}% bal acc` : (project.trained ? 'trained' : 'not trained');
    const state = project.ready ? 'ready' : (project.trained ? 'review' : 'draft');
    const tone = project.ready ? 'good' : (project.trained ? 'warn' : '');
    const reportPath = String(project.report_path || '');
    const encodedReportPath = encodeURIComponent(reportPath);
    const packagePath = String(project.export_dir || project.export_path || project.package_path || '');
    const encodedPackagePath = encodeURIComponent(packagePath);
    return `
      <div class="pipeline-result-row pipeline-project-row">
        <strong>${escapeHTML(project.name || `Project ${idx + 1}`)}</strong>
        <span class="${project.ready ? 'good' : (project.trained ? 'warn' : '')}">${escapeHTML(`${state} | ${project.n_windows || 0} windows | ${labels} labels | ${score}`)}</span>
      </div>
      <div class="pipeline-pill-row">
        <span class="pipeline-pill ${tone}">${escapeHTML(project.source || 'source')}</span>
        <span class="pipeline-pill">${escapeHTML(`${project.recipe_versions || 0} recipes`)}</span>
        ${project.dataset_path ? `<span class="pipeline-pill">${escapeHTML('dataset')}</span>` : ''}
        ${active?.project_id === id ? `<span class="pipeline-pill good">${escapeHTML('loaded')}</span>` : ''}
        <button class="scope-command-btn" type="button" onclick="window.loadPipelineProject(decodeURIComponent('${escapeHTML(encodedId)}'))">Load</button>
        <button class="scope-command-btn" type="button" onclick="window.rebuildPipelineProject(decodeURIComponent('${escapeHTML(encodedId)}'))">Rebuild</button>
        ${reportPath ? `<button class="scope-command-btn" type="button" onclick="window.openPipelineProjectReport(decodeURIComponent('${escapeHTML(encodedId)}'))">Open Report</button>` : ''}
        ${reportPath ? `<button class="scope-command-btn" type="button" onclick="window.copyPipelineRegistryPath(decodeURIComponent('${escapeHTML(encodedReportPath)}'))">Copy Report</button>` : ''}
        ${packagePath ? `<button class="scope-command-btn" type="button" onclick="window.downloadPipelineExportPackage(decodeURIComponent('${escapeHTML(encodedPackagePath)}'))">Download Package</button>` : ''}
        ${packagePath ? `<button class="scope-command-btn" type="button" onclick="window.copyPipelineRegistryPath(decodeURIComponent('${escapeHTML(encodedPackagePath)}'))">Copy Package</button>` : ''}
        <button class="scope-command-btn" type="button" onclick="window.deletePipelineProject(decodeURIComponent('${escapeHTML(encodedId)}'))">Delete</button>
      </div>
    `;
  }).join('');
  const jobRows = jobs.slice(0, 8).map((job, idx) => {
    const id = String(job.job_id || '');
    const encodedId = encodeURIComponent(id);
    const status = String(job.status || 'unknown');
    const tone = status === 'completed' ? 'good' : (['failed', 'interrupted'].includes(status) ? 'bad' : 'warn');
    const steps = Array.isArray(job.steps) ? job.steps.length : 0;
    const reportPath = String(job.result?.report_path || '');
    const encodedReportPath = encodeURIComponent(reportPath);
    const packagePath = String(job.result?.export_dir || job.result?.project?.export_dir || job.result?.project?.package_path || '');
    const encodedPackagePath = encodeURIComponent(packagePath);
    return `
      <div class="pipeline-result-row pipeline-project-row">
        <strong>${escapeHTML(job.project_id || `Job ${idx + 1}`)}</strong>
        <span class="${tone}">${escapeHTML(`${status} | ${steps} steps | ${job.updated_at || job.created_at || ''}`)}</span>
      </div>
      <div class="pipeline-pill-row">
        <span class="pipeline-pill ${tone}">${escapeHTML(status)}</span>
        <span class="pipeline-pill">${escapeHTML(job.current_step || 'queued')}</span>
        <button class="scope-command-btn" type="button" onclick="window.loadPipelineRebuildJob(decodeURIComponent('${escapeHTML(encodedId)}'))">Details</button>
        ${reportPath ? `<button class="scope-command-btn" type="button" onclick="window.openPipelineRebuildReport(decodeURIComponent('${escapeHTML(encodedId)}'))">Open Report</button>` : ''}
        ${reportPath ? `<button class="scope-command-btn" type="button" onclick="window.copyPipelineRegistryPath(decodeURIComponent('${escapeHTML(encodedReportPath)}'))">Copy Report</button>` : ''}
        ${packagePath ? `<button class="scope-command-btn" type="button" onclick="window.downloadPipelineExportPackage(decodeURIComponent('${escapeHTML(encodedPackagePath)}'))">Download Package</button>` : ''}
        ${packagePath ? `<button class="scope-command-btn" type="button" onclick="window.copyPipelineRegistryPath(decodeURIComponent('${escapeHTML(encodedPackagePath)}'))">Copy Package</button>` : ''}
        ${['completed', 'failed', 'interrupted'].includes(status) ? `<button class="scope-command-btn" type="button" onclick="window.deletePipelineRebuildJob(decodeURIComponent('${escapeHTML(encodedId)}'))">Delete Job</button>` : ''}
      </div>
    `;
  }).join('');
  const importers = Array.isArray(registry.importer_status) ? registry.importer_status : [];
  const importerLines = importers
    .filter(item => String(filters.format || 'all') === 'all' || String(item.format || '').toLowerCase().includes(String(filters.format || '').toLowerCase()))
    .map(item => `${item.format}: ${item.status} - ${item.detail}`);
  const evaluation = active?.evaluation || {};
  const report = String(active?.report || '').trim();
  const progress = Array.isArray(S.pipelineBuilder.rebuildProgress) ? S.pipelineBuilder.rebuildProgress : [];
  const progressRows = progress.map(item => `${item.at} | ${item.status}: ${item.label}`);
  return `
    <div class="pipeline-section">
      <strong>Project Registry</strong>
      <div class="pipeline-pill-row">
        <span class="pipeline-pill ${projects.length ? 'good' : 'warn'}">${escapeHTML(`${projects.length}/${allProjects.length} projects`)}</span>
        <span class="pipeline-pill ${jobs.length ? 'good' : ''}">${escapeHTML(`${jobs.length}/${allJobs.length} jobs`)}</span>
        ${active ? `<span class="pipeline-pill good">${escapeHTML(active.project_id || 'active')}</span>` : ''}
        ${registry.project_dir ? `<span class="pipeline-pill">${escapeHTML('versioned recipes')}</span>` : ''}
      </div>
      <div class="pipeline-recipe-grid">
        <input class="pipeline-mini-input pipeline-registry-filter" data-registry-filter="query" type="search" placeholder="Search projects, labels, jobs" value="${escapeHTML(filters.query || '')}">
        <select class="pipeline-mini-select pipeline-registry-filter" data-registry-filter="status">
          ${[
            ['all', 'All status'],
            ['ready', 'Ready projects'],
            ['review', 'Review projects'],
            ['draft', 'Draft projects'],
            ['completed', 'Completed jobs'],
            ['failed', 'Failed jobs'],
            ['interrupted', 'Interrupted jobs'],
            ['running', 'Running jobs'],
            ['queued', 'Queued jobs'],
          ].map(([value, label]) => `<option value="${escapeHTML(value)}" ${String(filters.status || 'all') === value ? 'selected' : ''}>${escapeHTML(label)}</option>`).join('')}
        </select>
        <select class="pipeline-mini-select pipeline-registry-filter" data-registry-filter="format">
          ${[
            ['all', 'All formats'],
            ['csv', 'CSV'],
            ['npz', 'NPZ'],
            ['edf', 'EDF/BDF'],
            ['xdf', 'XDF'],
            ['mat', 'MAT/WAV'],
            ['hdf5', 'HDF5'],
            ['parquet', 'Parquet'],
          ].map(([value, label]) => `<option value="${escapeHTML(value)}" ${String(filters.format || 'all') === value ? 'selected' : ''}>${escapeHTML(label)}</option>`).join('')}
        </select>
      </div>
      ${active ? `
        <div class="pipeline-result-row">
          <strong>${escapeHTML(active.name || 'Active project')}</strong>
          <span class="${evaluation.ready ? 'good' : 'warn'}">${escapeHTML(`${evaluation.n_windows || 0} windows | ${(evaluation.labels || []).length} labels | ready ${evaluation.ready}`)}</span>
        </div>
      ` : ''}
      <div class="pipeline-pill-row">
        <button class="scope-command-btn" type="button" onclick="window.savePipelineProject()">Save Current</button>
        <button class="scope-command-btn" type="button" onclick="window.runPipelineProjectList()">Refresh</button>
        <button class="scope-command-btn" type="button" onclick="window.clearPipelineRegistryFilters()">Clear Filters</button>
      </div>
      ${rows || '<p class="pipeline-summary">No saved projects yet.</p>'}
      <div class="pipeline-section">
        <strong>Rebuild Jobs</strong>
        ${jobRows || '<p class="pipeline-summary">No rebuild jobs yet.</p>'}
      </div>
      ${progressRows.length ? `<ul class="pipeline-list">${pipelineListHTML(progressRows)}</ul>` : ''}
      <ul class="pipeline-list">${pipelineListHTML(importerLines.slice(0, 8))}</ul>
      ${report ? `<pre class="pipeline-report">${escapeHTML(report)}</pre>` : ''}
    </div>
  `;
}

function renderPipelinePlan(plan = S.pipelineBuilder.plan) {
  const preview = $('pipeline-plan-preview');
  const meta = $('pipeline-plan-meta');
  if (!preview) return;
  if (!plan) {
    preview.innerHTML = `
      <div class="pipeline-empty">Use the prompt line above. KYMA will open setup, tasks, training, and code only when each step is needed.</div>
      ${renderPipelineAutopilotHTML(S.pipelineBuilder.autopilot)}
      ${renderPipelineCodeWorkbenchHTML()}
      ${renderPipelineProjectRegistryHTML(S.pipelineBuilder.projectRegistry)}
      ${renderPipelineModelManagerHTML(S.pipelineBuilder.modelManager)}
      ${renderPipelineExportManagerHTML(S.pipelineBuilder.exportManager)}
    `;
    if (meta) meta.textContent = 'No plan generated';
    return;
  }
  const signal = plan.signal || {};
  const task = pipelinePromptRequestsFrequencyApp()
    ? { ...(plan.task || {}), type: 'spectral visualization' }
    : (plan.task || {});
  const protocol = plan.protocol || {};
  const windowing = plan.windowing || {};
  const labelProtocol = plan.label_protocol || {};
  const channels = Array.isArray(signal.channels) ? signal.channels : [];
  const filters = (Array.isArray(plan.filters) ? plan.filters : []).map(formatPipelineFilter);
  const candidates = Array.isArray(plan.model_candidates) && plan.model_candidates.length ? plan.model_candidates : plan.models;
  const models = (Array.isArray(candidates) ? candidates : []).map(model => {
    const name = String(model?.name || 'Model');
    const kind = String(model?.kind || model?.type || '');
    const why = String(model?.why || '');
    const notes = String(model?.notes || '');
    const latency = Number(model?.latency_target_ms);
    const status = String(model?.status || '');
    return `${name}${kind ? ` (${kind})` : ''}${status ? `, ${status.replace(/_/g, ' ')}` : ''}${Number.isFinite(latency) ? `, target ${latency} ms` : ''}${why || notes ? `: ${why || notes}` : ''}`;
  });
  const labelTargets = (Array.isArray(labelProtocol.labels) ? labelProtocol.labels : []).map(item => {
    const target = Number(item?.target_windows || 0);
    const seconds = Number(item?.min_seconds || 0);
    return `${item?.name || 'label'}${target ? `, ${target} windows` : ''}${seconds ? `, ${seconds}s` : ''}`;
  });
  const windowLines = [
    `${windowing.window_size_ms || '--'} ms window`,
    `${windowing.window_step_ms || '--'} ms step`,
    `${windowing.sample_rate_hz || S.sampleRate || 250} Hz`,
    windowing.normalization || 'per-session normalization',
  ];
  const channelPreview = channels.slice(0, 8).join(', ');
  const estimated = Number(protocol.estimated_minutes || 0);
  if (meta) {
    meta.textContent = estimated ? `${estimated} min first pass` : 'Generated plan';
  }
  const displayTitle = pipelinePromptRequestsFrequencyApp()
    ? `${S.signalProfileName || 'EMG'} Frequency Band Visualizer`
    : (plan.title || 'Biosignal Build');
  const displaySummary = pipelinePromptRequestsFrequencyApp()
    ? 'Build a web app that consumes live EMG samples and visualizes motion, low, mid, high, and line-noise frequency bands.'
    : (plan.summary || '');
  preview.innerHTML = `
    <div class="pipeline-title">
      <h4>${escapeHTML(displayTitle)}</h4>
    </div>
    <p class="pipeline-summary">${escapeHTML(displaySummary)}</p>
    ${renderPipelineAutopilotHTML(S.pipelineBuilder.autopilot)}
    <div class="pipeline-pill-row">
      <span class="pipeline-pill">${escapeHTML(signal.display_name || signal.profile || 'Signal')}</span>
      <span class="pipeline-pill">${escapeHTML(String(plan.source || 'live').replace(/_/g, ' '))}</span>
      <span class="pipeline-pill">${escapeHTML(String(plan.output || 'model').replace(/_/g, ' '))}</span>
      <span class="pipeline-pill">${escapeHTML(String(task.type || 'classification').replace(/_/g, ' '))}</span>
    </div>
    <details class="pipeline-plan-details">
      <summary>Generated Plan</summary>
      ${renderPipelineRecipeEditorHTML(plan)}
      <div class="pipeline-section">
        <strong>Channels</strong>
        <p class="pipeline-summary">${escapeHTML(channelPreview || 'No channel map yet.')}</p>
      </div>
      <div class="pipeline-columns">
        <div class="pipeline-section">
          <strong>Protocol</strong>
          <ul class="pipeline-list">${pipelineListHTML(protocol.steps || [])}</ul>
        </div>
        <div class="pipeline-section">
          <strong>Labels</strong>
          <ul class="pipeline-list">${pipelineListHTML(labelTargets.length ? labelTargets : task.labels || [])}</ul>
        </div>
        <div class="pipeline-section">
          <strong>Windowing</strong>
          <ul class="pipeline-list">${pipelineListHTML(windowLines)}</ul>
        </div>
        <div class="pipeline-section">
          <strong>Filters</strong>
          <ul class="pipeline-list">${pipelineListHTML(filters)}</ul>
        </div>
        <div class="pipeline-section">
          <strong>Models</strong>
          <ul class="pipeline-list">${pipelineListHTML(models)}</ul>
        </div>
        <div class="pipeline-section">
          <strong>Exports</strong>
          <ul class="pipeline-list">${pipelineListHTML(plan.exports || [])}</ul>
        </div>
        <div class="pipeline-section">
          <strong>Next</strong>
          <ul class="pipeline-list">${pipelineListHTML(plan.next_actions || [])}</ul>
        </div>
      </div>
    </details>
    ${renderPipelineCodeWorkbenchHTML()}
    <details class="pipeline-plan-details">
      <summary>Workspace Details</summary>
      ${renderPipelineQAHTML(S.pipelineBuilder.qa)}
      ${renderPipelineTrainHTML(S.pipelineBuilder.trainResult)}
      ${renderPipelineModelManagerHTML(S.pipelineBuilder.modelManager)}
      ${renderPipelineAcquisitionHTML(S.pipelineBuilder.acquisition)}
      ${renderPipelineLabelSuggestionsHTML(S.pipelineBuilder.labelSuggestions)}
      ${renderPipelineDatasetHTML(S.pipelineBuilder.dataset)}
      ${renderPipelineComparisonHTML(S.pipelineBuilder.comparison)}
      ${renderPipelineRuntimeHTML(S.pipelineBuilder.runtime)}
      ${renderPipelineEmbeddingHTML(S.pipelineBuilder.embedding)}
      ${renderPipelineThroughputHTML(S.pipelineBuilder.throughput)}
      ${renderPipelineExportHTML(S.pipelineBuilder.exportResult)}
      ${renderPipelineExportManagerHTML(S.pipelineBuilder.exportManager)}
      ${renderPipelineProjectRegistryHTML(S.pipelineBuilder.projectRegistry)}
    </details>
  `;
}

function syncPipelineBuilderUI() {
  const drawer = $('pipeline-builder-drawer');
  const toggle = $('btn-pipeline-builder');
  const prompt = $('pipeline-prompt');
  const source = $('pipeline-source');
  const output = $('pipeline-output');
  const datasetPath = $('pipeline-dataset-path');
  const status = $('pipeline-builder-status');
  const meta = $('pipeline-builder-meta');
  const autopilotBtn = $('btn-pipeline-autopilot');
  const createBtn = $('btn-pipeline-create-model');
  const runBtn = $('btn-pipeline-plan');
  const qaBtn = $('btn-pipeline-qa');
  const trainBtn = $('btn-pipeline-train');
  const acquireBtn = $('btn-pipeline-acquire');
  const runAcquireBtn = $('btn-pipeline-run-acquire');
  const trainAcquireBtn = $('btn-pipeline-train-acquire');
  const modelsBtn = $('btn-pipeline-models');
  const labelsBtn = $('btn-pipeline-labels');
  const datasetBtn = $('btn-pipeline-dataset');
  const ingestBtn = $('btn-pipeline-ingest');
  const compareBtn = $('btn-pipeline-compare');
  const runtimeBtn = $('btn-pipeline-runtime');
  const deployBtn = $('btn-pipeline-deploy');
  const embeddingBtn = $('btn-pipeline-embedding');
  const throughputBtn = $('btn-pipeline-throughput');
  const exportBtn = $('btn-pipeline-export');
  const exportsBtn = $('btn-pipeline-exports');
  const projectsBtn = $('btn-pipeline-projects');
  const saveProjectBtn = $('btn-pipeline-save-project');
  const busy = !!(S.pipelineBuilder.autopilotLoading || S.pipelineBuilder.createLoading || S.pipelineBuilder.loading || S.pipelineBuilder.qaLoading || S.pipelineBuilder.trainLoading
    || S.pipelineBuilder.acquireLoading || S.pipelineBuilder.acquisitionRunLoading
    || S.pipelineBuilder.acquisitionTrainLoading
    || S.pipelineBuilder.labelsLoading || S.pipelineBuilder.datasetLoading || S.pipelineBuilder.ingestLoading
    || S.pipelineBuilder.compareLoading || S.pipelineBuilder.runtimeLoading || S.pipelineBuilder.deployLoading || S.pipelineBuilder.embeddingLoading
    || S.pipelineBuilder.throughputLoading || S.pipelineBuilder.modelsLoading || S.pipelineBuilder.modelActionLoading
    || S.pipelineBuilder.exportsLoading || S.pipelineBuilder.exportActionLoading || S.pipelineBuilder.exportLoading
    || S.pipelineBuilder.projectsLoading || S.pipelineBuilder.projectActionLoading);
  if (drawer) drawer.classList.toggle('is-hidden', !S.pipelineBuilder.open);
  if (toggle) toggle.textContent = S.pipelineBuilder.autopilotLoading ? 'Building' : (S.pipelineBuilder.open ? 'Rebuild' : 'Build');
  if (prompt && prompt.value !== S.pipelineBuilder.prompt) prompt.value = S.pipelineBuilder.prompt;
  if (source && source.value !== S.pipelineBuilder.source) source.value = S.pipelineBuilder.source;
  if (output && output.value !== S.pipelineBuilder.output) output.value = S.pipelineBuilder.output;
  if (datasetPath && datasetPath.value !== S.pipelineBuilder.datasetPath) datasetPath.value = S.pipelineBuilder.datasetPath;
  if (meta) {
    const channelCount = Math.max(0, (S.channelLabels || []).length || N_CH);
    const sourceLabel = S.pipelineBuilder.source === 'live'
      ? 'live electrodes'
      : (S.pipelineBuilder.source === 'synthetic' ? 'synthetic demo' : (S.pipelineBuilder.source || S.streamSource || 'source'));
    meta.textContent = `${S.signalProfileName || 'Signal'} | ${channelCount} ch | ${sourceLabel}`;
  }
  if (status) {
    let label = 'Ready';
    if (S.pipelineBuilder.autopilotLoading) label = S.pipelineBuilder.createStage || 'Autopilot running...';
    else if (S.pipelineBuilder.createLoading) label = S.pipelineBuilder.createStage || 'Creating model...';
    else if (S.pipelineBuilder.loading) label = 'Planning...';
    else if (S.pipelineBuilder.qaLoading) label = 'Running QA...';
    else if (S.pipelineBuilder.trainLoading) label = 'Training baseline...';
    else if (S.pipelineBuilder.acquireLoading) label = 'Building acquisition...';
    else if (S.pipelineBuilder.acquisitionRunLoading) label = 'Updating acquisition...';
    else if (S.pipelineBuilder.acquisitionTrainLoading) label = 'Training prompt model...';
    else if (S.pipelineBuilder.modelsLoading) label = 'Loading models...';
    else if (S.pipelineBuilder.modelActionLoading) label = 'Switching model...';
    else if (S.pipelineBuilder.exportsLoading) label = 'Loading packages...';
    else if (S.pipelineBuilder.exportActionLoading) label = 'Updating package...';
    else if (S.pipelineBuilder.projectsLoading) label = 'Loading projects...';
    else if (S.pipelineBuilder.projectActionLoading) label = 'Updating project...';
    else if (S.pipelineBuilder.labelsLoading) label = 'Suggesting labels...';
    else if (S.pipelineBuilder.datasetLoading) label = 'Scanning dataset...';
    else if (S.pipelineBuilder.ingestLoading) label = 'Ingesting dataset...';
    else if (S.pipelineBuilder.compareLoading) label = 'Comparing models...';
    else if (S.pipelineBuilder.runtimeLoading) label = 'Checking runtime...';
    else if (S.pipelineBuilder.deployLoading) label = 'Verifying deploy...';
    else if (S.pipelineBuilder.embeddingLoading) label = 'Reading embeddings...';
    else if (S.pipelineBuilder.throughputLoading) label = 'Benchmarking...';
    else if (S.pipelineBuilder.exportLoading) label = 'Exporting...';
    else if (S.pipelineBuilder.error) label = S.pipelineBuilder.error;
    else if (S.pipelineBuilder.autopilot?.status === 'ready') label = 'Autopilot ready';
    else if (S.pipelineBuilder.autopilot?.status === 'waiting_setup') label = 'Connect electrodes';
    else if (S.pipelineBuilder.autopilot?.status === 'collecting') label = 'Collecting task data';
    else if (S.pipelineBuilder.autopilot?.status === 'needs_data') label = 'Autopilot needs data';
    else if (S.pipelineBuilder.autopilot?.status === 'failed') label = 'Autopilot failed';
    else if (S.pipelineBuilder.createStage === 'Dataset model ready') label = 'Dataset model ready';
    else if (S.pipelineBuilder.createStage === 'Model live') label = 'Model live';
    else if (S.pipelineBuilder.exportResult) label = 'Export ready';
    else if (S.pipelineBuilder.throughput) label = 'Speed ready';
    else if (S.pipelineBuilder.embedding) label = 'Readiness ready';
    else if (S.pipelineBuilder.runtime) label = 'Runtime checked';
    else if (S.pipelineBuilder.deploy) label = 'Deploy verified';
    else if (S.pipelineBuilder.comparison) label = 'Comparison ready';
    else if (S.pipelineBuilder.ingest) label = 'Dataset ingested';
    else if (S.pipelineBuilder.dataset) label = 'Import scan ready';
    else if (S.pipelineBuilder.labelSuggestions) label = 'Labels ready';
    else if (S.pipelineBuilder.acquisition) label = 'Acquisition ready';
    else if (S.pipelineBuilder.trainResult) label = S.pipelineBuilder.trainResult.trained ? 'Prompt model live' : 'Training needs data';
    else if (S.pipelineBuilder.qa) label = 'QA ready';
    else if (S.pipelineBuilder.plan) label = 'Plan ready';
    status.textContent = label;
  }
  if (autopilotBtn) {
    autopilotBtn.disabled = busy;
    autopilotBtn.textContent = S.pipelineBuilder.autopilotLoading ? 'Building...' : 'Build Automatically';
  }
  if (createBtn) {
    createBtn.disabled = busy;
    createBtn.textContent = S.pipelineBuilder.createLoading ? 'Creating...' : 'Create Model';
  }
  if (runBtn) {
    runBtn.disabled = busy;
    runBtn.textContent = S.pipelineBuilder.loading ? 'Planning' : 'Generate Plan';
  }
  if (qaBtn) {
    qaBtn.disabled = busy || !S.pipelineBuilder.plan;
    qaBtn.textContent = S.pipelineBuilder.qaLoading ? 'QA...' : 'Run QA';
  }
  if (trainBtn) {
    trainBtn.disabled = busy || !S.pipelineBuilder.plan;
    trainBtn.textContent = S.pipelineBuilder.trainLoading ? 'Training...' : 'Train Baseline';
  }
  if (acquireBtn) {
    acquireBtn.disabled = busy || !S.pipelineBuilder.plan;
    acquireBtn.textContent = S.pipelineBuilder.acquireLoading ? 'Acquire...' : 'Acquire';
  }
  if (runAcquireBtn) {
    const running = !!S.pipelineBuilder.acquisitionRun?.running;
    runAcquireBtn.disabled = busy || !S.pipelineBuilder.plan;
    runAcquireBtn.textContent = S.pipelineBuilder.acquisitionRunLoading ? 'Run...' : (running ? 'Stop Acquire' : 'Run Acquire');
  }
  if (trainAcquireBtn) {
    trainAcquireBtn.disabled = busy || !S.pipelineBuilder.plan;
    trainAcquireBtn.textContent = S.pipelineBuilder.acquisitionTrainLoading ? 'Training...' : 'Train Auto';
  }
  if (modelsBtn) {
    modelsBtn.disabled = busy;
    modelsBtn.textContent = S.pipelineBuilder.modelsLoading ? 'Models...' : 'Models';
  }
  if (labelsBtn) {
    labelsBtn.disabled = busy || !S.pipelineBuilder.plan;
    labelsBtn.textContent = S.pipelineBuilder.labelsLoading ? 'Labels...' : 'Labels';
  }
  if (datasetBtn) {
    datasetBtn.disabled = busy || !S.pipelineBuilder.plan;
    datasetBtn.textContent = S.pipelineBuilder.datasetLoading ? 'Import...' : 'Import';
  }
  if (ingestBtn) {
    ingestBtn.disabled = busy || !S.pipelineBuilder.plan;
    ingestBtn.textContent = S.pipelineBuilder.ingestLoading ? 'Ingest...' : 'Ingest';
  }
  if (compareBtn) {
    compareBtn.disabled = busy || !S.pipelineBuilder.plan;
    compareBtn.textContent = S.pipelineBuilder.compareLoading ? 'Compare...' : 'Compare';
  }
  if (runtimeBtn) {
    runtimeBtn.disabled = busy || !S.pipelineBuilder.plan;
    runtimeBtn.textContent = S.pipelineBuilder.runtimeLoading ? 'Runtime...' : 'Runtime';
  }
  if (deployBtn) {
    deployBtn.disabled = busy || !S.pipelineBuilder.plan;
    deployBtn.textContent = S.pipelineBuilder.deployLoading ? 'Verify...' : 'Verify';
  }
  if (embeddingBtn) {
    embeddingBtn.disabled = busy || !S.pipelineBuilder.plan;
    embeddingBtn.textContent = S.pipelineBuilder.embeddingLoading ? 'Embed...' : 'Embeddings';
  }
  if (throughputBtn) {
    throughputBtn.disabled = busy || !S.pipelineBuilder.plan;
    throughputBtn.textContent = S.pipelineBuilder.throughputLoading ? 'Speed...' : 'Speed';
  }
  if (exportBtn) {
    exportBtn.disabled = busy || !S.pipelineBuilder.plan;
    exportBtn.textContent = S.pipelineBuilder.exportLoading ? 'Export...' : 'Export';
  }
  if (exportsBtn) {
    exportsBtn.disabled = busy;
    exportsBtn.textContent = S.pipelineBuilder.exportsLoading ? 'Packages...' : 'Packages';
  }
  if (projectsBtn) {
    projectsBtn.disabled = busy;
    projectsBtn.textContent = S.pipelineBuilder.projectsLoading ? 'Projects...' : 'Projects';
  }
  if (saveProjectBtn) {
    saveProjectBtn.disabled = busy || !S.pipelineBuilder.plan;
    saveProjectBtn.textContent = S.pipelineBuilder.projectActionLoading ? 'Saving...' : 'Save';
  }
  renderPipelinePlan();
}

function togglePipelineBuilder(force) {
  const next = typeof force === 'boolean' ? force : !S.pipelineBuilder.open;
  S.pipelineBuilder.open = next;
  localStorage.setItem('kyma-pipeline-builder-open', next ? '1' : '0');
  if (next) {
    const inlinePrompt = String($('scope-copilot-input')?.value || '').trim();
    if (inlinePrompt) S.pipelineBuilder.prompt = inlinePrompt;
  }
  syncPipelineBuilderUI();
  if (next && !S.pipelineBuilder.projectRegistry) runPipelineProjectList();
}

async function runTopPromptBuild() {
  const input = $('scope-copilot-input');
  const inlinePrompt = String(input?.value || '').trim();
  const profileName = S.signalProfileName || 'biosignal';
  const article = /^[aeiou]/i.test(profileName) ? 'an' : 'a';
  S.pipelineBuilder.prompt = inlinePrompt || `Build ${article} ${profileName} model from this live stream in 5 minutes.`;
  S.pipelineBuilder.open = true;
  S.pipelineBuilder.sourceChoiceConfirmed = false;
  S.pipelineBuilder.appChoiceConfirmed = false;
  S.pipelineBuilder.codeWorkbenchOpen = false;
  S.pipelineBuilder.error = '';
  S.pipelineBuilder.autopilot = {
    status: 'choose_setup',
    mode: '',
    summary: 'Choose the source and app type before KYMA records or trains anything.',
    next_action: 'Pick live electrodes, synthetic demo, or dataset import. Then choose the app KYMA should generate.',
    action: 'choose_setup',
    steps: [
      { id: 'setup', label: 'Choose source and app', status: 'running' },
      { id: 'plan', label: 'Create pipeline recipe', status: 'pending' },
      { id: 'quality', label: 'Check signal or dataset quality', status: 'pending' },
      { id: 'validate', label: 'Train and grade model', status: 'pending' },
      { id: 'code', label: 'Generate app code', status: 'pending' },
    ],
  };
  localStorage.setItem('kyma-pipeline-builder-open', '1');
  syncPipelineBuilderUI();
}

function choosePipelineBuildSource(source) {
  const value = ['live', 'synthetic', 'dataset'].includes(String(source || '')) ? String(source) : 'live';
  S.pipelineBuilder.source = value;
  S.pipelineBuilder.sourceChoiceConfirmed = true;
  if (value !== 'dataset') S.pipelineBuilder.datasetPath = '';
  localStorage.setItem('kyma-pipeline-source', value);
  const sourceSelect = $('pipeline-source');
  if (sourceSelect) sourceSelect.value = value;
  syncPipelineBuilderUI();
}

function choosePipelineAppType(type) {
  const value = ['dashboard', 'alert', 'control', 'research'].includes(String(type || '')) ? String(type) : 'dashboard';
  S.pipelineBuilder.appType = value;
  S.pipelineBuilder.appChoiceConfirmed = true;
  const outputMap = {
    dashboard: 'live_model',
    alert: 'software_control',
    control: 'software_control',
    research: 'analysis_report',
  };
  S.pipelineBuilder.output = outputMap[value] || 'live_model';
  localStorage.setItem('kyma-pipeline-output', S.pipelineBuilder.output);
  const outputSelect = $('pipeline-output');
  if (outputSelect) outputSelect.value = S.pipelineBuilder.output;
  syncPipelineBuilderUI();
}

async function beginPipelineBuildWithChoices() {
  const datasetChoice = $('pipeline-choice-dataset-path');
  if (datasetChoice) {
    S.pipelineBuilder.datasetPath = String(datasetChoice.value || '').trim();
    localStorage.setItem('kyma-pipeline-dataset-path', S.pipelineBuilder.datasetPath);
  }
  if (!['live', 'synthetic', 'dataset'].includes(S.pipelineBuilder.source || '')) {
    toast('Choose live, synthetic, or dataset first.', 'yellow');
    return;
  }
  if (!S.pipelineBuilder.appType) {
    toast('Choose the app KYMA should build first.', 'yellow');
    return;
  }
  if (S.pipelineBuilder.source === 'dataset' && !S.pipelineBuilder.datasetPath) {
    toast('Enter the dataset path first.', 'yellow');
    syncPipelineBuilderUI();
    return;
  }
  S.pipelineBuilder.sourceChoiceConfirmed = true;
  S.pipelineBuilder.appChoiceConfirmed = true;
  await runPipelineAutopilot();
}

function buildPipelinePlanPayload() {
  const aiPayload = buildAICopilotPayload();
  return {
    prompt: S.pipelineBuilder.prompt,
    source: S.pipelineBuilder.source,
    output: S.pipelineBuilder.output,
    signal_profile: S.signalProfileKey || 'emg',
    channel_labels: (S.channelLabels || []).slice(0, N_CH),
    context: {
      dataset_path: S.pipelineBuilder.datasetPath || '',
      dataset_mode: S.pipelineBuilder.source === 'dataset',
      stream_source: S.streamSource,
      sample_rate_hz: S.sampleRate,
      decoder_mode: S.decoderMode,
      live_ml: S.liveMl ? {
        inference_ms: S.liveMl.inference_ms,
        artifact_label: S.liveMl.artifact_label,
        foundation_embedding_count: S.liveMl.foundation_embedding_count,
      } : {},
      diagnostics: aiPayload.diagnostics || {},
      review_stats: aiPayload.review?.stats || null,
      last_prediction: aiPayload.last_prediction || {},
      session: aiPayload.session || {},
    },
  };
}

function syncPipelineBuilderInputsFromForm() {
  const topPrompt = String($('scope-copilot-input')?.value || '').trim();
  S.pipelineBuilder.prompt = String(topPrompt || $('pipeline-prompt')?.value || S.pipelineBuilder.prompt || '').trim();
  S.pipelineBuilder.source = String($('pipeline-source')?.value || S.pipelineBuilder.source || 'live');
  S.pipelineBuilder.output = String($('pipeline-output')?.value || S.pipelineBuilder.output || 'live_model');
  S.pipelineBuilder.datasetPath = String($('pipeline-dataset-path')?.value || S.pipelineBuilder.datasetPath || '').trim();
  localStorage.setItem('kyma-pipeline-source', S.pipelineBuilder.source);
  localStorage.setItem('kyma-pipeline-output', S.pipelineBuilder.output);
  localStorage.setItem('kyma-pipeline-dataset-path', S.pipelineBuilder.datasetPath);
}

function pipelineBuilderUsesDatasetSource() {
  return S.pipelineBuilder.source === 'dataset' || !!String(S.pipelineBuilder.datasetPath || '').trim();
}

function pipelineBuilderUsesSyntheticSource() {
  return S.pipelineBuilder.source === 'synthetic';
}

function pipelinePromptRequestsSynthetic() {
  const prompt = String(S.pipelineBuilder.prompt || '').toLowerCase();
  return /\b(synthetic|demo|simulate|simulated|mock|fake|test data)\b/.test(prompt);
}

function pipelinePromptRequestsLive() {
  const prompt = String(S.pipelineBuilder.prompt || '').toLowerCase();
  return /\b(live|electrode|electrodes|sensor|hardware|real[- ]?time|record|recording|my arm|my muscle)\b/.test(prompt);
}

function normalizePipelineSourceForPrompt() {
  const source = $('pipeline-source');
  if (source && S.pipelineBuilder.source && source.value !== S.pipelineBuilder.source) source.value = S.pipelineBuilder.source;
}

function improvePipelinePrompt() {
  syncPipelineBuilderInputsFromForm();
  const raw = String(S.pipelineBuilder.prompt || '').trim();
  const profile = String(S.signalProfileName || 'biosignal');
  const source = S.pipelineBuilder.source === 'synthetic' && !pipelinePromptRequestsSynthetic()
    ? 'live electrodes'
    : String(S.pipelineBuilder.source || 'live').replace(/_/g, ' ');
  const output = String(S.pipelineBuilder.output || 'live_model').replace(/_/g, ' ');
  const goal = raw || `Build a ${profile} model`;
  const wantsFatigue = /\bfatigue|tired|endurance\b/i.test(goal);
  const wantsControl = /\bcontrol|app|game|keyboard|mouse|robot|software\b/i.test(goal);
  const labels = wantsFatigue
    ? ['rest baseline', 'fresh contraction', 'fatigued hold', 'recovery']
    : (wantsControl ? ['rest', 'intent on', 'intent off'] : ['rest', 'active', 'artifact']);
  const improved = [
    `Build: ${goal}.`,
    `Input: ${source}; signal profile: ${profile}; output: ${output}.`,
    `Guide the user one step at a time: electrode setup, live channel check, baseline, then one recording task per label.`,
    `Labels/tasks: ${labels.join(', ')}.`,
    `Reject noisy, clipped, duplicated, or flat-channel data and ask the user to repeat only the failed task.`,
    `After training, generate the code needed to test the model: status check, QA command, WebSocket prediction listener, and a small app integration example.`,
  ].join('\n');
  S.pipelineBuilder.prompt = improved;
  const prompt = $('pipeline-prompt');
  const topPrompt = $('scope-copilot-input');
  if (prompt) {
    prompt.value = improved;
  }
  if (topPrompt) {
    topPrompt.value = improved;
    topPrompt.focus();
  }
  normalizePipelineSourceForPrompt();
  syncPipelineBuilderUI();
  toast('Prompt improved for guided build');
}

function buildPipelineActionPayload() {
  return {
    plan: applyPipelineRecipeDraftToPlan() || S.pipelineBuilder.plan || {},
    qa: S.pipelineBuilder.qa || {},
    train_result: S.pipelineBuilder.trainResult || {},
    acquisition: S.pipelineBuilder.acquisition || {},
    acquisition_run: S.pipelineBuilder.acquisitionRun || {},
    labels: S.pipelineBuilder.labelSuggestions || {},
    dataset: S.pipelineBuilder.dataset || {},
    throughput: S.pipelineBuilder.throughput || {},
    context: {
      ...((buildPipelinePlanPayload() || {}).context || {}),
      webgpu_available: !!navigator.gpu,
      user_agent: navigator.userAgent || '',
    },
  };
}

async function refreshPipelineAcquisitionRun() {
  if (!S.pipelineBuilder.acquisitionRun?.id || S.pipelineBuilder.acquisitionRunLoading) return;
  try {
    const res = await post('/api/pipeline/acquisition_run', {
      ...buildPipelineActionPayload(),
      protocol: S.pipelineBuilder.acquisition || {},
      action: 'status',
    });
    S.pipelineBuilder.acquisitionRun = res.run || null;
    syncPipelineBuilderUI();
  } catch {
    // Status refresh should not interrupt the live stream UI.
  }
}

async function runPipelinePlan() {
  if (S.pipelineBuilder.loading) return;
  syncPipelineBuilderInputsFromForm();
  if (!S.pipelineBuilder.prompt) {
    S.pipelineBuilder.error = 'Prompt required';
    syncPipelineBuilderUI();
    toast('Describe what the pipeline should build first.', 'yellow');
    return;
  }
  S.pipelineBuilder.loading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/plan', buildPipelinePlanPayload());
    S.pipelineBuilder.plan = res.plan || null;
    S.pipelineBuilder.recipeDraft = null;
    S.pipelineBuilder.qa = null;
    S.pipelineBuilder.trainResult = null;
    S.pipelineBuilder.acquisition = null;
    S.pipelineBuilder.acquisitionRun = null;
    S.pipelineBuilder.labelSuggestions = null;
    S.pipelineBuilder.dataset = null;
    S.pipelineBuilder.ingest = null;
    S.pipelineBuilder.comparison = null;
    S.pipelineBuilder.exportResult = null;
    S.pipelineBuilder.runtime = null;
    S.pipelineBuilder.deploy = null;
    S.pipelineBuilder.embedding = null;
    S.pipelineBuilder.throughput = null;
    S.pipelineBuilder.error = '';
    toast('Pipeline plan generated');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Plan failed';
    toast(`Pipeline plan failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.loading = false;
    syncPipelineBuilderUI();
  }
}

async function runPipelineAcquisition() {
  if (S.pipelineBuilder.acquireLoading || !S.pipelineBuilder.plan) return;
  S.pipelineBuilder.acquireLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/acquisition_protocol', buildPipelineActionPayload());
    S.pipelineBuilder.acquisition = res.acquisition || null;
    toast('Guided acquisition ready');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Acquisition failed';
    toast(`Acquisition failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.acquireLoading = false;
    syncPipelineBuilderUI();
  }
}

async function runPipelineAcquisitionControl() {
  if (S.pipelineBuilder.acquisitionRunLoading || !S.pipelineBuilder.plan) return;
  if (!S.pipelineBuilder.acquisition) {
    await runPipelineAcquisition();
    if (!S.pipelineBuilder.acquisition) return;
  }
  const running = !!S.pipelineBuilder.acquisitionRun?.running;
  S.pipelineBuilder.acquisitionRunLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/acquisition_run', {
      ...buildPipelineActionPayload(),
      protocol: S.pipelineBuilder.acquisition || {},
      action: running ? 'stop' : 'start',
    });
    S.pipelineBuilder.acquisitionRun = res.run || null;
    toast(running ? 'Guided acquisition stopped' : 'Guided acquisition running');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Acquisition run failed';
    toast(`Acquisition run failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.acquisitionRunLoading = false;
    syncPipelineBuilderUI();
  }
}

async function runPipelineAcquisitionTrain() {
  if (S.pipelineBuilder.acquisitionTrainLoading || !S.pipelineBuilder.plan) return;
  S.pipelineBuilder.acquisitionTrainLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/train_acquisition', buildPipelineActionPayload());
    S.pipelineBuilder.trainResult = res || null;
    if (res?.trained) {
      const manager = await get('/api/pipeline/models').catch(() => null);
      if (manager) S.pipelineBuilder.modelManager = manager;
      if (manager?.active) S.promptModel.server = manager.active;
    }
    toast(res.trained ? 'Prompt model trained' : 'Prompt model needs more labels', res.trained ? '' : 'yellow');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Prompt training failed';
    toast(`Prompt training failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.acquisitionTrainLoading = false;
    syncPipelineBuilderUI();
  }
}

async function runPipelineModelList() {
  if (S.pipelineBuilder.modelsLoading) return;
  S.pipelineBuilder.modelsLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await get('/api/pipeline/models');
    S.pipelineBuilder.modelManager = res || null;
    if (res?.active) S.promptModel.server = res.active;
    toast(`${Array.isArray(res?.models) ? res.models.length : 0} saved model${Array.isArray(res?.models) && res.models.length === 1 ? '' : 's'} found`);
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Model list failed';
    toast(`Model list failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.modelsLoading = false;
    syncPipelineBuilderUI();
  }
}

async function loadPipelinePromptModel(path) {
  if (!path || S.pipelineBuilder.modelActionLoading) return;
  S.pipelineBuilder.modelActionLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/models/load', { path });
    S.pipelineBuilder.modelManager = res || S.pipelineBuilder.modelManager;
    if (res?.active) S.promptModel.server = res.active;
    toast('Prompt model loaded');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Model load failed';
    toast(`Model load failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.modelActionLoading = false;
    syncPipelineBuilderUI();
  }
}

async function deletePipelinePromptModel(path) {
  if (!path || S.pipelineBuilder.modelActionLoading) return;
  const fileName = path.split(/[\\/]/).pop() || 'this model';
  if (!window.confirm(`Delete ${fileName}?`)) return;
  S.pipelineBuilder.modelActionLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/models/delete', { path });
    S.pipelineBuilder.modelManager = res || S.pipelineBuilder.modelManager;
    if (res?.active) S.promptModel.server = res.active;
    toast(res?.was_active ? 'Active model deleted' : 'Saved model deleted', res?.was_active ? 'yellow' : '');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Model delete failed';
    toast(`Model delete failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.modelActionLoading = false;
    syncPipelineBuilderUI();
  }
}

window.loadPipelinePromptModel = loadPipelinePromptModel;
window.deletePipelinePromptModel = deletePipelinePromptModel;

async function runPipelineExportList() {
  if (S.pipelineBuilder.exportsLoading) return;
  S.pipelineBuilder.exportsLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await get('/api/pipeline/exports');
    S.pipelineBuilder.exportManager = res || null;
    toast(`${Array.isArray(res?.exports) ? res.exports.length : 0} export package${Array.isArray(res?.exports) && res.exports.length === 1 ? '' : 's'} found`);
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Package list failed';
    toast(`Export package list failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.exportsLoading = false;
    syncPipelineBuilderUI();
  }
}

async function verifyPipelineExportPackage(path) {
  if (!path || S.pipelineBuilder.exportActionLoading) return;
  S.pipelineBuilder.exportActionLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/exports/verify', { path });
    S.pipelineBuilder.deploy = res.deploy || S.pipelineBuilder.deploy;
    if (!S.pipelineBuilder.exportManager) S.pipelineBuilder.exportManager = { exports: [] };
    S.pipelineBuilder.exportManager.lastVerify = res || null;
    if (res.package && S.pipelineBuilder.exportManager) {
      const list = Array.isArray(S.pipelineBuilder.exportManager.exports) ? S.pipelineBuilder.exportManager.exports : [];
      S.pipelineBuilder.exportManager.exports = list.map(item => item.path === res.package.path ? res.package : item);
    }
    toast(res.deploy?.ready ? 'Export package verified' : 'Export package needs review', res.deploy?.ready ? '' : 'yellow');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Package verify failed';
    toast(`Export package verify failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.exportActionLoading = false;
    syncPipelineBuilderUI();
  }
}

async function deletePipelineExportPackage(path) {
  if (!path || S.pipelineBuilder.exportActionLoading) return;
  const fileName = path.split(/[\\/]/).pop() || 'this export package';
  if (!window.confirm(`Delete ${fileName}?`)) return;
  S.pipelineBuilder.exportActionLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/exports/delete', { path });
    S.pipelineBuilder.exportManager = res || S.pipelineBuilder.exportManager;
    if (S.pipelineBuilder.exportResult?.export_dir === path) S.pipelineBuilder.exportResult = null;
    toast('Export package deleted');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Package delete failed';
    toast(`Export package delete failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.exportActionLoading = false;
    syncPipelineBuilderUI();
  }
}

function openPipelineExportDemo(path) {
  if (!path) return;
  window.open(pipelineExportDemoUrl(path), '_blank', 'noopener');
}

function downloadPipelineExportPackage(path) {
  if (!path) return;
  window.open(pipelineExportDownloadUrl(path), '_blank', 'noopener');
}

async function copyPipelineExportPath(path) {
  if (!path) return;
  try {
    await navigator.clipboard.writeText(path);
    toast('Export path copied');
  } catch {
    toast(path);
  }
}

async function copyPipelineRegistryPath(path) {
  if (!path) return;
  try {
    await navigator.clipboard.writeText(path);
    toast('Path copied');
  } catch {
    toast(path);
  }
}

window.verifyPipelineExportPackage = verifyPipelineExportPackage;
window.deletePipelineExportPackage = deletePipelineExportPackage;
window.openPipelineExportDemo = openPipelineExportDemo;
window.downloadPipelineExportPackage = downloadPipelineExportPackage;
window.copyPipelineExportPath = copyPipelineExportPath;
window.copyPipelineRegistryPath = copyPipelineRegistryPath;

async function runPipelineProjectList() {
  if (S.pipelineBuilder.projectsLoading) return;
  S.pipelineBuilder.projectsLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await get('/api/pipeline/projects');
    S.pipelineBuilder.projectRegistry = res || null;
    toast(`${Array.isArray(res?.projects) ? res.projects.length : 0} project${Array.isArray(res?.projects) && res.projects.length === 1 ? '' : 's'} found`);
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Project list failed';
    toast(`Project list failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.projectsLoading = false;
    syncPipelineBuilderUI();
  }
}

async function savePipelineProject() {
  if (S.pipelineBuilder.projectActionLoading || !S.pipelineBuilder.plan) return;
  S.pipelineBuilder.projectActionLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const snapshot = buildPipelineProjectSnapshot();
    const res = await post('/api/pipeline/projects/save', {
      project_id: S.pipelineBuilder.activeProject?.project_id || snapshot.project_id || '',
      name: snapshot.name || 'Biosignal project',
      snapshot,
    });
    S.pipelineBuilder.projectRegistry = res || S.pipelineBuilder.projectRegistry;
    S.pipelineBuilder.activeProject = res.project || null;
    toast('Project saved with recipe version and report');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Project save failed';
    toast(`Project save failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.projectActionLoading = false;
    syncPipelineBuilderUI();
  }
}

async function loadPipelineProject(projectId) {
  if (!projectId || S.pipelineBuilder.projectActionLoading) return;
  S.pipelineBuilder.projectActionLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/projects/load', { project_id: projectId });
    S.pipelineBuilder.activeProject = res.project || null;
    restorePipelineProjectSnapshot(res.project?.snapshot || {});
    toast('Project loaded');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Project load failed';
    toast(`Project load failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.projectActionLoading = false;
    syncPipelineBuilderUI();
  }
}

async function deletePipelineProject(projectId) {
  if (!projectId || S.pipelineBuilder.projectActionLoading) return;
  if (!window.confirm(`Delete project ${projectId}?`)) return;
  S.pipelineBuilder.projectActionLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/projects/delete', { project_id: projectId });
    S.pipelineBuilder.projectRegistry = res || S.pipelineBuilder.projectRegistry;
    if (S.pipelineBuilder.activeProject?.project_id === projectId) S.pipelineBuilder.activeProject = null;
    toast('Project deleted');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Project delete failed';
    toast(`Project delete failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.projectActionLoading = false;
    syncPipelineBuilderUI();
  }
}

async function loadPipelineRebuildJob(jobId) {
  if (!jobId || S.pipelineBuilder.projectActionLoading) return;
  S.pipelineBuilder.projectActionLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await get(`/api/pipeline/projects/rebuild/${encodeURIComponent(jobId)}`);
    syncPipelineRebuildJob(res.job || null);
    toast('Rebuild job details loaded');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Job load failed';
    toast(`Job load failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.projectActionLoading = false;
    syncPipelineBuilderUI();
  }
}

async function deletePipelineRebuildJob(jobId) {
  if (!jobId || S.pipelineBuilder.projectActionLoading) return;
  if (!window.confirm(`Delete rebuild job ${jobId}?`)) return;
  S.pipelineBuilder.projectActionLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/projects/rebuild/delete', { project_id: jobId });
    if (!S.pipelineBuilder.projectRegistry) S.pipelineBuilder.projectRegistry = {};
    S.pipelineBuilder.projectRegistry.rebuild_jobs = Array.isArray(res?.rebuild_jobs) ? res.rebuild_jobs : [];
    if (S.pipelineBuilder.rebuildJob?.job_id === jobId) {
      S.pipelineBuilder.rebuildJob = null;
      S.pipelineBuilder.rebuildProgress = [];
    }
    toast('Rebuild job deleted');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Job delete failed';
    toast(`Job delete failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.projectActionLoading = false;
    syncPipelineBuilderUI();
  }
}

function clearPipelineRegistryFilters() {
  S.pipelineBuilder.registryFilters = { query: '', status: 'all', format: 'all' };
  syncPipelineBuilderUI();
}

async function rebuildPipelineProject(projectId) {
  if (!projectId || S.pipelineBuilder.createLoading) return;
  S.pipelineBuilder.rebuildProgress = [];
  S.pipelineBuilder.rebuildJob = null;
  S.pipelineBuilder.createLoading = true;
  S.pipelineBuilder.projectActionLoading = true;
  S.pipelineBuilder.createStage = 'Starting backend rebuild...';
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const start = await post('/api/pipeline/projects/rebuild', { project_id: projectId });
    syncPipelineRebuildJob(start.job || null);
    const jobId = start.job?.job_id || '';
    if (!jobId) throw new Error('Backend rebuild did not return a job id.');
    let latest = start.job || null;
    for (let idx = 0; idx < 900; idx += 1) {
      await pipelineSleep(900);
      const res = await get(`/api/pipeline/projects/rebuild/${encodeURIComponent(jobId)}`);
      latest = res.job || latest;
      syncPipelineRebuildJob(latest);
      syncPipelineBuilderUI();
      if (['completed', 'failed'].includes(String(latest?.status || ''))) break;
    }
    if (String(latest?.status || '') !== 'completed') {
      throw new Error(latest?.error || 'Backend rebuild failed.');
    }
    await runPipelineProjectList();
    await loadPipelineProject(projectId);
    S.pipelineBuilder.createStage = 'Backend rebuild complete';
    toast('Project rebuild complete');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Project rebuild failed';
    if (!S.pipelineBuilder.rebuildProgress.length) setPipelineRebuildProgress(S.pipelineBuilder.error, 'error');
    toast(`Project rebuild failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.createLoading = false;
    S.pipelineBuilder.projectActionLoading = false;
    syncPipelineBuilderUI();
  }
}

window.runPipelineProjectList = runPipelineProjectList;
window.savePipelineProject = savePipelineProject;
window.loadPipelineProject = loadPipelineProject;
window.deletePipelineProject = deletePipelineProject;
window.loadPipelineRebuildJob = loadPipelineRebuildJob;
window.deletePipelineRebuildJob = deletePipelineRebuildJob;
window.clearPipelineRegistryFilters = clearPipelineRegistryFilters;
window.openPipelineProjectReport = openPipelineProjectReport;
window.openPipelineRebuildReport = openPipelineRebuildReport;
window.rebuildPipelineProject = rebuildPipelineProject;
window.runPipelineAutopilot = runPipelineAutopilot;
window.choosePipelineBuildSource = choosePipelineBuildSource;
window.choosePipelineAppType = choosePipelineAppType;
window.beginPipelineBuildWithChoices = beginPipelineBuildWithChoices;
window.startPipelineAutopilotStream = startPipelineAutopilotStream;
window.continuePipelineAutopilotLive = continuePipelineAutopilotLive;
window.startPipelineAutopilotTask = startPipelineAutopilotTask;

function pipelineSleep(ms) {
  return new Promise(resolve => setTimeout(resolve, Math.max(0, Number(ms || 0))));
}

function setPipelineAutopilotStep(id, status, detail = '') {
  const autopilot = S.pipelineBuilder.autopilot || { steps: [] };
  const steps = Array.isArray(autopilot.steps) ? autopilot.steps : [];
  const index = steps.findIndex(step => step.id === id);
  if (index >= 0) {
    steps[index] = { ...steps[index], status, detail };
  }
  S.pipelineBuilder.autopilot = { ...autopilot, steps };
  const active = steps.find(step => step.status === 'running');
  S.pipelineBuilder.createStage = active?.label || autopilot.summary || '';
  syncPipelineBuilderUI();
}

async function runPipelineAutopilotStep(id, fn, doneDetail = '') {
  setPipelineAutopilotStep(id, 'running');
  S.pipelineBuilder.error = '';
  await fn();
  if (S.pipelineBuilder.error) {
    setPipelineAutopilotStep(id, 'failed', S.pipelineBuilder.error);
    throw new Error(S.pipelineBuilder.error);
  }
  setPipelineAutopilotStep(id, 'done', doneDetail);
}

function resetPipelineAutopilot(mode) {
  const datasetMode = mode === 'dataset';
  const syntheticMode = mode === 'synthetic';
  S.pipelineBuilder.codeWorkbenchOpen = false;
  if (pipelineCodeTypingTimer) {
    clearInterval(pipelineCodeTypingTimer);
    pipelineCodeTypingTimer = null;
  }
  S.pipelineBuilder.autopilot = {
    status: 'running',
    mode: datasetMode ? 'dataset' : (syntheticMode ? 'synthetic demo' : 'live electrodes'),
    summary: datasetMode
      ? 'Autopilot will inspect the dataset, save a project, train when labels are present, and create a package.'
      : (syntheticMode
        ? 'Autopilot will simulate task collection, train, export, and save the package.'
        : 'Autopilot will open electrode setup, show task prompts, check quality, train when data is clean, and create a package.'),
    next_action: '',
    steps: [
      { id: 'setup', label: `${datasetMode ? 'Dataset' : (syntheticMode ? 'Synthetic' : 'Live')} source and ${pipelineAppTypeLabel(S.pipelineBuilder.appType)}`, status: 'done', detail: 'User selected' },
      { id: 'plan', label: 'Create pipeline recipe', status: 'pending' },
      datasetMode
        ? { id: 'inspect', label: 'Inspect dataset', status: 'pending' }
        : { id: 'quality', label: 'Check live signal quality', status: 'pending' },
      datasetMode
        ? { id: 'ingest', label: 'Ingest and train dataset', status: 'pending' }
        : { id: 'labels', label: 'Prepare labels or acquisition protocol', status: 'pending' },
      { id: 'validate', label: 'Validate model readiness', status: 'pending' },
      { id: 'package', label: 'Export package', status: 'pending' },
      { id: 'save', label: 'Save project and report', status: 'pending' },
      { id: 'code', label: 'Generate app code', status: 'pending' },
    ],
  };
  syncPipelineBuilderUI();
}

async function finishPipelineAutopilot(status, summary, nextAction) {
  S.pipelineBuilder.autopilot = {
    ...(S.pipelineBuilder.autopilot || {}),
    status,
    summary,
    next_action: nextAction,
    action: '',
  };
  S.pipelineBuilder.createStage = summary;
  if (status === 'ready') {
    setPipelineAutopilotStep('code', 'running', `Generating ${pipelineAppTypeLabel(S.pipelineBuilder.appType).toLowerCase()} files.`);
    startPipelineCodeWorkbench({ force: true });
    setPipelineAutopilotStep('code', 'done', 'Code workspace opened.');
    return;
  }
  syncPipelineBuilderUI();
}

async function startPipelineAutopilotStream() {
  const selected = pipelineBuilderUsesSyntheticSource() ? 'synthetic' : (S.pipelineBuilder.source === 'live' ? 'hardware' : getSelectedSource());
  const source = selected === 'playback' ? 'hardware' : selected;
  if (S.streaming) {
    if (String(S.streamSource || '') === source) {
      await loadStatus().catch(() => null);
      toast('Stream is already running');
      return true;
    }
    await post('/api/stream/stop').catch(() => null);
    S.streaming = false;
    await loadStatus().catch(() => null);
  }
  const body = { source };
  if (source === 'synthetic') body.synthetic_scenario = getSelectedSyntheticScenario();
  const cPort = $('cyton-port')?.value || '';
  if (source === 'hardware' && cPort) body.cyton_port = cPort;
  if (source === 'lsl') {
    const active = S.lslInputs.find(stream => (stream.source_id || stream.uid || stream.name) === getSelectedLSLInput());
    if (!active) throw new Error('Select an external LSL stream first.');
    body.lsl_stream_name = active.name;
    body.lsl_source_id = active.source_id || null;
  }
  const aPort = $('arduino-port')?.value || '';
  if (aPort) body.arduino_port = aPort;
  const res = await post('/api/stream/start', body);
  S.streamSource = res.stream_source || source;
  S.streamDetails = res.stream_details || {};
  resetSignalBuffers();
  resetReviewState({ clearMarkers: true });
  applySignalProfile(res.signal_profile || {});
  S.review.liveAutoScale = true;
  S.review.zoomY = 1;
  S.review.lastAutoScaleAt = 0;
  syncProfileUI();
  await loadStatus().catch(() => null);
  await refreshLSLStatus().catch(() => null);
  toast(`${S.signalProfileName || 'Biosignal'} stream started`);
  return true;
}

function pipelineAutopilotTaskLabels() {
  return (S.pipelineBuilder.acquisition?.steps || [])
    .filter(step => String(step?.action || '') === 'record_label')
    .map(step => String(step.label || step.name || step.id || '').trim())
    .filter(Boolean)
    .slice(0, 6);
}

function setPipelineGuidedTaskReady(index = 0) {
  const taskSteps = (S.pipelineBuilder.acquisition?.steps || [])
    .filter(step => String(step?.action || '') === 'record_label');
  const safeIndex = clamp(Number(index || 0), 0, Math.max(taskSteps.length - 1, 0));
  S.pipelineBuilder.guidedTaskIndex = safeIndex;
  S.pipelineBuilder.acquisitionRun = {
    ...(S.pipelineBuilder.acquisitionRun || {}),
    running: false,
    current_step: taskSteps[safeIndex] || {},
  };
  setPipelineAutopilotAction(
    'waiting_task',
    'task_ready',
    `Task ${safeIndex + 1} is ready.`,
    taskSteps[safeIndex]?.label
      ? `Prepare for ${taskSteps[safeIndex].label}, then press Start This Task.`
      : 'Prepare, then press Start This Task.'
  );
}

function setPipelineAutopilotAction(status, action, summary, nextAction) {
  S.pipelineBuilder.autopilot = {
    ...(S.pipelineBuilder.autopilot || {}),
    status,
    action,
    summary,
    next_action: nextAction,
  };
  syncPipelineBuilderUI();
}

async function completePipelineAutopilotFromTrainedLiveModel() {
  await runPipelineCompare().catch(() => null);
  await runPipelineEmbedding().catch(() => null);
  await runPipelineThroughput().catch(() => null);
  await runPipelineAutopilotStep('package', runPipelineExport, 'Export package created');
  await runPipelineRuntime().catch(() => null);
  await runPipelineDeploySmoke().catch(() => null);
  await runPipelineAutopilotStep('save', savePipelineProject, 'Project report saved');
  await runPipelineProjectList().catch(() => null);
  await finishPipelineAutopilot(
    'ready',
    'Autopilot finished. The live model, project report, and package are ready.',
    'Open the project report or download the package from the Project Registry.'
  );
  toast('Autopilot complete');
}

async function completePipelineAutopilotFromSignalAnalysis() {
  const qaScore = Number(S.pipelineBuilder.qa?.score || 0);
  S.pipelineBuilder.trainResult = {
    trained: true,
    ready: true,
    analysis_app: true,
    classifier: 'raw_emg_frequency_bands',
    result: {
      quality_score: Number.isFinite(qaScore) ? qaScore : null,
      ready: true,
      app_type: 'frequency_band_visualizer',
    },
    next_actions: [
      'Generate the EMG frequency band web app.',
      'Keep KYMA streaming so the app receives raw EMG chunks over WebSocket.',
    ],
  };
  setPipelineAutopilotStep('validate', 'done', 'No classifier training required for this signal visualization app.');
  await finishPipelineAutopilot(
    'ready',
    'Signal analysis app ready. The code workspace will generate an EMG frequency-band visualizer.',
    'Open Preview after the code finishes writing.'
  );
  toast('Signal visualization app ready');
}

async function startPipelineAutopilotTask() {
  if (S.pipelineBuilder.autopilotLoading) return;
  const taskSteps = (S.pipelineBuilder.acquisition?.steps || [])
    .filter(step => String(step?.action || '') === 'record_label');
  const index = clamp(Number(S.pipelineBuilder.guidedTaskIndex || 0), 0, Math.max(taskSteps.length - 1, 0));
  const step = taskSteps[index];
  if (!step) {
    toast('No guided task is ready yet.', 'yellow');
    return;
  }
  S.pipelineBuilder.autopilotLoading = true;
  S.pipelineBuilder.error = '';
  setPipelineAutopilotAction('collecting', '', `Recording ${step.label || `task ${index + 1}`}.`, 'Hold the task until KYMA finishes this timed capture.');
  try {
    if (!S.streaming) await startPipelineAutopilotStream();
    const singleProtocol = {
      ...(S.pipelineBuilder.guidedProtocol || S.pipelineBuilder.acquisition || {}),
      mode: 'guided_step',
      estimated_seconds: Number(step.duration_s || 8),
      steps: [step],
    };
    const start = await post('/api/pipeline/acquisition_run', {
      ...buildPipelineActionPayload(),
      protocol: singleProtocol,
      action: 'start',
      context: {
        ...(buildPipelineActionPayload().context || {}),
        append_acquisition: index > 0,
      },
    });
    S.pipelineBuilder.acquisitionRun = start.run || null;
    await waitForPipelineAcquisitionCompletion(singleProtocol);
    const run = S.pipelineBuilder.acquisitionRun || {};
    const accepted = Number(run.accepted_windows || 0);
    const rejected = Number(run.rejected_windows || 0);
    if (run.last_reject_reason && rejected > accepted) {
      setPipelineAutopilotAction('needs_data', 'retry', 'That task was too noisy.', `${run.last_reject_reason} Fix the setup and retry the task.`);
      toast('Task was too noisy. Retry it after fixing setup.', 'yellow');
      return;
    }
    if (index + 1 < taskSteps.length) {
      setPipelineGuidedTaskReady(index + 1);
      toast(`Task ${index + 1} complete. Prepare for the next task.`);
      return;
    }
    setPipelineAutopilotStep('validate', 'running', 'Training from guided task data.');
    await runPipelineAutopilotStep('validate', runPipelineAcquisitionTrain, `${accepted} clean windows accepted`);
    if (!S.pipelineBuilder.trainResult?.trained || S.pipelineBuilder.trainResult?.ready === false) {
      const warning = (S.pipelineBuilder.trainResult?.result?.quality_warnings || S.pipelineBuilder.trainResult?.next_actions || [])[0] || 'Model quality is not high enough yet.';
      setPipelineAutopilotStep('validate', 'needs_data', warning);
      setPipelineAutopilotAction('needs_data', 'retry', 'The model needs clearer examples before export.', `${warning} Click Retry Tasks after improving the recording.`);
      toast('Model needs clearer examples', 'yellow');
      return;
    }
    await completePipelineAutopilotFromTrainedLiveModel();
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Task recording failed';
    setPipelineAutopilotAction('failed', 'retry', 'Task recording stopped before the model was ready.', `${S.pipelineBuilder.error} Fix the setup and retry.`);
    toast(`Task recording failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.autopilotLoading = false;
    syncPipelineBuilderUI();
  }
}

async function continuePipelineAutopilotLive(options = {}) {
  if (S.pipelineBuilder.autopilotLoading) return;
  S.pipelineBuilder.autopilotLoading = true;
  S.pipelineBuilder.error = '';
  setPipelineAutopilotAction('collecting', '', 'Starting guided task collection.', 'Keep electrodes attached and follow each task prompt.');
  try {
    if (!S.pipelineBuilder.plan) await runPipelineAutopilotStep('plan', runPipelinePlan, 'Recipe generated');
    if (!S.pipelineBuilder.acquisition) await runPipelineAutopilotStep('labels', runPipelineAcquisition, 'Acquisition protocol ready');
    if (!S.streaming) await startPipelineAutopilotStream();
    await pipelineSleep(1200);
    await runPipelineAutopilotStep('quality', runPipelineQA, 'Signal QA complete');
    const qaScore = Number(S.pipelineBuilder.qa?.score || 0);
    if (Number.isFinite(qaScore) && qaScore < 55) {
      setPipelineAutopilotStep('validate', 'needs_data', `Signal QA is low (${qaScore.toFixed(1)}/100).`);
      const reason = (S.pipelineBuilder.qa?.artifacts || [])[0]?.detail || 'Re-seat electrodes, reduce movement, and check channel contact.';
      setPipelineAutopilotAction('needs_data', 'retry', 'The signal is too noisy to train yet.', `${reason} Then click Retry Tasks.`);
      toast('Signal is too noisy. Fix electrodes and retry.', 'yellow');
      return;
    }
    if (!pipelineBuildNeedsTraining()) {
      await completePipelineAutopilotFromSignalAnalysis();
      return;
    }
    const runnableProtocol = buildRunnablePromptAcquisitionProtocol(S.pipelineBuilder.acquisition);
    if (!runnableProtocol.steps.length) throw new Error('No trainable task prompts are available.');
    S.pipelineBuilder.acquisition = runnableProtocol;
    S.pipelineBuilder.guidedProtocol = runnableProtocol;
    S.pipelineBuilder.guidedTaskIndex = options.retry ? Number(S.pipelineBuilder.guidedTaskIndex || 0) : 0;
    setPipelineAutopilotStep('validate', 'needs_data', 'Waiting for the user to start each guided task.');
    setPipelineGuidedTaskReady(S.pipelineBuilder.guidedTaskIndex);
    toast('Prepare for the first task, then press Start This Task');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Live Autopilot failed';
    setPipelineAutopilotStep('validate', 'failed', S.pipelineBuilder.error);
    setPipelineAutopilotAction('failed', 'retry', 'Live Autopilot stopped before the model was ready.', `${S.pipelineBuilder.error} Fix the setup and click Retry Tasks.`);
    toast(`Live Autopilot failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.autopilotLoading = false;
    syncPipelineBuilderUI();
  }
}

async function runPipelineAutopilot() {
  if (S.pipelineBuilder.autopilotLoading) return;
  syncPipelineBuilderInputsFromForm();
  normalizePipelineSourceForPrompt();
  if (!S.pipelineBuilder.prompt) {
    S.pipelineBuilder.error = 'Prompt required';
    syncPipelineBuilderUI();
    toast('Describe what you want KYMA to build first.', 'yellow');
    return;
  }
  const datasetMode = pipelineBuilderUsesDatasetSource();
  resetPipelineAutopilot(datasetMode ? 'dataset' : (pipelineBuilderUsesSyntheticSource() ? 'synthetic' : 'live'));
  if (!S.pipelineBuilder.sourceChoiceConfirmed || !S.pipelineBuilder.appChoiceConfirmed) {
    setPipelineAutopilotAction(
      'choose_setup',
      'choose_setup',
      'Choose the source and app type before KYMA records or trains anything.',
      'Pick live electrodes, synthetic demo, or dataset import. Then choose the app KYMA should generate.'
    );
    return;
  }
  S.pipelineBuilder.autopilotLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    await runPipelineAutopilotStep('plan', runPipelinePlan, 'Recipe generated');
    if (!S.pipelineBuilder.plan) throw new Error('Autopilot could not create a pipeline plan.');

    if (datasetMode) {
      await runPipelineAutopilotStep('inspect', runPipelineDatasetInspect, 'Dataset scanned');
      const readiness = S.pipelineBuilder.dataset?.readiness || {};
      if (!readiness.ready) {
        const gaps = Array.isArray(readiness.gaps) ? readiness.gaps.filter(Boolean) : [];
        setPipelineAutopilotStep('ingest', 'needs_data', gaps[0] || 'Dataset needs labels or channel mapping before training.');
        await runPipelineAutopilotStep('save', savePipelineProject, 'Draft project saved');
        await runPipelineProjectList().catch(() => null);
        await finishPipelineAutopilot(
          'needs_data',
          'Autopilot created a draft project, but the dataset is not trainable yet.',
          gaps[0] || 'Add or map labels, then click Build Automatically again.'
        );
        toast('Autopilot needs dataset labels or mapping', 'yellow');
        return;
      }
      await runPipelineAutopilotStep('ingest', runPipelineDatasetIngest, 'Dataset ingested');
    } else {
      const wantsSynthetic = pipelineBuilderUsesSyntheticSource();
      if (!wantsSynthetic && S.streaming && String(S.streamSource || '').toLowerCase() === 'synthetic') {
        await post('/api/stream/stop').catch(() => null);
        S.streaming = false;
        await loadStatus().catch(() => null);
      }
      if (!S.streaming && wantsSynthetic) {
        await post('/api/stream/start', { source: 'synthetic', synthetic_scenario: 'clean' }).catch(() => null);
        await loadStatus().catch(() => null);
        await pipelineSleep(1200);
      }
      await runPipelineAutopilotStep('labels', runPipelineAcquisition, 'Acquisition protocol ready');
      const isSynthetic = wantsSynthetic && String(S.streamSource || '').toLowerCase() === 'synthetic';
      if (!isSynthetic) {
        prepareTourView({
          tab: 'dashboard',
          workspace: 'live',
          expand: ['#card-stream', '#card-channel-activity', '#card-review-markers'],
        });
        setSelectedSource('hardware');
        syncStreamModeUI();
        window.setTimeout(() => {
          $('card-stream')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 150);
        setPipelineAutopilotStep('quality', 'needs_data', 'Connect electrodes and start the live stream before QA.');
        setPipelineAutopilotStep('validate', 'needs_data', 'Waiting for electrode setup and task recording.');
        await runPipelineAutopilotStep('save', savePipelineProject, 'Draft project saved');
        await runPipelineProjectList().catch(() => null);
        const labels = (S.pipelineBuilder.acquisition?.steps || [])
          .map(step => step.label || step.name)
          .filter(Boolean)
          .slice(0, 4)
          .join(', ');
        setPipelineAutopilotAction(
          'waiting_setup',
          'setup',
          'Connect the electrodes, start the stream, then run the guided tasks.',
          labels ? `Tasks KYMA will prompt: ${labels}. Check the channel activity bars, then click I Am Ready - Start Tasks.` : 'Check the channel activity bars, then click I Am Ready - Start Tasks.'
        );
        toast('Connect electrodes, then start the guided tasks', 'yellow');
        return;
      }
      await runPipelineAutopilotStep('quality', runPipelineQA, 'Signal QA complete');
      if (!pipelineBuildNeedsTraining()) {
        await completePipelineAutopilotFromSignalAnalysis();
        return;
      }
      const runnableProtocol = buildRunnablePromptAcquisitionProtocol(S.pipelineBuilder.acquisition);
      if (!runnableProtocol.steps.length) throw new Error('Autopilot could not create trainable labels.');
      S.pipelineBuilder.acquisition = runnableProtocol;
      const start = await post('/api/pipeline/acquisition_run', {
        ...buildPipelineActionPayload(),
        protocol: runnableProtocol,
        action: 'start',
      });
      S.pipelineBuilder.acquisitionRun = start.run || null;
      await waitForPipelineAcquisitionCompletion(runnableProtocol);
      await runPipelineAutopilotStep('validate', runPipelineAcquisitionTrain, 'Synthetic labels trained');
    }

    if (datasetMode) {
      await runPipelineAutopilotStep('validate', async () => {
        if (!S.pipelineBuilder.trainResult?.trained) await runPipelineBaselineTrain();
        if (!S.pipelineBuilder.trainResult?.trained) {
          throw new Error('Training needs more labeled windows.');
        }
      }, 'Model trained');
    } else if (!S.pipelineBuilder.trainResult?.trained || S.pipelineBuilder.trainResult?.ready === false) {
      setPipelineAutopilotStep('validate', 'needs_data', 'Training needs more accepted label windows.');
      const warning = (S.pipelineBuilder.trainResult?.result?.quality_warnings || S.pipelineBuilder.trainResult?.next_actions || [])[0] || 'Record clearer examples, then continue.';
      await finishPipelineAutopilot('needs_data', 'Autopilot needs a stronger model before code generation.', warning);
      return;
    }

    await runPipelineCompare().catch(() => null);
    await runPipelineEmbedding().catch(() => null);
    await runPipelineThroughput().catch(() => null);
    await runPipelineAutopilotStep('package', runPipelineExport, 'Export package created');
    await runPipelineRuntime().catch(() => null);
    await runPipelineDeploySmoke().catch(() => null);
    await runPipelineAutopilotStep('save', savePipelineProject, 'Project report saved');
    await runPipelineProjectList().catch(() => null);
    await finishPipelineAutopilot(
      'ready',
      'Autopilot finished. The project, report, and package are ready.',
      'Open the project report or download the package from the Project Registry.'
    );
    toast('Autopilot complete');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Autopilot failed';
    await finishPipelineAutopilot(
      'failed',
      'Autopilot stopped before completion.',
      S.pipelineBuilder.error
    );
    toast(`Autopilot failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.autopilotLoading = false;
    syncPipelineBuilderUI();
  }
}

function buildRunnablePromptAcquisitionProtocol(acquisition) {
  const sourceSteps = Array.isArray(acquisition?.steps) ? acquisition.steps : [];
  const syntheticScenarios = ['clean', 'fatigue', 'contraction', 'motion', 'drift', 'line_noise'];
  const recordSteps = sourceSteps
    .filter(step => String(step?.action || '') === 'record_label')
    .slice(0, 6)
    .map((step, index) => ({
      ...step,
      id: step.id || `label_${index + 1}`,
      label: String(step.label || step.name || `label_${index + 1}`).trim() || `label_${index + 1}`,
      action: 'record_label',
      duration_s: clamp(Number(step.duration_s || 8), 4, 12),
      target_windows: clamp(Number(step.target_windows || 12), 6, 25),
      synthetic_scenario: step.synthetic_scenario || syntheticScenarios[index % syntheticScenarios.length],
    }));
  return {
    ...(acquisition || {}),
    mode: 'guided_auto',
    estimated_seconds: recordSteps.reduce((sum, step) => sum + Number(step.duration_s || 0), 0),
    steps: recordSteps,
    automation: [
      'Record prompt labels directly into the live acquisition buffer.',
      'Train immediately when the guided run finishes.',
      'Start live prompt-model inference and package the model card.',
    ],
  };
}

async function waitForPipelineAcquisitionCompletion(protocol) {
  const startedAt = Date.now();
  const expectedMs = Math.max(12000, Number(protocol?.estimated_seconds || 30) * 1000 + 10000);
  let latest = null;
  while (Date.now() - startedAt < expectedMs) {
    const res = await post('/api/pipeline/acquisition_run', {
      ...buildPipelineActionPayload(),
      protocol,
      action: 'status',
    });
    latest = res.run || null;
    S.pipelineBuilder.acquisitionRun = latest;
    const accepted = Number(latest?.accepted_windows || 0);
    const perLabel = latest?.per_label || {};
    S.pipelineBuilder.createStage = accepted
      ? `Collecting labels... ${accepted} windows`
      : 'Collecting labels...';
    syncPipelineBuilderUI();
    if (latest && !latest.running) return latest;
    await pipelineSleep(900);
  }
  throw new Error('Acquisition timed out before training.');
}

async function runPipelineCreateModel() {
  if (S.pipelineBuilder.createLoading) return;
  syncPipelineBuilderInputsFromForm();
  if (!S.pipelineBuilder.prompt) {
    S.pipelineBuilder.error = 'Prompt required';
    syncPipelineBuilderUI();
    toast('Describe the model first.', 'yellow');
    return;
  }
  S.pipelineBuilder.createLoading = true;
  S.pipelineBuilder.createStage = pipelineBuilderUsesDatasetSource() ? 'Preparing dataset...' : 'Preparing stream...';
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    if (pipelineBuilderUsesDatasetSource()) {
      S.pipelineBuilder.createStage = 'Generating dataset plan...';
      syncPipelineBuilderUI();
      await runPipelinePlan();
      if (!S.pipelineBuilder.plan) throw new Error('Plan was not created.');

      S.pipelineBuilder.createStage = 'Scanning dataset readiness...';
      syncPipelineBuilderUI();
      await runPipelineDatasetInspect();
      const readiness = S.pipelineBuilder.dataset?.readiness || {};
      if (!readiness.ready) {
        const gaps = Array.isArray(readiness.gaps) ? readiness.gaps.filter(Boolean) : [];
        throw new Error(gaps[0] || 'Dataset is not ready for supervised training yet.');
      }

      S.pipelineBuilder.createStage = 'Ingesting and training...';
      syncPipelineBuilderUI();
      await runPipelineDatasetIngest();
      if (!S.pipelineBuilder.trainResult?.trained) throw new Error('Dataset ingest did not produce a trained model.');
      if (S.pipelineBuilder.trainResult?.ready === false) {
        throw new Error('Dataset model trained, but validation is too low for a finished model.');
      }

      S.pipelineBuilder.createStage = 'Comparing models...';
      syncPipelineBuilderUI();
      await runPipelineCompare();

      S.pipelineBuilder.createStage = 'Reading embeddings...';
      syncPipelineBuilderUI();
      await runPipelineEmbedding();

      S.pipelineBuilder.createStage = 'Benchmarking speed...';
      syncPipelineBuilderUI();
      await runPipelineThroughput();

      S.pipelineBuilder.createStage = 'Packaging export...';
      syncPipelineBuilderUI();
      await runPipelineExport();

      S.pipelineBuilder.createStage = 'Checking browser runtime...';
      syncPipelineBuilderUI();
      await runPipelineRuntime();

      S.pipelineBuilder.createStage = 'Verifying deploy package...';
      syncPipelineBuilderUI();
      await runPipelineDeploySmoke();

      S.pipelineBuilder.createStage = 'Dataset model ready';
      toast('Create Model complete. Dataset model package is ready.');
      return;
    }

    if (!S.streaming) {
      await post('/api/stream/start', {
        source: S.streamSource || 'synthetic',
        synthetic_scenario: 'clean',
      });
      await loadStatus();
    }

    S.pipelineBuilder.createStage = 'Generating plan...';
    syncPipelineBuilderUI();
    await runPipelinePlan();
    if (!S.pipelineBuilder.plan) throw new Error('Plan was not created.');

    S.pipelineBuilder.createStage = 'Checking signal quality...';
    syncPipelineBuilderUI();
    await runPipelineQA();

    S.pipelineBuilder.createStage = 'Building guided labels...';
    syncPipelineBuilderUI();
    await runPipelineAcquisition();
    const runnableProtocol = buildRunnablePromptAcquisitionProtocol(S.pipelineBuilder.acquisition);
    if (!runnableProtocol.steps.length) throw new Error('No trainable labels were generated.');
    S.pipelineBuilder.acquisition = runnableProtocol;

    S.pipelineBuilder.createStage = 'Collecting labels...';
    syncPipelineBuilderUI();
    const start = await post('/api/pipeline/acquisition_run', {
      ...buildPipelineActionPayload(),
      protocol: runnableProtocol,
      action: 'start',
    });
    S.pipelineBuilder.acquisitionRun = start.run || null;
    syncPipelineBuilderUI();
    await waitForPipelineAcquisitionCompletion(runnableProtocol);

    S.pipelineBuilder.createStage = 'Training prompt model...';
    syncPipelineBuilderUI();
    await runPipelineAcquisitionTrain();
    if (!S.pipelineBuilder.trainResult?.trained) throw new Error('Prompt model needs more accepted label windows.');
    if (S.pipelineBuilder.trainResult?.ready === false) {
      throw new Error('Prompt model trained, but validation is too low for a finished live model.');
    }

    S.pipelineBuilder.createStage = 'Checking live runtime...';
    syncPipelineBuilderUI();
    await runPipelineRuntime();

    S.pipelineBuilder.createStage = 'Reading embeddings...';
    syncPipelineBuilderUI();
    await runPipelineEmbedding();

    S.pipelineBuilder.createStage = 'Packaging export...';
    syncPipelineBuilderUI();
    await runPipelineExport();

    S.pipelineBuilder.createStage = 'Model live';
    toast('Create Model complete. Live prompt inference is running.');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Create Model failed';
    toast(`Create Model failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.createLoading = false;
    syncPipelineBuilderUI();
  }
}

async function runPipelineLabelSuggestions() {
  if (S.pipelineBuilder.labelsLoading || !S.pipelineBuilder.plan) return;
  S.pipelineBuilder.labelsLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/label_suggestions', buildPipelineActionPayload());
    S.pipelineBuilder.labelSuggestions = res.labels || null;
    toast('Label suggestions ready');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Labels failed';
    toast(`Label suggestions failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.labelsLoading = false;
    syncPipelineBuilderUI();
  }
}

async function runPipelineDatasetInspect() {
  if (S.pipelineBuilder.datasetLoading || !S.pipelineBuilder.plan) return;
  S.pipelineBuilder.datasetLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/dataset_inspect', {
      ...buildPipelineActionPayload(),
      path: S.pipelineBuilder.datasetPath || '',
      schema_mapping: buildPipelineSchemaMappingPayload(),
    });
    S.pipelineBuilder.dataset = res.dataset || null;
    ensurePipelineSchemaMapping(S.pipelineBuilder.dataset);
    toast('Dataset import scan ready');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Import scan failed';
    toast(`Dataset scan failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.datasetLoading = false;
    syncPipelineBuilderUI();
  }
}

async function runPipelineDatasetIngest() {
  if (S.pipelineBuilder.ingestLoading || !S.pipelineBuilder.plan) return;
  S.pipelineBuilder.ingestLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/dataset_ingest', {
      ...buildPipelineActionPayload(),
      path: S.pipelineBuilder.datasetPath || '',
      schema_mapping: buildPipelineSchemaMappingPayload(),
      append: false,
      train_after: true,
      max_windows: 240,
      max_windows_per_label: 80,
    });
    S.pipelineBuilder.ingest = res.ingest || null;
    S.pipelineBuilder.acquisitionRun = res.run || S.pipelineBuilder.acquisitionRun;
    if (res.training) {
      S.pipelineBuilder.trainResult = res.training;
      const manager = await get('/api/pipeline/models').catch(() => null);
      if (manager) S.pipelineBuilder.modelManager = manager;
      if (manager?.active) S.promptModel.server = manager.active;
    }
    const count = Number(S.pipelineBuilder.ingest?.added_windows || 0);
    toast(res.training?.trained ? `Dataset ingested and model trained (${count} windows)` : `Dataset ingested (${count} windows)`, res.training?.trained ? '' : 'yellow');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Dataset ingest failed';
    toast(`Dataset ingest failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.ingestLoading = false;
    syncPipelineBuilderUI();
  }
}

async function runPipelineThroughput() {
  if (S.pipelineBuilder.throughputLoading || !S.pipelineBuilder.plan) return;
  S.pipelineBuilder.throughputLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/throughput', {
      ...buildPipelineActionPayload(),
      quantization_step: 0.02,
      max_workers: Math.min(8, Math.max(2, Number(navigator.hardwareConcurrency || 4))),
    });
    S.pipelineBuilder.throughput = res.throughput || null;
    toast('Speed benchmark ready');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Speed failed';
    toast(`Speed benchmark failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.throughputLoading = false;
    syncPipelineBuilderUI();
  }
}

async function runPipelineCompare() {
  if (S.pipelineBuilder.compareLoading || !S.pipelineBuilder.plan) return;
  S.pipelineBuilder.compareLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/compare', buildPipelineActionPayload());
    S.pipelineBuilder.comparison = res.comparison || null;
    toast('Model comparison ready');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Compare failed';
    toast(`Model comparison failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.compareLoading = false;
    syncPipelineBuilderUI();
  }
}

async function runPipelineRuntime() {
  if (S.pipelineBuilder.runtimeLoading || !S.pipelineBuilder.plan) return;
  S.pipelineBuilder.runtimeLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/browser_runtime', buildPipelineActionPayload());
    S.pipelineBuilder.runtime = res.runtime || null;
    toast(S.pipelineBuilder.runtime?.ready ? 'Browser runtime ready' : 'Browser runtime needs export', S.pipelineBuilder.runtime?.ready ? '' : 'yellow');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Runtime failed';
    toast(`Runtime check failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.runtimeLoading = false;
    syncPipelineBuilderUI();
  }
}

async function runPipelineDeploySmoke() {
  if (S.pipelineBuilder.deployLoading || !S.pipelineBuilder.plan) return;
  S.pipelineBuilder.deployLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/deploy_smoke', buildPipelineActionPayload());
    S.pipelineBuilder.deploy = res.deploy || null;
    S.pipelineBuilder.runtime = res.runtime || S.pipelineBuilder.runtime;
    toast(S.pipelineBuilder.deploy?.ready ? 'Deploy smoke passed' : 'Deploy smoke needs review', S.pipelineBuilder.deploy?.ready ? '' : 'yellow');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Deploy verify failed';
    toast(`Deploy verify failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.deployLoading = false;
    syncPipelineBuilderUI();
  }
}

async function runPipelineEmbedding() {
  if (S.pipelineBuilder.embeddingLoading || !S.pipelineBuilder.plan) return;
  S.pipelineBuilder.embeddingLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/embedding_timeline', buildPipelineActionPayload());
    S.pipelineBuilder.embedding = res.embedding || null;
    const score = Number(S.pipelineBuilder.embedding?.readiness?.score || 0);
    toast(`Dataset readiness ${score.toFixed(1)} / 100`);
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Embeddings failed';
    toast(`Embedding timeline failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.embeddingLoading = false;
    syncPipelineBuilderUI();
  }
}

async function runPipelineExport() {
  if (S.pipelineBuilder.exportLoading || !S.pipelineBuilder.plan) return;
  S.pipelineBuilder.exportLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/export', buildPipelineActionPayload());
    S.pipelineBuilder.exportResult = res.export || null;
    const manager = await get('/api/pipeline/exports').catch(() => null);
    if (manager) S.pipelineBuilder.exportManager = manager;
    toast('Pipeline export package created');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Export failed';
    toast(`Pipeline export failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.exportLoading = false;
    syncPipelineBuilderUI();
  }
}

async function runPipelineQA() {
  if (S.pipelineBuilder.qaLoading || !S.pipelineBuilder.plan) return;
  S.pipelineBuilder.qaLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/qa', {
      plan: S.pipelineBuilder.plan || {},
      context: buildPipelinePlanPayload().context || {},
    });
    S.pipelineBuilder.qa = res.qa || null;
    const score = Number(S.pipelineBuilder.qa?.score || 0);
    toast(`Signal QA ${score.toFixed(1)} / 100${S.pipelineBuilder.qa?.ready_for_training ? ' - ready' : ' - review cleanup'}`);
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'QA failed';
    toast(`Pipeline QA failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.qaLoading = false;
    syncPipelineBuilderUI();
  }
}

async function runPipelineBaselineTrain() {
  if (S.pipelineBuilder.trainLoading || !S.pipelineBuilder.plan) return;
  S.pipelineBuilder.trainLoading = true;
  S.pipelineBuilder.error = '';
  syncPipelineBuilderUI();
  try {
    const res = await post('/api/pipeline/train_baseline', {
      plan: S.pipelineBuilder.plan || {},
      classifier: 'LDA',
    });
    S.pipelineBuilder.trainResult = res || null;
    await refreshTrainSummary();
    toast(res.trained ? 'Baseline model trained' : 'Baseline needs labeled windows', res.trained ? '' : 'yellow');
  } catch (e) {
    S.pipelineBuilder.error = e.message || 'Training failed';
    toast(`Baseline training failed: ${S.pipelineBuilder.error}`, 'red');
  } finally {
    S.pipelineBuilder.trainLoading = false;
    syncPipelineBuilderUI();
  }
}

function syncSignalLogicUI() {
  populateSignalLogicTemplateOptions();
  syncSignalLogicSummary();
  renderSignalLogicRules();
  syncSignalLogicRuleStates();
  renderSignalLogicFeed();
  syncSignalLogicLiveReadout();
  syncSignalStudioDrawer();
}

function pushSignalLogicFeed(title, detail, extra = {}) {
  S.logic.feed.unshift({ title, detail, at: Date.now(), ...extra });
  if (S.logic.feed.length > SIGNAL_LOGIC_FEED_MAX) S.logic.feed.length = SIGNAL_LOGIC_FEED_MAX;
  renderSignalLogicFeed();
}

function addSignalLogicRule(rule = {}) {
  S.logic.rules.push(normalizeSignalLogicRule(rule));
  persistSignalLogicConfig();
  syncSignalLogicUI();
}

function addSignalLogicRuleForSource(sourceKey) {
  const meta = getSignalLogicSourceMeta(sourceKey);
  const operator = signalLogicOperatorsForType(meta.type)[0]?.key || (meta.type === 'text' ? 'equals' : '>');
  const value = meta.type === 'text' ? '' : '0';
  addSignalLogicRule({ source: meta.key, operator, value });
}

function removeSignalLogicRule(ruleId) {
  S.logic.rules = (S.logic.rules || []).filter(rule => rule.id !== ruleId);
  cleanupSignalLogicRuntime();
  persistSignalLogicConfig();
  syncSignalLogicUI();
}

function updateSignalLogicRule(ruleId, field, value) {
  const rule = (S.logic.rules || []).find(item => item.id === ruleId);
  if (!rule) return;
  if (field === 'enabled') rule.enabled = !!value;
  else if (field === 'hold_ms' || field === 'cooldown_ms') rule[field] = Math.max(0, Number(value || 0));
  else rule[field] = value;
  const normalized = normalizeSignalLogicRule(rule);
  Object.assign(rule, normalized);
  if (field === 'action' && !String(rule.payload || '').trim()) {
    rule.payload = getSignalLogicActionMeta(rule.action).placeholder || '';
  }
  persistSignalLogicConfig();
  syncSignalLogicUI();
}

function parseSignalLogicNumeric(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseSignalLogicRange(value) {
  const raw = String(value || '').trim();
  if (!raw) return null;
  const parts = raw.split(/\s*(?:\.\.|,|:)\s*/).filter(Boolean);
  if (parts.length < 2) return null;
  const low = Number(parts[0]);
  const high = Number(parts[1]);
  if (!Number.isFinite(low) || !Number.isFinite(high)) return null;
  return { low: Math.min(low, high), high: Math.max(low, high) };
}

function signalLogicConditionMatched(rule, value, runtime, sourceMeta) {
  if (sourceMeta.type === 'text') {
    const current = String(value || '').trim().toLowerCase();
    const target = String(rule.value || '').trim().toLowerCase();
    const previous = String(runtime.lastValue || '').trim().toLowerCase();
    if (rule.operator === 'equals') return current === target && !!target;
    if (rule.operator === 'not_equals') return !!current && current !== target;
    if (rule.operator === 'contains') return !!target && current.includes(target);
    if (rule.operator === 'starts_with') return !!target && current.startsWith(target);
    if (rule.operator === 'changes_to') return !!target && current === target && previous !== current;
    return false;
  }
  const current = Number(value || 0);
  const target = parseSignalLogicNumeric(rule.value);
  const previous = Number(runtime.lastValue);
  if (rule.operator === 'between') {
    const range = parseSignalLogicRange(rule.value);
    return !!range && current >= range.low && current <= range.high;
  }
  if (rule.operator === 'crosses_above') {
    return target != null && Number.isFinite(previous) && previous <= target && current > target;
  }
  if (rule.operator === 'crosses_below') {
    return target != null && Number.isFinite(previous) && previous >= target && current < target;
  }
  if (rule.operator === 'rises_by') {
    return target != null && Number.isFinite(previous) && (current - previous) >= target;
  }
  if (rule.operator === 'falls_by') {
    return target != null && Number.isFinite(previous) && (previous - current) >= target;
  }
  if (target == null) return false;
  if (rule.operator === '>') return current > target;
  if (rule.operator === '>=') return current >= target;
  if (rule.operator === '<') return current < target;
  if (rule.operator === '<=') return current <= target;
  if (rule.operator === '==') return current === target;
  if (rule.operator === '!=') return current !== target;
  return false;
}

function signalLogicDefaultSelection(state = getReviewRenderState()) {
  const artifact = Array.isArray(S.review.artifacts) && S.review.artifacts.length ? S.review.artifacts[0] : null;
  if (artifact) {
    return { startSample: artifact.startSample, endSample: artifact.endSample };
  }
  if (S.review.lastStats?.samples > 1) {
    return { startSample: S.review.lastStats.startSample, endSample: S.review.lastStats.endSample };
  }
  const endSample = Math.max(Number(state.baseAbs || 0) + Math.max(Number(state.filled || 0) - 1, 0), 0);
  const span = Math.max(2, Math.round(Number(state.sampleRate || S.sampleRate || 250) * 0.2));
  return {
    startSample: Math.max(Number(state.baseAbs || 0), endSample - span + 1),
    endSample,
  };
}

function estimateSignalLogicDominantHz(state, startSample, endSample, channel) {
  const sampleRate = Math.max(Number(state?.sampleRate || S.sampleRate || 250), 1);
  const count = Math.max(0, Number(endSample) - Number(startSample) + 1);
  if (count < 4) return 0;
  let zeroCrossings = 0;
  let lastSign = 0;
  for (let abs = startSample; abs <= endSample; abs += 1) {
    const idx = bufferIndexForAbsSample(state, abs);
    const value = Number(state.emg[channel]?.[idx] || 0);
    const sign = value === 0 ? lastSign : (value > 0 ? 1 : -1);
    if (lastSign && sign && sign !== lastSign) zeroCrossings += 1;
    if (sign) lastSign = sign;
  }
  const durationS = count / sampleRate;
  return durationS > 0 ? zeroCrossings / (2 * durationS) : 0;
}

function buildSignalLogicSnapshotForRange(state, startSample, endSample) {
  const selection = { startSample, endSample };
  const stats = computeSelectionStats(selection, state);
  if (!stats) return null;
  const prediction = (S.review.paused && S.review.predictionSnapshot)
    ? S.review.predictionSnapshot
    : S.lastPrediction;
  const overlappingArtifacts = (S.review.artifacts || []).filter(item => Number(item.endSample) >= startSample && Number(item.startSample) <= endSample);
  const focusChannel = Number(stats.focusChannel || 0);
  const baseline = Number(S.propRestRms?.[focusChannel] || 0);
  return {
    profile: S.signalProfileName || 'Signal',
    focus_rms: Number(stats.focusRms || 0),
    rms: Number(stats.rms || 0),
    peak_to_peak: Number(stats.peakToPeak || 0),
    dominant_hz: Number(estimateSignalLogicDominantHz(state, startSample, endSample, focusChannel) || 0),
    visible_span_ms: Number(stats.durationMs || 0),
    focus_channel: String(stats.focusLabel || S.channelLabels[focusChannel] || `CH${focusChannel + 1}`),
    clip_pct: Number(S.diagnostics?.noise?.clip_pct || 0),
    hum_db: Math.max(Number(S.diagnostics?.noise?.hum_50_db || 0), Number(S.diagnostics?.noise?.hum_60_db || 0)),
    drift_db: Number(S.diagnostics?.noise?.drift_db || 0),
    issue_count: Number(overlappingArtifacts.length || 0),
    baseline_delta: Number(stats.focusRms || 0) - baseline,
    prediction_label: String(prediction?.label || ''),
    prediction_confidence: Number(prediction?.confidence || 0),
    artifact_label: overlappingArtifacts[0] ? normalizeArtifactKind(overlappingArtifacts[0].kind || overlappingArtifacts[0].label || 'artifact') : 'clean',
    artifact_confidence: overlappingArtifacts[0] ? 0.9 : 0,
    qa_score: readAIOverallScore(),
    signal_age_ms: Number(S.diagnostics?.timing?.signal_age_ms || 0),
    streaming: S.streaming ? 1 : 0,
  };
}

async function runSignalLogicReplayTest() {
  if (!S.review.paused && getReviewRenderState().filled) toggleReviewPause(true);
  const state = getReviewRenderState();
  if (!state?.filled) {
    toast('Freeze or stream signal first', 'yellow');
    return;
  }
  const viewport = getReviewViewport(state, canvas.width || 1);
  const selected = getSelectionRange(S.review.selection);
  const range = selected || {
    start: Number(viewport.viewStart || state.baseAbs || 0),
    end: Number(viewport.viewEnd || state.baseAbs || 0),
  };
  const sampleRate = Math.max(Number(state.sampleRate || S.sampleRate || 250), 1);
  const windowSamples = Math.max(12, Math.round(sampleRate * 0.2));
  const stepSamples = Math.max(4, Math.round(sampleRate * 0.05));
  const replayRuntime = {};
  const hits = [];

  for (let start = range.start; start <= Math.max(range.start, range.end - windowSamples + 1); start += stepSamples) {
    const end = Math.min(range.end, start + windowSamples - 1);
    const snapshot = buildSignalLogicSnapshotForRange(state, start, end);
    if (!snapshot) continue;
    (S.logic.rules || []).forEach((rule) => {
      if (!rule.enabled) return;
      if (!replayRuntime[rule.id]) replayRuntime[rule.id] = { lastValue: undefined, activeSince: 0, lastFiredAt: -Infinity };
      const runtime = replayRuntime[rule.id];
      const currentValue = snapshot[rule.source];
      const sourceMeta = getSignalLogicSourceMeta(rule.source);
      const matched = signalLogicConditionMatched(rule, currentValue, runtime, sourceMeta);
      if (matched) {
        if (!runtime.activeSince) runtime.activeSince = start;
        const heldLongEnough = ((end - runtime.activeSince + 1) / sampleRate) * 1000 >= Math.max(0, Number(rule.hold_ms || 0));
        const cooledDown = ((start - Number(runtime.lastFiredAt || -Infinity)) / sampleRate) * 1000 >= Math.max(0, Number(rule.cooldown_ms || 0));
        if (heldLongEnough && cooledDown) {
          runtime.lastFiredAt = end;
          hits.push({
            title: `Replay · ${getSignalLogicActionMeta(rule.action).label}`,
            detail: signalLogicRuleSummary(rule),
            startSample: start,
            endSample: end,
            at: Date.now(),
          });
        }
      } else {
        runtime.activeSince = 0;
      }
      runtime.lastValue = currentValue;
    });
  }

  pushSignalLogicFeed(
    'Replay Test',
    hits.length
      ? `${hits.length} match${hits.length === 1 ? '' : 'es'} from ${reviewPointLabel(range.start, state)} to ${reviewPointLabel(range.end, state)}.`
      : `No rule matches from ${reviewPointLabel(range.start, state)} to ${reviewPointLabel(range.end, state)}.`,
    { startSample: range.start, endSample: range.end },
  );
  hits.slice(0, 10).forEach(item => pushSignalLogicFeed(item.title, item.detail, item));
  if (hits.length) {
    focusSignalLogicFeedRange(hits[0].startSample, hits[0].endSample);
    toast(`Replay found ${hits.length} match${hits.length === 1 ? '' : 'es'}`);
  } else {
    toast('Replay found no matches', 'yellow');
  }
  syncSignalStudioDrawer();
}

async function saveProgrammaticReviewMarker(event, note = '', selection = signalLogicDefaultSelection()) {
  const state = getReviewRenderState();
  const startSample = Number(selection?.startSample);
  const endSample = Number(selection?.endSample);
  const hasRange = Number.isFinite(startSample) && Number.isFinite(endSample) && Math.abs(endSample - startSample) >= 1;
  const orderedStart = hasRange ? Math.min(startSample, endSample) : null;
  const orderedEnd = hasRange ? Math.max(startSample, endSample) : null;
  const pointSample = hasRange ? null : Math.max(Number(state.baseAbs || 0) + Math.max(Number(state.filled || 0) - 1, 0), 0);
  const body = {
    event,
    note,
    selection_start_s: hasRange ? (orderedStart - state.baseAbs) / Math.max(state.sampleRate, 1) : undefined,
    selection_end_s: hasRange ? (orderedEnd - state.baseAbs) / Math.max(state.sampleRate, 1) : undefined,
    selection_start_sample: hasRange ? orderedStart : undefined,
    selection_end_sample: hasRange ? orderedEnd : undefined,
    sample_time_s: Number.isFinite(pointSample) ? (pointSample - state.baseAbs) / Math.max(state.sampleRate, 1) : undefined,
    sample_index: Number.isFinite(pointSample) ? pointSample : undefined,
  };
  const localMarker = {
    event,
    note,
    selection: hasRange ? { start_sample: orderedStart, end_sample: orderedEnd } : null,
    sampleIndex: Number.isFinite(pointSample) ? pointSample : orderedEnd,
    createdAt: Date.now(),
  };
  if (!S.streaming) {
    pushReviewMarker(localMarker);
    return;
  }
  try {
    const out = await post('/api/review/marker', body);
    if (!S.ws || S.ws.readyState !== WebSocket.OPEN) {
      pushReviewMarker({
        ...(out.marker || {}),
        createdAt: Date.now(),
        sampleIndex: Number.isFinite(out.marker?.sample_index) ? Number(out.marker.sample_index) : localMarker.sampleIndex,
      });
    }
  } catch {
    pushReviewMarker(localMarker);
  }
}

function selectionMetaForLogicFeed(selection = signalLogicDefaultSelection()) {
  const startSample = Number(selection?.startSample);
  const endSample = Number(selection?.endSample);
  if (!Number.isFinite(startSample) || !Number.isFinite(endSample)) return {};
  return {
    startSample: Math.min(startSample, endSample),
    endSample: Math.max(startSample, endSample),
  };
}

async function runSignalLogicAction(rule, snapshot) {
  const runtime = ensureSignalLogicRuntime(rule.id);
  const sourceValue = snapshot[rule.source];
  const action = rule.action;
  const payload = String(rule.payload || '').trim();
  const label = signalLogicRuleSummary(rule);
  if (action === 'toast') {
    const msg = payload || `${getSignalLogicSourceMeta(rule.source).label} matched at ${formatSignalLogicValue(rule.source, sourceValue)}`;
    toast(msg);
    runtime.lastMessage = msg;
    pushSignalLogicFeed('Toast', msg, selectionMetaForLogicFeed());
    return;
  }
  if (action === 'save_marker') {
    const event = payload || `logic_${rule.source}`;
    const note = `${label} | value ${formatSignalLogicValue(rule.source, sourceValue)}`;
    const selection = signalLogicDefaultSelection();
    await saveProgrammaticReviewMarker(event, note, selection);
    runtime.lastMessage = `Saved marker ${event}`;
    pushSignalLogicFeed('Marker Saved', `${event} from ${formatSignalLogicValue(rule.source, sourceValue)}`, selectionMetaForLogicFeed(selection));
    return;
  }
  if (action === 'freeze_focus') {
    const selection = signalLogicDefaultSelection();
    spotlightReviewRange(selection.startSample, selection.endSample, 'Logic Focus');
    runtime.lastMessage = 'Review scope frozen and focused';
    pushSignalLogicFeed('Freeze + Focus', label, selectionMetaForLogicFeed(selection));
    return;
  }
  if (action === 'spotlight_artifact') {
    const segments = artifactRadarSegments();
    if (segments.length) {
      spotlightReviewRange(segments[0].startSample, segments[0].endSample, `Logic Focus: ${segments[0].label}`);
      runtime.lastMessage = `Spotlighted ${segments[0].label.toLowerCase()}`;
      pushSignalLogicFeed('Artifact Spotlight', segments[0].detail || segments[0].label, {
        startSample: segments[0].startSample,
        endSample: segments[0].endSample,
      });
    } else {
      const selection = signalLogicDefaultSelection();
      spotlightReviewRange(selection.startSample, selection.endSample, 'Logic Focus');
      runtime.lastMessage = 'No artifact segment; focused recent signal instead';
      pushSignalLogicFeed('Artifact Spotlight', 'No artifact segment was active, so the runtime focused the latest window.', selectionMetaForLogicFeed(selection));
    }
    return;
  }
  if (action === 'run_ai_scan') {
    await runAICopilot({ silent: true, background: false, autopilot: true });
    runtime.lastMessage = 'Triggered AI Scan';
    pushSignalLogicFeed('AI Scan', label, selectionMetaForLogicFeed());
    return;
  }
  if (action === 'gesture') {
    const gesture = payload || String(snapshot.prediction_label || '').trim();
    if (!gesture) throw new Error('Gesture payload is empty');
    await post(`/api/gesture/${encodeURIComponent(gesture)}`);
    runtime.lastMessage = `Sent gesture ${gesture}`;
    pushSignalLogicFeed('Gesture Sent', gesture, selectionMetaForLogicFeed());
    return;
  }
  if (action === 'move_joint') {
    const [jointRaw, angleRaw] = payload.split(':');
    const jointId = Number(jointRaw);
    const angle = Number(angleRaw);
    if (!Number.isFinite(jointId) || !Number.isFinite(angle)) throw new Error('Use joint:angle payload');
    await post('/api/move', { joint_id: jointId, angle });
    runtime.lastMessage = `Moved joint ${jointId} to ${angle}`;
    pushSignalLogicFeed('Joint Move', `Joint ${jointId} -> ${angle}`, selectionMetaForLogicFeed());
    return;
  }
  if (action === 'digital_write') {
    const [pinRaw, valueRaw] = payload.split(':');
    const pin = Number(pinRaw);
    const value = Number(valueRaw);
    if (!Number.isFinite(pin) || !Number.isFinite(value)) throw new Error('Use pin:value payload');
    await post('/api/digital_write', { pin, value });
    runtime.lastMessage = `Pin ${pin} -> ${value}`;
    pushSignalLogicFeed('Digital Write', `Pin ${pin} -> ${value}`, selectionMetaForLogicFeed());
    return;
  }
  if (action === 'osc_message') {
    const [addressPart, rawValue] = payload.split('=');
    const address = String(addressPart || '').trim();
    if (!address) throw new Error('Use /address=value payload');
    let value = rawValue;
    if (typeof rawValue === 'string') {
      const trimmed = rawValue.trim();
      if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
        try { value = JSON.parse(trimmed); } catch {}
      } else if (/^-?\d+(?:\.\d+)?$/.test(trimmed)) {
        value = Number(trimmed);
      }
    }
    await post('/api/osc/send', { address, value });
    runtime.lastMessage = `OSC ${address}`;
    pushSignalLogicFeed('OSC Message', `${address} -> ${typeof value === 'string' ? value : JSON.stringify(value)}`, selectionMetaForLogicFeed());
    return;
  }
  if (action === 'serial_write') {
    if (!payload) throw new Error('Serial payload is empty');
    await post('/api/arduino/serial_write', { text: payload });
    runtime.lastMessage = 'Serial text sent';
    pushSignalLogicFeed('Serial Write', payload, selectionMetaForLogicFeed());
    return;
  }
  if (action === 'webhook') {
    const separator = payload.indexOf('|');
    const url = separator >= 0 ? payload.slice(0, separator).trim() : payload.trim();
    const bodyText = separator >= 0 ? payload.slice(separator + 1).trim() : '';
    if (!url) throw new Error('Webhook payload must start with a URL');
    let body = {
      rule: signalLogicRuleSummary(rule),
      source: rule.source,
      value: sourceValue,
      profile: snapshot.profile,
      prediction_label: snapshot.prediction_label,
      artifact_label: snapshot.artifact_label,
      qa_score: snapshot.qa_score,
    };
    if (bodyText) {
      try {
        body = JSON.parse(bodyText);
      } catch {
        body = { ...body, payload: bodyText };
      }
    }
    await post('/api/integrations/webhook', { url, body });
    runtime.lastMessage = 'Webhook delivered';
    pushSignalLogicFeed('Webhook', url, selectionMetaForLogicFeed());
    return;
  }
}

function evaluateSignalLogicRuntime() {
  const snapshot = buildSignalLogicSnapshot();
  syncSignalLogicLiveReadout(snapshot);
  if (!S.logic.enabled) return;
  (S.logic.rules || []).forEach(rule => {
    const runtime = ensureSignalLogicRuntime(rule.id);
    if (!rule.enabled) {
      runtime.activeSince = 0;
      runtime.lastValue = snapshot[rule.source];
      return;
    }
    const sourceMeta = getSignalLogicSourceMeta(rule.source);
    const currentValue = snapshot[rule.source];
    const matched = signalLogicConditionMatched(rule, currentValue, runtime, sourceMeta);
    if (matched) {
      if (!runtime.activeSince) runtime.activeSince = Date.now();
      const heldLongEnough = Date.now() - runtime.activeSince >= Math.max(0, Number(rule.hold_ms || 0));
      const cooledDown = Date.now() - Number(runtime.lastFiredAt || 0) >= Math.max(0, Number(rule.cooldown_ms || 0));
      if (heldLongEnough && cooledDown && !runtime.busy) {
        runtime.busy = true;
        runtime.lastFiredAt = Date.now();
        runtime.hotUntil = Date.now() + 1200;
        Promise.resolve(runSignalLogicAction(rule, snapshot))
          .catch(err => {
            runtime.lastMessage = err.message || 'Action failed';
            pushSignalLogicFeed('Logic Error', `${signalLogicRuleSummary(rule)} | ${runtime.lastMessage}`);
            toast(`Signal logic: ${runtime.lastMessage}`, 'red');
          })
          .finally(() => {
            runtime.busy = false;
            syncSignalLogicRuleStates();
          });
      }
    } else {
      runtime.activeSince = 0;
    }
    runtime.lastValue = currentValue;
  });
  syncSignalLogicRuleStates();
}

function artifactStyle(kind) {
  return ({
    hum: { label: 'Hum', color: 'rgba(212,78,92,0.18)', line: 'rgba(212,78,92,0.92)' },
    clip: { label: 'Clip', color: 'rgba(193,92,112,0.18)', line: 'rgba(193,92,112,0.88)' },
    drift: { label: 'Drift', color: 'rgba(178,131,39,0.16)', line: 'rgba(178,131,39,0.84)' },
    flatline: { label: 'Flatline', color: 'rgba(99,114,184,0.16)', line: 'rgba(99,114,184,0.82)' },
    spike: { label: 'Spike', color: 'rgba(214,111,62,0.16)', line: 'rgba(214,111,62,0.84)' },
    motion_artifact: { label: 'Motion', color: 'rgba(79,143,105,0.16)', line: 'rgba(79,143,105,0.84)' },
    low_signal: { label: 'Low Signal', color: 'rgba(99,114,184,0.16)', line: 'rgba(99,114,184,0.82)' },
    artifact: { label: 'Artifact', color: 'rgba(212,78,92,0.16)', line: 'rgba(212,78,92,0.86)' },
    focus: { label: 'Focus', color: 'rgba(88,111,218,0.14)', line: 'rgba(88,111,218,0.82)' },
    prompt_model: { label: 'Model', color: 'rgba(88,111,218,0.16)', line: 'rgba(88,111,218,0.88)' },
  })[kind] || { label: 'Artifact', color: 'rgba(212,78,92,0.16)', line: 'rgba(212,78,92,0.86)' };
}

function promptModelLabelStyle(label) {
  const palette = [
    ['rgba(88,111,218,0.18)', 'rgba(88,111,218,0.92)'],
    ['rgba(79,143,105,0.18)', 'rgba(79,143,105,0.92)'],
    ['rgba(193,92,112,0.18)', 'rgba(193,92,112,0.90)'],
    ['rgba(178,131,39,0.18)', 'rgba(178,131,39,0.90)'],
    ['rgba(134,100,214,0.18)', 'rgba(134,100,214,0.92)'],
    ['rgba(63,143,147,0.18)', 'rgba(63,143,147,0.90)'],
    ['rgba(200,122,71,0.18)', 'rgba(200,122,71,0.90)'],
    ['rgba(99,114,184,0.18)', 'rgba(99,114,184,0.90)'],
  ];
  const raw = String(label || 'model');
  let hash = 0;
  for (let i = 0; i < raw.length; i += 1) hash = ((hash << 5) - hash + raw.charCodeAt(i)) | 0;
  const pair = palette[Math.abs(hash) % palette.length];
  return { label: raw || 'Model', color: pair[0], line: pair[1] };
}

function spotlightStyle(kind) {
  return artifactStyle(normalizeArtifactKind(kind || 'focus'));
}

function buildSpotlightPayload(startSample, endSample, label = 'AI Focus', kind = 'focus', channel = null) {
  const style = spotlightStyle(kind);
  const safeChannel = Number.isFinite(channel) ? clamp(Number(channel), 0, N_CH - 1) : null;
  return {
    startSample: Number(startSample),
    endSample: Number(endSample),
    label: String(label || 'AI Focus'),
    kind: normalizeArtifactKind(kind || 'focus'),
    channel: safeChannel,
    color: style.color,
    line: style.line,
    until: Number.POSITIVE_INFINITY,
  };
}

function buildAIHighlightRegions(result = S.ai.result, state = getReviewRenderState()) {
  if (!state?.filled) return [];
  const artifacts = Array.isArray(S.review.artifacts) ? S.review.artifacts : [];
  if (artifacts.length) {
    return artifacts.slice(0, 6).map(item => ({
      startSample: Number(item.startSample),
      endSample: Number(item.endSample),
      kind: normalizeArtifactKind(item.kind || 'artifact'),
      label: `AI ${item.label || artifactStyle(item.kind).label}`,
      channel: Number.isFinite(item.channel) ? Number(item.channel) : null,
      channelLabel: item.channelLabel || '',
      detail: item.detail || '',
      source: 'ai',
      until: Number.POSITIVE_INFINITY,
    }));
  }

  const viewport = getReviewViewport(state, canvas.width || canvas.clientWidth || 1);
  const start = Number(viewport.viewStart || 0);
  const end = Number(viewport.viewEnd || start);
  const count = Math.max(1, end - start + 1);
  const sampleRate = Math.max(Number(state.sampleRate || S.sampleRate || 250), 1);
  const win = Math.max(12, Math.min(count, Math.round(sampleRate * 0.22)));
  const hop = Math.max(4, Math.floor(win / 3));
  const topIssue = normalizeArtifactKind(result?.artifact_summary?.top_issue || result?.local_model_insights?.artifact_classifier?.label || 'focus');
  const cleanIssue = ['', 'clean', 'none', 'normal', 'stable', 'ok', 'clean_window'].includes(topIssue);
  const kind = cleanIssue ? 'focus' : topIssue;
  const candidates = [];
  const focusChannel = clamp(getReviewFocusChannelIndex(state), 0, N_CH - 1);
  const scanChannels = cleanIssue
    ? [focusChannel]
    : Array.from({ length: N_CH }, (_, i) => i).filter(isChannelVisible);

  for (const ch of scanChannels) {
    if (!isChannelVisible(ch)) continue;
    const buf = state.emg?.[ch];
    if (!buf) continue;
    for (let offset = 0; offset <= count - win; offset += hop) {
      let sum = 0;
      let peak = 0;
      for (let j = 0; j < win; j += 1) {
        const value = Math.abs(Number(buf[bufferIndexForAbsSample(state, start + offset + j)] || 0));
        sum += value;
        if (value > peak) peak = value;
      }
      candidates.push({
        channel: ch,
        startSample: start + offset,
        endSample: start + offset + win - 1,
        score: (sum / win) + peak * 0.35,
      });
    }
  }

  const usedChannels = new Set();
  return candidates
    .sort((a, b) => Number(b.score || 0) - Number(a.score || 0))
    .filter(item => {
      if (usedChannels.has(item.channel)) return false;
      usedChannels.add(item.channel);
      return true;
    })
    .slice(0, cleanIssue ? 1 : 4)
    .map(item => ({
      ...item,
      kind,
      label: cleanIssue ? 'AI Activity Focus' : `AI ${artifactStyle(kind).label}`,
      channelLabel: S.channelLabels[item.channel] || `CH${item.channel + 1}`,
      detail: cleanIssue
        ? `Strongest localized activity on ${S.channelLabels[item.channel] || `CH${item.channel + 1}`}.`
        : 'AI-localized region from the current scan.',
      source: 'ai',
      until: Number.POSITIVE_INFINITY,
    }));
}

function regionsOverlapReviewViewport(regions = [], state = getReviewRenderState()) {
  if (!Array.isArray(regions) || !regions.length || !state?.filled) return false;
  const viewport = getReviewViewport(state, canvas.width || canvas.clientWidth || 1);
  const viewStart = Number(viewport.viewStart || viewport.fullStart || 0);
  const viewEnd = Number(viewport.viewEnd || viewport.fullEnd || viewStart);
  return regions.some(item => (
    Number(item.endSample) >= viewStart
    && Number(item.startSample) <= viewEnd
  ));
}

function ensureAIHighlightRegions(state = getReviewRenderState()) {
  const current = Array.isArray(S.ai.regions) ? S.ai.regions : [];
  if (current.length && regionsOverlapReviewViewport(current, state)) return current;
  if (S.ai.loading || !S.ai.result) return current;
  const rebuilt = buildAIHighlightRegions(S.ai.result, state);
  if (!rebuilt.length) return current;
  S.ai.regions = rebuilt;
  const first = rebuilt[0];
  S.ai.spotlight = buildSpotlightPayload(first.startSample, first.endSample, first.label, first.kind, first.channel);
  return S.ai.regions;
}

function normalizeArtifactKind(kind) {
  const raw = String(kind || '').toLowerCase().trim();
  if (!raw) return 'artifact';
  if (raw.includes('clip')) return 'clip';
  if (raw.includes('hum')) return 'hum';
  if (raw.includes('drift')) return 'drift';
  if (raw.includes('flat')) return 'flatline';
  if (raw.includes('spike')) return 'spike';
  if (raw.includes('motion')) return 'motion_artifact';
  if (raw.includes('low')) return 'low_signal';
  return raw;
}

function artifactRadarSegments(state = getReviewRenderState(), result = S.ai.result) {
  if (!state?.filled) return [];
  const viewport = getReviewViewport(state, canvas.width || canvas.clientWidth || 1);
  const fullStart = Number(viewport.fullStart || 0);
  const fullEnd = Number(viewport.fullEnd || fullStart);
  const fullSpan = Math.max(1, fullEnd - fullStart + 1);
  const segments = [];
  const pushSegment = (segment) => {
    if (!segment) return;
    const kind = normalizeArtifactKind(segment.kind || segment.label || 'artifact');
    const startSample = clamp(Math.round(Number(segment.startSample ?? fullStart)), fullStart, fullEnd);
    const endSample = clamp(Math.round(Number(segment.endSample ?? startSample)), startSample, fullEnd);
    const confidence = clamp(Number(segment.confidence ?? segment.score ?? 0.72), 0, 0.99);
    const coverage = Math.max(0.014, (endSample - startSample + 1) / fullSpan);
    segments.push({
      kind,
      label: String(segment.label || artifactStyle(kind).label || 'Artifact'),
      detail: String(segment.detail || ''),
      channel: Number.isFinite(Number(segment.channel)) ? Number(segment.channel) : null,
      channelLabel: String(segment.channelLabel || ''),
      startSample,
      endSample,
      confidence,
      widthPct: coverage,
      leftPct: clamp((startSample - fullStart) / fullSpan, 0, 1),
      coverage,
      source: String(segment.source || 'localized'),
      localized: segment.localized !== false && coverage < 0.72,
    });
  };

  (S.review.artifacts || []).forEach(item => {
    pushSegment({
      kind: item.kind,
      label: item.label,
      detail: item.detail,
      startSample: item.startSample,
      endSample: item.endSample,
      channel: item.channel,
      channelLabel: item.channelLabel,
      confidence: item.score,
      source: 'localized',
      localized: true,
    });
  });

  const topIssue = String(result?.artifact_summary?.top_issue || '').trim();
  const artifactHead = result?.local_model_insights?.artifact_classifier || null;
  const headKind = normalizeArtifactKind(artifactHead?.label || topIssue || '');
  const diagnostics = S.diagnostics?.noise || {};
  const addGlobal = (kind, detail, confidence = 0.7) => {
    if (segments.some(item => item.kind === kind)) return;
    pushSegment({
      kind,
      label: artifactStyle(kind).label,
      detail,
      startSample: fullStart,
      endSample: fullEnd,
      confidence,
      source: 'global',
      localized: false,
    });
  };

  if (Number(diagnostics.clip_pct || 0) > 0.2 || topIssue === 'clipping' || headKind === 'clip') {
    addGlobal('clip', artifactHead?.detail || `Clip percentage is ${Number(diagnostics.clip_pct || 0).toFixed(2)} percent in the visible window.`, 0.9);
  }
  const humDb = Math.max(Number(diagnostics.hum_50_db || -Infinity), Number(diagnostics.hum_60_db || -Infinity));
  if (humDb >= -28 || topIssue === 'hum' || headKind === 'hum') {
    addGlobal('hum', artifactHead?.detail || `Line-noise energy is elevated at ${humDb.toFixed(1)} dB relative power.`, 0.86);
  }
  if (Number(diagnostics.drift_db || -Infinity) >= -26 || topIssue === 'baseline_drift' || headKind === 'drift') {
    addGlobal('drift', artifactHead?.detail || `Baseline drift energy is elevated at ${Number(diagnostics.drift_db || 0).toFixed(1)} dB relative power.`, 0.84);
  }
  if (headKind && !['clean', 'none', 'normal', 'stable', 'ok', 'clip', 'hum', 'drift'].includes(headKind)) {
    addGlobal(headKind, artifactHead?.detail || 'The built-in artifact head flagged this window for review.', Number(artifactHead?.confidence || 0.76));
  }

  return segments
    .sort((a, b) => a.leftPct - b.leftPct || b.widthPct - a.widthPct || String(a.kind).localeCompare(String(b.kind)))
    .slice(0, 12);
}

function spotlightArtifactRadarSegment(index) {
  const segments = artifactRadarSegments();
  const segment = segments[Number(index)];
  if (!segment) return;
  if (!S.review.paused) toggleReviewPause(true);
  S.review.selection = {
    startSample: Number(segment.startSample),
    endSample: Number(segment.endSample),
  };
  S.review.viewCenterSample = Math.round((Number(segment.startSample) + Number(segment.endSample)) / 2);
  S.review.lastStats = computeSelectionStats(S.review.selection, getReviewRenderState());
  spotlightReviewRange(segment.startSample, segment.endSample, `${segment.label} Focus`, segment.kind, segment.channel);
  if (Number.isFinite(segment.channel)) {
    S.review.lastStats = {
      ...(S.review.lastStats || computeSelectionStats(S.review.selection, getReviewRenderState()) || {}),
      focusChannel: Number(segment.channel),
      focusLabel: segment.channelLabel || S.channelLabels[segment.channel] || `CH${segment.channel + 1}`,
    };
  }
  syncReviewUI();
  syncWorkshopUI();
  syncAICopilotUI();
  toast(`Focused ${segment.label.toLowerCase()} region`);
}

window.spotlightArtifactRadarSegment = spotlightArtifactRadarSegment;

function syncArtifactRadar(state = getReviewRenderState(), result = S.ai.result) {
  const track = $('artifact-radar-track');
  const empty = $('artifact-radar-empty');
  const meta = $('artifact-radar-meta');
  if (!track) return;
  Array.from(track.querySelectorAll('.artifact-radar-segment')).forEach(node => node.remove());
  const segments = artifactRadarSegments(state, result);
  const viewport = getReviewViewport(state, canvas.width || canvas.clientWidth || 1);
  const fullSpan = Math.max(1, Number(viewport.fullEnd || 0) - Number(viewport.fullStart || 0) + 1);

  if (meta) {
    if (!state?.filled) {
      meta.textContent = 'No review span';
    } else if (!segments.length) {
      meta.textContent = formatAxisDuration((fullSpan / Math.max(Number(state.sampleRate || 250), 1)) * 1000);
    } else {
      meta.textContent = `${segments.length} region${segments.length === 1 ? '' : 's'} in ${formatAxisDuration((fullSpan / Math.max(Number(state.sampleRate || 250), 1)) * 1000)}`;
    }
  }
  if (empty) {
    empty.textContent = state?.filled
      ? (segments.length ? 'Click to focus.' : 'No issues in view.')
      : 'Scan or freeze to map issues.';
    empty.style.display = segments.length ? 'none' : 'flex';
  }

  const selected = getSelectionRange(S.review.selection);
  segments.forEach((segment, index) => {
    const style = artifactStyle(segment.kind);
    const node = document.createElement('button');
    node.type = 'button';
    node.className = `artifact-radar-segment ${segment.kind}`;
    const widthPct = Math.max(1.4, segment.widthPct * 100);
    node.style.left = `${Math.max(0, segment.leftPct * 100)}%`;
    node.style.width = `${Math.min(100, widthPct)}%`;
    node.style.opacity = String(clamp(0.58 + segment.confidence * 0.34, 0.58, 0.98));
    node.title = `${segment.label}${segment.channelLabel ? ` on ${segment.channelLabel}` : ''}: ${segment.detail || style.label}`;
    if (
      selected
      && Number(segment.startSample) <= Number(selected.end)
      && Number(segment.endSample) >= Number(selected.start)
    ) {
      node.classList.add('active');
    }
    node.addEventListener('click', () => spotlightArtifactRadarSegment(index));
    track.appendChild(node);
  });
}

function mergeReviewArtifactRegions(regions) {
  if (!Array.isArray(regions) || !regions.length) return [];
  const sorted = regions
    .map(item => ({
      ...item,
      startSample: Math.round(Number(item.startSample || 0)),
      endSample: Math.round(Number(item.endSample || item.startSample || 0)),
    }))
    .sort((a, b) => a.startSample - b.startSample || String(a.kind || '').localeCompare(String(b.kind || '')));
  const merged = [];
  sorted.forEach(item => {
    const prev = merged[merged.length - 1];
    if (
      prev
      && prev.kind === item.kind
      && String(prev.source || '') === String(item.source || '')
      && String(prev.label || '') === String(item.label || '')
      && Number(prev.channel ?? -1) === Number(item.channel ?? -1)
      && item.startSample <= prev.endSample + 4
    ) {
      prev.endSample = Math.max(prev.endSample, item.endSample);
      if (Number(item.score || 0) > Number(prev.score || 0)) {
        prev.score = item.score;
        prev.detail = item.detail;
      }
      return;
    }
    merged.push({ ...item });
  });
  return merged;
}

function detectReviewArtifacts(state) {
  if (!state?.filled) return [];
  const width = canvas.width || canvas.clientWidth || 1;
  const viewport = getReviewViewport(state, width);
  const start = Number(viewport.viewStart || 0);
  const end = Number(viewport.viewEnd || start);
  const count = Math.max(0, end - start + 1);
  if (!count) return [];

  const sampleRate = Math.max(Number(state.sampleRate || 250), 1);
  const fullScale = Math.max(Number(S.signalFullScale || 200), 1);
  const regions = [];
  const driftWindow = Math.max(8, Math.round(sampleRate * 0.25));
  const flatWindow = Math.max(8, Math.round(sampleRate * 0.20));

  const channels = Array.from({ length: N_CH }, (_, i) => i).filter(isChannelVisible);
  channels.forEach(channel => {
    const values = new Array(count);
    let maxAbs = 0;
    for (let i = 0; i < count; i += 1) {
      const absSample = start + i;
      const idx = bufferIndexForAbsSample(state, absSample);
      const value = Number(state.emg[channel]?.[idx] || 0);
      values[i] = value;
      if (Math.abs(value) > maxAbs) maxAbs = Math.abs(value);
    }

    let regionStart = null;
    for (let i = 0; i < count; i += 1) {
      const flagged = Math.abs(values[i]) >= fullScale * 0.92;
      if (flagged && regionStart === null) regionStart = i;
      if ((!flagged || i === count - 1) && regionStart !== null) {
        const endIndex = flagged && i === count - 1 ? i : i - 1;
        regions.push({
          kind: 'clip',
          channel,
          startSample: start + regionStart,
          endSample: start + endIndex,
          score: 1,
          detail: `Peak ${formatSignalValue(values.slice(regionStart, endIndex + 1).reduce((best, value) => Math.abs(value) > Math.abs(best) ? value : best, 0))}`,
        });
        regionStart = null;
      }
    }

    for (let i = 0; i <= count - driftWindow; i += Math.max(1, Math.floor(driftWindow / 3))) {
      let sum = 0;
      for (let j = 0; j < driftWindow; j += 1) sum += values[i + j];
      const mean = sum / driftWindow;
      if (Math.abs(mean) >= fullScale * 0.18) {
        regions.push({
          kind: 'drift',
          channel,
          startSample: start + i,
          endSample: start + i + driftWindow - 1,
          score: Math.abs(mean) / fullScale,
          detail: `Mean offset ${formatSignalValue(mean)}`,
        });
      }
    }

    for (let i = 0; i <= count - flatWindow; i += Math.max(1, Math.floor(flatWindow / 2))) {
      let min = Infinity;
      let max = -Infinity;
      for (let j = 0; j < flatWindow; j += 1) {
        const value = values[i + j];
        if (value < min) min = value;
        if (value > max) max = value;
      }
      const peakToPeak = max - min;
      if (peakToPeak <= fullScale * 0.015 && maxAbs >= fullScale * 0.02) {
        regions.push({
          kind: 'flatline',
          channel,
          startSample: start + i,
          endSample: start + i + flatWindow - 1,
          score: 1 - peakToPeak / Math.max(fullScale * 0.015, 1e-6),
          detail: `Peak-to-peak ${formatSignalValue(peakToPeak)}`,
        });
      }
    }

    const deltaThreshold = Math.max(fullScale * 0.42, Math.max(maxAbs, 1) * 0.75);
    for (let i = 1; i < count - 1; i += 1) {
      const leftStep = Number(values[i] || 0) - Number(values[i - 1] || 0);
      const rightStep = Number(values[i + 1] || 0) - Number(values[i] || 0);
      const transient = Math.abs(Number(values[i] || 0) - ((Number(values[i - 1] || 0) + Number(values[i + 1] || 0)) * 0.5));
      if (
        Math.abs(leftStep) >= deltaThreshold
        && Math.abs(rightStep) >= deltaThreshold * 0.55
        && Math.sign(leftStep) !== Math.sign(rightStep)
        && transient >= deltaThreshold * 0.45
      ) {
        regions.push({
          kind: 'spike',
          channel,
          startSample: start + Math.max(i - 1, 0),
          endSample: start + Math.min(i + 1, count - 1),
          score: transient / fullScale,
          detail: `Transient ${formatSignalValue(transient)}`,
        });
      }
    }
  });

  return mergeReviewArtifactRegions(regions)
    .sort((a, b) => Number(b.score || 0) - Number(a.score || 0))
    .slice(0, 18)
    .map(item => ({
      ...item,
      channelLabel: S.channelLabels[item.channel] || `CH${Number(item.channel || 0) + 1}`,
      label: artifactStyle(item.kind).label,
    }));
}

function refreshReviewArtifacts(state = getReviewRenderState()) {
  const summary = $('review-artifact-summary');
  const list = $('review-artifact-list');
  const artifacts = detectReviewArtifacts(state);
  S.review.artifacts = artifacts;

  if (summary) {
    if (!state?.filled) {
      summary.textContent = 'No signal window is available yet.';
    } else if (!artifacts.length) {
      summary.textContent = `No obvious artifacts flagged in the current ${state.paused ? 'frozen' : 'live'} window.`;
    } else {
      const channelCount = new Set(artifacts.map(item => Number(item.channel))).size;
      summary.textContent = `${artifacts.length} candidate issue${artifacts.length === 1 ? '' : 's'} across ${channelCount} channel${channelCount === 1 ? '' : 's'}. Focus or mark them from here.`;
    }
  }

  if (list) {
    list.innerHTML = '';
    if (!artifacts.length) {
    list.innerHTML = '<div class="setup-copy">No issues in view.</div>';
    } else {
      artifacts.forEach((item, idx) => {
        const row = document.createElement('div');
        row.className = 'review-marker-item';
        const startMs = reviewSampleOffsetMs(item.startSample, state);
        const endMs = reviewSampleOffsetMs(item.endSample, state);
        row.innerHTML = `
          <div class="review-marker-head">
            <span class="review-marker-name">${item.label}</span>
            <span class="review-marker-range">${startMs.toFixed(1)} to ${endMs.toFixed(1)} ms</span>
          </div>
          <div class="review-marker-note">${item.detail} | ${item.channelLabel}</div>
          <div class="session-import-actions review-marker-actions" style="margin-top:6px">
            <button class="btn" type="button" onclick="focusReviewArtifact(${idx})">Focus</button>
            <button class="btn" type="button" onclick="markReviewArtifact(${idx})">Mark</button>
          </div>
        `;
        list.appendChild(row);
      });
    }
  }

  syncArtifactRadar(state);
}

function hoveredReviewRegion() {
  const hoverSample = Number.isFinite(S.review.hoverSample) ? Number(S.review.hoverSample) : null;
  if (hoverSample === null) return null;
  const hoverChannel = Number.isFinite(S.review.hoverChannel) ? Number(S.review.hoverChannel) : null;
  const regions = [
    ...(Array.isArray(S.review.artifacts) ? S.review.artifacts : []),
    ...(Array.isArray(S.ai.regions) ? S.ai.regions : []),
    ...(Array.isArray(S.promptModel?.regions) ? S.promptModel.regions : []),
  ];
  const direct = regions.find(item =>
    hoverSample >= Number(item.startSample)
    && hoverSample <= Number(item.endSample)
    && (hoverChannel === null || !Number.isFinite(item.channel) || Number(item.channel) === hoverChannel)
  );
  if (direct) return direct;
  let nearest = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  const sampleRate = Math.max(Number(S.sampleRate || 250), 1);
  const hoverToleranceSamples = Math.max(12, Math.round(sampleRate * 0.25));
  regions.forEach(item => {
    if (hoverChannel !== null && Number.isFinite(item.channel) && Number(item.channel) !== hoverChannel) return;
    const center = Math.round((Number(item.startSample) + Number(item.endSample)) / 2);
    const distance = Math.abs(center - hoverSample);
    if (distance < bestDistance) {
      bestDistance = distance;
      nearest = item;
    }
  });
  if (!nearest) return null;
  if (nearest.source === 'ai') return nearest;
  return bestDistance <= hoverToleranceSamples ? nearest : null;
}

function getReviewFocusChannelIndex(state) {
  if (S.review.lastStats && Number.isFinite(S.review.lastStats.focusChannel)) {
    return Number(S.review.lastStats.focusChannel);
  }
  const visible = Array.from({ length: N_CH }, (_, i) => i).filter(i => isChannelVisible(i));
  return visible.length ? visible[0] : 0;
}

function resolveReviewCursorSample(cursor, state) {
  if (!cursor || !state?.filled) return null;
  const viewport = getReviewViewport(state, canvas.width || 1);
  if (!state.paused && Number.isFinite(cursor.ratio)) {
    return clamp(
      Math.round(Number(viewport.viewStart || 0) + Number(cursor.ratio) * Math.max(Number(viewport.viewSamples || 1) - 1, 0)),
      Number(viewport.fullStart || 0),
      Number(viewport.fullEnd || 0),
    );
  }
  if (Number.isFinite(cursor.sample)) {
    return clamp(
      Number(cursor.sample),
      Number(viewport.fullStart || 0),
      Number(viewport.fullEnd || 0),
    );
  }
  return null;
}

function setReviewCursor(cursorId, sample, state, width) {
  const viewport = getReviewViewport(state, width || canvas.width || 1);
  const clamped = clamp(
    Number(sample || 0),
    Number(viewport.fullStart || 0),
    Number(viewport.fullEnd || 0),
  );
  const ratio = viewport.viewSamples > 1
    ? clamp((clamped - Number(viewport.viewStart || 0)) / Math.max(Number(viewport.viewSamples || 1) - 1, 1), 0, 1)
    : 0;
  S.review.cursors[cursorId] = { sample: clamped, ratio };
}

function hitTestReviewCursor(x, state, width) {
  const thresholdPx = 10;
  return ['a', 'b'].find(id => {
    const sample = resolveReviewCursorSample(S.review.cursors[id], state);
    if (!Number.isFinite(sample)) return false;
    const cursorX = canvasXFromSample(sample, state, width);
    return Math.abs(cursorX - x) <= thresholdPx;
  }) || null;
}

function getReviewCursorMetrics(state) {
  if (!state?.filled) return null;
  const focusChannel = getReviewFocusChannelIndex(state);
  const focusLabel = S.channelLabels[focusChannel] || `CH${focusChannel + 1}`;
  const metrics = { focusChannel, focusLabel, cursors: {} };

  ['a', 'b'].forEach(id => {
    const sample = resolveReviewCursorSample(S.review.cursors[id], state);
    if (!Number.isFinite(sample)) return;
    const point = getReviewSampleValue(state, sample, focusChannel);
    if (!point) return;
    metrics.cursors[id] = {
      sample: point.sample,
      value: point.value,
      timeLabel: point.timeLabel,
      valueLabel: point.valueLabel,
    };
  });

  if (metrics.cursors.a && metrics.cursors.b) {
    const deltaSamples = Number(metrics.cursors.b.sample) - Number(metrics.cursors.a.sample);
    const deltaMs = (deltaSamples / Math.max(Number(state.sampleRate || 250), 1)) * 1000;
    const deltaValue = Number(metrics.cursors.b.value) - Number(metrics.cursors.a.value);
    metrics.delta = {
      samples: deltaSamples,
      timeLabel: formatAxisDuration(Math.abs(deltaMs)),
      valueLabel: formatSignalValue(deltaValue),
      sign: deltaMs >= 0 ? '' : '-',
    };
  }
  return metrics;
}

function stepZoomLevel(levels, current, dir) {
  const values = Array.isArray(levels) && levels.length ? levels : [1];
  const currentValue = Number(current || values[0]);
  let bestIndex = 0;
  let bestDistance = Infinity;
  values.forEach((value, idx) => {
    const distance = Math.abs(Number(value) - currentValue);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = idx;
    }
  });
  const nextIndex = clamp(bestIndex + dir, 0, values.length - 1);
  return Number(values[nextIndex]);
}

function adjustReviewZoom(axis, dir) {
  const state = getReviewRenderState();
  if (!state.filled) return;
  S.review.viewCenterSample = resolveReviewAnchorSample(state);
  if (axis === 'x') {
    S.review.zoomX = stepZoomLevel(REVIEW_X_ZOOM_LEVELS, S.review.zoomX, dir);
  } else {
    S.review.liveAutoScale = false;
    S.review.zoomY = stepZoomLevel(REVIEW_Y_ZOOM_LEVELS, S.review.zoomY, dir);
  }
  syncReviewUI();
}

function resetReviewZoom() {
  const state = getReviewRenderState();
  if (!state.filled) return;
  S.review.zoomX = 1;
  S.review.zoomY = 1;
  S.review.liveAutoScale = false;
  S.review.lastAutoScaleAt = 0;
  S.review.viewCenterSample = resolveReviewAnchorSample(state);
  syncReviewUI();
}


function normalizeChannelMask(mask) {
  const next = Array.isArray(mask) ? mask.slice(0, N_CH).map(v => v !== false) : [];
  while (next.length < N_CH) next.push(true);
  if (!next.some(Boolean)) return new Array(N_CH).fill(true);
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

function setChannelMask(mask, options = {}) {
  const persist = options.persist !== false;
  S.channelEnabled = normalizeChannelMask(mask);
  if (persist) saveChannelMask();
  buildLegend();
  buildRmsBars();
  buildQualityGrid();
  syncInspectorTelemetry();
  if (S.review.selection) {
    S.review.lastStats = computeSelectionStats(S.review.selection, getReviewRenderState());
  }
  syncReviewUI();
}

function setAllChannelVisibility(value) {
  setChannelMask(new Array(N_CH).fill(!!value));
}

function toggleChannelVisibility(index) {
  S.manualChannelOverrideUntil = performance.now() + 30000;
  const next = S.channelEnabled.slice();
  next[index] = !next[index];
  setChannelMask(next);
}


// =============================================================================
// BOOT — kicks everything off
// =============================================================================

async function boot() {
  // theme first so the page doesn't flash the wrong mode
  applyTheme(currentTheme);
  $('theme-select').onchange = e => applyTheme(e.target.value);
  loadSignalLogicConfig();

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
  syncSignalLogicUI();
  bindButtons();
  bindReviewCanvas();
  bindAILens();
  bindCommandPalette();
  connectWS();
  resizeCanvas();
  window.onresize = resizeCanvas;
  renderLoop();
  await loadSessions();
  await loadSubjects();
  await loadDatasets();
  await loadExperiments();

  switchWorkspace(S.dashboardWorkspace);
  if (typeof window.switchTab === 'function') window.switchTab('code');
  switchViz(S.activeViz);
  syncReviewUI();
  syncScopeCopilotUI();
  syncPipelineBuilderUI();

  // model selector
  const modelSel = $('model-select');
  if (modelSel) {
    modelSel.onchange = () => {
      if (window._arm3d) window._arm3d.setModel(modelSel.value);
    };
  }

  // update performance stats once per second
  setInterval(updatePerformance, 1000);
  setInterval(maybeAutoRunAICopilot, 1500);
  setInterval(evaluateSignalLogicRuntime, 250);
  setInterval(refreshPipelineAcquisitionRun, 1000);
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
    S.promptModel.server = s.prompt_model || null;
    S.recSession = s.is_recording;
    applyLSLStatus(s.lsl || {});
    applyOSCStatus(s.osc || {});
    applySignalProfile(s.signal_profile || {});
    S.review.liveAutoScale = false;
    if (S.streamSource === 'playback') S.review.zoomY = 1;
    S.review.lastAutoScaleAt = 0;
    applyProtocolTemplates(s.protocol_templates || []);
    applyEEGExperiments(s.eeg_experiments || []);
    applyCalibrationState(s.calibration || {});
    applyEEGBrainView(s.eeg_brain || {});
    applyDiagnostics(s.diagnostics || {});
    applySafety(s.safety || {});
    applyFilterLabStatus(s.filter_lab || {});
    applyWorkshopStatus(s.workshop || {});
    applyFirmwareStatus(s.firmware || {});
    applyAIStatus(s.ai || {});
    S.lastPrediction = clonePredictionPayload(s.last_prediction);
    await refreshLieDetectorStatus();
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
    syncFirmwareUI();
    syncAICopilotUI();
    syncSignalLogicUI();
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
    S.controlConfidenceThreshold = Number(c.confidence_threshold ?? S.controlConfidenceThreshold ?? 0.55);
    S.promptModel.server = c.prompt_model || S.promptModel.server || null;
    applySignalProfile(c.signal_profile || {});
    S.review.liveAutoScale = false;
    if (S.streamSource === 'playback') S.review.zoomY = 1;
    S.review.lastAutoScaleAt = 0;
    applyProtocolTemplates(c.protocol_templates || []);
    applyEEGExperiments(c.eeg_experiments || []);
    applyCalibrationState({ protocol: c.calibration_protocol || null });
    applyEEGBrainView(c.eeg_brain || {});
    applyOSCStatus(c.osc || S.osc);
    applyDiagnostics(c.diagnostics || {});
    applySafety(c.safety || {});
    applyFilterLabStatus(c.filter_lab || {});
    applyWorkshopStatus(c.workshop || {});
    applyFirmwareStatus(c.firmware || {});
    applyAIStatus(c.ai || {});
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
    syncFirmwareUI();
    syncAICopilotUI();
    syncSignalLogicUI();
    syncControlSensitivityUI();
  } catch { /* config fetch can fail during startup */ }
}

function syncControlSensitivityUI() {
  const slider = $('control-sensitivity-slider');
  const value = $('control-sensitivity-value');
  const pct = Math.round(Math.max(0.3, Math.min(0.95, Number(S.controlConfidenceThreshold || 0.55))) * 100);
  if (slider && document.activeElement !== slider) slider.value = String(pct);
  if (value) value.textContent = `${pct}%`;
}

async function applyControlSensitivity(threshold, quiet = false) {
  const clamped = Math.max(0.30, Math.min(0.95, Number(threshold || 0.55)));
  S.controlConfidenceThreshold = clamped;
  syncControlSensitivityUI();
  try {
    const result = await post('/api/control/sensitivity', { confidence_threshold: clamped });
    S.controlConfidenceThreshold = Number(result.confidence_threshold ?? clamped);
    syncControlSensitivityUI();
    if (!quiet) toast(`Control trigger set to ${Math.round(S.controlConfidenceThreshold * 100)}%`, 'green');
  } catch(e) {
    if (!quiet) toast(`Could not update sensitivity: ${e.message || e}`, 'red');
  }
}

function queueControlSensitivityFromSlider(value) {
  const threshold = Math.max(30, Math.min(95, Number(value || 55))) / 100;
  S.controlConfidenceThreshold = threshold;
  syncControlSensitivityUI();
  clearTimeout(S.controlSensitivityTimer);
  S.controlSensitivityTimer = setTimeout(() => applyControlSensitivity(threshold, true), 180);
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
  S.workshop.lastSaved = null;
  S.workshop.saving = false;
  S.workshop.exporting = false;
  loadChannelMask();
  resetReviewState({ clearMarkers: true });
  syncResearchUI();
  primeFilterDefaultsFromProfile();
  syncFilterChainLabel();
  populateSyntheticScenarioOptions(S.signalProfileKey);
  syncAICopilotUI();
  syncSignalLogicUI();
}

function applyProtocolTemplates(templates) {
  S.protocolTemplates = Array.isArray(templates) ? templates : [];
  const select = $('protocol-template');
  const current = select?.value || '';
  const valid = S.protocolTemplates.some(template => template.key === current);
  if (valid) return;
  resetProtocolRun();
  S.protocolRunner.phase = 'idle';
  S.protocolRunner.step = null;
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
  })[key] || 'Workshop';
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

function applyFirmwareStatus(payload) {
  if (!payload) return;
  S.firmware.root_dir = String(payload.root_dir || S.firmware.root_dir || '');
  S.firmware.generated_dir = String(payload.generated_dir || S.firmware.generated_dir || '');
  S.firmware.arduino_cli = {
    ...(S.firmware.arduino_cli || {}),
    ...((payload && payload.arduino_cli) || {}),
  };
  if (Array.isArray(payload.files)) {
    S.firmware.files = payload.files.slice();
    if (S.firmware.selectedPath && !S.firmware.files.some(item => item.path === S.firmware.selectedPath)) {
      S.firmware.selectedPath = '';
      S.firmware.selectedKind = '';
      S.firmware.content = '';
      S.firmware.dirty = false;
    }
  }
}

function applyAIStatus(payload) {
  if (!payload) return;
  S.ai.available = payload.available !== false;
  S.ai.configured = !!payload.configured;
  S.ai.mode = String(payload.mode || (S.ai.configured ? 'remote' : 'heuristic'));
  S.ai.provider = String(payload.provider || (S.ai.configured ? 'api' : 'local'));
  S.ai.model = String(payload.model || S.ai.model || '');
  S.ai.base_url = String(payload.base_url || S.ai.base_url || '');
  S.ai.backend_label = String(payload.backend_label || S.ai.backend_label || '');
  S.ai.credential_source = String(payload.credential_source || 'none');
  S.ai.config_source = String(payload.config_source || 'none');
  S.ai.config_saved = !!payload.config_saved;
  S.ai.key_hint = String(payload.key_hint || '');
  S.ai.storage_path = String(payload.storage_path || '');
  S.ai.last_error = String(payload.last_error || '');
  S.ai.localModels = payload.local_models ? structuredClone(payload.local_models) : S.ai.localModels;
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
  const matlabBtn = $('btn-workshop-matlab');
  const saveBtn = $('btn-workshop-save');
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
  if (matlabBtn) {
    matlabBtn.disabled = S.workshop.loading || S.workshop.saving || S.workshop.exporting || !S.workshop.lastRequest;
    matlabBtn.textContent = S.workshop.exporting ? 'Exporting...' : 'Export MATLAB';
  }
  if (saveBtn) {
    saveBtn.disabled = S.workshop.loading || S.workshop.saving || S.workshop.exporting || !S.workshop.lastRequest;
    saveBtn.textContent = S.workshop.saving ? 'Saving...' : 'Save Selection';
  }
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
    empty.textContent = 'Run analysis to fill bands.';
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
    empty.textContent = 'No analysis yet.';
      details.appendChild(empty);
    } else {
      const rows = [
        ['Sample Rate', `${Number(S.workshop.result?.sample_rate || 0).toFixed(0)} Hz`],
        ['Selection Label', S.workshop.result?.selection_label || 'selected_chunk'],
        ['Min / Max', `${Number(summary.min || 0).toFixed(5)} / ${Number(summary.max || 0).toFixed(5)} ${S.signalUnits}`],
        ['Std Dev', `${Number(summary.std || 0).toFixed(5)} ${S.signalUnits}`],
        ['Area', `${Number(summary.area || 0).toFixed(6)}`],
      ];
      if (S.workshop.lastSaved?.filename) rows.push(['Saved File', S.workshop.lastSaved.filename]);
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
  status.textContent = S.workshop.last_error || 'Workshop is unavailable on this runtime.';
    } else if (S.workshop.loading) {
      status.textContent = 'Analyzing the selected chunk on the server.';
    } else if (S.workshop.saving) {
      status.textContent = 'Saving the selected chunk info to the session workspace.';
    } else if (S.workshop.result) {
      const savedNote = S.workshop.lastSaved?.filename ? ` Saved as ${S.workshop.lastSaved.filename}.` : '';
      status.textContent = `Ready: ${workshopViewTitle(S.workshop.view)} for ${S.workshop.result.channel_labels?.length || 0} channel(s).${savedNote}`;
    } else if (S.review.snapshot) {
      status.textContent = 'Paused review chunk is ready. Analyze it here or send a new range from the waveform.';
    } else {
    status.textContent = 'Awaiting chunk.';
    }
  }

  renderWorkshopView();
}

function firmwareLogLine(text) {
  const line = `[${new Date().toLocaleTimeString()}] ${text}`;
  S.firmware.actionOutput = S.firmware.actionOutput
    ? `${S.firmware.actionOutput}\n${line}`
    : line;
}

function getSelectedFirmwareFile() {
  return (S.firmware.files || []).find(item => item.path === S.firmware.selectedPath) || null;
}

async function loadFirmwareFiles(preferredPath = '') {
  try {
    const payload = await get('/api/firmware/files');
    applyFirmwareStatus(payload || {});
    const fallbackPath = preferredPath || S.firmware.selectedPath || S.firmware.files[0]?.path || '';
    if (fallbackPath) {
      await openFirmwareFile(fallbackPath, { silent: true });
    } else {
      syncFirmwareUI();
    }
  } catch (e) {
    firmwareLogLine(`Firmware list failed: ${e.message}`);
    syncFirmwareUI();
    toast(e.message, 'red');
  }
}

async function openFirmwareFile(path, { silent = false } = {}) {
  const nextPath = String(path || '').trim();
  if (!nextPath) return;
  if (S.firmware.dirty && S.firmware.selectedPath && S.firmware.selectedPath !== nextPath) {
    const keepGoing = window.confirm('Discard unsaved firmware edits?');
    if (!keepGoing) return;
  }
  try {
    S.firmware.loading = true;
    syncFirmwareUI();
    const payload = await post('/api/firmware/file/read', { path: nextPath });
    S.firmware.selectedPath = String(payload.path || nextPath);
    S.firmware.selectedKind = String(payload.kind || '');
    S.firmware.content = String(payload.content || '');
    S.firmware.dirty = false;
    syncFirmwareUI();
    if (!silent) toast(`Opened ${S.firmware.selectedPath}`);
  } catch (e) {
    toast(e.message, 'red');
  } finally {
    S.firmware.loading = false;
    syncFirmwareUI();
  }
}

async function saveFirmwareEditor() {
  if (!S.firmware.selectedPath) {
    toast('Choose a firmware file first', 'yellow');
    return;
  }
  try {
    S.firmware.saving = true;
    syncFirmwareUI();
    const payload = await post('/api/firmware/file/save', {
      path: S.firmware.selectedPath,
      content: S.firmware.content,
    });
    applyFirmwareStatus(payload || {});
    S.firmware.dirty = false;
    firmwareLogLine(`Saved ${S.firmware.selectedPath}`);
    syncFirmwareUI();
    toast(`Saved ${S.firmware.selectedPath}`);
  } catch (e) {
    toast(e.message, 'red');
  } finally {
    S.firmware.saving = false;
    syncFirmwareUI();
  }
}

async function compileFirmwareSketch() {
  const file = getSelectedFirmwareFile();
  if (!file || file.kind !== 'sketch') {
    toast('Select an .ino sketch to compile', 'yellow');
    return;
  }
  try {
    const payload = await post('/api/firmware/compile', {
      path: file.path,
      fqbn: S.firmware.fqbn,
    });
    S.firmware.lastCompile = payload;
    firmwareLogLine(payload.ok ? `Compile OK: ${file.path}` : `Compile failed: ${file.path}`);
    if (payload.stdout) firmwareLogLine(payload.stdout.trim());
    if (payload.stderr) firmwareLogLine(payload.stderr.trim());
    syncFirmwareUI();
    toast(payload.ok ? 'Firmware compile finished' : 'Firmware compile failed', payload.ok ? undefined : 'red');
  } catch (e) {
    firmwareLogLine(`Compile failed: ${e.message}`);
    syncFirmwareUI();
    toast(e.message, 'red');
  }
}

async function uploadFirmwareSketch() {
  const file = getSelectedFirmwareFile();
  if (!file || file.kind !== 'sketch') {
    toast('Select an .ino sketch to upload', 'yellow');
    return;
  }
  if (!S.firmware.port.trim()) {
    toast('Enter a serial port first', 'yellow');
    return;
  }
  try {
    const payload = await post('/api/firmware/upload', {
      path: file.path,
      fqbn: S.firmware.fqbn,
      port: S.firmware.port,
    });
    firmwareLogLine(payload.ok ? `Upload OK: ${file.path} -> ${S.firmware.port}` : `Upload failed: ${file.path}`);
    if (payload.stdout) firmwareLogLine(payload.stdout.trim());
    if (payload.stderr) firmwareLogLine(payload.stderr.trim());
    syncFirmwareUI();
    toast(payload.ok ? 'Firmware upload finished' : 'Firmware upload failed', payload.ok ? undefined : 'red');
  } catch (e) {
    firmwareLogLine(`Upload failed: ${e.message}`);
    syncFirmwareUI();
    toast(e.message, 'red');
  }
}

async function sendCodeToFirmwareLab() {
  if (!S.exportMeta?.sendToFirmware) {
    toast('Only generated Arduino sketches can be sent to Firmware', 'yellow');
    return;
  }
  const content = $('code-output')?.value || '';
  if (!content.trim()) {
    toast('Nothing to send to Firmware', 'yellow');
    return;
  }
  const filename = S.exportMeta?.filename || `${(S.exportMeta?.name || 'generated_sketch').replace(/[^A-Za-z0-9_.-]+/g, '_')}.ino`;
  try {
    const payload = await post('/api/firmware/generated', {
      name: S.exportMeta?.name || 'generated_sketch',
      filename,
      content,
      target: 'arduino',
    });
    applyFirmwareStatus(payload || {});
    const path = payload.saved?.path || '';
    firmwareLogLine(`Generated sketch saved: ${path || filename}`);
    if (path) {
      await openFirmwareFile(path, { silent: true });
    }
    window.switchTab('firmware');
    toast(`Sent ${filename} to Firmware`);
  } catch (e) {
    toast(e.message, 'red');
  }
}

function syncFirmwareUI() {
  const status = $('firmware-status');
  const pathLabel = $('firmware-file-path');
  const editor = $('firmware-editor');
  const list = $('firmware-file-list');
  const saveBtn = $('btn-firmware-save');
  const compileBtn = $('btn-firmware-compile');
  const uploadBtn = $('btn-firmware-upload');
  const fqbnInput = $('firmware-fqbn');
  const portInput = $('firmware-port');
  const output = $('firmware-action-log');

  if (fqbnInput && fqbnInput.value !== S.firmware.fqbn) fqbnInput.value = S.firmware.fqbn;
  if (portInput && portInput.value !== S.firmware.port) portInput.value = S.firmware.port;
  if (pathLabel) pathLabel.textContent = S.firmware.selectedPath || 'No file selected';
  if (editor && editor.value !== S.firmware.content) editor.value = S.firmware.content || '';
  if (editor) editor.disabled = S.firmware.loading;

  const selected = getSelectedFirmwareFile();
  if (saveBtn) {
    saveBtn.disabled = !S.firmware.selectedPath || !S.firmware.dirty || S.firmware.saving;
    saveBtn.textContent = S.firmware.saving ? 'Saving...' : 'Save File';
  }
  if (compileBtn) compileBtn.disabled = !selected || selected.kind !== 'sketch' || !S.firmware.arduino_cli?.available || S.firmware.dirty;
  if (uploadBtn) uploadBtn.disabled = !selected || selected.kind !== 'sketch' || !S.firmware.arduino_cli?.available || !S.firmware.port.trim() || S.firmware.dirty;

  if (status) {
    if (!S.firmware.arduino_cli?.available) {
      status.textContent = S.firmware.arduino_cli?.note || 'arduino-cli unavailable. Edit and export still work.';
    } else if (selected) {
      status.textContent = `Ready on ${selected.path}. Compile with ${S.firmware.fqbn}. Upload uses ${S.firmware.port || 'no port set'}.`;
    } else {
      status.textContent = S.firmware.arduino_cli?.note || 'Firmware ready.';
    }
  }

  if (list) {
    list.innerHTML = '';
    if (!S.firmware.files.length) {
      list.innerHTML = '<div class="setup-copy">No firmware files found.</div>';
    } else {
      S.firmware.files.forEach(item => {
        const row = document.createElement('button');
        row.type = 'button';
        row.className = `firmware-file-item${item.path === S.firmware.selectedPath ? ' active' : ''}`;
        row.innerHTML = `
          <div class="firmware-file-name">${item.name}</div>
          <div class="firmware-file-meta">${item.dir || 'root'} | ${item.kind}${item.generated ? ' | generated' : ''}</div>
        `;
        row.onclick = () => openFirmwareFile(item.path);
        list.appendChild(row);
      });
    }
  }

  if (output) output.value = S.firmware.actionOutput || '';
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
  return THEME_CANVAS[currentTheme] || THEME_CANVAS.neutral;
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
    toast(S.workshop.last_error || 'Workshop is unavailable.', 'red');
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
  S.workshop.lastSaved = null;
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

async function saveWorkshopSelection() {
  if (!S.workshop.lastRequest) {
    toast('Analyze a selection first so there is workshop data to save.', 'yellow');
    return;
  }

  S.workshop.saving = true;
  syncWorkshopUI();

  try {
    const out = await post('/api/workshop/save', {
      request: S.workshop.lastRequest,
      analysis: S.workshop.result || {},
      selection_meta: {
        ...(S.workshop.selectionMeta || {}),
        view: S.workshop.view || 'fft',
        signal_units: S.signalUnits,
      },
    });
    S.workshop.lastSaved = out?.saved || null;
    const filename = S.workshop.lastSaved?.filename || 'selection.json';
    toast(`Workshop selection saved: ${filename}`);
  } catch (e) {
    toast(e.message || 'Workshop save failed.', 'red');
  } finally {
    S.workshop.saving = false;
    syncWorkshopUI();
  }
}

async function exportWorkshopMatlab() {
  if (!S.workshop.lastRequest) {
    toast('Analyze a selection first so there is workshop data to export.', 'yellow');
    return;
  }

  S.workshop.exporting = true;
  syncWorkshopUI();

  try {
    const out = await post('/api/workshop/export/matlab', {
      request: S.workshop.lastRequest,
      analysis: S.workshop.result || {},
      selection_meta: {
        ...(S.workshop.selectionMeta || {}),
        view: S.workshop.view || 'fft',
        signal_units: S.signalUnits,
      },
    });
    const exported = out?.exported || null;
    const filename = exported?.filename || 'selection.mat';
    toast(`MATLAB export saved: ${filename}`);
  } catch (e) {
    toast(e.message || 'MATLAB export failed.', 'red');
  } finally {
    S.workshop.exporting = false;
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
      opt.textContent = 'No subjects';
      select.appendChild(opt);
    } else {
      const blank = document.createElement('option');
      blank.value = '';
      blank.textContent = 'Select';
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
      status.textContent = bits.join(' | ') || 'Subject loaded.';
    }
  } else {
    if (currentSubjectId && !findSubjectRecord(currentSubjectId)) {
      populateSubjectRegistryForm({ subject_id: currentSubjectId });
      if (status) status.textContent = 'Unsaved subject ID.';
    } else if (!currentSubjectId && !select?.value) {
      populateSubjectRegistryForm(null);
      if (status) status.textContent = 'Subjects support LOSO and tags.';
    }
  }
}

function syncCalibrationUI() {
  const summary = $('calibration-summary');
  if (!summary) return;

  const protocol = S.calibrationProtocol;
  if (!protocol) {
    summary.textContent = 'Loading calibration.';
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

const SYNTHETIC_SCENARIOS = {
  emg: [
    { value: 'clean', label: 'Clean Window', note: 'Low-noise EMG baseline for a normal usable-window scan.' },
    { value: 'contraction', label: 'Strong Contraction', note: 'Higher-amplitude activation bursts for training-quality EMG chunks.' },
    { value: 'fatigue', label: 'Fatigue Trend', note: 'Activation gradually softens so the AI can talk about weaker EMG readiness.' },
    { value: 'line_noise', label: 'Line Noise', note: 'Injects 50/60 Hz mains contamination on purpose.' },
    { value: 'drift', label: 'Baseline Drift', note: 'Adds slow baseline movement so the AI should call out drift.' },
    { value: 'motion', label: 'Motion Artifact', note: 'Adds spikes and step changes to imitate cable bumps or movement.' },
    { value: 'clipping', label: 'Clipping', note: 'Overdrives the waveform so clipping and bad export readiness are obvious.' },
  ],
  eeg: [
    { value: 'clean', label: 'Clean Window', note: 'Balanced rhythmic EEG without strong artifact cues.' },
    { value: 'alpha_focus', label: 'Alpha Focus', note: 'Boosts alpha-like rhythmic activity for a cleaner EEG review case.' },
    { value: 'line_noise', label: 'Line Noise', note: 'Injects mains contamination for artifact testing.' },
    { value: 'drift', label: 'Slow Drift', note: 'Adds low-frequency baseline wander.' },
    { value: 'blink_artifact', label: 'Blink Artifact', note: 'Adds frontal blink-like pulses the AI should flag.' },
    { value: 'spike_burst', label: 'Spike Burst', note: 'Adds sharp burst events for event-heavy review testing.' },
  ],
  ecg: [
    { value: 'clean', label: 'Sinus Rhythm', note: 'Stable synthetic ECG for a clean QA pass.' },
    { value: 'tachycardia', label: 'Faster Rhythm', note: 'Raises the heart-rate pattern so the AI sees a different rhythm regime.' },
    { value: 'baseline_wander', label: 'Baseline Wander', note: 'Adds strong slow drift like loose-contact or breathing wander.' },
    { value: 'motion', label: 'Motion Artifact', note: 'Adds transient movement artifacts on top of the beats.' },
    { value: 'clipping', label: 'Clipping', note: 'Overdrives the trace to trigger clipping-style warnings.' },
  ],
  eog: [
    { value: 'clean', label: 'Clean Window', note: 'Mild eye-motion activity without a dominant artifact burst.' },
    { value: 'saccades', label: 'Saccades', note: 'Stronger directional eye-movement pattern.' },
    { value: 'blink_burst', label: 'Blink Burst', note: 'Frequent blink-like pulses for artifact and marker testing.' },
    { value: 'drift', label: 'Drift', note: 'Slow baseline wander for review testing.' },
    { value: 'motion', label: 'Motion Artifact', note: 'Adds harsher transient movement noise.' },
  ],
  eda: [
    { value: 'clean', label: 'Clean Window', note: 'Low-noise tonic/phasic activity.' },
    { value: 'phasic_bursts', label: 'Phasic Bursts', note: 'Adds stronger event-like conductance responses.' },
    { value: 'drift', label: 'Drift', note: 'Adds a slow baseline trend.' },
    { value: 'noisy', label: 'Noisy', note: 'Raises random noise for QA testing.' },
  ],
  ppg: [
    { value: 'clean', label: 'Clean Window', note: 'Stable pulse waveform for a clean scan.' },
    { value: 'tachycardia', label: 'Faster Pulse', note: 'Speeds up the pulse pattern.' },
    { value: 'motion', label: 'Motion Artifact', note: 'Adds pulse-corrupting motion transients.' },
    { value: 'clipping', label: 'Clipping', note: 'Overdrives the pulse waveform.' },
  ],
  resp: [
    { value: 'clean', label: 'Clean Window', note: 'Steady respiration pattern.' },
    { value: 'deep_breathing', label: 'Deep Breathing', note: 'Slower, larger breathing cycles.' },
    { value: 'apnea_like', label: 'Apnea-like Gating', note: 'Drops sections of the cycle to create obvious interruptions.' },
    { value: 'drift', label: 'Drift', note: 'Adds a slow baseline shift.' },
  ],
  temp: [
    { value: 'clean', label: 'Clean Window', note: 'Stable temperature drift.' },
    { value: 'slow_rise', label: 'Slow Rise', note: 'Adds a gradual upward trend.' },
    { value: 'drift', label: 'Drift', note: 'Adds stronger low-frequency drift.' },
  ],
  default: [
    { value: 'clean', label: 'Clean Window', note: 'Low-noise synthetic signal for AI testing.' },
    { value: 'line_noise', label: 'Line Noise', note: 'Injects mains-style contamination.' },
    { value: 'drift', label: 'Drift', note: 'Adds a slow baseline shift.' },
    { value: 'motion', label: 'Motion Artifact', note: 'Adds transient motion-like artifacts.' },
    { value: 'clipping', label: 'Clipping', note: 'Overdrives the waveform.' },
  ],
};

function syntheticScenarioOptions(profileKey = S.signalProfileKey) {
  return SYNTHETIC_SCENARIOS[String(profileKey || '').toLowerCase()] || SYNTHETIC_SCENARIOS.default;
}

function syntheticScenarioStorageKey(profileKey = S.signalProfileKey) {
  return `kyma-synthetic-scenario-${String(profileKey || 'default').toLowerCase()}`;
}

function populateSyntheticScenarioOptions(profileKey = S.signalProfileKey) {
  const sel = $('synthetic-scenario');
  const note = $('synthetic-scenario-note');
  if (!sel) return;
  const options = syntheticScenarioOptions(profileKey);
  const preferred = String(S.streamDetails?.scenario || localStorage.getItem(syntheticScenarioStorageKey(profileKey)) || options[0]?.value || 'clean');
  sel.innerHTML = '';
  options.forEach(item => {
    const opt = document.createElement('option');
    opt.value = item.value;
    opt.textContent = item.label;
    sel.appendChild(opt);
  });
  sel.value = options.some(item => item.value === preferred) ? preferred : (options[0]?.value || 'clean');
  if (note) {
    const active = options.find(item => item.value === sel.value) || options[0];
  note.textContent = active?.note || 'Scenario for QA.';
  }
}

function getSelectedSyntheticScenario() {
  const profileKey = S.signalProfileKey;
  return $('synthetic-scenario')?.value
    || localStorage.getItem(syntheticScenarioStorageKey(profileKey))
    || syntheticScenarioOptions(profileKey)[0]?.value
    || 'clean';
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
    opt.textContent = 'No sessions';
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

function protocolCountdownMs() {
  if (!S.protocolRunner.active || !Number.isFinite(S.protocolRunner.endsAt)) return 0;
  return Math.max(0, Number(S.protocolRunner.endsAt) - Date.now());
}

function stopProtocolTicker() {
  if (S.protocolRunner.tickId) {
    window.clearInterval(S.protocolRunner.tickId);
    S.protocolRunner.tickId = 0;
  }
}

function startProtocolTicker() {
  if (S.protocolRunner.tickId) return;
  S.protocolRunner.tickId = window.setInterval(() => {
    if (!S.protocolRunner.active) {
      stopProtocolTicker();
      return;
    }
    syncProtocolUI();
  }, 200);
}

function finishProtocolRunner({ phase = 'idle', keepStep = false } = {}) {
  stopProtocolTicker();
  S.protocolRunner.active = false;
  S.protocolRunner.phase = phase;
  S.protocolRunner.endsAt = 0;
  S.protocolRunner.token = 0;
  if (!keepStep) S.protocolRunner.step = null;
  syncProtocolUI();
}

async function waitProtocolDuration(durationMs, token) {
  const total = Math.max(0, Number(durationMs || 0));
  const started = Date.now();
  while (Date.now() - started < total) {
    if (!S.protocolRunner.active || S.protocolRunner.token !== token) return false;
    const remaining = total - (Date.now() - started);
    await new Promise(resolve => window.setTimeout(resolve, Math.min(250, Math.max(25, remaining))));
  }
  return S.protocolRunner.active && S.protocolRunner.token === token;
}

async function postProtocolMarker(event, step, phase) {
  if (!S.streaming) return;
  try {
    await post('/api/review/marker', {
      event,
      note: `${phase}: ${step?.label || 'protocol'}`,
      metrics: {
        protocol_key: S.protocolRunner.templateKey || '',
        run_id: S.protocolRunner.runId || '',
        phase,
        label: step?.label || '',
        trial_index: step?.trial_index ?? null,
        repetition_index: step?.repetition_index ?? null,
      },
    });
  } catch {}
}

async function stopSessionRecordingForRunner() {
  if (!S.recSession) return;
  const r = await post('/api/session/stop');
  S.recSession = false;
      $('btn-record-session').textContent = 'Record';
  $('btn-record-session').className = 'btn';
  await loadSessions();
  return r;
}

async function runProtocolRunner() {
  const template = getSelectedProtocolTemplate();
  if (!template) {
    toast('Select a protocol first', 'red');
    return;
  }
  if (!S.streaming || S.streamSource === 'playback') {
    toast('Protocol runner needs a live hardware, synthetic, or LSL stream', 'red');
    return;
  }
  if (S.recSession) {
    toast('Stop the active session before starting the protocol runner', 'red');
    return;
  }

  const plan = buildProtocolPlan(template);
  const nextStep = plan[S.protocolStepIndex];
  if (!nextStep) {
    toast('This protocol run is already complete. Reset the run first.', 'yellow');
    return;
  }

  const runId = ensureProtocolRunId(template);
  const token = Date.now();
  S.protocolRunner.active = true;
  S.protocolRunner.phase = 'arming';
  S.protocolRunner.step = nextStep;
  S.protocolRunner.endsAt = 0;
  S.protocolRunner.token = token;
  S.protocolRunner.runId = runId;
  S.protocolRunner.templateKey = template.key || '';
  startProtocolTicker();
  syncProtocolUI();

  try {
    while (S.protocolRunner.active && S.protocolRunner.token === token) {
      const step = plan[S.protocolStepIndex];
      if (!step) break;

      S.protocolRunner.step = step;
      S.protocolRunner.phase = 'trial';
      S.protocolRunner.endsAt = Date.now() + Math.max(0, Number(template.trial_duration_s || 0) * 1000);
      syncProtocolUI();

      if ($('session-label')) $('session-label').value = step.label;
      await postProtocolMarker('protocol_trial_start', step, 'trial_start');
      await startSessionRecording({
        label: step.label,
        protocol_key: template.key,
        protocol_title: template.title || template.key,
        session_group_id: runId,
        trial_index: step.trial_index,
        repetition_index: step.repetition_index,
      });

      const trialDone = await waitProtocolDuration(Number(template.trial_duration_s || 0) * 1000, token);
      if (!trialDone) break;

      await stopSessionRecordingForRunner();
      await postProtocolMarker('protocol_trial_end', step, 'trial_end');
      S.protocolStepIndex += 1;
      syncProtocolUI();

      const upcoming = plan[S.protocolStepIndex];
      if (!upcoming) break;

      const restMs = Math.max(0, Number(template.rest_duration_s || 0) * 1000);
      if (restMs > 0) {
        S.protocolRunner.phase = 'rest';
        S.protocolRunner.endsAt = Date.now() + restMs;
        await postProtocolMarker('protocol_rest_start', step, 'rest_start');
        syncProtocolUI();
        const restDone = await waitProtocolDuration(restMs, token);
        if (!restDone) break;
        await postProtocolMarker('protocol_rest_end', step, 'rest_end');
      }
    }

    if (S.protocolRunner.active && S.protocolRunner.token === token && !plan[S.protocolStepIndex]) {
      finishProtocolRunner({ phase: 'complete', keepStep: true });
      toast('Protocol run complete');
      return;
    }
  } catch (e) {
    toast(e.message, 'red');
  }

  if (S.recSession) {
    try {
      await stopSessionRecordingForRunner();
    } catch {}
  }
  finishProtocolRunner({ phase: 'idle' });
}

async function stopProtocolRunner() {
  if (!S.protocolRunner.active) {
    finishProtocolRunner({ phase: 'idle' });
    return;
  }
  S.protocolRunner.active = false;
  if (S.recSession) {
    try {
      await stopSessionRecordingForRunner();
    } catch (e) {
      toast(e.message, 'yellow');
    }
  }
  finishProtocolRunner({ phase: 'idle' });
  toast('Protocol runner stopped', 'yellow');
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
  refreshReviewArtifacts(getReviewRenderState());
  if ($('bench-report')?.classList.contains('active')) refreshBenchReportUI();
}

function liveMLResultFromPayload(payload) {
  const local = payload?.local_model_insights || payload || {};
  const artifact = local?.artifact_classifier || {};
  const label = String(artifact?.label || '').toLowerCase();
  const clean = !label || ['clean', 'none', 'normal', 'stable', 'ok'].includes(label);
  const stats = payload?.stats || {};
  const qaScore = local?.signal_qa?.score ?? (clean ? 86 : 58);
  const embeddings = Array.isArray(local?.foundation_embeddings) ? local.foundation_embeddings : [];
  const primaryEmbedding = embeddings[0] || null;
  const runtimeParts = [
    Number.isFinite(Number(payload?.inference_ms)) ? `${Number(payload.inference_ms).toFixed(1)} ms ML` : '',
    Number(payload?.channel_count || 0) ? `${Number(payload.channel_count || 0)} ch` : '',
    Number(payload?.window_samples || 0) ? `${Number(payload.window_samples || 0)} samples` : '',
    primaryEmbedding ? `${primaryEmbedding.name || primaryEmbedding.model_id || 'Foundation'} ${Number(primaryEmbedding.embedding_dim || 0)}-d` : '',
  ].filter(Boolean);
  return {
    source: 'live_ml',
    summary: clean
      ? `Live model pass sees no artifact threshold crossing${runtimeParts.length ? ` (${runtimeParts.join(' | ')})` : ''}.`
      : `Live model pass is leaning toward ${label.replace(/_/g, ' ')}${runtimeParts.length ? ` (${runtimeParts.join(' | ')})` : ''}.`,
    measurements: {
      sample_rate_hz: Number(payload?.sample_rate_hz || S.sampleRate || 0),
      visible_span_ms: Number(stats.duration_ms || 0),
      rms: Number(stats.rms || 0),
      peak_to_peak: Number(stats.peak_to_peak || 0),
      focus_label: String(stats.focus_label || ''),
      focus_rms: Number(stats.focus_rms || 0),
      units: S.signalUnits || 'a.u.',
      inference_ms: Number(payload?.inference_ms || 0),
      channel_count: Number(payload?.channel_count || 0),
      window_samples: Number(payload?.window_samples || 0),
      foundation_embedding_count: Number(payload?.foundation_embedding_count || embeddings.length || 0),
    },
    qa_score: {
      overall: Math.round(Number(qaScore || 0)),
      grade: Number(qaScore || 0) >= 80 ? 'A' : (Number(qaScore || 0) >= 65 ? 'B' : 'C'),
      dimensions: [],
    },
    artifact_summary: {
      count: clean ? 0 : 1,
      top_issue: clean ? 'clean_window' : label,
      model_hint: label,
    },
    local_model_insights: local,
    research_watchlist: Array.isArray(local?.condition_screen) ? local.condition_screen : [],
    next_actions: clean
      ? (primaryEmbedding ? [{
        title: 'Track foundation embedding drift',
        detail: `${primaryEmbedding.name || primaryEmbedding.model_id || 'Foundation encoder'} produced a ${Number(primaryEmbedding.embedding_dim || 0)}-d representation for the live window.`,
        priority: 'low',
      }] : [])
      : [{
      title: `Check ${label.replace(/_/g, ' ')} risk`,
      detail: artifact.detail || 'The live local model flagged the current stream window.',
      priority: Number(artifact.confidence || 0) >= 0.8 ? 'high' : 'medium',
    }],
  };
}

function applyLiveMLInsights(payload) {
  if (!payload) return;
  S.liveMl = payload;
  S.liveMlSeenAt = Date.now();
  const local = payload.local_model_insights || {};
  if (local.model_count || local.foundation_model_count || local.builtin_heads) {
    S.ai.localModels = {
      ...(S.ai.localModels || {}),
      ...local,
    };
  }
  const liveResult = liveMLResultFromPayload(payload);
  const artifactLabel = String(local?.artifact_classifier?.label || '').toLowerCase();
  const nonClean = artifactLabel && !['clean', 'none', 'normal', 'stable', 'ok'].includes(artifactLabel);
  const reviewState = getReviewRenderState();
  const regionsVisible = regionsOverlapReviewViewport(S.ai.regions || [], reviewState);
  const shouldRefreshLiveResult = !S.ai.loading && (!S.ai.result || S.ai.result.source === 'live_ml' || nonClean || !regionsVisible);
  if (shouldRefreshLiveResult) {
    S.ai.result = liveResult;
    S.ai.regions = buildAIHighlightRegions(liveResult, reviewState);
    if (S.ai.regions.length) {
      const first = S.ai.regions[0];
      S.ai.spotlight = buildSpotlightPayload(first.startSample, first.endSample, first.label, first.kind, first.channel);
    }
    syncAICopilotUI();
  }
}

function applyPromptModelPrediction(payload) {
  if (!payload) return;
  const label = String(payload.label || '--');
  const confidence = Number(payload.confidence || 0);
  const windowSamples = Math.max(1, Number(payload.window_samples || 0) || Math.round(Number(S.sampleRate || 250) * 0.25));
  const endSample = Math.max(0, Number(S.emgTotal || 0) - 1);
  const startSample = Math.max(0, endSample - windowSamples + 1);
  const style = promptModelLabelStyle(label);
  const evidence = Array.isArray(payload.evidence_channels) ? payload.evidence_channels : [];
  const selectedEvidence = evidence.length
    ? evidence
    : [{ channel: getReviewFocusChannelIndex(getReviewRenderState()), channel_label: S.channelLabels[getReviewFocusChannelIndex(getReviewRenderState())] || 'CH1', score: confidence }];

  const nextRegions = selectedEvidence
    .filter(item => Number.isFinite(Number(item.channel)))
    .map(item => {
      const ch = clamp(Number(item.channel), 0, N_CH - 1);
      const score = Number(item.score || confidence || 0);
      return {
        kind: 'prompt_model',
        label: `Model ${label}`,
        detail: `${label} ${(confidence * 100).toFixed(1)}% | evidence ${(score * 100).toFixed(0)}%`,
        channel: ch,
        channelLabel: String(item.channel_label || S.channelLabels[ch] || `CH${ch + 1}`),
        startSample,
        endSample,
        score,
        confidence,
        source: 'prompt_model',
        color: style.color,
        line: style.line,
        probabilities: payload.probabilities || {},
        inferenceMs: Number(payload.inference_ms || 0),
        modelPath: String(payload.model_path || ''),
        until: Number.POSITIVE_INFINITY,
      };
    });

  const existing = Array.isArray(S.promptModel.regions) ? S.promptModel.regions : [];
  const merged = [...existing, ...nextRegions]
    .sort((a, b) => Number(a.startSample || 0) - Number(b.startSample || 0));
  S.promptModel.regions = mergeReviewArtifactRegions(merged)
    .slice(-Math.max(12, Number(S.promptModel.maxRegions || 96)));
  S.promptModel.prediction = {
    ...payload,
    label,
    confidence,
    regions: nextRegions,
  };
  S.promptModel.seenAt = Date.now();

  S.lastPrediction = clonePredictionPayload({
    ...payload,
    label,
    gesture: label,
    summary: `Prompt model: ${label} ${(confidence * 100).toFixed(1)}%`,
  });
  if (!S.review.paused) {
    renderPredictionPanel(S.lastPrediction, { animate: true });
  }
  if (nextRegions.length) {
    const first = nextRegions[0];
    S.ai.spotlight = buildSpotlightPayload(first.startSample, first.endSample, first.label, first.kind, first.channel);
    S.ai.spotlight.color = first.color;
    S.ai.spotlight.line = first.line;
  }
  S.predCount++;
  const gIdx = S.gestures.indexOf(label);
  S.timeline.push({ g: gIdx >= 0 ? gIdx : 0, c: confidence });
  if (S.timeline.length > S.timelineMax) S.timeline.shift();
  drawTimeline();
  syncInspectorTelemetry();
}

function liveMLRuntimeText(payload = S.liveMl) {
  if (!payload) return '';
  const local = payload.local_model_insights || {};
  const embeddings = Array.isArray(local.foundation_embeddings) ? local.foundation_embeddings : [];
  const primary = embeddings[0] || null;
  const parts = [];
  if (Number.isFinite(Number(payload.inference_ms))) parts.push(`${Number(payload.inference_ms).toFixed(1)} ms`);
  if (Number(payload.channel_count || 0) || Number(payload.window_samples || 0)) {
    parts.push(`${Number(payload.channel_count || 0)} ch / ${Number(payload.window_samples || 0)} samples`);
  }
  if (primary) {
    parts.push(`${primary.name || primary.model_id || 'Foundation'} ${Number(primary.embedding_dim || 0)}-d`);
  } else if (Number(payload.foundation_embedding_count || 0) > 0) {
    parts.push(`${Number(payload.foundation_embedding_count || 0)} embedding${Number(payload.foundation_embedding_count || 0) === 1 ? '' : 's'}`);
  }
  const artifact = local.artifact_classifier || {};
  if (artifact.label) parts.push(`${String(artifact.label).replace(/_/g, ' ')} ${Math.round(Number(artifact.confidence || 0) * 100)}%`);
  return parts.join(' | ');
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
  const autoBtn = $('btn-protocol-auto');
  const stopBtn = $('btn-protocol-stop');
  const runnerStatus = $('protocol-runner-status');
  const phaseValue = $('protocol-phase');
  const stepValue = $('protocol-step');
  const countdownValue = $('protocol-countdown');
  const template = getSelectedProtocolTemplate() || (S.protocolTemplates[0] || null);

  if (select) {
    const current = select.value || template?.key || '';
    select.innerHTML = '';
    if (!S.protocolTemplates.length) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No templates';
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
      summary.textContent = 'Load profile for templates.';
    } else {
      const labels = (activeTemplate.labels || []).join(', ') || 'no labels';
      const nextText = nextStep
        ? `Next: ${nextStep.label} (trial ${nextStep.trial_index}, rep ${nextStep.repetition_index})`
        : 'Run complete.';
      const duration = Number(activeTemplate.estimated_duration_s || 0).toFixed(1);
      summary.textContent = `${activeTemplate.summary} Labels: ${labels}. ${activeTemplate.repetitions} reps. ~${duration}s. ${nextText}`;
    }
  }

  if (nextBtn) {
    nextBtn.disabled = !activeTemplate || !plan.length || S.recSession || !S.streaming || S.streamSource === 'playback' || S.protocolRunner.active;
    nextBtn.textContent = nextStep ? `Next ${nextStep.label}` : 'Complete';
  }

  if (autoBtn) {
    autoBtn.disabled = !activeTemplate || !plan.length || !S.streaming || S.streamSource === 'playback' || S.recSession || S.protocolRunner.active;
    autoBtn.textContent = S.protocolRunner.active ? 'Auto On' : 'Auto';
  }
  if (stopBtn) stopBtn.disabled = !S.protocolRunner.active && !S.recSession;

  const runnerStep = S.protocolRunner.step || nextStep || null;
  const countdown = protocolCountdownMs();
  if (runnerStatus) {
    if (S.protocolRunner.active) {
      const phaseText = S.protocolRunner.phase === 'rest' ? 'Rest' : 'Trial';
      runnerStatus.textContent = `${phaseText} mode active.`;
    } else if (S.protocolRunner.phase === 'complete') {
      runnerStatus.textContent = 'Run finished.';
    } else {
      runnerStatus.textContent = 'Manual step mode ready. Auto Run advances trial and rest.';
    }
  }
  if (phaseValue) {
    phaseValue.textContent = S.protocolRunner.active
      ? String(S.protocolRunner.phase || 'trial').replace(/_/g, ' ')
      : (S.protocolRunner.phase === 'complete' ? 'complete' : 'manual');
  }
  if (stepValue) {
    stepValue.textContent = runnerStep
      ? `${runnerStep.label} (${runnerStep.trial_index}/${Math.max(plan.length, 1)})`
      : '--';
  }
  if (countdownValue) {
    countdownValue.textContent = S.protocolRunner.active ? formatAxisDuration(countdown) : '--';
  }
  if ($('btn-record-session')) $('btn-record-session').disabled = !!S.protocolRunner.active;
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
      opt.textContent = 'No EEG presets';
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
      summary.textContent = 'Switch to EEG for presets.';
    } else if (!active) {
      summary.textContent = 'No EEG presets.';
    } else {
      const exportText = Array.isArray(active.recommended_export) && active.recommended_export.length
        ? active.recommended_export.join(' + ').toUpperCase()
        : 'session export';
      const steps = Array.isArray(active.instructions) && active.instructions.length
        ? ` ${active.instructions.join(' | ')}`
        : '';
      summary.textContent = `${active.summary} Mode: ${(active.mode || 'record_only').replace(/_/g, ' ')}. Source: ${active.recommended_source || 'hardware'}. Export: ${exportText}.${steps}`;
    }
  }

  if (applyBtn) {
    applyBtn.disabled = !active || S.signalProfileKey !== 'eeg';
    applyBtn.textContent = S.signalProfileKey === 'eeg' ? 'Apply' : 'EEG Only';
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
    summary.textContent = 'Switch to EEG for marker hints.';
    strategy.textContent = '--';
    markers.textContent = '--';
    blocks.textContent = '--';
    return;
  }

  if (!active) {
    summary.textContent = 'Pick an EEG preset first.';
    strategy.textContent = '--';
    markers.textContent = '--';
    blocks.textContent = '--';
    return;
  }

  summary.textContent = `${active.title}: marker names and block notes.`;
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
    opt.textContent = 'No preset markers';
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
    status.textContent = 'Switch to EEG for preset markers.';
  } else if (!lslReady) {
    status.textContent = 'Start LSL in Control, then test here.';
  } else {
    status.textContent = `Sending to ${S.lsl.marker_stream_name || 'KYMA marker stream'}.`;
  }

  sendBtn.disabled = !lslReady;
}

function syncReviewUI() {
  const pauseBtn = $('btn-review-pause');
  const clearBtn = $('btn-review-clear-selection');
  const hZoomOutBtn = $('btn-review-hzoom-out');
  const hZoomInBtn = $('btn-review-hzoom-in');
  const vZoomOutBtn = $('btn-review-vzoom-out');
  const vZoomInBtn = $('btn-review-vzoom-in');
  const markerToolBtn = $('btn-review-marker-tool');
  const envelopeBtn = $('btn-scope-envelope');
  const thresholdBtn = $('btn-scope-thresholds');
  const markerReadout = $('review-cursor-readout');
  const status = $('review-status');
  const chip = $('scope-status-chip');
  const list = $('review-marker-list');
  const markerBtn = $('btn-review-marker');
  const workshopBtn = $('btn-review-workshop');
  const container = $('canvas-container');
  const timelineLabel = $('timeline-label');
  const scrubber = $('review-scrubber');
  const scrubStart = $('review-scrub-start');
  const scrubEnd = $('review-scrub-end');
  const scrubCaption = $('review-scrub-caption');
  const stats = S.review.lastStats;
  const paused = !!S.review.paused;
  const state = getReviewRenderState();
  const cursorMetrics = getReviewCursorMetrics(state);
  const hoverMetrics = getReviewHoverMetrics(state);
  const cursorSample = Number.isFinite(S.review.hoverSample) ? Number(S.review.hoverSample) : null;
  const markerMode = stats && stats.samples > 1
    ? 'Save Range Marker'
    : (cursorSample !== null ? 'Save Point Marker' : 'Save Marker');

  if (pauseBtn) {
    pauseBtn.textContent = paused ? 'Resume Review' : 'Freeze Review';
    pauseBtn.disabled = !S.streaming && !S.review.snapshot;
  }
  if (clearBtn) clearBtn.disabled = !S.review.selection;
  if (markerBtn) {
    markerBtn.disabled = !S.streaming && !S.review.snapshot;
    markerBtn.textContent = markerMode;
  }
  if (workshopBtn) workshopBtn.disabled = !S.review.snapshot;
  const viewport = getReviewViewport(state, canvas.width || 1);
  const xSpanMs = (Math.max(Number(viewport.viewSamples || 0), 1) / Math.max(Number(state.sampleRate || 250), 1)) * 1000;
  const yFullScale = Math.max(Number(getPlaybackDisplayFullScale(state) || S.signalFullScale || 200), 0.001) / Math.max(Number(S.review.zoomY || 1), 0.25);
  if ($('review-hzoom-label')) $('review-hzoom-label').textContent = formatAxisDuration(xSpanMs);
  if ($('review-hzoom-meta')) $('review-hzoom-meta').textContent = 'visible span';
  if ($('review-vzoom-label')) $('review-vzoom-label').textContent = formatAxisAmplitude(yFullScale, S.signalUnits);
  if ($('review-vzoom-meta')) $('review-vzoom-meta').textContent = S.signalUnits || 'units';
  const hasSignal = !!state.filled;
  if (hZoomOutBtn) hZoomOutBtn.disabled = !hasSignal || Number(S.review.zoomX || 1) <= REVIEW_X_ZOOM_LEVELS[0];
  if (hZoomInBtn) hZoomInBtn.disabled = !hasSignal || Number(S.review.zoomX || 1) >= REVIEW_X_ZOOM_LEVELS[REVIEW_X_ZOOM_LEVELS.length - 1];
  if (vZoomOutBtn) vZoomOutBtn.disabled = !hasSignal || Number(S.review.zoomY || 1) <= REVIEW_Y_ZOOM_LEVELS[0];
  if (vZoomInBtn) vZoomInBtn.disabled = !hasSignal || Number(S.review.zoomY || 1) >= REVIEW_Y_ZOOM_LEVELS[REVIEW_Y_ZOOM_LEVELS.length - 1];
  if (markerToolBtn) {
    markerToolBtn.disabled = !hasSignal;
    markerToolBtn.classList.toggle('active', !!S.review.markerTool);
    markerToolBtn.textContent = S.review.markerTool ? 'Marker On' : 'Marker';
  }
  if (envelopeBtn) {
    envelopeBtn.classList.toggle('active', !!S.scopeOverlays.envelope);
    envelopeBtn.textContent = S.scopeOverlays.envelope ? 'Envelope On' : 'Envelope';
  }
  if (thresholdBtn) {
    thresholdBtn.classList.toggle('active', !!S.scopeOverlays.thresholds);
    thresholdBtn.textContent = S.scopeOverlays.thresholds ? 'Threshold On' : 'Threshold';
  }
  if (markerReadout) {
    if (!hasSignal) {
      markerReadout.textContent = 'No signal window.';
    } else if (!cursorMetrics || (!cursorMetrics.cursors.a && !cursorMetrics.cursors.b)) {
      markerReadout.textContent = S.review.markerTool
        ? 'Click the plot to place A, then B. Drag a cursor line to move it.'
        : 'Marker tool off.';
    } else {
      const lines = [`Focus ${cursorMetrics.focusLabel}`];
      if (cursorMetrics.cursors.a) {
        lines.push(`A  ${cursorMetrics.cursors.a.timeLabel}`);
        lines.push(`   ${cursorMetrics.cursors.a.valueLabel}`);
      }
      if (cursorMetrics.cursors.b) {
        lines.push(`B  ${cursorMetrics.cursors.b.timeLabel}`);
        lines.push(`   ${cursorMetrics.cursors.b.valueLabel}`);
      }
      if (cursorMetrics.delta) {
        lines.push(`Δt ${cursorMetrics.delta.sign}${cursorMetrics.delta.timeLabel}`);
        lines.push(`Δv ${cursorMetrics.delta.valueLabel}`);
      }
      markerReadout.textContent = lines.join('\n');
    }
  }
  if (container) container.classList.toggle('review-paused', paused);
  if (timelineLabel) timelineLabel.textContent = paused ? 'Frozen decoded output snapshot' : 'Last 60 decoded outputs';
  if (chip) {
    chip.textContent = paused ? 'Paused' : 'Live';
    chip.classList.toggle('paused', paused);
  }

  if ($('review-cursor-pos')) {
    $('review-cursor-pos').textContent = hoverMetrics
      ? `${hoverMetrics.timeLabel} | ${hoverMetrics.channelLabel} ${hoverMetrics.valueLabel}`
      : '--';
  }
  if ($('review-selection-span')) $('review-selection-span').textContent = stats ? `${stats.samples} samples` : 'None';
  if ($('review-selection-range')) $('review-selection-range').textContent = stats ? reviewRangeLabel(stats, state) : '--';
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
      status.textContent = 'Start a stream first.';
    } else if (paused && stats) {
      status.textContent = 'Frozen. Hover for voltage. Drag for range. Slider or wheel to pan.';
    } else if (paused) {
      status.textContent = 'Frozen. Hover for voltage. Drag for range. Slider or wheel to pan.';
    } else {
      status.textContent = 'Live trace. Freeze to inspect.';
    }
  }

  if (scrubber) {
    const fullStart = Number(viewport.fullStart || 0);
    const fullEnd = Number(viewport.fullEnd || fullStart);
    const centerSample = clampReviewCenterSample(
      Number.isFinite(S.review.viewCenterSample)
        ? Number(S.review.viewCenterSample)
        : (Number(viewport.viewStart || fullStart) + Math.floor(Math.max(Number(viewport.viewSamples || 1) - 1, 0) / 2)),
      state,
      canvas.width || 1,
    );
    scrubber.min = String(fullStart);
    scrubber.max = String(Math.max(fullStart, fullEnd));
    scrubber.step = '1';
    scrubber.value = String(centerSample);
    scrubber.disabled = !paused || !hasSignal || fullEnd <= fullStart;
  }
  if (scrubStart) scrubStart.textContent = hasSignal ? reviewPointLabel(Number(viewport.fullStart || 0), state) : '--';
  if (scrubEnd) scrubEnd.textContent = hasSignal ? reviewPointLabel(Number(viewport.fullEnd || 0), state) : '--';
  if (scrubCaption) {
    if (!paused || !hasSignal) {
      scrubCaption.textContent = 'Freeze review to scrub through the capture.';
    } else {
      scrubCaption.textContent = `Viewing ${reviewPointLabel(Number(viewport.viewStart || 0), state)} to ${reviewPointLabel(Number(viewport.viewEnd || 0), state)}`;
    }
  }

  refreshAICopilotAvailability(state);
  refreshReviewArtifacts(state);

  if (list) {
    list.innerHTML = '';
    if (!S.review.markers.length) {
      const empty = document.createElement('div');
      empty.className = 'setup-copy';
      empty.textContent = 'No markers yet.';
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
          : (Number.isFinite(item.sampleIndex) ? ` ${reviewPointLabel(item.sampleIndex, state)}` : '');
        note.textContent = `${item.note || 'No note.'}${selectionText}`.trim();

        head.append(name, range);
        row.append(head, note);
        list.appendChild(row);
      });
    }
  }
}

function buildAICopilotWindow(state) {
  if (!state?.filled) return null;
  const selected = getSelectionRange(S.review.selection);
  const viewport = getReviewViewport(state, Math.max(canvas.width || 0, 1));
  const startSample = selected?.start ?? viewport.viewStart;
  const endSample = selected?.end ?? viewport.viewEnd;
  const totalSamples = Math.max(1, endSample - startSample + 1);
  const targetSamples = Math.min(256, Math.max(64, totalSamples));
  const stepSamples = Math.max(1, Math.ceil(totalSamples / targetSamples));
  const focusChannel = Number(S.review.lastStats?.focusChannel || 0);
  const channels = [];

  for (let ch = 0; ch < N_CH; ch += 1) {
    const buf = state.emg?.[ch];
    if (!buf) {
      channels.push([]);
      continue;
    }
    const series = [];
    for (let sample = startSample; sample <= endSample; sample += stepSamples) {
      const idx = bufferIndexForAbsSample(state, sample);
      series.push(Number(Number(buf[idx] || 0).toFixed(6)));
    }
    if ((startSample + (series.length - 1) * stepSamples) < endSample) {
      const tailIdx = bufferIndexForAbsSample(state, endSample);
      series.push(Number(Number(buf[tailIdx] || 0).toFixed(6)));
    }
    channels.push(series);
  }

  return {
    sample_rate_hz: Number(state.sampleRate || 250),
    start_sample: Number(startSample),
    end_sample: Number(endSample),
    samples: Number(totalSamples),
    step_samples: Number(stepSamples),
    focus_channel: focusChannel,
    focus_label: String(S.review.lastStats?.focusLabel || S.channelLabels[focusChannel] || `CH${focusChannel + 1}`),
    channels,
  };
}

function buildAICopilotPayload() {
  const state = getReviewRenderState();
  const stats = S.review.lastStats || null;
  const template = getSelectedProtocolTemplate();
  const prediction = clonePredictionPayload(S.lastPrediction);
  const workshopSummary = S.workshop.result?.summary || null;
  const reviewWindow = buildAICopilotWindow(state);
  const selection = stats ? {
    start_ms: Number(reviewSampleOffsetMs(stats.startSample, state).toFixed(3)),
    end_ms: Number(reviewSampleOffsetMs(stats.endSample, state).toFixed(3)),
    duration_ms: Number((stats.durationMs || 0).toFixed(3)),
  } : null;
  return {
    channel_labels: S.channelLabels.slice(0, N_CH),
    diagnostics: structuredClone(S.diagnostics || {}),
    review: {
      paused: !!S.review.paused,
      cursor_label: Number.isFinite(S.review.hoverSample) ? reviewPointLabel(S.review.hoverSample, state) : '',
      stats: stats ? {
        samples: Number(stats.samples || 0),
        duration_ms: Number((stats.durationMs || 0).toFixed(3)),
        mean: Number((stats.mean || 0).toFixed(6)),
        rms: Number((stats.rms || 0).toFixed(6)),
        peak_to_peak: Number((stats.peakToPeak || 0).toFixed(6)),
        focus_channel: Number(stats.focusChannel || 0),
        focus_label: String(stats.focusLabel || ''),
        focus_rms: Number((stats.focusRms || 0).toFixed(6)),
      } : null,
      selection,
      artifacts: (S.review.artifacts || []).map(item => ({
        kind: String(item.kind || ''),
        label: String(item.label || ''),
        channel: Number(item.channel ?? -1),
        channel_label: String(item.channelLabel || ''),
        detail: String(item.detail || ''),
        score: Number(item.score || 0),
        start_ms: Number(reviewSampleOffsetMs(item.startSample, state).toFixed(3)),
        end_ms: Number(reviewSampleOffsetMs(item.endSample, state).toFixed(3)),
      })),
      window: reviewWindow,
    },
    protocol: {
      active: !!S.protocolRunner.active,
      template_key: String(S.protocolRunner.templateKey || template?.key || ''),
      template_title: String(template?.title || ''),
      phase: String(S.protocolRunner.phase || 'idle'),
      step_label: String(S.protocolRunner.step?.label || ''),
      remaining_ms: Number(protocolCountdownMs().toFixed(3)),
    },
    workshop: {
      has_result: !!S.workshop.result,
      view: String(S.workshop.view || ''),
      selection_label: String(S.workshop.result?.selection_label || S.workshop.selectionMeta?.rangeLabel || ''),
      summary: workshopSummary ? structuredClone(workshopSummary) : {},
    },
    last_prediction: prediction || {},
    filter_chain: {
      label: String($('filter-chain-label')?.textContent || '').trim(),
    },
    session: {
      session_id: String(S.streamDetails?.session_id || ''),
      label: String($('session-label')?.value || ''),
      subject_id: String($('session-subject')?.value || ''),
      condition: String($('session-condition')?.value || ''),
      notes: String($('session-notes')?.value || ''),
      workspace: String(S.dashboardWorkspace || 'live'),
    },
  };
}

function aiScoreColor(score) {
  const numeric = Number(score || 0);
  if (numeric >= 85) return 'var(--green)';
  if (numeric >= 65) return 'var(--yellow)';
  return 'var(--red)';
}

function aiIssueCounts(result = S.ai.result) {
  const counts = { hum: 0, drift: 0, clip: 0, artifact: 0 };
  (S.review.artifacts || []).forEach(item => {
    const kind = String(item?.kind || '').toLowerCase();
    if (kind === 'drift') counts.drift += 1;
    else if (kind === 'clip') counts.clip += 1;
    else if (kind === 'hum') counts.hum += 1;
    else counts.artifact += 1;
  });
  if (Number(S.diagnostics?.noise?.hum_50_db ?? -Infinity) >= -20 || Number(S.diagnostics?.noise?.hum_60_db ?? -Infinity) >= -20) {
    counts.hum = Math.max(counts.hum, 1);
  }
  if (Number(S.diagnostics?.noise?.drift_db ?? -Infinity) >= -20) {
    counts.drift = Math.max(counts.drift, 1);
  }
  if (Number(S.diagnostics?.noise?.clip_pct ?? 0) > 1) {
    counts.clip = Math.max(counts.clip, 1);
  }
  const modelHint = String(result?.local_model_insights?.artifact_classifier?.label || '').toLowerCase();
  if (modelHint && !['clean', 'none', 'normal', 'stable'].includes(modelHint)) {
    if (modelHint.includes('hum')) counts.hum = Math.max(counts.hum, 1);
    else if (modelHint.includes('drift')) counts.drift = Math.max(counts.drift, 1);
    else if (modelHint.includes('clip')) counts.clip = Math.max(counts.clip, 1);
    else counts.artifact = Math.max(counts.artifact, 1);
  }
  return counts;
}

function aiResultSignature(result) {
  if (!result) return '';
  return JSON.stringify({
    overall: Number(result?.qa_score?.overall || 0),
    grade: String(result?.qa_score?.grade || ''),
    top_issue: String(result?.artifact_summary?.top_issue || ''),
    artifact_count: Number(result?.artifact_summary?.count || 0),
    summary: compactAISummary(result?.summary, 96),
    next: (result?.next_actions || []).map(item => String(item?.title || '')).slice(0, 2),
    watch: (result?.research_watchlist || []).map(item => String(item?.label || '')).slice(0, 2),
    signal_qa: Number(result?.local_model_insights?.signal_qa?.score || 0),
    artifact_model: String(result?.local_model_insights?.artifact_classifier?.label || ''),
    readiness: Number(result?.local_model_insights?.fatigue_readiness?.score || 0),
  });
}

function setAICopilotTab(tab, { persist = true } = {}) {
  const next = ['summary', 'actions', 'markers', 'exports'].includes(String(tab || ''))
    ? String(tab)
    : 'summary';
  S.ai.activeTab = next;
  if (persist) localStorage.setItem('kyma-ai-tab', next);
}

function stopAIThinkingLoop() {
  if (S.ai.thinkingTimer) {
    clearInterval(S.ai.thinkingTimer);
    S.ai.thinkingTimer = 0;
  }
  S.ai.thinkingStep = 0;
}

function startAIThinkingLoop() {
  stopAIThinkingLoop();
}

function currentAIReasoningStep() {
  return AI_REASONING_STEPS[0] || { stage: 'scan', line: 'Scanning window.' };
}

function aiCopilotModeLabel() {
  if (Date.now() - Number(S.liveMlSeenAt || 0) < 2500) return 'Live ML';
  if (S.ai.loading) return S.ai.background ? 'Quiet' : 'Thinking';
  if (Number(S.ai.localModels?.model_count || 0) > 0 || Number(S.ai.localModels?.foundation_model_count || 0) > 0) return 'Hybrid';
  if (S.ai.configured) return 'Linked';
  return 'Local';
}

function aiPrimaryFinding(result = S.ai.result) {
  if (!result) {
    return {
      title: 'No scan yet',
      detail: 'Scan the window.',
      tone: 'idle',
    };
  }
  const counts = aiIssueCounts(result);
  const artifactHead = result?.local_model_insights?.artifact_classifier || null;
  const artifactHeadLabel = String(artifactHead?.label || '').trim();
  const artifactHeadDetail = String(artifactHead?.detail || '').trim();
  const top = Array.isArray(S.review.artifacts) && S.review.artifacts.length ? S.review.artifacts[0] : null;
  if (top) {
    return {
      title: `${top.label} on ${top.channelLabel}`,
      detail: String(top.detail || top.kind || '').replace(/_/g, ' '),
      tone: top.kind === 'clip' ? 'high' : 'medium',
    };
  }
  if (artifactHeadLabel && !['clean', 'none', 'normal', 'stable', 'ok'].includes(artifactHeadLabel.toLowerCase())) {
    const nice = artifactHeadLabel.replace(/_/g, ' ');
    return {
      title: `${nice.charAt(0).toUpperCase()}${nice.slice(1)} detected`,
      detail: artifactHeadDetail || 'A local artifact head flagged this window for review.',
      tone: artifactHeadLabel.includes('clip') ? 'high' : 'medium',
    };
  }
  if (counts.clip > 0) {
    return {
      title: 'Clipping risk present',
      detail: 'The current window is likely saturating and can distort amplitude readings.',
      tone: 'high',
    };
  }
  if (counts.hum > 0) {
    return {
      title: 'Line noise is present',
      detail: 'Mains contamination is strong enough to affect the spectrum and visual clarity.',
      tone: 'medium',
    };
  }
  if (counts.drift > 0) {
    return {
      title: 'Baseline drift is present',
      detail: 'The baseline is moving and can mask slower changes or inflate range.',
      tone: 'medium',
    };
  }
  if (counts.artifact > 0) {
    return {
      title: 'Artifact activity detected',
      detail: 'The scan found non-clean motion or signal-shape behavior in the current window.',
      tone: 'medium',
    };
  }
  const watch = Array.isArray(result?.research_watchlist) && result.research_watchlist.length ? result.research_watchlist[0] : null;
  if (watch?.kind === 'screening_result') {
    return {
      title: String(watch.display_label || watch.label || 'Research cue'),
      detail: String(watch.detail || 'A research-only screening cue was raised for this window.'),
      tone: watch.status === 'active' ? 'medium' : 'low',
    };
  }
  const count = Number(result?.artifact_summary?.count || 0);
  if (count > 0) {
    return {
      title: `${count} issue${count === 1 ? '' : 's'} flagged`,
      detail: 'The scan found one or more issues but did not localize them to a stronger region yet.',
      tone: 'medium',
    };
  }
  return {
    title: 'Artifact sweep complete',
    detail: 'No clip, hum, drift, or motion threshold crossed in this pass.',
    tone: 'low',
  };
}

function aiTopIssueLabel(result = S.ai.result) {
  return aiPrimaryFinding(result).title;
}

function aiMeaningText(result = S.ai.result, metrics = aiStripMeasurements()) {
  if (!result) return 'The lens has not scanned this signal yet.';
  const finding = aiPrimaryFinding(result);
  if (finding.tone === 'high') return `${finding.detail} Inspect this region before trusting the current amplitude or classifier output.`;
  if (finding.tone === 'medium') return `${finding.detail} This is the first thing to fix or verify before you trust the current pass.`;
  const dom = metrics?.dominantHz > 0 ? ` Dominant activity is around ${metrics.domText}.` : '';
  return `No clip, hum, drift, or motion threshold crossed in this pass.${dom}`;
}

function compactAISummary(text, limit = 160) {
  const raw = String(text || '').replace(/\s+/g, ' ').trim();
  if (!raw) return '';
  const sentence = raw.match(/^(.{1,200}?[.!?])(\s|$)/)?.[1]?.trim() || '';
  if (sentence && sentence.length <= limit) return sentence;
  if (raw.length <= limit) return raw;
  const clipped = raw.slice(0, Math.max(40, limit - 1));
  const safe = clipped.slice(0, Math.max(24, clipped.lastIndexOf(' ')));
  return `${safe || clipped}...`;
}

function aiVisibleStats(state) {
  if (state?.paused) return S.review.lastStats || pinFrozenReviewStats(state);
  if (S.review.lastStats) return S.review.lastStats;
  if (!state?.filled) return null;
  const viewport = getReviewViewport(state, canvas.width || 1);
  if (!Number(viewport?.viewSamples || 0)) return null;
  return computeSelectionStats(
    {
      startSample: Number(viewport.viewStart || 0),
      endSample: Number(viewport.viewEnd || 0),
    },
    state,
  );
}

function aiDominantSpectrumFrequency(result = S.ai.result, diagnostics = S.diagnostics) {
  const fromResult = Number(result?.measurements?.dominant_frequency_hz || 0);
  if (fromResult > 0) return fromResult;
  const freq = Array.isArray(diagnostics?.spectrum?.freq_hz) ? diagnostics.spectrum.freq_hz : [];
  const mag = Array.isArray(diagnostics?.spectrum?.mag_db) ? diagnostics.spectrum.mag_db : [];
  if (!freq.length || !mag.length) return 0;
  let bestHz = 0;
  let bestDb = -Infinity;
  for (let i = 0; i < Math.min(freq.length, mag.length); i += 1) {
    const hz = Number(freq[i] || 0);
    const db = Number(mag[i] || 0);
    if (hz <= 0.5) continue;
    if (db > bestDb) {
      bestDb = db;
      bestHz = hz;
    }
  }
  return bestHz;
}

function aiStripMeasurements(state = getReviewRenderState(), result = S.ai.result) {
  const stats = aiVisibleStats(state);
  const diagnostics = state?.paused ? (S.review.diagnosticsSnapshot || S.diagnostics) : S.diagnostics;
  const measurement = result?.measurements || {};
  const frozen = !!state?.paused;
  const sampleRate = frozen
    ? Number(state?.sampleRate || measurement.sample_rate_hz || S.sampleRate || 0)
    : Number(measurement.sample_rate_hz || state?.sampleRate || S.sampleRate || 0);
  const viewport = getReviewViewport(state || {}, canvas.width || 1);
  const viewportSpanMs = sampleRate > 0 ? (Number(viewport?.viewSamples || 0) / sampleRate) * 1000 : 0;
  const visibleSpanMs = frozen
    ? Number(viewportSpanMs || measurement.visible_span_ms || 0)
    : Number(measurement.visible_span_ms || viewportSpanMs || 0);
  const rms = frozen
    ? Number(stats?.rms || measurement.rms || 0)
    : Number(measurement.rms || stats?.rms || 0);
  const peakToPeak = frozen
    ? Number(stats?.peakToPeak || measurement.peak_to_peak || 0)
    : Number(measurement.peak_to_peak || stats?.peakToPeak || 0);
  const focusRms = frozen
    ? Number(stats?.focusRms || measurement.focus_rms || 0)
    : Number(measurement.focus_rms || stats?.focusRms || 0);
  const dominantHz = aiDominantSpectrumFrequency(result, diagnostics);
  const focusLabel = frozen
    ? String(stats?.focusLabel || measurement.focus_label || (S.channelLabels[getReviewFocusChannelIndex(state)] || 'CH1'))
    : String(measurement.focus_label || stats?.focusLabel || (S.channelLabels[getReviewFocusChannelIndex(state)] || 'CH1'));
  const units = String(measurement.units || S.signalUnits || 'a.u.');
  return {
    sampleRate,
    visibleSpanMs,
    rms,
    peakToPeak,
    focusRms,
    dominantHz,
    focusLabel,
    units,
    rateText: sampleRate > 0 ? `${sampleRate >= 100 ? sampleRate.toFixed(0) : sampleRate.toFixed(1)} Hz` : '--',
    spanText: visibleSpanMs > 0 ? formatAxisDuration(visibleSpanMs) : '--',
    domText: dominantHz > 0 ? `${dominantHz >= 100 ? dominantHz.toFixed(0) : (dominantHz >= 10 ? dominantHz.toFixed(1) : dominantHz.toFixed(2))} Hz` : '--',
    rmsText: rms ? formatSignalValue(rms) : '--',
    ptpText: peakToPeak ? formatSignalValue(peakToPeak) : '--',
    focusText: `${focusLabel} · ${units}`,
  };
}

function aiFocusReason(result = S.ai.result, metrics = aiStripMeasurements()) {
  const parts = [];
  const topArtifact = Array.isArray(S.review.artifacts) && S.review.artifacts.length ? S.review.artifacts[0] : null;
  if (topArtifact) parts.push(`${topArtifact.label} on ${topArtifact.channelLabel}`);
  else if (result?.artifact_summary?.top_issue && result.artifact_summary.top_issue !== 'clean_window') {
    parts.push(String(result.artifact_summary.top_issue).replace(/_/g, ' '));
  }
  if (metrics?.dominantHz > 0) parts.push(`dominant ${metrics.domText}`);
  if (metrics?.focusRms > 0) parts.push(`focus rms ${formatSignalValue(metrics.focusRms)}`);
  const watch = Array.isArray(result?.research_watchlist) && result.research_watchlist.length ? result.research_watchlist[0] : null;
  if (watch?.kind === 'screening_result') {
    parts.push(`research screen ${String(watch.display_label || watch.label || '').toLowerCase()}`);
  }
  return parts.length ? `Why focus: ${parts.join(' | ')}.` : 'Measurements and routing update here after each AI scan.';
}

function aiQuickSummary(result = S.ai.result, metrics = aiStripMeasurements()) {
  if (!result) return 'Tap AI Scan to inspect the active window.';
  const finding = aiPrimaryFinding(result);
  const pieces = [];
  if (finding?.title) pieces.push(finding.title);
  if (metrics?.dominantHz > 0) pieces.push(`dom ${metrics.domText}`);
  if (metrics?.rmsText && metrics.rmsText !== '--') pieces.push(`rms ${metrics.rmsText}`);
  const watch = Array.isArray(result?.research_watchlist) && result.research_watchlist.length ? result.research_watchlist[0] : null;
  if (watch?.kind === 'screening_result') {
    pieces.push(`screen ${String(watch.display_label || watch.label || '').toLowerCase()} ${Number(watch.score || 0)}%`);
  }
  const action = result?.top_action || (Array.isArray(result?.next_actions) ? result.next_actions[0] : null);
  if (action?.title) pieces.push(`next ${String(action.title).toLowerCase()}`);
  return pieces.length ? pieces.join(' | ') : (compactAISummary(result?.summary, 120) || 'AI scan complete.');
}

function persistAILensState() {
  localStorage.setItem('kyma-ai-lens-minimized', S.ai.lensMinimized ? '1' : '0');
  if (Number.isFinite(S.ai.lensX)) localStorage.setItem('kyma-ai-lens-x', String(Math.round(S.ai.lensX)));
  if (Number.isFinite(S.ai.lensY)) localStorage.setItem('kyma-ai-lens-y', String(Math.round(S.ai.lensY)));
  if (Number.isFinite(S.ai.lensWidth)) localStorage.setItem('kyma-ai-lens-width', String(Math.round(S.ai.lensWidth)));
  if (Number.isFinite(S.ai.lensHeight)) localStorage.setItem('kyma-ai-lens-height', String(Math.round(S.ai.lensHeight)));
}

function positionAILens() {
  const lens = $('ai-scope-hud');
  if (!lens) return;
  const boundsW = Math.max(0, window.innerWidth || document.documentElement.clientWidth || 0);
  const boundsH = Math.max(0, window.innerHeight || document.documentElement.clientHeight || 0);
  const minW = 340;
  const minH = 280;
  const maxW = Math.max(minW, boundsW - 24);
  const maxH = Math.max(minH, boundsH - 24);
  if (!Number.isFinite(S.ai.lensWidth)) S.ai.lensWidth = Math.min(420, maxW);
  if (!Number.isFinite(S.ai.lensHeight)) S.ai.lensHeight = Math.min(520, maxH);
  S.ai.lensWidth = clamp(Number(S.ai.lensWidth || minW), minW, maxW);
  S.ai.lensHeight = clamp(Number(S.ai.lensHeight || minH), minH, maxH);
  lens.style.width = `${Math.round(S.ai.lensWidth)}px`;
  lens.style.height = `${Math.round(S.ai.lensHeight)}px`;
  const maxX = Math.max(12, boundsW - Number(S.ai.lensWidth || minW) - 12);
  const maxY = Math.max(12, boundsH - Number(S.ai.lensHeight || minH) - 12);
  if (!Number.isFinite(S.ai.lensX)) S.ai.lensX = maxX;
  if (!Number.isFinite(S.ai.lensY)) S.ai.lensY = 16;
  S.ai.lensX = clamp(Number(S.ai.lensX || 0), 12, maxX);
  S.ai.lensY = clamp(Number(S.ai.lensY || 0), 12, maxY);
  lens.style.left = `${Math.round(S.ai.lensX)}px`;
  lens.style.top = `${Math.round(S.ai.lensY)}px`;
}

function setAILensMinimized(minimized, { persist = true } = {}) {
  S.ai.lensMinimized = !!minimized;
  if (S.ai.lensMinimized) S.ai.expandUntil = 0;
  if (persist) persistAILensState();
  syncAICopilotUI();
}

window.openAILens = function() {
  setAILensMinimized(false);
  positionAILens();
};

function bindAILens() {
  const lens = $('ai-scope-hud');
  const head = $('ai-scope-head');
  const toggle = $('btn-ai-lens-toggle');
  const minBtn = $('btn-ai-lens-min');
  const resizeHandle = $('ai-scope-resize');
  if (!lens || !head || document.body.dataset.aiLensBound === '1') return;
  document.body.dataset.aiLensBound = '1';

  lens.addEventListener('pointerdown', e => e.stopPropagation());
  toggle?.addEventListener('click', () => {
    setAILensMinimized(!S.ai.lensMinimized);
  });
  minBtn?.addEventListener('click', e => {
    e.preventDefault();
    e.stopPropagation();
    setAILensMinimized(true);
  });

  const startLensDrag = e => {
    if (e.target && (e.target.closest('button') || e.target.closest('#ai-scope-resize'))) return;
    e.preventDefault();
    e.stopPropagation();
    const rect = lens.getBoundingClientRect();
    S.ai.lensDragging = true;
    S.ai.lensDragOffsetX = e.clientX - rect.left;
    S.ai.lensDragOffsetY = e.clientY - rect.top;
  };

  head.addEventListener('pointerdown', startLensDrag);
  lens.addEventListener('pointerdown', startLensDrag);

  resizeHandle?.addEventListener('pointerdown', e => {
    e.preventDefault();
    e.stopPropagation();
    S.ai.lensResizing = true;
    S.ai.lensResizeStartX = e.clientX;
    S.ai.lensResizeStartY = e.clientY;
    S.ai.lensResizeStartW = Number(S.ai.lensWidth || lens.offsetWidth || 420);
    S.ai.lensResizeStartH = Number(S.ai.lensHeight || lens.offsetHeight || 520);
  });

  window.addEventListener('pointermove', e => {
    if (S.ai.lensDragging) {
      S.ai.lensX = e.clientX - Number(S.ai.lensDragOffsetX || 0);
      S.ai.lensY = e.clientY - Number(S.ai.lensDragOffsetY || 0);
      positionAILens();
      return;
    }
    if (!S.ai.lensResizing) return;
    S.ai.lensWidth = Number(S.ai.lensResizeStartW || 420) + (e.clientX - Number(S.ai.lensResizeStartX || 0));
    S.ai.lensHeight = Number(S.ai.lensResizeStartH || 520) + (e.clientY - Number(S.ai.lensResizeStartY || 0));
    positionAILens();
  });

  window.addEventListener('pointerup', () => {
    if (!S.ai.lensDragging && !S.ai.lensResizing) return;
    S.ai.lensDragging = false;
    S.ai.lensResizing = false;
    persistAILensState();
  });
}

function nearestReviewZoom(levels, target) {
  let best = levels[0];
  let error = Math.abs(best - target);
  levels.forEach(level => {
    const delta = Math.abs(level - target);
    if (delta < error) {
      best = level;
      error = delta;
    }
  });
  return best;
}

function autoFitLiveReviewScale() {
  const atDefaultZoom = Math.abs(Number(S.review.zoomY || 1) - 1) < 0.001;
  if (S.review.paused) return;
  if (!S.review.liveAutoScale && !atDefaultZoom) return;
  const now = Date.now();
  if ((now - Number(S.review.lastAutoScaleAt || 0)) < 240) return;
  const state = getReviewRenderState();
  if (!state?.filled) return;
  const viewport = getReviewViewport(state, canvas.width || canvas.clientWidth || 1);
  const start = Number(viewport.viewStart || state.baseAbs || 0);
  const end = Number(viewport.viewEnd || start);
  let peak = 0;
  for (let ch = 0; ch < Math.min(N_CH, state.emg?.length || 0); ch += 1) {
    if (S.channelEnabled[ch] === false) continue;
    const buf = state.emg?.[ch];
    if (!buf) continue;
    for (let sample = start; sample <= end; sample += 1) {
      const idx = bufferIndexForAbsSample(state, sample);
      peak = Math.max(peak, Math.abs(Number(buf[idx] || 0)));
    }
  }
  if (!(peak > 0)) return;
  const fullScale = Math.max(Number(S.signalFullScale || 200), 1e-6);
  const target = clamp((fullScale * 0.40) / peak, REVIEW_Y_ZOOM_LEVELS[0], REVIEW_Y_ZOOM_LEVELS[REVIEW_Y_ZOOM_LEVELS.length - 1]);
  const nextZoom = nearestReviewZoom(REVIEW_Y_ZOOM_LEVELS, target);
  if (Math.abs(Number(S.review.zoomY || 1) - nextZoom) > 0.001) {
    S.review.zoomY = nextZoom;
    syncReviewUI();
  }
  S.review.lastAutoScaleAt = now;
}

function autoAdjustReviewZoomYForRange(startSample, endSample) {
  const state = getReviewRenderState();
  if (!state?.filled) return;
  const focus = getReviewFocusChannelIndex(state);
  const buf = state.emg?.[focus];
  if (!buf) return;
  const rangeStart = Math.max(Number(state.baseAbs || 0), Math.min(Number(startSample || 0), Number(endSample || 0)));
  const rangeEnd = Math.min(Number(state.baseAbs || 0) + Math.max(Number(state.filled || 0) - 1, 0), Math.max(Number(startSample || 0), Number(endSample || 0)));
  let peak = 0;
  for (let sample = rangeStart; sample <= rangeEnd; sample += 1) {
    const idx = bufferIndexForAbsSample(state, sample);
    peak = Math.max(peak, Math.abs(Number(buf[idx] || 0)));
  }
  const fullScale = Math.max(Number(S.signalFullScale || 200), 1);
  const target = peak > 0 ? clamp((fullScale * 0.58) / peak, 0.5, 8) : 1;
  S.review.zoomY = nearestReviewZoom(REVIEW_Y_ZOOM_LEVELS, target);
}

function spotlightReviewRange(startSample, endSample, label = 'AI Focus', kind = 'focus', channel = null) {
  const stateBefore = getReviewRenderState();
  if (!stateBefore.filled) {
    toast('No signal window to spotlight yet.', 'yellow');
    return false;
  }
  if (!S.review.paused) toggleReviewPause(true);
  const state = getReviewRenderState();
  const fullStart = Number(state.baseAbs || 0);
  const fullEnd = fullStart + Math.max(Number(state.filled || 0) - 1, 0);
  const start = clamp(Math.round(Number(startSample)), fullStart, fullEnd);
  const end = clamp(Math.round(Number(endSample ?? startSample)), fullStart, fullEnd);
  const rangeStart = Math.min(start, end);
  const rangeEnd = Math.max(start, end);
  S.review.selection = { startSample: rangeStart, endSample: rangeEnd };
  S.review.viewCenterSample = Math.round((rangeStart + rangeEnd) / 2);
  const sampleSpan = Math.max(1, rangeEnd - rangeStart + 1);
  const zoomTarget = Math.max(1, Math.min((Number(state.filled || 1) / sampleSpan) * 0.65, REVIEW_X_ZOOM_LEVELS[REVIEW_X_ZOOM_LEVELS.length - 1]));
  let zoomIndex = 0;
  let bestError = Number.POSITIVE_INFINITY;
  REVIEW_X_ZOOM_LEVELS.forEach((level, idx) => {
    const error = Math.abs(level - zoomTarget);
    if (error < bestError) {
      bestError = error;
      zoomIndex = idx;
    }
  });
  S.review.zoomX = REVIEW_X_ZOOM_LEVELS[zoomIndex];
  autoAdjustReviewZoomYForRange(rangeStart, rangeEnd);
  S.review.lastStats = computeSelectionStats(S.review.selection, getReviewRenderState());
  const focusChannel = Number.isFinite(channel)
    ? clamp(Number(channel), 0, N_CH - 1)
    : Number.isFinite(S.review.lastStats?.focusChannel)
    ? Number(S.review.lastStats.focusChannel)
    : getReviewFocusChannelIndex(getReviewRenderState());
  if (Number.isFinite(channel) && S.review.lastStats) {
    S.review.lastStats.focusChannel = focusChannel;
    S.review.lastStats.focusLabel = S.channelLabels[focusChannel] || `CH${focusChannel + 1}`;
  }
  S.ai.spotlight = buildSpotlightPayload(rangeStart, rangeEnd, label, kind, focusChannel);
  syncReviewUI();
  syncWorkshopUI();
  return true;
}

function runAIAutopilotPass() {
  const state = getReviewRenderState();
  S.ai.regions = buildAIHighlightRegions(S.ai.result, state);
  const artifacts = Array.isArray(S.review.artifacts) ? S.review.artifacts : [];
  if (artifacts.length) {
    const primary = artifacts[0];
    spotlightReviewRange(primary.startSample, primary.endSample, `AI Focus: ${primary.label}`, primary.kind, primary.channel);
    if ($('review-marker-event')) $('review-marker-event').value = `artifact_${primary.kind}`;
    if ($('review-marker-note')) $('review-marker-note').value = primary.detail || `${primary.label} on ${primary.channelLabel}`;
    return true;
  }
  if (S.ai.regions.length) {
    const primary = S.ai.regions[0];
    S.ai.spotlight = buildSpotlightPayload(
      primary.startSample,
      primary.endSample,
      primary.label,
      primary.kind,
      primary.channel,
    );
    syncReviewUI();
    syncWorkshopUI();
    return true;
  }
  return false;
}

function resolveAIMarkerArtifactIndex(suggestion) {
  const event = String(suggestion?.event || '').toLowerCase();
  const artifacts = Array.isArray(S.review.artifacts) ? S.review.artifacts : [];
  if (!artifacts.length) return -1;
  const matchers = [
    event.includes('clip') ? 'clip' : '',
    event.includes('flat') ? 'flatline' : '',
    event.includes('drift') ? 'drift' : '',
    event.includes('hum') ? 'hum' : '',
  ].filter(Boolean);
  if (matchers.length) {
    const hit = artifacts.findIndex(item => matchers.includes(String(item.kind || '').toLowerCase()));
    if (hit >= 0) return hit;
  }
  return 0;
}

function refreshAICopilotAvailability(state = getReviewRenderState()) {
  const buttons = [$('btn-ai-copilot-run'), $('btn-ai-copilot-run-inline')].filter(Boolean);
  const context = $('ai-copilot-context');
  const hasSignal = !!state?.filled;
  buttons.forEach(btn => {
    btn.disabled = S.ai.loading;
    btn.textContent = S.ai.loading ? 'Scanning...' : (btn.id === 'btn-ai-copilot-run-inline' ? 'AI Scan' : 'Run Copilot');
  });
  if (context) {
    if (!hasSignal) {
      context.textContent = 'No signal window yet. Copilot can still run on the current session state and diagnostics.';
    } else if (S.review.paused && S.review.lastStats) {
      context.textContent = `Using frozen review range on ${S.review.lastStats.focusLabel} (${Number(S.review.lastStats.durationMs || 0).toFixed(1)} ms).`;
    } else if (S.review.paused) {
      context.textContent = 'Using the full frozen review buffer.';
    } else {
      context.textContent = 'Using the current live diagnostics and review context.';
    }
  }
}

function syncAICopilotUILegacy() {
  const shell = $('ai-copilot-shell');
  const status = $('ai-copilot-status');
  const hud = $('ai-scope-hud');
  const stageChip = $('ai-scope-stage-chip');
  const modeChip = $('ai-scope-mode-chip');
  const focusChip = $('ai-scope-focus');
  const autoscaleChip = $('ai-scope-autoscale');
  const scopeLine = $('ai-scope-line');
  const scopeDetailLine = $('ai-scope-detail-line');
  const scopeSummary = $('ai-scope-summary');
  const metricRate = $('ai-metric-rate');
  const metricSpan = $('ai-metric-span');
  const metricDom = $('ai-metric-dom');
  const metricRms = $('ai-metric-rms');
  const metricPtp = $('ai-metric-ptp');
  const metricFocus = $('ai-metric-focus');
  const scopeAction = $('ai-scope-action');
  const scopeWatchlist = $('ai-scope-watchlist');
  const countHum = $('ai-count-hum');
  const countDrift = $('ai-count-drift');
  const countClip = $('ai-count-clip');
  const countArtifact = $('ai-count-artifact');
  const qaScore = $('ai-qa-score');
  const qaGrade = $('ai-qa-grade');
  const qaMode = $('ai-qa-mode');
  const qaFlags = $('ai-qa-flags');
  const qaBar = $('ai-qa-bar');
  const keyInput = $('ai-api-key');
  const keyNote = $('ai-key-note');
  const modelNote = $('ai-model-note');
  const keySaveBtn = $('btn-ai-key-save');
  const keyClearBtn = $('btn-ai-key-clear');
  const result = S.ai.result;
  const signalState = getReviewRenderState();
  const issueCounts = aiIssueCounts(result);
  const metrics = aiStripMeasurements(signalState, result);
  const shouldExpand = !S.ai.lensMinimized;

  refreshAICopilotAvailability(signalState);

  stopAIThinkingLoop();

  if (hud) {
    hud.classList.toggle('is-busy', false);
    hud.classList.toggle('is-expanded', shouldExpand);
  }
  if (shell) shell.classList.toggle('is-busy', false);
  if (status) {
    if (S.ai.loading && !S.ai.background) {
      status.textContent = 'AI scan is reading the visible signal, scoring quality, and selecting a focus region automatically.';
    } else if (S.ai.last_error) {
      status.textContent = `AI scan fell back locally. ${S.ai.last_error}`;
    } else if (Number(S.ai.localModels?.model_count || 0) > 0) {
      status.textContent = 'Hybrid runtime active. Heuristics, local model packs, and optional remote text refinement are all available.';
    } else {
      status.textContent = 'AI scan runs from the scope header, reads the current live or frozen window, adjusts zoom automatically, and spotlights the strongest artifact or marker candidate.';
    }
  }
  if (stageChip) stageChip.textContent = S.ai.loading ? 'scan' : 'ready';
  if (modeChip) modeChip.textContent = aiCopilotModeLabel();
  if (focusChip) focusChip.textContent = aiTopIssueLabel();
  if (autoscaleChip) {
    autoscaleChip.textContent = signalState?.filled
      ? `X ${Number(S.review.zoomX || 1).toFixed(1)}x · Y ${Number(S.review.zoomY || 1).toFixed(1)}x`
      : 'Auto scale ready';
  }
  if (scopeLine) {
    scopeLine.textContent = S.ai.loading
      ? 'Scanning window.'
      : (S.ai.last_error
        ? `Last AI pass fell back locally. ${S.ai.last_error}`
        : (result?.summary
          ? aiQuickSummary(result, metrics)
    : 'Scan to find the strongest region.'));
  }
  if (scopeDetailLine) {
    if (S.ai.loading) {
      scopeDetailLine.textContent = 'Updating measurements, watchlist, and focus guidance.';
    } else {
      scopeDetailLine.textContent = aiFocusReason(result, metrics);
    }
  }
  if (scopeSummary) {
    scopeSummary.textContent = S.ai.loading
      ? 'Scanning the signal field and preparing a focus pass.'
      : (compactAISummary(result?.summary, 150) || 'No AI scan yet.');
  }
  if (metricRate) metricRate.textContent = metrics.rateText;
  if (metricSpan) metricSpan.textContent = metrics.spanText;
  if (metricDom) metricDom.textContent = metrics.domText;
  if (metricRms) metricRms.textContent = metrics.rmsText;
  if (metricPtp) metricPtp.textContent = metrics.ptpText;
  if (metricFocus) metricFocus.textContent = metrics.focusText;
  if (scopeAction) {
    const action = result?.top_action || (Array.isArray(result?.next_actions) ? result.next_actions[0] : null);
    scopeAction.className = 'ai-scope-action';
    if (S.ai.loading) {
      scopeAction.textContent = 'Building next action from the live window.';
    } else if (action) {
      scopeAction.classList.add(String(action.priority || 'medium'));
      scopeAction.textContent = `${String(action.title || 'Next action')}: ${String(action.detail || '').trim()}`;
    } else {
      scopeAction.textContent = 'No action queued.';
    }
  }
  if (scopeWatchlist) {
    scopeWatchlist.innerHTML = '';
    const watchItems = Array.isArray(result?.research_watchlist) ? result.research_watchlist.slice(0, 3) : [];
    if (watchItems.length) {
      watchItems.forEach(item => {
        const pill = document.createElement('span');
        const statusClass = ['active', 'ready', 'standby', 'idle'].includes(String(item?.status || ''))
          ? String(item.status)
          : 'idle';
        pill.className = `ai-scope-watch ${statusClass}`;
        pill.textContent = String(item?.display_label || item?.label || 'Research watch');
        if (item?.detail) pill.title = String(item.detail);
        scopeWatchlist.appendChild(pill);
      });
    } else {
      const pill = document.createElement('span');
      pill.className = 'ai-scope-watch idle';
      pill.textContent = 'No research watchlist yet';
      scopeWatchlist.appendChild(pill);
    }
  }
  if (countHum) countHum.textContent = String(issueCounts.hum || 0);
  if (countDrift) countDrift.textContent = String(issueCounts.drift || 0);
  if (countClip) countClip.textContent = String(issueCounts.clip || 0);
  if (countArtifact) countArtifact.textContent = String(issueCounts.artifact || 0);
  if (qaScore) qaScore.textContent = result?.qa_score ? `${result.qa_score.overall}` : '--';
  if (qaGrade) qaGrade.textContent = result?.qa_score?.grade || '--';
  if (qaMode) qaMode.textContent = S.ai.loading ? 'thinking' : (result?.source || (S.ai.configured ? 'linked' : 'local') || '--');
  if (qaFlags) qaFlags.textContent = result?.artifact_summary ? `${result.artifact_summary.count} flagged` : '--';
  if (keyNote) {
    const sourceLabel = ({
      saved: 'saved locally',
      env: 'loaded from environment',
      none: 'not set',
    })[S.ai.config_source] || S.ai.config_source || 'not set';
    const keyLabel = S.ai.key_hint ? `Key ${S.ai.key_hint}` : 'No API key configured.';
    const storage = S.ai.storage_path ? ` Local path: ${S.ai.storage_path}` : '';
    keyNote.textContent = `${keyLabel} Source: ${sourceLabel}.${storage}`;
  }
  if (modelNote) {
    if (Number(S.ai.localModels?.model_count || 0) > 0) {
      const tasks = (S.ai.localModels.tasks || []).map(task => task.replace(/_/g, ' ')).join(' | ');
      modelNote.textContent = `Local pack ready: ${tasks || 'biosignal runtime'} (${Number(S.ai.localModels.model_count || 0)} loaded).`;
    } else if (Number(S.ai.localModels?.foundation_model_count || 0) > 0 || Number((S.ai.localModels?.builtin_heads || []).length || 0) > 0) {
      const foundationCount = Number(S.ai.localModels?.foundation_model_count || 0);
      const builtinCount = Number((S.ai.localModels?.builtin_heads || []).length || 0);
      modelNote.textContent = `Local runtime ${foundationCount} encoder${foundationCount === 1 ? '' : 's'} + ${builtinCount} built-in head${builtinCount === 1 ? '' : 's'}.`;
    } else if (S.ai.localModels?.last_error) {
      modelNote.textContent = `Local runtime idle. ${S.ai.localModels.last_error}`;
    } else {
      modelNote.textContent = 'No local pack loaded yet.';
    }
  }
  if (keyInput) {
    keyInput.disabled = false;
  }
  if (keySaveBtn) {
    keySaveBtn.disabled = S.ai.loading;
    keySaveBtn.textContent = 'Save Key';
  }
  if (keyClearBtn) {
    keyClearBtn.disabled = S.ai.loading || !S.ai.config_saved;
  }
  if (qaBar) {
    const score = Number(result?.qa_score?.overall || 0);
    qaBar.style.width = `${score}%`;
    qaBar.style.background = aiScoreColor(score);
  }
}

function syncAICopilotUI() {
  const shell = $('ai-copilot-shell');
  const status = $('ai-copilot-status');
  const hud = $('ai-scope-hud');
  const anchor = $('btn-ai-lens-toggle');
  const anchorBadge = $('ai-lens-anchor-badge');
  const anchorText = $('ai-lens-anchor-text');
  const anchorMeta = $('ai-lens-anchor-meta');
  const caption = $('ai-scope-caption');
  const stageChip = $('ai-scope-stage-chip');
  const modeChip = $('ai-scope-mode-chip');
  const focusChip = $('ai-scope-focus');
  const autoscaleChip = $('ai-scope-autoscale');
  const scopeLine = $('ai-scope-line');
  const scopeDetailLine = $('ai-scope-detail-line');
  const scopeSummary = $('ai-scope-summary');
  const metricRate = $('ai-metric-rate');
  const metricSpan = $('ai-metric-span');
  const metricDom = $('ai-metric-dom');
  const metricRms = $('ai-metric-rms');
  const metricPtp = $('ai-metric-ptp');
  const metricFocus = $('ai-metric-focus');
  const scopeAction = $('ai-scope-action');
  const scopeWatchlist = $('ai-scope-watchlist');
  const countHum = $('ai-count-hum');
  const countDrift = $('ai-count-drift');
  const countClip = $('ai-count-clip');
  const countArtifact = $('ai-count-artifact');
  const qaScore = $('ai-qa-score');
  const qaGrade = $('ai-qa-grade');
  const qaMode = $('ai-qa-mode');
  const qaFlags = $('ai-qa-flags');
  const qaBar = $('ai-qa-bar');
  const keyInput = $('ai-api-key');
  const keyNote = $('ai-key-note');
  const modelNote = $('ai-model-note');
  const keySaveBtn = $('btn-ai-key-save');
  const keyClearBtn = $('btn-ai-key-clear');
  const result = S.ai.result;
  const signalState = getReviewRenderState();
  const issueCounts = aiIssueCounts(result);
  const metrics = aiStripMeasurements(signalState, result);
  const finding = aiPrimaryFinding(result);
  const runtimeText = liveMLRuntimeText();
  const shouldShowLens = !S.ai.lensMinimized
    || (!S.ai.background && !!S.ai.loading);

  refreshAICopilotAvailability(signalState);
  syncArtifactRadar(signalState, result);

  stopAIThinkingLoop();

  if (hud) {
    hud.classList.toggle('is-busy', false);
    hud.classList.toggle('is-hidden', !shouldShowLens);
    positionAILens();
  }
  if (anchor) anchor.classList.toggle('is-busy', false);
  if (anchorBadge) anchorBadge.textContent = S.ai.loading ? 'Scan' : 'Lens';
  if (anchorText) anchorText.textContent = S.ai.loading ? 'Scanning window.' : compactAISummary(aiQuickSummary(result, metrics), 56);
  if (anchorMeta) anchorMeta.textContent = S.ai.loading ? 'Live' : (S.ai.lensMinimized ? 'Closed' : 'Open');
  if (shell) shell.classList.toggle('is-busy', false);

  if (status) {
    if (S.ai.loading && !S.ai.background) {
      status.textContent = 'Scanning window.';
    } else if (S.ai.last_error) {
      status.textContent = `Fallback runtime active. ${S.ai.last_error}`;
    } else if (runtimeText) {
      status.textContent = `Live ML active: ${runtimeText}.`;
    } else if (Number(S.ai.localModels?.model_count || 0) > 0) {
      status.textContent = 'Hybrid runtime active.';
    } else {
      status.textContent = 'Header scan. Runtime settings.';
    }
  }

  if (caption) {
    caption.textContent = S.ai.loading
      ? 'Scanning window.'
      : (S.ai.lensMinimized ? 'Use Lens to reopen.' : 'Drag anywhere. Resize corner.');
  }
  if (stageChip) stageChip.textContent = S.ai.loading ? 'scan' : 'ready';
  if (modeChip) modeChip.textContent = aiCopilotModeLabel();
  if (focusChip) focusChip.textContent = S.ai.loading ? 'Scanning' : finding.title;
  if (autoscaleChip) {
    autoscaleChip.textContent = signalState?.filled
      ? `X ${Number(S.review.zoomX || 1).toFixed(1)}x / Y ${Number(S.review.zoomY || 1).toFixed(1)}x`
      : 'Scale ready';
  }
  if (scopeLine) {
    scopeLine.textContent = S.ai.loading
      ? 'Scanning window.'
      : (S.ai.last_error ? `Last AI pass fell back locally. ${S.ai.last_error}` : finding.title);
  }
  if (scopeDetailLine) {
    const detailStrong = scopeDetailLine.querySelector('strong');
    const detailSpan = scopeDetailLine.querySelector('span');
    if (detailStrong) detailStrong.textContent = 'Meaning';
    if (detailSpan) {
      detailSpan.textContent = S.ai.loading
        ? 'Updating focus and measurements.'
        : (runtimeText ? `${aiMeaningText(result, metrics)} Runtime: ${runtimeText}.` : aiMeaningText(result, metrics));
    }
  }
  if (scopeSummary) {
    scopeSummary.textContent = S.ai.loading
      ? 'Measuring the visible window.'
      : (compactAISummary(result?.summary, 110) || finding.detail);
  }
  if (metricRate) metricRate.textContent = metrics.rateText;
  if (metricSpan) metricSpan.textContent = metrics.spanText;
  if (metricDom) metricDom.textContent = metrics.domText;
  if (metricRms) metricRms.textContent = metrics.rmsText;
  if (metricPtp) metricPtp.textContent = metrics.ptpText;
  if (metricFocus) metricFocus.textContent = metrics.focusText;

  if (scopeAction) {
    const action = result?.top_action || (Array.isArray(result?.next_actions) ? result.next_actions[0] : null);
    scopeAction.className = 'ai-scope-action';
    const actionStrong = scopeAction.querySelector('strong');
    const actionSpan = scopeAction.querySelector('span');
    if (actionStrong) actionStrong.textContent = 'Next';
    if (S.ai.loading) {
      if (actionSpan) actionSpan.textContent = 'Building the next action from the live window.';
    } else if (action) {
      scopeAction.classList.add(String(action.priority || 'medium'));
      if (actionSpan) actionSpan.textContent = `${String(action.title || 'Next action')}: ${String(action.detail || '').trim()}`;
    } else {
      if (actionSpan) actionSpan.textContent = 'No action queued.';
    }
  }

  if (scopeWatchlist) {
    scopeWatchlist.innerHTML = '';
    const watchItems = Array.isArray(result?.research_watchlist) ? result.research_watchlist.slice(0, 3) : [];
    if (watchItems.length) {
      watchItems.forEach(item => {
        const pill = document.createElement('span');
        const statusClass = ['active', 'ready', 'standby', 'idle'].includes(String(item?.status || ''))
          ? String(item.status)
          : 'idle';
        pill.className = `ai-scope-watch ${statusClass}`;
        pill.textContent = String(item?.display_label || item?.label || 'Research watch');
        if (item?.detail) pill.title = String(item.detail);
        scopeWatchlist.appendChild(pill);
      });
    } else {
      const pill = document.createElement('span');
      pill.className = 'ai-scope-watch idle';
      pill.textContent = 'No condition-style cue triggered';
      scopeWatchlist.appendChild(pill);
    }
  }

  if (countHum) countHum.textContent = String(issueCounts.hum || 0);
  if (countDrift) countDrift.textContent = String(issueCounts.drift || 0);
  if (countClip) countClip.textContent = String(issueCounts.clip || 0);
  if (countArtifact) countArtifact.textContent = String(issueCounts.artifact || 0);
  if (qaScore) qaScore.textContent = result?.qa_score ? `${result.qa_score.overall}` : '--';
  if (qaGrade) qaGrade.textContent = result?.qa_score?.grade || '--';
  if (qaMode) qaMode.textContent = S.ai.loading ? 'thinking' : (result?.source || (S.ai.configured ? 'linked' : 'local') || '--');
  if (qaFlags) qaFlags.textContent = result?.artifact_summary ? `${result.artifact_summary.count} flagged` : '--';
  if (keyNote) {
    const sourceLabel = ({
      saved: 'saved locally',
      env: 'loaded from environment',
      none: 'not set',
    })[S.ai.config_source] || S.ai.config_source || 'not set';
    const keyLabel = S.ai.key_hint ? `Key ${S.ai.key_hint}` : 'No API key configured.';
    const storage = S.ai.storage_path ? ` Local path: ${S.ai.storage_path}` : '';
    keyNote.textContent = `${keyLabel} Source: ${sourceLabel}.${storage}`;
  }
  if (modelNote) {
    if (Number(S.ai.localModels?.model_count || 0) > 0) {
      const tasks = (S.ai.localModels.tasks || []).map(task => task.replace(/_/g, ' ')).join(' | ');
      modelNote.textContent = `Local model pack ready: ${tasks || 'biosignal runtime'} (${Number(S.ai.localModels.model_count || 0)} loaded).`;
    } else if (runtimeText) {
      modelNote.textContent = `Live biosignal runtime: ${runtimeText}.`;
    } else if (Number(S.ai.localModels?.foundation_model_count || 0) > 0 || Number((S.ai.localModels?.builtin_heads || []).length || 0) > 0) {
      const foundationCount = Number(S.ai.localModels?.foundation_model_count || 0);
      const builtinCount = Number((S.ai.localModels?.builtin_heads || []).length || 0);
      modelNote.textContent = `Local runtime active: ${foundationCount} foundation encoder${foundationCount === 1 ? '' : 's'} and ${builtinCount} built-in artifact head${builtinCount === 1 ? '' : 's'}.`;
    } else if (S.ai.localModels?.last_error) {
      modelNote.textContent = `Local model runtime idle. ${S.ai.localModels.last_error}`;
    } else {
      modelNote.textContent = 'No local model pack loaded. Drop TorchScript packs into server/models or sessions/ai_models to extend the scan.';
    }
  }
  if (keyInput) keyInput.disabled = false;
  if (keySaveBtn) {
    keySaveBtn.disabled = S.ai.loading;
    keySaveBtn.textContent = 'Save Key';
  }
  if (keyClearBtn) keyClearBtn.disabled = S.ai.loading || !S.ai.config_saved;
  if (qaBar) {
    const score = Number(result?.qa_score?.overall || 0);
    qaBar.style.width = `${score}%`;
    qaBar.style.background = aiScoreColor(score);
  }
}

function applyAIMarkerSuggestion(index) {
  const suggestion = Array.isArray(S.ai.result?.suggested_markers) ? S.ai.result.suggested_markers[index] : null;
  if (!suggestion) return;
  if ($('review-marker-event')) $('review-marker-event').value = String(suggestion.event || '');
  if ($('review-marker-note')) $('review-marker-note').value = String(suggestion.note || '');
  toast(`Marker template loaded: ${suggestion.event}`);
}

function spotlightAIMarkerSuggestion(index) {
  const suggestion = Array.isArray(S.ai.result?.suggested_markers) ? S.ai.result.suggested_markers[index] : null;
  if (!suggestion) return;
  applyAIMarkerSuggestion(index);
  const artifactIndex = resolveAIMarkerArtifactIndex(suggestion);
  if (artifactIndex >= 0 && S.review.artifacts?.[artifactIndex]) {
    const item = S.review.artifacts[artifactIndex];
    if (spotlightReviewRange(item.startSample, item.endSample, `AI Focus: ${suggestion.event}`, item.kind || suggestion.event, item.channel)) {
      toast(`Spotlighted ${suggestion.event}`);
      return;
    }
  }
  const stats = S.review.lastStats;
  if (stats && Number.isFinite(stats.startSample) && Number.isFinite(stats.endSample)) {
    if (spotlightReviewRange(stats.startSample, stats.endSample, `AI Focus: ${suggestion.event}`)) {
      toast(`Spotlighted ${suggestion.event}`);
      return;
    }
  }
  if (!S.review.paused && getReviewRenderState().filled) {
    toggleReviewPause(true);
    const state = getReviewRenderState();
    const start = Number(state.baseAbs || 0);
    const end = start + Math.max(Number(state.filled || 0) - 1, 0);
    spotlightReviewRange(start, end, `AI Focus: ${suggestion.event}`);
    toast(`Spotlighted ${suggestion.event}`);
    return;
  }
  toast('No review span available to spotlight yet.', 'yellow');
}

function runAICopilotAction(index) {
  const item = Array.isArray(S.ai.result?.next_actions) ? S.ai.result.next_actions[index] : null;
  if (!item) return;
  const title = String(item.title || '').toLowerCase();
  if (title.includes('freeze')) {
    toggleReviewPause(true);
    const state = getReviewRenderState();
    const start = Number(state.baseAbs || 0);
    const end = start + Math.max(Number(state.filled || 0) - 1, 0);
    spotlightReviewRange(start, end, 'AI Focus');
    toast('Review window frozen for inspection');
    return;
  }
  if (title.includes('workshop') || title.includes('export')) {
    window.switchTab?.('workshop');
    toast('Opened Workshop');
    return;
  }
  if (S.review.artifacts?.length) {
    window.focusReviewArtifact?.(0);
    S.ai.spotlight = {
      ...buildSpotlightPayload(
        Number(S.review.artifacts[0].startSample),
        Number(S.review.artifacts[0].endSample),
        'AI Focus',
        S.review.artifacts[0].kind || 'artifact',
        Number.isFinite(S.review.artifacts[0].channel) ? Number(S.review.artifacts[0].channel) : null,
      ),
    };
    toast('Focused the top artifact candidate');
    return;
  }
  if (S.review.lastStats) {
    spotlightReviewRange(S.review.lastStats.startSample, S.review.lastStats.endSample, 'AI Focus');
    toast('Focused the current review span');
    return;
  }
  toast('Nothing focused yet. Freeze or select a review span first.', 'yellow');
}

function runAIExportAction(index) {
  const item = Array.isArray(S.ai.result?.export_recommendations) ? S.ai.result.export_recommendations[index] : '';
  const text = String(item || '').toLowerCase();
  if (text.includes('matlab')) {
    window.switchTab?.('workshop');
    toast('Opened Workshop for export');
    return;
  }
  if (text.includes('marker')) {
    setAICopilotTab('markers');
    syncAICopilotUI();
    toast('Opened marker suggestions');
    return;
  }
  if (text.includes('workshop')) {
    window.switchTab?.('workshop');
    toast('Opened Workshop');
    return;
  }
  setAICopilotTab('summary');
  syncAICopilotUI();
}

function maybeAutoRunAICopilot() {
  if (document.hidden || S.ai.loading || !S.streaming) return;
  const state = getReviewRenderState();
  const hasSignal = !!state?.filled;
  const hasContext = hasSignal || Number(S.diagnostics?.timing?.window_count || 0) > 0 || !!S.lastPrediction;
  if (!hasContext) return;
  if ((Date.now() - Number(S.ai.lastAutoRunAt || 0)) < Number(S.ai.autoEveryMs || 7000)) return;
  runAICopilot({ silent: true, background: true, autopilot: false });
}

async function runAICopilot({ silent = false, background = false, autopilot = true } = {}) {
  if (S.ai.loading) return;
  if (!background) {
    S.ai.lensMinimized = false;
    persistAILensState();
  }
  S.ai.thinkingStep = 0;
  S.ai.thinkingStartedAt = Date.now();
  S.ai.expandUntil = 0;
  S.ai.changedPulseUntil = 0;
  S.ai.regions = [];
  S.ai.spotlight = null;
  S.ai.loading = true;
  S.ai.background = !!background;
  syncAICopilotUI();
  try {
    const payload = buildAICopilotPayload();
    const res = await post('/api/ai/copilot', payload);
    const nextResult = res.result || null;
    const nextSignature = aiResultSignature(nextResult);
    applyAIStatus(res.ai || {});
    S.ai.result = nextResult;
    S.ai.lastSignature = nextSignature;
    S.ai.lastAutoRunAt = Date.now();
    if (autopilot) runAIAutopilotPass();
    else S.ai.regions = buildAIHighlightRegions(nextResult, getReviewRenderState());
    syncAICopilotUI();
    if (!silent) toast(background ? 'AI scan refreshed quietly' : aiQuickSummary(nextResult, aiStripMeasurements(getReviewRenderState(), nextResult)));
  } catch (e) {
    if (!silent || !background) toast(`Copilot failed: ${e.message}`, 'red');
  } finally {
    S.ai.loading = false;
    S.ai.background = false;
    syncAICopilotUI();
  }
}

function parseRequestedChannel(text) {
  const normalized = String(text || '').toLowerCase();
  const match = normalized.match(/\b(?:channel|ch|muscle)\s*(\d{1,2})\b/);
  if (!match) return -1;
  const idx = Number(match[1]) - 1;
  return Number.isFinite(idx) && idx >= 0 && idx < N_CH ? idx : -1;
}

function pickHumCenterHz() {
  const noise = S.diagnostics?.noise || {};
  const hum50 = Number(noise.hum_50_db || -Infinity);
  const hum60 = Number(noise.hum_60_db || -Infinity);
  return hum60 >= hum50 ? 60 : 50;
}

function stageFilterFields(spec = {}, { preview = true, switchToFilters = true } = {}) {
  if (!spec || !spec.responseType) return Promise.resolve(null);
  if (switchToFilters) window.switchTab?.('filters');
  if ($('filter-method')) $('filter-method').value = String(spec.method || 'butter');
  if ($('filter-response-type')) $('filter-response-type').value = String(spec.responseType || 'bandpass');
  if ($('filter-order')) $('filter-order').value = String(spec.order || 2);
  if ($('filter-cutoff-hz')) $('filter-cutoff-hz').value = spec.cutoffHz != null ? String(spec.cutoffHz) : '';
  if ($('filter-low-hz')) $('filter-low-hz').value = spec.lowHz != null ? String(spec.lowHz) : '';
  if ($('filter-high-hz')) $('filter-high-hz').value = spec.highHz != null ? String(spec.highHz) : '';
  if ($('filter-rp-db')) $('filter-rp-db').value = String(spec.rpDb != null ? spec.rpDb : 1);
  if ($('filter-rs-db')) $('filter-rs-db').value = String(spec.rsDb != null ? spec.rsDb : 40);
  if ($('filter-apply-mode')) $('filter-apply-mode').value = String(spec.applyMode || 'append');
  if ($('filter-name')) $('filter-name').value = String(spec.name || 'Copilot Cleanup');
  updateFilterFieldVisibility();
  refreshFilterLabUI();
  return preview ? previewFilterDesign().then(() => spec) : Promise.resolve(spec);
}

function recommendedFilterForFinding(finding) {
  const kind = normalizeArtifactKind(finding?.kind || '');
  const hz = pickHumCenterHz();
  const emg = String(S.signalProfileKey || '').toLowerCase() === 'emg';
  if (kind === 'hum') {
    return {
      mode: 'filter',
      responseType: 'bandstop',
      method: 'butter',
      order: 2,
      lowHz: hz - 2,
      highHz: hz + 2,
      applyMode: 'append',
      name: `Hum Notch ${hz}Hz`,
      label: `${hz} Hz notch`,
    };
  }
  if (kind === 'drift') {
    const cutoff = emg ? 20 : 1;
    return {
      mode: 'filter',
      responseType: 'highpass',
      method: 'butter',
      order: 2,
      cutoffHz: cutoff,
      applyMode: 'append',
      name: `Baseline Drift Cleanup ${cutoff}Hz`,
      label: `${cutoff} Hz high-pass`,
    };
  }
  if (kind === 'motion_artifact' || kind === 'spike') {
    return {
      mode: 'filter',
      responseType: 'bandpass',
      method: 'butter',
      order: 3,
      lowHz: emg ? 20 : 1,
      highHz: emg ? 120 : Math.max(10, Math.round(Number(S.sampleRate || 250) * 0.35)),
      applyMode: 'append',
      name: emg ? 'Motion Cleanup Bandpass' : 'Cleanup Bandpass',
      label: emg ? '20-120 Hz bandpass' : 'Bandpass cleanup',
    };
  }
  if (kind === 'clip') {
    return { mode: 'hardware', label: 'reduce gain / amplitude' };
  }
  if (kind === 'flatline' || kind === 'low_signal') {
    return { mode: 'setup', label: 're-seat electrode / check contact' };
  }
  return {
    mode: 'inspect',
    label: 'inspect and mark region',
  };
}

function formatFindingRangeLabel(finding, state = getReviewRenderState()) {
  if (finding && finding.localized === false) return 'full span';
  const startMs = reviewSampleOffsetMs(Number(finding?.startSample || 0), state);
  const endMs = reviewSampleOffsetMs(Number(finding?.endSample || finding?.startSample || 0), state);
  return `${startMs.toFixed(0)}-${endMs.toFixed(0)} ms`;
}

function getScopeCopilotNoiseFindings(state = getReviewRenderState(), result = S.ai.result) {
  const findings = artifactRadarSegments(state, result)
    .map((item) => {
      const recommendation = recommendedFilterForFinding(item);
      return {
        ...item,
        recommendation,
        recommendationLabel: recommendation?.label || 'inspect region',
        rangeLabel: formatFindingRangeLabel(item, state),
      };
    })
    .sort((a, b) => {
      if (Boolean(a.localized) !== Boolean(b.localized)) return a.localized ? -1 : 1;
      const conf = Number(b.confidence || 0) - Number(a.confidence || 0);
      if (Math.abs(conf) > 0.001) return conf;
      return (Number(a.startSample || 0) - Number(b.startSample || 0));
    });
  return findings.slice(0, 6);
}

function focusScopeCopilotFinding(finding, { updateStatus = true } = {}) {
  if (!finding) return false;
  if (!finding.localized) {
    if (updateStatus) {
      S.copilot.status = `${finding.label} is elevated across the visible span, but there is no tight region to bracket yet.`;
      S.copilot.steps = ['ranked noise regions', 'no localized span found', `recommended ${finding.recommendationLabel}`];
      syncScopeCopilotUI();
    }
    return false;
  }
  const focused = spotlightReviewRange(finding.startSample, finding.endSample, `${finding.label} Focus`, finding.kind, finding.channel);
  if (!focused) return false;
  placeReviewCursorsForRange(finding.startSample, finding.endSample);
  if ($('review-marker-event')) $('review-marker-event').value = `artifact_${normalizeArtifactKind(finding.kind)}`;
  if ($('review-marker-note')) $('review-marker-note').value = `${finding.label} | ${finding.detail || finding.recommendationLabel}`;
  if (updateStatus) {
    S.copilot.status = `${finding.label} isolated at ${finding.rangeLabel}.`;
    S.copilot.steps = ['ranked noise regions', `selected ${String(finding.label || '').toLowerCase()}`, `recommended ${finding.recommendationLabel}`];
    syncScopeCopilotUI();
  }
  return true;
}

async function stageRecommendedFilterForFinding(finding, { preview = true, switchToFilters = true } = {}) {
  if (!finding?.recommendation) return null;
  const recommendation = finding.recommendation;
  if (recommendation.mode !== 'filter') return recommendation;
  await stageFilterFields(recommendation, { preview, switchToFilters });
  return recommendation;
}

function placeReviewCursorsForRange(startSample, endSample) {
  const state = getReviewRenderState();
  const width = canvas.width || 1;
  setReviewCursor('a', Number(startSample), state, width);
  setReviewCursor('b', Number(endSample), state, width);
  syncReviewUI();
}

async function stageNoiseCleanupFilter({ preview = true, switchToFilters = true } = {}) {
  const hz = pickHumCenterHz();
  if (switchToFilters) window.switchTab?.('filters');
  if ($('filter-method')) $('filter-method').value = 'butter';
  if ($('filter-response-type')) $('filter-response-type').value = 'bandstop';
  if ($('filter-order')) $('filter-order').value = '2';
  if ($('filter-low-hz')) $('filter-low-hz').value = String(hz - 2);
  if ($('filter-high-hz')) $('filter-high-hz').value = String(hz + 2);
  if ($('filter-apply-mode')) $('filter-apply-mode').value = 'append';
  if ($('filter-name')) $('filter-name').value = `Noise Cleanup ${hz}Hz`;
  updateFilterFieldVisibility();
  refreshFilterLabUI();
  if (preview) await previewFilterDesign();
  return hz;
}

function buildScopeCopilotActions({ channelIndex = -1, noise = false, artifact = null } = {}) {
  const findings = getScopeCopilotNoiseFindings();
  const actions = [];
  if (channelIndex >= 0) {
    actions.push({
      label: 'Reset View',
      tone: '',
      run: () => {
        setCopilotChannelFocus(-1, { active: false });
        S.copilot.status = 'Restored the full stack.';
        S.copilot.steps = ['restored full stack'];
        S.copilot.actions = [];
        syncScopeCopilotUI();
      },
    });
  }
  actions.push({
    label: 'Rescan',
    tone: 'primary',
    run: async () => {
      await runAICopilot({ silent: true, background: false, autopilot: true });
      const result = S.ai.result;
      S.copilot.status = compactAISummary(result?.summary, 140) || 'Scan complete.';
      S.copilot.steps = ['measured window', 'updated findings', 'prepared next action'];
      syncScopeCopilotUI();
    },
  });
  if (noise || artifact || Number(aiIssueCounts(S.ai.result).hum || 0) > 0) {
    findings.slice(0, 3).forEach((finding, idx) => {
      actions.push({
        label: idx === 0 ? `${finding.localized ? 'Focus' : 'Inspect'} ${finding.label}` : `${finding.label} ${idx + 1}`,
        tone: idx === 0 ? 'warn' : '',
        run: () => {
          if (!focusScopeCopilotFinding(finding)) {
            toast(`${finding.label} is a full-span issue right now. Use the filter suggestion first.`, 'yellow');
          }
        },
      });
    });
    actions.push({
      label: 'Spotlight Noise',
      tone: 'warn',
      run: () => {
        const candidate = findings.find(item => item.localized) || artifact;
        if (candidate && focusScopeCopilotFinding(candidate)) {
          return;
        }
        if (S.review.artifacts?.length) {
          const first = S.review.artifacts[0];
          spotlightReviewRange(first.startSample, first.endSample, `Noise Focus: ${first.label}`, first.kind, first.channel);
          placeReviewCursorsForRange(first.startSample, first.endSample);
          S.copilot.status = `${first.label} highlighted and bracketed.`;
          S.copilot.steps = ['locked noisy span', 'placed A/B markers', 'ready for cleanup'];
          syncScopeCopilotUI();
          return;
        }
        toast('No localized noise segment yet.', 'yellow');
      },
    });
    actions.push({
      label: findings[0]?.recommendation?.mode === 'filter'
        ? findings[0].recommendation.label
        : 'Cleanup Advice',
      tone: 'primary',
      run: async () => {
        const candidate = findings.find(item => item.localized) || findings[0] || artifact;
        if (!candidate) {
          const hz = await stageNoiseCleanupFilter({ preview: true, switchToFilters: true });
          S.copilot.status = `${hz} Hz notch staged in Filters with a live preview.`;
          S.copilot.steps = ['read hum energy', `staged ${hz} hz bandstop`, 'opened filter workspace'];
          syncScopeCopilotUI();
          return;
        }
        const recommendation = await stageRecommendedFilterForFinding(candidate, { preview: true, switchToFilters: true });
        if (recommendation?.mode === 'filter') {
          S.copilot.status = `${candidate.label} mapped to ${recommendation.label}. Preview is staged in Filters.`;
          S.copilot.steps = ['ranked noisy regions', `picked ${candidate.label.toLowerCase()}`, `staged ${recommendation.label}`];
        } else {
          S.copilot.status = `${candidate.label} does not map to a safe filter pass. Recommended action: ${candidate.recommendationLabel}.`;
          S.copilot.steps = ['ranked noisy regions', `picked ${candidate.label.toLowerCase()}`, 'returned setup guidance'];
        }
        syncScopeCopilotUI();
      },
    });
    actions.push({
      label: 'Mark Noise',
      tone: '',
      run: async () => {
        const candidate = findings.find(item => item.localized) || artifact;
        if (candidate) {
          spotlightReviewRange(candidate.startSample, candidate.endSample, `Noise Marker: ${candidate.label}`, candidate.kind, candidate.channel);
          if ($('review-marker-event')) $('review-marker-event').value = `artifact_${normalizeArtifactKind(candidate.kind)}`;
          if ($('review-marker-note')) $('review-marker-note').value = candidate.detail || `${candidate.label} on ${candidate.channelLabel}`;
          await saveReviewMarker();
          S.copilot.status = 'Noise marker saved on the highlighted region.';
          S.copilot.steps = ['focused artifact', 'loaded marker fields', 'saved review marker'];
          syncScopeCopilotUI();
          return;
        }
        toast('No noise region is active yet.', 'yellow');
      },
    });
  }
  return actions;
}

async function handleScopeCopilotCommand(rawText = '') {
  const text = String(rawText || '').trim();
  if (!text) return;
  S.copilot.input = text;
  S.copilot.lastCommand = text;
  S.copilot.steps = ['reading command'];
  S.copilot.actions = [];
  S.copilot.findings = [];
  syncScopeCopilotUI();
  const normalized = text.toLowerCase();
  const channelIndex = parseRequestedChannel(normalized);
  const wantsNoise = /\b(noise|hum|drift|artifact|clip|dirty|cleanup|clean up|filter)\b/.test(normalized);
  const wantsMarker = /\b(marker|mark|segment|discretize|locate)\b/.test(normalized);
  const wantsReset = /\b(reset|show all|clear focus|unfocus)\b/.test(normalized);
  const wantsCompare = /\b(compare|versus|vs)\b/.test(normalized);

  if (wantsReset) {
    setCopilotChannelFocus(-1, { active: false });
    S.copilot.status = 'Restored the full channel stack.';
    S.copilot.steps = ['cleared focus', 'reset stack emphasis'];
    S.copilot.actions = [];
    S.copilot.findings = [];
    syncScopeCopilotUI();
    return;
  }

  if (channelIndex >= 0) {
    setCopilotChannelFocus(channelIndex, { label: channelDisplayLabel(channelIndex), active: true });
  }

  const shouldScan = /\b(analy[sz]e|scan|inspect|find|show|noise|artifact|filter|mark|segment)\b/.test(normalized);
  if (shouldScan) {
    await runAICopilot({ silent: true, background: false, autopilot: true });
  }

  refreshReviewArtifacts(getReviewRenderState());
  const findings = getScopeCopilotNoiseFindings();
  S.copilot.findings = findings;
  const artifact = findings.find(item => item.localized) || (Array.isArray(S.review.artifacts) && S.review.artifacts.length ? S.review.artifacts[0] : null);

  if (wantsCompare) {
    S.copilot.status = 'Compare mode is next. For now I can focus one channel, spotlight artifacts, and stage cleanup filters.';
    S.copilot.steps = ['parsed compare request', 'kept current channel view'];
    S.copilot.actions = buildScopeCopilotActions({ channelIndex, noise: wantsNoise, artifact });
    syncScopeCopilotUI();
    return;
  }

  if (wantsMarker && artifact) {
    focusScopeCopilotFinding(artifact, { updateStatus: false });
    S.copilot.status = `${artifact.label} isolated. Marker fields are ready.`;
    S.copilot.steps = ['ranked noisy regions', 'placed A/B markers', 'staged marker fields'];
    S.copilot.actions = buildScopeCopilotActions({ channelIndex, noise: true, artifact });
    syncScopeCopilotUI();
    return;
  }

  if (wantsNoise) {
    if (artifact) {
      focusScopeCopilotFinding(artifact, { updateStatus: false });
      const total = findings.length || 1;
      const recommendation = artifact.recommendationLabel || 'inspect region';
      S.copilot.status = `${total} noise region${total === 1 ? '' : 's'} flagged. ${artifact.label} is selected first and ${recommendation} is recommended.`;
      S.copilot.steps = ['scanned active span', `ranked ${total} noisy region${total === 1 ? '' : 's'}`, 'prepared cleanup actions'];
    } else if (findings.length) {
      const primary = findings[0];
      S.copilot.status = `${primary.label} is elevated across the visible span. I did not bracket the whole window; use ${primary.recommendationLabel} first or freeze a noisier segment.`;
      S.copilot.steps = ['scanned active span', 'found full-span issue', 'prepared cleanup actions'];
    } else {
      const finding = aiPrimaryFinding(S.ai.result);
      S.copilot.status = `${finding.title}. I staged cleanup options even though no tight region was localized yet.`;
      S.copilot.steps = ['scanned active span', 'read artifact head', 'prepared cleanup actions'];
    }
    S.copilot.actions = buildScopeCopilotActions({ channelIndex, noise: true, artifact });
    syncScopeCopilotUI();
    return;
  }

  if (channelIndex >= 0) {
    const label = channelDisplayLabel(channelIndex);
    const state = getReviewRenderState();
    const channelRms = Number((state.paused ? (state.rms || [])[channelIndex] : (S.rms || [])[channelIndex]) || 0);
    S.copilot.status = `${label} is now in focus. Current channel RMS is ${formatSignalValue(channelRms)} and the rest of the stack is dimmed.`;
    S.copilot.steps = ['parsed channel request', `focused ${label.toLowerCase()}`, 'read live channel rms'];
    S.copilot.actions = buildScopeCopilotActions({ channelIndex, noise: false, artifact });
    syncScopeCopilotUI();
    return;
  }

  S.copilot.status = compactAISummary(S.ai.result?.summary, 150) || 'Scan complete.';
  S.copilot.steps = ['measured window', 'updated findings', 'ready for next action'];
  S.copilot.actions = buildScopeCopilotActions({ channelIndex, noise: wantsNoise, artifact });
  syncScopeCopilotUI();
}

function runScopeCopilotAction(index) {
  const action = Array.isArray(S.copilot.actions) ? S.copilot.actions[Number(index)] : null;
  if (!action || typeof action.run !== 'function') return;
  Promise.resolve(action.run()).catch((e) => toast(e.message, 'red'));
}

function runScopeCopilotFinding(index) {
  const finding = Array.isArray(S.copilot.findings) ? S.copilot.findings[Number(index)] : null;
  if (!finding) return;
  focusScopeCopilotFinding(finding);
}

window.handleScopeCopilotCommand = handleScopeCopilotCommand;
window.runScopeCopilotAction = runScopeCopilotAction;
window.runScopeCopilotFinding = runScopeCopilotFinding;

async function saveAIKey() {
  const input = $('ai-api-key');
  const apiKey = String(input?.value || '').trim();
  if (!apiKey) {
    toast('Enter an API key first', 'yellow');
    return;
  }
  S.ai.loading = true;
  syncAICopilotUI();
  try {
    const res = await post('/api/ai/config', { api_key: apiKey });
    applyAIStatus(res.ai || {});
    if (input) input.value = '';
    syncAICopilotUI();
    toast('API key saved locally');
  } catch (e) {
    toast(`Save failed: ${e.message}`, 'red');
  } finally {
    S.ai.loading = false;
    syncAICopilotUI();
  }
}

async function clearAIKey() {
  S.ai.loading = true;
  syncAICopilotUI();
  try {
    const res = await post('/api/ai/config/clear', {});
    applyAIStatus(res.ai || {});
    if ($('ai-api-key')) $('ai-api-key').value = '';
    syncAICopilotUI();
    toast('Saved API key cleared', 'yellow');
  } catch (e) {
    toast(`Clear failed: ${e.message}`, 'red');
  } finally {
    S.ai.loading = false;
    syncAICopilotUI();
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
    return 'Sessions, protocols, and tags.';
  }
  if (name === 'train') {
    return 'Fit, datasets, and eval.';
  }
  if (name === 'control') {
    return 'Outputs and actuation.';
  }
  return 'Live signal, review, and control.';
}

function syncWorkspaceUI() {
  const current = S.dashboardWorkspace || 'live';
  document.body.classList.remove('workspace-live', 'workspace-record', 'workspace-train', 'workspace-control');
  document.body.classList.add(`workspace-${current}`);
  document.querySelectorAll('.workspace-btn').forEach(btn => {
    btn.classList.toggle('active', btn.id === `workspace-${current}`);
  });
  const activeShell = getActiveShellTab();
  const workspaceTabMap = {
    live: 'tab-dashboard',
    record: 'tab-blocks',
    train: 'tab-filters',
    control: 'tab-dashboard',
  };
  if (activeShell === 'dashboard') {
    ['tab-dashboard', 'tab-blocks', 'tab-filters', 'tab-workshop', 'tab-code', 'tab-firmware'].forEach(id => {
      const btn = $(id);
      if (btn) btn.classList.toggle('active', id === (workspaceTabMap[current] || 'tab-dashboard'));
    });
  }
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
  if (current === 'live') {
    ['#card-signal-type', '#card-stream', '#card-signal-logic', '#card-decoded-output', '#card-channel-activity', '#card-review-markers'].forEach(selector => {
      setCardExpanded(selector, true);
    });
  } else if (current === 'record') {
    ['#card-signal-type', '#card-stream', '#card-session-metadata', '#card-protocol-template', '#card-sessions', '#card-review-markers', '#card-decoded-output', '#card-channel-activity', '#card-timing-safety'].forEach(selector => {
      setCardExpanded(selector, true);
    });
  } else if (current === 'train') {
    ['#card-signal-type', '#card-training', '#card-research', '#card-sessions', '#card-subjects', '#card-decoded-output', '#card-timing-safety'].forEach(selector => {
      setCardExpanded(selector, true);
    });
  } else if (current === 'control') {
    ['#card-signal-type', '#card-stream', '#card-signal-logic', '#card-lsl', '#card-osc', '#card-decoded-output', '#card-timing-safety'].forEach(selector => {
      setCardExpanded(selector, true);
    });
  }
}

window.switchWorkspace = function(name) {
  const wanted = ['live', 'record', 'train', 'control'].includes(name) ? name : 'live';
  S.dashboardWorkspace = wanted;
  localStorage.setItem('kyma-dashboard-workspace', wanted);
  syncWorkspaceUI();
};

window.switchPipelineTab = function() {
  window.switchTab('dashboard');
  window.switchWorkspace('live');
  if (typeof togglePipelineBuilder === 'function') togglePipelineBuilder(true);
  ['tab-dashboard', 'tab-blocks', 'tab-filters', 'tab-workshop', 'tab-code', 'tab-firmware'].forEach(id => {
    const btn = $(id);
    if (btn) btn.classList.toggle('active', id === 'tab-workshop');
  });
};

function getActiveShellTab() {
  if ($('tab-blocks')?.classList.contains('active')) return 'blocks';
  if ($('tab-filters')?.classList.contains('active')) return 'filters';
  if ($('tab-workshop')?.classList.contains('active')) return 'workshop';
  if ($('tab-bench')?.classList.contains('active')) return 'bench';
  if ($('tab-firmware')?.classList.contains('active')) return 'firmware';
  if ($('tab-hwdocs')?.classList.contains('active')) return 'hwdocs';
  return 'dashboard';
}

function getCommandPaletteActions() {
  const streamTitle = S.streaming ? 'Stop stream' : 'Start stream';
  const freezeTitle = S.review.paused ? 'Resume review' : 'Freeze review';
  return [
    { title: 'Dashboard', subtitle: 'Open the live shell and inspectors.', tag: 'Tab', run: () => window.switchTab('dashboard') },
    { title: 'Blocks', subtitle: 'Open the node graph editor.', tag: 'Tab', run: () => window.switchTab('blocks') },
    { title: 'Filters', subtitle: 'Open the filter design workspace.', tag: 'Tab', run: () => window.switchTab('filters') },
    { title: 'Workshop', subtitle: 'Open frozen chunk DSP analysis.', tag: 'Tab', run: () => window.switchTab('workshop') },
    { title: 'Bench', subtitle: 'Open the engineering report workspace.', tag: 'Tab', run: () => window.switchTab('bench') },
    { title: 'Firmware', subtitle: 'Open the firmware editor.', tag: 'Tab', run: () => window.switchTab('firmware') },
    { title: 'Guide', subtitle: 'Open the hardware guide.', tag: 'Tab', run: () => window.switchTab('hwdocs') },
    { title: 'Live workspace', subtitle: 'Switch dashboard mode to live signal work.', tag: 'Mode', run: () => { window.switchTab('dashboard'); window.switchWorkspace('live'); } },
    { title: 'Record workspace', subtitle: 'Switch dashboard mode to sessions and protocols.', tag: 'Mode', run: () => { window.switchTab('dashboard'); window.switchWorkspace('record'); } },
    { title: 'Train workspace', subtitle: 'Switch dashboard mode to fit and datasets.', tag: 'Mode', run: () => { window.switchTab('dashboard'); window.switchWorkspace('train'); } },
    { title: 'Control workspace', subtitle: 'Switch dashboard mode to outputs and actuation.', tag: 'Mode', run: () => { window.switchTab('dashboard'); window.switchWorkspace('control'); } },
    { title: 'AI scan', subtitle: 'Run the copilot on the active live or frozen window.', tag: 'Run', run: () => { window.switchTab('dashboard'); window.switchWorkspace('live'); runAICopilot({ silent: false, background: false, autopilot: true }); } },
    { title: 'Build pipeline', subtitle: 'Turn a biosignal prompt into filters, labels, models, and exports.', tag: 'AI', run: () => { window.switchTab('dashboard'); window.switchWorkspace('live'); togglePipelineBuilder(true); } },
    { title: 'Open AI lens', subtitle: 'Show the floating AI lens panel.', tag: 'Panel', run: () => { window.switchTab('dashboard'); window.switchWorkspace('live'); window.openAILens?.(); } },
    { title: freezeTitle, subtitle: 'Pause or resume the review buffer.', tag: 'Review', run: () => { window.switchTab('dashboard'); window.switchWorkspace('live'); toggleReviewPause(); } },
    { title: streamTitle, subtitle: 'Toggle the active source stream.', tag: 'Stream', run: () => { window.switchTab('dashboard'); $('btn-stream')?.click(); } },
    { title: 'Home arm', subtitle: 'Send the arm back to neutral.', tag: 'Run', run: () => $('btn-home')?.click() },
    { title: 'Tidy graph', subtitle: 'Auto-layout the active blocks graph.', tag: 'Blocks', run: () => { window.switchTab('blocks'); window.autoLayoutActiveProgram?.(); } },
    { title: 'Start tour', subtitle: 'Launch the guided product tour.', tag: 'Help', run: () => window.startQuickTour?.() },
  ];
}

function getFilteredCommandPaletteItems(query = '') {
  const q = String(query || '').trim().toLowerCase();
  const actions = getCommandPaletteActions();
  if (!q) return actions;
  return actions.filter((item) => {
    const haystack = `${item.title} ${item.subtitle} ${item.tag}`.toLowerCase();
    return haystack.includes(q);
  });
}

function renderCommandPalette() {
  const list = $('command-palette-results');
  const empty = $('command-palette-empty');
  if (!list || !empty) return;

  COMMAND_PALETTE.items = getFilteredCommandPaletteItems(COMMAND_PALETTE.query);
  if (COMMAND_PALETTE.index >= COMMAND_PALETTE.items.length) {
    COMMAND_PALETTE.index = Math.max(0, COMMAND_PALETTE.items.length - 1);
  }

  list.innerHTML = '';
  COMMAND_PALETTE.items.forEach((item, index) => {
    const row = document.createElement('button');
    row.className = `command-item${index === COMMAND_PALETTE.index ? ' active' : ''}`;
    row.type = 'button';
    row.innerHTML = `
      <div class="command-copy">
        <div class="command-title">${item.title}</div>
        <div class="command-subtitle">${item.subtitle}</div>
      </div>
      <span class="command-tag">${item.tag}</span>
    `;
    row.onclick = () => runCommandPaletteItem(index);
    list.appendChild(row);
  });
  empty.classList.toggle('active', COMMAND_PALETTE.items.length === 0);
}

function openCommandPalette(prefill = '') {
  const root = $('command-palette');
  const input = $('command-palette-input');
  if (!root || !input) return;
  COMMAND_PALETTE.open = true;
  COMMAND_PALETTE.query = String(prefill || '');
  COMMAND_PALETTE.index = 0;
  root.classList.add('active');
  input.value = COMMAND_PALETTE.query;
  renderCommandPalette();
  window.requestAnimationFrame(() => input.focus());
}

function closeCommandPalette() {
  const root = $('command-palette');
  const input = $('command-palette-input');
  if (!root) return;
  COMMAND_PALETTE.open = false;
  root.classList.remove('active');
  COMMAND_PALETTE.query = '';
  COMMAND_PALETTE.index = 0;
  if (input) input.value = '';
}

function runCommandPaletteItem(index = COMMAND_PALETTE.index) {
  const item = COMMAND_PALETTE.items[index];
  if (!item) return;
  closeCommandPalette();
  try {
    const result = item.run();
    if (result && typeof result.then === 'function') {
      result.catch((e) => toast(e.message || 'Command failed', 'red'));
    }
  } catch (e) {
    toast(e.message || 'Command failed', 'red');
  }
}

function bindCommandPalette() {
  if (document.body.dataset.commandPaletteBound === '1') return;
  document.body.dataset.commandPaletteBound = '1';

  const root = $('command-palette');
  const input = $('command-palette-input');
  const trigger = $('btn-command-palette');
  if (!root || !input || !trigger) return;

  trigger.onclick = () => openCommandPalette();
  root.addEventListener('pointerdown', (e) => {
    if (e.target === root) closeCommandPalette();
  });
  input.addEventListener('input', () => {
    COMMAND_PALETTE.query = input.value;
    COMMAND_PALETTE.index = 0;
    renderCommandPalette();
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      COMMAND_PALETTE.index = Math.min(COMMAND_PALETTE.index + 1, Math.max(0, COMMAND_PALETTE.items.length - 1));
      renderCommandPalette();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      COMMAND_PALETTE.index = Math.max(0, COMMAND_PALETTE.index - 1);
      renderCommandPalette();
    } else if (e.key === 'Enter') {
      e.preventDefault();
      runCommandPaletteItem();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      closeCommandPalette();
    }
  });

  document.addEventListener('keydown', (e) => {
    const isCommandKey = (e.ctrlKey || e.metaKey) && !e.altKey;
    if (isCommandKey && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      if (COMMAND_PALETTE.open) closeCommandPalette();
      else openCommandPalette();
      return;
    }
    if (!COMMAND_PALETTE.open) return;
    if (e.key === 'Escape') {
      e.preventDefault();
      closeCommandPalette();
    }
  });
}

function getTourFactory() {
  return window.driver?.js?.driver || null;
}

function setCardExpanded(target, expanded = true) {
  const card = typeof target === 'string' ? document.querySelector(target) : target;
  if (!card) return false;
  card.classList.toggle('collapsed', !expanded);
  return true;
}

function compactCodeDashboard() {
  S.dashboardWorkspace = 'live';
  localStorage.setItem('kyma-dashboard-workspace', 'live');
  syncWorkspaceUI();
  moveCodeInspectCards(true);
  document.querySelectorAll('#left-panel .card-collapsible, #right-panel .card-collapsible').forEach(card => {
    card.classList.add('collapsed');
  });
  ['#card-signal-type', '#card-stream', '#card-decoded-output', '#card-channel-activity', '#card-timing-safety', '#card-review-markers'].forEach(selector => {
    const el = document.querySelector(selector);
    if (el) el.classList.remove('workspace-hidden');
  });
}

function moveCodeInspectCards(intoLeft) {
  const left = $('left-panel');
  const right = $('right-panel');
  if (!left || !right) return;
  const ids = ['card-decoded-output', 'card-channel-activity', 'card-timing-safety', 'card-review-markers'];
  ids.forEach(id => {
    const card = $(id);
    if (!card) return;
    if (!card.dataset.originalPanel) card.dataset.originalPanel = 'right-panel';
    if (intoLeft && card.parentElement !== left) {
      card.dataset.codeMoved = '1';
      left.appendChild(card);
    } else if (!intoLeft && card.dataset.codeMoved === '1' && card.parentElement !== right) {
      right.appendChild(card);
      delete card.dataset.codeMoved;
    }
  });
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

function preparePipelineTourView({ registry = false } = {}) {
  prepareTourView({
    tab: 'dashboard',
    workspace: 'live',
    viz: S.signalProfileKey === 'eeg' ? 'eeg' : 'hand',
    expand: ['#card-signal-type', '#card-stream', '#card-decoded-output', '#card-review-markers'],
  });
  if (typeof togglePipelineBuilder === 'function') togglePipelineBuilder(true);
  if (registry && typeof runPipelineProjectList === 'function') return runPipelineProjectList();
  return null;
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
        title: 'Marker Test',
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
        title: 'Fit Models',
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
        description: 'Blocks handles automation, Filters handles DSP, Workshop analyzes frozen chunks, and Guide covers wiring and firmware. Each tab has its own tour.',
      },
    },
  ]);

  return tour;
}

function createCurrentWorkflowTour() {
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
        title: 'KYMA Prompt Pipeline',
        description: 'This tutorial follows the current workflow: choose a source, prompt the pipeline, inspect data, train or rebuild, then export a report and package.',
        nextBtnText: 'Start Tutorial',
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
        title: 'Workflow Stages',
        description: 'Live is where the signal and AI pipeline start. Record, Train, and Control support capture, model validation, and deployment after the project is saved.',
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
        title: 'Pick The Biosignal',
        description: 'Choose EMG, EEG, ECG, EOG, or another profile first. KYMA uses this to set labels, filters, model hints, QA checks, and output assumptions.',
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
        title: 'Choose Live Or Dataset',
        description: 'Use live hardware, synthetic, playback, LSL, OSC, serial, or a dataset path. Import supports CSV, NPZ, EDF/BDF, XDF, MAT, WAV, HDF5, Parquet, and KYMA sessions.',
      },
      onNextClick: advance(() => preparePipelineTourView()),
    },
    {
      element: '#scope-copilot-input',
      onHighlightStarted: onTourShow(() => preparePipelineTourView()),
      popover: {
        title: 'Describe The Goal',
        description: 'Use the AI command bar for quick analysis, cleanup ideas, artifact checks, or a product goal like building a gesture classifier.',
      },
      onNextClick: advance(() => preparePipelineTourView()),
    },
    {
      element: '#btn-pipeline-builder',
      onHighlightStarted: onTourShow(() => preparePipelineTourView()),
      popover: {
        title: 'Open Build',
        description: 'Build opens the no-code prompt pipeline. This is the main path from biosignal data to recipe, model, report, and package.',
      },
      onNextClick: advance(() => preparePipelineTourView()),
    },
    {
      element: '#pipeline-prompt',
      onHighlightStarted: onTourShow(() => preparePipelineTourView()),
      popover: {
        title: 'Prompt The Pipeline',
        description: 'Write the outcome in plain language: fatigue detector, blink control, EEG attention model, ECG quality screen, or labeled dataset cleanup.',
      },
      onNextClick: advance(() => preparePipelineTourView()),
    },
    {
      element: '#pipeline-source',
      onHighlightStarted: onTourShow(() => preparePipelineTourView()),
      popover: {
        title: 'Set Input And Output',
        description: 'Select live electrodes, dataset import, LSL, or playback. Then choose live model, research dataset, software control, hardware control, or analysis report.',
      },
      onNextClick: advance(() => preparePipelineTourView()),
    },
    {
      element: '#pipeline-dataset-path',
      onHighlightStarted: onTourShow(() => preparePipelineTourView()),
      popover: {
        title: 'Dataset Path',
        description: 'Paste a dataset or session path here when you are not using live electrodes. Import inspects it; Ingest saves it as a KYMA project.',
      },
      onNextClick: advance(() => preparePipelineTourView()),
    },
    {
      element: '#btn-pipeline-autopilot',
      onHighlightStarted: onTourShow(() => preparePipelineTourView()),
      popover: {
        title: 'Build Automatically First',
        description: 'Use this as the default path. For live electrodes, KYMA opens setup, waits for you to connect electrodes, runs task prompts, rejects bad data, trains when clean windows are accepted, then exports and saves.',
      },
      onNextClick: advance(() => preparePipelineTourView()),
    },
    {
      element: '#pipeline-plan-preview',
      onHighlightStarted: onTourShow(() => preparePipelineTourView()),
      popover: {
        title: 'Plan Preview',
        description: 'This panel shows recipe steps, channels, filters, labels, model candidates, QA, training results, exports, and the project registry.',
      },
      onNextClick: advance(() => preparePipelineTourView({ registry: true })),
    },
    {
      element: '#btn-pipeline-projects',
      onHighlightStarted: onTourShow(() => preparePipelineTourView({ registry: true })),
      popover: {
        title: 'Projects And Jobs',
        description: 'Projects opens the searchable registry. Load saved work, rebuild recipes, inspect job status, open reports, and download export packages here.',
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
        title: 'Review Signal Spans',
        description: 'Freeze the live trace, inspect highlighted regions, add markers, and keep important spans tied to the session before training or export.',
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
        title: 'Live ML Review',
        description: 'Decoded output shows the active model state, confidence, and runtime result. Treat it as workflow evidence after the project, labels, QA, and metrics are saved.',
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
        title: 'Record Clean Sessions',
        description: 'When live data needs labels, add subject, condition, notes, and session labels here. Those fields become part of the reproducible project record.',
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
        title: 'Train And Compare',
        description: 'Train still supports classic supervised models and datasets. Use it for validation while Build handles the prompt-driven project workflow.',
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
        title: 'Deploy Outputs',
        description: 'Control publishes samples, markers, labels, and state changes to LSL, OSC, serial, software tools, or hardware targets after validation.',
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
        description: 'E-STOP remains visible for hardware workflows. Use it any time an output or actuator needs to halt immediately.',
      },
      onNextClick: advance(() => prepareTourView({
        tab: 'dashboard',
        workspace: 'live',
      })),
    },
    {
      element: '#tab-hwdocs',
      onHighlightStarted: onTourShow(() => prepareTourView({
        tab: 'dashboard',
        workspace: 'live',
      })),
      popover: {
        title: 'Guide And Specialist Tools',
        description: 'Guide documents this exact workflow. Blocks, Filters, Workshop, Bench, and Firmware are specialist tools for automation, DSP, analysis, reports, and deployable outputs.',
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
  const tour = createCurrentWorkflowTour();
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
        title: 'Blocks',
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
        title: 'Starter',
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
        title: 'Routing',
        description: 'This is the live KYMA path. Map decoded labels to programs when you want the server to trigger scripts from predictions instead of exporting standalone firmware.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'blocks' })),
    },
    {
      element: '#card-block-help',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'blocks' })),
      popover: {
        title: 'Export Vs Live',
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
        title: 'Filters',
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
        title: 'Magnitude',
        description: 'This graph is the magnitude response of the current preview or saved filter. Use it to verify your passband, stopband, and overall shaping before activation.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'filters' })),
    },
    {
      element: '#filter-polezero-canvas',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'filters' })),
      popover: {
        title: 'Poles / Zeros',
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
        title: 'Library',
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
        title: 'Active Path',
        description: 'Use a saved filter here to insert it into the live KYMA host processing path. Append adds it after the profile defaults; replace bypasses the default BrainFlow stages for the hardware path.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'filters' })),
    },
    {
      element: '#card-filter-export',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'filters' })),
      popover: {
        title: 'Export',
        description: 'Export targets now include host code, C++ reuse, Bode CSV, pole-zero JSON, and a fixed-point header. That keeps the design grounded in existing DSP workflows instead of rewriting the math yourself.',
      },
      onNextClick: advance(() => prepareTourView({ tab: 'bench' })),
    },
    {
      element: '#card-bench-preview',
      onHighlightStarted: onTourShow(() => prepareTourView({ tab: 'bench' })),
      popover: {
        title: 'Bench',
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
    switchViz('none');
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
  const syntheticSetup = $('synthetic-setup');
  const lslInputSetup = $('lsl-input-setup');
  const playbackSetup = $('playback-setup');
  const playbackSel = $('playback-session');
  const playbackRateSel = $('playback-rate');
  const syntheticScenarioSel = $('synthetic-scenario');
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
  if (syntheticSetup) syntheticSetup.style.display = isSynthetic ? 'block' : 'none';
  if (lslInputSetup) lslInputSetup.style.display = isLSL ? 'block' : 'none';
  if (playbackSetup) playbackSetup.style.display = isPlayback ? 'block' : 'none';
  if (playbackSel) playbackSel.disabled = S.streaming || !isPlayback;
  if (playbackRateSel) playbackRateSel.disabled = S.streaming || !isPlayback;
  if (syntheticScenarioSel) syntheticScenarioSel.disabled = S.streaming || !isSynthetic;
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
    recordBtn.textContent = S.recSession ? 'Stop' : 'Record';
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
      hint.textContent = `Playback ${sid} at ${rate}x.`;
    } else if (S.streamSource === 'lsl') {
      const name = S.streamDetails?.name || S.streamDetails?.stream_name || 'LSL stream';
      hint.textContent = `LSL ${name}.`;
    } else {
      const scenario = S.streamDetails?.scenario ? ` (${String(S.streamDetails.scenario).replace(/_/g, ' ')})` : '';
      hint.textContent = `${S.streamSource}${scenario}.`;
    }
  } else if (isSynthetic) {
    const active = syntheticScenarioOptions(S.signalProfileKey).find(item => item.value === getSelectedSyntheticScenario());
    hint.textContent = `Synthetic ${S.signalProfileName}. ${active?.label || 'Clean Window'}.`;
  } else if (isLSL && !hasLSLStream) {
    hint.textContent = 'Select an LSL stream.';
  } else if (isLSL) {
    const active = S.lslInputs.find(stream => (stream.source_id || stream.uid || stream.name) === getSelectedLSLInput());
    const rate = active?.sample_rate ? `${Number(active.sample_rate).toFixed(0)} Hz` : 'irregular rate';
    hint.textContent = `${active?.name || 'LSL stream'} ready (${rate}).`;
  } else if (isPlayback && !hasPlaybackSession) {
    hint.textContent = 'Select a session.';
  } else if (isPlayback) {
    const sid = playbackSel?.value || 'session';
    const rate = Number(playbackRateSel?.value || 1).toFixed(2);
    hint.textContent = `Playback ${sid} at ${rate}x.`;
  } else if (requiresSynthetic) {
    hint.textContent = `${S.signalProfileName} uses synthetic or playback in this build.`;
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
      case 'ml_insights': applyLiveMLInsights(m.data); break;
      case 'prompt_prediction': applyPromptModelPrediction(m.data); break;
      case 'safety':      applySafety(m.data); break;
      case 'review_marker':
        pushReviewMarker({
          ...(m.data || {}),
          createdAt: Date.now(),
          sampleIndex: Number.isFinite(m.data?.sample_index)
            ? Number(m.data.sample_index)
            : (Number.isFinite(m.data?.selection?.end_sample) ? Number(m.data.selection.end_sample) : Math.max(0, S.emgTotal - 1)),
        });
        break;
      case 'state':       setSysState(m.data.state); break;
      case 'calibration': onCalibration(m.data); break;
      case 'signal_confidence': onSignalConfidence(m.data); break;
      case 'signal_recovery': onSignalRecovery(m.data); break;
      case 'guided_step': onGuidedStep(m.data); break;
      case 'guided_progress': onGuidedProgress(m.data); break;
      case 'codegen_complete': onCodegenComplete(m.data); break;
      case 'codegen_status':
        if (m.data?.message) { const s = $('intent-status'); if (s) s.innerHTML = m.data.message; }
        break;
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

function signalCorrelation(a, b) {
  const n = Math.min(a?.length || 0, b?.length || 0);
  if (n < 12) return 0;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) {
    ma += Number(a[i] || 0);
    mb += Number(b[i] || 0);
  }
  ma /= n;
  mb /= n;
  let num = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < n; i++) {
    const da = Number(a[i] || 0) - ma;
    const db = Number(b[i] || 0) - mb;
    num += da * db;
    va += da * da;
    vb += db * db;
  }
  if (va <= 1e-9 || vb <= 1e-9) return 0;
  return num / Math.sqrt(va * vb);
}

function duplicateChannelGroups(pairs = []) {
  const parent = Array.from({ length: N_CH }, (_, index) => index);
  const find = (value) => {
    let v = value;
    while (parent[v] !== v) v = parent[v];
    while (parent[value] !== value) {
      const next = parent[value];
      parent[value] = v;
      value = next;
    }
    return v;
  };
  const union = (a, b) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[rb] = ra;
  };
  pairs.forEach(pair => union(pair[0], pair[1]));
  const groups = new Map();
  for (let i = 0; i < N_CH; i++) {
    const root = find(i);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root).push(i);
  }
  return Array.from(groups.values()).filter(group => group.length > 1);
}

function applyAutoChannelVisibility(activeChannels = [], duplicatePairs = [], rms = []) {
  if (!S.autoChannelVisibility) return;
  const now = performance.now();
  if (now < Number(S.manualChannelOverrideUntil || 0)) return;
  if (now - Number(S.autoChannelLastAt || 0) < 800) return;

  const sharedNoise = new Set();
  duplicateChannelGroups(duplicatePairs).forEach(group => {
    if (group.length >= 3) {
      group.forEach(index => sharedNoise.add(index));
      return;
    }
    const strongest = group.reduce((best, index) => Number(rms[index] || 0) > Number(rms[best] || 0) ? index : best, group[0]);
    group.forEach(index => {
      if (index !== strongest) sharedNoise.add(index);
    });
  });

  const mask = new Array(N_CH).fill(false);
  activeChannels.forEach(index => {
    if (!sharedNoise.has(index)) mask[index] = true;
  });
  if (!mask.some(Boolean)) {
    const best = rms
      .map((value, index) => ({ value: Number(value || 0), index }))
      .sort((a, b) => b.value - a.value)[0];
    if (best && best.value > 0.5) mask[best.index] = true;
  }
  if (!mask.some(Boolean)) return;
  const key = mask.map(Boolean).join('');
  if (key === S.autoChannelLastMask) return;
  S.autoChannelLastMask = key;
  S.autoChannelLastAt = now;
  S.signalHealth.autoHiddenChannels = mask
    .map((visible, index) => visible ? -1 : index)
    .filter(index => index >= 0);
  setChannelMask(mask, { persist: false });
}

function updateSignalHealthFromChunk(channels = []) {
  const rms = [];
  const ptp = [];
  for (let i = 0; i < Math.min(channels.length, N_CH); i++) {
    const values = channels[i] || [];
    let sumSq = 0;
    let min = Infinity;
    let max = -Infinity;
    for (const raw of values) {
      const v = Number(raw || 0);
      sumSq += v * v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    rms[i] = values.length ? Math.sqrt(sumSq / values.length) : 0;
    ptp[i] = values.length ? max - min : 0;
  }
  const flatChannels = rms
    .map((value, index) => ({ value, index }))
    .filter(item => item.value < 0.01 || Number(ptp[item.index] || 0) < 0.01)
    .map(item => item.index);
  const duplicateChannels = new Set();
  const pairs = [];
  for (let i = 0; i < Math.min(channels.length, N_CH); i++) {
    if (flatChannels.includes(i) || rms[i] < 1) continue;
    for (let j = i + 1; j < Math.min(channels.length, N_CH); j++) {
      if (flatChannels.includes(j) || rms[j] < 1) continue;
      const ratio = Math.min(rms[i], rms[j]) / Math.max(rms[i], rms[j], 1e-6);
      const corr = signalCorrelation(channels[i], channels[j]);
      if (ratio > 0.85 && Math.abs(corr) > 0.995) {
        duplicateChannels.add(i);
        duplicateChannels.add(j);
        pairs.push([i, j]);
      }
    }
  }
  const duplicateSharedNoise = new Set();
  duplicateChannelGroups(pairs).forEach(group => {
    if (group.length >= 3) group.forEach(index => duplicateSharedNoise.add(index));
  });
  const activeChannels = rms
    .map((value, index) => ({ value, index }))
    .filter(item => item.value >= 5 && Number(ptp[item.index] || 0) >= 15 && !duplicateSharedNoise.has(item.index))
    .map(item => item.index);
  const warnings = [];
  if (duplicateChannels.size >= 2) {
    warnings.push(`Channels ${Array.from(duplicateChannels).map(index => index + 1).join(', ')} are moving almost identically; that usually means shared noise, floating inputs, or duplicated hardware data.`);
  }
  const activeCount = rms.filter(v => v >= 0.01).length;
  if (activeCount > 0 && flatChannels.length >= Math.max(3, N_CH - 3)) {
    warnings.push(`Several channels are flat. Turn off unused channels or connect/reference them so they do not confuse training.`);
  }
  S.signalHealth = {
    warnings,
    flatChannels,
    duplicateChannels: Array.from(duplicateChannels),
    duplicatePairs: pairs.map(pair => `${pair[0] + 1}/${pair[1] + 1}`),
    activeChannels,
    autoHiddenChannels: Array.isArray(S.signalHealth?.autoHiddenChannels) ? S.signalHealth.autoHiddenChannels : [],
    updatedAt: performance.now(),
  };
  const note = $('signal-health-note');
  if (note) {
    const activeText = activeChannels.length
      ? `Showing CH${activeChannels.map(index => index + 1).join(', CH')}.`
      : 'No clean active channel detected yet.';
    const hiddenText = S.autoChannelVisibility
      ? 'Unused or duplicated channels are hidden automatically.'
      : 'Automatic channel hiding is off.';
    note.textContent = warnings[0]
      ? `${warnings[0]} ${activeText} ${hiddenText}`
      : `${activeText} ${hiddenText}`;
    note.classList.toggle('warn', warnings.length > 0);
  }
  applyAutoChannelVisibility(activeChannels, pairs, rms);
}

/**
 * EMG data comes in as the new increment only (not the full window).
 * we just append each sample to the ring buffer and it scrolls naturally.
 */
function onEmg(d) {
  const ch = d.channels;
  const n = ch[0]?.length ?? 0;
  S.lastSignalAtClient = performance.now();
  updateSignalHealthFromChunk(ch);
  for (let c = 0; c < Math.min(ch.length, N_CH); c++) {
    for (let i = 0; i < n; i++) {
      const value = ch[c][i];
      S.emg[c][(S.emgHead + i) % DISPLAY_SAMPLES] = value;
      S.reviewArchive[c][(S.reviewArchiveHead + i) % REVIEW_ARCHIVE_SAMPLES] = value;
    }
  }
  S.emgHead = (S.emgHead + n) % DISPLAY_SAMPLES;
  S.emgTotal += n;
  S.reviewArchiveHead = (S.reviewArchiveHead + n) % REVIEW_ARCHIVE_SAMPLES;
  S.reviewArchiveTotal += n;
    if (d.rms) {
      S.rms = d.rms;
      S.lastQuality = Array.isArray(d.quality) ? d.quality.slice() : (Array.isArray(d.rms) ? d.rms.slice() : []);
      if (!S.review.paused) {
        updateFatigue(d.rms);
      }
      syncInspectorTelemetry();
      if (S.proportional) applyProportional(d.rms);
    }
    if (S.signalProfileKey === 'eeg' && S.activeViz === 'eeg' && !S.review.paused) {
      refreshEEGBrainView(false);
    }
    syncReviewUI();
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
        <button onclick="window._removePropMap(${i})" style="font-size:9px;padding:1px 5px;background:var(--red);color:#fff;border:none;border-radius:2px;cursor:pointer;opacity:0.7" title="Remove">x</button>
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
      try {
        await ensureBabylonLoaded();
      } catch (err) {
        console.warn('[HandView] Babylon failed to load', err);
        return;
      }
      if (typeof BABYLON === 'undefined') return;
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
  const vizSection = $('viz-section');
  const sections = {
    none: document.getElementById('viz-empty-state'),
    hand: document.getElementById('hand-section'),
    arm: document.getElementById('arm-section'),
    eeg: document.getElementById('eeg-brain-section'),
  };
  if (!sections.none || !sections.hand || !sections.arm || !sections.eeg) return;
  const requested = sections[which] ? which : 'none';
  if (requested === S.activeViz && requested !== 'eeg') {
    S.activeViz = 'none';
  } else {
    S.activeViz = requested;
  }
  Object.entries(sections).forEach(([key, el]) => {
    if (!el) return;
    const active = key === S.activeViz;
    el.style.display = active ? (key === 'hand' || key === 'none' ? 'flex' : 'block') : 'none';
  });
  if (vizSection) vizSection.classList.toggle('viz-collapsed', S.activeViz === 'none');
  const armLabel = $('arm-gesture-label');
  if (armLabel) armLabel.style.display = S.activeViz === 'arm' && S.supportsArmGestures ? 'block' : 'none';
  document.querySelectorAll('.viz-tab').forEach(b => {
    b.classList.toggle('active', b.dataset.viz === S.activeViz);
  });
  localStorage.setItem('kyma-active-viz', S.activeViz);
  if (S.activeViz === 'hand') {
    HandView.init();
  } else if (S.activeViz === 'arm') {
    init3DArm();
  }
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
    if (S.protocolRunner.active) {
      stopProtocolRunner();
    }
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
    S.review.diagnosticsSnapshot = clonePlainData(S.diagnostics);
    S.review.aiResultSnapshot = clonePlainData(S.ai.result);
    S.review.artifactSnapshot = clonePlainData(S.review.artifacts);
    S.review.qualitySnapshot = Array.isArray(S.lastQuality) ? S.lastQuality.slice() : [];
    S.review.fatigueSnapshot = Number(S.fatigue || 1);
    S.review.paused = true;
    S.review.hoverSample = null;
    S.review.hoverChannel = null;
    if (S.review.snapshot) {
      const snapshotTotal = Number(S.review.snapshot.total || 0);
      const snapshotCapacity = Math.max(1, Number(S.review.snapshot.capacity || DISPLAY_SAMPLES));
      const filled = Math.max(Math.min(snapshotTotal, snapshotCapacity), 1);
      const visibleTarget = Math.max(1, Math.min(filled, DISPLAY_SAMPLES));
      const zoomTarget = Math.max(1, filled / visibleTarget);
      S.review.zoomX = nearestReviewZoom(REVIEW_X_ZOOM_LEVELS, zoomTarget);
      S.review.viewCenterSample = Math.max(0, snapshotTotal - 1);
    } else {
      S.review.zoomX = 1;
      S.review.viewCenterSample = null;
    }
    pinFrozenReviewStats(getReviewRenderState());
  } else {
    S.review.paused = false;
    S.review.snapshot = null;
    S.review.timelineSnapshot = null;
    S.review.predictionSnapshot = null;
    S.review.eegBrainSnapshot = null;
    S.review.diagnosticsSnapshot = null;
    S.review.aiResultSnapshot = null;
    S.review.artifactSnapshot = null;
    S.review.qualitySnapshot = null;
    S.review.fatigueSnapshot = null;
    S.review.hoverSample = null;
    S.review.hoverChannel = null;
    clearReviewSelection();
  }
  syncReviewUI();
  syncInspectorTelemetry();
  syncPredictionPanel();
  drawTimeline();
  syncWorkshopUI();
  if (!next && S.activeViz === 'eeg') refreshEEGBrainView(true);
}

async function saveReviewMarker() {
  const event = ($('review-marker-event')?.value || '').trim();
  const note = ($('review-marker-note')?.value || '').trim();
  if (!event) {
    toast('Enter a marker event first', 'red');
    return;
  }

  const state = getReviewRenderState();
  const stats = S.review.lastStats;
  const hasRange = !!(stats && stats.samples > 1);
  const pointSample = hasRange
    ? null
    : (stats?.startSample ?? (Number.isFinite(S.review.hoverSample) ? Number(S.review.hoverSample) : Math.max(state.baseAbs + state.filled - 1, 0)));
  const selection = hasRange ? {
    start_s: (stats.startSample - state.baseAbs) / Math.max(state.sampleRate, 1),
    end_s: (stats.endSample - state.baseAbs) / Math.max(state.sampleRate, 1),
    start_sample: stats.startSample,
    end_sample: stats.endSample,
  } : null;
  const pointTimeS = Number.isFinite(pointSample)
    ? (Number(pointSample) - state.baseAbs) / Math.max(state.sampleRate, 1)
    : null;

  const body = {
    event,
    note,
    selection_start_s: selection?.start_s,
    selection_end_s: selection?.end_s,
    selection_start_sample: selection?.start_sample,
    selection_end_sample: selection?.end_sample,
    sample_time_s: pointTimeS ?? undefined,
    sample_index: Number.isFinite(pointSample) ? Number(pointSample) : undefined,
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

  const localMarker = {
    event,
    note,
    selection,
    sampleIndex: Number.isFinite(pointSample) ? Number(pointSample) : (selection ? selection.end_sample : Math.max(0, state.total - 1)),
    createdAt: Date.now(),
  };

  if (!S.streaming) {
    pushReviewMarker(localMarker);
    toast(hasRange ? `Range marker saved locally: ${event}` : `Point marker saved locally: ${event}`, 'yellow');
    return;
  }

  try {
    const out = await post('/api/review/marker', body);
    if (!S.ws || S.ws.readyState !== WebSocket.OPEN) {
      pushReviewMarker({
        ...(out.marker || {}),
        createdAt: Date.now(),
        sampleIndex: Number.isFinite(out.marker?.sample_index)
          ? Number(out.marker.sample_index)
          : localMarker.sampleIndex,
      });
    }
    toast(hasRange ? `Range marker saved: ${event}` : `Point marker saved: ${event}`);
  } catch (e) {
    pushReviewMarker(localMarker);
    toast(`Marker saved locally only: ${e.message}`, 'yellow');
  }
}

window.focusReviewArtifact = function(index) {
  const item = S.review.artifacts?.[Number(index)];
  if (!item) return;
  if (!S.review.paused) {
    toggleReviewPause(true);
  }
  S.review.selection = {
    startSample: Number(item.startSample),
    endSample: Number(item.endSample),
  };
  S.review.viewCenterSample = Math.round((Number(item.startSample) + Number(item.endSample)) / 2);
  S.review.lastStats = computeSelectionStats(S.review.selection, getReviewRenderState());
  if (Number.isFinite(item.channel) && S.review.lastStats) {
    S.review.lastStats.focusChannel = Number(item.channel);
    S.review.lastStats.focusLabel = item.channelLabel || S.channelLabels[item.channel] || `CH${Number(item.channel) + 1}`;
  }
  S.ai.spotlight = buildSpotlightPayload(
    item.startSample,
    item.endSample,
    `${item.label} Focus`,
    item.kind,
    Number.isFinite(item.channel) ? Number(item.channel) : null,
  );
  if ($('review-marker-event') && !$('review-marker-event').value.trim()) {
    $('review-marker-event').value = `artifact_${item.kind}`;
  }
  if ($('review-marker-note') && !$('review-marker-note').value.trim()) {
    $('review-marker-note').value = item.detail || `${item.label} on ${item.channelLabel}`;
  }
  syncReviewUI();
  syncWorkshopUI();
};

window.markReviewArtifact = async function(index) {
  const item = S.review.artifacts?.[Number(index)];
  if (!item) return;
  window.focusReviewArtifact(index);
  if ($('review-marker-event')) $('review-marker-event').value = `artifact_${item.kind}`;
  if ($('review-marker-note')) $('review-marker-note').value = item.detail || `${item.label} on ${item.channelLabel}`;
  await saveReviewMarker();
};

function bindReviewCanvas() {
  const plotWrap = $('scope-plot-wrap') || $('canvas-container');
  if (!plotWrap || plotWrap.dataset.reviewBound === '1') return;
  plotWrap.dataset.reviewBound = '1';

  plotWrap.addEventListener('pointerdown', e => {
    if (e.button !== 0) return;
    const state = getReviewRenderState();
    if (!state.filled) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const sample = sampleFromCanvasX(x, state, canvas.width || rect.width || 1);
    const hitCursor = hitTestReviewCursor(x, state, canvas.width || rect.width || 1);
    if (hitCursor) {
      S.review.draggingCursor = hitCursor;
      S.review.hoverSample = sample;
      setReviewCursor(hitCursor, sample, state, canvas.width || rect.width || 1);
      syncReviewUI();
      return;
    }
    if (S.review.markerTool) {
      const slot = !S.review.cursors.a ? 'a'
        : (!S.review.cursors.b ? 'b' : S.review.nextCursorSlot || 'a');
      setReviewCursor(slot, sample, state, canvas.width || rect.width || 1);
      S.review.draggingCursor = slot;
      S.review.nextCursorSlot = slot === 'a' ? 'b' : 'a';
      S.review.hoverSample = sample;
      syncReviewUI();
      return;
    }
    if (!S.review.paused) return;
    S.review.dragging = true;
    S.review.dragOriginX = x;
    S.review.hoverSample = sample;
    S.review.viewCenterSample = sample;
    S.review.selection = { startSample: sample, endSample: sample };
    S.review.lastStats = computeSelectionStats(S.review.selection, state);
    syncReviewUI();
    syncWorkshopUI();
  });

  window.addEventListener('pointermove', e => {
    const rect = canvas.getBoundingClientRect();
    const inside = e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom;
    if (inside) {
      const hoverX = clamp(e.clientX - rect.left, 0, rect.width);
      const hoverY = clamp(e.clientY - rect.top, 0, rect.height);
      const hoverState = getReviewRenderState();
      S.review.hoverSample = sampleFromCanvasX(hoverX, hoverState, canvas.width || rect.width || 1);
      S.review.hoverChannel = getReviewChannelAtCanvasY(hoverY, canvas.height || rect.height || 1);
    } else if (S.review.draggingCursor && inside) {
      const hoverX = clamp(e.clientX - rect.left, 0, rect.width);
      const hoverY = clamp(e.clientY - rect.top, 0, rect.height);
      const hoverState = getReviewRenderState();
      S.review.hoverSample = sampleFromCanvasX(hoverX, hoverState, canvas.width || rect.width || 1);
      S.review.hoverChannel = getReviewChannelAtCanvasY(hoverY, canvas.height || rect.height || 1);
    } else if (!S.review.dragging) {
      S.review.hoverSample = null;
      S.review.hoverChannel = null;
    }
    if (S.review.draggingCursor) {
      const x = clamp(e.clientX - rect.left, 0, rect.width);
      const state = getReviewRenderState();
      const sample = sampleFromCanvasX(x, state, canvas.width || rect.width || 1);
      setReviewCursor(S.review.draggingCursor, sample, state, canvas.width || rect.width || 1);
      syncReviewUI();
      return;
    }
    if (!S.review.dragging || !S.review.paused) {
      if (S.review.paused) syncReviewUI();
      return;
    }
    const x = clamp(e.clientX - rect.left, 0, rect.width);
    const state = getReviewRenderState();
    const sample = sampleFromCanvasX(x, state, canvas.width || rect.width || 1);
    if (!S.review.selection) return;
    S.review.selection.endSample = sample;
    S.review.viewCenterSample = Math.round((Number(S.review.selection.startSample || sample) + sample) / 2);
    S.review.lastStats = computeSelectionStats(S.review.selection, state);
    syncReviewUI();
    syncWorkshopUI();
  });

  window.addEventListener('pointerup', () => {
    if (S.review.draggingCursor) {
      S.review.draggingCursor = null;
      syncReviewUI();
      return;
    }
    if (!S.review.dragging) return;
    S.review.dragging = false;
    const state = getReviewRenderState();
    S.review.lastStats = computeSelectionStats(S.review.selection, state);
    if (S.review.lastStats) {
      S.review.viewCenterSample = Math.round((Number(S.review.lastStats.startSample || 0) + Number(S.review.lastStats.endSample || 0)) / 2);
    }
    syncReviewUI();
    syncWorkshopUI();
  });

  plotWrap.addEventListener('wheel', e => {
    if (!S.review.paused) return;
    const state = getReviewRenderState();
    if (!state?.filled) return;
    const width = canvas.width || plotWrap.clientWidth || 1;
    const viewport = getReviewViewport(state, width);
    if (!Number(viewport.viewSamples || 0)) return;
    e.preventDefault();
    clearReviewSelection();
    S.review.hoverSample = null;
    S.review.hoverChannel = null;
    const span = Math.max(1, Number(viewport.viewSamples || 1));
    const stride = Math.max(1, Math.round(span * (e.shiftKey ? 0.45 : 0.14)));
    const direction = e.deltaY > 0 ? 1 : -1;
    const currentCenter = Number.isFinite(S.review.viewCenterSample)
      ? Number(S.review.viewCenterSample)
      : resolveReviewAnchorSample(state);
    S.review.viewCenterSample = clampReviewCenterSample(currentCenter + (direction * stride), state, width);
    pinFrozenReviewStats(getReviewRenderState());
    syncReviewUI();
    syncWorkshopUI();
  }, { passive: false });
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
  const syntheticScenarioSel = $('synthetic-scenario');
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
  const blockTidyBtn = $('btn-block-tidy');
  if (blockTidyBtn) {
    blockTidyBtn.onclick = () => window.autoLayoutActiveProgram();
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
  $('btn-workshop-matlab')?.addEventListener('click', exportWorkshopMatlab);
  $('btn-workshop-save')?.addEventListener('click', saveWorkshopSelection);
  $('btn-ai-copilot-run')?.addEventListener('click', runAICopilot);
  $('btn-ai-copilot-run-inline')?.addEventListener('click', runAICopilot);
  $('btn-pipeline-builder')?.addEventListener('click', runTopPromptBuild);
  $('btn-pipeline-close')?.addEventListener('click', () => togglePipelineBuilder(false));
  $('btn-pipeline-improve-prompt')?.addEventListener('click', improvePipelinePrompt);
  $('btn-pipeline-autopilot')?.addEventListener('click', runPipelineAutopilot);
  $('btn-pipeline-create-model')?.addEventListener('click', runPipelineCreateModel);
  $('btn-pipeline-plan')?.addEventListener('click', runPipelinePlan);
  $('btn-pipeline-qa')?.addEventListener('click', runPipelineQA);
  $('btn-pipeline-train')?.addEventListener('click', runPipelineBaselineTrain);
  $('btn-pipeline-acquire')?.addEventListener('click', runPipelineAcquisition);
  $('btn-pipeline-run-acquire')?.addEventListener('click', runPipelineAcquisitionControl);
  $('btn-pipeline-train-acquire')?.addEventListener('click', runPipelineAcquisitionTrain);
  $('btn-pipeline-models')?.addEventListener('click', runPipelineModelList);
  $('btn-pipeline-labels')?.addEventListener('click', runPipelineLabelSuggestions);
  $('btn-pipeline-dataset')?.addEventListener('click', runPipelineDatasetInspect);
  $('btn-pipeline-ingest')?.addEventListener('click', runPipelineDatasetIngest);
  $('btn-pipeline-compare')?.addEventListener('click', runPipelineCompare);
  $('btn-pipeline-runtime')?.addEventListener('click', runPipelineRuntime);
  $('btn-pipeline-deploy')?.addEventListener('click', runPipelineDeploySmoke);
  $('btn-pipeline-embedding')?.addEventListener('click', runPipelineEmbedding);
  $('btn-pipeline-throughput')?.addEventListener('click', runPipelineThroughput);
  $('btn-pipeline-export')?.addEventListener('click', runPipelineExport);
  $('btn-pipeline-exports')?.addEventListener('click', runPipelineExportList);
  $('btn-pipeline-projects')?.addEventListener('click', runPipelineProjectList);
  $('btn-pipeline-save-project')?.addEventListener('click', savePipelineProject);
  $('pipeline-prompt')?.addEventListener('input', (e) => {
    S.pipelineBuilder.prompt = String(e.target?.value || '');
  });
  $('pipeline-prompt')?.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      runPipelinePlan();
    }
  });
  $('pipeline-source')?.addEventListener('change', (e) => {
    S.pipelineBuilder.source = String(e.target?.value || 'live');
    localStorage.setItem('kyma-pipeline-source', S.pipelineBuilder.source);
    syncPipelineBuilderUI();
  });
  $('pipeline-output')?.addEventListener('change', (e) => {
    S.pipelineBuilder.output = String(e.target?.value || 'live_model');
    localStorage.setItem('kyma-pipeline-output', S.pipelineBuilder.output);
    syncPipelineBuilderUI();
  });
  $('pipeline-dataset-path')?.addEventListener('input', (e) => {
    S.pipelineBuilder.datasetPath = String(e.target?.value || '');
    localStorage.setItem('kyma-pipeline-dataset-path', S.pipelineBuilder.datasetPath);
  });
  $('pipeline-plan-preview')?.addEventListener('change', (e) => {
    const target = e.target;
    if (target?.classList?.contains('pipeline-schema-role')) {
      const column = String(target.dataset.column || '');
      if (!column) return;
      const mapping = ensurePipelineSchemaMapping() || { roles: {} };
      mapping.roles = { ...(mapping.roles || {}), [column]: String(target.value || 'ignore') };
      S.pipelineBuilder.schemaMapping = mapping;
      ensurePipelineSchemaMapping();
      syncPipelineBuilderUI();
      return;
    }
    if (target?.classList?.contains('pipeline-recipe-input')) {
      const field = String(target.dataset.recipeField || '');
      if (!field) return;
      const draft = ensurePipelineRecipeDraft() || {};
      draft[field] = target.type === 'number' ? Number(target.value || 0) : String(target.value || '');
      S.pipelineBuilder.recipeDraft = draft;
      applyPipelineRecipeDraftToPlan();
      syncPipelineBuilderUI();
      return;
    }
    if (target?.classList?.contains('pipeline-recipe-filter')) {
      const idx = Number(target.dataset.filterIndex);
      const field = String(target.dataset.filterField || '');
      const draft = ensurePipelineRecipeDraft() || {};
      if (!Array.isArray(draft.filters) || !Number.isInteger(idx) || !draft.filters[idx] || !field) return;
      draft.filters[idx][field] = target.type === 'number' ? Number(target.value || 0) : String(target.value || '');
      S.pipelineBuilder.recipeDraft = draft;
      applyPipelineRecipeDraftToPlan();
      syncPipelineBuilderUI();
      return;
    }
    if (target?.classList?.contains('pipeline-recipe-label')) {
      const idx = Number(target.dataset.labelIndex);
      const field = String(target.dataset.labelField || '');
      const draft = ensurePipelineRecipeDraft() || {};
      if (!Array.isArray(draft.labels) || !Number.isInteger(idx) || !draft.labels[idx] || !field) return;
      draft.labels[idx][field] = target.type === 'number' ? Number(target.value || 0) : String(target.value || '');
      S.pipelineBuilder.recipeDraft = draft;
      applyPipelineRecipeDraftToPlan();
      syncPipelineBuilderUI();
      return;
    }
    if (target?.classList?.contains('pipeline-registry-filter')) {
      const field = String(target.dataset.registryFilter || '');
      if (!field) return;
      S.pipelineBuilder.registryFilters = {
        ...(S.pipelineBuilder.registryFilters || { query: '', status: 'all', format: 'all' }),
        [field]: String(target.value || (field === 'query' ? '' : 'all')),
      };
      syncPipelineBuilderUI();
    }
  });
  $('pipeline-plan-preview')?.addEventListener('input', (e) => {
    const target = e.target;
    if (target?.id === 'pipeline-choice-dataset-path') {
      S.pipelineBuilder.datasetPath = String(target.value || '').trim();
      localStorage.setItem('kyma-pipeline-dataset-path', S.pipelineBuilder.datasetPath);
      return;
    }
    if (target?.classList?.contains('pipeline-code-textarea')) {
      const path = String(target.dataset.codeEditor || '');
      const wb = ensurePipelineCodeWorkbench();
      const file = wb.files.find(item => item.path === path);
      if (file) {
        file.content = String(target.value || '');
        wb.typedPath = file.path;
        wb.typedChars = file.content.length;
        wb.previewHTML = '';
        S.pipelineBuilder.codeWorkbench = wb;
      }
      return;
    }
    if (!target?.classList?.contains('pipeline-registry-filter')) return;
    const field = String(target.dataset.registryFilter || '');
    if (!field) return;
    S.pipelineBuilder.registryFilters = {
      ...(S.pipelineBuilder.registryFilters || { query: '', status: 'all', format: 'all' }),
      [field]: String(target.value || (field === 'query' ? '' : 'all')),
    };
    syncPipelineBuilderUI();
  });
  $('pipeline-plan-preview')?.addEventListener('click', (e) => {
    const fileBtn = e.target?.closest?.('[data-code-file]');
    if (fileBtn) {
      const wb = ensurePipelineCodeWorkbench();
      wb.selectedPath = String(fileBtn.dataset.codeFile || '');
      wb.mode = 'code';
      wb.typedPath = wb.selectedPath;
      wb.typedChars = 0;
      S.pipelineBuilder.codeWorkbench = wb;
      startPipelineCodeWorkbench();
      return;
    }
    const codeBtn = e.target?.closest?.('[data-code-action]');
    if (codeBtn) {
      const action = String(codeBtn.dataset.codeAction || '');
      if (action === 'regenerate') {
        startPipelineCodeWorkbench({ force: true });
        addPipelineCodeLog('Regenerated code files from the current pipeline.');
        syncPipelineBuilderUI();
      } else if (action === 'show-code') {
        const wb = ensurePipelineCodeWorkbench();
        wb.mode = 'code';
        S.pipelineBuilder.codeWorkbench = wb;
        syncPipelineBuilderUI();
      } else if (action === 'run-preview') {
        runPipelineCodePreview();
      } else if (action === 'run-smoke') {
        runPipelineCodeSmoke();
      } else if (action === 'download') {
        downloadSelectedPipelineCodeFile();
      } else if (action === 'copy-status') {
        copyPipelineCodeCommand('status');
      } else if (action === 'copy-qa') {
        copyPipelineCodeCommand('qa');
      }
      return;
    }
    const btn = e.target?.closest?.('[data-recipe-action]');
    if (!btn) return;
    const action = String(btn.dataset.recipeAction || '');
    const draft = ensurePipelineRecipeDraft() || {};
    if (action === 'add-filter') {
      draft.filters = Array.isArray(draft.filters) ? draft.filters : [];
      draft.filters.push({ kind: 'bandstop', low_hz: 58, high_hz: 62, cutoff_hz: '', order: 2 });
    } else if (action === 'remove-filter') {
      const idx = Number(btn.dataset.filterIndex);
      if (Array.isArray(draft.filters) && Number.isInteger(idx)) draft.filters.splice(idx, 1);
    } else if (action === 'add-label') {
      draft.labels = Array.isArray(draft.labels) ? draft.labels : [];
      draft.labels.push({ name: `label_${draft.labels.length + 1}`, target_windows: 25, min_seconds: 8, cue: '' });
    } else if (action === 'remove-label') {
      const idx = Number(btn.dataset.labelIndex);
      if (Array.isArray(draft.labels) && Number.isInteger(idx)) draft.labels.splice(idx, 1);
    } else {
      return;
    }
    S.pipelineBuilder.recipeDraft = draft;
    applyPipelineRecipeDraftToPlan();
    syncPipelineBuilderUI();
  });
  $('btn-scope-copilot-send')?.addEventListener('click', () => {
    handleScopeCopilotCommand($('scope-copilot-input')?.value || '');
  });
  $('scope-copilot-input')?.addEventListener('input', (e) => {
    S.copilot.input = String(e.target?.value || '');
  });
  $('scope-copilot-input')?.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' || e.shiftKey) return;
    e.preventDefault();
    handleScopeCopilotCommand($('scope-copilot-input')?.value || '');
  });
  $('scope-copilot-actions')?.addEventListener('click', (e) => {
    const btn = e.target?.closest?.('[data-copilot-action-index]');
    if (!btn) return;
    runScopeCopilotAction(Number(btn.dataset.copilotActionIndex || -1));
  });
  $('scope-copilot-findings')?.addEventListener('click', (e) => {
    const btn = e.target?.closest?.('[data-copilot-finding-index]');
    if (!btn) return;
    runScopeCopilotFinding(Number(btn.dataset.copilotFindingIndex || -1));
  });
  $('btn-ai-key-save')?.addEventListener('click', saveAIKey);
  $('btn-ai-key-clear')?.addEventListener('click', clearAIKey);
  $('btn-studio-toggle')?.addEventListener('click', () => {
    S.logic.drawerOpen = !S.logic.drawerOpen;
    localStorage.setItem('kyma-signal-logic-drawer', S.logic.drawerOpen ? '1' : '0');
    syncSignalStudioDrawer();
  });
  $('btn-logic-toggle')?.addEventListener('click', () => {
    S.logic.enabled = !S.logic.enabled;
    persistSignalLogicConfig();
    syncSignalLogicUI();
    toast(S.logic.enabled ? 'Signal Logic runtime enabled' : 'Signal Logic runtime disabled', S.logic.enabled ? '' : 'yellow');
  });
  $('btn-logic-add')?.addEventListener('click', () => {
    addSignalLogicRule();
    toast('Signal logic rule added');
  });
  $('btn-logic-replay')?.addEventListener('click', () => {
    runSignalLogicReplayTest();
  });
  $('btn-logic-template')?.addEventListener('click', () => {
    const select = $('logic-template');
    const idx = Number(select?.value || NaN);
    const templates = signalLogicTemplates();
    const selected = Number.isFinite(idx) ? templates[idx] : null;
    if (!selected) {
      toast('Choose a starter rule first', 'yellow');
      return;
    }
    addSignalLogicRule(selected.rule);
    toast(`Starter added: ${selected.label}`);
  });
  const logicRuleList = $('logic-rule-list');
  const logicFeed = $('logic-runtime-feed');
  if (logicRuleList && logicRuleList.dataset.bound !== '1') {
    logicRuleList.dataset.bound = '1';
    const handleRuleInput = event => {
      const target = event.target;
      const field = target?.dataset?.logicField;
      const row = target?.closest?.('[data-logic-rule-id]');
      if (!field || !row) return;
      const ruleId = row.dataset.logicRuleId;
      const value = target.type === 'checkbox' ? !!target.checked : target.value;
      updateSignalLogicRule(ruleId, field, value);
    };
    logicRuleList.addEventListener('change', handleRuleInput);
    logicRuleList.addEventListener('click', event => {
      const btn = event.target?.closest?.('[data-logic-action]');
      const row = event.target?.closest?.('[data-logic-rule-id]');
      if (!btn || !row) return;
      if (btn.dataset.logicAction === 'delete') {
        removeSignalLogicRule(row.dataset.logicRuleId);
        toast('Signal logic rule removed', 'yellow');
      }
    });
  }
  if (logicFeed && logicFeed.dataset.bound !== '1') {
    logicFeed.dataset.bound = '1';
    logicFeed.addEventListener('click', event => {
      const row = event.target?.closest?.('[data-logic-feed-start]');
      if (!row) return;
      focusSignalLogicFeedRange(Number(row.dataset.logicFeedStart), Number(row.dataset.logicFeedEnd));
      toast('Focused logic replay hit');
    });
  }
  const logicSourceFilter = $('logic-source-filter');
  if (logicSourceFilter && logicSourceFilter.dataset.bound !== '1') {
    logicSourceFilter.dataset.bound = '1';
    logicSourceFilter.addEventListener('input', () => renderSignalLogicSourceBrowser());
  }
  $('btn-logic-source-clear')?.addEventListener('click', () => {
    if ($('logic-source-filter')) $('logic-source-filter').value = '';
    renderSignalLogicSourceBrowser();
  });
  const logicSourceList = $('logic-source-list');
  if (logicSourceList && logicSourceList.dataset.bound !== '1') {
    logicSourceList.dataset.bound = '1';
    logicSourceList.addEventListener('click', async (event) => {
      const btn = event.target?.closest?.('[data-logic-source-action]');
      const row = event.target?.closest?.('[data-logic-source-key]');
      if (!btn || !row) return;
      const sourceKey = String(row.dataset.logicSourceKey || '');
      if (!sourceKey) return;
      if (btn.dataset.logicSourceAction === 'use') {
        addSignalLogicRuleForSource(sourceKey);
        toast(`Rule added for ${getSignalLogicSourceMeta(sourceKey).label}`);
        return;
      }
      if (btn.dataset.logicSourceAction === 'copy') {
        try {
          await navigator.clipboard.writeText(sourceKey);
          toast(`Copied ${sourceKey}`);
        } catch {
          toast('Clipboard unavailable', 'yellow');
        }
      }
    });
  }
  $('btn-code-send-firmware')?.addEventListener('click', sendCodeToFirmwareLab);
  if ($('btn-code-send-firmware')) $('btn-code-send-firmware').disabled = true;
  $('btn-firmware-refresh')?.addEventListener('click', () => loadFirmwareFiles());
  $('btn-firmware-save')?.addEventListener('click', saveFirmwareEditor);
  $('btn-firmware-compile')?.addEventListener('click', compileFirmwareSketch);
  $('btn-firmware-upload')?.addEventListener('click', uploadFirmwareSketch);
  const firmwareFqbn = $('firmware-fqbn');
  if (firmwareFqbn) {
    firmwareFqbn.value = S.firmware.fqbn;
    firmwareFqbn.onchange = () => {
      S.firmware.fqbn = firmwareFqbn.value.trim();
      localStorage.setItem('kyma-firmware-fqbn', S.firmware.fqbn);
      syncFirmwareUI();
    };
  }
  const firmwarePort = $('firmware-port');
  if (firmwarePort) {
    firmwarePort.value = S.firmware.port;
    firmwarePort.onchange = () => {
      S.firmware.port = firmwarePort.value.trim();
      localStorage.setItem('kyma-firmware-port', S.firmware.port);
      syncFirmwareUI();
    };
  }
  const firmwareEditor = $('firmware-editor');
  if (firmwareEditor) {
    firmwareEditor.oninput = () => {
      S.firmware.content = firmwareEditor.value;
      S.firmware.dirty = true;
      syncFirmwareUI();
    };
  }
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
  if (syntheticScenarioSel) {
    populateSyntheticScenarioOptions(S.signalProfileKey);
    syntheticScenarioSel.onchange = () => {
      localStorage.setItem(syntheticScenarioStorageKey(S.signalProfileKey), syntheticScenarioSel.value);
      populateSyntheticScenarioOptions(S.signalProfileKey);
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
    toast('Switch to EEG first', 'yellow');
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
        if (source === 'synthetic') {
          body.synthetic_scenario = getSelectedSyntheticScenario();
        }
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
        resetSignalBuffers();
        resetReviewState({ clearMarkers: true });
        applySignalProfile(r.signal_profile || {});
        S.review.liveAutoScale = true;
        S.review.zoomY = 1;
        S.review.lastAutoScaleAt = 0;
        syncProfileUI();
        await refreshLSLStatus();
        if (S.streamSource === 'synthetic') {
          const scenario = String(S.streamDetails?.scenario || body.synthetic_scenario || 'clean').replace(/_/g, ' ');
          toast(`Synthetic ${S.signalProfileName} started: ${scenario}`);
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
    resetSignalBuffers();
    resetReviewState({ clearMarkers: true });
    toast('Stream cleared');
  };

  const reviewPauseBtn = $('btn-review-pause');
  if (reviewPauseBtn) {
    reviewPauseBtn.onclick = () => toggleReviewPause();
  }
  const controlSensitivitySlider = $('control-sensitivity-slider');
  if (controlSensitivitySlider) {
    controlSensitivitySlider.oninput = () => queueControlSensitivityFromSlider(controlSensitivitySlider.value);
    controlSensitivitySlider.onchange = () => applyControlSensitivity(Number(controlSensitivitySlider.value || 55) / 100);
  }
  if ($('btn-lie-baseline')) $('btn-lie-baseline').onclick = () => trainLieBaseline();
  if ($('btn-lie-reset')) $('btn-lie-reset').onclick = () => resetLieDetector();
  if ($('btn-lie-truth')) $('btn-lie-truth').onclick = () => sampleLieDetector('truth');
  if ($('btn-lie-false')) $('btn-lie-false').onclick = () => sampleLieDetector('lie');
  if ($('btn-lie-start')) $('btn-lie-start').onclick = () => startLieQuestion();
  if ($('btn-lie-score')) $('btn-lie-score').onclick = () => scoreLieQuestion();
  const reviewClearBtn = $('btn-review-clear-selection');
  if (reviewClearBtn) {
    reviewClearBtn.onclick = () => {
      clearReviewSelection();
      S.review.viewCenterSample = resolveReviewAnchorSample(getReviewRenderState());
      syncReviewUI();
      syncWorkshopUI();
    };
  }
  const reviewHZoomOutBtn = $('btn-review-hzoom-out');
  if (reviewHZoomOutBtn) reviewHZoomOutBtn.onclick = () => adjustReviewZoom('x', -1);
  const reviewHZoomInBtn = $('btn-review-hzoom-in');
  if (reviewHZoomInBtn) reviewHZoomInBtn.onclick = () => adjustReviewZoom('x', 1);
  const reviewScrubber = $('review-scrubber');
  if (reviewScrubber) {
    reviewScrubber.oninput = () => {
      const state = getReviewRenderState();
      if (!S.review.paused || !state?.filled) return;
      clearReviewSelection();
      S.review.hoverSample = null;
      S.review.hoverChannel = null;
      S.review.viewCenterSample = clampReviewCenterSample(Number(reviewScrubber.value || 0), state, canvas.width || 1);
      pinFrozenReviewStats(getReviewRenderState());
      syncReviewUI();
      syncWorkshopUI();
    };
  }
  const reviewVZoomOutBtn = $('btn-review-vzoom-out');
  if (reviewVZoomOutBtn) reviewVZoomOutBtn.onclick = () => adjustReviewZoom('y', -1);
  const reviewVZoomInBtn = $('btn-review-vzoom-in');
  if (reviewVZoomInBtn) reviewVZoomInBtn.onclick = () => adjustReviewZoom('y', 1);
  const scopeEnvelopeBtn = $('btn-scope-envelope');
  if (scopeEnvelopeBtn) {
    scopeEnvelopeBtn.onclick = () => {
      S.scopeOverlays.envelope = !S.scopeOverlays.envelope;
      localStorage.setItem('kyma-scope-envelope', S.scopeOverlays.envelope ? '1' : '0');
      syncReviewUI();
    };
  }
  const scopeThresholdBtn = $('btn-scope-thresholds');
  if (scopeThresholdBtn) {
    scopeThresholdBtn.onclick = () => {
      S.scopeOverlays.thresholds = !S.scopeOverlays.thresholds;
      localStorage.setItem('kyma-scope-thresholds', S.scopeOverlays.thresholds ? '1' : '0');
      syncReviewUI();
    };
  }
  const reviewMarkerToolBtn = $('btn-review-marker-tool');
  if (reviewMarkerToolBtn) {
    reviewMarkerToolBtn.onclick = () => {
      const state = getReviewRenderState();
      if (!state.filled) return;
      S.review.markerTool = !S.review.markerTool;
      if (!S.review.markerTool) {
        S.review.draggingCursor = null;
      }
      syncReviewUI();
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
      $('btn-record-session').textContent = 'Record';
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
    toast('Select a protocol first', 'red');
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
      if (S.protocolRunner.active) return;
      resetProtocolRun();
      S.protocolRunner.phase = 'idle';
      S.protocolRunner.step = null;
      syncProtocolUI();
      toast('Protocol run reset');
    };
  }
  const protocolAutoBtn = $('btn-protocol-auto');
  if (protocolAutoBtn) {
    protocolAutoBtn.onclick = async () => {
      await runProtocolRunner();
    };
  }
  const protocolStopBtn = $('btn-protocol-stop');
  if (protocolStopBtn) {
    protocolStopBtn.onclick = async () => {
      await stopProtocolRunner();
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
        const res = await post(`/api/gesture/${g}`);
        toast(`${res?.preview_only ? 'Shortcut preview' : 'Arm gesture'}: ${g}`);
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
  const c = $('scope-plot-wrap') || $('canvas-container');
  canvas.width = c.clientWidth;
  canvas.height = c.clientHeight;
  positionAILens();

  // also tell the 3d arm to resize if it exists
  if (window._arm3d) window._arm3d.resize();
}

function estimateActivationThreshold(state, channel, viewport) {
  const buf = state?.emg?.[channel];
  if (!buf || !state?.filled) return Math.max(Number(S.muteFloor || 0.5) * 2.5, 1e-6);
  const start = Number(viewport.viewStart || 0);
  const end = Number(viewport.viewEnd || start);
  const span = Math.max(1, end - start + 1);
  const stride = Math.max(1, Math.floor(span / 384));
  const absValues = [];
  let sumSq = 0;
  for (let sample = start; sample <= end; sample += stride) {
    const value = Math.abs(Number(buf[bufferIndexForAbsSample(state, sample)] || 0));
    absValues.push(value);
    sumSq += value * value;
  }
  if (!absValues.length) return Math.max(Number(S.muteFloor || 0.5) * 2.5, 1e-6);
  absValues.sort((a, b) => a - b);
  const pct = (p) => absValues[Math.min(absValues.length - 1, Math.max(0, Math.floor((absValues.length - 1) * p)))];
  const median = pct(0.50);
  const p75 = pct(0.75);
  const p90 = pct(0.90);
  const rms = Math.sqrt(sumSq / absValues.length);
  const rest = Number(S.propRestRms?.[channel] || 0);
  const floor = Math.max(Number(S.muteFloor || 0.5) * 2.4, Number(S.signalFullScale || 200) * 0.018);
  const adaptive = Math.max(floor, rest * 1.8, median * 3.0, p75 * 1.45, rms * 0.92);
  return Math.min(adaptive, Math.max(p90 * 1.12, Number(S.signalFullScale || 200) * 0.82));
}

function drawActivationOverlay({ channel, rowTop, rowH, mid, scale, reviewState, viewport, drawStart, endX, visibleWidth, color }) {
  const showEnvelope = !!S.scopeOverlays.envelope;
  const showThresholds = !!S.scopeOverlays.thresholds;
  if (!showEnvelope && !showThresholds) return;

  const threshold = estimateActivationThreshold(reviewState, channel, viewport);
  const thresholdOffset = threshold * scale;
  const yUpperThreshold = Math.max(rowTop + 2, Math.min(rowTop + rowH - 2, mid - thresholdOffset));
  const yLowerThreshold = Math.max(rowTop + 2, Math.min(rowTop + rowH - 2, mid + thresholdOffset));

  if (showThresholds && thresholdOffset > 1) {
    ctx.save();
    ctx.fillStyle = 'rgba(178,131,39,0.055)';
    ctx.fillRect(0, yUpperThreshold, canvas.width, Math.max(1, yLowerThreshold - yUpperThreshold));
    ctx.strokeStyle = 'rgba(178,131,39,0.58)';
    ctx.lineWidth = 0.9;
    ctx.setLineDash([4, 5]);
    ctx.beginPath();
    ctx.moveTo(0, yUpperThreshold);
    ctx.lineTo(canvas.width, yUpperThreshold);
    ctx.moveTo(0, yLowerThreshold);
    ctx.lineTo(canvas.width, yLowerThreshold);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  if (!showEnvelope) return;
  const buf = reviewState?.emg?.[channel];
  if (!buf) return;
  const viewStart = Number(viewport.viewStart || 0);
  const viewSamples = Math.max(1, Number(viewport.viewSamples || 1));
  const fullStart = Number(viewport.fullStart || viewStart);
  const fullEnd = Number(viewport.fullEnd || viewStart);
  const pixelCount = Math.max(1, endX - drawStart + 1);
  const upper = [];
  const lower = [];
  let smoothAbs = 0;
  for (let x = drawStart; x <= endX; x += 1) {
    const px = x - drawStart;
    const startOffset = Math.floor((px / pixelCount) * viewSamples);
    const endOffset = Math.max(startOffset, Math.floor(((px + 1) / pixelCount) * viewSamples) - 1);
    const startSample = clamp(viewStart + startOffset, fullStart, fullEnd);
    const endSample = clamp(viewStart + endOffset, fullStart, fullEnd);
    const span = Math.max(1, endSample - startSample + 1);
    const stride = Math.max(1, Math.floor(span / 32));
    let maxAbs = 0;
    for (let sample = startSample; sample <= endSample; sample += stride) {
      const value = Math.abs(Number(buf[bufferIndexForAbsSample(reviewState, sample)] || 0));
      if (value > maxAbs) maxAbs = value;
    }
    smoothAbs = smoothAbs <= 0 ? maxAbs : (smoothAbs * 0.78 + maxAbs * 0.22);
    const yTop = Math.max(rowTop + 2, Math.min(rowTop + rowH - 2, mid - smoothAbs * scale));
    const yBottom = Math.max(rowTop + 2, Math.min(rowTop + rowH - 2, mid + smoothAbs * scale));
    upper.push([x, yTop]);
    lower.push([x, yBottom]);
  }
  if (upper.length < 2) return;

  ctx.save();
  ctx.fillStyle = 'rgba(79,143,105,0.040)';
  ctx.beginPath();
  upper.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  for (let i = lower.length - 1; i >= 0; i -= 1) {
    ctx.lineTo(lower[i][0], lower[i][1]);
  }
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function renderLoop() {
  autoFitLiveReviewScale();
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
  stepCopilotChannelFocus();

  const t = THEME_CANVAS[currentTheme] || THEME_CANVAS.neutral;
  const cols = chColors();
  const reviewState = getReviewRenderState();

  // clear + fill background
  ctx.fillStyle = t.bg;
  ctx.fillRect(0, 0, W, H);

  const rows = getChannelDrawLayout(H);

  // How many samples have actually been written to the buffer?
  // Don't draw the zero-filled portion — that causes square-wave artifacts.
  const filled = reviewState.filled;
  const viewport = getReviewViewport(reviewState, W);
  const { drawStart, visibleWidth, viewStart, viewSamples, fullStart, fullEnd } = viewport;
  const cursorMetrics = getReviewCursorMetrics(reviewState);
  const rmsSource = reviewState.paused ? (reviewState.rms || S.review.snapshot?.rms || S.rms || []) : (S.rms || []);
  const topLabelLanes = [];
  const reserveTopLabelBox = (preferredX, width, laneHeight = 18) => {
    const left = clamp(Number(preferredX || 0), 4, Math.max(4, W - width - 4));
    const right = left + width;
    let lane = 0;
    for (; lane < topLabelLanes.length; lane += 1) {
      const overlaps = topLabelLanes[lane].some((item) => !(right + 6 < item.left || left - 6 > item.right));
      if (!overlaps) break;
    }
    if (!topLabelLanes[lane]) topLabelLanes[lane] = [];
    topLabelLanes[lane].push({ left, right });
    return { x: left, y: 6 + lane * laneHeight };
  };

  // ── Debug-friendly mute: use only an absolute floor ──
  // Relative muting against the strongest channel hides real low-amplitude
  // channels whenever unused leads are floating and picking up large noise.
  const MUTE_ABS_FLOOR = 0.5;   // µV — absolute minimum to even consider
  for (let ch = 0; ch < N_CH; ch++) {
    if (!isChannelVisible(ch)) continue;
    const raw = rmsSource ? rmsSource[ch] : 0;
    S.rmsSmooth[ch] = S.rmsSmooth[ch] * 0.90 + raw * 0.10;
  }
  const playbackLowAmpMode = S.streamSource === 'playback';
  const muteFloor = playbackLowAmpMode
    ? 0.001
    : (S.muteFloor || MUTE_ABS_FLOOR);
  const muteThresh = muteFloor;

  for (let ch = 0; ch < N_CH; ch++) {
    const buf = reviewState.emg[ch];
    const row = rows[ch];
    const rowH = Math.max(Number(row?.height || 0), 1);
    const rowTop = Number(row?.top || 0);
    const mid = Number(row?.mid || 0);
    const focusAlpha = row?.dimmed ? 0.24 : 1;

    // subtle alternating row shading
    if (ch % 2 === 0) {
      ctx.fillStyle = t.alt;
      ctx.fillRect(0, rowTop, W, rowH);
    }
    if (row?.focused) {
      ctx.save();
      ctx.fillStyle = 'rgba(88,111,218,0.10)';
      ctx.fillRect(0, rowTop, W, rowH);
      ctx.strokeStyle = 'rgba(88,111,218,0.22)';
      ctx.lineWidth = 1.2;
      ctx.strokeRect(0.5, rowTop + 0.5, Math.max(0, W - 1), Math.max(0, rowH - 1));
      ctx.restore();
    }

    // center line
    ctx.beginPath();
    ctx.strokeStyle = t.grid;
    ctx.lineWidth = 1;
    ctx.globalAlpha = row?.dimmed ? 0.32 : 1;
    ctx.moveTo(0, mid); ctx.lineTo(W, mid);
    ctx.stroke();
    ctx.globalAlpha = 1;

    // ── Mute with hysteresis ──
    const sr = S.rmsSmooth[ch];
    if (S.chMuted[ch]) {
      if (sr > muteThresh * 1.3) S.chMuted[ch] = false;   // need 30 % above to turn on
    } else {
      if (sr < muteThresh * 0.7) S.chMuted[ch] = true;    // 30 % below to turn off
    }
    const hidden = !isChannelVisible(ch);
    const muted = hidden || (!playbackLowAmpMode && S.chMuted[ch]);

    // ── Fixed scale — NO auto-scale for noise ──
    // Use a fixed sensitivity that shows real EMG well.
    // Real EMG from Cyton is typically 50-500+ µV.  Noise is 1-20 µV.
    // A fixed scale of (rowH*0.4 / 200µV) means 200 µV fills 80 % of the row.
    // Strong contractions (>200 µV) get clamped at the row edge — that's fine.
    const FULL_SCALE_UV = getPlaybackDisplayFullScale(reviewState);
    const usable = rowH * 0.40;
    const scale = (usable / FULL_SCALE_UV) * Math.max(Number(S.review.zoomY || 1), 0.25);
    const endX = Math.max(drawStart, drawStart + visibleWidth - 1);

    if (!muted) {
      drawActivationOverlay({
        channel: ch,
        rowTop,
        rowH,
        mid,
        scale,
        reviewState,
        viewport,
        drawStart,
        endX,
        visibleWidth,
        color: cols[ch],
      });
    }

    // the actual waveform
    ctx.beginPath();
    if (muted) {
      ctx.strokeStyle = t.grid;
      ctx.lineWidth = 0.6;
      ctx.globalAlpha = 0.25;
    } else {
      ctx.strokeStyle = cols[ch];
      ctx.lineWidth = 1.2;
    }
    ctx.lineJoin = 'round';
    ctx.globalAlpha = muted ? 0.25 : focusAlpha;

    const pixelCount = Math.max(1, endX - drawStart + 1);
    let lastY = mid;
    let connected = false;
    for (let x = drawStart; x <= endX; x++) {
      const px = x - drawStart;
      const startOffset = Math.floor((px / pixelCount) * viewSamples);
      const endOffset = Math.max(startOffset, Math.floor(((px + 1) / pixelCount) * viewSamples) - 1);
      const startSample = clamp(viewStart + startOffset, fullStart, fullEnd);
      const endSample = clamp(viewStart + endOffset, fullStart, fullEnd);

      if (muted) {
        if (x === drawStart) ctx.moveTo(x, mid);
        else ctx.lineTo(x, mid);
        continue;
      }

      let minVal = Infinity;
      let maxVal = -Infinity;
      const span = Math.max(1, endSample - startSample + 1);
      const stride = Math.max(1, Math.floor(span / 48));
      for (let sample = startSample; sample <= endSample; sample += stride) {
        const value = Number(buf[bufferIndexForAbsSample(reviewState, sample)] || 0);
        if (value < minVal) minVal = value;
        if (value > maxVal) maxVal = value;
      }
      if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) {
        minVal = 0;
        maxVal = 0;
      }

      const yHigh = Math.max(rowTop + 2, Math.min(rowTop + rowH - 2, mid - maxVal * scale));
      const yLow = Math.max(rowTop + 2, Math.min(rowTop + rowH - 2, mid - minVal * scale));
      if (span <= 1) {
        if (!connected) {
          ctx.moveTo(x, yHigh);
          connected = true;
        } else {
          ctx.lineTo(x, yHigh);
        }
        lastY = yHigh;
      } else {
        if (!connected) {
          ctx.moveTo(x, (yHigh + yLow) * 0.5);
          connected = true;
        } else {
          ctx.lineTo(x, lastY);
        }
        ctx.moveTo(x, yHigh);
        ctx.lineTo(x, yLow);
        lastY = (yHigh + yLow) * 0.5;
        ctx.moveTo(x, lastY);
      }
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1.0;

    // channel label
    ctx.fillStyle = muted ? t.grid : cols[ch];
    ctx.globalAlpha = muted ? 0.38 : (row?.dimmed ? 0.40 : 1);
    ctx.font = '11px Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    const baseLabel = S.channelLabels[ch] || `CH${ch+1}`;
    const label = hidden ? `${baseLabel} (hidden)` : (muted ? `${baseLabel} (off)` : baseLabel);
    ctx.fillText(label, 4, rowTop + 12);
    ctx.globalAlpha = 1;
  }

  const waveformYForSample = (channel, sample) => {
    const safeChannel = Math.max(0, Math.min(N_CH - 1, Number(channel || 0)));
    const row = rows[safeChannel];
    const buf = reviewState.emg?.[safeChannel];
    if (!row || !buf) return H * 0.5;
    const rowH = Math.max(Number(row.height || 0), 1);
    const rowTop = Number(row.top || 0);
    const mid = Number(row.mid || 0);
    const playbackLowAmpMode = S.streamSource === 'playback';
    const hidden = !isChannelVisible(safeChannel);
    const muted = hidden || (!playbackLowAmpMode && S.chMuted[safeChannel]);
    if (muted) return mid;
    const fullScaleUv = getPlaybackDisplayFullScale(reviewState);
    const usable = rowH * 0.40;
    const scale = (usable / fullScaleUv) * Math.max(Number(S.review.zoomY || 1), 0.25);
    const idx = bufferIndexForAbsSample(reviewState, sample);
    const y = mid - Number(buf[idx] || 0) * scale;
    return Math.max(rowTop + 2, Math.min(rowTop + rowH - 2, y));
  };

  const resolveOverlayChannel = (...candidates) => {
    for (const candidate of candidates) {
      if (Number.isFinite(candidate)) return clamp(Number(candidate), 0, N_CH - 1);
    }
    return clamp(getReviewFocusChannelIndex(reviewState), 0, N_CH - 1);
  };

  const reviewRowRect = (...candidates) => {
    const channel = resolveOverlayChannel(...candidates);
    const row = rows[channel];
    if (!row) return { channel, top: 0, height: H };
    return {
      channel,
      top: Number(row.top || 0),
      height: Math.max(Number(row.height || 0), 1),
    };
  };

  if (filled > 0) {
    const showTrainingOverlays = !!(S.review.showOverlays || S.guidedLabeling?.active);
    const activeAIRegions = showTrainingOverlays ? ensureAIHighlightRegions(reviewState) : [];
    const promptRegions = showTrainingOverlays ? (S.promptModel?.regions || []).filter(item => (
      Number(item.endSample) >= fullStart && Number(item.startSample) <= fullEnd
    )) : [];
    const artifactRegions = showTrainingOverlays ? (S.review.artifacts || []) : [];
    const guidedRegions = showTrainingOverlays ? (S.guidedLabeling?.regions || []) : [];
    [...artifactRegions, ...activeAIRegions, ...promptRegions, ...guidedRegions].forEach(item => {
      const startSample = Number(item.startSample);
      const endSample = Number(item.endSample);
      if (endSample < fullStart || startSample > fullEnd) return;
      const baseStyle = artifactStyle(item.kind);
      const style = {
        ...baseStyle,
        color: item.color || baseStyle.color,
        line: item.line || baseStyle.line,
      };
      const centerSample = Math.round((startSample + endSample) / 2);
      const x1 = canvasXFromSample(startSample, reviewState, W);
      const x2 = canvasXFromSample(endSample, reviewState, W);
      const left = Math.min(x1, x2);
      const width = Math.max(2, Math.abs(x2 - x1));
      const artifactRow = reviewRowRect(item.channel);
      const dotX = canvasXFromSample(centerSample, reviewState, W);
      const dotY = waveformYForSample(artifactRow.channel, centerSample);
      const focused = !!(S.ai.spotlight
        && Number(startSample) <= Number(S.ai.spotlight.endSample || endSample)
        && Number(endSample) >= Number(S.ai.spotlight.startSample || startSample));
      const aiRegion = item.source === 'ai' || item.source === 'prompt_model' || activeAIRegions.includes(item);
      const dotRadius = focused ? (5 + ((Math.sin(Date.now() / 140) + 1) * 0.5 * 2.2)) : 4;
      ctx.save();
      ctx.fillStyle = aiRegion && item.kind === 'focus' ? 'rgba(88,111,218,0.30)' : style.color;
      ctx.fillRect(left, artifactRow.top, width, artifactRow.height);
      ctx.strokeStyle = style.line;
      ctx.lineWidth = aiRegion ? 2 : 1;
      ctx.setLineDash(aiRegion ? [2, 3] : [4, 4]);
      ctx.beginPath();
      ctx.moveTo(left, artifactRow.top);
      ctx.lineTo(left, artifactRow.top + artifactRow.height);
      ctx.moveTo(left + width, artifactRow.top);
      ctx.lineTo(left + width, artifactRow.top + artifactRow.height);
      ctx.stroke();
      ctx.setLineDash([]);
      if (aiRegion) {
        const badge = String(item.label || style.label || 'AI Focus').slice(0, 18);
        const badgeW = Math.min(104, Math.max(38, badge.length * 6.2 + 14));
        const badgeX = clamp(left + 4, 2, Math.max(2, W - badgeW - 2));
        const badgeY = clamp(artifactRow.top + 3, artifactRow.top + 2, artifactRow.top + Math.max(2, artifactRow.height - 17));
        ctx.fillStyle = 'rgba(37,40,49,0.88)';
        ctx.fillRect(badgeX, badgeY, badgeW, 15);
        ctx.fillStyle = '#f5f7fb';
        ctx.font = '9px Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.fillText(badge, badgeX + 6, badgeY + 11);
      }
      ctx.beginPath();
      ctx.fillStyle = style.line;
      ctx.shadowColor = style.line;
      ctx.shadowBlur = focused ? 16 : 8;
      ctx.arc(dotX, dotY, dotRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(255,255,255,0.92)';
      ctx.lineWidth = focused ? 2 : 1.4;
      ctx.arc(dotX, dotY, dotRadius + 2, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    });

    const hoveredRegion = hoveredReviewRegion();
    if (hoveredRegion) {
      const baseStyle = artifactStyle(hoveredRegion.kind);
      const style = {
        ...baseStyle,
        color: hoveredRegion.color || baseStyle.color,
        line: hoveredRegion.line || baseStyle.line,
      };
      const centerSample = Math.round((Number(hoveredRegion.startSample) + Number(hoveredRegion.endSample)) / 2);
      const dotX = canvasXFromSample(centerSample, reviewState, W);
      const dotY = waveformYForSample(Number(hoveredRegion.channel || 0), centerSample);
      const startMs = reviewSampleOffsetMs(Number(hoveredRegion.startSample), reviewState);
      const endMs = reviewSampleOffsetMs(Number(hoveredRegion.endSample), reviewState);
      const label = `${String(hoveredRegion.label || style.label)} | ${hoveredRegion.channelLabel || S.channelLabels[hoveredRegion.channel] || 'Signal'}`;
      const detail = String(hoveredRegion.detail || 'AI-selected signal region.');
      const timing = `${startMs.toFixed(1)} to ${endMs.toFixed(1)} ms`;
      const width = Math.min(Math.max(168, Math.max(label.length, detail.length, timing.length) * 6.2), 280);
      const boxX = clamp(dotX + 10, 6, Math.max(6, W - width - 6));
      const boxY = clamp(dotY - 48, 6, Math.max(6, H - 52));
      ctx.save();
      ctx.fillStyle = 'rgba(37,40,49,0.92)';
      ctx.fillRect(boxX, boxY, width, 46);
      ctx.strokeStyle = style.line;
      ctx.lineWidth = 1;
      ctx.strokeRect(boxX, boxY, width, 46);
      ctx.fillStyle = '#f5f7fb';
      ctx.font = '10px Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillText(label, boxX + 8, boxY + 13);
      ctx.fillStyle = 'rgba(245,247,251,0.78)';
      ctx.fillText(detail || 'Artifact candidate', boxX + 8, boxY + 27);
      ctx.fillStyle = 'rgba(245,247,251,0.58)';
      ctx.fillText(timing, boxX + 8, boxY + 40);
      ctx.restore();
    }

    if (S.ai.spotlight && Number(S.ai.spotlight.until || 0) > Date.now()) {
      const spotlight = S.ai.spotlight || {};
      const fill = String(spotlight.color || spotlightStyle(spotlight.kind).color);
      const line = String(spotlight.line || spotlightStyle(spotlight.kind).line);
      const x1 = canvasXFromSample(Number(S.ai.spotlight.startSample || fullStart), reviewState, W);
      const x2 = canvasXFromSample(Number(S.ai.spotlight.endSample || S.ai.spotlight.startSample || fullEnd), reviewState, W);
      const left = Math.min(x1, x2);
      const width = Math.max(3, Math.abs(x2 - x1));
      const spotlightRow = reviewRowRect(spotlight.channel, S.review.lastStats?.focusChannel);
      ctx.save();
      ctx.fillStyle = fill;
      ctx.fillRect(left, spotlightRow.top, width, spotlightRow.height);
      ctx.strokeStyle = line;
      ctx.lineWidth = 2;
      ctx.strokeRect(left, spotlightRow.top + 1, width, Math.max(0, spotlightRow.height - 2));
      const labelBox = reserveTopLabelBox(left + 6, 132, 22);
      ctx.fillStyle = 'rgba(37,40,49,0.90)';
      ctx.fillRect(labelBox.x, labelBox.y, 132, 20);
      ctx.fillStyle = '#f5f7fb';
      ctx.font = '10px Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillText(String(S.ai.spotlight.label || 'AI Focus'), labelBox.x + 4, labelBox.y + 14);
      ctx.restore();
    } else if (S.ai.spotlight && Number.isFinite(Number(S.ai.spotlight.until || 0)) && Number(S.ai.spotlight.until || 0) <= Date.now()) {
      S.ai.spotlight = null;
    }

    ['a', 'b'].forEach((id, idx) => {
      const cursor = cursorMetrics?.cursors?.[id];
      if (!cursor) return;
      const x = canvasXFromSample(Number(cursor.sample), reviewState, W);
      const color = idx === 0 ? 'rgba(193,92,112,0.92)' : 'rgba(79,143,105,0.92)';
      const label = `${id.toUpperCase()} ${cursor.timeLabel}`;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x - 6, 10);
      ctx.lineTo(x + 6, 10);
      ctx.closePath();
      ctx.fill();
      const cursorBox = reserveTopLabelBox(x + 6, 112);
      ctx.fillStyle = 'rgba(37,40,49,0.84)';
      ctx.fillRect(cursorBox.x, cursorBox.y, 112, 16);
      ctx.fillStyle = '#f5f7fb';
      ctx.font = '10px Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillText(label, cursorBox.x + 4, cursorBox.y + 12);
      ctx.restore();
    });

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
        ctx.font = '10px Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        const markerWidth = Math.min(Math.max(68, String(label).length * 6.6 + 10), 146);
        const markerBox = reserveTopLabelBox(x + 5, markerWidth);
        ctx.fillStyle = 'rgba(37,40,49,0.84)';
        ctx.fillRect(markerBox.x, markerBox.y, markerWidth, 16);
        ctx.fillStyle = color;
        ctx.fillText(label, markerBox.x + 4, markerBox.y + 12);
      }
      ctx.restore();
    };

    S.review.markers.forEach(marker => {
      const range = marker.selection || null;
      if (range && Number.isFinite(range.start_sample) && Number.isFinite(range.end_sample)) {
        const start = Number(range.start_sample);
        const end = Number(range.end_sample);
        if (end < fullStart || start > fullEnd) return;
        const x1 = canvasXFromSample(start, reviewState, W);
        const x2 = canvasXFromSample(end, reviewState, W);
        const markerRow = reviewRowRect(range.focus_channel, marker.channel, marker.focusChannel, S.review.lastStats?.focusChannel);
        ctx.save();
        ctx.fillStyle = 'rgba(88,111,218,0.10)';
        ctx.fillRect(Math.min(x1, x2), markerRow.top, Math.max(2, Math.abs(x2 - x1)), markerRow.height);
        ctx.restore();
        drawMarkerLine(x1, marker.event, 'rgba(88,111,218,0.82)');
        drawMarkerLine(x2, '', 'rgba(88,111,218,0.62)', true);
      } else if (Number.isFinite(marker.sampleIndex)) {
        const sampleIndex = Number(marker.sampleIndex);
        if (sampleIndex < fullStart || sampleIndex > fullEnd) return;
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
      const stats = S.review.lastStats;
      const selectionRow = reviewRowRect(stats?.focusChannel, S.ai.spotlight?.channel);
      ctx.save();
      ctx.fillStyle = 'rgba(88,111,218,0.12)';
      ctx.fillRect(left, selectionRow.top, width, selectionRow.height);
      ctx.strokeStyle = 'rgba(88,111,218,0.74)';
      ctx.lineWidth = 1.3;
      ctx.strokeRect(left, selectionRow.top + 1, width, Math.max(0, selectionRow.height - 2));
      const detail = stats
        ? `${reviewPointLabel(stats.startSample, reviewState)} -> ${reviewPointLabel(stats.endSample, reviewState)} | ${stats.durationMs.toFixed(1)} ms`
        : '';
      if (detail) {
        const detailBox = reserveTopLabelBox(left + 4, 210);
        ctx.fillStyle = 'rgba(37,40,49,0.84)';
        ctx.fillRect(detailBox.x, detailBox.y, 210, 18);
        ctx.fillStyle = '#f5f7fb';
        ctx.font = '10px Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.fillText(detail, detailBox.x + 4, detailBox.y + 13);
      }
      ctx.restore();
    }
  }

  if (reviewState.paused && Number.isFinite(S.review.hoverSample)) {
    const hoverMetrics = getReviewHoverMetrics(reviewState);
    if (hoverMetrics && hoverMetrics.sample >= fullStart && hoverMetrics.sample <= fullEnd) {
      const hoverX = canvasXFromSample(hoverMetrics.sample, reviewState, W);
      const label = `${hoverMetrics.channelLabel} ${hoverMetrics.timeLabel} · ${hoverMetrics.valueLabel}`;
      const boxWidth = Math.min(Math.max(156, label.length * 6.15), 260);
      ctx.save();
      ctx.strokeStyle = 'rgba(193,92,112,0.82)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hoverX, 0);
      ctx.lineTo(hoverX, H);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(37,40,49,0.84)';
      const boxX = clamp(hoverX + 6, 4, Math.max(4, W - boxWidth - 4));
      ctx.fillRect(boxX, H - 24, boxWidth, 18);
      ctx.fillStyle = '#f5f7fb';
      ctx.font = '10px Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillText(label, boxX + 4, H - 11);
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
  if (window._arm3d) return;
  if (typeof THREE === 'undefined') {
    ensureThreeLoaded()
      .then(() => init3DArm())
      .catch((err) => console.warn('three.js failed to load', err));
    return;
  }

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
      const c = new THREE.Color(THEME_SCENE_BG[name] || THEME_SCENE_BG.neutral);
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
      datasetStatus.textContent = `Select sessions to build a ${S.signalProfileName} dataset.`;
    } else if (mismatchedSessions.length) {
      datasetStatus.textContent = `${selectedSessions.length} selected. ${matchingSessions.length} match. ${mismatchedSessions.length} rejected.`;
    } else {
      datasetStatus.textContent = `${matchingSessions.length} ${S.signalProfileName} session(s) selected. Ready to build.`;
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
      experimentStatus.textContent = 'Pick a dataset, then run eval.';
    } else if (datasetProfileMismatch) {
      experimentStatus.textContent = `Switch to ${(activeDataset.signal_profile_name || activeDataset.signal_profile || 'that profile')}.`;
    } else if (!activeDataset.ready) {
      experimentStatus.textContent = `Dataset "${activeDataset.name || activeDataset.dataset_id}" is not ready yet.`;
    } else if (losoEnabled && !losoReady) {
      const groups = (activeDataset.full_session_groups || []).length || 0;
      experimentStatus.textContent = `LOSO not ready. Full runs: ${groups}.`;
    } else if (subjectHoldoutEnabled && !subjectHoldoutReady) {
      const subjects = (activeDataset.full_subjects || []).length || 0;
      experimentStatus.textContent = `Subject holdout not ready. Full subjects: ${subjects}.`;
    } else {
      const labels = (activeDataset.labels_present || []).join(', ') || 'no labels';
      const gapText = temporalGapEnabled && !losoEnabled && !subjectHoldoutEnabled ? ` with ${(Number(gapInput?.value || 0.2)).toFixed(2)} s gap` : '';
      const losoText = losoEnabled ? ` Full runs: ${(activeDataset.full_session_groups || []).length || 0}.` : '';
      const subjectText = subjectHoldoutEnabled ? ` Full subjects: ${(activeDataset.full_subjects || []).length || 0}.` : '';
      experimentStatus.textContent = `Ready: ${activeDataset.estimated_windows || 0} windows | ${labels} | ${humanizeExperimentSplit(splitStrategy)}${gapText}.${losoText}${subjectText}`;
    }
  }

  if (datasetList) {
    datasetList.innerHTML = '';
    if (!S.datasets.length) {
      datasetList.innerHTML = '<div style="font-size:11px;color:var(--text-dim)">No datasets yet</div>';
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
      experimentList.innerHTML = '<div style="font-size:11px;color:var(--text-dim)">No evals yet</div>';
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
  renderFatigueValue(S.fatigue);
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
  tx.fillStyle = (THEME_CANVAS[currentTheme] || THEME_CANVAS.neutral).bg;
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

  const theme = THEME_CANVAS[currentTheme] || THEME_CANVAS.neutral;
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
    if (Array.isArray(S.signalHealth?.duplicateChannels) && S.signalHealth.duplicateChannels.includes(i)) {
      cls = 'bad'; label = 'DUP';
    } else if (v < 0.02) {
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
  const theme = THEME_CANVAS[currentTheme] || THEME_CANVAS.neutral;
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
      gx.fillText('Preview a filter to inspect response', 14, H / 2);
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

  const theme = THEME_CANVAS[currentTheme] || THEME_CANVAS.neutral;
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
  area.value = '// Preview or select a filter to export';
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
    '# KYMA Bench',
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
  list.innerHTML = '<div class="filter-note">No filters yet.</div>';
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
    : 'Response';
  buildFilterSavedList();
  renderSavedFilterPalette();
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
    sendToFirmware: false,
  };
  if ($('btn-code-send-firmware')) $('btn-code-send-firmware').disabled = true;
  $('code-modal')?.classList.add('active');
}

function openBenchReportModal() {
  refreshBenchReportUI();
  const output = $('code-output');
  if (output) output.value = buildBenchReportText();
  if ($('code-modal-title')) $('code-modal-title').textContent = 'Bench Export';
  S.exportMeta = {
    filename: `kyma_bench_report_${S.signalProfileKey}.md`,
    name: `kyma_bench_report_${S.signalProfileKey}`,
    sendToFirmware: false,
  };
  if ($('btn-code-send-firmware')) $('btn-code-send-firmware').disabled = true;
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

const BLOCK_GRID_SIZE = 28;
const BLOCK_ALIGN_THRESHOLD = 10;
const BLOCK_PORT_SNAP_PX = 28;

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
function getPaletteBlockLabel(el) {
  if (!el) return 'Block';
  return String(el.dataset.blockLabel || BLOCK_TYPES[el.dataset.type]?.label || el.dataset.type || 'Block');
}

function applyPalettePresetToNode(node, el) {
  if (!node || !el) return node;
  if (node.type === 'saved_filter') {
    const filterId = String(el.dataset.filterId || '').trim();
    if (filterId) node.params.filter_id = filterId;
    const exportTarget = String(el.dataset.exportTarget || '').trim();
    if (exportTarget) node.params.export_target = exportTarget;
  }
  return node;
}

function addNodeFromPaletteElement(el, x, y) {
  const type = el?.dataset?.type;
  if (!type) return null;
  const prog = getActiveProgram();
  if (!prog) {
    toast('Create a program first', 'red');
    return null;
  }
  const node = applyPalettePresetToNode(createNode(type, x, y), el);
  if (!node) return null;
  prog.nodes[node.id] = node;
  savePrograms();
  renderCanvas();
  if (node.type === 'saved_filter' && node.params.filter_id) {
    ensureFilterRecord(String(node.params.filter_id)).catch(() => {});
  }
  toast(`Added: ${getPaletteBlockLabel(el)}`);
  return node;
}

function wirePaletteBlockElement(el) {
  if (!el || el.dataset.paletteBound === '1') return;
  el.dataset.paletteBound = '1';
  el.draggable = false;

  el.addEventListener('dblclick', (e) => {
    e.preventDefault();
    const prog = getActiveProgram();
    if (!prog) {
      toast('Create a program first', 'red');
      return;
    }
    const nodeCount = Object.keys(prog.nodes).length;
    const x = 100 + (nodeCount % 4) * 200;
    const y = 80 + Math.floor(nodeCount / 4) * 120;
    addNodeFromPaletteElement(el, x, y);
  });

  el.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    const type = el.dataset.type;
    if (!type) return;

    const startX = e.clientX;
    const startY = e.clientY;
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
        ghost.appendChild(document.createTextNode(getPaletteBlockLabel(el)));
        document.body.appendChild(ghost);
      }
      ghost.style.left = `${ev.clientX + 12}px`;
      ghost.style.top = `${ev.clientY + 12}px`;

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
      if (!prog || !container) return;

      const r = container.getBoundingClientRect();
      if (ev.clientX < r.left || ev.clientX > r.right || ev.clientY < r.top || ev.clientY > r.bottom) return;

      const zoom = prog.viewZoom || 1;
      const ox = prog.viewOffset?.x || 0;
      const oy = prog.viewOffset?.y || 0;
      const x = (ev.clientX - r.left - ox) / zoom;
      const y = (ev.clientY - r.top - oy) / zoom;
      addNodeFromPaletteElement(el, x, y);
    };

    document.addEventListener('pointermove', onMove);
    document.addEventListener('pointerup', onUp);
  });
}

function renderSavedFilterPalette() {
  const container = $('saved-filter-palette');
  if (!container) return;
  const filters = S.filterLab.filters || [];
  container.innerHTML = '';
  if (!filters.length) {
    container.innerHTML = '<div class="filter-note">Save a filter in Filters and it will appear here for Blocks.</div>';
    return;
  }

  filters.forEach(item => {
    const block = document.createElement('div');
    const title = String(item.name || item.id || 'Saved Filter');
    const active = item.id === S.filterLab.active_filter_id;

    block.className = 'palette-block';
    block.dataset.type = 'saved_filter';
    block.dataset.filterId = String(item.id || '');
    block.dataset.exportTarget = 'fixed_point_header';
    block.dataset.blockLabel = title;
    block.title = `${title} | ${item.method} ${item.response_type}`;

    const icon = document.createElement('span');
    icon.className = 'block-icon';
    icon.style.background = '#7ee787';

    const group = document.createElement('div');
    group.className = 'block-label-group';

    const label = document.createElement('div');
    label.textContent = title;

    const meta = document.createElement('div');
    meta.className = 'block-meta';
    meta.textContent = `${item.method} ${item.response_type}${active ? ' · ACTIVE' : ''}`;

    group.appendChild(label);
    group.appendChild(meta);
    block.appendChild(icon);
    block.appendChild(group);
    container.appendChild(block);
    wirePaletteBlockElement(block);
  });
}

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
    hint.className = 'node-block node-block-empty'; // reuse class for easy cleanup
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
  el.dataset.nodeType = node.type;
  el.dataset.nodeCategory = String(def.category || '');
  el.style.setProperty('--node-accent', def.color);
  el.style.left = node.x + 'px';
  el.style.top = node.y + 'px';

  if (node.id === S.selectedNodeId) el.classList.add('selected');

  // header
  const header = document.createElement('div');
  header.className = 'node-header';
  header.innerHTML = `<span class="node-tone"></span><span class="node-title">${def.label}</span>`;
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
    html += '<div style="font-size:9px;color:var(--text-dim)">No filters for this profile.</div>';
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

function snapBlockValue(value) {
  return Math.round(Number(value || 0) / BLOCK_GRID_SIZE) * BLOCK_GRID_SIZE;
}

function clearNodeGuides() {
  $('node-guide-v')?.classList.remove('visible');
  $('node-guide-h')?.classList.remove('visible');
}

function showNodeGuides(guides) {
  const guideV = $('node-guide-v');
  const guideH = $('node-guide-h');
  if (!guideV || !guideH) return;

  if (Number.isFinite(guides?.x)) {
    guideV.style.left = `${guides.x}px`;
    guideV.classList.add('visible');
  } else {
    guideV.classList.remove('visible');
  }

  if (Number.isFinite(guides?.y)) {
    guideH.style.top = `${guides.y}px`;
    guideH.classList.add('visible');
  } else {
    guideH.classList.remove('visible');
  }
}

function getNodeCanvasMetrics(activeNodeId = '') {
  const metrics = [];
  document.querySelectorAll('.node-block[data-node-id]').forEach((el) => {
    const nodeId = String(el.dataset.nodeId || '');
    if (!nodeId || nodeId === activeNodeId) return;
    const left = parseFloat(el.style.left || '0');
    const top = parseFloat(el.style.top || '0');
    const width = el.offsetWidth || 0;
    const height = el.offsetHeight || 0;
    metrics.push({
      id: nodeId,
      left,
      top,
      width,
      height,
      centerX: left + width / 2,
      centerY: top + height / 2,
    });
  });
  return metrics;
}

function computeDragSnap(node, draftX, draftY, el) {
  const width = el?.offsetWidth || 0;
  const height = el?.offsetHeight || 0;
  let x = snapBlockValue(draftX);
  let y = snapBlockValue(draftY);
  let guideX = null;
  let guideY = null;
  const centerX = x + width / 2;
  const centerY = y + height / 2;

  getNodeCanvasMetrics(node.id).forEach((other) => {
    if (Math.abs(other.left - x) <= BLOCK_ALIGN_THRESHOLD) {
      x = other.left;
      guideX = other.left;
    } else if (Math.abs(other.centerX - centerX) <= BLOCK_ALIGN_THRESHOLD) {
      x = other.centerX - width / 2;
      guideX = other.centerX;
    }

    if (Math.abs(other.top - y) <= BLOCK_ALIGN_THRESHOLD) {
      y = other.top;
      guideY = other.top;
    } else if (Math.abs(other.centerY - centerY) <= BLOCK_ALIGN_THRESHOLD) {
      y = other.centerY - height / 2;
      guideY = other.centerY;
    }
  });

  return { x, y, guideX, guideY };
}

function getNearestInputPort(clientX, clientY, prog, fromNodeId = '') {
  let best = null;
  document.querySelectorAll('.port-dot[data-dir="in"]').forEach((dot) => {
    const toNodeId = String(dot.dataset.nodeId || '');
    const toPortId = String(dot.dataset.portId || '');
    if (!toNodeId || !toPortId) return;
    if (toNodeId === fromNodeId) return;
    if (prog.connections.some(c => c.toNode === toNodeId && c.toPort === toPortId)) return;

    const rect = dot.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = clientX - cx;
    const dy = clientY - cy;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance > BLOCK_PORT_SNAP_PX) return;
    if (!best || distance < best.distance) {
      best = { dot, distance, clientX: cx, clientY: cy };
    }
  });
  return best;
}

function clearPortTargets() {
  document.querySelectorAll('.port-dot.port-target').forEach(d => d.classList.remove('port-target'));
  document.querySelectorAll('.node-block.port-target').forEach(n => n.classList.remove('port-target'));
}

function markPortTarget(dot) {
  clearPortTargets();
  if (!dot) return;
  dot.classList.add('port-target');
  const nodeEl = dot.closest('.node-block');
  nodeEl?.classList.add('port-target');
}

function autoLayoutActiveProgram() {
  const prog = getActiveProgram();
  if (!prog) {
    toast('Choose a program first', 'yellow');
    return;
  }
  const nodes = Object.values(prog.nodes || {});
  if (!nodes.length) return;

  const rankMap = {};
  const indegree = {};
  const outgoing = {};
  nodes.forEach((node) => {
    indegree[node.id] = 0;
    outgoing[node.id] = [];
  });
  (prog.connections || []).forEach((conn) => {
    if (outgoing[conn.fromNode]) outgoing[conn.fromNode].push(conn.toNode);
    if (Object.hasOwn(indegree, conn.toNode)) indegree[conn.toNode] += 1;
  });

  const startNode = nodes.find(node => node.type === 'start') || nodes[0];
  const queue = [startNode.id];
  rankMap[startNode.id] = 0;
  const visited = new Set();
  while (queue.length) {
    const current = queue.shift();
    if (visited.has(current)) continue;
    visited.add(current);
    const currentRank = rankMap[current] || 0;
    (outgoing[current] || []).forEach((nextId) => {
      rankMap[nextId] = Math.max(rankMap[nextId] || 0, currentRank + 1);
      queue.push(nextId);
    });
  }

  let fallbackRank = Math.max(0, ...Object.values(rankMap));
  nodes.forEach((node) => {
    if (!Object.hasOwn(rankMap, node.id)) {
      fallbackRank += 1;
      rankMap[node.id] = fallbackRank;
    }
  });

  const columns = new Map();
  nodes.forEach((node) => {
    const rank = rankMap[node.id] || 0;
    if (!columns.has(rank)) columns.set(rank, []);
    columns.get(rank).push(node);
  });

  Array.from(columns.values()).forEach((col) => {
    col.sort((a, b) => String(a.type).localeCompare(String(b.type)) || a.id.localeCompare(b.id));
  });

  const startX = 96;
  const startY = 88;
  const colStep = 248;
  const rowStep = 132;
  Array.from(columns.entries()).sort((a, b) => a[0] - b[0]).forEach(([rank, col]) => {
    col.forEach((node, rowIndex) => {
      node.x = snapBlockValue(startX + rank * colStep);
      node.y = snapBlockValue(startY + rowIndex * rowStep);
    });
  });

  savePrograms();
  renderCanvas();
  toast('Graph tidied');
}
window.autoLayoutActiveProgram = autoLayoutActiveProgram;

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
      const draftX = origX + (ev.clientX - startX) / zoom;
      const draftY = origY + (ev.clientY - startY) / zoom;
      const snapped = computeDragSnap(node, draftX, draftY, el);
      node.x = snapped.x;
      node.y = snapped.y;
      el.style.left = node.x + 'px';
      el.style.top = node.y + 'px';
      showNodeGuides({ x: snapped.guideX, y: snapped.guideY });
      renderWires();
    };

    const onUp = () => {
      document.removeEventListener('pointermove', onMove);
      document.removeEventListener('pointerup', onUp);
      clearNodeGuides();
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
      const nearest = getNearestInputPort(ev.clientX, ev.clientY, prog, fromNodeId);
      const targetX = nearest ? nearest.clientX : ev.clientX;
      const targetY = nearest ? nearest.clientY : ev.clientY;
      const mx = (targetX - canvasRect.left) / zoom;
      const my = (targetY - canvasRect.top) / zoom;
      const dx = Math.max(Math.abs(mx - startPos.x) * 0.5, 40);
      const d = `M ${startPos.x} ${startPos.y} C ${startPos.x+dx} ${startPos.y}, ${mx-dx} ${my}, ${mx} ${my}`;
      tempPath.setAttribute('d', d);
      markPortTarget(nearest?.dot || null);
    };

    const onUp = (ev) => {
      document.removeEventListener('pointermove', onMove);
      document.removeEventListener('pointerup', onUp);
      tempPath.remove();
      const nearest = getNearestInputPort(ev.clientX, ev.clientY, prog, fromNodeId);
      clearPortTargets();

      // check if dropped on or near an input port
      const target = nearest?.dot || document.elementFromPoint(ev.clientX, ev.clientY);
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
      toast('Connected');
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
  document.querySelectorAll('.palette-block').forEach(wirePaletteBlockElement);
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
    sendToFirmware: true,
  };
  if ($('btn-code-send-firmware')) $('btn-code-send-firmware').disabled = false;
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
    const forceExpanded = key === 'decoded-output';

    if (forceExpanded) {
      card.classList.remove('collapsed');
      localStorage.removeItem(storageKey);
    } else if (saved === '1') card.classList.add('collapsed');
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
  const firmware = $('firmware-workspace');
  const hwdocs = $('hw-docs');
  const codeIde = $('code-ide');
  const tabDash = $('tab-dashboard');
  const tabBlocks = $('tab-blocks');
  const tabFilters = $('tab-filters');
  const tabWorkshop = $('tab-workshop');
  const tabBench = $('tab-bench');
  const tabFirmware = $('tab-firmware');
  const tabCode = $('tab-code');
  const tabHwdocs = $('tab-hwdocs');

  document.body.classList.toggle('code-split', tab === 'code');
  if (tab !== 'code') moveCodeInspectCards(false);

  // Hide all
  main.classList.add('hidden');
  editor.classList.remove('active');
  filterLab.classList.remove('active');
  workshop.classList.remove('active');
  benchReport.classList.remove('active');
  firmware.classList.remove('active');
  hwdocs.classList.remove('active');
  if (codeIde) codeIde.style.display = 'none';
  tabDash.classList.remove('active');
  tabBlocks.classList.remove('active');
  tabFilters.classList.remove('active');
  tabWorkshop.classList.remove('active');
  tabBench.classList.remove('active');
  tabFirmware.classList.remove('active');
  if (tabCode) tabCode.classList.remove('active');
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
  } else if (tab === 'firmware') {
    firmware.classList.add('active');
    tabFirmware.classList.add('active');
    syncFirmwareUI();
    loadFirmwareFiles();
  } else if (tab === 'code') {
    compactCodeDashboard();
    main.classList.remove('hidden');
    if (codeIde) codeIde.style.display = 'flex';
    if (tabCode) tabCode.classList.add('active');
    if (tabDash) tabDash.classList.remove('active');
    if (tabBlocks) tabBlocks.classList.remove('active');
    if (tabFilters) tabFilters.classList.remove('active');
    if (tabWorkshop) tabWorkshop.classList.remove('active');
    restoreCodeIdePanel();
    ensureBlankCodeWorkspace();
  } else if (tab === 'hwdocs') {
    hwdocs.classList.add('active');
    tabHwdocs.classList.add('active');
  } else {
    main.classList.remove('hidden');
    syncWorkspaceUI();
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
// INTELLIGENCE LAYER — Signal Confidence, Recovery, Intent, Guided Recording
// =============================================================================

// ── Confidence Rings ──────────────────────────────────────────────────────────

function _trustColor(t) {
  if (t > 0.7) return 'var(--green)';
  if (t > 0.4) return 'var(--yellow)';
  return 'var(--red)';
}

function _buildRing(ch, trust, status) {
  const pct = Math.max(0, Math.min(100, trust * 100));
  const dashLen = 2 * Math.PI * 17;
  const offset = dashLen * (1 - trust);
  const c = _trustColor(trust);
  return `<div style="display:flex;flex-direction:column;align-items:center;gap:2px">
    <svg viewBox="0 0 40 40" style="width:32px;height:32px">
      <circle cx="20" cy="20" r="17" fill="none" stroke="var(--border)" stroke-width="3"/>
      <circle cx="20" cy="20" r="17" fill="none" stroke="${c}" stroke-width="3"
        stroke-dasharray="${dashLen}" stroke-dashoffset="${offset}"
        stroke-linecap="round" transform="rotate(-90 20 20)"
        style="transition: stroke-dashoffset .8s cubic-bezier(0.16,1,0.3,1), stroke .5s"/>
      <text x="20" y="23" text-anchor="middle" fill="var(--text)" font-size="10" font-weight="600" font-family="var(--font)">${Math.round(pct)}</text>
    </svg>
    <span style="font-size:8px;color:var(--text-dim);text-transform:uppercase;letter-spacing:.4px">CH${ch+1}</span>
  </div>`;
}

function onSignalConfidence(d) {
  const ringsEl = $('confidence-rings');
  const msgEl = $('confidence-message');
  const viabilityFill = $('viability-fill');
  const viabilityText = $('viability-text');
  if (ringsEl && d.channels) {
    ringsEl.innerHTML = d.channels.map(ch => _buildRing(ch.channel, ch.trust_score, ch.status)).join('');
  }
  if (msgEl) msgEl.textContent = d.message || '';
  if (viabilityFill) viabilityFill.style.width = `${(d.model_viability || 0) * 100}%`;
  if (viabilityText) {
    const v = Math.round((d.model_viability || 0) * 100);
    const imp = Math.round((d.improvement_potential || 0) * 100);
    viabilityText.textContent = `${v}% viable` + (imp > 5 ? ` · ${imp}% improvement possible` : '');
  }
  updatePipelineTimeline(d);
}

// ── Signal Recovery / Intervention Cards ──────────────────────────────────────

function onSignalRecovery(d) {
  const container = $('intervention-container');
  if (!container) return;
  const suggestions = d.suggestions || [];
  if (!suggestions.length) { container.innerHTML = ''; return; }
  // Show max 3 suggestions
  container.innerHTML = suggestions.slice(0, 3).map(s => {
    const icon = s.severity === 'critical' ? 'Critical' : s.severity === 'warning' ? 'Warning' : 'Info';
    const bg = s.severity === 'critical' ? 'rgba(193,92,112,.12)' : s.severity === 'warning' ? 'rgba(255,176,32,.08)' : 'rgba(88,111,218,.08)';
    const border = s.severity === 'critical' ? 'rgba(193,92,112,.25)' : s.severity === 'warning' ? 'rgba(255,176,32,.2)' : 'rgba(88,111,218,.15)';
    return `<div style="padding:8px 10px;background:${bg};border:1px solid ${border};border-radius:10px;backdrop-filter:blur(12px);font-size:10px;line-height:1.5;animation:fadeInUp .3s ease-out">
      <div style="font-weight:600;margin-bottom:2px">${icon}: ${s.action}</div>
      <div style="color:var(--text-dim);font-size:9px">${s.detail}${s.auto_applied ? ' · auto-applied' : ''}</div>
    </div>`;
  }).join('');
}

// ── Guided Recording Overlay ─────────────────────────────────────────────────

function onGuidedStep(d) {
  const overlay = $('guided-overlay');
  const status = $('intent-status');
  if (!overlay) return;
  if (d.phase === 'complete' || !d.active) {
    S.guidedLabeling.active = false;
    S.guidedLabeling.activeRegionKey = '';
    S.review.showOverlays = false;
    overlay.style.display = 'none';
    if (d.phase === 'complete') stopGuidedStatusPolling();
    if (d.phase === 'complete') {
      if (status) status.innerHTML = '<strong>Training complete</strong> — generating project code...';
      toast('Recording complete — generating code', 'green');
      _autoTrainAndGenCode();
    }
    return;
  }
  S.guidedLabeling.active = true;
  S.review.showOverlays = true;
  updateGuidedLabelRegions(d);
  overlay.style.display = 'flex';
  const instr = $('guided-instruction');
  const phase = $('guided-phase');
  const stepLabel = $('guided-step-label');
  const userPrompt = $('guided-user-prompt');
  const tipsEl = $('guided-tips');
  const tipsList = $('guided-tips-list');
  const countdown = $('guided-countdown');
  const meters = $('guided-meters');
  const quality = $('guided-quality');
  const btnReady = $('guided-btn-ready');
  const btnRedo = $('guided-btn-redo');
  const btnNext = $('guided-btn-next');

  if (instr) instr.textContent = d.message || '';
  if (phase) phase.textContent = d.phase || '';
  if (stepLabel) stepLabel.textContent = `Step ${(d.current_step||0)+1} / ${d.total_steps||'?'}`;
  if (userPrompt) userPrompt.textContent = d.current_user_prompt || '';

  // Tips
  if (tipsEl && tipsList) {
    const tips = d.current_tips || [];
    if (tips.length > 0) {
      tipsEl.style.display = 'block';
      tipsList.innerHTML = tips.map(t => `<div>• ${t}</div>`).join('');
    } else {
      tipsEl.style.display = 'none';
    }
  }

  // Phase-dependent UI
  if (countdown) countdown.style.display = 'none';
  if (meters) meters.style.display = 'none';
  if (quality) quality.style.display = 'none';
  if (btnReady) btnReady.style.display = 'none';
  if (btnRedo) btnRedo.style.display = 'none';
  if (btnNext) btnNext.style.display = 'none';

  if (d.phase === 'waiting' && d.waiting_for_user) {
    if (btnReady) btnReady.style.display = 'inline-flex';
  } else if (d.phase === 'countdown') {
    if (countdown) {
      countdown.style.display = 'block';
      const match = (d.message || '').match(/(\d)/);
      countdown.textContent = match ? match[1] : '•';
    }
  } else if (d.phase === 'recording') {
    if (meters) meters.style.display = 'block';
  } else if (d.phase === 'quality') {
    if (quality) {
      quality.style.display = 'block';
      const sq = d.step_quality || {};
      const gradeEl = $('guided-quality-grade');
      const detailEl = $('guided-quality-detail');
      const icons = { good: 'Good', fair: 'Fair', poor: 'Poor' };
      const colors = { good: 'rgba(62,207,142,0.06)', fair: 'rgba(255,176,32,0.06)', poor: 'rgba(193,92,112,0.06)' };
      const borders = { good: 'rgba(62,207,142,0.15)', fair: 'rgba(255,176,32,0.15)', poor: 'rgba(193,92,112,0.15)' };
      if (gradeEl) gradeEl.textContent = icons[sq.grade] || sq.grade;
      if (detailEl) detailEl.textContent = sq.message || `${sq.windows || 0} windows · SNR ${sq.snr || 0}`;
      quality.style.background = colors[sq.grade] || colors.fair;
      quality.style.borderColor = borders[sq.grade] || borders.fair;
    }
    if (d.can_redo) {
      if (btnRedo) btnRedo.style.display = 'inline-flex';
      if (btnNext) btnNext.style.display = 'inline-flex';
    }
  }
  _updateGuidedMeters(d);
}

let _guidedStatusPollTimer = null;

function stopGuidedStatusPolling() {
  if (_guidedStatusPollTimer) {
    clearInterval(_guidedStatusPollTimer);
    _guidedStatusPollTimer = null;
  }
}

function startGuidedStatusPolling() {
  stopGuidedStatusPolling();
  const poll = async () => {
    try {
      const resp = await fetch('/api/recording/guided/status');
      if (!resp.ok) return;
      const state = await resp.json();
      if (!state || !state.active) {
        if (state?.phase === 'complete') onGuidedStep(state);
        stopGuidedStatusPolling();
        return;
      }
      onGuidedStep(state);
    } catch (e) {
      // WebSocket remains the primary path; polling is only a visibility fallback.
    }
  };
  poll();
  _guidedStatusPollTimer = setInterval(poll, 500);
}

function updateGuidedLabelRegions(d) {
  if (!S.guidedLabeling?.active || !d) return;
  if (!Array.isArray(S.guidedLabeling.regions)) S.guidedLabeling.regions = [];

  const phase = String(d.phase || '');
  const stepIdx = Number(d.current_step || 0);
  const step = Array.isArray(d.steps) ? d.steps[stepIdx] : null;
  const gesture = String(d.current_gesture || step?.gesture || '').trim();
  const instruction = String(d.current_instruction || step?.instruction || d.message || '').trim();

  if (phase !== 'recording') {
    S.guidedLabeling.activeRegionKey = '';
    return;
  }

  const label = gesture || instruction || `Step ${stepIdx + 1}`;
  const rep = Number(step?.current_rep || 0);
  const key = `${stepIdx}:${rep}:${label}`;
  const endSample = Math.max(0, Number(S.emgTotal || 0) - 1);
  if (S.guidedLabeling.activeRegionKey !== key) {
    S.guidedLabeling.activeRegionKey = key;
    S.guidedLabeling.activeRegionStartSample = Math.max(0, endSample - Math.round(Number(S.sampleRate || 250) * 0.2));
  }

  const startSample = Math.max(0, Number(S.guidedLabeling.activeRegionStartSample || 0));
  const detail = `Training label: ${label}`;
  S.guidedLabeling.regions = (S.guidedLabeling.regions || [])
    .filter(item => item.source !== 'guided_label' || item.key !== key);

  for (let channel = 0; channel < N_CH; channel += 1) {
    S.guidedLabeling.regions.push({
      key,
      source: 'guided_label',
      kind: 'focus',
      label,
      detail,
      channel,
      startSample,
      endSample,
      color: 'rgba(217,119,86,0.13)',
      line: 'rgba(217,119,86,0.82)',
    });
  }

  const maxRegions = N_CH * 18;
  if (S.guidedLabeling.regions.length > maxRegions) {
    S.guidedLabeling.regions = S.guidedLabeling.regions.slice(-maxRegions);
  }
}

async function _autoTrainAndGenCode() {
  const status = $('intent-status');
  try {
    const clf = 'LDA';
    const r = await post(`/api/train/fit?classifier=${clf}`);
    if (r && r.accuracy !== undefined) {
      const pct = Math.round(r.accuracy * 100);
      if (status) status.innerHTML = `<strong>Model trained: ${pct}%</strong> — generating project code...`;
      toast(`Model trained: ${pct}% accuracy`, 'green');
    }
  } catch(e) {
    if (status) status.innerHTML = 'Auto-train note: ' + (e.message || e);
  }
  // Generate code
  try {
    const prompt = S.guidedLabeling?.prompt || $('intent-input')?.value || 'Biosignal project';
    const apiKey = getCodeAIKey();
    const provider = getCodeAIProvider();
    const baseUrl = getCodeAIBaseUrl();
    const model = getCodeAIModel();
    const resp = await fetch('/api/codegen', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, api_key: apiKey, provider, base_url: baseUrl, model })
    });
    const data = await resp.json();
    if (data.ok && data.project) {
      _loadProjectIntoIDE(data.project);
      window.switchTab('code');
      if (status) status.innerHTML = `<strong>Done!</strong> ${data.project.files?.length || 0} files generated — view in Code tab`;
      toast('Code generated — opening IDE', 'green');
    }
  } catch(e) {
    if (status) status.innerHTML += ' | Code gen: ' + (e.message || e);
  }
}

function onGuidedProgress(d) {
  const overlay = $('guided-overlay');
  if (!overlay || overlay.style.display === 'none') return;
  _updateGuidedMeters(d);
}

function _updateGuidedMeters(d) {
  const act = $('guided-activation');
  const stab = $('guided-stability');
  const prog = $('guided-progress');
  const ready = $('guided-readiness');
  if (act) act.style.width = `${Math.min(100, (d.live_activation || 0) * 200)}%`;
  if (stab) stab.style.width = `${(d.live_stability || 0) * 100}%`;
  if (prog) prog.style.width = `${(d.progress || 0) * 100}%`;
  if (ready) ready.textContent = `Model Readiness: ${Math.round((d.model_readiness || 0) * 100)}% · ${d.windows_collected || 0}/${d.windows_target || 200} windows`;
}

window.confirmGuidedStep = async function() {
  const btnReady = $('guided-btn-ready');
  const btnNext = $('guided-btn-next');
  if (btnReady) btnReady.disabled = true;
  if (btnNext) btnNext.disabled = true;
  try {
    const resp = await fetch('/api/recording/guided/confirm', { method: 'POST' });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || data.ok === false) {
      throw new Error(data.detail || data.error || `Guided confirm failed (${resp.status})`);
    }
    if (data.session) onGuidedStep(data.session);
  } catch(e) {
    toast('Could not advance guided step: ' + (e.message || e), 'red');
  } finally {
    if (btnReady) btnReady.disabled = false;
    if (btnNext) btnNext.disabled = false;
  }
};

window.redoGuidedStep = async function() {
  try {
    const resp = await fetch('/api/recording/guided/redo', { method: 'POST' });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || data.ok === false) {
      throw new Error(data.detail || data.error || `Guided redo failed (${resp.status})`);
    }
    if (data.session) onGuidedStep(data.session);
  } catch(e) {
    toast('Could not redo guided step: ' + (e.message || e), 'red');
  }
};

window.cancelGuidedRecording = async function() {
  try {
    await fetch('/api/recording/guided/cancel', { method: 'POST' });
  } catch(e) {}
  S.guidedLabeling.active = false;
  S.guidedLabeling.activeRegionKey = '';
  S.guidedLabeling.regions = [];
  S.review.showOverlays = false;
  const overlay = $('guided-overlay');
  if (overlay) overlay.style.display = 'none';
};

// ── Intent Bar — Full Auto-Pipeline Orchestrator ─────────────────────────────

(function initIntentBar() {
  const input = $('intent-input');
  const submit = $('intent-submit');
  const status = $('intent-status');
  const glow = $('intent-glow');
  const apiKeyInput = $('intent-api-key');
  const providerSelect = $('intent-provider');
  if (!input || !submit) return;

  input.addEventListener('focus', () => { if (glow) glow.style.opacity = '.35'; });
  input.addEventListener('blur', () => { if (glow) glow.style.opacity = '0'; });

  async function _setStatus(html) { if (status) status.innerHTML = html; }

  async function _waitForStream(maxWaitMs = 8000) {
    const start = Date.now();
    while (Date.now() - start < maxWaitMs) {
      await new Promise(r => setTimeout(r, 400));
      if (S.streaming) return true;
    }
    return S.streaming;
  }

  async function submitIntent() {
    const prompt = input.value.trim();
    if (!prompt) return;
    submit.disabled = true;
    if (glow) glow.style.opacity = '.6';

    const apiKey = apiKeyInput?.value?.trim() || getCodeAIKey();
    const provider = apiKeyInput?.value?.trim() ? (providerSelect?.value || 'deepseek') : getCodeAIProvider();
    const baseUrl = apiKeyInput?.value?.trim() ? '' : getCodeAIBaseUrl();
    const model = apiKeyInput?.value?.trim() ? '' : getCodeAIModel();

    try {
      // ─── Step 1: Parse intent ───────────────────────────────────
      _setStatus('<strong>Step 1/5</strong> — Analyzing intent...');
      const resp = await fetch('/api/intent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, api_key: apiKey, provider, base_url: baseUrl, model })
      });
      const data = await resp.json();
      if (!data.ok || !data.plan) {
        _setStatus(data.detail || 'Intent parsing failed');
        return;
      }
      const plan = data.plan;
      toast(`${plan.task_type}: ${plan.suggested_gestures?.join(', ')}`, 'green');

      // ─── Step 2: Start synthetic stream if not streaming ─────────
      if (!S.streaming) {
        _setStatus('<strong>Step 2/5</strong> — Starting synthetic stream...');
        try {
          await fetch('/api/stream/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: 'synthetic' })
          });
          await _waitForStream(6000);
        } catch(e) {
          _setStatus('Stream start failed: ' + e.message);
          return;
        }
      }

      // ─── Step 3: Start guided recording ──────────────────────────
      _setStatus(`<strong>Step 3/5</strong> — Guided recording (${plan.suggested_gestures?.length || '?'} gestures). Follow overlay prompts.`);
      await new Promise(r => setTimeout(r, 800));
      const protocol = plan.recording_protocol?.length ? plan.recording_protocol : null;
      const targetWindows = plan.dataset_targets?.recommended_windows || 200;
      try {
        const guidedResp = await fetch('/api/recording/guided/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ protocol, target_windows: targetWindows })
        });
        const guidedData = await guidedResp.json().catch(() => ({}));
        if (!guidedResp.ok || guidedData.ok === false) {
          throw new Error(guidedData.detail || guidedData.error || `Guided recording failed (${guidedResp.status})`);
        }
        if (guidedData.session) onGuidedStep(guidedData.session);
        startGuidedStatusPolling();
        toast('Guided recording started — follow the prompts', 'green');
        // Steps 4 (train) and 5 (codegen) fire automatically when recording completes
      } catch(e) {
        _setStatus('Could not start guided recording: ' + e.message);
      }
    } catch(e) {
      _setStatus(e.message);
    } finally {
      submit.disabled = false;
      if (glow) glow.style.opacity = '0';
    }
  }

  submit.addEventListener('click', submitIntent);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') submitIntent(); });
})();

// ── Pipeline Timeline Updater ────────────────────────────────────────────────

function updatePipelineTimeline(conf) {
  const stages = document.querySelectorAll('.timeline-stage');
  if (!stages.length) return;
  // Determine active stage from system state
  let activeStage = 'setup';
  const state = S.sysState || 'idle';
  if (state === 'plan_generated' || state === 'streaming') activeStage = 'record';
  if (state === 'guided_recording') activeStage = 'record';
  if (state === 'training' || state === 'progressive_training') activeStage = 'train';
  if (S.modelTrained) activeStage = 'validate';
  if (conf && conf.model_viability > 0.85 && S.modelTrained) activeStage = 'export';

  const stageOrder = ['setup', 'record', 'train', 'validate', 'export'];
  const activeIdx = stageOrder.indexOf(activeStage);

  stages.forEach(el => {
    const stage = el.dataset.stage;
    const idx = stageOrder.indexOf(stage);
    const dot = el.querySelector('div > div');
    const ring = el.querySelector('div');
    const label = el.querySelector('span');
    if (idx <= activeIdx) {
      if (ring) { ring.style.borderColor = 'var(--accent)'; ring.style.boxShadow = '0 0 8px rgba(88,111,218,.3)'; }
      if (dot) dot.style.background = 'var(--accent)';
      if (label) { label.style.color = 'var(--accent)'; }
      el.classList.add('active');
    } else {
      if (ring) { ring.style.borderColor = 'var(--border)'; ring.style.boxShadow = 'none'; }
      if (dot) dot.style.background = 'var(--text-dim)';
      if (label) label.style.color = 'var(--text-dim)';
      el.classList.remove('active');
    }
  });
  // Color connectors
  const connectors = document.querySelectorAll('#pipeline-timeline > div > div:not(.timeline-stage)');
  connectors.forEach((c, i) => {
    c.style.background = i < activeIdx ? 'var(--accent)' : 'var(--border)';
  });
}

// ── Dark Surgical Theme Toggle ───────────────────────────────────────────────

(function initSurgicalTheme() {
  const sel = $('theme-select');
  if (!sel) return;
  // Add 'surgical' option if missing
  if (!sel.querySelector('option[value="surgical"]')) {
    const opt = document.createElement('option');
    opt.value = 'surgical';
    opt.textContent = 'SURGICAL';
    sel.appendChild(opt);
  }
  const saved = localStorage.getItem('kyma-theme');
  if (saved === 'surgical') {
    sel.value = 'surgical';
    applySurgicalTheme(true);
  }
  sel.addEventListener('change', () => {
    if (sel.value === 'surgical') {
      applySurgicalTheme(true);
      localStorage.setItem('kyma-theme', 'surgical');
    } else {
      applySurgicalTheme(false);
      localStorage.setItem('kyma-theme', sel.value);
    }
  });
})();

function applySurgicalTheme(on) {
  const r = document.documentElement.style;
  if (on) {
    document.body.classList.add('theme-surgical');
    r.setProperty('--bg', '#0a0e14');
    r.setProperty('--surface', 'rgba(16,20,28,0.85)');
    r.setProperty('--surface-strong', 'rgba(20,24,34,0.95)');
    r.setProperty('--text', '#e8ecf5');
    r.setProperty('--text-dim', '#6b7a99');
    r.setProperty('--border', 'rgba(255,255,255,0.06)');
    r.setProperty('--glass-border', 'rgba(255,255,255,0.08)');
    r.setProperty('--glass-shadow', '0 4px 24px rgba(0,0,0,0.5)');
    r.setProperty('--accent', '#586fda');
    r.setProperty('--cyan', '#00d4ff');
    r.setProperty('--green', '#3ecf8e');
    r.setProperty('--yellow', '#ffb020');
    r.setProperty('--red', '#c15c70');
  } else {
    document.body.classList.remove('theme-surgical');
    r.removeProperty('--bg');
    r.removeProperty('--surface');
    r.removeProperty('--surface-strong');
    r.removeProperty('--text');
    r.removeProperty('--text-dim');
    r.removeProperty('--border');
    r.removeProperty('--glass-border');
    r.removeProperty('--glass-shadow');
    r.removeProperty('--accent');
    r.removeProperty('--cyan');
    r.removeProperty('--green');
    r.removeProperty('--yellow');
    r.removeProperty('--red');
  }
}

// ── Fade-in animation for intervention cards ─────────────────────────────────

(function() {
  const style = document.createElement('style');
  style.textContent = `
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(8px); }
      to   { opacity: 1; transform: translateY(0); }
    }
  `;
  document.head.appendChild(style);
})();

// =============================================================================
// CODE IDE — File Viewer, Syntax Highlighting, Download
// =============================================================================

let _codeProject = null;
let _codeActiveIdx = 0;
let _codePendingProject = null;
let _codeEditor = null;
let _codeEditorModel = null;
let _codeMonacoLoading = null;
let _codeMonacoReady = false;
let _codeEditorViewStates = new Map();
let _codeEditorMarkers = new Map();
const MONACO_EDITOR_VERSION = '0.55.1';
const MONACO_EDITOR_BASE = `https://cdn.jsdelivr.net/npm/monaco-editor@${MONACO_EDITOR_VERSION}/min`;

function codeLanguageForMonaco(lang) {
  const map = {
    python: 'python', javascript: 'javascript', typescript: 'typescript',
    html: 'html', css: 'css', json: 'json', cpp: 'cpp', c: 'c',
    markdown: 'markdown', yaml: 'yaml', bash: 'shell', batch: 'bat',
    plaintext: 'plaintext'
  };
  return map[lang] || 'plaintext';
}

function codeLanguageFromPath(path = '') {
  const ext = String(path).toLowerCase().split('.').pop();
  const map = {
    py: 'python', js: 'javascript', mjs: 'javascript', cjs: 'javascript',
    ts: 'typescript', html: 'html', htm: 'html', css: 'css',
    json: 'json', cpp: 'cpp', cc: 'cpp', cxx: 'cpp', h: 'c', c: 'c',
    md: 'markdown', markdown: 'markdown', yml: 'yaml', yaml: 'yaml',
    sh: 'bash', bash: 'bash', bat: 'batch', txt: 'plaintext',
  };
  return map[ext] || 'plaintext';
}

function codeMonacoUriForPath(path = 'untitled.txt') {
  const clean = String(path || 'untitled.txt').replaceAll('\\', '/').replace(/^\/+/, '');
  const encoded = clean.split('/').map(part => encodeURIComponent(part)).join('/');
  return monaco.Uri.parse(`file:///kyma/${encoded || 'untitled.txt'}`);
}

function codeDisposeMonacoModels() {
  if (!window.monaco) return;
  monaco.editor.getModels()
    .filter(model => String(model.uri || '').startsWith('file:///kyma/'))
    .forEach(model => model.dispose());
  _codeEditorModel = null;
  _codeEditorViewStates = new Map();
  _codeEditorMarkers = new Map();
}

function codeEnsureMonacoWorkerConfig() {
  if (window.MonacoEnvironment?.__kymaConfigured) return;
  const workerMain = `${MONACO_EDITOR_BASE}/vs/base/worker/workerMain.js`;
  window.MonacoEnvironment = {
    __kymaConfigured: true,
    getWorkerUrl() {
      const source = `self.MonacoEnvironment={baseUrl:'${MONACO_EDITOR_BASE}/'};importScripts('${workerMain}');`;
      return `data:text/javascript;charset=utf-8,${encodeURIComponent(source)}`;
    },
  };
}

function codeApplyMonacoMarkers(path, markers = []) {
  if (!window.monaco) return;
  const model = monaco.editor.getModel(codeMonacoUriForPath(path));
  if (!model) return;
  monaco.editor.setModelMarkers(model, 'kyma-checks', markers);
}

function codeParseCheckLine(message = '') {
  const match = String(message).match(/(?:line|:)(\d+)(?::(\d+))?/i);
  return match ? Math.max(1, Number(match[1]) || 1) : 1;
}

function codeUpdateEditorDiagnostics(results = []) {
  _codeEditorMarkers = new Map();
  results.forEach(item => {
    if (String(item.status || '').toLowerCase() !== 'fail') return;
    const path = String(item.path || '');
    const line = codeParseCheckLine(item.message || '');
    const markers = _codeEditorMarkers.get(path) || [];
    markers.push({
      severity: window.monaco?.MarkerSeverity?.Error || 8,
      message: String(item.message || 'Check failed'),
      startLineNumber: line,
      startColumn: 1,
      endLineNumber: line,
      endColumn: 120,
    });
    _codeEditorMarkers.set(path, markers);
  });
  if (!window.monaco) return;
  (_codeProject?.files || []).forEach(file => {
    codeApplyMonacoMarkers(file.path, _codeEditorMarkers.get(file.path) || []);
  });
}

function ensureMonacoEditor() {
  if (_codeEditor) return Promise.resolve(_codeEditor);
  if (_codeMonacoLoading) return _codeMonacoLoading;
  _codeMonacoLoading = new Promise((resolve, reject) => {
    const container = $('code-monaco-editor');
    if (!container) {
      reject(new Error('Monaco container missing'));
      return;
    }
    const finish = () => {
      try {
        codeEnsureMonacoWorkerConfig();
        require.config({ paths: { vs: `${MONACO_EDITOR_BASE}/vs` } });
        require(['vs/editor/editor.main'], () => {
          monaco.editor.defineTheme('kyma-code-dark', {
            base: 'vs-dark',
            inherit: true,
            rules: [
              { token: 'comment', foreground: '6a737d', fontStyle: 'italic' },
              { token: 'keyword', foreground: 'ff7b72' },
              { token: 'string', foreground: 'a5d6ff' },
              { token: 'number', foreground: '79c0ff' },
            ],
            colors: {
              'editor.background': '#0d1117',
              'editor.foreground': '#c9d1d9',
              'editorLineNumber.foreground': '#484f58',
              'editorLineNumber.activeForeground': '#8b949e',
              'editorCursor.foreground': '#f78166',
              'editor.selectionBackground': '#264f78',
              'editor.inactiveSelectionBackground': '#1f6feb33',
            },
          });
          _codeEditor = monaco.editor.create(container, {
            value: '',
            language: 'plaintext',
            theme: 'kyma-code-dark',
            automaticLayout: true,
            minimap: { enabled: true, showSlider: 'always' },
            fontSize: 12,
            lineHeight: 20,
            wordWrap: 'off',
            scrollBeyondLastLine: false,
            tabSize: 2,
            insertSpaces: true,
            detectIndentation: true,
            renderWhitespace: 'selection',
            renderControlCharacters: true,
            bracketPairColorization: { enabled: true },
            guides: { bracketPairs: true, indentation: true },
            folding: true,
            glyphMargin: true,
            lightbulb: { enabled: true },
            links: true,
            contextmenu: true,
            quickSuggestions: true,
            suggestOnTriggerCharacters: true,
            parameterHints: { enabled: true },
            formatOnPaste: true,
            formatOnType: true,
            smoothScrolling: true,
            cursorSmoothCaretAnimation: 'on',
            mouseWheelZoom: true,
          });
          _codeEditor.onDidChangeModelContent(() => {
            if (!_codeProject || !_codeProject.files?.[_codeActiveIdx]) return;
            _codeProject.files[_codeActiveIdx].content = _codeEditor.getValue();
          });
          _codeMonacoReady = true;
          resolve(_codeEditor);
        });
      } catch (e) {
        reject(e);
      }
    };
    if (window.monaco && window.require) {
      finish();
      return;
    }
    const loader = document.createElement('script');
    loader.src = `${MONACO_EDITOR_BASE}/vs/loader.js`;
    loader.onload = finish;
    loader.onerror = () => reject(new Error('Failed to load Monaco editor'));
    document.head.appendChild(loader);
  }).catch(e => {
    const monacoEl = $('code-monaco-editor');
    const preEl = $('code-ide-pre');
    if (monacoEl) monacoEl.style.display = 'none';
    if (preEl) preEl.style.display = 'block';
    throw e;
  });
  return _codeMonacoLoading;
}

function syncActiveCodeFromEditor() {
  if (_codeEditor && _codeProject?.files?.[_codeActiveIdx]) {
    _codeProject.files[_codeActiveIdx].content = _codeEditor.getValue();
  }
}

function ensureBlankCodeWorkspace() {
  if (_codeProject?.files?.length) return;
  _loadProjectIntoIDE({
    summary: 'untitled_project',
    files: [
      {
        path: 'untitled.txt',
        language: 'plaintext',
        description: 'Untitled file',
        content: '',
      },
    ],
  });
  const statusEl = $('code-ide-ai-status');
  if (statusEl) statusEl.textContent = 'Editor ready. Ask KYMA for changes when you want help.';
  setCodeAgentOutput('Editor ready. Create or edit files, then use Check or Save.');
}

window.addCodeIdeFile = function() {
  ensureBlankCodeWorkspace();
  syncActiveCodeFromEditor();
  const path = window.prompt('New file path:', 'src/app.js');
  const cleanPath = String(path || '').trim().replaceAll('\\', '/').replace(/^\/+/, '');
  if (!cleanPath) return;
  if (_codeProject.files.some(file => file.path === cleanPath)) {
    toast('File already exists', 'yellow');
    return;
  }
  _codeProject.files.push({
    path: cleanPath,
    language: codeLanguageFromPath(cleanPath),
    description: 'User-created file',
    content: '',
  });
  _loadProjectIntoIDE(_codeProject);
  _selectCodeFile(_codeProject.files.length - 1);
  appendCodeChatMessage('assistant', `Created ${cleanPath}.`);
};

window.addCodeIdeFolder = function() {
  ensureBlankCodeWorkspace();
  syncActiveCodeFromEditor();
  const name = window.prompt('New folder path:', 'src');
  const folder = String(name || '').trim().replaceAll('\\', '/').replace(/^\/+|\/+$/g, '');
  if (!folder) return;
  const path = `${folder}/.gitkeep`;
  if (_codeProject.files.some(file => file.path === path)) {
    toast('Folder already exists', 'yellow');
    return;
  }
  _codeProject.files.push({
    path,
    language: 'plaintext',
    description: 'Folder placeholder',
    content: '',
  });
  _loadProjectIntoIDE(_codeProject);
  _selectCodeFile(_codeProject.files.length - 1);
  appendCodeChatMessage('assistant', `Created folder ${folder}.`);
};

function _loadProjectIntoIDE(project) {
  syncActiveCodeFromEditor();
  codeDisposeMonacoModels();
  _codeProject = project;
  _codeActiveIdx = 0;
  const fileList = $('code-ide-file-list');
  const emptyEl = $('code-ide-empty');
  const preEl = $('code-ide-pre');
  const nameEl = $('code-ide-project-name');
  const countEl = $('code-ide-file-count');
  if (!fileList) return;

  const files = project.files || [];
  if (nameEl) nameEl.textContent = project.summary || '';
  if (countEl) countEl.textContent = `${files.length} files`;

  // Build file tree
  const iconMap = {
    python: 'PY', javascript: 'JS', typescript: 'TS', html: 'HTML', css: 'CSS',
    cpp: 'CPP', c: 'C', json: 'JSON', markdown: 'MD', yaml: 'YAML',
    bash: 'SH', plaintext: 'TXT', batch: 'BAT'
  };
  fileList.innerHTML = files.map((f, i) => {
    const icon = iconMap[f.language] || 'TXT';
    const name = f.path.split('/').pop();
    const dir = f.path.includes('/') ? f.path.substring(0, f.path.lastIndexOf('/') + 1) : '';
    return `<div class="code-file-item ${i === 0 ? 'active' : ''}" data-idx="${i}" onclick="_selectCodeFile(${i})">
      <span>${icon}</span>
      <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${f.path}">
        ${dir ? '<span style="color:#484f58">' + dir + '</span>' : ''}${name}
      </span>
    </div>`;
  }).join('');

  if (files.length > 0) {
    if (emptyEl) emptyEl.style.display = 'none';
    if (preEl) preEl.style.display = 'block';
    _selectCodeFile(0);
  }
}

window._loadProjectIntoIDE = _loadProjectIntoIDE;

function _selectCodeFile(idx) {
  if (!_codeProject) return;
  const files = _codeProject.files || [];
  if (idx < 0 || idx >= files.length) return;
  if (_codeEditor && files[_codeActiveIdx]) {
    _codeEditorViewStates.set(files[_codeActiveIdx].path, _codeEditor.saveViewState());
  }
  syncActiveCodeFromEditor();
  _codeActiveIdx = idx;
  const f = files[idx];

  // Update tree selection
  document.querySelectorAll('#code-ide-file-list .code-file-item').forEach(el => {
    el.classList.toggle('active', parseInt(el.dataset.idx) === idx);
  });

  // Update header
  const nameEl = $('code-ide-active-file');
  const langEl = $('code-ide-active-lang');
  const descEl = $('code-ide-active-desc');
  if (nameEl) nameEl.textContent = f.path;
  if (langEl) langEl.textContent = f.language;
  if (descEl) descEl.textContent = f.description;

  // Render code with line numbers + syntax coloring
  const codeEl = $('code-ide-code');
  if (codeEl) {
    const lines = f.content.split('\n');
    const highlighted = lines.map((line, i) => {
      const num = `<span class="line-number">${i + 1}</span>`;
      return num + _highlightLine(_escHtml(line), f.language);
    }).join('\n');
    codeEl.innerHTML = highlighted;
  }
  ensureMonacoEditor().then(editor => {
    const monacoEl = $('code-monaco-editor');
    const preEl = $('code-ide-pre');
    if (monacoEl) monacoEl.style.display = 'block';
    if (preEl) preEl.style.display = 'none';
    const uri = codeMonacoUriForPath(f.path);
    const language = codeLanguageForMonaco(f.language || codeLanguageFromPath(f.path));
    _codeEditorModel = monaco.editor.getModel(uri);
    if (!_codeEditorModel) {
      _codeEditorModel = monaco.editor.createModel(f.content || '', language, uri);
    } else {
      if (_codeEditorModel.getValue() !== (f.content || '')) {
        _codeEditorModel.setValue(f.content || '');
      }
      monaco.editor.setModelLanguage(_codeEditorModel, language);
    }
    editor.setModel(_codeEditorModel);
    const viewState = _codeEditorViewStates.get(f.path);
    if (viewState) editor.restoreViewState(viewState);
    codeApplyMonacoMarkers(f.path, _codeEditorMarkers.get(f.path) || []);
    editor.layout();
    editor.focus();
  }).catch(() => {
    const monacoEl = $('code-monaco-editor');
    const preEl = $('code-ide-pre');
    if (monacoEl) monacoEl.style.display = 'none';
    if (preEl) preEl.style.display = 'block';
  });

  // Scroll to top
  const content = $('code-ide-content');
  if (content) content.scrollTop = 0;
}

function _escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function _highlightLine(line, lang) {
  if (!line.trim()) return line;

  // Comments
  if (lang === 'python' || lang === 'bash') {
    const cm = line.match(/^(\s*)(#.*)$/);
    if (cm) return cm[1] + '<span class="syn-cm">' + cm[2] + '</span>';
  }
  if (lang === 'cpp' || lang === 'c' || lang === 'javascript' || lang === 'typescript') {
    const cm = line.match(/^(\s*)(\/\/.*)$/);
    if (cm) return cm[1] + '<span class="syn-cm">' + cm[2] + '</span>';
  }
  if (lang === 'html') {
    const cm = line.match(/^(\s*)(&lt;!--.*)$/);
    if (cm) return cm[1] + '<span class="syn-cm">' + cm[2] + '</span>';
  }

  // Strings
  line = line.replace(/(["'])(?:(?!\1|\\).|\\.)*?\1/g, '<span class="syn-str">$&</span>');
  line = line.replace(/("""[\s\S]*?"""|'''[\s\S]*?''')/g, '<span class="syn-str">$&</span>');

  // Numbers
  line = line.replace(/\b(\d+\.?\d*)\b/g, '<span class="syn-num">$1</span>');

  // Python keywords
  if (lang === 'python') {
    line = line.replace(/\b(import|from|def|class|return|if|elif|else|for|while|in|not|and|or|is|try|except|finally|with|as|pass|break|continue|raise|yield|async|await|True|False|None|self|print|lambda)\b/g, '<span class="syn-kw">$1</span>');
  }
  // JS/TS keywords
  if (lang === 'javascript' || lang === 'typescript') {
    line = line.replace(/\b(const|let|var|function|return|if|else|for|while|switch|case|break|continue|new|this|class|extends|import|export|from|async|await|try|catch|throw|true|false|null|undefined|typeof|instanceof)\b/g, '<span class="syn-kw">$1</span>');
  }
  // C/C++ keywords
  if (lang === 'cpp' || lang === 'c') {
    line = line.replace(/\b(void|int|float|double|char|bool|long|unsigned|signed|const|static|struct|class|return|if|else|for|while|switch|case|break|continue|include|define|true|false|NULL|Serial|String|Servo|HIGH|LOW|OUTPUT|INPUT|pinMode|digitalWrite|digitalRead|analogWrite|analogRead|delay|setup|loop|Serial)\b/g, '<span class="syn-kw">$1</span>');
  }
  // HTML tags
  if (lang === 'html') {
    line = line.replace(/(&lt;\/?)([\w-]+)/g, '$1<span class="syn-tag">$2</span>');
    line = line.replace(/\b(style|class|id|src|href|type|value|placeholder|onclick|onchange)\b(?==)/g, '<span class="syn-attr">$1</span>');
  }

  return line;
}

window.copyActiveCodeFile = function() {
  if (!_codeProject) return;
  syncActiveCodeFromEditor();
  const f = _codeProject.files[_codeActiveIdx];
  if (f) {
    navigator.clipboard.writeText(f.content);
    toast('Copied to clipboard!');
  }
};

window.downloadAllCodeFiles = function() {
  if (!_codeProject || !_codeProject.files.length) return;
  syncActiveCodeFromEditor();
  _codeProject.files.forEach(f => {
    const blob = new Blob([f.content], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = f.path.split('/').pop();
    a.click();
    URL.revokeObjectURL(a.href);
  });
  toast(`Downloaded ${_codeProject.files.length} files`);
};

function projectFilesByPath(project) {
  const map = new Map();
  (project?.files || []).forEach(file => map.set(file.path, file));
  return map;
}

function describeProjectDiff(before, after) {
  const a = projectFilesByPath(before);
  const b = projectFilesByPath(after);
  const paths = new Set([...a.keys(), ...b.keys()]);
  const changes = [];
  paths.forEach(path => {
    if (!a.has(path)) changes.push({ path, kind: 'added' });
    else if (!b.has(path)) changes.push({ path, kind: 'removed' });
    else if ((a.get(path).content || '') !== (b.get(path).content || '')) changes.push({ path, kind: 'modified' });
  });
  return changes;
}

function renderPendingCodeReview(project) {
  _codePendingProject = project;
  const panel = $('code-agent-review');
  const summary = $('code-agent-review-summary');
  const list = $('code-agent-review-list');
  const changes = describeProjectDiff(_codeProject, project);
  if (!panel || !summary || !list) return;
  panel.style.display = 'block';
  summary.textContent = `${changes.length} file change${changes.length === 1 ? '' : 's'} ready`;
  list.innerHTML = (changes.length ? changes : [{ path: 'No file content changes detected', kind: 'info' }]).map(change =>
    `<span class="code-agent-file-pill">${escapeHTML(change.kind)}: ${escapeHTML(change.path)}</span>`
  ).join('');
}

window.applyPendingCodeProject = function() {
  if (!_codePendingProject) return;
  _loadProjectIntoIDE(_codePendingProject);
  _codePendingProject = null;
  const panel = $('code-agent-review');
  if (panel) panel.style.display = 'none';
  appendCodeChatMessage('assistant', 'Applied the pending file changes.');
};

window.rejectPendingCodeProject = function() {
  _codePendingProject = null;
  const panel = $('code-agent-review');
  if (panel) panel.style.display = 'none';
  appendCodeChatMessage('assistant', 'Rejected the pending file changes.');
};

function setCodeAgentOutput(lines) {
  const log = $('code-agent-output-log');
  if (log) log.textContent = Array.isArray(lines) ? lines.join('\n') : String(lines || '');
}

window.checkGeneratedProject = async function() {
  if (!_codeProject?.files?.length) {
    toast('Generate code before running checks.', 'yellow');
    return;
  }
  syncActiveCodeFromEditor();
  setCodeAgentOutput('Running safe static checks...');
  try {
    const resp = await fetch('/api/codegen/check', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project: _codeProject })
    });
    const data = await resp.json().catch(() => ({}));
    const lines = (data.results || []).map(item => `[${String(item.status || '').toUpperCase()}] ${item.path}: ${item.message}`);
    codeUpdateEditorDiagnostics(data.results || []);
    setCodeAgentOutput(lines.length ? lines : 'No check results.');
    toast(data.ok ? 'Generated project checks passed' : 'Generated project has check failures', data.ok ? 'green' : 'red');
  } catch(e) {
    setCodeAgentOutput(e.message || String(e));
    toast('Generated project check failed: ' + (e.message || e), 'red');
  }
};

window.saveGeneratedProject = async function() {
  if (!_codeProject?.files?.length) {
    toast('Generate code before saving.', 'yellow');
    return;
  }
  syncActiveCodeFromEditor();
  try {
    const resp = await fetch('/api/codegen/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project: _codeProject })
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || data.ok === false) throw new Error(data.detail || data.error || `Save failed (${resp.status})`);
    setCodeAgentOutput(`Saved generated project:\n${data.root}`);
    appendCodeChatMessage('assistant', `Saved the generated project to ${data.root}.`);
    toast('Generated project saved', 'green');
  } catch(e) {
    setCodeAgentOutput(e.message || String(e));
    toast('Generated project save failed: ' + (e.message || e), 'red');
  }
};

function bindCodeIdeAIResize() {
  const handle = $('code-ide-dragbar');
  const shell = $('code-ide');
  if (!handle || !shell || handle.dataset.bound === '1') return;
  handle.dataset.bound = '1';

  let startX = 0;
  let startWidth = 0;
  let startY = 0;
  let startTop = 68;
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  function onSplitMove(e) {
    const minWidth = Math.max(520, Math.round(window.innerWidth * 0.5));
    const maxWidth = Math.max(minWidth, window.innerWidth - 28);
    const next = clamp(startWidth - (e.clientX - startX), minWidth, maxWidth);
    document.body.style.setProperty('--code-split-width', `${Math.round(next)}px`);
    document.body.classList.toggle('code-ide-max', next >= window.innerWidth - 96);
    shell.classList.remove('code-ide-peek');
  }

  function onSplitUp() {
    handle.classList.remove('active');
    document.body.classList.remove('resizing');
    window.removeEventListener('pointermove', onSplitMove);
    window.removeEventListener('pointerup', onSplitUp);
  }

  function onMove(e) {
    const maxTop = Math.max(96, window.innerHeight - 124);
    const nextTop = clamp(startTop + (e.clientY - startY), 54, maxTop);
    shell.classList.remove('code-ide-peek');
    shell.style.top = `${Math.round(nextTop)}px`;
    shell.style.bottom = '18px';
  }

  function onUp() {
    handle.classList.remove('active');
    document.body.classList.remove('resizing-v');
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
  }

  handle.addEventListener('pointerdown', e => {
    if (e.target && e.target.closest && e.target.closest('button')) return;
    if (document.body.classList.contains('code-split')) {
      const rect = shell.getBoundingClientRect();
      if (e.clientX <= rect.left + 18) {
        startX = e.clientX;
        startWidth = rect.width || 590;
        handle.classList.add('active');
        document.body.classList.add('resizing');
        window.addEventListener('pointermove', onSplitMove);
        window.addEventListener('pointerup', onSplitUp);
        e.preventDefault();
        return;
      }
      return;
    }
    startY = e.clientY;
    startTop = shell.getBoundingClientRect().top;
    handle.classList.add('active');
    document.body.classList.add('resizing-v');
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    e.preventDefault();
  });

  shell.addEventListener('pointerdown', e => {
    if (!document.body.classList.contains('code-split')) return;
    if (e.target && e.target.closest && e.target.closest('button, textarea, input, select')) return;
    const rect = shell.getBoundingClientRect();
    if (e.clientX > rect.left + 12) return;
    startX = e.clientX;
    startWidth = rect.width || 590;
    handle.classList.add('active');
    document.body.classList.add('resizing');
    window.addEventListener('pointermove', onSplitMove);
    window.addEventListener('pointerup', onSplitUp);
    e.preventDefault();
  });
}

function bindCodeIdeAIResizeLegacy() {
  const handle = $('code-ide-ai-resize');
  const shell = $('code-ide');
  const panel = $('code-ide-ai-panel');
  if (!handle || !shell || !panel || handle.dataset.bound === '1') return;
  handle.dataset.bound = '1';

  let startX = 0;
  let startWidth = 292;
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  function applyWidth(width) {
    const next = Math.round(width);
    shell.style.setProperty('--kyma-ai-panel-width', `${next}px`);
    panel.style.width = `${next}px`;
  }

  function onMove(e) {
    const maxWidth = Math.min(480, Math.max(220, window.innerWidth - 420));
    const next = clamp(startWidth - (e.clientX - startX), 220, maxWidth);
    applyWidth(next);
  }

  function onUp() {
    handle.classList.remove('active');
    document.body.classList.remove('resizing-h');
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
  }

  handle.addEventListener('pointerdown', e => {
    startX = e.clientX;
    startWidth = panel.getBoundingClientRect().width || 292;
    handle.classList.add('active');
    document.body.classList.add('resizing-h');
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    e.preventDefault();
  });

  applyWidth(panel.getBoundingClientRect().width || 292);
}

function restoreCodeIdePanel() {
  const shell = $('code-ide');
  if (!shell) return;
  shell.classList.remove('code-ide-peek');
  shell.style.top = '68px';
  shell.style.bottom = '14px';
}

window.collapseCodeIde = function() {
  const shell = $('code-ide');
  if (!shell) return;
  shell.classList.toggle('code-ide-peek');
  shell.style.top = '';
  shell.style.bottom = '18px';
};

function appendCodeChatMessage(role, text) {
  const log = $('code-ide-chat-log');
  if (!log) return;
  const msg = document.createElement('div');
  msg.className = `code-chat-msg ${role === 'user' ? 'user' : 'assistant'}`;
  const label = document.createElement('div');
  label.className = 'code-chat-role';
  label.textContent = role === 'user' ? 'You' : 'KYMA';
  const body = document.createElement('div');
  body.className = 'code-chat-body';
  body.textContent = text;
  msg.appendChild(label);
  msg.appendChild(body);
  log.appendChild(msg);
  log.scrollTop = log.scrollHeight;
}

const CODE_AI_KEY_PROVIDERS = ['ollama', 'nvidia', 'deepseek', 'openai', 'anthropic'];
const CODE_AI_PROVIDER_LABELS = {
  ollama: 'Local Ollama',
  nvidia: 'NVIDIA',
  deepseek: 'DeepSeek',
  openai: 'OpenAI',
  anthropic: 'Anthropic',
};
const CODE_AI_PROVIDER_DEFAULTS = {
  ollama: {
    api_key: 'ollama',
    base_url: 'http://127.0.0.1:11434/v1',
    model: 'qwen2.5-coder:7b',
  },
  nvidia: {
    base_url: 'https://integrate.api.nvidia.com/v1',
    model: 'deepseek-ai/deepseek-v4-pro',
  },
};

function normalizeCodeAIConfig(provider, config = {}) {
  const defaults = CODE_AI_PROVIDER_DEFAULTS[provider] || {};
  const apiKey = String(config.api_key || defaults.api_key || '').trim();
  const baseUrl = String(config.base_url || defaults.base_url || '').trim();
  const model = String(config.model || defaults.model || '').trim();
  const present = !!apiKey || (provider === 'ollama' && (!!baseUrl || !!model));
  if (!present) return null;
  return { api_key: apiKey, base_url: baseUrl, model };
}

function readCodeAIKeyStore() {
  let configs = {};
  try {
    const parsed = JSON.parse(localStorage.getItem('kyma-code-ai-keys') || '{}');
    if (parsed && typeof parsed === 'object') configs = parsed;
  } catch(e) {
    configs = {};
  }
  const legacyKey = (localStorage.getItem('kyma-code-ai-key') || '').trim();
  const legacyProvider = localStorage.getItem('kyma-code-ai-provider') || 'deepseek';
  if (legacyKey && CODE_AI_KEY_PROVIDERS.includes(legacyProvider) && !configs[legacyProvider]) {
    configs[legacyProvider] = legacyKey;
  }
  return CODE_AI_KEY_PROVIDERS.reduce((clean, provider) => {
    const raw = configs[provider];
    const config = raw && typeof raw === 'object' ? raw : { api_key: raw };
    const normalized = normalizeCodeAIConfig(provider, config);
    if (normalized) clean[provider] = normalized;
    return clean;
  }, {});
}

function writeCodeAIKeyStore(configs) {
  const clean = CODE_AI_KEY_PROVIDERS.reduce((out, provider) => {
    const config = configs?.[provider] || {};
    const normalized = normalizeCodeAIConfig(provider, config);
    if (normalized) out[provider] = normalized;
    return out;
  }, {});
  localStorage.setItem('kyma-code-ai-keys', JSON.stringify(clean));
  const active = getActiveCodeAIConfig(clean);
  if (active.api_key) {
    localStorage.setItem('kyma-code-ai-provider', active.provider);
    localStorage.setItem('kyma-code-ai-key', active.api_key);
  } else {
    localStorage.removeItem('kyma-code-ai-key');
  }
  return clean;
}

function getActiveCodeAIConfig(configs = readCodeAIKeyStore()) {
  const provider = CODE_AI_KEY_PROVIDERS.find(name => configs[name]?.api_key || (name === 'ollama' && configs[name]?.model)) || 'deepseek';
  const config = configs[provider] || {};
  return {
    provider,
    api_key: String(config.api_key || CODE_AI_PROVIDER_DEFAULTS[provider]?.api_key || '').trim(),
    base_url: String(config.base_url || CODE_AI_PROVIDER_DEFAULTS[provider]?.base_url || '').trim(),
    model: String(config.model || CODE_AI_PROVIDER_DEFAULTS[provider]?.model || '').trim(),
  };
}

function getCodeAIProvider() {
  const active = getActiveCodeAIConfig();
  return active.api_key ? active.provider : ($('intent-provider')?.value || 'deepseek');
}

function getCodeAIKey() {
  const active = getActiveCodeAIConfig();
  return active.api_key || ($('intent-api-key')?.value || '').trim();
}

function getCodeAIBaseUrl() {
  return getActiveCodeAIConfig().base_url;
}

function getCodeAIModel() {
  return getActiveCodeAIConfig().model;
}

function collectCodeAIKeyInputs() {
  return CODE_AI_KEY_PROVIDERS.reduce((configs, provider) => {
    const input = $(`code-ai-key-${provider}`);
    const apiKey = String(input?.value || '').trim();
    const baseUrl = String($(`code-ai-base-url-${provider}`)?.value || CODE_AI_PROVIDER_DEFAULTS[provider]?.base_url || '').trim();
    const model = String($(`code-ai-model-${provider}`)?.value || CODE_AI_PROVIDER_DEFAULTS[provider]?.model || '').trim();
    const normalized = normalizeCodeAIConfig(provider, { api_key: apiKey, base_url: baseUrl, model });
    if (normalized) configs[provider] = normalized;
    return configs;
  }, {});
}

function syncCodeAIKeyInputs() {
  const configs = readCodeAIKeyStore();
  const active = getActiveCodeAIConfig(configs);
  const savedProviders = CODE_AI_KEY_PROVIDERS.filter(provider => configs[provider]?.api_key);
  CODE_AI_KEY_PROVIDERS.forEach(provider => {
    const config = configs[provider] || {};
    const keyInput = $(`code-ai-key-${provider}`);
    const baseInput = $(`code-ai-base-url-${provider}`);
    const modelInput = $(`code-ai-model-${provider}`);
    if (keyInput && !keyInput.value) keyInput.value = config.api_key || '';
    if (baseInput && !baseInput.value) baseInput.value = config.base_url || CODE_AI_PROVIDER_DEFAULTS[provider]?.base_url || '';
    if (modelInput && !modelInput.value) modelInput.value = config.model || CODE_AI_PROVIDER_DEFAULTS[provider]?.model || '';
  });
  if ($('intent-provider')) $('intent-provider').value = active.provider;
  if ($('intent-api-key') && active.api_key && !$('intent-api-key').value) $('intent-api-key').value = active.api_key;
  const summaryEl = $('code-ai-key-summary');
  if (summaryEl) {
    summaryEl.textContent = savedProviders.length
      ? `Saved: ${savedProviders.map(provider => CODE_AI_PROVIDER_LABELS[provider]).join(', ')}`
      : 'No API keys saved.';
  }
  const statusEl = $('code-ide-ai-status');
  if (statusEl && active.api_key) statusEl.textContent = `AI key ready for ${CODE_AI_PROVIDER_LABELS[active.provider]}.`;
}

window.openCodeAIKeyPopup = function() {
  syncCodeAIKeyInputs();
  const modal = $('code-ai-key-modal');
  if (!modal) return;
  modal.hidden = false;
  const firstEmpty = CODE_AI_KEY_PROVIDERS
    .map(provider => $(`code-ai-key-${provider}`))
    .find(input => input && !input.value);
  (firstEmpty || $('code-ai-key-deepseek'))?.focus();
};

window.closeCodeAIKeyPopup = function() {
  const modal = $('code-ai-key-modal');
  if (modal) modal.hidden = true;
};

window.saveCodeAIKey = async function() {
  const keys = writeCodeAIKeyStore(collectCodeAIKeyInputs());
  const active = getActiveCodeAIConfig(keys);
  if (!active.api_key) {
    toast('Paste an API key first', 'yellow');
    return;
  }
  if ($('intent-provider')) $('intent-provider').value = active.provider;
  if ($('intent-api-key')) $('intent-api-key').value = active.api_key;
  const statusEl = $('code-ide-ai-status');
  if (statusEl) statusEl.textContent = `Saving key for ${CODE_AI_PROVIDER_LABELS[active.provider]}...`;
  try {
    await fetch('/api/intent/api_key', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: active.api_key,
        provider: active.provider,
        base_url: active.base_url,
        model: active.model,
      }),
    });
  } catch(e) {
    // The frontend still sends the saved key with each request if this call fails.
  }
  syncCodeAIKeyInputs();
  window.closeCodeAIKeyPopup();
  toast('AI keys saved for Code and Training', 'green');
};

window.testCodeAIKey = async function() {
  const configs = collectCodeAIKeyInputs();
  const active = getActiveCodeAIConfig(configs);
  if (!active.api_key) {
    toast('Paste an API key first', 'yellow');
    return;
  }
  const btn = $('code-ai-key-test');
  const statusEl = $('code-ide-ai-status');
  if (btn) btn.disabled = true;
  const label = CODE_AI_PROVIDER_LABELS[active.provider] || active.provider;
  if (statusEl) statusEl.textContent = `Testing ${label}. Local models can take a minute on first load...`;
  try {
    const resp = await fetch('/api/intent/api_key/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: active.api_key,
        provider: active.provider,
        base_url: active.base_url,
        model: active.model,
      }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || data.ok === false) {
      throw new Error(data.detail || data.error || `${label} test failed (${resp.status})`);
    }
    if (statusEl) statusEl.textContent = `${label} test passed in ${data.elapsed_s || '?'}s.`;
    toast(`${label} works (${data.elapsed_s || '?'}s)`, 'green');
  } catch(e) {
    if (statusEl) statusEl.textContent = e.message || `${label} test failed.`;
    toast(`${label} test failed: ${e.message || e}`, 'red');
  } finally {
    if (btn) btn.disabled = false;
  }
};

window.clearCodeAIKey = async function() {
  const provider = getCodeAIProvider();
  localStorage.removeItem('kyma-code-ai-keys');
  localStorage.removeItem('kyma-code-ai-key');
  CODE_AI_KEY_PROVIDERS.forEach(name => {
    const keyInput = $(`code-ai-key-${name}`);
    if (keyInput) keyInput.value = '';
  });
  if ($('intent-api-key')) $('intent-api-key').value = '';
  const statusEl = $('code-ide-ai-status');
  if (statusEl) statusEl.textContent = 'AI keys cleared. Add a key when you need cloud AI.';
  try {
    await fetch('/api/intent/api_key', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: '', provider }),
    });
  } catch(e) {}
  syncCodeAIKeyInputs();
  toast('AI keys cleared', 'yellow');
};

function bindCodeAIKeyControls() {
  CODE_AI_KEY_PROVIDERS.forEach(provider => {
    const input = $(`code-ai-key-${provider}`);
    if (input && input.dataset.bound !== '1') {
      input.dataset.bound = '1';
      input.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
          e.preventDefault();
          window.saveCodeAIKey();
        }
      });
    }
  });
  const modal = $('code-ai-key-modal');
  if (modal && modal.dataset.bound !== '1') {
    modal.dataset.bound = '1';
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && !modal.hidden) window.closeCodeAIKeyPopup();
    });
  }
  syncCodeAIKeyInputs();
}

window.refineGeneratedProject = async function() {
  const promptEl = $('code-ide-ai-prompt');
  const statusEl = $('code-ide-ai-status');
  const btn = $('code-ide-ai-apply');
  const prompt = (promptEl?.value || '').trim();
  if (!prompt) {
    toast('Describe the code change you want.', 'yellow');
    return;
  }
  ensureBlankCodeWorkspace();
  syncActiveCodeFromEditor();
  appendCodeChatMessage('user', prompt);
  if (codePromptWantsTraining(prompt)) {
    if (btn) btn.disabled = true;
    try {
      await beginCodeGuidedTraining(prompt);
      if (promptEl) promptEl.value = '';
    } catch(e) {
      if (statusEl) statusEl.textContent = e.message || 'Could not start guided labeling.';
      appendCodeChatMessage('assistant', e.message || 'Could not start guided labeling.');
      toast('Could not start guided labeling: ' + (e.message || e), 'red');
    } finally {
      if (btn) btn.disabled = false;
    }
    return;
  }
  if (!_codeProject || !Array.isArray(_codeProject.files) || !_codeProject.files.length) {
    toast('Create or open a file first, then ask for changes.', 'yellow');
    return;
  }
  if (btn) btn.disabled = true;
  if (statusEl) statusEl.textContent = 'Applying requested changes...';
  appendCodeChatMessage('assistant', 'Working on the generated files...');
  try {
    const apiKey = getCodeAIKey();
    const provider = getCodeAIProvider();
    const baseUrl = getCodeAIBaseUrl();
    const model = getCodeAIModel();
    const resp = await fetch('/api/codegen/refine', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, project: _codeProject, api_key: apiKey, provider, base_url: baseUrl, model })
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || data.ok === false || !data.project) {
      throw new Error(data.detail || data.error || `Code change failed (${resp.status})`);
    }
    renderPendingCodeReview(data.project);
    if (promptEl) promptEl.value = '';
    if (statusEl) statusEl.textContent = data.project.summary || 'Changes ready for review.';
    appendCodeChatMessage('assistant', data.project.summary || 'I prepared file changes. Review and apply them when ready.');
    toast('Generated project changes ready', 'green');
  } catch(e) {
    if (statusEl) statusEl.textContent = e.message || 'Could not apply changes.';
    appendCodeChatMessage('assistant', e.message || 'Could not apply changes.');
    toast('Could not apply code changes: ' + (e.message || e), 'red');
  } finally {
    if (btn) btn.disabled = false;
  }
};

function bindCodeChatComposer() {
  const promptEl = $('code-ide-ai-prompt');
  if (!promptEl || promptEl.dataset.bound === '1') return;
  promptEl.dataset.bound = '1';
  promptEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      window.refineGeneratedProject();
    }
  });
}

function codePromptWantsTraining(prompt) {
  const text = String(prompt || '').toLowerCase();
  return /\b(train|training|label|labels|labeling|collect|record|calibrate|classifier|classification|detect|recognize|gesture|blink|attention|focus|fatigue|model)\b/.test(text)
    && !/\b(just|only)\s+(edit|change|refactor|style|css|rename|format)\b/.test(text);
}

async function waitForCodeStream(maxWaitMs = 8000) {
  const start = Date.now();
  while (Date.now() - start < maxWaitMs) {
    if (S.streaming) return true;
    await new Promise(resolve => setTimeout(resolve, 350));
  }
  return !!S.streaming;
}

async function beginCodeGuidedTraining(prompt) {
  appendCodeChatMessage('assistant', 'I will turn that into a guided labeling run. I will ask you to perform each task, highlight accepted signal windows, and save them for training.');
  const statusEl = $('code-ide-ai-status');
  if (statusEl) statusEl.textContent = 'Planning guided labels...';
  setCodeAgentOutput('Planning labels from prompt...');
  const apiKey = getCodeAIKey();
  const provider = getCodeAIProvider();
  const baseUrl = getCodeAIBaseUrl();
  const model = getCodeAIModel();
  const resp = await fetch('/api/intent', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, api_key: apiKey, provider, base_url: baseUrl, model }),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || data.ok === false || !data.plan) {
    throw new Error(data.detail || data.error || `Intent planning failed (${resp.status})`);
  }
  const plan = data.plan || {};
  const labels = Array.isArray(plan.suggested_gestures) ? plan.suggested_gestures : [];
  S.guidedLabeling = {
    active: true,
    prompt,
    labels,
    regions: [],
    activeRegionKey: '',
    activeRegionStartSample: 0,
  };
  S.review.showOverlays = true;
  appendCodeChatMessage('assistant', `Label plan: ${labels.length ? labels.join(', ') : 'default profile labels'}.`);
  setCodeAgentOutput([
    `Prompt: ${prompt}`,
    `Task type: ${plan.task_type || 'guided labeling'}`,
    `Labels: ${labels.length ? labels.join(', ') : 'default profile labels'}`,
    'Starting stream and guided recording...',
  ]);
  if (!S.streaming) {
    if (statusEl) statusEl.textContent = 'Starting synthetic stream for guided labeling...';
    await fetch('/api/stream/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: 'synthetic', synthetic_scenario: 'clean' }),
    });
    await waitForCodeStream(7000);
  }
  if (!S.streaming) throw new Error('Stream did not start. Start hardware or synthetic stream, then retry.');
  const protocol = Array.isArray(plan.recording_protocol) && plan.recording_protocol.length ? plan.recording_protocol : null;
  const targetWindows = Number(plan.dataset_targets?.recommended_windows || 200);
  const guidedResp = await fetch('/api/recording/guided/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ protocol, target_windows: targetWindows }),
  });
  const guidedData = await guidedResp.json().catch(() => ({}));
  if (!guidedResp.ok || guidedData.ok === false) {
    throw new Error(guidedData.detail || guidedData.error || `Guided recording failed (${guidedResp.status})`);
  }
  if (guidedData.session) onGuidedStep(guidedData.session);
  startGuidedStatusPolling();
  if (statusEl) statusEl.textContent = 'Guided labeling active. Follow the popup tasks.';
  appendCodeChatMessage('assistant', 'Guided labeling is active. Follow the popup; I will keep accepted windows for training.');
  toast('Guided labeling started', 'green');
}

// Handle codegen_complete from WebSocket
function onCodegenComplete(d) {
  if (d && d.files) {
    _loadProjectIntoIDE(d);
    toast(`Code ready: ${d.files.length} files`, 'green');
  }
}

// =============================================================================
// GO
// =============================================================================
bindCodeIdeAIResize();
bindCodeIdeAIResizeLegacy();
bindCodeAIKeyControls();
bindCodeChatComposer();
boot();
