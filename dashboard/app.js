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
    '--bg':'#0d1117','--surface':'#161b22','--border':'#30363d',
    '--accent':'#58a6ff','--green':'#3fb950','--yellow':'#d29922',
    '--red':'#f85149','--text':'#c9d1d9','--text-dim':'#8b949e',
    '--radius':'8px','--font':"'Segoe UI', system-ui, sans-serif",
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
  dark:      { bg:'#0d1117', grid:'rgba(255,255,255,0.06)', alt:'rgba(255,255,255,0.015)' },
  retro:     { bg:'#000000', grid:'rgba(0,255,0,0.12)',     alt:'rgba(0,255,0,0.04)' },
  mario:     { bg:'#000040', grid:'rgba(252,160,68,0.15)',  alt:'rgba(252,160,68,0.06)' },
  gameboy:   { bg:'#9bbc0f', grid:'rgba(15,56,15,0.2)',     alt:'rgba(15,56,15,0.08)' },
  cyberpunk: { bg:'#0a0010', grid:'rgba(255,45,149,0.12)',  alt:'rgba(0,255,255,0.04)' },
};

// per-channel waveform colors — different palette for each theme
const CH_COLORS = {
  dark:      ['#58a6ff','#3fb950','#ff7b72','#d29922','#bc8cff','#39d353','#ffa657','#79c0ff'],
  retro:     ['#00ff00','#00cc00','#33ff33','#00ff66','#66ff66','#00ff99','#99ff99','#00ffcc'],
  mario:     ['#e40712','#00a800','#fca044','#049cd8','#ff6b6b','#43b047','#fcb514','#ffffff'],
  gameboy:   ['#0f380f','#306230','#0f380f','#306230','#0f380f','#306230','#0f380f','#306230'],
  cyberpunk: ['#ff2d95','#00ffff','#ff2d95','#00ffff','#ff2d95','#00ffff','#ff2d95','#00ffff'],
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

const API = `http://${location.hostname}:8000`;
const WS_URL = `ws://${location.hostname}:8000/ws`;
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

  // proportional control — maps EMG channels directly to joints (no classifier needed)
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


// =============================================================================
// BOOT — kicks everything off
// =============================================================================

async function boot() {
  // theme first so the page doesn't flash white
  applyTheme(currentTheme);
  $('theme-select').onchange = e => applyTheme(e.target.value);

  // grab server state before building the UI
  await loadStatus();
  await detectMode();
  buildLegend();
  buildRmsBars();
  buildServos();
  buildQualityGrid();
  initProportionalUI();
  bindButtons();
  connectWS();
  resizeCanvas();
  window.onresize = resizeCanvas;
  renderLoop();
  loadSessions();

  // fire up the 3d arm once three.js is loaded
  init3DArm();

  // model selector
  const modelSel = $('model-select');
  if (modelSel) {
    modelSel.onchange = () => {
      if (window._arm3d) window._arm3d.setModel(modelSel.value);
    };
  }

  // update performance stats once per second
  setInterval(updatePerformance, 1000);
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

async function detectMode() {
  try {
    const cfg = await get('/api/config');
    if (cfg.mock) {
      $('stream-mode').value = 'mock';
      $('port-settings').style.display = 'none';
    } else {
      $('stream-mode').value = 'real';
      $('port-settings').style.display = 'block';
      await scanPorts();
    }
  } catch (_) {}
}

async function loadStatus() {
  try {
    const s = await get('/api/status');
    S.gestures   = s.gestures || [];
    S.trained    = s.model_trained;
    S.streaming  = s.stream_running;
    S.recSession = s.is_recording;
    setSysState(s.state);
    buildGestureList();
    buildQuickGestures();
    refreshTrainSummary();
  } catch { /* server might not be up yet, that's fine */ }
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
    updateQualityGrid(d.rms);
    if (S.proportional) applyProportional(d.rms);
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

/**
 * prediction comes in ~20 times/sec with the gesture name + confidence.
 * we bounce the label when the gesture changes and update the 3d arm.
 */
function onPrediction(d) {
  // skip classifier-driven arm updates when proportional control is active
  if (S.proportional) return;

  const el = $('pred-gesture');
  const g = d.gesture || '--';

  // bounce animation when the gesture actually changes
  if (g !== S.lastGesture && g !== '--') {
    el.classList.remove('bounce');
    void el.offsetWidth;  // forces reflow so the animation restarts
    el.classList.add('bounce');
    S.lastGesture = g;

    // update 3d arm to match the new gesture
    if (window._arm3d) window._arm3d.setGesture(g);
    $('arm-gesture-name').textContent = g.toUpperCase();

    // optional sound beep on gesture change
    if ($('chk-sound').checked) playBeep(g);

    // check if a block program is mapped to this gesture
    checkGestureProgramMapping(g);
  }

  el.textContent = g;
  const pct = (d.confidence * 100).toFixed(1);
  $('pred-confidence').textContent = `Confidence: ${pct}%`;
  const bar = $('conf-bar');
  bar.style.width = `${d.confidence * 100}%`;
  bar.style.background = d.confidence > 0.8 ? 'var(--green)'
    : d.confidence > 0.55 ? 'var(--yellow)' : 'var(--red)';

  // track prediction rate and add to timeline
  S.predCount++;
  const gIdx = S.gestures.indexOf(g);
  S.timeline.push({g: gIdx >= 0 ? gIdx : 0, c: d.confidence});
  if (S.timeline.length > S.timelineMax) S.timeline.shift();
  drawTimeline();
}

/** calibration steps get appended to the log box one at a time */
function onCalibration(d) {
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
    btn.textContent = 'Start Stream';
    btn.className = 'btn primary';
  }
  $('btn-calibrate').disabled = !S.streaming;
  $('btn-fit').disabled = s === 'estop';
}


// =============================================================================
// BUTTON WIRING
//
// every button gets its click handler here. i like having them all in one
// place so you don't have to hunt through the HTML for onclick handlers.
// =============================================================================

function bindButtons() {
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
      if (window._arm3d) window._arm3d.setGesture('rest');
    } catch (e) { toast(e.message, 'red'); }
  };

  // --- mode toggle: show/hide port settings ---
  $('stream-mode').onchange = () => {
    const isReal = $('stream-mode').value === 'real';
    $('port-settings').style.display = isReal ? 'block' : 'none';
    if (isReal) scanPorts();
  };

  // --- scan serial ports ---
  $('btn-refresh-ports').onclick = () => scanPorts();

  // --- start/stop the brainflow stream ---
  $('btn-stream').onclick = async () => {
    try {
      if (S.streaming) {
        await post('/api/stream/stop');
      } else {
        const mode = $('stream-mode').value;
        const body = {};
        if (mode === 'real') {
          const cPort = $('cyton-port').value;
          if (cPort) body.cyton_port = cPort;
          const aPort = $('arduino-port').value;
          if (aPort) body.arduino_port = aPort;
        }
        await post('/api/stream/start', body);
      }
    } catch (e) { toast(e.message, 'red'); }
  };

  // --- run the 3-stage calibration routine ---
  $('btn-calibrate').onclick = async () => {
    $('cal-log').innerHTML = '';
    try { await post('/api/calibrate'); toast('Calibration started'); }
    catch (e) { toast(e.message, 'red'); }
  };

  // --- fit the selected classifier on all recorded data ---
  // LDA takes ~1 second, TCN takes 30-60 seconds, Mamba takes 60-120 seconds
  $('btn-fit').onclick = async () => {
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
    toast('Stream cleared');
  };

  // --- toggle session recording (saves raw EMG to disk) ---
  $('btn-record-session').onclick = async () => {
    try {
      if (S.recSession) {
        const r = await post('/api/session/stop');
        S.recSession = false;
        $('btn-record-session').textContent = 'Rec Session';
        $('btn-record-session').className = 'btn';
        toast(`Saved: ${r.saved_to}`);
        loadSessions();
      } else {
        const label = prompt('Session label (optional):') || '';
        const r = await post('/api/session/start', {label});
        S.recSession = true;
        $('btn-record-session').textContent = 'Stop Rec';
        $('btn-record-session').className = 'btn danger';
        toast(`Recording: ${r.session_id}`);
      }
    } catch (e) { toast(e.message, 'red'); }
  };

  $('btn-sessions-refresh').onclick = loadSessions;
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

  // built-in gestures
  S.gestures.forEach(g => {
    const b = document.createElement('button');
    b.className = 'btn'; b.textContent = g;
    b.style.width = 'auto'; b.style.flex = '1';
    b.onclick = async () => {
      try {
        await post(`/api/gesture/${g}`);
        toast(`Gesture: ${g}`);
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
    c.innerHTML += `<div class="rms-row">
      <span class="rms-label">CH${i+1}</span>
      <div class="rms-bar-bg"><div class="rms-bar" id="rb-${i}" style="background:${cols[i]}"></div></div>
      <span class="rms-val" id="rv-${i}">0.000</span>
    </div>`;
  }
}

function updateRmsBars(rms) {
  rms.forEach((v,i) => {
    const bar = $(`rb-${i}`);
    const val = $(`rv-${i}`);
    if (bar) bar.style.width = `${Math.min(v / 0.2 * 100, 100)}%`;
    if (val) val.textContent = v.toFixed(4);
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
    c.innerHTML += `<span class="ch-label">
      <span class="ch-dot" style="background:${cols[i]}"></span>CH${i+1}</span>`;
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

  // clear + fill background
  ctx.fillStyle = t.bg;
  ctx.fillRect(0, 0, W, H);

  const rowH = H / N_CH;
  const head = S.emgHead;

  // How many samples have actually been written to the buffer?
  // Don't draw the zero-filled portion — that causes square-wave artifacts.
  const filled = Math.min(S.emgTotal, DISPLAY_SAMPLES);

  // ── Adaptive mute: compare each channel's RMS to the BEST channel ──
  // Disconnected channels on Cyton still pick up crosstalk so a fixed
  // threshold doesn't work.  Instead, find the strongest channel and mute
  // anything that's less than 20 % of it.
  const MUTE_ABS_FLOOR = 0.5;   // µV — absolute minimum to even consider
  let maxRms = 0;
  for (let ch = 0; ch < N_CH; ch++) {
    const raw = S.rms ? S.rms[ch] : 0;
    S.rmsSmooth[ch] = S.rmsSmooth[ch] * 0.90 + raw * 0.10;
    if (S.rmsSmooth[ch] > maxRms) maxRms = S.rmsSmooth[ch];
  }
  const muteThresh = Math.max(maxRms * 0.20, MUTE_ABS_FLOOR);

  for (let ch = 0; ch < N_CH; ch++) {
    const buf = S.emg[ch];
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
    const muted = S.chMuted[ch];

    // ── Fixed scale — NO auto-scale for noise ──
    // Use a fixed sensitivity that shows real EMG well.
    // Real EMG from Cyton is typically 50-500+ µV.  Noise is 1-20 µV.
    // A fixed scale of (rowH*0.4 / 200µV) means 200 µV fills 80 % of the row.
    // Strong contractions (>200 µV) get clamped at the row edge — that's fine.
    const FULL_SCALE_UV = 200.0;
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
    const drawStart = filled < DISPLAY_SAMPLES
        ? Math.floor(W * (1 - filled / DISPLAY_SAMPLES))   // start partway across
        : 0;

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
    ctx.font = currentTheme === 'dark' ? '10px monospace' : '7px "Press Start 2P", monospace';
    const label = muted ? `CH${ch+1} (off)` : `CH${ch+1}`;
    ctx.fillText(label, 4, ch * rowH + 12);
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
      const bgs = {dark:0x0d1117, retro:0x000a00, mario:0x000040, gameboy:0x0f380f, cyberpunk:0x0a0010};
      const c = new THREE.Color(bgs[name] || bgs.dark);
      scene.background = c; scene.fog.color = c;
    },
  };
  window._arm3d.setTheme(currentTheme);
}


// =============================================================================
// SESSIONS (past recordings from disk)
// =============================================================================

async function loadSessions() {
  try {
    const list = await get('/api/sessions');
    const el = $('session-list');
    el.innerHTML = '';
    if (!list.length) {
      el.innerHTML = '<div style="font-size:11px;color:var(--text-dim)">No sessions</div>';
      return;
    }
    list.slice(0,10).forEach(s => {
      const d = document.createElement('div');
      d.className = 'session-item';
      d.innerHTML = `<div class="sid">${s.session_id||'?'}</div>
        <div>${s.duration_s?s.duration_s.toFixed(1)+'s':''} | ${s.n_events??''} events</div>`;
      el.appendChild(d);
    });
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

  tx.clearRect(0, 0, W, H);
  const tlBgs = {dark:'#0d1117', retro:'#000', mario:'#000040', gameboy:'#9bbc0f', cyberpunk:'#0a0010'};
  tx.fillStyle = tlBgs[currentTheme] || '#0d1117';
  tx.fillRect(0, 0, W, H);

  const n = S.timeline.length;
  if (!n) return;

  const bw = W / S.timelineMax;  // block width
  for (let i = 0; i < n; i++) {
    const t = S.timeline[i];
    const x = i * bw;
    const h = t.c * H;  // height = confidence
    tx.fillStyle = cols[t.g % cols.length];
    tx.fillRect(x, H - h, bw - 1, h);
  }
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
    c.innerHTML += `<div class="q-block dead" id="qb-${i}">CH${i+1}<br>--</div>`;
  }
}

function updateQualityGrid(rms) {
  for (let i = 0; i < N_CH; i++) {
    const el = $(`qb-${i}`);
    if (!el) continue;
    const v = rms[i];
    let cls, label;
    if (v < 0.0005) {
      cls = 'dead'; label = 'NO SIG';      // electrode probably fell off
    } else if (v > 0.5) {
      cls = 'bad';  label = 'NOISY';        // saturated or bad contact
    } else if (v < 0.005) {
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

  // training window count
  let total = 0;
  Object.values(S.trainCounts).forEach(n => total += n);
  $('win-count').textContent = total;
}


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
    const freq = freqs[gesture] || 440;

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
  gesture:       { color:'#f85149', label:'Run Gesture',    category:'preset', defaults:{ gesture:'open' },
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
  if_rms:        { color:'#79c0ff', label:'If RMS >',       category:'control', defaults:{ channel:0, threshold:0.05 },
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'true_out',kind:'flow',dir:'out',label:'true'},{id:'false_out',kind:'flow',dir:'out',label:'false'},{id:'flow_out',kind:'flow',dir:'out',label:'done'}] },
  if_gesture:    { color:'#50e3c2', label:'If Gesture',     category:'control', defaults:{ gesture:'open' },
    ports:[{id:'flow_in',kind:'flow',dir:'in',label:''},{id:'true_out',kind:'flow',dir:'out',label:'yes'},{id:'false_out',kind:'flow',dir:'out',label:'no'},{id:'flow_out',kind:'flow',dir:'out',label:'done'}] },
  wait_gesture:  { color:'#50fa7b', label:'Wait Gesture',   category:'control', defaults:{ gesture:'close', timeout_s:10 },
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
      const gestures = ['rest','open','close','pinch','point'];
      let html = '<label>Gesture <select data-key="gesture">';
      gestures.forEach(g => html += `<option value="${g}" ${p.gesture===g?'selected':''}>${g}</option>`);
      container.innerHTML = html + '</select></label>';
      break;
    }
    case 'wait_gesture': {
      const gestures = ['rest','open','close','pinch','point'];
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
    input.addEventListener('change', () => {
      const val = input.type === 'number' ? parseFloat(input.value) : input.value;
      p[key] = val;
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

      case 'gesture':
        execLog(`Gesture: ${p.gesture}`);
        await post('/api/gesture', { gesture: p.gesture });
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
        const rmsVal = S.rms?.[p.channel] || 0;
        if (rmsVal > (p.threshold || 0)) {
          execLog(`RMS ch${p.channel} (${rmsVal.toFixed(3)}) > ${p.threshold} → true`);
          await followFlow(prog, node.id, 'true_out');
        } else {
          execLog(`RMS ch${p.channel} (${rmsVal.toFixed(3)}) ≤ ${p.threshold} → false`);
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
        loopLines.push(`${pad}// ${BLOCK_TYPES[node.type]?.label || node.type} (EMG server only)`);
        walkFlow(node.id, 'flow_out', indent);
        break;
      case 'if_gesture':
        loopLines.push(`${pad}// If Gesture "${p.gesture}" (requires EMG classifier)`);
        loopLines.push(`${pad}// true branch:`);
        walkFlow(node.id, 'true_out', indent);
        loopLines.push(`${pad}// false branch:`);
        walkFlow(node.id, 'false_out', indent);
        walkFlow(node.id, 'flow_out', indent);
        break;
      case 'wait_gesture':
        loopLines.push(`${pad}// Wait for gesture "${p.gesture}" (requires EMG classifier)`);
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
  const prog = getActiveProgram();
  const code = $('code-output')?.value || '';
  const name = (prog?.name || 'program').replace(/[^a-zA-Z0-9_]/g, '_');
  const blob = new Blob([code], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name + '.ino';
  a.click();
  URL.revokeObjectURL(a.href);
  toast('Downloaded ' + name + '.ino');
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

document.querySelectorAll('.card-collapsible h3').forEach(h3 => {
  h3.addEventListener('click', () => {
    h3.parentElement.classList.toggle('collapsed');
  });
});

// ── Tab switching ────────────────────────────────────────────────────────────

window.switchTab = function(tab) {
  const main = $('main');
  const editor = $('block-editor');
  const hwdocs = $('hw-docs');
  const tabDash = $('tab-dashboard');
  const tabBlocks = $('tab-blocks');
  const tabHwdocs = $('tab-hwdocs');

  // Hide all
  main.classList.add('hidden');
  editor.classList.remove('active');
  hwdocs.classList.remove('active');
  tabDash.classList.remove('active');
  tabBlocks.classList.remove('active');
  tabHwdocs.classList.remove('active');

  if (tab === 'blocks') {
    editor.classList.add('active');
    tabBlocks.classList.add('active');
    refreshProgramSelect();
    renderCanvas();
    buildGestureMappingUI();
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
      execLog(`[EMG] Gesture "${gestureName}" -> "${prog.name}"`);
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
            newW + 'px 5px 1fr 5px ' + (rightPanel.style.width || (kind === 'main-left' ? '240px' : '260px'));
        } else if (kind === 'main-right' || kind === 'blocks-right') {
          const newW = Math.max(140, Math.min(w * 0.4, w - x));
          rightPanel.style.width = newW + 'px';
          parent.style.gridTemplateColumns =
            (leftPanel.style.width || (kind === 'main-right' ? '220px' : '220px')) + ' 5px 1fr 5px ' + newW + 'px';
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
