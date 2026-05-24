# KYMA Product Strategy 2026

## Position

KYMA should become the research-grade biosignal operating room: a beautiful live signal workspace where EEG, EMG, ECG, PPG, EDA, EOG, IMU, events, models, and exports all stay synchronized.

The strongest niche is not another generic OpenBCI-style viewer and not a clinical diagnostic product. The best wedge is:

> A live multimodal biosignal ML workbench for researchers, neurotech teams, and wearable AI companies who need to see, label, trust, and export clean signal windows quickly.

## Market Signals

- Research tools win when they interoperate. MNE, LSL, XDF, BIDS, MATLAB, Python notebooks, and hardware-neutral streams matter more than one proprietary device path.
- OpenBCI-style apps prove that people want real-time visualization, filters, recording, and routing to UDP, OSC, LSL, Serial, and MATLAB-like tools.
- MNE-LSL and MNE-Python show the serious research center of gravity: real-time LSL streams, preprocessing, visualization, time-frequency analysis, source estimation, connectivity, ML, and statistics.
- NeuroKit2 proves there is demand for simple physiological processing across ECG, RSP, EDA, EEG, EMG, and PPG.
- Braindecode, EEG-FM-Bench, PaPaGei, PhysioLite, BioTrain, and wearable edge-AI papers point toward foundation encoders, benchmarkable model packs, on-device adaptation, and small low-latency models.
- VC/neurotech funding is clustering around brain-health analytics, wearable EEG, EMG intent interfaces, closed-loop sleep/audio systems, edge AI, and longitudinal biomarkers.

## Product Pillars

### 1. Signal Trust Layer

- Per-channel quality score with explanations: contact, flatline, saturation, hum, drift, motion, electrode pop, spike, low signal.
- Latency strip: device latency, server ingest, filter time, model inference, websocket age, UI frame age.
- Stream audit: sample rate stability, dropped windows, chunk size, jitter, clock offset, LSL drift.
- Provenance ledger: device, firmware, profile, filters, model version, export hash, subject/session metadata.

### 2. AI Signal Lens

- Model-selected signal spans, persistent per channel.
- Hover cards with event type, confidence, model, evidence, timing, and recommended action.
- Foundation embedding timeline: norm, drift, cluster, novelty, repeated pattern detection.
- AI-guided cleanup: "try notch", "tighten bandpass", "mark artifact", "rerun baseline".
- Opt-in remote AI waveform mode: send tiny compressed waveform summaries, never raw by default.

### 3. Research Recording

- Protocol builder with randomized blocks, rest/task timing, repetition counts, condition labels, and marker schema.
- Marker tracks: manual, LSL, hardware trigger, AI-suggested, accepted/rejected.
- Trial quality gate before saving.
- Session replay with synchronized signals, model outputs, markers, and notes.

### 4. Model Lab

- Local model registry with TorchScript, ONNX, foundation encoders, tiny edge models, and metadata.
- Benchmark view: subject-independent split, leave-one-session-out, calibration curve, confusion matrix, drift over time.
- Live model compare: classic features + LDA/SVM/RF vs CNN/Transformer/foundation encoder.
- Export model card: training data, preprocessing, channels, window length, metrics, failure modes.

### 5. Visual System

- Keep the main waveform calm and premium, not game-like.
- Let each channel keep its identity color, but use AI overlays sparingly and persistently.
- Add dense but readable research panels: quality, markers, model, latency, export.
- Use replay and hover interactions instead of explanatory text.
- Make the first screen answer: Is signal live? Is it clean? What did the model see? What should I do next?

## High-Value Feature Backlog

### Immediate

- Live ML latency and embedding metadata in the AI Lens.
- Server-side `/api/debug/live_signal` expanded with ML runtime, websocket queue depth, and last inference.
- Browser frame/WS age indicator near FPS.
- Save AI-selected windows as markers with one click.
- Foundation embedding history for last 2-5 minutes.

### Near-Term

- MNE-LSL client mode for research-grade EEG streams.
- XDF round-trip: record and replay synchronized signal + markers.
- BIDS export wizard with channel types/units/events.
- NeuroKit2-derived ECG/PPG/EDA feature panels.
- Braindecode/ONNX adapter for model packs.
- Remote AI waveform summary opt-in.

### Product Differentiators

- "Signal receipt" export: a PDF/JSON record proving what was captured, cleaned, modeled, and exported.
- "Dataset readiness" score: subject/session balance, label counts, artifact burden, drift, model leakage warnings.
- "Foundation signal map": project live windows into embedding space and show novelty/repeated states.
- "Research mode vs prototype mode": strict provenance for labs, low-friction mode for demos.
- "Edge deployment preview": estimate whether a model fits MCU/NPU latency and memory constraints.

## Niche To Own

Do not lead with "BCI control." That market is noisy and overpromised.

Lead with:

> Beautiful live biosignal data infrastructure for teams building wearable AI.

Researchers care about trust, reproducibility, export, and annotations. Big tech cares about clean data pipelines, low latency, edge models, and faster prototyping. KYMA can sit exactly between those needs.

