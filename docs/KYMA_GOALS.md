# KYMA Goals

## One-Line Vision

KYMA is a prompt-to-biosignal platform: plug in electrodes or upload a dataset, describe what you want, and KYMA builds the signal pipeline, recording protocol, model, visualization, report, and export.

## Product North Star

In five minutes, a user should be able to say:

> "Build me a biosignal pipeline for this data."

KYMA should then:

1. Identify the signal type.
2. Understand the target task.
3. Create the data protocol or import plan.
4. Clean and segment the signal.
5. Train or select a model.
6. Explain the result.
7. Export the dataset, report, model, or live software integration.

This is not only hardware control. KYMA should support software workflows, research reports, dataset cleaning, model creation, live visualization, annotations, and deployment.

## Signals To Support

- EEG: brain electrical activity.
- EMG: muscle activity.
- ECG/EKG: heart electrical activity.
- EOG: eye movement.
- ERG: retinal response.
- EDA/GSR: skin conductance.
- PPG: optical pulse/blood-volume signal.
- Respiration.
- Temperature.
- IMU/motion.
- Multimodal streams that combine any of the above.

## Data Sources

- Live hardware through BrainFlow, serial, BLE, UDP, OSC, WebSocket, and vendor SDKs.
- Lab Streaming Layer streams.
- Uploaded datasets: CSV, EDF/BDF, XDF, MAT, HDF5, Parquet, Zarr, NWB, BIDS.
- Existing research datasets.
- Synthetic signal generator for demos, stress tests, and regression tests.

## Core User Prompts

- "Build an EMG fatigue detector from this sleeve."
- "Make an EOG blink and gaze detector."
- "Clean this ECG dataset and mark unusable windows."
- "Turn this EEG recording into a BIDS-ready dataset."
- "Compare two electrode placements and tell me which gives better signal."
- "Train a model that detects rest vs task from this EDA and PPG stream."
- "Create a live dashboard for this LSL stream."
- "Find artifacts in this dataset and export only clean segments."
- "Use foundation embeddings to cluster signal states."
- "Export this model to ONNX and run it in the browser."

## Short-Term Goals

### 1. Make Live Signal Trust Obvious

- Show signal age, websocket age, server queue size, dropped packets, UI FPS, and ML inference time.
- Keep AI-highlighted regions persistent and channel-specific.
- Show hover cards with timing, channel, confidence, model, and reason.
- Add a "why flagged" panel for every artifact or model-selected region.

### 2. Make Local ML Real

- Keep raw/downsampled signal arrays local by default.
- Run local artifact QA on every live window.
- Run foundation encoders on live windows.
- Show embedding dimension, norm, drift, and novelty.
- Store a rolling embedding history.

### 3. Make Recording Useful

- One-click marker from AI-selected region.
- Prompt-generated protocols.
- Trial quality scoring.
- Re-record bad trials automatically.
- Session metadata: subject, condition, device, filters, model versions.

### 4. Make Export Research-Grade

- CSV for quick use.
- XDF for synchronized LSL workflows.
- BIDS for EEG/neuro-style datasets.
- NWB for neurophysiology archival workflows.
- MATLAB export.
- Model card and data provenance JSON.

### 5. Make The UI Feel Premium

- Signal-first workspace.
- Calm, dense, readable research UI.
- Strong color identity per channel.
- Minimal panels, no clutter.
- Visual answer to: Is the stream live? Is it clean? What did AI see? What should I do next?

## Long-Term Goals

### 1. Prompt-To-Pipeline

The user should not manually assemble filters, windowing, labeling, training, evaluation, and export. KYMA should generate and run the pipeline from a prompt, then expose each step for inspection.

### 2. Biosignal Model Operating System

KYMA should manage:

- Datasets.
- Protocols.
- Models.
- Foundation embeddings.
- Artifacts.
- Labels.
- Exports.
- Deployment targets.
- Provenance.

### 3. Foundation-Model Workbench

KYMA should support biosignal foundation models across EEG, EMG, ECG, EOG, PPG, and multimodal streams.

Core features:

- Embedding timeline.
- Signal-state clustering.
- Novelty and drift detection.
- Subject/session adaptation.
- Small task heads on top of frozen encoders.
- Benchmark reports.

### 4. Deployment Layer

KYMA should export pipelines to:

- Python package.
- ONNX model.
- TorchScript model.
- Browser/WebGPU runtime.
- Edge/MCU feasibility report.
- API endpoint.
- OSC/MIDI/Webhook/software control integration.
- Optional hardware control.

### 5. Research + Enterprise Layer

Long-term enterprise features:

- Team projects.
- Annotation review queues.
- Dataset versioning.
- Model registry.
- Access control.
- Audit logs.
- De-identification.
- Cloud sync optional, local-first by default.

## 2026 Tool Stack

### Stable Research Standards

- Lab Streaming Layer: live multimodal stream synchronization.
- MNE and MNE-LSL: EEG/MEG/neurophysiology analysis and real-time LSL workflows.
- BrainFlow: hardware access for OpenBCI and other biosignal devices.
- NeuroKit2: physiological signal processing for ECG, PPG, EDA, EMG, RSP, and related signals.
- Braindecode: deep learning for electrophysiological decoding.
- BIDS: organized neuroimaging and EEG-style dataset structure.
- NWB: neurophysiology data standard.
- XDF and pyxdf: synchronized multimodal recording and replay.
- ONNX and TorchScript: portable model deployment.

### Newer / Emerging 2026 Direction

- WebGPU inference for browser-native AI.
- WebNN for hardware-native browser inference where supported.
- ONNX Runtime Web with WebGPU/WebNN backends.
- Transformers.js and WebLLM for local browser agents.
- BioGAP-Ultra-style multimodal wearable edge AI platforms.
- BioTrain-style on-device fine-tuning for EEG/EOG/wearable drift adaptation.
- SilentWear-style EMG silent speech and textile sensing workflows.
- SignalMC-MED-style multimodal ECG/PPG foundation-model benchmarks.
- Biosignal foundation models such as EEG, EMG, ECG, PPG, and multimodal encoders.
- Edge model reports: model size, memory, latency, battery feasibility.

## Differentiation

OpenBCI shows signals.

MNE analyzes signals.

NeuroKit processes physiological signals.

Beacon and clinical platforms monetize specific regulated workflows.

KYMA should become the AI-native workspace that turns live or uploaded biosignals into clean datasets, usable models, reports, and deployable software.

## Near-Term Build Order

1. Add a "Prompt Pipeline" drawer.
2. Let the user choose live stream or upload dataset.
3. Parse prompt into a structured plan.
4. Generate protocol, labels, filters, window size, and model candidates.
5. Run signal QA and artifact detection.
6. Train baseline models.
7. Show model comparison and failure cases.
8. Export dataset/model/report.
9. Add browser-native ONNX/WebGPU inference for exported small models.
10. Add embedding timeline and dataset readiness score.

## Product Principle

Do not make the user become a DSP engineer to use biosignals.

KYMA should make the hard steps visible, editable, and reproducible, but it should do the work automatically first.

