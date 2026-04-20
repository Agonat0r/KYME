# KYMA Research Stack

KYMA already runs as a browser-based biosignal control platform. This document
defines the next layer above that runtime: the research tooling needed to make
it credible as a reusable biosignal and BCI development environment instead of
only an EMG arm demo.

## Install Tiers

Use the dependency files by role instead of forcing every machine to carry the
entire stack.

- `requirements.txt`: core runtime for the current dashboard, server, streaming,
  and arm-control demo.
- `requirements-research.txt`: optional research and ML tooling for analysis,
  multimodal acquisition, benchmarking, notebooks, and experiment design.
- `requirements-dev.txt`: test and lint tooling for day-to-day development.

Recommended setup:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-research.txt
python -m pip install -r requirements-dev.txt
```

## Tooling Layers

### 1. Runtime platform

This is the shipping app layer and should stay lean.

- FastAPI + WebSocket server
- Browser dashboard
- BrainFlow hardware streaming
- Synthetic data and playback support
- Biosignal profiles and live analyzers
- Arduino or simulated actuation backends

### 2. Research acquisition and data standards

These tools turn KYMA into a real lab workflow instead of a standalone demo.

- `pylsl`: low-level LSL compatibility for the current runtime and quick stream
  publishing
- `mne-lsl`: higher-level LSL stream discovery, inlets/outlets, replay, and
  real-time processing scaffolding
- `pyxdf`: import and replay XDF sessions
- `mne`: offline analysis, preprocessing, visualization, decoding, and review
- `mne-bids`: standardized dataset export and import
- `pybids`: query and index exported BIDS datasets after they leave KYMA;
  best kept in notebook or validation workflows rather than the live runtime
- `pyEDFlib`: EDF/BDF interchange for clinical and legacy acquisition workflows
- `h5py`: structured binary storage for intermediate artifacts and large runs

### 3. Biosignal analytics

These cover signal cleaning and modality-aware feature work beyond EMG.

- `neurokit2`: ECG, PPG, EDA, respiration, EOG, and general physiology helpers
- `mne-features`: reusable handcrafted feature extraction for M/EEG-style
  multichannel windows
- `mne-icalabel`: automatic ICA component labeling for EEG cleanup and artifact
  review
- `scipy`: filters, spectral analysis, statistics, and signal transforms
- `pandas`: annotation tables, event logs, and experiment metadata
- `matplotlib`, `seaborn`, `plotly`: publication plots, diagnostics, and
  interactive review notebooks

### 4. Classical BCI and machine learning

This is the baseline research stack for cross-session and cross-subject work.

- `scikit-learn`: baseline models, metrics, CV, and preprocessing pipelines
- `pyriemann`: covariance and Riemannian methods for BCI pipelines
- `moabb`: benchmark harness for standard BCI evaluation
- `optuna`: hyperparameter search
- `hydra-core`: reproducible experiment configuration
- `wandb`: run tracking, artifacts, and sweep management

### 5. Deep learning

Use the existing PyTorch path, but keep it export-friendly.

- `torch`: core training and inference runtime
- `lightning`: distributed and mixed-precision training without hiding the loop
- `braindecode`: deep EEG/ECoG/MEG decoding workflows
- `torcheeg`: dataset wrappers, transforms, and EEG model zoo

Recommended model policy:

- Keep every decoder behind a profile-aware adapter.
- Support both feature-based and raw-window inputs.
- Prefer CPU-safe inference for live demos.
- Design models so they can be exported later without rewriting the stack.

### 6. Experiments and human studies

- `psychopy`: stimulus delivery, tasks, timing, and subject-facing paradigms
- `jupyterlab`: quick exploration and study iteration
- `ipykernel`: clean notebook kernels per environment

### 7. Developer tooling

- `pytest`, `pytest-asyncio`: backend and API regression tests
- `httpx`: API test client and endpoint checks
- `ruff`: linting and formatting
- `pre-commit`: consistent local hooks

## External Tools

These are not part of the Python requirements, but they matter for the full
platform story.

- **LabRecorder**: record synchronized LSL streams to XDF
- **BIDS Validator**: validate exported BIDS datasets before sharing or running
  downstream pipelines; keep it as a separate Node-based validation tool
- **MNE-BIDS-Pipeline**: optional EEG/MEG batch-processing layer once KYMA
  exports stable BIDS datasets
- **OpenBCI GUI**: sanity-check hardware, channels, and electrode contact
- **Blender**: create digital twins, anatomy overlays, electrode placement
  guides, and rigged arm/hand assets

## Added Upstream Packages

These are the most useful additions for the current roadmap because they close
real gaps in the repo instead of just expanding the package list.

| Package / Tool | Focus | Why it belongs in KYMA |
|---|---|---|
| `mne-lsl` | LSL input, discovery, replay | Better foundation for subscribing to external LSL sources, not just publishing them |
| `pyxdf` + **LabRecorder** | XDF ingest / record | Standard path for multimodal biosignal recording and replay |
| `mne-bids` + `pybids` + **BIDS Validator** | BIDS export / query / validation | Turns session archives into reusable datasets instead of project-local files |
| `pyEDFlib` | EDF/BDF interoperability | Adds a practical interchange path for clinical or legacy electrophysiology tools |
| `mne-features` | Feature engineering | Speeds up non-deep-learning biosignal benchmarks and profile-specific features |
| `mne-icalabel` | EEG artifact handling | Useful once EEG moves beyond simple live band-state decoding |
| **MNE-BIDS-Pipeline** | EEG/MEG batch analysis | Optional downstream pipeline after KYMA exports validated BIDS data |

## Blender and 3D Asset Direction

The 3D layer should support both presentation value and engineering use.

- Export browser-ready `.glb` assets, not editor-only scenes
- Keep a named joint hierarchy that matches KYMA control labels
- Maintain one low-poly live dashboard asset and one high-detail presentation
  asset for the same mechanism
- Build a reusable forearm anatomy model with electrode placement overlays
- Keep end-effector and gripper variants as separate attachable assets

Suggested model set:

- rigged robotic arm twin
- rigged hand/prosthetic hand
- forearm anatomy sleeve with electrode map
- calibration scene with reference poses
- hardware rack or benchtop layout for system demos

## Platform Roadmap

### Phase 1. Foundation

Goal: make every session reproducible and every signal profile comparable.

- Add LSL stream in/out support
- Add LSL stream discovery and external inlet support
- Export sessions with modality metadata and event markers
- Add XDF import plus BIDS validation utilities
- Add profile-aware signal quality metrics
- Add playback sessions as a first-class stream source

### Phase 2. Multimodal decoding

Goal: move from EMG-first to true biosignal decoding.

- Keep EMG as the actuator demo path
- Add profile-specific feature extractors for EEG, ECG, EOG, EDA, PPG, and
  respiration
- Add benchmark scripts for within-session, cross-session, and cross-subject
  evaluation
- Standardize output as `label`, `metrics`, and `action`, not only `gesture`
- Add experiment configs for training, validation, and replay

### Phase 3. Deployment and digital twin

Goal: make KYMA usable for demos, edge deployment, and reproducible studies.

- Add exportable inference paths for trained models
- Keep hardware and simulation outputs behind the same translation layer
- Build production-quality Blender assets for the dashboard and presentations
- Add richer replay and annotation tooling for recorded sessions
- Add device backends beyond the robotic arm

## Immediate Priorities

If the goal is to get the most value with the least disruption, do these next:

1. Add LSL inlet discovery and subscription with `mne-lsl`.
2. Add XDF import/export workflow around `pyxdf` and LabRecorder.
3. Harden BIDS export with `mne-bids`, `pybids`, and BIDS validation.
4. Add one non-EMG training path, preferably EOG or ECG.
5. Add benchmark scripts, notebook templates, and one polished Blender `.glb`
   asset tied to KYMA joint names.
