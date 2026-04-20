<p align="center">
  <img src="https://img.shields.io/badge/KYMA-00bfff?style=for-the-badge&labelColor=0a0f1a" alt="KYMA" height="40"/>
</p>

<h1 align="center">KYMA</h1>

<p align="center">
  <em>Open-source biosignal control platform - from wave to action.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/OpenBCI-Cyton-green?style=flat-square" alt="OpenBCI"/>
  <img src="https://img.shields.io/badge/board-Arduino-00979D?style=flat-square&logo=arduino&logoColor=white" alt="Arduino"/>
  <img src="https://img.shields.io/badge/status-alpha-orange?style=flat-square" alt="Status"/>
</p>

<p align="center">
  <a href="https://buymeacoffee.com/SageFlugel"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"/></a>
  <a href="https://x.com/pericleshimself"><img src="https://img.shields.io/badge/Follow_@pericleshimself-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow on X"/></a>
</p>

---

## Overview

**KYMA** is a real-time biosignal acquisition, analysis, and control platform built around OpenBCI-class hardware and a browser-first research UI. It supports multiple biosignal profiles, live monitoring, recording, offline experiments, reusable digital filter design, and hardware/software control outputs from one dashboard.

The current stack includes:
- profile-aware biosignal modes: `EMG`, `EEG`, `ECG`, `EOG`, `PPG`, `EDA`, `Respiration`, `Temperature`
- `Live`, `Record`, `Train`, and `Control` workspaces
- `Filter Lab` for digital filter design and export
- `Signal Workshop` for selected-chunk DSP analysis
- `Blocks` for visual logic and Arduino code generation
- LSL, XDF, BIDS-style export, playback, OSC, and Arduino interop

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/architecture.svg"/>
    <img src="docs/architecture.svg" alt="KYMA Architecture" width="720"/>
  </picture>
</p>

### Key Features

| Feature | Description |
|---------|-------------|
| **Profile-aware biosignal stack** | Per-profile labels, filters, metrics, and training support for EMG, EEG, ECG, EOG, PPG, EDA, respiration, and temperature |
| **Live diagnostics** | Waveforms, channel toggles, decoded output, signal health, FFT/PSD-style diagnostics, hum metrics, latency, and watchdog state |
| **Review freeze + markers** | Freeze the visible review surfaces, drag a chunk like a lightweight scope capture, inspect local stats, and save custom markers |
| **Signal Workshop** | Analyze frozen chunks with FFT, PSD, spectrogram, autocorrelation, histogram, Hilbert envelope, correlation, and Laplace views |
| **Filter Lab** | Design, preview, save, activate, and export digital filters with Bode view, pole/zero map, and fixed-point estimates |
| **Training and experiments** | Live training plus offline datasets, temporal holdout, leave-one-session-out, and leave-one-subject-out evaluation |
| **Research interoperability** | LSL input/output, XDF import, BIDS-style export, session playback, subject registry, and EEG experiment presets |
| **Control outputs** | Arduino serial bridge, OSC output, proportional control, block programming, `.ino` export, and E-STOP safety |

---

## Architecture

```text
KYMA/
|-- server/                     # Python backend (FastAPI + WebSocket)
|   |-- main.py                 # API routes, WS server, runtime orchestration
|   |-- brainflow_stream.py     # BrainFlow/Cyton, synthetic, playback, LSL input
|   |-- biosignal_pipeline.py   # Profile-aware decoder routing
|   |-- *_pipeline.py           # EMG/EEG/ECG/EOG/PPG/EDA/resp/temp pipelines
|   |-- live_diagnostics.py     # Spectrum, timing, noise, watchdog metrics
|   |-- signal_workshop.py      # Selected-chunk DSP analysis service
|   |-- filter_lab.py           # Digital filter design, storage, export
|   |-- session_recorder.py     # Raw chunk + event persistence
|   |-- research_manager.py     # Datasets and offline experiments
|   |-- lsl_bridge.py           # LSL output
|   |-- lsl_input.py            # LSL inlet discovery and streaming
|   |-- xdf_import.py           # XDF inspect/import
|   |-- session_export.py       # BIDS-style export
|   |-- osc_bridge.py           # OSC output
|   `-- models.py               # Pydantic request/response schemas
|
|-- dashboard/                  # Browser frontend (vanilla JS, no build step)
|   |-- index.html              # App shell, layout, styles
|   `-- app.js                  # Workspaces, charts, tours, WS client, blocks
|
|-- firmware/
|   `-- arm_controller/
|       `-- arm_controller.ino  # Arduino servo driver (binary protocol)
|
|-- docs/                       # Diagrams, notes, slide assets, research stack docs
|-- requirements.txt            # Runtime dependencies
|-- requirements-research.txt   # Optional research/interoperability stack
|-- requirements-dev.txt        # Test/lint/dev tooling
`-- LICENSE
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **OpenBCI Cyton** board + USB dongle if using live hardware
- **Arduino** only if using physical servo control

### Quick Launch (Windows)

```bash
git clone https://github.com/YOUR_USERNAME/kyma.git
cd kyma
```

Then run:

```powershell
.\KYMA.bat
```

Optional custom port:

```powershell
.\KYMA.bat --browser --port 8007
```

### Manual Setup

```bash
pip install -r requirements.txt
```

### Optional Research Stack

Install this only if you want the broader BCI/research toolchain:

```bash
pip install -r requirements-research.txt
pip install -r requirements-dev.txt
```

This adds packages such as `mne-lsl`, `pyxdf`, `mne-bids`, `pybids`, `pyEDFlib`, `mne-features`, and `mne-icalabel`.

See [docs/kyma_research_stack.md](docs/kyma_research_stack.md) for the research roadmap and tooling rationale.

### Run (synthetic mode, no hardware)

```bash
cd server

# Windows PowerShell
$env:BOARD_ID="-1"; python main.py

# Linux/macOS
BOARD_ID=-1 python main.py
```

Open **http://localhost:8000**.

In the dashboard:
1. choose **Signal Type**
2. choose **Data Source**
3. pick **Synthetic**
4. click **Start Stream**

### Run (real Cyton board)

```bash
cd server

# Windows
$env:CYTON_PORT="COM8"; python main.py

# Linux
CYTON_PORT=/dev/ttyUSB0 python main.py
```

### Run (with Arduino servo arm)

```bash
cd server
$env:CYTON_PORT="COM8"; $env:ARDUINO_PORT="COM4"; python main.py
```

---

## Configuration

Most runtime defaults live in `server/config.py` and can be overridden with environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `SIGNAL_PROFILE` | `emg` | Active biosignal profile at startup |
| `BOARD_ID` | `0` | BrainFlow board ID (`0` = Cyton, `-1` = synthetic) |
| `CYTON_PORT` | `COM8` | Serial port for OpenBCI dongle |
| `ARDUINO_PORT` | `COM4` | Serial port for Arduino |
| `PORT` | `8000` | HTTP server port |

---

## Dashboard Layout

### Workspaces

- **Live**: monitoring, diagnostics, review freeze, markers, EEG/arm views
- **Record**: session metadata, protocol templates, XDF import, EEG experiment presets, marker helper
- **Train**: model fitting, datasets, offline experiments
- **Control**: LSL output, OSC output, arm/proportional control

### Top-Level Tabs

- **Dashboard**: the main workspace UI
- **Blocks**: visual logic builder and `.ino` export
- **Filter Lab**: digital filter design, activation, and export
- **Signal Workshop**: selected-chunk DSP analysis
- **Bench Report**: engineering snapshot export
- **Arduino Guide**: wiring, protocol, and firmware notes

### Built-In Tours

KYMA now ships with guided tours for:
- Quick Start
- Blocks
- Filter Lab
- Signal Workshop

Use the **Tour** button in the top bar, or the tab-specific tour buttons.

---

## Usage Guide

### Fastest Path: synthetic demo

1. Start KYMA
2. Choose a **Signal Type**
3. Choose **Synthetic**
4. Click **Start Stream**
5. Open the relevant visualization or control panels

### Review freeze and chunk analysis

1. Start a stream
2. In **Review & Markers**, click **Freeze Review**
3. Drag across the waveform to select a chunk
4. Save a marker, or click **Send To Workshop**
5. In **Signal Workshop**, choose the transform view and analyze the chunk

This works like a lightweight oscilloscope review pass on the visible buffer. It freezes the review surfaces while the stream continues underneath.

### Filter Lab

1. Open **Filter Lab**
2. Choose method, response type, order, and cutoff frequencies
3. Click **Preview**
4. Inspect:
   - Bode magnitude
   - pole/zero map
   - fixed-point estimate
5. Save and activate the filter for the current profile
6. Export to host code, C++, Arduino-targeted helpers, or engineering data files

Important: KYMA designs and applies custom filters on the host path. It does **not** program arbitrary custom filters onto the Cyton board itself.

### Blocks and code generation

1. Open **Blocks**
2. Load the example program or create your own
3. Use metric-threshold blocks for exportable Arduino logic
4. Use label-driven blocks when KYMA stays in the loop
5. Export a `.ino` sketch from the current graph

### Proportional control

Best used with EMG:

1. Start the stream
2. Open **Control**
3. Enable **Proportional Control**
4. Map channels to joints
5. Adjust gain and dead zone
6. Drive the simulated or physical arm directly from signal magnitude

### Recording and offline experiments

1. Open **Record**
2. Fill in session metadata
3. Use a protocol template if you want a structured run
4. Record sessions
5. Open **Train**
6. Create a dataset from saved sessions
7. Run offline experiments with temporal/session/subject holdout

---

## Signal Workshop

The `Signal Workshop` is the selected-chunk DSP analysis space. It currently supports:

- `Fourier / FFT`
- `Power Spectral Density`
- `Spectrogram`
- `Autocorrelation`
- `Amplitude Histogram`
- `Hilbert Envelope`
- `Cross-Channel Correlation`
- `Laplace Surface`

It also reports chunk-level summary metrics such as:
- mean
- RMS
- standard deviation
- min / max
- peak-to-peak
- derivative RMS
- zero crossings
- dominant frequency
- spectral centroid

The Laplace view is a **numeric magnitude surface** over frequency and sigma for the selected chunk. It is not transfer-function identification.

---

## Research Interoperability

KYMA now includes:

- **LSL output** for live samples and markers
- **LSL input** for subscribing to external streams
- **XDF import** into replayable KYMA sessions
- **session playback**
- **BIDS-style export**
- **subject registry**
- **EEG experiment presets** for oddball, SSVEP, and N170 recording workflows

The current research layer is strongest for acquisition, review, metadata, export, and offline experiments. It is not trying to replace dedicated offline packages like MNE or Brainstorm.

---

## Classifiers

| Backend | Input | Training Data | Speed | Best For |
|---------|-------|--------------|-------|----------|
| **LDA** | Hand-crafted features | 30+ windows/label | Fast | Quick prototyping and baseline models |
| **TCN** | Raw windows | 60+ windows/label | Medium | Higher-capacity temporal modeling |
| **Mamba** | Raw windows | 60+ windows/label | Medium/slow | State-space temporal modeling |

Not every profile uses every backend equally. EMG is still the strongest end-to-end arm-control path, while the other profiles now share the broader biosignal platform and research workflow.

---

## Arduino Firmware

The firmware in `firmware/arm_controller/` uses a compact binary protocol over serial at `115200` baud:

| Command | Bytes | Response | Description |
|---------|-------|----------|-------------|
| `MOVE` | `[0x01, joint, angle]` | `[0xAA, joint]` | Set servo angle (`0-180`) |
| `ESTOP` | `[0x02]` | `[0xAA, 0x00]` | Emergency stop all servos |
| `HOME` | `[0x03]` | `[0xAA, 0x00]` | Return to neutral |
| `PING` | `[0x04]` | `[0xBB, 0x00]` | Heartbeat check |

Full wiring, protocol notes, and `.ino` export instructions are available in the **Arduino Guide** tab inside the dashboard.

---

## API Reference

The server exposes a REST API and WebSocket interface. See `/docs` while the server is running for the live schema.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | System state, active profile, diagnostics, safety, filter lab, workshop status |
| `/api/config` | GET | Runtime config, available profiles, filters, calibration, experiments |
| `/api/stream/start` | POST | Start hardware, synthetic, playback, or LSL input streaming |
| `/api/stream/stop` | POST | Stop streaming |
| `/api/calibrate` | POST | Run calibration |
| `/api/session/start` | POST | Start a recording session |
| `/api/session/stop` | POST | Stop a recording session |
| `/api/sessions` | GET | List recorded sessions |
| `/api/review/marker` | POST | Save a custom review marker from the frozen display |
| `/api/filterlab/status` | GET | Filter Lab capabilities and saved filters |
| `/api/filterlab/design` | POST | Preview a digital filter design |
| `/api/workshop/status` | GET | Signal Workshop availability and supported views |
| `/api/workshop/analyze` | POST | Analyze a selected chunk in the Signal Workshop |
| `/api/lsl/start` | POST | Start LSL output |
| `/api/lsl/stop` | POST | Stop LSL output |
| `/api/osc/start` | POST | Start OSC output |
| `/api/osc/stop` | POST | Stop OSC output |
| `/api/move` | POST | Move one servo joint |
| `/api/gesture/*` | POST | Trigger named gesture actions |
| `/api/estop` | POST | Emergency stop |
| `/ws` | WebSocket | Live samples, predictions, diagnostics, calibration, markers, state |

---

## Contributing

Contributions are welcome under the terms of the AGPL-3.0 license.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push the branch
5. Open a pull request

---

## License

**KYMA** is licensed under the [GNU Affero General Public License v3.0](LICENSE).

This means:
- you can use, study, and share the software
- you can modify and distribute your own versions
- you must release modifications under the same license
- if you run a modified version as a network service, you must provide source code to users
- you cannot make closed-source derivatives

---

<p align="center">
  <em>KYMA - from wave to action</em>
</p>
