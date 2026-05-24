# KYMA AI Workflow Roadmap

## Goal

Make KYMA an AI-accelerated biosignal workflow platform without turning v1 into a regulated diagnostic product.

The AI should help users:

- collect cleaner sessions
- understand what happened in a session
- detect bad data faster
- label and review data with less expertise
- move from signal to deployment faster

## Product Principle

Use AI for workflow acceleration first, not diagnosis first.

Good v1:

- signal quality copilot
- artifact detection
- protocol guidance
- session summarization
- marker suggestion
- export and firmware assistance

Avoid in v1:

- diagnosis
- treatment recommendation
- clinical triage
- patient self-testing claims
- claims that a model identifies disease or impairment

## V1 Now

### 1. Session Copilot

Use an LLM to turn current KYMA state into structured workflow output.

Inputs:

- active signal profile
- diagnostics snapshot
- filter chain
- review selection stats
- workshop summary
- protocol state
- channel labels

Outputs:

- session summary
- suggested next actions
- suggested marker labels
- flagged channels
- export recommendations

Why this matters:

- reduces operator skill burden
- gives cleaner handoff between collection and analysis
- keeps AI in a workflow-support role

Implementation:

- backend route that packages session state
- OpenAI Responses API
- Structured Outputs with a strict JSON schema
- frontend copilot card that renders typed fields instead of free-form text

### 2. Voice Operator Assistant

Use realtime voice to guide users during setup and collection.

Examples:

- "Channel 2 looks disconnected."
- "You have high hum on the current setup."
- "Start the next contraction now."
- "This protocol block is complete."
- "The current chunk looks good for training."

Why this matters:

- makes the system usable by non-experts
- helps hands-busy hardware workflows
- fits rehab, robotics, and lab setup use cases

Implementation:

- OpenAI Realtime API
- push KYMA state events into a narrow operator-agent prompt
- allow only workflow-safe tool calls

### 3. Artifact / Anomaly Assistant

Add a model layer that detects windows that differ from known-good signal behavior.

Use cases:

- loose electrodes
- flatline / disconnect
- clipping
- motion bursts
- unusual contractions
- out-of-distribution gestures

Why this matters:

- improves data quality before training
- reduces wasted collection sessions
- makes review faster

Implementation options:

- start with deterministic heuristics already in KYMA
- add anomaly scoring on top
- expose confidence and reason tags

### 4. Marker Suggestion Engine

Have AI suggest useful marker names and notes from waveform context plus session state.

Examples:

- `rest_start`
- `contraction_onset`
- `artifact_motion`
- `bad_contact`
- `usable_training_chunk`

Why this matters:

- makes later analysis much cleaner
- helps novice users label sessions consistently

Implementation:

- combine review stats + artifact flags + protocol phase
- return 3 to 5 suggested markers with short rationales

### 5. Session QA Score

Produce a simple session-quality score for a collected run.

Score dimensions:

- channel stability
- noise / hum
- clipping
- artifact burden
- protocol completeness
- label balance
- usable windows for training

Why this matters:

- gives teams a decision point before they waste time training
- useful for labs, robotics teams, and enterprise workflows

## V2 Later

### 6. Cross-Session Comparison Assistant

Compare two sessions and answer:

- what changed
- which channels degraded
- whether baseline shifted
- whether the second session is train-compatible

### 7. Training Readiness Assistant

Recommend:

- whether to train now
- which labels are weak
- which chunks to exclude
- whether more balanced data is needed

### 8. Firmware Copilot

Turn control logic and thresholds into firmware-ready outputs.

Examples:

- generate Arduino sketch skeletons
- generate threshold constants
- generate comments and operator notes
- explain what a generated sketch does

This should remain an engineering assistant, not an autonomous clinical controller.

### 9. Embedded TinyML Path

Once datasets and control tasks are stable, support export to small embedded models.

Targets:

- TensorFlow Lite for Microcontrollers
- simple anomaly models
- tiny gesture classifiers

Use this only after the workflow and QA layer is reliable.

### 10. Forecasting / Drift Modeling

Use time-series forecasting for long-session trends, not for core diagnosis.

Possible uses:

- expected baseline drift
- fatigue trend estimation
- signal degradation forecasting
- maintenance alerts for electrodes and setups

## Recommended AI Stack

### OpenAI

Use for:

- structured session summaries
- marker suggestions
- workflow copilot
- voice operator assistant
- firmware explanation and code-generation assistance

Best fit:

- high-value orchestration
- natural-language reasoning
- typed JSON outputs
- interactive operator UX

### Anomaly Detection Layer

Use for:

- bad-window scoring
- out-of-distribution detection
- unknown gesture flagging
- unusual signal behavior

Best fit:

- signal QA
- guardrails before classification

### Tiny Embedded Inference

Use for:

- low-latency on-device gesture recognition
- low-cost anomaly detection on microcontrollers
- deployment beyond KYMA-hosted inference

Best fit:

- later-stage deployment
- hardware productization

### Time-Series Foundation Models

Use sparingly for:

- long-horizon trend analysis
- drift forecasting
- experiment-level comparisons

Do not make this the main decoder for v1.

## Safe Claims

Safer early claims:

- "helps users collect cleaner biosignal sessions"
- "flags likely artifacts and noisy channels"
- "guides protocol-based recording"
- "summarizes sessions and suggests annotations"
- "accelerates review, export, and deployment workflows"
- "helps engineering and research teams build biosignal-powered interfaces"

## Risky Claims

Avoid saying:

- "diagnoses conditions"
- "detects disease"
- "replaces clinical testing"
- "lets patients perform medical testing at home"
- "recommends treatment"
- "tells doctors what diagnosis to make"

## Best Customer Wedge

Start with:

- EMG device teams
- rehab-tech builders
- robotics / HCI teams
- OpenBCI / BrainFlow developers
- labs that need better workflow, not diagnosis

Avoid starting with:

- direct-to-consumer medical testing
- doctor-facing diagnostic decision support
- patient self-testing products

## Sequencing

### Phase 1

- session copilot
- artifact assistant
- marker suggestions
- session QA score

### Phase 2

- voice operator assistant
- cross-session comparison
- training readiness assistant

### Phase 3

- firmware copilot
- embedded TinyML export
- trend forecasting

## What Success Looks Like

The right v1 outcome is not "the AI is medically smart."

The right v1 outcome is:

- less operator confusion
- cleaner sessions
- faster review
- better annotations
- higher training success
- shorter time from signal to working control demo

## Short Positioning Statement

KYMA is an AI-accelerated workflow platform for biosignal teams. It helps users collect cleaner sessions, detect bad data faster, summarize what happened, and move from raw signals to deployable control logic with much less specialized expertise.

## Source Notes

- OpenAI Structured Outputs and Realtime API are strong fits for typed copilot output and voice-guided workflows.
- Anomaly detection is a good fit for out-of-distribution and bad-signal windows.
- TensorFlow Lite Micro is the practical embedded path for later deployment.
- Time-series foundation models are better for trend analysis than for a first-pass real-time decoder.
