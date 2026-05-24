# AI Condition Model Registry

Updated: 2026-04-21

This is the short list of research-oriented condition and screening model families that actually make sense for KYMA right now.

Guardrail:
- These are research and workflow screening models.
- They are not patient-facing diagnosis.
- KYMA should surface them as `research watchlist`, `screening output`, or `offline review candidate`.

## Best Fits

### 1. SleepFM-Clinical

Use when:
- the session is overnight or PSG-like
- multiple modalities are available together
- the goal is disease-risk screening or sleep staging

Inputs:
- multimodal sleep signals such as EEG, ECG, EMG, EOG, and respiratory channels

Outputs:
- disease-risk scores
- sleep-stage outputs
- shared latent sleep embeddings

Why it matters:
- this is the strongest current fit for broad condition screening across biosignals
- it already has a released disease model and sleep-staging model in the official repo

Practical KYMA role:
- registry entry
- future `condition_screen` local pack
- only enable when the capture looks like a real multimodal sleep workflow

Sources:
- https://github.com/zou-group/sleepfm-clinical
- https://www.nature.com/articles/s41591-025-04133-4

### 2. ECG-FM

Use when:
- the active profile is ECG
- the goal is arrhythmia-style screening or ECG representation learning

Inputs:
- ECG waveforms

Outputs:
- ECG embeddings from the base model
- downstream label probabilities after fine-tuning heads

Why it matters:
- open repository
- checkpoints are published
- clear path for arrhythmia and reduced-LVEF style heads

Practical KYMA role:
- best first ECG foundation family to support
- use for rhythm-screening research heads, not bedside diagnosis

Sources:
- https://github.com/bowang-lab/ECG-FM
- https://huggingface.co/papers/2408.05178

### 3. LUNA / BioFoundation

Use when:
- the active profile is EEG
- the goal is abnormality, slowing, or artifact-oriented screening

Inputs:
- multi-channel EEG

Outputs:
- topology-agnostic EEG embeddings
- downstream abnormality and slowing heads after fine-tuning

Why it matters:
- strong recent EEG foundation route
- explicitly designed for heterogeneous EEG layouts

Practical KYMA role:
- primary EEG foundation encoder family
- pair with separate screening heads for abnormality or other review tasks

Sources:
- https://thorirmar.com/publication/2025-neurips-luna/
- https://thorirmar.com/project/biofoundation/

### 4. EEG Artifact + Seizure Screening

Use when:
- the active profile is EEG
- the goal is event review, artifact separation, or seizure-like screening in research workflows

Inputs:
- EEG windows

Outputs:
- seizure-vs-artifact style screening labels
- event probabilities

Why it matters:
- clearer immediate value than pretending a general EEG model can diagnose anything

Practical KYMA role:
- use as a specialized `condition_screen` or event-screen head
- keep language as `seizure-like event candidate` or `review flagged event`

Sources:
- https://github.com/pulp-bio/Artifact-Seizure
- https://www.nature.com/articles/s41598-026-41358-w

## EMG Reality Check

EMG is weaker today for open-weight clinical-condition heads than ECG or PSG.

What is realistic:
- fatigue / readiness
- asymmetric activation
- abnormal contraction pattern screening
- neuromuscular-pattern research heads

What is not mature enough to present as turnkey:
- robust open-weight diagnostic EMG condition models with clear deployment paths

Research anchor:
- https://www.nature.com/articles/s41598-025-03766-2

## What KYMA Should Do

### Now

- Keep condition outputs behind `research_only`.
- Route by modality before any model is touched.
- Show `active screening head`, `missing head`, or `not valid for this modality`.
- Expose outputs as:
  - screening label
  - confidence
  - top ranked candidates
  - source model
  - research-only warning

### Next

- add `condition_screen` TorchScript manifest support
- ship at least one real local screening head
- prefer ECG or PSG first, not EMG

### Avoid

- diagnosis language
- treatment recommendations
- patient self-testing claims
- implying a foundation encoder alone is a diagnosis model
