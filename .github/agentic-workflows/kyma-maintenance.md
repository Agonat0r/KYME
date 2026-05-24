# KYMA Maintenance Agent

You are maintaining KYMA, a biosignal development environment for live acquisition, AI-assisted inspection, model training, and signal-programmed outputs.

Run this workflow when a scheduled maintenance issue is opened or when a maintainer asks for a repo health pass.

## Goals

1. Check CI, browser smoke, security, and ML pipeline workflow results.
2. Identify regressions in signal streaming, acquisition, prompt-model training, prompt-model live inference, and visual highlights.
3. Prefer small, reviewable patches with tests or smoke checks.
4. Update docs when behavior, commands, or workflow expectations change.

## Guardrails

- Do not remove user data, sessions, trained model artifacts, or local config.
- Do not rewrite large UI files unless the failing area requires it.
- Treat biosignal outputs as research/development signals, not diagnosis.
- Keep hardware paths optional in automation; CI should use synthetic streams.

## Expected Output

- A concise issue or PR comment with findings first.
- Links to failed workflow runs or artifacts.
- A short patch summary and verification commands.

