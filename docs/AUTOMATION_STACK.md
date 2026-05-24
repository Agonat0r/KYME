# KYMA Automation Stack

KYMA uses automation for four separate risks: broken app code, broken signal UI rendering, unsafe changes, and model regressions.

## Workflows

- `CI`: fast syntax checks for Python and dashboard JavaScript.
- `Browser Smoke`: starts KYMA with a synthetic stream and verifies the live canvas renders nonblank signal pixels in Chromium.
- `Security`: runs Gitleaks and Semgrep.
- `ML Pipeline`: runs the DVC prompt-model smoke stage, logs MLflow artifacts, and posts a CML report on pull requests.
- `Repo Maintenance`: opens an agent-ready maintenance issue on a schedule.

## Local Commands

```powershell
python -m py_compile server\main.py server\models.py scripts\ml_prompt_smoke.py
node --check dashboard\app.js
node --check tests\e2e\kyma-smoke.spec.js
python scripts\ml_prompt_smoke.py --out-dir reports\ml --artifact-dir artifacts\ml
```

For browser automation:

```powershell
npm install
npm run test:e2e
```

For the DVC stage:

```powershell
pip install dvc mlflow
dvc repro --no-commit prompt_model_smoke
```

## Notes

Automation should use synthetic streams by default. Hardware, LSL, OSC, and serial paths should stay optional so CI can run without lab equipment.
