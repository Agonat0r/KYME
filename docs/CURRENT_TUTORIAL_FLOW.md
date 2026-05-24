# KYMA Current Tutorial Flow

This document mirrors the in-app `Tour` button. Keep it aligned with `docs/CURRENT_WORKFLOW_GUIDE.md` and the current Build panel workflow.

## Tutorial Goal

Teach a new user how to move from biosignal source selection to a saved project, rebuild job, report, and package.

The tutorial should not feel like an Arduino-only walkthrough. Hardware output is one possible deployment target after the data, recipe, QA, and model are trustworthy.

## Current In-App Steps

1. **KYMA Prompt Pipeline**
   - Introduces the source -> prompt -> inspect -> train/rebuild -> report/package loop.
2. **Workflow Stages**
   - Explains Live, Record, Train, and Control as stages around the core pipeline.
3. **Pick The Biosignal**
   - Select EMG, EEG, ECG, EOG, or another profile before building.
4. **Choose Live Or Dataset**
   - Explains live hardware, synthetic, playback, LSL, OSC, serial, and dataset paths.
5. **Describe The Goal**
   - Shows where to ask for cleanup, artifact review, or product outcomes.
6. **Open Build**
   - Opens the prompt pipeline drawer.
7. **Prompt The Pipeline**
   - Shows the natural-language goal field.
8. **Set Input And Output**
   - Explains live electrodes, dataset import, LSL, playback, and output targets.
9. **Dataset Path**
   - Shows where CSV, NPZ, EDF/BDF, XDF, MAT, WAV, HDF5, Parquet, or session paths go.
10. **Generate Plan First**
   - Teaches that `Build Automatically` should be the default path and manual buttons are for inspection or repair.
11. **Plan Preview**
   - Shows recipe, channels, filters, labels, models, QA, training, exports, and registry.
12. **Projects And Jobs**
   - Shows saved projects, rebuild jobs, reports, and package downloads.
13. **Live Electrode Autopilot**
   - Explains that KYMA opens setup, asks the user to connect electrodes, then starts guided tasks only after the user is ready.
14. **Review Signal Spans**
   - Shows live review, persistent highlights, markers, and frozen regions.
15. **Live ML Review**
   - Shows decoded output and reminds users to save labels, QA, and metrics before trusting results.
16. **Record Clean Sessions**
   - Shows metadata and labels for live acquisition.
17. **Train And Compare**
   - Points to classic supervised training and validation.
18. **Deploy Outputs**
   - Shows LSL, OSC, serial, software, and hardware output paths.
19. **Safety**
   - Shows E-STOP for actuator workflows.
20. **Guide And Specialist Tools**
   - Points to Guide, Blocks, Filters, Workshop, Bench, and Firmware.

## Maintenance Rule

When the Build panel workflow changes, update:
- `dashboard/app.js` current workflow tour
- `dashboard/index.html` Guide tab
- `docs/CURRENT_WORKFLOW_GUIDE.md`
- this file
