# Graded Anomaly Detection

Detection of anomalies in surveillance video with graded severity levels using 3D-CNN.

## Overview

This project addresses anomaly detection in restricted-zone surveillance, where actions are classified not only as normal or anomalous, but by **severity level**. This allows the system to distinguish between minor deviations, suspicious behavior, and explicit violations — providing more useful information to security operators than a binary decision.

## Severity Levels

| Level | Label | Description | Examples |
|---|---|---|---|
| 0 | Normal | Allowed actions | walking, reading, writing |
| 1 | Low | Allowed but not always | computer use, phone use, counting money |
| 2 | High | Suspicious, requires attention | smoking, raking photo, arguing |
| 3 | Critical | Explicit violations | fighting, arson, robbery, shooting |

## Architecture

The model uses a shared 3D-CNN backbone with two heads:

- **Head 1** — soft binary classifier predicting a severity score (0.0–1.0).
- **Head 2** — multi-class classifier predicting the action type (N target classes).

At inference, a high threshold on Head 1 is used for global scene analysis (detecting only explicit anomalies), while a lower threshold is used for local analysis (detecting all severity levels).

## Dataset

The dataset combines action classes from open sources (Kinetics-700, UCF101, UCF-Crime, DCSASS, RLVD) selected for relevance to restricted-zone security. It includes both physical violations (fighting, robbery) and information security violations (computer use, phone use, photography). Some of the actions were compiled manually based on YouTube clips.

## Status

Work in progress. Training pipeline and evaluation are under development.

## License

TBD


