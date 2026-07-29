# Article 107 — Bayesian damping data-sufficiency criterion

Reproducibility repository for:

*Criterio bayesiano de suficiencia de datos para estimación de amortiguamiento estructural: calibración sintética reproducible y caso operacional externo*

This repository contains the supplementary code, public-data provenance, derived results, figures, and SHA-256 manifest used for the MIIUM30-62 revision. It does not contain author manuscripts, response letters, personal information, or local audit traces.

## Contents

- `code/`: synthetic validation, operational acceleration pilot, modal-contrast source, and reproduction note.
- `data/`: public-data archives, extracted public acceleration records, and provenance.
- `results/`: CSV/JSON outputs and generated figures from the reported checks.
- `figures/`: manuscript figures at 300 dpi.
- `checksums_sha256.txt`: SHA-256 manifest for every file in the supplementary package.

## Reproduction

From the repository root:

```bash
python code/run_synthetic_validation.py
python code/run_real_acceleration_pilot.py
```

The synthetic workflow is fully specified and reproducible from the included scripts and configuration. The operational case is an exploratory re-estimation and is not presented as an independent damping ground truth.

## Data and claim boundary

The bridge acceleration source is the public Mendeley Data record `10.17632/d3by55pjh7.2`. Third-party source terms remain subject to their original license and attribution requirements. The repository supports reproducibility of the reported computational and exploratory checks; it does not claim complete experimental validation of damping or structural safety certification.

## Integrity

The supplementary package shipped with the revision contains the same files as this repository and records their SHA-256 values in `checksums_sha256.txt`. The journal-facing DOCX files and response letter remain in the separate submission package.
