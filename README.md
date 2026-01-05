# CXR Multimodal Baseline

Baseline project for chest X-ray analysis with image and tabular data.
Designed to be multimodal-ready, starting from image-only baselines.


## Project Structure

```
cxr-multimodal-baseline/
├─ src/cxrcli/          # Core dataset and pipeline modules
│  ├─ **init**.py
│  └─ dataset.py
├─ scripts/             # Local development and sanity-check scripts
├─ tests/               # Unit tests
├─ data/                # Local datasets (not versioned)
├─ README.md
└─ pyproject.toml
```


## Scope
- Chest X-ray image loading
- Structured metadata handling
- Reproducible baseline pipelines