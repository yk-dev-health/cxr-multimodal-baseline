# CXR Multimodal Baseline

This project demonstrates a baseline multimodal machine learning pipeline
combining chest X-ray images and structured clinical tabular data.

The focus is on data handling, pipeline structure, and reproducibility,
rather than achieving state-of-the-art model performance.

It is intended for engineers and data practitioners working with
real-world healthcare data, where multimodal inputs and data quality constraints are common.


## Why This Project

In clinical practice, important patient information is often split across multiple data sources:

* Medical images (e.g. chest X-rays)
* Structured clinical or administrative data (e.g. age, sex, admission details)

Single-modal models can miss clinically relevant context.

This project explores a simple and transparent multimodal baseline to show:

* how different data modalities can be combined
* how data preprocessing and alignment affect modelling
* how engineering decisions influence downstream ML results


## Project Positioning

This repository is intentionally designed as a **baseline implementation**.

* It prioritises clarity and structure over model complexity
* It avoids aggressive optimisation or complex fusion strategies
* It is suitable as a starting point for further experimentation

The goal is to demonstrate engineering and data handling skills in a healthcare ML context,
not research novelty.


## Core Approach

The pipeline consists of:

* Image preprocessing for chest X-ray data
* Tabular feature preprocessing for clinical variables
* Simple feature fusion at the model level
* Clear separation between data loading, training, and evaluation

All steps are implemented with reproducibility and readability in mind.


## Example Use Case

This project is intended for scenarios such as:

* Prototyping multimodal clinical ML pipelines
* Exploring feasibility before more complex modelling
* Understanding how clinical tabular data complements imaging data
* Demonstrating ML system structure in a healthcare context


## Relation to Clinical Data Quality Analysis

This project complements the
**clinical-data-quality-analysis** repository.

While that project focuses on detecting and reporting clinical data quality issues,
this repository demonstrates how prepared datasets can be used in a downstream multimodal ML pipeline.

Together, the two projects reflect a typical real-world workflow:

1. Assess and understand data quality
2. Prepare and validate datasets
3. Apply baseline modelling


## Dataset

The project uses publicly available chest X-ray and clinical datasets
(for demonstration and educational purposes only).

No private or identifiable patient data is included in this repository.


## Reproducibility

* Deterministic preprocessing steps
* Configuration-driven parameters
* Clear module boundaries
* Minimal external dependencies

This structure makes the pipeline easier to understand, test, and extend.


## Project Structure

```
cxr-multimodal-baseline/
├─ data/            # Dataset references and loaders
├─ src/             # Core pipeline and model code
├─ config/          # Configuration files
├─ notebooks/       # Exploratory experiments (non-production)
├─ outputs/         # Model outputs and logs
```


## Project Status

* Baseline multimodal pipeline implemented
* Image + tabular fusion demonstrated
* Designed for extension and experimentation

Future work may include alternative fusion strategies or extended evaluation,
but these are intentionally out of scope for the baseline.


## Intended Audience

This project is aimed at:

* Healthcare data engineers
* Backend or data-focused software engineers
* ML practitioners interested in applied clinical ML
* Engineers transitioning from healthcare or medical backgrounds into ML


## Disclaimer

This repository is for educational and demonstration purposes only.
It is not intended for clinical use or medical decision-making.
