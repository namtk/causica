[//]: # (This repository was cloned and modified locally. See the "Changes in this clone" section below for a summary of local additions and how to push them to your GitHub fork.)

## Changes in this clone (local additions)

This copy of the repository was edited locally to add small helper scripts and documentation to make it easier to run the example notebook from the command line, build graph constraints, and extract learned adjacency matrices from saved models.

New files added in this clone
- `docs/run_and_results_report.md` — a short run report capturing the dataset used, CLI commands, trainer settings, and where outputs/checkpoints were written for the example run performed locally.
- `docs/learned_adjacency_report.md` — explanation of how adjacency is represented in the repo, where to find the ENCO adjacency implementation, and sample code snippets for extracting learned graphs.
- `docs/hyperparameter_tuning_guide.md` — practical tuning guidance (data preprocessing, LR, batch size, auglag settings) and CLI examples for quick sweeps.
- `docs/terminal_command_history.txt` — a lightweight record of the terminal commands used during the local experiment session for reproducibility.
- `docs/multi_investment_sales_attribution_cli.md` — a concise markdown report that translates `examples/multi_investment_sales_attribution.ipynb` into a reproducible `causica.lightning.main` CLI command and checklist.
- `scripts/build_constraints.py` — helper script that reads `variables.json` (or `train.csv` header) and writes `constraints.npy` implementing the notebook's example constraints (Revenue row constraints, non-child attribute columns, engagement-node block constraints).
- `scripts/extract_sem_and_save_adj.py` — helper script that loads a saved SEM module or Lightning checkpoint (e.g., `outputs/last.ckpt`), extracts the learned adjacency (mode) and optionally mean, and writes CSV artifacts to `outputs/`.

Local artifacts produced by running the helper scripts
- `./tmp_causica/constraints.npy` — constraints file created by `scripts/build_constraints.py` (if you ran the generator script against `tmp_causica`).
- `./outputs/learned_adjacency_mode.csv` — adjacency CSV produced by `scripts/extract_sem_and_save_adj.py` when extracting from a local checkpoint.
- `./outputs/deci_sem_module.pt` — saved SEM module object produced by the extractor for later programmatic use.

Notes about these changes
- The added scripts and docs are utilities to reproduce the notebook workflow via CLI and to make it easy to re-run experiments and extract adjacency CSVs for downstream analysis. They are intentionally small, dependency-light, and conservative about overwriting existing files.
- No core library code or model algorithms were modified. These changes are purely utilities and documentation to simplify local experimentation.

---


[![Causica CI Build](https://github.com/microsoft/causica/actions/workflows/ci-build.yml/badge.svg)](https://github.com/microsoft/causica/actions/workflows/ci-build.yml)
[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/microsoft/causica)


# Causica

## Overview

Causal machine learning enables individuals and organizations to make better data-driven decisions. In particular, causal ML allows us to answer “what if” questions about the effect of potential actions on outcomes.

Causal ML is a nascent area, we aim  to enable a **scalable**, **flexible**, **real-world applicable end-to-end** causal inference framework. In perticular, we bridge between causal discovery, causal inference, and deep learning to achieve the goal.  We aim to develop technology can automate causal decision-making using existing observational data alone, output both the discovered causal relationships and estimate the effect of actions simultaneously.

Causica is a deep learning library for end-to-end causal inference, including both causal discovery and inference.  It implements deep end-to-end inference framework [2] and different alternatives.

This project splits the interventional decision making from observational decision making Azua repo found here [Azua](https://github.com/microsoft/project-azua).

This codebase has been heavily refactored, you can find the previous version of the code [here](https://github.com/microsoft/causica/releases/tag/v0.0.0).

# DECI: End to End Causal Inference

## Installation

The Causica repo is on PyPI so you can be pip installed:

```
pip install causica
```

## About

Real-world data-driven decision making requires causal inference to ensure the validity of drawn conclusions. However, it is very uncommon to have a-priori perfect knowledge of the causal relationships underlying relevant variables. DECI allows the end user to perform causal inference without having complete knowledge of the causal graph. This is done by combining the causal discovery and causal inference steps in a single model. DECI takes in observational data and outputs ATE and CATE estimates.

For more information, please refer to the [paper](https://arxiv.org/abs/2202.02195).


**Model Description**

DECI is a generative model that employs an additive noise structural equation model (ANM-SEM) to capture the functional relationships among variables and exogenous noise, while simultaneously learning a variational distribution over causal graphs. Specifically, the relationships among variables are captured with flexible neural networks while the exogenous noise is modelled as either a Gaussian or spline-flow noise model. The SEM is reversible, meaning that we can generate an observation vector from an exogenous noise vector through forward simulation and given a observation vector we can recover a unique corresponding exogenous noise vector. In this sense, the DECI SEM can be seen as a flow from exogenous noise to observations. We employ a mean-field approximate posterior distribution over graphs, which is learnt together with the functional relationships among variables by optimising an evidence lower bound (ELBO). Additionally, DECI supports learning under partially observed data.

**Simulation-based Causal Inference**

DECI estimates causal quantities (ATE) by applying the relevant interventions to its learnt causal graph (i.e. mutilating incoming edges to intervened variables) and then sampling from the generative model. This process involves first sampling a vector of exogenous noise from the learnt noise distribution and then forward simulating the SEM until an observation vector is obtained. ATE can be computed via estimating an expectation over the effect variable of interest using MonteCarlo samples of the intervened distribution of observations.

## How to run

The best place to start is the `examples/multi_investment_sales_attribution.ipynb` notebook. This explains how to fit a model using PyTorch Lightning and test ATE and ITE results.

For a more detailed introduction to the components and how they fit together, see the notebook `examples/csuite_example.ipynb`, for how to train a DECI model and check the causal discovery.

This will download the data from the CSuite Azure blob storage and train DECI on it. See [here](https://github.com/microsoft/csuite) for more info about CSuite datasets. The notebook will work on any of the available CSuite datasets.


**Specifying a noise model**

The noise exogenous model can be modified by changing the `noise_dist` field within `TrainingConfig`, either Gaussian or Spline are allowed.

The Gaussian model has Gaussian exogenous noise distribution with mean set to 0 while its variance is learnt.

The Spline model uses a flexible spline flow that is learnt from the data. This model provides most gains in heavy-tailed noise settings, where the Gaussian model is at risk of overfitting to outliers, but can take longer to train.

**Using a known Causal graph**

To use DECI to learn the functional relationships, remove the variational distribution terms from the loss and replace the sample with the known graph.

**Example using the CLI**

An example of how to run a training job with the noise distribution specified in the config `src/causica/config/lightning/default_gaussian.yaml` and the data configuration specified in `src/causica/config/lightning/default_data.yaml`:

```
python -m causica.lightning.main \
    --config src/causica/config/lightning/default_gaussian.yaml --data src/causica/config/lightning/default_data.yaml
```


## Further extensions

For now, we have removed Rhino and DDECI from the codebase but they will be added back. You can still access the previously released versions [here](https://github.com/microsoft/causica/releases/tag/v0.0.0).

# References

If you have used the models in our code base, please consider to cite the corresponding papers:

[1], **(VISL)** Pablo Morales-Alvarez, Wenbo Gong, Angus Lamb, Simon Woodhead, Simon Peyton Jones, Nick Pawlowski, Miltiadis Allamanis, Cheng Zhang, "Simultaneous Missing Value Imputation and Structure Learning with Groups", [ArXiv preprint](https://arxiv.org/abs/2110.08223)

[2], **(DECI)** Tomas Geffner, Javier Antoran, Adam Foster, Wenbo Gong, Chao Ma, Emre Kiciman, Amit Sharma, Angus Lamb, Martin Kukla, Nick Pawlowski, Miltiadis Allamanis, Cheng Zhang. Deep End-to-end Causal Inference. [Arxiv preprint](https://arxiv.org/abs/2202.02195) (2022)

[3], **(DDECI)** Matthew Ashman, Chao Ma, Agrin Hilmkil, Joel Jennings, Cheng Zhang. Causal Reasoning in the Presence of Latent Confounders via Neural ADMG Learning. [ICLR](https://openreview.net/forum?id=dcN0CaXQhT) (2023)

[4], **(Rhino)** Wenbo Gong, Joel Jennings, Cheng Zhang, Nick Pawlowski. Rhino: Deep Causal Temporal Relationship Learning with History-dependent Noise. [ICLR](https://openreview.net/forum?id=i_1rbq8yFWC) (2023)


# Development

## Poetry

We use Poetry to manage the project dependencies, they're specified in the [pyproject.toml](pyproject.toml). To install poetry run:

```
    curl -sSL https://install.python-poetry.org | python3 -
```

To install the environment run `poetry install`, this will create a virtualenv that you can use by running either `poetry shell` or `poetry run {command}`. It's also a virtualenv that you can interact with in the normal way too.

More information about poetry can be found [here](https://python-poetry.org/)

## mlflow

We use [mlflow](https://mlflow.org/) for logging metrics and artifacts. By default it will run locally and store results in `./mlruns`.
