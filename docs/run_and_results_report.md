# Run & Results — how to run, generated data, learned adjacency, metrics, and normalization

This document explains how to run the project (quick commands), what the synthetic dataset files mean, how the model reconstructs a learned adjacency (graph), how the evaluation metrics are computed and interpreted, and how data scale affects metrics (with normalization guidance).

## 1 — Quick run (environment + install)

The project requires Python 3.10 (see `pyproject.toml`). Example quick setup that worked in this session using a local venv named `.venv310`:

```bash
# create venv (example: using conda-run to get python 3.10, then standard venv)
conda create -n causica_py310 python=3.10 -y
conda run -n causica_py310 python -m venv .venv310
source .venv310/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Run the training CLI (fast smoke test):

```bash
# smoke test (fast_dev_run)
.venv310/bin/python -m causica.lightning.main \
  --config src/causica/config/lightning/default_gaussian.yaml \
  --data src/causica/config/lightning/default_data.yaml \
  --data.init_args.dataset_path . \
  --data.init_args.dataset_name tmp_causica \
  --trainer.fast_dev_run True
```

### CLI arguments (what the common flags mean)

The `causica.lightning.main` entrypoint exposes the Lightning CLI configured with the project's YAML configs. All available options can be inspected with `--help`:

```bash
.venv310/bin/python -m causica.lightning.main --help
```

Common arguments and what they control:

- `--config <path>`: path to a YAML file with default model/trainer/hyperparameter values (example: `src/causica/config/lightning/default_gaussian.yaml`). This file provides defaults for the run.
- `--data <path>`: path to a YAML file configuring the data module (example: `src/causica/config/lightning/default_data.yaml`).
- `--data.init_args.<name> <value>`: override an initialization argument passed to the DataModule. Common keys:
  - `dataset_path`: directory containing datasets (e.g., `.`)
  - `dataset_name`: dataset directory name under `dataset_path` (e.g., `tmp_causica`)
  - `batch_size`: training batch size
  - other data-specific flags defined in the chosen DataModule
- `--trainer.<name> <value>`: override `pytorch_lightning.Trainer` arguments exposed by the CLI. Common trainer keys:
  - `max_epochs`: number of training epochs
  - `fast_dev_run`: run a single batch/epoch for smoke testing (`True` or `False`)
  - `accelerator`: device accelerator to use (e.g., `cpu`, `mps`, `gpu`)
  - `devices`: number of devices (e.g., `1`)
  - `logger`: logger config (depends on configured loggers)
- `--seed <int>`: RNG seed to make runs reproducible.
- `--ckpt_path <path>`: path to a checkpoint to resume from (Lightning uses `'last'` by default when resuming via CLI behavior).

Examples of nested overrides:

```bash
# set data batch size and run 10 epochs on CPU
.venv310/bin/python -m causica.lightning.main \
  --config src/causica/config/lightning/default_gaussian.yaml \
  --data src/causica/config/lightning/default_data.yaml \
  --data.init_args.dataset_path . \
  --data.init_args.dataset_name tmp_causica \
  --data.init_args.batch_size 32 \
  --trainer.max_epochs 10 \
  --trainer.accelerator cpu
```

Notes on precedence:
- CLI overrides take precedence over values in the `--config` YAML. Use the YAML files for sensible defaults and the CLI to tweak experiments quickly.


Run a short real training (1 epoch on MPS with fallback):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv310/bin/python -m causica.lightning.main \
  --config src/causica/config/lightning/default_gaussian.yaml \
  --data src/causica/config/lightning/default_data.yaml \
  --data.init_args.dataset_path . \
  --data.init_args.dataset_name tmp_causica \
  --trainer.max_epochs 1 \
  --trainer.accelerator mps \
  --trainer.devices 1
```

Notes:
- If MPS fails for unsupported ops, set `PYTORCH_ENABLE_MPS_FALLBACK=1` to let unsupported operators fall back to CPU (affects performance, but usually not correctness).
- By default training logs and checkpoints are written to `./outputs` (e.g., `outputs/last.ckpt`). If an old checkpoint is incompatible with new model shapes, move it with `mv outputs outputs.bak` and re-run.


## 2 — Generated dataset (what files mean)

When running the data generator or use the provided synthetic dataset, the dataset directory (e.g., `tmp_causica`) contains:

- `variables.json`
  - Metadata for each variable in the dataset. Includes variable name, type (distribution), index/order, and other variable-specific metadata. The order here defines rows/columns for adjacency matrices and the columns in CSVs.
- `train.csv`, `test.csv`, `val.csv`
  - Observational samples. Each CSV row is one sample/observation. Columns correspond to variables (in the same order as `variables.json`).
- `adj_matrix.csv`
  - The ground-truth adjacency matrix used to generate the synthetic data. This is the true DAG (directed edges) used by the generator; use it to evaluate recovered graphs. Rows and columns correspond to `variables.json` ordering.
- `interventions.json`
  - If present, per-sample or per-experiment intervention specifications used for intervention-aware evaluation.
- `counterfactuals.json`
  - Optional counterfactuals used for ITE/ATE evaluation.
- `adj_matrix.csv` should be considered the canonical ground truth for graph recovery metrics.

Quick checks:

```bash
jq . tmp_causica/variables.json
head -n 5 tmp_causica/train.csv
```


### Running-example dataset used in this session

The examples and runs in this report used a locally generated synthetic dataset saved in `./tmp_causica`.
You can reproduce it with the repository's generator; the exact command used in this session was:

```bash
.venv310/bin/python -m causica.data_generation.generate_data \
  --num-samples-train 1024 \
  --num-samples-test 128 \
  --num-variables 12 \
  --num-edges 20 \
  --datadir ./tmp_causica \
  --overwrite \
  --plot-kind ''
```

Quick facts observed from the dataset that was generated here:
- Training rows: 1,023 (CSV excludes header; generator asked for 1,024 samples).
- Number of variables: 12 (matches `variables.json`).
- Variable scales: means near zero but heterogeneous standard deviations — some variables have small std (~0.5–2), others much larger (tens to hundreds). This heterogeneity explains large-magnitude log-likelihoods and ATE_RMSE values unless data are standardized.

Files to inspect for this dataset:
- `tmp_causica/variables.json` — variable metadata and ordering
- `tmp_causica/train.csv`, `tmp_causica/test.csv`, `tmp_causica/val.csv` — observational samples
- `tmp_causica/adj_matrix.csv` — ground-truth adjacency matrix (DAG used to generate data)

For a smaller quick-debug dataset, re-run the generator with fewer samples (for example `--num-samples-train 8 --num-samples-test 2 --num-variables 5`).



## 3 — How the model represents & produces a learned adjacency

Key implementation pieces (where to look in the repo):

- `src/causica/distributions/adjacency/enco.py`
  - ENCO parameterization: learns `logits_exist` (an n×n matrix) and `logits_orient` (triangular vector of orientations).
  - Has `.mean` (per-edge probabilities in [0,1]), `.mode` (most likely binary adjacency), `sample()` and `relaxed_sample()` (Gumbel-softmax) used for training.
- `src/causica/sem/sem_distribution.py`
  - Combines adjacency distribution + functional relationships + noise module into a distribution over SEMs. Each SEM carries a `.graph` field (a binary adjacency).
- `src/causica/lightning/modules/deci_module.py`
  - `DECIModule` constructs the adjacency module and, at test time, samples multiple graphs `sem_module().sample(torch.Size([NUM_GRAPH_SAMPLES]))` and extracts `sem.graph` from each sample.

Practical extraction (recommended):

- Load the Lightning module checkpoint and call the adjacency module API to get probabilities or mode:

```bash
.venv310/bin/python - <<'PY'
from causica.lightning.modules.deci_module import DECIModule
import glob, json, pandas as pd
ckpts = glob.glob('outputs/**/*.ckpt', recursive=True) + glob.glob('outputs/*.ckpt')
ckpt = ckpts[0]
module = DECIModule.load_from_checkpoint(ckpt, map_location='cpu')
adj_module = module.sem_module.adjacency_module
prob_adj = adj_module().mean.detach().cpu().numpy()
mode_adj = adj_module().mode.detach().cpu().numpy().astype(int)
vars_meta = json.load(open('tmp_causica/variables.json'))
names = [v.get('name', f'v{i}') for i,v in enumerate(vars_meta)]
import pandas as pd
pd.DataFrame(prob_adj, index=names, columns=names).to_csv('outputs/learned_adjacency_probs.csv')
pd.DataFrame(mode_adj, index=names, columns=names).to_csv('outputs/learned_adjacency_mode.csv')
print('Saved outputs/learned_adjacency_probs.csv and outputs/learned_adjacency_mode.csv')
PY
```

Fallback: inspect checkpoint `state_dict` keys — adjacency logits are usually under:
- `sem_module.adjacency_module.logits_exist`
- `sem_module.adjacency_module.logits_orient`


### Domain-specific graph constraints (CLI usage)

A constraint matrix can be supplied to force or prohibit edges during learning. The `DECIModule` accepts a
`constraint_matrix_path` argument (expected to be a `.npy` file) which is loaded in `prepare_data()` and applied to the
adjacency distribution via `ConstrainedAdjacency` during `setup()`.

Constraint format and semantics:
- File format: NumPy `.npy` containing an `(n, n)` float32 matrix where `n` is the number of variables.
- Values:
  - `NaN` — no constraint
  - `0.0` — negative constraint (edge prohibited)
  - `1.0` — positive constraint (edge required)
- The matrix rows/columns must follow the variable ordering in `tmp_causica/variables.json`.

Create and save an example constraint matrix (Python):

```python
import numpy as np
n = 12
cm = np.full((n, n), np.nan, dtype=np.float32)
# prohibit edge 2->5 and require edge 7->3 as examples
cm[2, 5] = 0.0
cm[7, 3] = 1.0
np.fill_diagonal(cm, np.nan)
np.save('tmp_causica/constraints.npy', cm)
```

Pass the constraint file to the CLI with a model init-arg override:

```bash
.venv310/bin/python -m causica.lightning.main \
  --config src/causica/config/lightning/default_gaussian.yaml \
  --data src/causica/config/lightning/default_data.yaml \
  --data.init_args.dataset_path . \
  --data.init_args.dataset_name tmp_causica \
  --model.init_args.constraint_matrix_path tmp_causica/constraints.npy \
  --trainer.max_epochs 5
```

Notes:
- Only `.npy` is supported by `prepare_data()` (other formats will raise an error).
- Make sure constraints match variable ordering and count; changing `num_variables` requires regenerating constraints.
- Can also set `constraint_matrix_path` in a YAML config and override via CLI. If programmatic control is needed, set
  `lightning_module.constraint_matrix` directly in a notebook before training.



## 4 — The metrics: what was logged and how they are computed

Metrics in logs (examples):
- `eval/test_LL` — test log-likelihood per-datapoint.
- `eval/adjacency.f1` — adjacency F1 score (edge presence recovery), averaged across sampled graphs.
- `eval/orientation.f1` — orientation F1 score (direction correctness) averaged across sampled graphs.
- `eval/Interventional_LL` — log-likelihood on interventional data (if present).
- `eval/ATE_RMSE` / `eval/ITE_RMSE` — errors for causal effect estimation.

How adjacency/orientation metrics are computed (code reference):
- `DECIModule.test_step_graph` calls `sem_module().sample(NUM_GRAPH_SAMPLES)` to get sampled SEMs; it extracts `sem.graph` from each and computes:
  - `adj_f1 = mean(adjacency_f1(true_adj, sampled_graph) for sampled_graph in samples)`
  - `orient_f1 = mean(orientation_f1(true_adj, sampled_graph) for sampled_graph in samples)`
- Metric implementations are in `src/causica/graph/evaluation_metrics.py`:
  - `adjacency_f1` and `orientation_f1` convert adjacency matrices to a vector form (via `_to_vector`), then compute precision and recall and derive F1.
  - `adjacency_f1` compares undirected presence/absence of an edge pair; `orientation_f1` compares signed directions.

Important: the logged F1s are expectations under the learned adjacency distribution (because they average many sampled discrete graphs). For a single deterministic evaluation, compute metrics on `.mode` or threshold `.mean`.


## 5 — How to interpret the metrics

- adjacency.f1 (0–1): higher means the model recovers more of the true undirected edges. 0.0 = no edges recovered; 1.0 = perfect recovery.
- orientation.f1 (0–1): higher means edge directions are correctly recovered. Orientation is usually harder and improves slower than adjacency.
- test_LL: higher (less negative) log-likelihood per-datapoint is better — but LL values are dependent on data scaling and noise models; compare relative improvements rather than absolute magnitudes unless data is normalized.
- ATE_RMSE / ITE_RMSE: lower is better; these measure counterfactual/interventional effect estimation errors and scale with variable magnitude.

Practical expectations:
- A single epoch is short; expect partial adjacency recovery (adjacency.f1 perhaps ~0.3–0.6 depending on dataset size and signal), and lower orientation.f1.
- Longer training and more data -> adjacency and orientation should improve.
- Use multiple seeds and average metrics to measure robustness.


## 6 — How data scale affects metrics & normalization guidance

Why scale matters
- Log-likelihood (`test_LL`) and ATE/ITE RMSE scale directly with the magnitude/variance of variables. If variables have large variance, LL will be large in magnitude and ATE_RMSE will be larger in absolute terms.
- Graph recovery metrics (F1) are less sensitive to scale, since they compare binary adjacency matrices. However, functional learning and likelihood optimization can be affected by heterogeneous scales across variables (training instability, slower convergence).

Recommended approaches
1. Standardize variables (z-score) per variable: subtract mean and divide by standard deviation computed on the training set. This makes LL and ATE magnitudes more interpretable and stabilizes optimization.

Simple in-place standardization script (Python):

```python
# scripts/standardize_dataset.py (run from repo root)
import json, numpy as np, pandas as pd
p = 'tmp_causica'
df = pd.read_csv(f'{p}/train.csv')
means = df.mean()
stds = df.std(ddof=0).replace(0, 1.0)
df_norm = (df - means) / stds
df_norm.to_csv(f'{p}/train.csv', index=False)
# apply same transform to val/test
for name in ('val.csv','test.csv'):
    df2 = pd.read_csv(f'{p}/{name}')
    pd.DataFrame((df2 - means) / stds).to_csv(f'{p}/{name}', index=False)
# save transform for later
json.dump({'means': means.to_dict(), 'stds': stds.to_dict()}, open(f'{p}/scaling.json','w'))
print('Standardized dataset in', p)
```

Run:

```bash
python scripts/standardize_dataset.py
```

2. Regenerate synthetic data with smaller noise / smaller coefficient scales if keeping raw variables is preferred (use the data generation CLI arguments that control noise magnitude and variable scaling).

Effect of normalization
- After z-scoring the dataset, `test_LL` values will be comparable across runs (units become "nats per standardized variable") and ATE_RMSE will be on a standard-scale; effects can be reverted by applying inverse transforms when interpreting predictions.


## 7 — Reproducible checklist & useful commands

Inspect data / variables:
```bash
jq . tmp_causica/variables.json
head -n 5 tmp_causica/train.csv
```

Extract learned adjacency from checkpoint (probabilities & mode):
```bash
# run the extraction snippet from docs or use the helper above
.venv310/bin/python - <<'PY'
# (see the extract snippet in this document)
PY
```

Compare learned mode vs ground truth using repo metrics:
```bash
.venv310/bin/python - <<'PY'
import pandas as pd, torch
from causica.graph.evaluation_metrics import adjacency_f1, orientation_f1
truth = pd.read_csv('tmp_causica/adj_matrix.csv', index_col=0).values
pred = pd.read_csv('outputs/learned_adjacency_mode.csv', index_col=0).values
print('adjacency_f1', adjacency_f1(torch.tensor(truth), torch.tensor(pred)).item())
print('orientation_f1', orientation_f1(torch.tensor(truth), torch.tensor(pred)).item())
PY
```

Run longer training (10 epochs) on MPS with fallback:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv310/bin/python -m causica.lightning.main \
  --config src/causica/config/lightning/default_gaussian.yaml \
  --data src/causica/config/lightning/default_data.yaml \
  --data.init_args.dataset_path . \
  --data.init_args.dataset_name tmp_causica \
  --trainer.max_epochs 10 \
  --trainer.accelerator mps \
  --trainer.devices 1
```


