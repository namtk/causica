# Learned adjacency — implementation and evaluation (Causica)

This document explains how the codebase represents a learnable adjacency distribution (graph), how that distribution is used to construct SEMs, and how the evaluation metrics `eval/adjacency.f1` and `eval/orientation.f1` are computed and logged.

It points to the canonical files in the repository and includes short example snippets to extract the learned adjacency from a Lightning checkpoint.

---

## Key files

- `src/causica/distributions/adjacency/enco.py` — ENCO adjacency distribution implementation (learnable logits and sampling).
- `src/causica/sem/sem_distribution.py` — wraps an adjacency distribution into SEMs (SEMDistribution / SEMDistributionModule).
- `src/causica/lightning/modules/deci_module.py` — the `DECIModule` LightningModule: constructs the adjacency module, samples graphs at test time, and logs metrics.
- `src/causica/graph/evaluation_metrics.py` — implementations of `adjacency_f1`, `orientation_f1` and helpers.
- `tmp_causica/adj_matrix.csv` — ground-truth adjacency matrix produced by the synthetic data generator (node ordering matches `tmp_causica/variables.json`).


## 1) How adjacency is parameterized (ENCO)

The default adjacency parameterization is the ENCO parameterization implemented in `src/causica/distributions/adjacency/enco.py`.

High-level summary:

- The ENCO parameterization separates edge existence from orientation.
  - `logits_exist`: a learnable `(n, n)` matrix of directed-edge logits representing the existence of each directed edge (diagonal ignored).
  - `logits_orient`: a learnable vector of length `n(n-1)/2` representing orientation logits for unordered node pairs (packed as a triangular vector).

- The distribution constructs a matrix of independent Bernoulli logits for each directed edge by combining `logits_exist` and `logits_orient` using a triangular transform (`fill_triangular`) and a log-sum-exp derivation (see `_get_independent_bernoulli_logits`).

- Provided APIs:
  - `sample()` — sample binary adjacency matrices (no gradients).
  - `relaxed_sample(sample_shape, temperature)` — Gumbel-softmax relaxed sampling used during training so gradients can flow.
  - `.mean` — per-edge probabilities in [0, 1] (Bernoulli means).
  - `.mode` — most probable binary adjacency (deterministic).

Practical consequence: the ENCO module learns parameters that can be used to sample multiple plausible binary graphs or to obtain a per-edge probability matrix.


## 2) How the adjacency distribution is used to create SEMs

- The `DECIModule` constructs an adjacency distribution module during setup:
  - `adjacency_dist = ENCOAdjacencyDistributionModule(num_nodes)`
  - Optionally wrapped by `ConstrainedAdjacency` if a constraint matrix is provided.

- The `SEMDistributionModule` composes this adjacency distribution with functional relationship modules and noise modules to create a distribution over SEMs.

- At test time, the evaluation code samples graphs by calling:

```python
sem_samples = self.sem_module().sample(torch.Size([NUM_GRAPH_SAMPLES]))
graph_samples = [sem.graph for sem in sem_samples]
```

Each `sem` in `sem_samples` contains a `graph` attribute (a binary adjacency tensor). By default, `NUM_GRAPH_SAMPLES = 100` so metrics are computed as an average across these sampled graphs.


## 3) How `eval/adjacency.f1` and `eval/orientation.f1` are computed

- The code that computes and logs these metrics is in `src/causica/lightning/modules/deci_module.py` (`test_step_graph`).

- Procedure:
  1. Sample `NUM_GRAPH_SAMPLES` binary graphs from the learned adjacency distribution (`sem_module().sample(...)`).
  2. For each sampled binary graph, compute `adjacency_f1(true_adj, graph)` and `orientation_f1(true_adj, graph)`.
  3. Average the per-sample F1 scores and log the averaged value as `eval/adjacency.f1` and `eval/orientation.f1`.

- Implementation details of the metrics are in `src/causica/graph/evaluation_metrics.py`:
  - `adjacency_f1` compares the undirected presence/absence of edges (converts adjacency matrices into a vector form and compares boolean presence).
  - `orientation_f1` compares edge directions (scores whether the signed difference matches the ground truth for non-zero edges).
  - Both metrics rely on `_to_vector` which extracts triangular parts and encodes no-edge, directed-edge, or double-edge cases.

- Important: the logged metrics are expectations over the sampled discrete graphs. They are *not* directly computed from the adjacency logits or probabilities but from sampled binary graphs (though you can also evaluate metrics on the distribution mean or mode if you prefer a single deterministic estimate).


## 4) How to extract the learned adjacency from a checkpoint

Preferred approach (recommended):

1. Load the trained `DECIModule` from a checkpoint and use the adjacency module API:

```python
from causica.lightning.modules.deci_module import DECIModule
module = DECIModule.load_from_checkpoint('outputs/last.ckpt', map_location='cpu')
# adjacency distribution module (ENCO) lives at:
adj_module = module.sem_module.adjacency_module
# probabilistic adjacency (per-edge probabilities)
prob_adj = adj_module().mean  # tensor (n, n), values in [0, 1]
# deterministic mode adjacency
mode_adj = adj_module().mode  # tensor (n, n), binary
```

2. Save `prob_adj` and/or `mode_adj` as CSV for inspection (use `tmp_causica/variables.json` to label rows/cols).

Fallback approach (state_dict inspection):

- If unable to instantiate the module for any reason, load the checkpoint `state_dict` and inspect parameter keys. Typical parameter keys for the ENCO module are:
  - `sem_module.adjacency_module.logits_exist`
  - `sem_module.adjacency_module.logits_orient`

- Reconstruct the per-edge logits using the same internal function `_get_independent_bernoulli_logits()` (requires `fill_triangular`) and compute Bernoulli means.

Example one-shot script to extract and save CSV (run from repo root using the repo Python environment):

```bash
.venv310/bin/python - <<'PY'
import glob, json, os
import torch, pandas as pd
from causica.lightning.modules.deci_module import DECIModule

ckpts = glob.glob('outputs/**/*.ckpt', recursive=True) + glob.glob('outputs/*.ckpt')
if not ckpts:
    raise SystemExit('No checkpoint found in outputs/')
ckpt = ckpts[0]
module = DECIModule.load_from_checkpoint(ckpt, map_location='cpu')
adj_module = module.sem_module.adjacency_module
prob_adj = adj_module().mean.detach().cpu().numpy()
mode_adj = adj_module().mode.detach().cpu().numpy().astype(int)
vars_meta = json.load(open('tmp_causica/variables.json'))
names = [v.get('name', f'v{i}') for i, v in enumerate(vars_meta)]

pd.DataFrame(prob_adj, index=names, columns=names).to_csv('outputs/learned_adjacency_probs.csv')
pd.DataFrame(mode_adj, index=names, columns=names).to_csv('outputs/learned_adjacency_mode.csv')
print('Saved outputs/learned_adjacency_probs.csv and outputs/learned_adjacency_mode.csv')
PY
```


## 5) How to compare to ground truth and reproduce the logged metrics

- `tmp_causica/adj_matrix.csv` contains the ground-truth adjacency matrix for the synthetic dataset. Node ordering matches `tmp_causica/variables.json`.

- To compare `outputs/learned_adjacency_mode.csv` directly using the repo metrics or compute F1 via simple code:

```python
import pandas as pd
from causica.graph.evaluation_metrics import adjacency_f1, orientation_f1
import torch

gt = pd.read_csv('tmp_causica/adj_matrix.csv', index_col=0).values
pred = pd.read_csv('outputs/learned_adjacency_mode.csv', index_col=0).values

# convert to torch tensors
gt_t = torch.tensor(gt, dtype=torch.float32)
pred_t = torch.tensor(pred, dtype=torch.float32)

print('adjacency_f1', adjacency_f1(gt_t, pred_t).item())
print('orientation_f1', orientation_f1(gt_t, pred_t).item())
```

Note: the logged metrics in training are averages over multiple sampled binary graphs. To reproduce the logged values, sample multiple graphs from the adjacency distribution and average per-sample metrics. For example:

```python
from causica.lightning.modules.deci_module import DECIModule
module = DECIModule.load_from_checkpoint('outputs/last.ckpt', map_location='cpu')
NUM_SAMPLES = 100
sem_samples = module.sem_module().sample(torch.Size([NUM_SAMPLES]))
graphs = [s.graph for s in sem_samples]
# compute adjacency_f1/orientation_f1 for each graph vs ground truth and average
```


## 6) Notes, caveats, and suggestions

- The ENCO parameterization separates existence and orientation; orientation is encoded in a triangular vector and is combined with `logits_exist` to get per-directed-edge logits.

- The training objective includes an Augmented Lagrangian DAG constraint and a graph prior (`GibbsDAGPrior`). These regularizers affect learned adjacency but are orthogonal to metric calculation.

- Logged `eval/adjacency.f1` and `eval/orientation.f1` are stochastic averages (since they average over sampled discrete graphs). If deterministic evaluation is preferred, compute metrics on the distribution `mean` (probabilities) thresholded at a cutoff, or on `.mode`.

- If variables have heterogeneous scales, standardizing inputs often stabilizes likelihood and ATE metrics; adjacency recovery is less sensitive to scale but functional-learning may be affected.


