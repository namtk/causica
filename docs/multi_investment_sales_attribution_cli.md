## Multi-investment Sales Attribution — Notebook Summary & CLI

This document translates the notebook `examples/multi_investment_sales_attribution.ipynb` into a compact markdown report and a reproducible CLI command for `causica.lightning.main`.

### Purpose
- Demonstrate learning causal graphs with DECI and estimating treatment effects (ATE and ITE) from observational data.
- Use a simulated multi-investment sales dataset with known ground-truth graph and effects to compare estimates.

### Main notebook steps (condensed)

1. Environment & imports
   - Uses PyTorch, PyTorch Lightning, Causica modules, pandas, networkx, numpy, matplotlib.
   - Optional: install `graphviz`/`pygraphviz` for nicer graph visuals.

2. Data loading and metadata
   - Load a CSV data table and a variable metadata JSON.
   - Create a `BasicDECIDataModule` with the dataframe (features only), a list of `Variable` metadata, `batch_size` (e.g. 128) and `normalize=True`.
   - Derive `num_nodes` from the data module's keys.

3. (Optional) Domain-specific graph constraints
   - Create a constraint matrix of shape `(num_nodes, num_nodes)` with values:
     - `NaN` = no constraint
     - `0.0` = negative constraint (edge cannot exist)
     - `1.0` = positive constraint (edge must exist)
   - Example constraints from the notebook:
     - Revenue cannot cause other nodes (set revenue row to 0.0 except allow Planning Summit).
     - Certain attributes are root variables (no parents) — set entire column(s) to 0.0.
     - Engagement nodes (Tech Support, Discount, New Engagement Strategy) do not cause each other.

4. Model configuration & training (DECI)
   - Set random seed (`pl.seed_everything`).
   - Create `DECIModule` with key hyperparameters (notebook examples):
     - `noise_dist = ContinuousNoiseDist.GAUSSIAN`
     - `prior_sparsity_lambda = 43.0`, `init_rho = 30.0`, `init_alpha = 0.20`
     - `auglag_config` with `max_inner_steps`, `max_outer_steps`, and `lr_init_dict` for submodules
   - Attach `constraint_matrix` (if used) to the lightning module via `lightning_module.constraint_matrix = torch.tensor(constraint_matrix)`.
   - Create `pl.Trainer` with `accelerator`, `max_epochs` (notebook uses 2000), `fast_dev_run` (for tests), callbacks, and checkpointing options (notebook example disables checkpointing).
   - Run `trainer.fit(lightning_module, datamodule=data_module)`.

5. Save/load trained model
   - Save the SEM module: `torch.save(lightning_module.sem_module, "deci.pt")`.
   - Load for inference: `sem_module: SEMDistributionModule = torch.load("deci.pt")` and extract the most likely SEM with `sem = sem_module().mode` and `sem.graph`.

6. Treatment effect estimation (ATE & ITE)
   - ATE: create intervened SEMs `sem.do(interventions=...)` for treatment=1 vs 0, sample many draws (e.g., 20k), apply `normalizer.inv()`, then compute mean difference and standard error.
   - ITE: compute `base_noise = sem.sample_to_noise(data_module.dataset_train)` then use `do_sem.noise_to_sample(base_noise)` to get individualized counterfactuals and compute per-sample differences.

7. Visualize and compare with ground truth
   - Plot ATE estimates vs ground-truth ATEs (error bars).
   - Scatter estimated ITEs against ground-truth ITEs.

### CLI translation (runnable example)
Below is a single CLI command for `causica.lightning.main` that reproduces the notebook workflow. Adjust the paths, dataset name, and hyperparameters to your environment.

Notes / assumptions:
- The dataset is organized in a dataset folder (for example `tmp_causica`) beneath a root `dataset_path` the CLI expects.
- If graph constraints are applied, create a NumPy file at `constraints.npy` with shape `(num_nodes, num_nodes)` and NaN/0/1 semantics.
- The CLI follows the project LightningCLI field naming (dotted flags like `model.init_args.*`).
- If running on Apple MPS and you observe unsupported op errors, add `PYTORCH_ENABLE_MPS_FALLBACK=1` to allow CPU fallback for missing operators.

Replace the bracketed paths/values below before running.

```bash
# Main training command (replace paths and flags as needed)
PYTORCH_ENABLE_MPS_FALLBACK=1 \
  .venv310/bin/python -m causica.lightning.main \
  --config src/causica/config/lightning/default_gaussian.yaml \
  --data src/causica/config/lightning/default_data.yaml \
  --data.init_args.dataset_path /path/to/datasets_root \
  --data.init_args.dataset_name tmp_causica \
  --data.init_args.batch_size 128 \
  --data.init_args.normalize True \
  --model.init_args.prior_sparsity_lambda 43.0 \
  --model.init_args.init_rho 30.0 \
  --model.init_args.init_alpha 0.20 \
  --model.init_args.auglag_config.max_inner_steps 3400 \
  --model.init_args.auglag_config.max_outer_steps 8 \
  --model.init_args.auglag_config.lr_init_dict.icgnn 0.00076 \
  --model.init_args.auglag_config.lr_init_dict.vardist 0.0098 \
  --model.init_args.auglag_config.lr_init_dict.functional_relationships 0.0003 \
  --model.init_args.auglag_config.lr_init_dict.noise_dist 0.0070 \
  --model.init_args.constraint_matrix_path /path/to/constraints.npy \
  --trainer.max_epochs 2000 \
  --trainer.fast_dev_run False \
  --trainer.accelerator auto \
  --trainer.devices 1 \
  --trainer.enable_checkpointing False \
  --trainer.callbacks '[{"class_path":"pytorch_lightning.callbacks.TQDMProgressBar","init_args":{"refresh_rate":19}}]'
```

Quick smoke/dev run (single-epoch):

```bash
.venv310/bin/python -m causica.lightning.main \
  --config src/causica/config/lightning/default_gaussian.yaml \
  --data src/causica/config/lightning/default_data.yaml \
  --data.init_args.dataset_path /path/to/datasets_root \
  --data.init_args.dataset_name tmp_causica \
  --data.init_args.batch_size 32 \
  --trainer.max_epochs 1 \
  --trainer.fast_dev_run True \
  --trainer.accelerator cpu
```

### How to extract and save the learned SEM / adjacency after training
Run a small Python snippet after training to save the most-likely SEM and adjacency matrix (this mirrors the notebook steps):

```python
import torch
import numpy as np

# If the notebook saved `deci.pt`, load that. Otherwise load from `outputs/` checkpoint as appropriate.
sem_module = torch.load("deci.pt")
sem = sem_module().mode
adj_matrix = sem.graph.cpu().numpy().astype(int)
np.savetxt("outputs/learned_adjacency_mode.csv", adj_matrix, fmt="%d", delimiter=',')
torch.save(sem_module, "outputs/deci_sem_module.pt")
```

### Quick checklist
- [ ] Prepare dataset folder (`/path/to/datasets_root/tmp_causica`) with `variables.json`, `train.csv`, `test.csv`, `val.csv`.
- [ ] (Optional) Build `constraints.npy` and place at `/path/to/constraints.npy`.
- [ ] Activate Python 3.10 venv (e.g., `source .venv310/bin/activate`) and ensure `causica` is installed (editable install recommended).
- [ ] Run the CLI command above, then run the extraction snippet to save `sem.graph` and evaluate ATE/ITE.

### Notes and troubleshooting
- If you change the number of variables in the dataset, remove or backup old checkpoints before resuming training (shape mismatches can break restore).
- For Apple MPS: set `PYTORCH_ENABLE_MPS_FALLBACK=1` if encountering unsupported operator errors (this falls back to CPU for those ops).
- For reproducibility set `pl.seed_everything(seed)` and save the SEM module plus the data normalizer.

