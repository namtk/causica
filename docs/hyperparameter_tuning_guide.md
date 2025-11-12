# Hyperparameter tuning guide — what to tune and how (Causica)

This guide summarizes the configuration parameters that most affect training and evaluation in this repository, how they interact with different kinds of input data, recommended starting ranges, and practical tuning strategies using the CLI.

---

## Summary — quick checklist

- Always standardize or check data scale before interpreting likelihood/ATE metrics.
- Start with small runs (fast_dev_run or 1 epoch) to validate wiring and shapes.
- Tune in this order: data preprocessing → learning rate & optimizer → batch size → graph prior (sparsity) → DAG penalty (auglag) → Gumbel / adjacency temps → model capacity (embedding size/layers) → longer runs and seeds.
- Use multiple seeds (3–5) for final evaluation; average metrics.

---

## 1) Data & preprocessing parameters

Why they matter
- Model likelihoods (test_LL) and causal-effect metrics (ATE/ITE RMSE) scale directly with variable magnitude/variance. Heterogeneous variable scales can destabilize training.
- Data splits, batch size, and drop_last affect how many training updates you actually get.

Parameters to tune / check
- `dataset` generation params (if generating synthetic data)
  - `num_samples_train`, `num_samples_test`, `num_variables`, `num_edges`, noise / coefficient scale
  - When data are small (<~1k samples) regularization and early stopping become important.
- DataModule init args (`--data.init_args.*`)
  - `batch_size` (important for gradient noise; see tuning below)
  - `standardize` / normalization flags (if available) or run standalone standardization script
  - `drop_last` (if True and batch_size exceeds dataset size you may get 0 training batches)

Practical advice
- Always compute per-variable mean/std on training set and either standardize or save the scaling.
- For small datasets, reduce batch_size to have multiple gradient updates per epoch (e.g., `batch_size` = 8–64 depending on dataset size).

CLI examples
```bash
# set batch size
--data.init_args.batch_size 32
# point to a standardized dataset
--data.init_args.dataset_path . --data.init_args.dataset_name tmp_causica
```

---

## 2) Optimizer & learning rate

Why tune
- Learning rate controls convergence speed. Too large → divergence; too small → slow learning (poor adjacency/orientation after limited epochs).

Parameters
- `auglag_config.lr_init_dict` and `optim` settings in `DECIModule.configure_optimizers` (module-specific LR groups are used for different submodules: functional relationships, adjacency module (vardist), noise module).
- Global defaults typically use Adam.

Recommended ranges
- Adam base LR: 1e-4 — 5e-3. Start at 1e-3 to 3e-4.
- If separate LR per module present, set `vardist` LR slightly lower than `functional_relationships` if graphs are sensitive (e.g., 0.5–1× base LR).

Tuning strategy
- Run short runs (1–3 epochs) and monitor training loss. If loss diverges, reduce LR by 5×.
- Optionally use LR warmup or scheduler (see AugLagLR callback for auglag components).

CLI example
```bash
# override trainer / hparams via config or CLI if exposed
--trainer.max_epochs 10 --seed 42
# per-module lr typically configured in YAML; update config file or override hparams if supported
```

---

## 3) Batch size

Why tune
- Affects gradient noise and effective learning rate.

Guidance
- Small datasets: use small batch_size (8–64) so that each epoch gives many updates.
- Large datasets: use larger batch_size (64–512) if memory allows. If using larger batch sizes, consider increasing LR proportionally (linear-scaling rule) but verify with experiments.

Caveat
- The DataModule may use `drop_last=True`. Ensure batch_size <= dataset size or you will lose last partial batch.

CLI example
```bash
--data.init_args.batch_size 128
```

---

## 4) Model capacity & architecture

Why tune
- Too small capacity → underfitting (poor adjacency recovery and likelihood). Too large → overfitting and slower training.

Key params in `DECIModule`
- `embedding_size` (default ~32) — size of node embeddings.
- `out_dim_g` — size of functional relationship outputs.
- `num_layers_g` / `num_layers_zeta` — depth of encoder/decoder networks.

Recommended ranges
- `embedding_size`: 8–128. Start at 32.
- `out_dim_g`: 16–128 depending on data complexity.
- `num_layers_g`: 1–4 (2 is a reasonable default).

Tuning strategy
- Start small for debugging. Increase capacity if both train and val losses are high.
- Monitor overfitting (train LL much better than val LL) and reduce capacity or increase regularization if needed.

CLI example
```bash
--hparams.embedding_size 64 --hparams.num_layers_g 3
# or change via YAML config
```

---

## 5) Graph prior & sparsity

Why tune
- The prior sparsity lambda balances likelihood vs graph complexity. Higher lambda encourages sparser graphs.

Params
- `prior_sparsity_lambda` (default ~0.05 in DECIModule constructor). Higher values → sparser graphs.

Recommended ranges
- 0.0 (no sparsity) to 0.5. Try: 0.0, 0.01, 0.05, 0.1, 0.2.

Tuning strategy
- If predicted graphs are too dense relative to truth, increase lambda. If predicted graphs miss true edges (very sparse), lower lambda.
- Use adjacency F1 as primary signal for this hyperparameter.

CLI example
```bash
--hparams.prior_sparsity_lambda 0.05
```

---

## 6) DAG constraint (Augmented Lagrangian)

Why tune
- AugLag balances enforcing acyclicity vs likelihood. Incorrect settings lead to poor DAGness or slow convergence.

Params
- `auglag_config` (scheduler), `init_alpha`, `init_rho`, `disable_auglag_epochs` in `DECIModule`.

Guidance
- `init_rho` and `init_alpha`: typical starting values from repo: `init_alpha=0.0`, `init_rho=1.0`.
- `disable_auglag_epochs`: useful to let the model learn functions before enforcing DAG strongly.

Tuning strategy
- If the learned graphs are not DAGs or have many cycles, increase enforcement (increase rho schedule). If enforcement prevents learning useful structure, delay enforcement (set `disable_auglag_epochs` to a few epochs).

CLI example
```bash
--hparams.init_rho 1.0 --hparams.disable_auglag_epochs 2
```

---

## 7) Gumbel / adjacency sampling temperature

Why tune
- `gumbel_temp` controls relaxation temperature for relaxed sampling (training). Low temperature → near-discrete samples; high temp → smoother gradients.

Tuning
- Typical default: `gumbel_temp = 0.25`.
- Consider annealing temperature: start higher (0.5) and reduce to 0.1–0.01 across training for more discrete solutions.

CLI example
```bash
--hparams.gumbel_temp 0.25
```

---

## 8) NUM_GRAPH_SAMPLES (evaluation)

Why tune
- More graph samples → more stable metric estimates (adjacency/orientation F1 average) but slower evaluation.

Recommendation
- Default `NUM_GRAPH_SAMPLES = 100` in code. Use 100 for accurate eval; for fast checks use 10–20.

---

## 9) Checkpointing & resume

- If you resume from `outputs/last.ckpt`, ensure dataset shape and model have compatible variable sizes (different num_variables will cause load_state_dict size mismatches).
- Keep backups of old outputs when changing dataset shapes (`mv outputs outputs.bak`).

---

## 10) Device / performance and MPS caveats

- If using MPS, some operators may not be implemented and will fall back to CPU unless you set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- MPS fallback affects runtime performance but should not change correctness in most cases.

---

## 11) Practical tuning workflow (step-by-step)

1. Data sanity checks
   - Inspect `variables.json` and `train.csv` scales. Standardize if needed.
2. Smoke test
   - Run `--trainer.fast_dev_run True` to validate model loads and shapes.
3. Short runs for hyperparam checks
   - Run 1–3 epoch runs to test learning rate and batch size. Monitor train/val loss for divergence.
4. Sweep graph-prior sparsity
   - Run multiple short runs with different `prior_sparsity_lambda` and monitor `eval/adjacency.f1`.
5. Tune DAG penalty
   - If graphs are cyclic or overly penalized, adjust `init_rho` or `disable_auglag_epochs`.
6. Capacity tuning
   - Increase `embedding_size` and `num_layers_g` if model underfits; decrease or regularize if overfitting.
7. Final runs
   - Run 10–50 epochs with 3 seeds and average metrics.

Automation
- Use a sweep tool (Weights & Biases, MLflow, or a simple bash loop) and vary one axis at a time. For each experiment, save a copy of the exact CLI command in `docs/terminal_command_history.txt`.

Example bash loop (sparsity sweep):
```bash
for s in 0.0 0.01 0.05 0.1; do
  .venv310/bin/python -m causica.lightning.main \
    --config src/causica/config/lightning/default_gaussian.yaml \
    --data src/causica/config/lightning/default_data.yaml \
    --data.init_args.dataset_path . \
    --data.init_args.dataset_name tmp_causica \
    --trainer.max_epochs 5 \
    --hparams.prior_sparsity_lambda ${s} \
    --seed 42
done
```

---

## 12) Metrics to monitor and stopping rules

- Primary: `eval/adjacency.f1`, `eval/orientation.f1` (graph recovery).
- Secondary: `eval/test_LL` (likelihood), `eval/Interventional_LL`, `eval/ATE_RMSE`.
- Stopping: early stop on validation `adjacency.f1` or validation `test_LL` if modeling the distribution.

---

## 13) Troubleshooting common issues

- Zero training batches: lower `batch_size` or set `drop_last=False` in DataModule.
- Checkpoint size mismatch: remove/backup `outputs` before changing dataset shape.
- MPS NotImplementedError: set `PYTORCH_ENABLE_MPS_FALLBACK=1` or run on CPU.
- Very negative LL / large ATE_RMSE: check and standardize data scales.

---

## 14) Quick reference — common CLI overrides

- Data pointers & batch size:
```bash
--data.init_args.dataset_path . --data.init_args.dataset_name tmp_causica --data.init_args.batch_size 64
```
- Trainer / device / epochs:
```bash
--trainer.max_epochs 10 --trainer.accelerator mps --trainer.devices 1
```
- Hparams exposed on DECIModule (examples):
```bash
--hparams.embedding_size 64 --hparams.prior_sparsity_lambda 0.05 --hparams.gumbel_temp 0.25
```
- Use seed for reproducibility:
```bash
--seed 42
```


