#!/usr/bin/env python3
"""Load a saved SEM (or Lightning checkpoint) and write adjacency CSVs to outputs/.

This script tries several common locations in order:
 - ./deci.pt
 - ./outputs/deci.pt
 - ./outputs/last.ckpt
 - ./outputs/best_model.ckpt

It will attempt to load a SEMDistributionModule directly, or load a DECIModule
Lightning checkpoint and extract its `sem_module` attribute. The most-likely
graph (mode) and the mean adjacency are saved as CSVs in the `outputs/`
directory. The SEM module object is also saved as `outputs/deci_sem_module.pt`.

Usage:
  .venv310/bin/python scripts/extract_sem_and_save_adj.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np


def find_candidate_paths() -> list[str]:
    cand = [
        "./deci.pt",
        "./outputs/deci.pt",
        "./outputs/last.ckpt",
        "./outputs/best_model.ckpt",
        "./outputs/last.ckpt.bak",
    ]
    return [p for p in cand if os.path.exists(p)]


def try_load_sem_module(path: str):
    """Try to load a SEM module directly via torch.load.

    Returns the loaded object on success, or None.
    """
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"torch.load failed for {path}: {e}")
        return None

    # Heuristics: if the object has a `mode` attribute callable after calling,
    # assume it's a SEMDistributionModule or similar.
    if hasattr(obj, "__call__") or hasattr(obj, "mode"):
        return obj

    return None


def try_load_lightning_checkpoint(path: str):
    """Attempt to load DECIModule from a Lightning checkpoint using the
    class in the repo. Returns the DECIModule instance or None.
    """
    try:
        # Import here to avoid heavy imports if not needed
        from causica.lightning.modules.deci_module import DECIModule

        module = DECIModule.load_from_checkpoint(path, map_location="cpu")
        return module
    except Exception as e:
        print(f"Loading DECIModule from checkpoint failed: {e}")
        return None


def extract_and_save(sem_module, out_dir: str = "outputs") -> None:
    os.makedirs(out_dir, exist_ok=True)

    # sem_module may be a SEMDistributionModule or a LightningModule with
    # attribute `sem_module`.
    if hasattr(sem_module, "sem_module"):
        sem_module = sem_module.sem_module

    # If sem_module is a callable factory, call it to get a SEMDistribution
    try:
        sem = sem_module() if callable(sem_module) else sem_module
    except Exception:
        sem = sem_module

    # Try to extract adjacency tensors
    adjacency_mode = None
    adjacency_mean = None

    # Common attribute names
    if hasattr(sem, "mode"):
        try:
            adjacency_mode = sem.mode.graph if hasattr(sem.mode, "graph") else getattr(sem.mode, "graph", None)
        except Exception:
            # sem.mode might be a Tensor or a callable
            try:
                m = sem.mode
                adjacency_mode = getattr(m, "graph", None)
            except Exception:
                adjacency_mode = None

    # Some SEMDistribution objects expose adjacency distribution on
    # sem_module.adjacency_module().mode or .mean
    if adjacency_mode is None and hasattr(sem_module, "adjacency_module"):
        try:
            adj_mod = sem_module.adjacency_module()
            adjacency_mode = getattr(adj_mod, "mode", None)
            adjacency_mean = getattr(adj_mod, "mean", None)
        except Exception:
            adjacency_mode = adjacency_mode or None

    # Fall back to attributes named `graph` or `adjacency`
    if adjacency_mode is None:
        if hasattr(sem, "graph"):
            adjacency_mode = sem.graph
        elif hasattr(sem, "adjacency"):
            adjacency_mode = sem.adjacency

    # adjacency_mean: try mean attribute on sem or sem_module
    if adjacency_mean is None:
        adjacency_mean = getattr(sem, "mean", None) or getattr(sem_module, "mean", None)

    # Convert to NumPy arrays if tensors
    def to_numpy(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        try:
            return x.cpu().numpy()
        except Exception:
            try:
                return np.array(x)
            except Exception:
                return None

    mode_np = to_numpy(adjacency_mode)
    mean_np = to_numpy(adjacency_mean)

    if mode_np is None and mean_np is None:
        print("Could not find adjacency tensors on the loaded object.")
        return

    if mode_np is not None:
        try:
            mode_path = os.path.join(out_dir, "learned_adjacency_mode.csv")
            np.savetxt(mode_path, mode_np.astype(int), fmt="%d", delimiter=",")
            print(f"Wrote adjacency mode to: {mode_path}")
        except Exception as e:
            print(f"Failed to save adjacency mode: {e}")

    if mean_np is not None:
        # mean_np might be a scalar or other non-2D value; only save if 1D/2D
        try:
            if getattr(mean_np, "ndim", None) is not None and mean_np.ndim >= 1:
                mean_path = os.path.join(out_dir, "learned_adjacency_mean.csv")
                np.savetxt(mean_path, mean_np, fmt="%.6f", delimiter=",")
                print(f"Wrote adjacency mean to: {mean_path}")
            else:
                print("Found adjacency mean but it is not 1D/2D; skipping save.")
        except Exception as e:
            print(f"Failed to save adjacency mean: {e}")

    # Save the sem_module object for later reuse
    try:
        torch.save(sem_module, os.path.join(out_dir, "deci_sem_module.pt"))
        print(f"Saved sem_module to: {os.path.join(out_dir, 'deci_sem_module.pt')}")
    except Exception as e:
        print(f"Failed to save sem_module object: {e}")


def main():
    candidates = find_candidate_paths()
    if not candidates:
        print("No candidate model files found (tried deci.pt and outputs/last.ckpt).")
        sys.exit(1)

    for p in candidates:
        print(f"Trying: {p}")
        sem_module = try_load_sem_module(p)
        if sem_module is not None:
            print(f"Loaded SEM module directly from: {p}")
            extract_and_save(sem_module)
            return

        # try loading as Lightning checkpoint
        module = try_load_lightning_checkpoint(p)
        if module is not None:
            print(f"Loaded DECIModule from checkpoint: {p}")
            extract_and_save(module)
            return

    print("Tried candidate paths but could not load a usable SEM/DECIModule.")
    sys.exit(2)


if __name__ == "__main__":
    main()
