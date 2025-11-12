#!/usr/bin/env python3
"""Build a constraint matrix (NumPy .npy) from dataset metadata using the notebook's rules.

This script implements the rules shown in
`examples/multi_investment_sales_attribution.ipynb`:

- Revenue cannot cause other nodes (except Planning Summit)
- Certain attributes have no parents (Commercial Flag, Major Flag, SMC Flag,
  PC Count, Employee Count, Global Flag, Size)
- Engagement nodes (Tech Support, Discount, New Engagement Strategy) do not
  cause each other

Usage:
  ./scripts/build_constraints.py --dataset-root ./tmp_causica --output ./tmp_causica/constraints.npy

The script is robust to a couple of different `variables.json` formats. If no
variables file can be parsed, it will fall back to reading the header from
`train.csv` in the dataset root to obtain the variable ordering.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import numpy as np


def load_variable_names_from_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Common formats:
    # - {"variables": [ {"name": "X"}, ... ] }
    # - [ {"name": "X"}, ... ]
    # - ["X", "Y", ...]
    if isinstance(payload, dict) and "variables" in payload:
        items = payload["variables"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("Unrecognized variables.json structure")

    names: List[str] = []
    for it in items:
        if isinstance(it, str):
            names.append(it)
        elif isinstance(it, dict):
            # try several common keys
            for k in ("name", "variable", "variable_name", "column_name", "id"):
                if k in it:
                    names.append(it[k])
                    break
            else:
                # fall back to stringifying the dict (should not happen)
                names.append(json.dumps(it))
        else:
            names.append(str(it))

    return names


def load_variable_names(dataset_root: str) -> List[str]:
    # Try variables.json then train.csv header
    cand = [
        os.path.join(dataset_root, "variables.json"),
        os.path.join(dataset_root, "_variables.json"),
    ]
    for p in cand:
        if os.path.exists(p):
            try:
                return load_variable_names_from_json(p)
            except Exception:
                # try next fallback
                pass

    # Fallback: read header of train.csv
    train_csv = os.path.join(dataset_root, "train.csv")
    if os.path.exists(train_csv):
        # read only header to avoid heavy IO dependencies
        import csv

        with open(train_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                return header

    raise FileNotFoundError(
        f"Could not determine variable names from {dataset_root}; place a variables.json or train.csv in the dataset root."
    )


def build_constraints(names: List[str]) -> np.ndarray:
    num_nodes = len(names)
    cm = np.full((num_nodes, num_nodes), np.nan, dtype=np.float32)

    name_to_idx = {n: i for i, n in enumerate(names)}

    def idx(name: str) -> Optional[int]:
        return name_to_idx.get(name)

    # Rule 1: Revenue cannot be a cause of other nodes except maybe Planning Summit
    revenue_idx = idx("Revenue")
    planning_idx = idx("Planning Summit")
    if revenue_idx is not None:
        cm[revenue_idx, :] = 0.0
        if planning_idx is not None:
            cm[revenue_idx, planning_idx] = np.nan

    # Rule 2: certain attributes have no parents (columns set to 0.0)
    non_child_nodes = [
        "Commercial Flag",
        "Major Flag",
        "SMC Flag",
        "PC Count",
        "Employee Count",
        "Global Flag",
        "Size",
    ]
    non_child_idxs = [i for n, i in name_to_idx.items() if n in non_child_nodes]
    for c in non_child_idxs:
        cm[:, c] = 0.0

    # Rule 3: engagement nodes do not cause each other
    engagement_nodes = ["Tech Support", "Discount", "New Engagement Strategy"]
    engagement_idxs = [idx(n) for n in engagement_nodes if idx(n) is not None]
    if engagement_idxs:
        # set edges among engagement nodes to 0 (no causal edges between them)
        for j in engagement_idxs:
            cm[engagement_idxs, j] = 0.0

    return cm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build constraints.npy for Causica DECI from dataset variables and canned rules."
    )
    parser.add_argument("--dataset-root", "-d", default="./tmp_causica", help="Path to dataset folder (contains variables.json or train.csv)")
    parser.add_argument("--output", "-o", default=None, help="Output .npy file path (defaults to <dataset_root>/constraints.npy)")

    args = parser.parse_args()

    dataset_root = args.dataset_root
    out = args.output or os.path.join(dataset_root, "constraints.npy")

    names = load_variable_names(dataset_root)
    cm = build_constraints(names)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, cm)

    print(f"Wrote constraint matrix to: {out}")
    print(f"Matrix shape: {cm.shape}")


if __name__ == "__main__":
    main()
