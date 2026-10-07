"""Scaffold splitting utilities using Bemis-Murcko frameworks."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


def murcko(smiles: str) -> Optional[str]:
    """Return the Bemis-Murcko generic scaffold SMILES, or None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        return Chem.MolToSmiles(generic)
    except Exception:
        return None


def scaffold_split(
    smiles: list[str],
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices by Bemis-Murcko scaffold.

    Molecules sharing a scaffold stay in the same fold.
    Large scaffold groups go to train; smaller groups are shuffled
    (controlled by *seed*) and assigned greedily until the test set
    reaches *test_frac*.

    Returns (train_indices, test_indices) as int arrays.
    """
    scaffolds: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        scaf = murcko(smi)
        key = scaf if scaf is not None else f"__no_scaffold_{i}"
        scaffolds[key].append(i)

    groups = list(scaffolds.values())
    groups.sort(key=len, reverse=True)

    n_total = len(smiles)
    n_test_target = int(n_total * test_frac)

    rng = np.random.RandomState(seed)
    big_cutoff = n_test_target
    big_groups = [g for g in groups if len(g) >= big_cutoff]
    small_groups = [g for g in groups if len(g) < big_cutoff]
    rng.shuffle(small_groups)

    train_idx: list[int] = []
    test_idx: list[int] = []

    for g in big_groups:
        train_idx.extend(g)

    for g in small_groups:
        if len(test_idx) < n_test_target:
            test_idx.extend(g)
        else:
            train_idx.extend(g)

    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)
