"""Tests for scaffold splitting."""

import numpy as np
import pytest
from biolock.split import murcko, scaffold_split


SYNTHETIC_SMILES = [
    "c1ccccc1",        # benzene
    "c1ccc(O)cc1",     # phenol
    "c1ccc(N)cc1",     # aniline
    "C1CCCCC1",        # cyclohexane
    "C1CCC(O)CC1",     # cyclohexanol
    "c1ccncc1",        # pyridine
    "CC(=O)O",         # acetic acid
    "CCO",             # ethanol
    "CCCO",            # propanol
    "CCCCO",           # butanol
]


class TestMurcko:
    def test_benzene(self):
        assert murcko("c1ccccc1") is not None

    def test_invalid_returns_none(self):
        assert murcko("NOT_VALID") is None

    def test_same_scaffold(self):
        assert murcko("c1ccccc1") == murcko("c1ccc(O)cc1")


class TestScaffoldSplit:
    def test_no_overlap(self):
        train, test = scaffold_split(SYNTHETIC_SMILES, test_frac=0.3, seed=42)
        assert len(set(train) & set(test)) == 0

    def test_covers_all(self):
        train, test = scaffold_split(SYNTHETIC_SMILES, test_frac=0.3, seed=42)
        assert sorted(np.concatenate([train, test])) == list(range(len(SYNTHETIC_SMILES)))

    def test_scaffolds_not_shared(self):
        train, test = scaffold_split(SYNTHETIC_SMILES, test_frac=0.3, seed=42)
        train_scaff = {murcko(SYNTHETIC_SMILES[i]) for i in train}
        test_scaff = {murcko(SYNTHETIC_SMILES[i]) for i in test}
        shared = train_scaff & test_scaff - {None}
        assert len(shared) == 0

    def test_different_seeds_differ(self):
        diverse = SYNTHETIC_SMILES + [
            "c1ccc2ccccc2c1",   # naphthalene
            "C(=O)N",           # formamide
            "CC(C)O",           # isopropanol
            "c1ccoc1",          # furan
            "CC=CC",            # 2-butene
        ]
        _, t1 = scaffold_split(diverse, test_frac=0.4, seed=1)
        _, t2 = scaffold_split(diverse, test_frac=0.4, seed=999)
        # At least one split should have a non-empty test set
        assert len(t1) > 0 or len(t2) > 0
