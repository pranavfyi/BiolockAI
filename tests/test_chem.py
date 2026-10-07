"""Unit tests for biolock.chem – uses only synthetic fixtures."""

import pytest
import numpy as np

from biolock.chem import (
    lipinski_violations,
    morgan_matrix,
    passes_lipinski,
    properties,
    standardize,
)


# ---------------------------------------------------------------------------
# standardize()
# ---------------------------------------------------------------------------

class TestStandardize:
    """Tests for SMILES standardisation."""

    def test_valid_smiles(self):
        """Canonical aspirin SMILES is returned."""
        result = standardize("CC(=O)Oc1ccccc1C(O)=O")
        assert result is not None
        assert isinstance(result, str)

    def test_invalid_smiles_returns_none(self):
        assert standardize("NOT_A_SMILES") is None
        assert standardize("") is None

    def test_salt_stripped(self):
        """Counter-ions are removed (e.g. sodium salt of benzoic acid)."""
        # Sodium benzoate -> benzoic acid
        result = standardize("[Na+].OC(=O)c1ccccc1")
        assert result is not None
        # Result should NOT contain Na
        assert "Na" not in result
        # Should be benzoic acid
        ref = standardize("OC(=O)c1ccccc1")
        assert result == ref

    def test_neutralised(self):
        """Charged forms are neutralised."""
        # Protonated amine
        result = standardize("[NH3+]CCCC")
        assert result is not None
        # Should match the neutral form
        neutral = standardize("NCCCC")
        assert result == neutral

    def test_canonical_is_consistent(self):
        """Different representations of the same molecule give the same output."""
        smi1 = standardize("c1ccccc1")
        smi2 = standardize("C1=CC=CC=C1")
        assert smi1 is not None
        assert smi1 == smi2


# ---------------------------------------------------------------------------
# properties()
# ---------------------------------------------------------------------------

class TestProperties:
    """Tests for molecular property computation."""

    def test_aspirin_properties(self):
        """Spot-check aspirin properties against known values."""
        props = properties("CC(=O)Oc1ccccc1C(O)=O")  # aspirin
        assert 170 < props["MW"] < 190  # ~180.04 (ExactMolWt)
        assert 0 < props["LogP"] < 2.5
        assert props["HBD"] == 1
        assert props["HBA"] in (3, 4)  # RDKit version-dependent
        assert isinstance(props["RotBonds"], int)
        assert isinstance(props["TPSA"], float)

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError):
            properties("INVALID_SMILES")

    def test_all_keys_present(self):
        props = properties("CCCCCC")  # hexane
        expected_keys = {"MW", "LogP", "HBD", "HBA", "RotBonds", "TPSA"}
        assert set(props.keys()) == expected_keys


# ---------------------------------------------------------------------------
# lipinski_violations() / passes_lipinski()
# ---------------------------------------------------------------------------

class TestLipinski:
    """Tests for Lipinski filter."""

    def test_no_violations(self):
        """Small drug-like molecule has zero violations."""
        props = properties("CC(=O)Oc1ccccc1C(O)=O")  # aspirin
        assert lipinski_violations(props) == 0
        assert passes_lipinski(props, max_violations=0) is True

    def test_mw_violation(self):
        """Synthetic heavy molecule dict exceeds MW > 500."""
        fake_props = {"MW": 600, "LogP": 2, "HBD": 1, "HBA": 3}
        assert lipinski_violations(fake_props) == 1
        assert passes_lipinski(fake_props, max_violations=0) is False
        assert passes_lipinski(fake_props, max_violations=1) is True

    def test_all_violations(self):
        fake_props = {"MW": 600, "LogP": 6, "HBD": 6, "HBA": 11}
        assert lipinski_violations(fake_props) == 4

    def test_boundary_values(self):
        """Values exactly at the threshold are NOT violations."""
        boundary = {"MW": 500, "LogP": 5, "HBD": 5, "HBA": 10}
        assert lipinski_violations(boundary) == 0


# ---------------------------------------------------------------------------
# morgan_matrix()
# ---------------------------------------------------------------------------

class TestMorganMatrix:
    """Tests for Morgan fingerprint generation."""

    def test_shape(self):
        smiles = ["CCCCCC", "c1ccccc1", "CC(=O)Oc1ccccc1C(O)=O"]
        X = morgan_matrix(smiles)
        assert X.shape == (3, 2048)
        assert X.dtype == np.uint8

    def test_binary_values(self):
        X = morgan_matrix(["CCCCCC"])
        assert set(np.unique(X)).issubset({0, 1})

    def test_different_molecules_differ(self):
        X = morgan_matrix(["CCCCCC", "c1ccccc1"])
        assert not np.array_equal(X[0], X[1])

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError):
            morgan_matrix(["INVALID"])

    def test_custom_bits(self):
        X = morgan_matrix(["CCCCCC"], n_bits=1024)
        assert X.shape == (1, 1024)
