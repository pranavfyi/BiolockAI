"""Chemistry utilities for SMILES handling, molecular properties, and fingerprints."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)


def standardize(smiles: str) -> Optional[str]:
    """Parse → cleanup → largest fragment → neutralise → canonical SMILES.

    Returns None on failure.
    """
    if not smiles or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        canon = Chem.MolToSmiles(mol)
        if Chem.MolFromSmiles(canon) is None:
            return None
        return canon
    except Exception:
        logger.debug("standardize failed for %s", smiles, exc_info=True)
        return None


def properties(smiles: str) -> dict[str, float]:
    """Compute MW, LogP, HBD, HBA, RotBonds, TPSA from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    return {
        "MW": round(Descriptors.ExactMolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "TPSA": round(Descriptors.TPSA(mol), 2),
    }


def lipinski_violations(props: dict[str, float]) -> int:
    """Count Lipinski Ro5 violations: MW≤500, LogP≤5, HBD≤5, HBA≤10."""
    return sum([
        props["MW"] > 500,
        props["LogP"] > 5,
        props["HBD"] > 5,
        props["HBA"] > 10,
    ])


def passes_lipinski(props: dict[str, float], max_violations: int = 0) -> bool:
    """True if the molecule has ≤ *max_violations* Lipinski violations."""
    return lipinski_violations(props) <= max_violations


def morgan_matrix(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> NDArray[np.uint8]:
    """Return an (N, n_bits) uint8 Morgan-fingerprint matrix."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Cannot parse SMILES: {smi}")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.uint8)
        for bit in fp.GetOnBits():
            arr[bit] = 1
        fps.append(arr)
    return np.vstack(fps)
