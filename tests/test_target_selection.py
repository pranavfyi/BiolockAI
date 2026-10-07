"""Local tests for ChEMBL target-selection rules (no network, no fake experimental IC50)."""

from pathlib import Path

from fetch_chembl import (
    evaluate_target,
    is_sars_cov2_organism,
    name_has_mpro_keyword,
    parse_target_fasta,
    parse_uniprot_mapping,
    relevant_candidates,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_sars_cov2_organism_canonical():
    assert is_sars_cov2_organism("Severe acute respiratory syndrome coronavirus 2")
    assert is_sars_cov2_organism("SARS-CoV-2")
    assert not is_sars_cov2_organism("Human SARS coronavirus")
    assert not is_sars_cov2_organism("Infectious bronchitis virus")


def test_mpro_name_keywords():
    assert name_has_mpro_keyword("SARS-CoV-2 3C-like proteinase")
    assert name_has_mpro_keyword("main protease")
    assert not name_has_mpro_keyword("Replicase polyprotein 1ab")
    assert not name_has_mpro_keyword("Spike glycoprotein")


def test_polyprotein_is_rejected_even_if_sars_cov2_single_protein():
    result = evaluate_target(
        {
            "target_chembl_id": "EXAMPLE_NOT_A_REAL_DOWNLOAD",
            "pref_name": "Replicase polyprotein 1ab",
            "organism": "Severe acute respiratory syndrome coronavirus 2",
            "target_type": "SINGLE PROTEIN",
        }
    )
    assert result["qualifies"] is False
    assert any("excluded" in r for r in result["rejection_reasons"])


def test_sars_cov1_3cl_is_rejected():
    result = evaluate_target(
        {
            "target_chembl_id": "EXAMPLE_SARS1",
            "pref_name": "3C-like proteinase",
            "organism": "Human SARS coronavirus",
            "target_type": "SINGLE PROTEIN",
        }
    )
    assert result["qualifies"] is False


def test_qualifying_mpro_single_protein():
    result = evaluate_target(
        {
            "target_chembl_id": "EXAMPLE_MPRO",
            "pref_name": "SARS-CoV-2 3C-like proteinase",
            "organism": "Severe acute respiratory syndrome coronavirus 2",
            "target_type": "SINGLE PROTEIN",
        }
    )
    assert result["qualifies"] is True
    assert result["rejection_reasons"] == []


def test_official_dump_layout_finds_no_standalone_sars2_mpro():
    targets = parse_uniprot_mapping(FIXTURE_DIR / "chembl_uniprot_mapping.txt")
    targets = parse_target_fasta(FIXTURE_DIR / "chembl_targets.fa", targets)
    inspected = relevant_candidates(targets)
    by_id = {r["target_chembl_id"]: r for r in inspected}

    assert by_id["CHEMBL4523582"]["organism"] == (
        "Severe acute respiratory syndrome coronavirus 2"
    )
    assert by_id["CHEMBL4523582"]["target_type"] == "SINGLE PROTEIN"
    assert by_id["CHEMBL4523582"]["qualifies"] is False
    assert by_id["CHEMBL3927"]["qualifies"] is False
    assert by_id["CHEMBL1293307"]["qualifies"] is False
    assert by_id["CHEMBL4662936"]["qualifies"] is False
    assert not any(r["qualifies"] for r in inspected)
