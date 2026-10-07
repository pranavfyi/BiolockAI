#!/usr/bin/env python
"""Fetch IC50 activity data from official ChEMBL sources.

Target discovery uses ChEMBL FTP dumps (chembl_uniprot_mapping.txt and
chembl_*.fa.gz), not the currently unreliable /target/search REST endpoint.

Activity download still uses the ChEMBL REST activity endpoint, with small
pages and retries, because that endpoint has been observed to work when
target-filter queries return HTTP 500.

Usage
-----
  python fetch_chembl.py select
  python fetch_chembl.py download CHEMBLxxxxxxxx --output data/raw.csv
  python fetch_chembl.py search "SARS-CoV-2 3C-like proteinase"
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
FTP_LATEST = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest"
FTP_README = f"{FTP_LATEST}/README"
MAPPING_NAME = "chembl_uniprot_mapping.txt"
FASTA_NAME = "chembl_37.fa.gz"

DEFAULT_OUT = Path("data/raw.csv")
DEFAULT_LOG = Path("data/target_selection.json")
CACHE_DIR = Path("data/chembl_cache")

SARS_COV2_ORGANISM = "Severe acute respiratory syndrome coronavirus 2"

POSITIVE_NAME_KEYWORDS = (
    "3c-like proteinase",
    "3c-like protease",
    "3clpro",
    "3cl-pro",
    "main protease",
    "mpro",
)
EXCLUDED_NAME_KEYWORDS = (
    "plpro",
    "papain-like",
    "polyprotein",
    "replicase",
    "complex",
)
SARS_COV1_MARKERS = (
    "sars-cov-1",
    "sars coronavirus (strain)",
    "human sars coronavirus",
    "severe acute respiratory syndrome coronavirus)",
)

ACTIVITY_FILTERS = {
    "standard_type": "IC50",
    "standard_units": "nM",
    "standard_relation": "=",
    "assay_type": "B",
}

SAFE_PAGE_SIZE = 20


class ChemblUnavailableError(RuntimeError):
    """Raised when ChEMBL cannot be queried reliably."""


class NoQualifyingTargetError(RuntimeError):
    """Raised when no target satisfies the project selection rules."""


def _get_json(url: str, params: dict | None = None, retries: int = 4, timeout: int = 60) -> dict:
    """GET a JSON resource with retries. HTTP 500 is treated as retryable."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code} for {r.url}", response=r)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning("Attempt %d failed (%s); retrying in %ds", attempt + 1, exc, wait)
            time.sleep(wait)
    raise ChemblUnavailableError(f"Failed to fetch {url} after {retries} attempts: {last_exc}")


def _download_file(url: str, dest: Path, retries: int = 4) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)
            logger.info("Cached official ChEMBL file: %s", dest)
            return dest
        except requests.RequestException as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning("Download attempt %d failed (%s); retrying in %ds", attempt + 1, exc, wait)
            time.sleep(wait)
    raise ChemblUnavailableError(f"Failed to download {url}: {last_exc}")


def detect_chembl_release(cache_dir: Path = CACHE_DIR) -> str:
    """Read the FTP README to record which ChEMBL release is being used."""
    readme_path = cache_dir / "README"
    if not readme_path.exists():
        _download_file(FTP_README, readme_path)
    text = readme_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Release:\s*(chembl_\d+)", text)
    if not match:
        raise ChemblUnavailableError("Could not parse ChEMBL release from official FTP README.")
    return match.group(1)


def fasta_filename_for_release(release: str) -> str:
    return f"{release}.fa.gz"


def _existing_data_file(name: str, cache_dir: Path) -> Path | None:
    for path in (cache_dir / name, Path("data") / name):
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def ensure_official_dumps(cache_dir: Path = CACHE_DIR) -> tuple[Path, Path, str]:
    """Download (if needed) the official UniProt mapping and target FASTA."""
    release = detect_chembl_release(cache_dir)
    mapping_name = MAPPING_NAME
    fasta_name = fasta_filename_for_release(release)
    mapping = _existing_data_file(mapping_name, cache_dir)
    fasta = _existing_data_file(fasta_name, cache_dir)
    if mapping is None:
        mapping = _download_file(f"{FTP_LATEST}/{mapping_name}", cache_dir / mapping_name)
    if fasta is None:
        fasta = _download_file(f"{FTP_LATEST}/{fasta_name}", cache_dir / fasta_name)
    return mapping, fasta, release


def _norm(text: str | None) -> str:
    return (text or "").strip().lower()


def is_sars_cov2_organism(organism: str | None) -> bool:
    org = _norm(organism)
    if not org:
        return False
    if SARS_COV2_ORGANISM.lower() in org:
        return True
    if "sars-cov-2" in org:
        return True
    return False


def looks_like_sars_cov1(organism: str | None) -> bool:
    org = _norm(organism)
    if is_sars_cov2_organism(org):
        return False
    return any(marker in org for marker in SARS_COV1_MARKERS) or (
        "sars" in org and "cov-2" not in org and "coronavirus 2" not in org
    )


def name_has_mpro_keyword(name: str | None) -> bool:
    n = _norm(name)
    return any(k in n for k in POSITIVE_NAME_KEYWORDS)


def name_has_excluded_keyword(name: str | None) -> bool:
    n = _norm(name)
    return any(k in n for k in EXCLUDED_NAME_KEYWORDS)


def evaluate_target(record: dict[str, Any]) -> dict[str, Any]:
    """Apply project target-selection rules. Does not invent IDs or activities."""
    reasons: list[str] = []
    organism = record.get("organism") or ""
    target_type = (record.get("target_type") or "").strip().upper()
    pref_name = record.get("pref_name") or ""

    if not is_sars_cov2_organism(organism):
        reasons.append(f"organism is not SARS-CoV-2 ({organism or 'missing'})")
    if looks_like_sars_cov1(organism):
        reasons.append("organism looks like SARS-CoV/SARS-CoV-1, not SARS-CoV-2")
    if target_type != "SINGLE PROTEIN":
        reasons.append(f"target_type is {target_type or 'missing'}, not SINGLE PROTEIN")
    if not name_has_mpro_keyword(pref_name):
        reasons.append(
            "pref_name does not contain 3C-like proteinase / main protease / Mpro"
        )
    if name_has_excluded_keyword(pref_name):
        reasons.append("pref_name contains an excluded term (PLpro/polyprotein/replicase/complex)")

    ok = not reasons
    return {
        **record,
        "qualifies": ok,
        "rejection_reasons": reasons,
    }


def parse_uniprot_mapping(path: Path) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            accession, chembl_id, pref_name, target_type = parts[0], parts[1], parts[2], parts[3]
            rec = targets.setdefault(
                chembl_id,
                {
                    "target_chembl_id": chembl_id,
                    "pref_name": pref_name,
                    "target_type": target_type,
                    "accessions": [],
                    "organism": None,
                    "fasta_header_name": None,
                    "sources": [],
                },
            )
            rec["pref_name"] = pref_name
            rec["target_type"] = target_type
            if accession and accession not in rec["accessions"]:
                rec["accessions"].append(accession)
            if "uniprot_mapping" not in rec["sources"]:
                rec["sources"].append("uniprot_mapping")
    return targets


def parse_target_fasta(path: Path, targets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Parse official target_dictionary FASTA headers.

    Header form:
      > CHEMBLid[,CHEMBLid...] [ACCESSION] Name (Organism)
    """
    header_re = re.compile(
        r"^>\s*(?P<ids>CHEMBL[0-9,CHEMBL]+)\s+\[(?P<acc>[^\]]+)\]\s+(?P<rest>.*)$"
    )
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            match = header_re.match(line.strip())
            if not match:
                continue
            ids = [x for x in match.group("ids").split(",") if x.startswith("CHEMBL")]
            acc = match.group("acc").strip()
            rest = match.group("rest").strip()
            organism = None
            name = rest
            if rest.endswith(")") and "(" in rest:
                name, organism = rest.rsplit("(", 1)
                name = name.strip()
                organism = organism[:-1].strip()
            for chembl_id in ids:
                rec = targets.setdefault(
                    chembl_id,
                    {
                        "target_chembl_id": chembl_id,
                        "pref_name": name,
                        "target_type": None,
                        "accessions": [],
                        "organism": organism,
                        "fasta_header_name": name,
                        "sources": [],
                    },
                )
                if not rec.get("pref_name"):
                    rec["pref_name"] = name
                rec["fasta_header_name"] = name
                rec["organism"] = organism or rec.get("organism")
                if acc and acc not in rec["accessions"]:
                    rec["accessions"].append(acc)
                if "target_fasta" not in rec["sources"]:
                    rec["sources"].append("target_fasta")
    return targets


def load_official_targets(cache_dir: Path = CACHE_DIR) -> tuple[dict[str, dict[str, Any]], str]:
    mapping, fasta, release = ensure_official_dumps(cache_dir)
    targets = parse_uniprot_mapping(mapping)
    targets = parse_target_fasta(fasta, targets)
    logger.info("Loaded %d ChEMBL target records from %s dumps", len(targets), release)
    return targets, release


def relevant_candidates(targets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return evaluated SARS-CoV-2 or 3C-like-named targets for the selection log."""
    rows = []
    for rec in targets.values():
        name = rec.get("pref_name") or ""
        fasta_name = rec.get("fasta_header_name") or ""
        organism = rec.get("organism") or ""
        interesting = (
            is_sars_cov2_organism(organism)
            or name_has_mpro_keyword(name)
            or name_has_mpro_keyword(fasta_name)
            or "sars" in _norm(organism)
            or "sars" in _norm(name)
        )
        if not interesting:
            continue
        rows.append(evaluate_target(rec))
    rows.sort(key=lambda r: (not r["qualifies"], r["target_chembl_id"]))
    return rows


def count_qualifying_ic50(target_id: str) -> int:
    """Count qualifying IC50 records. Uses limit=1 so only page_meta is needed."""
    params = {
        "target_chembl_id": target_id,
        **ACTIVITY_FILTERS,
        "limit": 1,
        "offset": 0,
    }
    data = _get_json(f"{BASE_URL}/activity.json", params)
    return int(data.get("page_meta", {}).get("total_count", 0))


def search_targets(query: str, limit: int = 20) -> list[dict]:
    """Offline search against official dumps. Does not call /target/search.json."""
    targets, _release = load_official_targets()
    q = _norm(query)
    tokens = [t for t in re.split(r"\s+", q) if t]
    hits = []
    for rec in targets.values():
        blob = " ".join(
            [
                rec.get("target_chembl_id") or "",
                rec.get("pref_name") or "",
                rec.get("fasta_header_name") or "",
                rec.get("organism") or "",
                rec.get("target_type") or "",
                " ".join(rec.get("accessions") or []),
            ]
        ).lower()
        if tokens and not all(tok in blob for tok in tokens):
            continue
        evaluated = evaluate_target(rec)
        hits.append(evaluated)
        if len(hits) >= limit:
            break
    return hits


def download_activities(
    target_id: str,
    standard_type: str = "IC50",
    units: str = "nM",
    relation: str = "=",
    assay_type: str = "B",
    limit_per_page: int = SAFE_PAGE_SIZE,
) -> list[dict]:
    """Download activity records with small pages to avoid HTTP 500s."""
    params = {
        "target_chembl_id": target_id,
        "standard_type": standard_type,
        "standard_units": units,
        "standard_relation": relation,
        "assay_type": assay_type,
        "limit": min(limit_per_page, SAFE_PAGE_SIZE),
        "offset": 0,
    }

    data = _get_json(f"{BASE_URL}/activity.json", params)
    total = int(data.get("page_meta", {}).get("total_count", 0))
    logger.info("Total records matching filters: %d", total)

    records: list[dict] = []
    seen = 0
    while True:
        activities = data.get("activities") or []
        for act in activities:
            seen += 1
            mol_id = act.get("molecule_chembl_id") or ""
            smiles = act.get("canonical_smiles") or ""
            value = act.get("standard_value")
            if mol_id and smiles and value not in (None, ""):
                records.append(
                    {
                        "molecule_id": mol_id,
                        "smiles": smiles,
                        "ic50_nM": value,
                    }
                )

        next_url = (data.get("page_meta") or {}).get("next")
        if not next_url:
            if params["offset"] + params["limit"] < total and activities:
                params["offset"] += params["limit"]
                data = _get_json(f"{BASE_URL}/activity.json", params)
                continue
            break

        logger.info("Fetched %d / %d records...", len(records), total)
        if str(next_url).startswith("http"):
            parsed = urlparse(next_url)
            if parsed.query:
                q = parse_qs(parsed.query)
                next_params = {k: v[0] for k, v in q.items()}
                if int(next_params.get("limit", SAFE_PAGE_SIZE)) > SAFE_PAGE_SIZE:
                    next_params["limit"] = str(SAFE_PAGE_SIZE)
                data = _get_json(f"{BASE_URL}/activity.json", next_params)
            else:
                data = _get_json(next_url)
        else:
            data = _get_json(f"https://www.ebi.ac.uk{next_url}")

    logger.info("Downloaded %d valid records (out of %d total)", len(records), total)
    return records


def save_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["molecule_id", "smiles", "ic50_nM"])
        writer.writeheader()
        writer.writerows(records)
    logger.info("Saved %d records to %s", len(records), path)


def write_selection_log(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote target-selection log to %s", path)


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_chembl_id": row.get("target_chembl_id"),
        "pref_name": row.get("pref_name"),
        "organism": row.get("organism"),
        "target_type": row.get("target_type"),
        "accessions": row.get("accessions") or [],
        "sources": row.get("sources") or [],
        "qualifies": bool(row.get("qualifies")),
        "rejection_reasons": row.get("rejection_reasons") or [],
        "qualifying_ic50_count": row.get("qualifying_ic50_count"),
        "ic50_count_error": row.get("ic50_count_error"),
    }


def select_mpro_target(
    cache_dir: Path = CACHE_DIR,
    count_ic50: bool = True,
) -> dict[str, Any]:
    """Identify a verified SARS-CoV-2 Mpro SINGLE PROTEIN target from official dumps."""
    targets, release = load_official_targets(cache_dir)
    inspected = relevant_candidates(targets)
    qualifying = [r for r in inspected if r["qualifies"]]

    rest_errors: list[str] = []
    if count_ic50:
        for row in qualifying:
            tid = row["target_chembl_id"]
            try:
                row["qualifying_ic50_count"] = count_qualifying_ic50(tid)
            except Exception as exc:
                row["qualifying_ic50_count"] = None
                row["ic50_count_error"] = str(exc)
                rest_errors.append(f"{tid}: {exc}")
                logger.warning("Could not count IC50 records for %s: %s", tid, exc)

    selected = None
    selection_error = None
    if not qualifying:
        selection_error = (
            "No ChEMBL target in the official "
            f"{release} UniProt mapping + target FASTA satisfies the project rules: "
            "SARS-CoV-2 organism, SINGLE PROTEIN, pref_name is Mpro/3C-like/main protease, "
            "and not PLpro/polyprotein/replicase/complex/SARS-CoV-1. "
            "Assay descriptions that mention 3C-like protease are not used as a substitute "
            "for target metadata. No target ID was assumed."
        )
    elif any(r.get("qualifying_ic50_count") is None for r in qualifying) and count_ic50:
        selection_error = (
            "One or more otherwise qualifying targets could not have IC50 records counted "
            "because the ChEMBL REST activity API failed. Refusing to guess among unverified counts."
        )
    else:
        qualifying.sort(
            key=lambda r: (-int(r.get("qualifying_ic50_count") or 0), r["target_chembl_id"])
        )
        selected = qualifying[0]
        if count_ic50 and int(selected.get("qualifying_ic50_count") or 0) == 0:
            selection_error = (
                f"{selected['target_chembl_id']} matches target metadata rules but has 0 "
                "qualifying IC50 records (IC50, nM, '=', assay_type B). Nothing was downloaded."
            )
            selected = None

    payload = {
        "chembl_release": release,
        "sources": {
            "ftp_latest": FTP_LATEST,
            "uniprot_mapping": MAPPING_NAME,
            "target_fasta": fasta_filename_for_release(release),
            "activity_endpoint": f"{BASE_URL}/activity.json",
            "activity_filters": ACTIVITY_FILTERS,
        },
        "selection_rules": {
            "organism": SARS_COV2_ORGANISM,
            "target_type": "SINGLE PROTEIN",
            "positive_name_keywords": list(POSITIVE_NAME_KEYWORDS),
            "excluded_name_keywords": list(EXCLUDED_NAME_KEYWORDS),
            "activity_filters": ACTIVITY_FILTERS,
            "tie_break": "largest qualifying IC50 count, then target_chembl_id",
        },
        "n_targets_in_dumps": len(targets),
        "n_relevant_inspected": len(inspected),
        "n_qualifying": len(qualifying),
        "inspected_relevant_targets": [_serialize_row(r) for r in inspected],
        "qualifying_targets": [_serialize_row(r) for r in qualifying],
        "selected_target": _serialize_row(selected) if selected else None,
        "rest_api_errors": rest_errors,
        "error": selection_error,
        "raw_csv_written": False,
        "records_downloaded": 0,
        "data_from_chembl": False,
    }
    return payload


def _print_inspected(rows: list[dict[str, Any]]) -> None:
    print("\nInspected SARS-CoV-2 / 3C-like related targets from official ChEMBL dumps")
    print("-" * 140)
    print(f"{'ChEMBL ID':<18} {'Type':<28} {'Qualifies':<10} {'Name / organism'}")
    for r in rows:
        flag = "YES" if r.get("qualifies") else "no"
        name = f"{r.get('pref_name') or '-'} | {r.get('organism') or '-'}"
        print(f"{r.get('target_chembl_id'):<18} {str(r.get('target_type') or '-'):<28} {flag:<10} {name}")
        if r.get("rejection_reasons"):
            for reason in r["rejection_reasons"]:
                print(f"{'':<18} reject: {reason}")


def verify_target_for_download(target_id: str, cache_dir: Path = CACHE_DIR) -> dict[str, Any]:
    targets, release = load_official_targets(cache_dir)
    rec = targets.get(target_id)
    if rec is None:
        raise NoQualifyingTargetError(
            f"{target_id} was not found in official {release} target dumps. "
            "Refusing to download an unverified target ID."
        )
    evaluated = evaluate_target(rec)
    evaluated["chembl_release"] = release
    if not evaluated["qualifies"]:
        reasons = "; ".join(evaluated["rejection_reasons"])
        raise NoQualifyingTargetError(
            f"{target_id} failed target-selection rules: {reasons}"
        )
    return evaluated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch IC50 data from official ChEMBL sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sp_search = sub.add_parser("search", help="Search official ChEMBL dumps (not /target/search).")
    sp_search.add_argument("query", help="Free-text target name or keyword.")
    sp_search.add_argument("--limit", type=int, default=20)

    sp_select = sub.add_parser("select", help="Select verified SARS-CoV-2 Mpro target and optionally download.")
    sp_select.add_argument("--output", type=Path, default=DEFAULT_OUT)
    sp_select.add_argument("--log", type=Path, default=DEFAULT_LOG)
    sp_select.add_argument("--no-download", action="store_true", help="Select only; do not write raw.csv.")
    sp_select.add_argument("--no-count", action="store_true", help="Skip REST IC50 counts (debug only).")

    sp_dl = sub.add_parser("download", help="Download IC50 records for a verified target.")
    sp_dl.add_argument("target_id", help="ChEMBL target ID.")
    sp_dl.add_argument("--output", type=Path, default=DEFAULT_OUT)
    sp_dl.add_argument("--log", type=Path, default=DEFAULT_LOG)
    sp_dl.add_argument("--standard-type", default="IC50")
    sp_dl.add_argument("--units", default="nM")
    sp_dl.add_argument("--relation", default="=")
    sp_dl.add_argument("--assay-type", default="B")
    sp_dl.add_argument(
        "--force-unverified",
        action="store_true",
        help="Unsafe: skip selection rules. Disabled by default and logged if used.",
    )

    args = parser.parse_args()

    if args.command == "search":
        try:
            results = search_targets(args.query, args.limit)
        except ChemblUnavailableError as exc:
            logger.error("%s", exc)
            sys.exit(2)
        if not results:
            print("No targets found in official ChEMBL dumps.")
            sys.exit(1)
        print(f"\n{'ChEMBL ID':<20} {'Name':<50} {'Organism':<45} {'Type':<20} Qualifies")
        print("-" * 150)
        for r in results:
            print(
                f"{r['target_chembl_id']:<20} {str(r.get('pref_name') or '-'):<50} "
                f"{str(r.get('organism') or '-'):<45} {str(r.get('target_type') or '-'):<20} "
                f"{'YES' if r['qualifies'] else 'no'}"
            )
        return

    if args.command == "select":
        try:
            payload = select_mpro_target(count_ic50=not args.no_count)
        except ChemblUnavailableError as exc:
            logger.error("%s", exc)
            sys.exit(2)

        _print_inspected(payload["inspected_relevant_targets"])
        write_selection_log(payload, args.log)

        if payload["error"]:
            logger.error("%s", payload["error"])
            print("\nSELECTED TARGET: none")
            print("raw.csv was not written. Existing files must not be treated as verified Mpro data.")
            sys.exit(1)

        selected = payload["selected_target"]
        print("\nSELECTED TARGET")
        print(f"  ID        : {selected['target_chembl_id']}")
        print(f"  Name      : {selected['pref_name']}")
        print(f"  Organism  : {selected['organism']}")
        print(f"  Type      : {selected['target_type']}")
        print(f"  UniProt   : {', '.join(selected['accessions']) or '-'}")
        print(f"  IC50 n=   : {selected['qualifying_ic50_count']}")
        print("  Why       : SARS-CoV-2 + SINGLE PROTEIN + Mpro/3C-like name + highest qualifying IC50 count")

        if args.no_download:
            sys.exit(0)

        records = download_activities(selected["target_chembl_id"])
        if not records:
            logger.error("No activity records downloaded. raw.csv was not written.")
            payload["error"] = "Activity download returned 0 usable molecule/SMILES/IC50 rows."
            write_selection_log(payload, args.log)
            sys.exit(1)
        save_csv(records, args.output)
        payload["raw_csv_written"] = True
        payload["records_downloaded"] = len(records)
        payload["data_from_chembl"] = True
        payload["raw_csv"] = str(args.output)
        write_selection_log(payload, args.log)
        return

    if args.command == "download":
        if args.force_unverified:
            logger.error(
                "--force-unverified is not supported: the project forbids downloading "
                "an unverified ChEMBL target."
            )
            sys.exit(2)
        try:
            evaluated = verify_target_for_download(args.target_id)
        except (ChemblUnavailableError, NoQualifyingTargetError) as exc:
            logger.error("%s", exc)
            sys.exit(1)

        logger.info(
            "Verified target %s | %s | %s | %s",
            evaluated["target_chembl_id"],
            evaluated.get("pref_name"),
            evaluated.get("organism"),
            evaluated.get("target_type"),
        )
        records = download_activities(
            args.target_id,
            standard_type=args.standard_type,
            units=args.units,
            relation=args.relation,
            assay_type=args.assay_type,
        )
        if not records:
            logger.error("No records found. Check target ID and filters.")
            sys.exit(1)
        save_csv(records, args.output)


if __name__ == "__main__":
    main()
