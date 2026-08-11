"""Fill in missing building-block names by querying PubChem.

The sink file names 808 of its 809 structured rows with a bare MetaNetX
identifier, and joining against `chem_prop.tsv` only recovers a chemical name for
about 36% of them. The remainder reach the agent as "MNXM1043", which carries no
information a model can reason with — the point of naming them at all.

This script closes the gap by looking the rest up in PubChem by InChIKey, and
writes a cache next to the sink:

    data/building_blocks/names.tsv     (smiles, label, source)

`core.building_blocks` reads that cache when present. Re-running is cheap: rows
already resolved are skipped, so an interrupted run resumes.

    python -m morganbiopilot.data_processing.name_building_blocks
    python -m morganbiopilot.data_processing.name_building_blocks --limit 20 --dry-run

Network etiquette: PubChem asks for at most 5 requests per second, so this sleeps
between calls and will take a few minutes over the whole sink. It queries one
public REST endpoint and downloads no files.
"""

import argparse
import csv
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Optional

from rdkit import Chem, RDLogger

from morganbiopilot.core.building_blocks import building_block_entries
from morganbiopilot.core.paths import BUILDING_BLOCKS_DIR

RDLogger.DisableLog("rdApp.*")

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
NAMES_TSV = BUILDING_BLOCKS_DIR / "names.tsv"
REQUESTS_PER_SECOND = 5


def looks_unnamed(label: str) -> bool:
    """True when the label is an identifier rather than a chemical name."""
    return label.startswith("MNXM") and label[4:].isdigit()


def pubchem_name(inchikey: str, timeout: float = 15.0) -> Optional[str]:
    """Preferred IUPAC-ish title for an InChIKey, or None if PubChem has none."""
    url = f"{PUBCHEM}/compound/inchikey/{urllib.parse.quote(inchikey)}/property/Title/TXT"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            title = response.read().decode("utf-8", "replace").strip().split("\n")[0]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:      # not in PubChem — a normal outcome, not a failure
            return None
        raise
    except (urllib.error.URLError, OSError):
        return None
    return title or None


def load_cache() -> Dict[str, tuple]:
    if not NAMES_TSV.exists():
        return {}
    with open(NAMES_TSV, encoding="utf-8", newline="") as fh:
        return {row["smiles"]: (row["label"], row["source"])
                for row in csv.DictReader(fh, delimiter="\t")}


def save_cache(cache: Dict[str, tuple]) -> None:
    NAMES_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(NAMES_TSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["smiles", "label", "source"])
        for smi, (label, source) in sorted(cache.items()):
            writer.writerow([smi, label, source])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=0, help="stop after N lookups")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be queried, hit no network")
    args = parser.parse_args()

    entries = building_block_entries()
    cache = load_cache()
    todo = [(smi, label) for smi, label in entries
            if looks_unnamed(label) and smi not in cache]

    named = sum(1 for _, label in entries if not looks_unnamed(label))
    print(f"sink: {len(entries)} metabolites | named from MetaNetX: {named} "
          f"({100*named/len(entries):.0f}%) | cached: {len(cache)} | to query: {len(todo)}")

    if args.dry_run:
        for smi, label in todo[:10]:
            print(f"  would query {label:14s} {smi[:60]}")
        return 0

    delay = 1.0 / REQUESTS_PER_SECOND
    found = failed = 0
    for i, (smi, label) in enumerate(todo, start=1):
        if args.limit and i > args.limit:
            break
        mol = Chem.MolFromSmiles(smi)
        key = Chem.MolToInchiKey(mol) if mol is not None else None
        if not key:
            failed += 1
            continue

        try:
            name = pubchem_name(key)
        except Exception as exc:
            print(f"  aborting at {label}: {exc}", file=sys.stderr)
            break

        if name:
            cache[smi] = (name, "pubchem")
            found += 1
        else:
            # Record the miss too, so a re-run does not query it again.
            cache[smi] = (label, "unresolved")
            failed += 1

        if i % 25 == 0:
            save_cache(cache)
            print(f"  {i}/{len(todo)} queried | resolved {found} | unresolved {failed}")
        time.sleep(delay)

    save_cache(cache)
    resolved = sum(1 for label, source in cache.values() if source == "pubchem")
    total_named = named + resolved
    print(f"\nwrote {NAMES_TSV}")
    print(f"  resolved this run : {found} | unresolved: {failed}")
    print(f"  sink now named    : {total_named}/{len(entries)} "
          f"({100*total_named/len(entries):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
