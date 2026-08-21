"""Post-generation sanity checks for a built C2M2 datapackage.

Complements `cfde-c2m2 validate` (schema + frictionless referential checks) with
data-level checks that matter for this submission: every file has a checksum, no
private crosswalk/id-map leaked into the package, referential integrity across the
tables, unique primary keys, and a Synapse-id leak scan (the disease firewall).

Usage:
    python verify_c2m2.py --package /path/to/c2m2_output [--work /path/with/crosswalks]

Exits non-zero if any HARD check fails; prints WARN/INFO for the rest.
"""
import os
import re
import glob
import argparse

import pandas as pd

CORE_TABLES = ["subject", "biosample", "file", "file_describes_biosample"]
SYNID_RE = re.compile(r"syn\d{6,}")


def _read(pkg, name):
    """Read a C2M2 TSV as all-strings, empty cells as '' (not NaN), or None if absent."""
    path = os.path.join(pkg, f"{name}.tsv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _keys(df, ns_col, id_col):
    return set(zip(df[ns_col], df[id_col]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", required=True, help="C2M2 output dir (the .tsv tables)")
    ap.add_argument("--work", default=None, help="dir holding the private id_map/crosswalk CSVs")
    args = ap.parse_args()
    pkg = args.package

    failures, warnings = [], []

    def hard(ok, msg):
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            failures.append(msg)

    def warn(cond, msg):
        if cond:
            print(f"  [WARN] {msg}")
            warnings.append(msg)

    tables = {t: _read(pkg, t) for t in CORE_TABLES}

    print("== core tables present + non-empty ==")
    for t in CORE_TABLES:
        df = tables[t]
        hard(df is not None and len(df) > 0,
             f"{t}.tsv present and non-empty ({0 if df is None else len(df)} rows)")
    if any(tables[t] is None for t in CORE_TABLES):
        print("\nmissing core tables; aborting further checks")
        raise SystemExit(1)

    print("== row counts ==")
    for f in sorted(glob.glob(os.path.join(pkg, "*.tsv"))):
        n = sum(1 for _ in open(f)) - 1
        print(f"  {os.path.basename(f):34} {max(n, 0)}")

    print("== every file has a checksum (sha256 or md5) ==")
    fdf = tables["file"]
    sha = fdf["sha256"] if "sha256" in fdf.columns else pd.Series([""] * len(fdf))
    md5 = fdf["md5"] if "md5" in fdf.columns else pd.Series([""] * len(fdf))
    missing = ((sha.fillna("") == "") & (md5.fillna("") == "")).sum()
    hard(missing == 0, f"file rows missing BOTH sha256 and md5: {missing}")

    print("== primary keys unique (id_namespace, local_id) ==")
    for t in ["subject", "biosample", "file"]:
        df = tables[t]
        if {"id_namespace", "local_id"} <= set(df.columns):
            dups = len(df) - len(df.drop_duplicates(["id_namespace", "local_id"]))
            hard(dups == 0, f"{t}.tsv duplicate primary keys: {dups}")

    print("== referential integrity ==")
    file_keys = _keys(fdf, "id_namespace", "local_id")
    bio_keys = _keys(tables["biosample"], "id_namespace", "local_id")
    fdb = tables["file_describes_biosample"]
    orphan_files = _keys(fdb, "file_id_namespace", "file_local_id") - file_keys
    orphan_bios = _keys(fdb, "biosample_id_namespace", "biosample_local_id") - bio_keys
    hard(not orphan_files, f"file_describes_biosample -> file orphans: {len(orphan_files)}")
    hard(not orphan_bios, f"file_describes_biosample -> biosample orphans: {len(orphan_bios)}")
    bd = _read(pkg, "biosample_disease")
    if bd is not None and len(bd):
        orphan_bd = _keys(bd, "biosample_id_namespace", "biosample_local_id") - bio_keys
        hard(not orphan_bd, f"biosample_disease -> biosample orphans: {len(orphan_bd)}")

    print("== disease firewall: no private crosswalk/id-map inside the package ==")
    leaked = [os.path.basename(p) for p in glob.glob(os.path.join(pkg, "*"))
              if re.search(r"crosswalk|id_map", os.path.basename(p), re.I)]
    hard(not leaked, f"crosswalk/id_map files inside package: {leaked}")

    print("== Synapse-id leak scan (warn) ==")
    syn_hits = {}
    for f in glob.glob(os.path.join(pkg, "*.tsv")):
        with open(f) as fh:
            c = sum(len(SYNID_RE.findall(line)) for line in fh)
        if c:
            syn_hits[os.path.basename(f)] = c
    warn(bool(syn_hits), f"Synapse ids appear in package tables: {syn_hits} "
                         "(controlled raw-audio rows must NOT carry synIds)")

    print("== leading zeros preserved in subject local_ids (eyeball) ==")
    print("  sample:", list(tables["subject"]["local_id"].head(5)))

    print(f"\n=== {'PASS' if not failures else 'FAIL'}: "
          f"{len(failures)} failure(s), {len(warnings)} warning(s) ===")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
