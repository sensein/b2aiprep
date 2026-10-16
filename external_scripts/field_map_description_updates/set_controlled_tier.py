#!/usr/bin/env python3
"""Set access_tier=controlled on the v4 medication fields that must be controlled-only:
specific-drug treatment lists (ET / Parkinson's / ALS) and the two free-text psychiatric
medication lists. Generic drug-class fields stay registered (Alex, 2026-10-07).

Surgical: untouched rows are re-emitted byte-for-byte; only the access_tier cell of the
targeted rows changes. Verifies that afterward. Dry run by default; --apply to write.
"""
import argparse, csv, io, sys
from pathlib import Path

FMAP = Path(__file__).resolve().parents[2] / 'src' / 'b2aiprep' / 'prepare' / 'resources' / 'bids_field_organization.csv'
BASES = [
    'et_medications_selection',
    'diagnosis_parkinsons_current_treatment_obtained_yes_medications',
    'diagnosis_parkinsons_treatment_obtained_yes_medications',
    'diagnosis_als_treatment_and_management_yes_medications',
]
FREETEXT = ['mph_current_medication', 'mph_previous_medication']


def targeted(col):
    if col in FREETEXT:
        return True
    return any(col == b or col.startswith(b + '___') for b in BASES)


def read_with_raw(path):
    lines = open(path, newline='', encoding='utf-8').readlines()
    buf = []
    def feed():
        for ln in lines:
            buf.append(ln); yield ln
    rows, raws, start = [], [], 0
    for rec in csv.reader(feed()):
        rows.append(rec); raws.append(''.join(buf[start:])); start = len(buf)
    return rows, raws


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--apply', action='store_true'); a = ap.parse_args()
    rows, raws = read_with_raw(FMAP)
    H = {n: i for i, n in enumerate(rows[0])}
    CI, TI = H['column_name'], H['access_tier']
    targets = [i for i in range(1, len(rows)) if targeted(rows[i][CI])]
    already = [i for i in targets if rows[i][TI] == 'controlled']
    todo = [i for i in targets if rows[i][TI] != 'controlled']
    print(f'field map: {FMAP}')
    print(f'targeted rows: {len(targets)} | already controlled: {len(already)} | to change: {len(todo)}')
    for i in todo:
        print(f'   {rows[i][CI]}: access_tier {rows[i][TI]!r} -> "controlled"')
    if not a.apply:
        print('\n(DRY RUN — nothing written. Re-run with --apply.)'); return

    new = [list(r) for r in rows]
    for i in todo:
        new[i][TI] = 'controlled'

    def ser(row):
        b = io.StringIO(); csv.writer(b, quoting=csv.QUOTE_MINIMAL, lineterminator='\n').writerow(row); return b.getvalue()
    with open(FMAP, 'w', newline='', encoding='utf-8') as f:
        f.write(raws[0])
        for i in range(1, len(new)):
            f.write(ser(new[i]) if i in set(todo) else raws[i])

    # verify: only access_tier cells of todo rows changed
    after = list(csv.reader(open(FMAP)))
    assert len(after) == len(rows), 'row count changed!'
    bad = []
    for i, (o, n) in enumerate(zip(rows, after)):
        for j, (x, y) in enumerate(zip(o, n)):
            if x != y and not (i in set(todo) and j == TI):
                bad.append((i, j))
    print(f'\nverified: {len(todo)} access_tier cells set to controlled; unintended changes: {len(bad)}')
    if bad:
        sys.exit(f'ABORT: unintended cell changes {bad[:8]}')
    print('OK: only the intended access_tier cells changed.')


if __name__ == '__main__':
    main()
