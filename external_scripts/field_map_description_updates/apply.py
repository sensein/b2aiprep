#!/usr/bin/env python3
import io
"""Apply the approved v4 description (and name) changes to bids_field_organization.csv.

The field map's `description` column is the single source of truth the pipeline uses
(dataset.py: it overrides the ReproSchema question). This script writes the approved
descriptions into that column, and the approved renames into `column_name`.

SAFETY:
  * Dry run by default — prints exactly what WOULD change and writes nothing.
  * Matches each change to one field-map row by (schema_name, column_name); never guesses.
  * On --apply: backs up the CSV, rewrites only the targeted cells (QUOTE_MINIMAL, LF),
    then re-reads and verifies cell-by-cell that ONLY the intended cells changed.

Usage:
  python apply.py              # dry run (report only)
  python apply.py --apply      # back up, write, verify
"""
import argparse, csv, datetime, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FMAP = HERE.parent.parent / 'src' / 'b2aiprep' / 'prepare' / 'resources' / 'bids_field_organization.csv'
CHANGES = HERE / 'approved_changes.tsv'


def read_csv_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.reader(f))


def read_rows_with_raw(path):
    """Parse CSV rows AND capture the exact raw text of each record (handles multi-line
    quoted fields), so untouched rows can be re-emitted byte-for-byte."""
    with open(path, newline='', encoding='utf-8') as f:
        lines = f.readlines()
    buf = []

    def feed():
        for ln in lines:
            buf.append(ln)
            yield ln

    reader = csv.reader(feed())
    rows, raws, start = [], [], 0
    for rec in reader:
        rows.append(rec)
        raws.append(''.join(buf[start:len(buf)]))
        start = len(buf)
    return rows, raws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='write the file (default: dry run)')
    args = ap.parse_args()

    all_rows, all_raws = read_rows_with_raw(FMAP)
    header, data = all_rows[0], all_rows[1:]
    raw_header, raw_data = all_raws[0], all_raws[1:]
    H = {name: i for i, name in enumerate(header)}
    SI, CI, DI = H['schema_name'], H['column_name'], H['description']

    # index field-map rows by (schema_name, column_name) and by column_name
    by_key, by_col = {}, {}
    for idx, r in enumerate(data):
        by_key.setdefault((r[SI], r[CI]), []).append(idx)
        by_col.setdefault(r[CI], []).append(idx)

    with open(CHANGES, newline='', encoding='utf-8') as f:
        changes = list(csv.reader(f, delimiter='\t'))
    chh = {n: i for i, n in enumerate(changes[0])}
    cs, cc = chh['schema_name'], chh['column']
    cdesc, cname, csrc = chh['new_description'], chh['new_name'], chh['source']

    planned = []          # (row_idx, kind, old, new, column, source)
    unmatched, ambiguous, desc_mismatch = [], [], []

    for ch in changes[1:]:
        schema, col = ch[cs], ch[cc]
        new_desc, new_name, src = ch[cdesc], ch[cname], ch[csrc]
        hits = by_key.get((schema, col)) or (by_col.get(col) if len(by_col.get(col, [])) == 1 else None)
        if not hits:
            unmatched.append((schema, col, src)); continue
        if len(hits) != 1:
            ambiguous.append((schema, col, [data[i][SI] for i in hits])); continue
        i = hits[0]
        cur_desc, cur_col = data[i][DI], data[i][CI]
        if new_desc and new_desc.strip() != cur_desc.strip():
            planned.append((i, 'description', cur_desc, new_desc, col, src))
        if new_name and new_name != cur_col:
            planned.append((i, 'column_name', cur_col, new_name, col, src))

    desc_changes = [p for p in planned if p[1] == 'description']
    name_changes = [p for p in planned if p[1] == 'column_name']

    print(f'Field map: {FMAP}')
    print(f'Approved changes read: {len(changes) - 1}')
    print(f'  description cells to change: {len(desc_changes)}')
    print(f'  column_name (rename) cells to change: {len(name_changes)}')
    print(f'  unmatched (NOT found, skipped): {len(unmatched)}')
    print(f'  ambiguous (multiple rows, skipped): {len(ambiguous)}')
    if unmatched:
        print('  UNMATCHED:', unmatched[:20])
    if ambiguous:
        print('  AMBIGUOUS:', ambiguous[:20])
    print('\nRenames (column_name), full list:')
    for i, k, old, new, col, src in name_changes:
        print(f'    {old}  ->  {new}')
    print('\nSample description changes (first 8):')
    for i, k, old, new, col, src in desc_changes[:8]:
        print(f'    [{col}] {old[:50]!r} -> {new[:60]!r}')

    if not args.apply:
        print('\n(DRY RUN — nothing written. Re-run with --apply to write.)')
        return

    if unmatched or ambiguous:
        sys.exit('REFUSING to write: unmatched or ambiguous changes present (see above).')

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = HERE / f'bids_field_organization.backup_{ts}.csv'
    shutil.copy2(FMAP, backup)
    print(f'\nBacked up original -> {backup}')

    changed_rows = {i for i, *_ in planned}
    new_data = [list(r) for r in data]
    for i, kind, old, new, col, src in planned:
        new_data[i][DI if kind == 'description' else CI] = new

    # Write: untouched rows re-emitted byte-for-byte from the original; only changed rows
    # are re-serialized. Keeps the diff to exactly the intended edits.
    def serialize(row):
        buf = io.StringIO()
        csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator='\n').writerow(row)
        return buf.getvalue()

    with open(FMAP, 'w', newline='', encoding='utf-8') as f:
        f.write(raw_header)
        for i, row in enumerate(new_data):
            f.write(serialize(row) if i in changed_rows else raw_data[i])

    # verify: re-read and confirm ONLY the intended cells differ
    after = read_csv_rows(FMAP)[1:]
    intended = {(i, DI if k == 'description' else CI) for i, k, *_ in planned}
    actual = set()
    assert len(after) == len(data), 'row count changed!'
    for i, (o_row, n_row) in enumerate(zip(data, after)):
        assert len(o_row) == len(n_row), f'column count changed at row {i}'
        for j, (a, b) in enumerate(zip(o_row, n_row)):
            if a != b:
                actual.add((i, j))
    extra = actual - intended
    missing = intended - actual
    print(f'\nVerification: intended cell edits={len(intended)}, actual cells changed={len(actual)}')
    if extra:
        print(f'  !! UNINTENDED changes in {len(extra)} cells:', list(extra)[:10])
        sys.exit('ABORTING semantics check: unintended cells changed (restore from backup).')
    if missing:
        print(f'  !! {len(missing)} intended edits did not take (likely identical text).')
    print('  OK: only the intended description/column_name cells changed.' if not extra else '')


if __name__ == '__main__':
    main()
