#!/usr/bin/env python3
"""Profile every phenotype field of a Bridge2AI-Voice release tree: TSVs -> profile.json.

Step 1 of the Data Explorer (data_explorer.html); build_explorer.py renders the profile.
See README.md in this folder.

Parsing rules:
  * TSVs use CSV quoting -> csv.reader(delimiter='\\t'), never split on tabs.
  * `base___option` columns are REDCap checkbox indicators.
  * non-numeric sentinels (e.g. "90 and above") are excluded from numeric ranges.

v2 (2026-09-29): --list-ids-max N records, for every value held by at most N rows (blanks
included, and a numeric column's minimum and maximum), the participant, session and site of
those rows, shown on hover in the page. --sites names a TSV with participant_id and
enrollment_institution (the internal tree's participant.tsv). On a deidentified tree the listed
IDs are its pseudonyms; on an internal tree, or with --sites or --org, they link participants to
rare answers, so keep that output internal.

v3 (2026-10-15): participant counts next to row counts. Each table gets n_participants (distinct
participant_id), each column n_participants_filled (distinct participants with a value), each
value np (distinct participants holding it; values_more entries become [value, rows, participants]),
a checkbox selected_participants, and meta.n_participants (distinct participants across all tables).

Usage:
  python3 analyze_phenotype.py ROOT OUT.json [--org bids_field_organization.csv] [--list-ids-max 10] [--sites participant.tsv]
Python 3.6 compatible (the cluster's /usr/bin/python3).
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
import re
import sys

csv.field_size_limit(sys.maxsize)

# ---- thresholds (judgment calls; documented in FIELD_PROFILE_CAVEATS.md) ----
NUMERIC_FRAC = 0.90      # >= this share of non-empty rows parse as numbers -> numeric
CODED_MAX_DISTINCT = 15  # numeric with <= this many distinct values -> numeric_coded
DATETIME_FRAC = 0.90     # > this share of distinct values look like ISO dates -> datetime
CATEGORICAL_MAX = 30     # <= this many distinct values -> categorical
TOP_N = 15               # text / datetime fields list only the top N values
HIST_BINS = 24
HIST_MIN_BINS = 6
FILES_LISTED = 3         # payload lists only the first N files of a merged table
ANCHOR_MAX_CHOICES = 3
NA_TOKENS = {'na', 'n/a', 'nan', 'null', 'none', '.'}
ISO_DATE = re.compile(r'^\d{4}-\d{2}-\d{2}')
SESSIONS_SCHEMA = 'session'  # merged sub-*_sessions table is joined against this schema
# deprecation markers written by the REDCap export into descriptions
DEPRECATED_FIELD = re.compile(r'^\s*\[\s*deprecated\s*\]', re.I)          # "[DEPRECATED]\n..."
DEPRECATED_OPTS = re.compile(r'\[\s*deprecated in ([^\]]*)\]', re.I)      # "...options [Deprecated in Protocol v4]\n2, Mexican"
KEY_COLUMNS = {'participant_id', 'session_id'}  # ignored when deciding if a whole form is deprecated


def to_float(s):
    try:
        f = float(s)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def fmt_num(x):
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return '%.4g' % x


def quantile(sorted_vals, q):
    """Linear interpolation between order statistics (numpy default)."""
    n = len(sorted_vals)
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def histogram(vals):
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return [{'lo': lo, 'hi': hi, 'n': len(vals)}]
    nb = max(HIST_MIN_BINS, min(HIST_BINS, len(set(vals))))
    w = (hi - lo) / nb
    counts = [0] * nb
    for v in vals:
        i = int((v - lo) / w)
        counts[min(max(i, 0), nb - 1)] += 1
    return [{'lo': lo + i * w, 'hi': lo + (i + 1) * w, 'n': counts[i]} for i in range(nb)]


def read_tsv(path):
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.reader(fh, delimiter='\t'))
    return (rows[0], rows[1:]) if rows else ([], [])


# ---------------------------------------------------------------- sidecars
def load_sidecar(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as fh:
        j = json.load(fh)
    if len(j) == 1:
        j = next(iter(j.values()))
    return j.get('data_elements', {}), j.get('description') or ''


def choice_list(el):
    out = []
    for ch in el.get('choices') or []:
        name = ch.get('name')
        label = name.get('en', '') if isinstance(name, dict) else (name if name is not None else '')
        out.append({'label': label, 'value': ch.get('value')})
    return out


def choice_matchers(choices):
    keys, nums = [], []
    for ch in choices:
        k = set()
        for x in (ch['label'], ch['value']):
            if x is None:
                continue
            k.add(str(x).strip().lower())
        keys.append(k)
        nums.append({f for f in (to_float(ch['label']), to_float(ch['value'])) if f is not None})
    return keys, nums


def match_choice(v, keys, nums):
    """Index of the first choice matching v (label/value, case-folded, or numerically)."""
    s = v.strip().lower()
    f = to_float(v)
    for i, (k, n) in enumerate(zip(keys, nums)):
        if s in k or (f is not None and f in n):
            return i
    return None


def deprecation(name, elements, org):
    """Reason string if the field is deprecated, else None.

    * description starts with "[DEPRECATED]" in the sidecar or the field map;
    * `base___code` checkbox columns whose code is listed after a
      "[Deprecated in Protocol vN]" note (the question stays, those options are retired).
    """
    el = elements.get(name) or {}
    if DEPRECATED_FIELD.search(el.get('description') or ''):
        return 'marked [DEPRECATED] in the sidecar'
    if org and DEPRECATED_FIELD.search(org.get('description') or ''):
        return 'marked [DEPRECATED] in the field map'
    if '___' in name:
        code = name.rsplit('___', 1)[1]
        m = DEPRECATED_OPTS.search(el.get('description') or '')
        if m:
            tail = (el.get('description') or '')[m.end():]
            codes = re.findall(r'^\s*([A-Za-z0-9_]+)\s*,', tail, re.M)
            if code in codes:
                return 'option deprecated in %s' % m.group(1).strip()
    return None


# ---------------------------------------------------------------- field map
def load_org(path):
    by_key, by_col = {}, collections.defaultdict(list)
    with open(path, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            by_key[(r['schema_name'], r['column_name'])] = r
            by_col[r['column_name']].append(r)
    return by_key, by_col


def org_entry(r):
    return collections.OrderedDict(
        # the field map's `delete` column is `legacy_delete` from v4.0.0 on; read either
        (k, r[k] if k != 'delete' else r.get('legacy_delete', r.get('delete', ''))) for k in ('disposition', 'delete', 'column_type', 'source', 'version_added',
                            'is_redcap_calculation', 'date_shift', 'group', 'description'))


def join_org(schema, col, org):
    by_key, by_col = org
    r = by_key.get((schema, col))
    if r is not None:
        return 'exact', r['disposition'], org_entry(r)
    cands = by_col.get(col, [])
    if len(cands) == 1:
        e = org_entry(cands[0])
        e['matched_schema'] = cands[0]['schema_name']
        return 'by-column', cands[0]['disposition'], e
    return 'none', None, {}


# ---------------------------------------------------------------- profiling
def row_refs(keys, sites):
    """[participant, session, site] for each row key (participant, session)."""
    return [[p, s or '', sites.get(p, '')] for p, s in keys]


LIST_KINDS = ('categorical', 'numeric_coded')  # answer-like columns whose rare values are listed


def rare_ids(cells, keys, counts, n_empty, filled_nums, list_max, sites, kind, flagged):
    """Participant/session/site behind rare values ('' = blank rows).

    Listed: blank rows; values held by <= list_max rows in categorical / coded columns;
    flagged values (undeclared, out of range, non-numeric in a numeric column) held by <= list_max
    rows; a numeric column's minimum and maximum. Free text, datetimes, continuous numbers and
    identifier columns are near-unique, so listing their values would map every row to a
    participant (and link free-text answers to them): they list only blanks and extremes.
    """
    if not list_max or keys is None:
        return None
    rare = {v for v, n in counts.items() if n <= list_max and (kind in LIST_KINDS or v in flagged)}
    if 0 < n_empty <= list_max:
        rare.add('')
    by_value = collections.defaultdict(list)
    for cell, key in zip(cells, keys):
        v = cell.strip()
        if v in rare:
            by_value[v].append(key)
    out = {v: row_refs(k, sites) for v, k in by_value.items()}
    # numeric extremes, whatever their count, when few rows hold them
    if filled_nums:
        for tag, target in (('__min__', min(filled_nums)), ('__max__', max(filled_nums))):
            k = [key for cell, key in zip(cells, keys) if to_float(cell.strip()) == target]
            if 0 < len(k) <= list_max:
                out[tag] = row_refs(k, sites)
    return out


def profile_column(name, cells, el, has_schema, row_keys=None, list_max=0, sites=None, pids=None):
    n_rows = len(cells)
    # values are profiled whitespace-trimmed: 'Dentures' and 'Dentures ' count as one
    filled = [c.strip() for c in cells if c.strip() != '']
    n_filled = len(filled)
    counts = collections.Counter(filled)
    # distinct participants per value (pids: participant_id of each row, None if the table has none)
    ppl = collections.defaultdict(set)
    if pids is not None:
        for cell, p in zip(cells, pids):
            v = cell.strip()
            if v != '':
                ppl[v].add(p)
    np_filled = len(set().union(*ppl.values())) if pids is not None else None
    np_of = (lambda v: len(ppl.get(v, ()))) if pids is not None else (lambda v: None)
    n_distinct = len(counts)
    is_na = lambda v: v.strip().lower() in NA_TOKENS
    na_token_n = sum(n for v, n in counts.items() if is_na(v))

    el = el or {}
    choices = choice_list(el)
    keys, nums = choice_matchers(choices)

    num_vals = [to_float(v) for v in filled]
    num_ok = [f for f in num_vals if f is not None]
    kind = 'empty'
    if n_filled:
        if len(num_ok) >= NUMERIC_FRAC * n_filled:
            kind = 'numeric_coded' if n_distinct <= CODED_MAX_DISTINCT else 'numeric'
        elif sum(1 for v in counts if ISO_DATE.match(v.strip())) > DATETIME_FRAC * n_distinct:
            kind = 'datetime'
        elif n_distinct <= CATEGORICAL_MAX:
            kind = 'categorical'
        else:
            kind = 'text'
    is_num = kind in ('numeric', 'numeric_coded')

    # --- checkbox / anchor / mismatch classification
    distinct_nums = {to_float(v) for v in counts}
    choice_nums = [to_float(c['value']) for c in choices]
    all_choices_numeric = bool(choices) and all(f is not None for f in choice_nums)
    checkbox = None
    if (is_num and choices and distinct_nums <= {0.0, 1.0}
            and (len(choices) == 1 or not any(f is not None for f in choice_nums))):
        sel = sum(n for v, n in counts.items() if to_float(v) == 1.0)
        checkbox = {'option': choices[0]['label'], 'selected': sel,
                    'pct': 100.0 * sel / n_rows if n_rows else 0.0}
        if pids is not None:
            checkbox['selected_participants'] = len(set().union(
                *[ppl[v] for v in counts if to_float(v) == 1.0]))
    anchor = None
    if (checkbox is None and kind == 'numeric' and all_choices_numeric
            and len(choices) <= ANCHOR_MAX_CHOICES):
        lo = min(choices, key=lambda c: to_float(c['value']))
        hi = max(choices, key=lambda c: to_float(c['value']))
        anchor = {'min': str(lo['value']), 'max': str(hi['value'])}
    type_mismatch = None
    slider_labels = False
    if (checkbox is None and is_num and choices
            and not any(f is not None for f in choice_nums)):
        type_mismatch = 'sidecar declares non-numeric options, data is numeric'
        # a slider whose sidecar lists its scale labels (e.g. MI/MO/SE) as options:
        # the data is right, the sidecar is not
        s_lo, s_hi = to_float(el.get('minValue')), to_float(el.get('maxValue'))
        if s_lo is not None and s_hi is not None and all(s_lo <= f <= s_hi for f in num_ok):
            slider_labels = True
            type_mismatch = ('slider: sidecar lists its scale labels as options (not numeric codes); '
                             'data is numeric within the declared %g–%g' % (s_lo, s_hi))

    # --- option conformance (skipped for slider labels: already reported as the mismatch)
    check_options = (bool(choices) and kind != 'empty' and checkbox is None and anchor is None
                     and not slider_labels)
    used = set()
    undeclared = collections.Counter()
    declared_flag = {}
    for v in counts:
        if is_na(v):
            declared_flag[v] = False
            continue
        i = match_choice(v, keys, nums) if choices else None
        declared_flag[v] = i is not None
        if i is not None:
            used.add(i)
        elif check_options:
            undeclared[v] = counts[v]
    unused = [c for i, c in enumerate(choices) if i not in used] if check_options else []
    out_of_range = []
    if anchor is not None:
        a_lo, a_hi = to_float(anchor['min']), to_float(anchor['max'])
        out_of_range = [v for v in counts
                        if to_float(v) is not None and not a_lo <= to_float(v) <= a_hi]

    # --- value distribution
    def vrow(v, n):
        return {'v': v, 'n': n, 'np': np_of(v), 'pct': 100.0 * n / n_filled,
                'declared': declared_flag.get(v, False), 'na': is_na(v)}
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    truncated = 0
    values_more = []
    if kind == 'numeric_coded':
        ordered = sorted(counts.items(),
                         key=lambda kv: (to_float(kv[0]) is None, to_float(kv[0]) or 0.0, kv[0]))
        values = [vrow(v, n) for v, n in ordered]
    elif kind in ('categorical', 'text', 'datetime'):
        # low-cardinality fields list every value; the rest only the top N
        keep = ranked if n_distinct <= CATEGORICAL_MAX else ranked[:TOP_N]
        values = [vrow(v, n) for v, n in keep]
        if kind == 'text' and not name.endswith('_id'):
            # free text: the page shows the top N but can expand to every value.
            # compact [value, count] pairs; identifier columns (*_id) are skipped,
            # they are near-unique and would add megabytes of IDs.
            values_more = [[v, n, np_of(v)] for v, n in ranked[len(keep):]]
        truncated = n_distinct - len(keep)
    else:
        values = []

    numeric, hist = None, []
    if is_num and num_ok:
        s = sorted(num_ok)
        numeric = collections.OrderedDict([
            ('min', fmt_num(s[0])), ('max', fmt_num(s[-1])),
            ('mean', fmt_num(sum(s) / len(s))), ('median', fmt_num(quantile(s, .5))),
            ('q1', fmt_num(quantile(s, .25))), ('q3', fmt_num(quantile(s, .75))),
            ('n', len(s)), ('min_raw', s[0]), ('max_raw', s[-1])])
        hist = histogram(s)
    elif kind == 'datetime':
        d = sorted(v for v in filled if ISO_DATE.match(v.strip()))
        numeric = collections.OrderedDict([('min', d[0]), ('max', d[-1]), ('n', len(d))])

    q = el.get('question')
    col = collections.OrderedDict([
        ('name', name), ('n_rows', n_rows), ('n_empty', n_rows - n_filled),
        ('pct_empty', 100.0 * (n_rows - n_filled) / n_rows if n_rows else 0.0),
        ('n_filled', n_filled), ('n_participants_filled', np_filled),
        ('n_distinct', n_distinct), ('na_token_n', na_token_n),
        ('kind', kind), ('values', values), ('truncated', truncated),
        ('values_more', values_more),
        ('numeric', numeric), ('hist', hist),
        ('declared_choices', choices), ('unused_choices', unused),
        ('anchor_choices', anchor), ('out_of_declared_range', out_of_range),
        ('type_mismatch', type_mismatch), ('checkbox', checkbox),
        ('undeclared_values', [v for v, _ in undeclared.most_common(TOP_N)]),
        ('description', el.get('description', '')),
        ('question', q.get('en', '') if isinstance(q, dict) else (q or '')),
        ('datatype', el.get('datatype') or el.get('valueType') or []),
    ])
    if has_schema:
        col['schema_min'] = el.get('minValue')
        col['schema_max'] = el.get('maxValue')
    if kind != 'empty':
        bad = collections.Counter(v for v in filled if to_float(v) is None and not is_na(v))
    else:
        bad = collections.Counter()
    flagged = set(undeclared) | set(out_of_range) | (set(bad) if is_num else set())
    if name not in KEY_COLUMNS and not name.endswith('_id'):
        ids = rare_ids(cells, row_keys, counts, n_rows - n_filled, num_ok if is_num else [], list_max,
                       sites or {}, kind, flagged)
        if ids is not None:
            col['ids'] = ids
    if kind != 'empty':
        col['non_numeric_in_numeric'] = ([{'v': v, 'n': n} for v, n in bad.most_common()]
                                         if is_num else [])
    return col


def profile_table(root, label, group, files, sidecar, schema_name, org, list_max=0, sites=None):
    header, rows, variants, ragged, nbytes, used = None, [], 0, 0, 0, []
    headers_seen = set()
    for f in files:
        h, r = read_tsv(os.path.join(root, f))
        headers_seen.add(tuple(h))
        if header is None:
            header = h
        elif h != header:
            continue  # differing header: skipped (reported via header_variants)
        used.append(f)
        nbytes += os.path.getsize(os.path.join(root, f))
        for row in r:
            if len(row) != len(header):
                ragged += 1
                row = (row + [''] * len(header))[:len(header)]
            rows.append(row)
    has_schema = sidecar is not None
    elements, table_desc = sidecar if has_schema else ({}, '')
    pids = [r[header.index('participant_id')] for r in rows] if 'participant_id' in header else None
    keys = None
    if list_max and 'participant_id' in header:
        pi = header.index('participant_id')
        # REDCap names each form's session column <form>_session_id (e.g. demographics_session_id)
        ses_cols = ['session_id'] if 'session_id' in header else [h for h in header if h.endswith('_session_id')]
        si = header.index(ses_cols[0]) if len(ses_cols) == 1 else None
        keys = [(r[pi], r[si] if si is not None else '') for r in rows]
    cols = []
    for i, name in enumerate(header):
        c = profile_column(name, [r[i] for r in rows], elements.get(name), has_schema,
                           keys, list_max, sites, pids)
        if org is not None:
            m, disp, entry = join_org(schema_name, name, org)
            c['org_match'], c['disposition'], c['org'] = m, disp, entry
        c['deprecated'] = deprecation(name, elements, c.get('org'))
        cols.append(c)
    content = [c for c in cols if c['name'] not in KEY_COLUMNS]
    if DEPRECATED_FIELD.search(table_desc) or re.search(r'deprecated', label, re.I):
        form_dep = 'form marked deprecated in its sidecar'
    elif content and all(c['deprecated'] for c in content):
        form_dep = 'every field in the form is deprecated'
    else:
        form_dep = None
    return collections.OrderedDict([
        ('label', label), ('group', group), ('files', files[:FILES_LISTED]), ('n_files', len(files)),
        ('n_rows', len(rows)), ('n_participants', len(set(pids)) if pids is not None else None),
        ('participant_ids', sorted(set(pids)) if pids is not None else []),
        ('n_cols', len(header)), ('columns', cols),
        ('schema_only_fields', [k for k in elements if k not in header] if has_schema else []),
        ('undocumented_fields', [h for h in header if h not in elements] if has_schema else []),
        ('header_variants', len(headers_seen)), ('ragged_rows', ragged),
        ('has_schema', has_schema), ('bytes', nbytes),
        ('deprecated', form_dep), ('n_deprecated', sum(1 for c in cols if c['deprecated']))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root')
    ap.add_argument('out')
    ap.add_argument('--org', help='bids_field_organization.csv; adds disposition/provenance')
    ap.add_argument('--list-ids-max', type=int, default=10,
                    help='list participant/session/site for values held by at most this many rows (0 = off)')
    ap.add_argument('--sites', help='TSV with participant_id and enrollment_institution')
    a = ap.parse_args()
    sites = {}
    if a.sites:
        with open(a.sites, newline='', encoding='utf-8') as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                sites[r['participant_id']] = r.get('enrollment_institution') or ''
    root = os.path.abspath(a.root)
    org = load_org(a.org) if a.org else None

    tables = []
    for p in sorted(glob.glob(os.path.join(root, 'phenotype', '*', '*.tsv'))):
        rel = os.path.relpath(p, root)
        label = os.path.splitext(os.path.basename(p))[0]
        side = load_sidecar(os.path.splitext(p)[0] + '.json')
        tables.append(profile_table(root, label, os.path.dirname(rel), [rel], side, label, org,
                                    a.list_ids_max, sites))
        print('%-40s %6d rows' % (label, tables[-1]['n_rows']), file=sys.stderr)
    sess = sorted(os.path.relpath(p, root) for p in glob.glob(os.path.join(root, 'sub-*', 'sub-*_sessions.tsv')))
    if sess:
        tables.append(profile_table(root, 'sub-*_sessions', 'subject-level', sess, None,
                                    SESSIONS_SCHEMA, org, a.list_ids_max, sites))
        print('%-40s %6d rows (%d files)' % ('sub-*_sessions', tables[-1]['n_rows'], len(sess)), file=sys.stderr)

    # distinct participants across all tables; the ID lists themselves stay out of the payload
    all_pids = set()
    for t in tables:
        all_pids.update(t.pop('participant_ids'))

    meta = collections.OrderedDict()
    meta['n_participants'] = len(all_pids)
    if org is not None:
        disp = collections.OrderedDict()
        for t in tables:
            for c in t['columns']:
                k = c['disposition'] or '(unmapped)'
                disp[k] = disp.get(k, 0) + 1
        meta['org_csv'] = os.path.abspath(a.org)
        meta['disposition_counts'] = disp
    meta['root'] = root
    meta['list_ids_max'] = a.list_ids_max
    meta['sites_from'] = os.path.abspath(a.sites) if a.sites else None
    meta['n_tables'] = len(tables)
    meta['n_tsv_files'] = sum(t['n_files'] for t in tables)
    meta['n_fields'] = sum(t['n_cols'] for t in tables)
    meta['n_rows'] = sum(t['n_rows'] for t in tables)
    meta['n_deprecated_fields'] = sum(t['n_deprecated'] for t in tables)
    meta['n_deprecated_forms'] = sum(1 for t in tables if t['deprecated'])
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(collections.OrderedDict([('meta', meta), ('tables', tables)]), fh, ensure_ascii=False)
    print('wrote %s: %d tables, %d fields' % (a.out, meta['n_tables'], meta['n_fields']), file=sys.stderr)


if __name__ == '__main__':
    main()
