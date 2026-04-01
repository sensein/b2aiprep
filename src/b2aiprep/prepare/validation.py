import pandas as pd
from collections import defaultdict
from pathlib import Path
import re


def _get_multiselect_groups(columns: list[str]) -> dict[str, list[str]]:
    """
    Identify multi-select column groups (e.g. race___1, race___2 -> 'race').
    Returns a dict mapping base_name -> list of constituent columns.
    """
    groups = defaultdict(list)
    for col in columns:
        match = re.match(r'^(.+)___(.+)$', col)
        if match:
            groups[match.group(1)].append(col)
    return dict(groups)


def _collapse_multiselect(row: pd.Series, cols: list[str]) -> frozenset:
    """
    Collapse a set of multi-select columns into a frozenset of selected option keys.
    Treats truthy values (1, True, '1', 'true', 'yes') as selected.
    """
    selected = set()
    for col in cols:
        val = row[col]
        if pd.isna(val):
            continue
        # Accept numeric 1/1.0 or string equivalents ('1', 'true', 'yes')
        try:
            is_selected = float(val) == 1.0
        except (ValueError, TypeError):
            is_selected = str(val).strip().lower() in {'true', 'yes'}
        if is_selected:
            # Store just the suffix after ___ as the selection label
            suffix = col.split('___', 1)[1]
            selected.add(suffix)
    return frozenset(selected)


def _find_session_cols(df: pd.DataFrame) -> list[str]:
    """Return all columns whose name contains 'session_id' as a substring."""
    return [c for c in df.columns if 'session_id' in c.lower()]


def check_multi_line_phenotypes(
    df: pd.DataFrame,
    participant_col: str = 'participant_id',
    session_col: str | None = None,
    output_dir: str | Path | None = None,
    save_per_column: bool = False,
    ignore_null_conflicts: bool = False,
) -> pd.DataFrame:
    """
    Check whether participants have inconsistent values across rows for any
    phenotype column. Multi-select column groups (base___option) are collapsed
    to a set before comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Must contain `participant_col`. A session ID column
        is optional — if none is found, rows are compared purely by participant.
    participant_col : str
        Name of the participant ID column.
    session_col : str, optional
        Exact name of the session ID column. If None, any column whose name
        contains 'session_id' (case-insensitive) is auto-detected and excluded
        from phenotype comparison. If no such column exists, the function
        simply checks for duplicate participant rows.
    output_dir : str or Path, optional
        If provided, write a summary TSV (and per-column TSVs if
        `save_per_column=True`) to this directory.
    save_per_column : bool
        Whether to also save one TSV per differing column.
    ignore_null_conflicts : bool
        If True, a difference is only flagged when at least two non-null values
        disagree with each other. NaN vs a real value is not considered a
        conflict. If False (default), any NaN alongside a non-NaN value is
        flagged as an inconsistency.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns:
        [participant_col, session_id_list, column, values]
        One row per (participant, column) pair that shows any inconsistency.
    """
    detected_session_cols = (
        [session_col] if session_col is not None else _find_session_cols(df)
    )

    id_cols = {participant_col, *detected_session_cols}
    payload_cols = [c for c in df.columns if c not in id_cols]

    ms_groups = _get_multiselect_groups(payload_cols)
    ms_member_cols = {col for cols in ms_groups.values() for col in cols}

    # Build the "effective" column list:
    #   - multi-select groups -> one virtual column named after the base
    #   - all other payload columns -> unchanged
    scalar_cols = [c for c in payload_cols if c not in ms_member_cols]
    virtual_cols = list(ms_groups.keys())   # one entry per group
    effective_cols = scalar_cols + virtual_cols

    # Compute a resolved value per row for every effective column
    resolved_rows = []
    for _, row in df.iterrows():
        rec = {participant_col: row[participant_col]}
        for sc in detected_session_cols:
            rec[sc] = row[sc]
        for col in scalar_cols:
            rec[col] = row[col]
        for base, members in ms_groups.items():
            rec[base] = _collapse_multiselect(row, members)
        resolved_rows.append(rec)

    resolved = pd.DataFrame(resolved_rows)

    # For each participant, check consistency across rows per effective column
    records = []

    for pid, group in resolved.groupby(participant_col):
        sessions = {sc: group[sc].tolist() for sc in detected_session_cols}

        for col in effective_cols:
            vals = group[col].tolist()

            # Normalise for comparison: treat NaN as a sentinel
            def _normalise(v):
                if isinstance(v, frozenset):
                    return v
                return v

            normed = [_normalise(v) for v in vals]

            def _is_nan(v):
                return not isinstance(v, frozenset) and pd.isna(v)

            def _eq(a, b):
                # NaN is never equal to anything, including another NaN
                if _is_nan(a) or _is_nan(b):
                    return False
                return a == b

            non_null = [v for v in normed if not _is_nan(v) and v != frozenset()]
            if ignore_null_conflicts:
                has_diff = len(non_null) > 1 and any(
                    not _eq(non_null[i], non_null[j])
                    for i in range(len(non_null))
                    for j in range(i + 1, len(non_null))
                )
            else:
                has_diff = len(non_null) > 0 and any(
                    not _eq(normed[i], normed[j])
                    for i in range(len(normed))
                    for j in range(i + 1, len(normed))
                )
            if has_diff:
                records.append({
                    participant_col: pid,
                    'session_ids': sessions,
                    'column': col,
                    'values': vals,
                })

    if not records:
        result = pd.DataFrame(columns=[participant_col, 'session_ids', 'column', 'values'])
        print("No inconsistencies found.")
        return result

    result = pd.DataFrame(records)

    # ------------------------------------------------------------------ #
    # Optionally serialise to disk                                         #
    # ------------------------------------------------------------------ #
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Flatten lists to strings for CSV compatibility
        flat = result.copy()
        flat['session_ids'] = flat['session_ids'].apply(
            lambda x: '; '.join(str(s) for s in x)
        )
        flat['values'] = flat['values'].apply(
            lambda x: ' | '.join(str(v) for v in x)
        )

        summary_path = out / 'phenotype_inconsistencies.tsv'
        flat.to_csv(summary_path, index=False, sep='\t')
        print(f"Summary written to {summary_path}")

        if save_per_column:
            for col_name, col_df in result.groupby('column'):
                safe_name = re.sub(r'[^\w\-]', '_', col_name)
                col_path = out / f'inconsistencies_{safe_name}.tsv'

                per_col_rows = []
                for _, row in col_df.iterrows():
                    sessions = row['session_ids']  # dict or {}
                    for i, val in enumerate(row['values']):
                        rec = {participant_col: row[participant_col]}
                        for sc, sv in sessions.items():
                            rec[sc] = sv[i] if i < len(sv) else None
                        rec['value'] = val
                        per_col_rows.append(rec)

                per_col_df = pd.DataFrame(per_col_rows)
                per_col_df.to_csv(col_path, index=False, sep='\t')

            print(f"Per-column files written to {out}/")

    return result
