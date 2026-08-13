# Phase 1 Contract: CLI surface

Two new commands, registered in `src/b2aiprep/cli.py` alongside the existing ones and following the same
verb-first naming as `generate-audio-features` and `validate-bundled-dataset`.

Existing commands gain no required arguments. Each grows an optional `--provenance-dir` (defaulting to
`<dataset_root>/provenance`) and writes a record as a side effect; no existing invocation changes
meaning, so every script in `post_3.0/v3.1/scripts/` keeps working unmodified.

---

## `b2aiprep-cli generate-release-metadata`

```
b2aiprep-cli generate-release-metadata DATASET_PATH
    --distribution {feature-bundle|deid-audio}
    --release-version TEXT              # e.g. 4.0.0; becomes the PhysioNet path segment
    --release-metadata PATH             # the single editorial source (release_metadata.yaml)
    [--provenance-dir PATH]             # default DATASET_PATH/provenance
    [--sibling PATH_OR_ID ...]          # other distributions of this release, for cross-reference
    [--c2m2-out PATH]                   # also emit the C2M2 submission here
    [--dry-run]                         # report what would be written; write nothing
```

**Preconditions**

| Condition | On failure |
|---|---|
| `DATASET_PATH` exists and contains a recognisable distribution layout | exit 2, name the missing marker |
| `<provenance-dir>/runs/` holds at least a `bundle` (or `deidentify`) record accounting for every file present | exit 3, list unaccounted files |
| every record has `status == complete` | exit 3, name the incomplete `run_id`s |
| every published family resolves to exactly one Activity | exit 3, name the ambiguous family and the competing `run_id`s |
| every `SoftwareComponent` version is release-identifiable (no local version segment) and has a resolving DOI | exit 4, name the component and its version |
| `--release-metadata` carries an approval record | exit 4 |
| no identifier-crosswalk file is reachable inside `DATASET_PATH` | exit 5, name the file |

**Postconditions on success (exit 0)**

Writes, at `DATASET_PATH`: `ro-crate-metadata.json`, `ro-crate-datasheet.html`,
`ro-crate-preview.html`, `ro-crate-croissant.json`, `ro-crate-prov-graph.{json,html}`,
`ro-crate-merkle-tree.json`, `ai_ready_score.json`. With `--c2m2-out`, also the C2M2 TSVs at that path.

Guarantees: every file in the distribution described exactly once with its own size and digest; every
identifier unique; every declared name agreeing with its location; no empty, null, or placeholder value;
every aggregate equal to the sum over described items; all dates ISO 8601 and inside the run→publication
interval. Byte-identical across runs given identical inputs, except a single generation timestamp.

**Failure behaviour**: nothing is written or overwritten on any non-zero exit. All violations are
reported in one pass, not one per invocation.

---

## `b2aiprep-cli validate-release-metadata`

```
b2aiprep-cli validate-release-metadata TARGET
    [--provenance-dir PATH]             # omit to validate published metadata alone
    [--inventory PATH]                  # file listing for a remote distribution (e.g. a Synapse manifest)
    [--sibling PATH_OR_ID ...]          # enables cross-distribution reconciliation checks
    [--format {text|json}]
```

`TARGET` is a dataset root or an `ro-crate-metadata.json`. With no `--provenance-dir`, it checks only
what can be checked from the artifact and the inventory — which is how an already-published release is
audited without regenerating anything (FR-022).

**Checks**, each mapping to a requirement:

| Check | Requirement |
|---|---|
| described-file set equals present-file set | FR-002 |
| every described file has its own size and digest | FR-002 |
| no empty, null, or placeholder value | FR-003 |
| identifiers unique; declared name agrees with location | FR-004 |
| aggregates equal the sum/cardinality over described items, in machine-comparable units | FR-005 |
| dates ISO 8601 and within the run→publication interval | FR-006 |
| each data dictionary's fields equal its file's columns | FR-007 |
| every software component versioned and durably identified; every model at an immutable revision | FR-008, FR-009 |
| published rows + non-`computed` unit rows == recording × family total | FR-011 |
| per-family coverage counts present | FR-012 |
| each distribution's host/licence/identifier its own; sibling identifiers resolve | FR-013 |
| overlap stated as counts, no containment rule asserted | FR-014 |
| pending facts marked pending, not empty or invented | FR-015 |
| shared facts identical between RO-Crate and C2M2 | FR-016 |
| no unmapped controlled-vocabulary term; no dropped row | FR-017 |
| every contributor identifier present resolves, the count lacking one is stated, award identifiers are structured | FR-018 |
| no participant identifiers, free text, or transcripts in any artifact | FR-025 |
| no crosswalk file inside a published dataset | FR-026 |

**Exit codes**: `0` all checks pass; `1` one or more failed (every failure listed with item and
artifact); `2` target unreadable or unrecognised.

Pointed at the published 3.0.0 metadata, this must report the catalogued defects — 286 empty values, the
two duplicated identifiers, the two name/location mismatches, the prose `contentSize`, the computation
dated after its outputs, and the undescribed files. That is the acceptance test for the validator itself.

---

## Records written by existing commands

| Command | Record | Notable content |
|---|---|---|
| `redcap2bids` | `runs/redcap2bids-<run_id>.json` | `source_export` (path, sha256, row count, access tier); `b2ai-redcap2rs` commit via `get_commit_sha` (`prepare/utils.py:52`) |
| `generate-audio-features` | `runs/…json` + `units/…tsv` | software and backend versions; three models with resolved shas; effective config; per `(recording, family)` outcome |
| `run-quality-control-on-audios` | `runs/…json` | senselab metric versions, thresholds, `deep_checks`/`skip_windowing`, row count |
| `deidentify-bids-dataset` | `runs/…json` + `units/…tsv` | config digests and per-rule affected counts; deidentified identifiers only |
| `create-bundled-dataset` | `runs/…json` + `units/…tsv` | per-output size and digest; `bundle_output_stats` (`commands.py:333`) promoted from a log line to a record; per-family coverage |

`--provenance-dir` is the only new option on each. Writing a record is not optional — a step that cannot
write one fails, rather than producing data that cannot later be described.

---

## Compatibility obligations

- `external_scripts/c2m2/*.py` keep their current argparse interfaces and continue to work as
  `external_scripts/c2m2/README.md` documents, now importing `b2aiprep.metadata.c2m2`.
- `sage_generate_manifest.py` and `verify_sage_contents.py` must agree on whether `provenance/` is
  uploaded; the choice is recorded (FR-028). `verify_sage_contents.py` additionally must exit non-zero on
  a digest mismatch, which today it does not.
- `c2m2_mappings` gains entries for the provenance file types, or an explicit exclusion, so the bare
  `rglob('*')` at `bundle_to_c2m2.py:34` cannot produce rows with empty format and data type.
