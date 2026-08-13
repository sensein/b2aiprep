# Quickstart: Generated Release Metadata

End-to-end on the synthetic fixtures already in the repo. No release data, no cluster, no credentials.

## Environment

The release environment is the `b2aiprep_test` conda env — the one every script in
`post_3.0/v3.1/scripts/` activates. Develop and test against it, not a fresh venv, so versions match
what actually runs.

```sh
conda activate b2aiprep_test
python -c "import sys; print(sys.version)"            # expect 3.12.13
python -m pip install -e '/path/to/b2aiprep[metadata]'  # adds fairscape-cli, fairscape-models
```

`fairscape-cli` and `fairscape-models` are absent from this env today; the `metadata` extra is what adds
them. Everything else the feature needs — senselab 1.3.0, the six backends, pyarrow, synapseclient — is
already installed.

One thing to know before generating anything for real: this env has `b2aiprep 3.1.0+51.g34174e7.dirty`
installed from a modified tree. A version with a local segment cannot resolve to a citable record, so a
PhysioNet-bound crate will be refused until the env is installed from a tagged commit. That refusal is
intended, not a bug.

## 1. Build a fixture release

```sh
export WORK=$(mktemp -d)
cd /path/to/b2aiprep

b2aiprep-cli redcap2bids data/sdv_redcap_synthetic_data_1000_rows.csv \
    --outdir $WORK/bids --audiodir data --sanitize_audio_format
b2aiprep-cli generate-audio-features $WORK/bids $WORK/bids --is_sequential True
b2aiprep-cli run-quality-control-on-audios $WORK/bids
b2aiprep-cli create-bundled-dataset $WORK/bids $WORK/bundle
```

Each step now also writes records:

```sh
ls $WORK/bids/provenance/runs/    # redcap2bids-*.json, generate-audio-features-*.json, quality-control-*.json
ls $WORK/bids/provenance/units/   # generate-audio-features-*.tsv
ls $WORK/bundle/provenance/runs/  # bundle-*.json
```

Sanity-check one:

```sh
python - <<'PY'
import json, glob
r = json.load(open(sorted(glob.glob('$WORK/bids/provenance/runs/generate-audio-features-*.json'))[0]))
print(r['status'], {s['name']: s['version'] for s in r['software']})
print([(m['role'], m['repo'], m['resolved_sha'][:12]) for m in r['models']])
PY
```

Expect `complete`, real versions for `b2aiprep`/`senselab`/each backend, and a resolved commit sha for
each of the three models — not `main`.

## 2. Generate the metadata

```sh
b2aiprep-cli generate-release-metadata $WORK/bundle \
    --distribution feature-bundle \
    --release-version 0.0.0-fixture \
    --release-metadata src/b2aiprep/prepare/resources/release_metadata.yaml \
    --c2m2-out $WORK/c2m2
```

Expect at `$WORK/bundle`: `ro-crate-metadata.json` plus the seven derived artifacts. Confirm the
invariant that mattered most in 3.0.0 — every file described, nothing empty:

```sh
python - <<'PY'
import json, os
root='$WORK/bundle'
d=json.load(open(f'{root}/ro-crate-metadata.json'))
described={e.get('contentUrl') for e in d['@graph'] if e.get('contentUrl')}
present={os.path.relpath(os.path.join(dp,f), root)
         for dp,_,fs in os.walk(root) for f in fs
         if 'provenance' not in dp and not f.startswith(('ro-crate-','ai_ready_score'))}
print('described but absent:', len(described)-len(present & {u.replace('file:///','') for u in described}))
empties=sum(1 for e in d['@graph'] for v in e.values() if v in ('',[],{},None))
print('empty property values:', empties)   # must be 0; 3.0.0 has 286
PY
```

## 3. Validate

```sh
b2aiprep-cli validate-release-metadata $WORK/bundle --provenance-dir $WORK/bundle/provenance
echo "exit: $?"     # 0
```

Then prove it actually checks something. Break one thing and re-run:

```sh
python -c "
import json; p='$WORK/bundle/ro-crate-metadata.json'
d=json.load(open(p)); d['@graph'][2]['description']=''; json.dump(d, open(p,'w'))"
b2aiprep-cli validate-release-metadata $WORK/bundle
echo "exit: $?"     # 1, naming the emptied property
```

## 4. Validate the published 3.0.0 crate

The validator's own acceptance test — it must find the real defects in what shipped:

```sh
b2aiprep-cli validate-release-metadata \
    "/orcd/data/satra/002/datasets/b2aivoice/claude_safe/ro-crate/voice-ro-crate-assets 2/ro-crate-metadata.json" \
    --inventory /orcd/data/satra/002/datasets/b2aivoice/post_3.0/data/adult/bundled_12_21_25
```

Expect a non-zero exit listing at least: 286 empty property values; `ark:59853/b2ai-voice-schema-phenotype-confounders`
used twice; `sparc-periodicity` named `sparc_loudness.parquet`; `torchaudio-pitch` named
`torchaudio_spectrogram.parquet`; `contentSize: "12.9 GB"` as prose against children summing to
13,789,023,450 bytes; the features computation dated `01/29/2026` after its outputs' `12/16/2025`; four
phenotype datasets with `contentUrl: []`; and ~83 of ~98 files undescribed.

This is a read-only audit — it regenerates nothing and touches no published file.

## 5. Cross-dataset reconciliation

Needs both distributions. Against the real v3.1 trees, read-only:

```sh
b2aiprep-cli validate-release-metadata \
    /orcd/data/satra/002/datasets/b2aivoice/post_3.0/v3.1/adult/deid_bids_registered_04_14_26 \
    --sibling /orcd/data/satra/002/datasets/b2aivoice/post_3.0/v3.1/adult/deid_bids_controlled_04_14_26
```

Expect participant overlap reported as 767 shared, 66 registered-only, 0 controlled-only — matching the
direct count — with no containment rule asserted, and the count of recordings that could not be matched
across trees reported rather than dropped. Two of the 767 shared participants have differing session sets
between the trees, so a non-zero unmatched count here is correct, not a bug.

## Tests

```sh
pytest tests/test_provenance_records.py tests/test_rocrate_generation.py \
       tests/test_metadata_validation.py tests/test_cross_dataset_reconcile.py
```

These use only the fixtures in `data/`. The 3.0.0 comparison in step 4 stays a documented manual
procedure — it needs the real bundle, so neither it nor any release data belongs in CI.
