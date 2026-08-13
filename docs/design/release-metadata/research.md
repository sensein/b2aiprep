# Phase 0 Research: Generated Release Metadata

All open questions carried out of the spec are resolved below. Each finding was verified against the
repository, the release scripts under `post_3.0/v3.1/scripts/`, or the built release trees — not
inferred.

---

## R1. Do the two published datasets share a pseudonym space?

**Decision**: Yes for participants. Participant overlap between the two published datasets is
computable directly from published identifiers, with no access to the private crosswalk.

**Evidence**: `adult_registered_generate_bids.sh` and `adult_controlled_generate_bids.sh` convert two
different RedCap exports (`registered_adult_07_01.csv`, `controlled_redcap_04_14_26.csv`) into two
separate BIDS trees, and `adult_deidentify.sh` deidentifies both using **one** config directory,
`post_3.0/config/shared_release_config_peds_all/`, which contains a single static `id_remapping.json`.
Same mapping file for both tiers, so the same source participant yields the same pseudonym in both.

Measured on the v3.1 adult trees:

| | count |
|---|---|
| `deid_bids_registered_04_14_26` subjects | 833 |
| `deid_bids_controlled_04_14_26` subjects | 767 |
| shared subject ids | 767 |
| controlled-only | **0** |
| registered-only | 66 |

This confirms the owner's account exactly: controlled ⊆ registered, 767 of 833, for this release and
cohort. FR-014's overlap counts are therefore directly computable.

**Caveat, and it is a real one**: session pseudonyms are *not* guaranteed to agree across trees.
`map_sequential_session_ids` (`dataset.py:1658-1730`) is called with the default `sequential=False`
(`dataset.py:1805`), so a session id is the source UUID truncated to 8 characters — deterministic and
content-based, which is good. But when two truncations collide *within a tree*, the colliding ids are
re-derived at 16 characters (`dataset.py:1716-1725`), and collision is evaluated over that tree's own
session set. A session could therefore be 8 characters in one published dataset and 16 in the other.
The code's own comment puts collision probability at ~1.2% for 10,000 8-character ids.

Measured: all 904 sessions of the 767 shared subjects are 8 characters in both trees, so no collision
fired in v3.1. The mechanism remains live for future releases.

Also measured: **2 of the 767 shared subjects have different session sets between the two trees.** So
recording-level agreement does not follow from participant-level agreement even where pseudonyms match.

**Implication for design**: participant overlap is a direct set operation. Recording overlap must match
on `(participant, session, task)` with length-tolerant session comparison (compare on the common
8-character prefix and flag any pair that agrees on the prefix but differs in length), and must report
the count of unmatched recordings rather than silently dropping them.

**Alternatives rejected**: requiring the private crosswalk (unnecessary, and it would put a
PHI-adjacent file in the generator's input path); assuming identity of session ids (would silently
undercount recording overlap the first time a collision fires).

---

## R2. What must senselab expose, and what can b2aiprep read itself?

**Decision**: b2aiprep reads every version itself via `importlib.metadata`. The senselab change is
narrow: report the models senselab selects internally.

**Evidence**: verified in the actual release environment, the `b2aiprep_test` conda env the release
scripts activate (Python 3.12.13). `importlib.metadata`
resolves every component the provenance graph needs:

| Component | Version in `b2aiprep_test` |
|---|---|
| `b2aiprep` | `3.1.0+51.g34174e7.dirty` |
| `senselab` | `1.3.0` |
| `opensmile` | `2.6.0` |
| `praat-parselmouth` | `0.4.7` |
| `torch` / `torchaudio` | `2.8.0` / `2.8.0` |
| `speech-articulatory-coding` | `0.1.0` |
| `ppgs` | `0.0.9` |
| `pyannote.audio` | `4.0.4` |
| `transformers` | `4.53.3` |
| `speechbrain` | `1.0.3` |
| `fairscape-cli` / `fairscape-models` | **absent** |

So the "senselab has no `__version__`" problem is not blocking: the installed distribution version is
discoverable from the caller, as are all six feature backends.

Three consequences of reading the real environment rather than assuming:

1. **The env installs `senselab 1.3.0`, not the `1.3.1a40` prerelease.** Development targets 1.3.0; the
   alpha is irrelevant unless the release env is deliberately moved.
2. **`b2aiprep` is installed from a dirty tree** (`3.1.0+51.g34174e7.dirty`). A `.dirty` version cannot
   resolve to a citable record, so FR-008 would fail against this environment as it stands today. The
   release env must be installed from a tagged commit before a PhysioNet-bound crate can be generated —
   and this is precisely the condition the generator should refuse on rather than paper over.
3. **`fairscape-cli` and `fairscape-models` are not installed**, so R3's dependency group is a real
   addition to this environment, not a version bump.

**Shipped data dictionaries corroborate the staleness risk.** Each feature dictionary hard-codes the
version of the library that produced its values — `ppgs 0.0.9`, `speech-articulatory-coding 0.1.0`,
`torchaudio 2.8.0` — and all three happen to match the env today, so nothing is currently wrong. But
they are string literals in package resources with `extraction_date` unset, checked against nothing. The
first backend upgrade makes every published dictionary silently wrong. Reading these from
`importlib.metadata` at bundle time removes the failure mode rather than deferring it.

Models b2aiprep constructs itself already carry their identity, and the immutable commit is one call
away: `HFModel.get_model_info().sha` (`senselab/src/senselab/utils/data_structures/model.py:93-100`).
That covers the transcription model (`prepare.py:339-341`) and the speaker-embedding model
(`prepare.py:321-323`).

The gap is `diarize_audios([audio_16k], device=device)` (`prepare.py:302-304`), called with **no**
model argument — senselab picks the default, and the explicit `HFModel` that would have named it is
commented out at `prepare.py:298-300`. b2aiprep cannot report what it did not choose.

**Design**: two paths, in order of preference.

1. **Preferred, no senselab change**: uncomment and pin the diarization model in b2aiprep, so all three
   models are constructed by the caller and reportable. This is a one-line change in code we own.
2. **Upstream, if (1) is undesirable**: senselab returns the resolved model identity alongside its
   results — the minimal version being that `diarize_audios` accepts and echoes an explicit model.

Either way this feature does not depend on a senselab release, which removes it from the critical path.

**Alternatives rejected**: a `senselab.provenance` module returning a full provenance dict (larger
upstream surface than needed, and blocks this feature on a senselab release); parsing versions from a
frozen environment file (drifts from what actually ran).

---

## R3. Which fairscape version, and does the generator use its CLI or its models?

**Decision**: depend on `fairscape-cli>=1.2.9` / `fairscape-models>=1.2.1` in a new optional
dependency group. Build the graph with the `fairscape_models` Python API, then invoke the
`fairscape-cli build` subcommands for the derived artifacts.

**Evidence** (established by running both, end to end, against the v3.1 bundle):

- `fairscape-models==1.0.24` accepts an `int` `contentSize`; `1.2.1` requires `str`. The 3.0.0 notebook
  emits `int` and therefore only runs under the older pin. Emitting `str` from the start makes the
  generator work on current releases.
- Under `fairscape-cli 1.2.9`, `build subcrate` on a crate directory produces
  `ro-crate-prov-graph.{json,html}`, `ro-crate-croissant.json`, `ro-crate-preview.html`, and
  `ro-crate-merkle-tree.json`; `build datasheet` adds `ro-crate-datasheet.html` and
  `ai_ready_score.json`. Together these reproduce all eight artifact names published for 3.0.0.
- Every one of those paths is `crate_path / "ro-crate-*"` (`fairscape_cli/utils/build_utils.py:181-327`),
  so **the crate metadata must sit at the root of the dataset it describes**. A separate `ro-crate/`
  directory is not a layout choice; it produces a crate whose relative paths are wrong.

**Alternatives rejected**: hand-writing the JSON-LD (loses datasheet, croissant, merkle, and AI-ready
score, all of which the current release publishes); pinning `1.0.24` to match the notebook (freezes us
on an unmaintained release to preserve a type error).

---

## R4. Where do run records live, and in what shape?

**Decision**: `<dataset_root>/provenance/`, with one JSON per invocation and one TSV of per-unit rows
per invocation. Records travel with the tree they describe.

```
<bids_or_bundle_root>/provenance/
  runs/<step>-<run_id>.json         # one per invocation: versions, models, config, inputs, outputs, counts, status
  units/<step>-<run_id>.tsv         # participant, session, task, family, outcome, reason
```

**Rationale**: one writer per file, so the 34 concurrent array tasks
(`adult_feature_extraction.sh --array=0-33`) never contend, and a re-run of one task replaces exactly
its own two files. At v3.1 scale this is ~70 files per release, not the per-recording fan-out an earlier
draft feared.

**Verified placement hazards** — these constrain the directory choice, and `provenance/` at the tree
root avoids all of them:

| Existing call site | Behaviour | Why `provenance/` is safe |
|---|---|---|
| `bids.py:53-64` `get_paths` | iterates only entries starting with `sub` | root-level dir invisible |
| `dataset.py:321` | `rglob("*/audio/*.json")` | would capture any JSON placed in an `audio/` dir — so none is |
| `dataset.py:618` | `glob("*.json")` per session | same constraint |
| `commands.py:371` | `copytree(bids/phenotype → bundle/phenotype)` | only the phenotype subtree |
| `commands.py:468` | `rglob("*.wav")` | extension-scoped |
| `commands.py:707`, `:752` | `glob("*.parquet")`, `phenotype.rglob("*.tsv")` | extension- and subtree-scoped |
| `commands.py:208`, `:215` | `copytree(source_bids → target_bids)` | records travel with the copy, which is intended; the reader keys on `run_id` so a copied record is not double-counted |
| `bundle_to_c2m2.py:34`, `controlled_to_c2m2.py:141` | bare `rglob('*')` | **does** see them — handled in R6 |
| `verify_sage_contents.py:106-118` | unfiltered recursion | **does** see them — handled in R6 |

**Alternatives rejected**: one JSONL line per recording carrying full run context (denormalized;
repeats versions on every line; and per-unit rows are needed only for outcome accounting, not
attribution); a single append-only file per step (concurrent writers); a database (no infrastructure for
one, and the records must travel with the data).

---

## R5. How is per-file provenance attributed when a release spans invocations?

**Decision**: attribute from the per-invocation records, and require the generator to verify that the
invocations contributing to one release agree. Refuse on a mixture it cannot attribute.

**Evidence**: each release recomputes all features into a fresh folder — `adult_feature_extraction.sh`
passes no `--update`, and no script in `v3.1/scripts/` does — so nothing is carried across releases. The
only within-release mixing arises when a failed run is resumed with `--update`, which merges newly
computed families into existing feature files (`prepare.py:189-286`).

That case is genuinely risky, because a retry often follows the environment change that fixed the
failure. It is also unrepresentable in the current artifact: the `.pt` has exactly one slot per config
key, and the `if not update:` guard (`prepare.py:288-294`) suppresses the write in update mode — which
is defensible there, since the config variables have by then been rewritten to describe what to *skip*
(`torch_config` becomes the literal `False` at `:238`), but is an accident on a fresh file, because the
guard tests the flag rather than whether a record was loaded.

**Design**: the run record makes this representable without touching the `.pt` format. Each invocation's
record names its own versions and effective configuration; the per-unit rows say which invocation
produced each `(recording, family)`. The generator then emits one activity per distinct
(versions, configuration) set and links each published file to the activities that contributed. If the
per-unit rows cannot resolve a family to a single activity, generation fails.

**Alternatives rejected**: fixing the `--update` guard alone (does not give multi-run provenance,
because the storage has one slot); forbidding heterogeneous releases outright (would block failure
recovery, which is a legitimate operational need).

---

## R6. Where does C2M2 generation live, and how do the new files avoid breaking existing tooling?

**Decision**: move the reusable logic into the installed package as `b2aiprep.metadata.c2m2`; leave the
existing `argparse` entry points in `external_scripts/c2m2/` as thin wrappers that import it. Invocation
stays a separate, explicitly ordered step.

**Rationale**: the scripts today can only run from their own directory, because they import `constants`
and `c2m2_mappings` as top-level modules (`external_scripts/c2m2/README.md:38-41`), and
`external_scripts/` is not part of the wheel — so an installed `b2aiprep-cli` cannot import them. C2M2
also spans both distributions and both cohorts, needs `cfde-c2m2` and Synapse credentials, and is
uploaded on its own cadence, so it cannot be folded into a per-dataset step.

**Required companion changes, all verified as present defects**:

- `c2m2_mappings.py:158-172` has no entry for the new file types, and lookups are `.get(suffix, '')`, so
  provenance files enumerated by the bare `rglob('*')` at `bundle_to_c2m2.py:34` would land in
  `file.tsv` with empty format and data type. Add `.json`/`.tsv` coverage for the provenance directory
  or exclude it explicitly — and record which was chosen, per FR-028.
- `verify_sage_contents.py:106-118` recurses unfiltered and reports any local file absent remotely.
  Either the provenance directory is uploaded or it is excluded there too; silence is not an option.
- Writers append with no de-duplication (`bundle_to_c2m2.py:497-533`, `fill_subject_files.py:146-152`),
  so a re-run doubles the submission. FR-019's no-accumulation requirement lands here.

**Alternatives rejected**: keeping the scripts byte-identical and shelling out with an explicit `cwd`
(works, but leaves the generator unable to reuse the mapping tables, and keeps the "run from this
directory" trap); moving them wholesale into the CLI (would make `cfde-c2m2` and `synapseclient` hard
dependencies of b2aiprep).

---

## R7. Command names

**Decision**: `b2aiprep-cli generate-release-metadata` and `b2aiprep-cli validate-release-metadata`.

**Rationale**: matches the established verb-first pattern in `cli.py` — `generate-audio-features`,
`validate-bundled-dataset`, `create-bundled-dataset`, `run-quality-control-on-audios`. Two commands
rather than one because the validator is independently useful: FR-022 requires it to run against an
already-published release, which is how 3.0.0 gets audited without regenerating anything.

**Alternatives rejected**: `build-metadata` (ambiguous about what is built — the confusion that "the
build" caused during specification); a single command with a `--check` flag (obscures that validation
has a different input contract: published metadata plus a file inventory, no run records required).

---

## R8. Release-level deposit to a citable archive

**Decision**: do not publish one now.

**Rationale**: FR-019 requires each dataset's metadata to be self-sufficient at its own root, and R3
shows the crate must live there regardless. A release-level deposit would therefore be purely additive,
and the spec already permits adding one later with no structural change. Nothing in the pipeline or in
either upload path depends on it.

**Alternatives rejected**: depositing the parent crate to Zenodo in this feature (adds an artifact to
maintain for no capability the two self-sufficient crates lack); putting it on Synapse (uses a data host
as a metadata patch venue).

---

## R9. Zenodo DOIs for the software

**Decision**: prerequisite, not scope. Register both repositories before the first release that cites
them.

**Rationale**: FR-008 requires a durable identifier that resolves to the exact version used, and it must
exist before publication because PhysioNet cannot be corrected afterwards. That forces an ordering:
tag and release senselab, then b2aiprep, then run the pipeline, then publish. Registration itself is an
administrative action on the `sensein` GitHub organization — outside this feature's automation. Neither
repository has a `.zenodo.json` today; senselab has no `CITATION.cff` either.

**Consequence for the generator**: it reads the concept and version DOIs from configuration and fails if
a PhysioNet-bound crate would cite an unresolvable identifier (FR-008 plus FR-021's fail-loud rule).
