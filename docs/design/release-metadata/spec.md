# Feature Specification: Pipeline-Generated Release Metadata

**Created**: 2026-08-13
**Status**: Draft
**Input**: Move RO-Crate and C2M2 publication metadata for Bridge2AI-Voice releases out of a hand-written notebook and standalone scripts and into the `b2aiprep` release pipeline, so that publication metadata is mechanically derived from the run that produced the data.

## Problem

Bridge2AI-Voice publishes two datasets per release: a feature-only dataset on **PhysioNet** (immutable once uploaded) and a deidentified audio dataset on **Sage/Synapse** (amendable, DOI minted manually). The `b2aiprep` pipeline builds both. Publication metadata is produced afterwards by other means: RO-Crate assets by a notebook in a separate repository (`b2ai-metadata-generation/0.1-alpha/ro-crate-generation/VOICE/generate_ro_crate.ipynb`), and the C2M2 catalogue submission by standalone scripts (`external_scripts/c2m2/`). `RELEASE.md` documents six pipeline steps and stops at `create-bundled-dataset`; it never mentions RO-Crate, C2M2, PhysioNet, Synapse, DOIs, or validation, and `grep` for `ro-crate|rocrate|c2m2|fairscape` over `src/` returns no hits outside unrelated docstrings. Publication metadata is therefore outside the pipeline entirely.

Two consequences, both verified.

**1. The pipeline computes facts and then throws them away.**

| Fact | Computed at | Fate |
| --- | --- | --- |
| Per-output example / participant / task counts, and `skipped(no-data)` status per published file | `src/b2aiprep/commands.py:333-336`, `:457-463`, `:615-621` | Formatted into one log line at `commands.py:623-637` and discarded |
| Effective feature configuration (`parselmouth_config`, `torch_config`, sample rate, duration) | `src/b2aiprep/prepare/prepare.py:175-187` | Written only when `--update` is absent (`prepare.py:288-294`). The v3.1 release scripts pass no `--update`, so these builds do record it — but the record is a side effect of a flag, and the `.pt` has exactly one slot per key, so a file merged across invocations cannot describe itself: whichever way the guard falls, one invocation's configuration is lost. The guard also keys off the flag rather than off whether an existing record was loaded (`prepare.py:139-145`), so a fresh file built with `--update` drops an accurate configuration for no reason |
| Installed PyTorch version, CUDA availability, GPU model | `prepare.py:120-126` | Logged, never persisted |
| Per-stage execution time | `prepare.py:163-165`, `:258-260`, `:308-310`, `:327-329`, `:348-350` | Logged, never persisted |
| Identity of the transcription and speaker-embedding models | `prepare.py:321-323`, `:339-341` | Instantiated with `revision="main"` — a mutable branch — and never written into the feature payload. The size actually used is also not the documented one: the CLI default and `RELEASE.md` example are `tiny` (`commands.py:157`), while the v3.1 script runs `-t "large-v3-turbo"`, so the model name a hand-written crate would carry is wrong even where it is present |
| Identity of the diarization model | `prepare.py:302-304` | Not specified at all; the intended explicit model is commented out at `prepare.py:298-300` |
| Why an individual item was skipped | `prepare.py:157-160`, `:311-314`, `:330-335`, `:351-354`; `bundle_data.py:49-53`, `:144-151`; `dataset.py:2157-2159`; `bids.py:71-72` | `None`, `continue`, or a bare `warning`. `bundle_data.py:145` guesses the reason in its own message ("likely due to sensitive") because the code does not know it |
| Count of records removed by each deidentification rule | `dataset.py:1963-1967`, `:1987-1991` | Logged, never persisted |
| Which feature families were withheld from which recording | `dataset.py:71-75` applied per recording at `dataset.py:2263-2268` | Not recorded anywhere |

The pipeline also already knows its own version (`src/b2aiprep/__init__.py:3`, via versioneer) and never records it. The library that computes the features exposes even less: `senselab` has no `__version__`, `extract_features_from_audios` (`senselab/src/senselab/audio/tasks/features_extraction/api.py:33-47`) returns numeric payloads only with no record of the effective parameters it resolved internally, and `HFModel.revision` stays `"main"` after validation even though the immutable commit hash was resolved and cached during validation and then dropped (`senselab/src/senselab/utils/data_structures/model.py:63`, `:232-239`; `senselab/src/senselab/utils/dependencies.py:300-313`).

**2. The hand-written metadata is measurably wrong.** The notebook reads only three things from the release: file sizes via `stat()`, inferred column names, and directory listings. Everything else — DOI, dates, licence and access URLs, publisher, funder, participant count, total size, the 117-name author list, every entity identifier, and the entire provenance graph — is typed into notebook cells, against a hard-coded root path on another machine (`cell 3 line 2`, `/mnt/data/b2ai-voice/physionet.org/files/...`, with alternates for this cluster commented out in cells 37-46). A `calculate_md5` helper is defined and then commented out at all eleven of its call sites, so the notebook computes no checksums. In the resulting published 3.0.0 RO-Crate (`ro-crate/voice-ro-crate-assets 2/ro-crate-metadata.json`, 75 graph entities, 1434 property values):

- **286 property values are empty** across **72 of 75** entities. Includes `hasPart: []` on the root crate (so a crate declaring `conformsTo: ro/crate/1.2` lists none of its parts), `EVI#inputs: []`, `usedMLModel: []` and `usedDataset: []` on both computation activities, `prov:wasDerivedFrom: []` and `derivedFrom: []` on all 15 dataset entities, `format: ""` on all 9 feature parquets, and `completeness: ""` and `irbProtocolId: ""` on the root.
- **Two entities carry the wrong filename.** `b2ai-voice-dataset-feature-sparc-periodicity` is named `sparc_loudness.parquet` while pointing at `sparc_periodicity.parquet`; `b2ai-voice-dataset-feature-torchaudio-pitch` is named `torchaudio_spectrogram.parquet` while pointing at `torchaudio_pitch.parquet`. Both errors are duplicated into the `generated` array of the features computation, so the mistake propagates.
- **Two identifiers are used twice.** `ark:59853/b2ai-voice-schema-phenotype-confounders` names both the confounders and the demographics schema; `ark:59853/b2ai-voice-schema-phenotype-voice-perception` appears twice. In JSON-LD one of each pair is unresolvable. In the notebook (`cell 21 lines 8-13`) the demographics schema is inferred from `confounders.tsv` and given the confounders schema's identifier — so the surviving node describes the wrong table.
- **Four of fifteen described datasets describe no file.** `phenotype-diagnosis`, `-enrollment`, `-questionnaire`, and `-task` all carry `contentUrl: []` and no digest, because the notebook globs paths missing the `phenotype/` segment (`cell 30 line 4`, `cell 32 line 3`, `cell 33 line 3`, `cell 34 line 3`) and `glob` on a nonexistent directory returns empty without error. Those four nodes stand for 42 of the release's 44 phenotype tables.
- **Six of fifteen datasets link to no schema at all.** No dataset entity carries `dataSchema`; nine carry `evi:Schema`. Confounders and demographics have neither, because the notebook sets `dataSchema` twice in the same dict literal with `None` second (`cell 27` lines 15 and 24; `cell 28` lines 17 and 22), so the schemas it generated for them dangle unreferenced.
- **No person in the release is identifiable.** 117 authors are bare name strings; the string `orcid` does not appear in the notebook or in any published artifact. The funder is one free-text string containing a visibly malformed award identifier (`Award #3Tf-OTOD03272001S2`) that nothing validates.
- **The declared total size is wrong and in the wrong unit.** The root asserts `contentSize: "12.9 GB"` as prose while its children carry integer bytes summing to 13,789,023,450 (12.84 GiB) — and the children omit `static_features.tsv` (79.8 MB in a real build) entirely.
- **Dates are inconsistent and one is impossible.** `datePublished` is `12/16/2025` everywhere except `sparc_pitch`, which says `08/18/2025`. The features computation has `dateCreated: 01/29/2026` — six weeks *after* the publication date of the files it claims to have generated.
- **File coverage is 11%.** Eleven of the fifteen dataset entities carry a digest, and the Merkle tree (`ro-crate-merkle-tree.json`) has 11 leaves: 9 parquets plus 2 of 44 phenotype TSVs. A real build of the same release shape (`.../post_3.0/data/adult/bundled_12_21_25`) contains 98 files. `static_features.tsv` has a declared schema entity (`b2ai-voice-schema-static-features`, 135 properties) but no dataset entity and no digest. All 44 JSON data dictionaries are absent. The Croissant export declares 15 distributions against 9 record sets, 4 of them with empty `contentUrl`.
- **The provenance graph names one piece of software.** `b2ai-voice-software-b2aiprep`, `version: "3.0.2"` hand-typed (the repository is now at `3.3.3-71-g2ca4f42`), `format: ".py"`, identified by `https://github.com/sensein/b2aiprep` — a mutable repository URL, not a durable identifier. `senselab`, openSMILE, Praat/parselmouth, torchaudio, SPARC, PPGS, Whisper, and SpeechBrain ECAPA appear **only in prose** (`rai:machineAnnotationTools`), not as graph entities. The human agent is recorded as `runBy: "Alastair"` — a bare first name, spelled differently from `Alistair Johnson` in the same document's author list.
- **The exported evidence graph covers one file** (`ro-crate-prov-graph.json` describes `ppgs.parquet` only) and ships the placeholder string `"a datafile description"`.
- **Existing quality scoring cannot see any of this.** `ai_ready_score.json` reports `has_content: true` on all 13 dimensions for this crate, including `traceable` and `interpretable`, on the strength of counting 2 computations and 1 software entity.

**3. The data dictionaries shipped inside the release have the same class of defect.** `src/b2aiprep/prepare/resources/feature_schemas/*.json` are copied into every bundle (`commands.py:602-607`). Two of nine name a column that does not exist: `ppgs.json` declares field `ppg` where the parquet column is `ppgs`, and `torchaudio_spectrogram.json` declares `spectrograms` where the column is `spectrogram` (verified against `bundled_12_21_25/features/*.parquet`). Every one carries a hand-typed provenance block with a frozen version string (`torchaudio` `"2.8.0"` at `torchaudio_mfcc.json:47`; `speech-articulatory-coding` `"0.1.0"` at `sparc_ema.json:59`; `ppgs` `"0.0.9"` at `ppgs.json:79`) and `"extraction_date": null`, none of which is checked against what is installed. Two different papers are cited by the *same* durable identifier: `arxiv.org/abs/2406.12998` is given for "High-Fidelity Neural Phonetic Posteriorgrams" (`ppgs.json:87`) and for "Coding Speech through Vocal Tract Kinematics" (`sparc_ema.json:66-67`). One of those citations is necessarily wrong. `static_features.json`, documenting 135 openSMILE / Praat / SQUIM columns, carries no provenance block at all — no openSMILE version, no feature-set name.

**4. The catalogue and upload paths have the same defects, plus their own.** The C2M2 submission is six manual phases and roughly ten hand-typed commands (`external_scripts/c2m2/README.md:48-138`), ordered only by prose: registered before controlled, per cohort, from inside the `c2m2/` directory, with the same identifier map passed to every invocation and validation errors fixed by hand before packaging. Every writer appends with no de-duplication (`bundle_to_c2m2.py:497-533`, `fill_subject_files.py:146-152`), so a re-run silently doubles the submission. It emits nothing about provenance: `assay_type` and `analysis_type` are `None` unconditionally in both writers (`bundle_to_c2m2.py:488-489`, `controlled_to_c2m2.py:93-94`), no software or analysis tables are produced, and `persistent_id` and `creation_time` are `None` for every file and biosample. Several values are silently empty or dropped: `.wav` and `.parquet` map to `''` for both format and data type (`c2m2_mappings.py:158-172`), so every audio row and every feature file is untyped; `peds_condition_to_DOID = {}` (`c2m2_mappings.py:153`), so the entire pediatric cohort contributes zero disease rows; seven adult conditions map to `[]` and are dropped with a print (`c2m2_mappings.py:132-145`); unmapped task names produce a file row with no biosample link (`bundle_to_c2m2.py:115-118`, `controlled_to_c2m2.py:248-251`); and PhysioNet access URLs are templated against the adult content slug regardless of cohort (`bundle_to_c2m2.py:468-469`), so pediatric URLs are wrong even though `static_files/project.tsv` records the correct pediatric location. On the Sage side, no annotations are attached to any uploaded file — the manifest carries only `path` and `parent` (`sage_generate_manifest.py:44-49`) — so nothing on Synapse records participant, session, task, cohort, release version, or the run that produced the file, and the C2M2 controlled writer has to recover all of it by regex on filenames. Verification is advisory: `verify_sage_contents.py` compares digests only when `--get_md5` is passed, logs mismatches at info level, and exits successfully either way; it never reads the manifest, so it cannot detect rows that were never uploaded. Its documented SLURM array step cannot be run as written — `README.md:54,66-67` invokes `sage_upload_array.sbatch`, and no `.sbatch` file exists anywhere in the repository.

**The organizing idea of this feature**: the pipeline emits a **release record** as a first-class output — a single accumulated account of what it read, what it computed, with what software and models and configuration, what it excluded and why, and what it wrote. Every publication artifact (RO-Crate for either dataset, C2M2 submission, data dictionaries) becomes a projection of that record. No publication value is authored by a human for a specific release, and any fact the record does not contain stops the release instead of appearing as an empty string.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Publication metadata for the feature dataset, from one command (Priority: P1)

A release engineer has just finished building the feature-only bundle for a release. They run one command against that bundle and receive the complete descriptive and structural metadata for the PhysioNet submission: every file in the bundle enumerated with its size, digest, media type, and role; every feature family's own participant, session, and recording counts; every column of every published table described; the licence, access conditions, and citation facts for this specific dataset and host. They type no values and edit no files afterwards.

**Why this priority**: It replaces the largest and most error-prone artifact — the RO-Crate for the immutable PhysioNet dataset, where a mistake requires an entire new version to correct. Everything it needs is recoverable from the built bundle, so it can ship before any change to the extraction path. It converts the 286 empty values and 11%-of-files coverage of 3.0.0 into a measured outcome.

**Independent Test**: Build (or take) a bundled feature dataset, run the command, and check the output against the bundle on disk: the set of described files equals the set of files present, sizes and digests match, counts match a direct count of the parquet contents, and no property is empty. Delivers a submittable metadata package for PhysioNet with no further human input.

**Acceptance Scenarios**:

1. **Given** a built feature-only bundle, **When** the engineer runs the metadata command naming the release version, **Then** the output describes every file in the bundle exactly once with its own byte size, content digest, and media type, and the described-file set differs from the on-disk set by zero entries.
2. **Given** the same bundle, **When** the engineer inspects the output, **Then** every declared file name agrees with the location it points to, no identifier is used twice, and no property holds an empty string, empty list, or placeholder.
3. **Given** the same bundle, **When** the engineer compares the declared aggregate size and participant count against the sum and cardinality of the described items, **Then** they agree exactly and are expressed in machine-comparable units.
4. **Given** a bundle in which one feature family produced no rows, **When** the command runs, **Then** the output records that family as produced-and-empty with the reason, rather than omitting it silently.

---

### User Story 2 - Provenance captured while it still exists (Priority: P2)

Feature extraction runs as a SLURM array over subject-split files — 34 tasks for the v3.1 adult cohort, each walking its list sequentially. A release builds every feature from scratch in a fresh folder, but a run that fails partway is resumed with `--update`, which skips recordings already complete; if the environment changed between the two attempts, one release's feature files straddle two versions. As each task runs, the pipeline records what only it can know: the versions of `b2aiprep`, `senselab`, and each feature backend as actually installed; the immutable revision of every model loaded; the configuration effectively used for each family in *this* invocation; and, for every recording that yielded no value, which of the possible reasons applied. When the release is later assembled, this history is carried into the publication metadata, so the provenance graph names every contributing software component and model with an observed version and a durable identifier.

**Why this priority**: These are the facts the request identifies as unrecoverable after the fact, and the ones the 3.0.0 crate is most wrong about (one software entity, a hand-typed version, mutable identifiers, no models at all). It is second because Story 1 is useful without it, and because it touches the highest-fan-out part of the pipeline.

**Independent Test**: Run extraction over a small subject set, then over the same set again with `--update` and a changed configuration. Inspect the accumulated record: it must name both invocations, the versions installed at each, the family each produced, and the configuration each used. No SLURM-scale run is needed to test the recording behaviour.

**Acceptance Scenarios**:

1. **Given** an extraction run, **When** the accumulated record is inspected, **Then** it names every software component that contributed a published value, each with the version installed at run time, and each resolvable through a durable identifier rather than a mutable branch or repository URL.
2. **Given** an extraction run that loads models, **When** the record is inspected, **Then** each model is recorded with an immutable revision, and no model appears with a branch name.
3. **Given** a release whose feature files were produced by two invocations under different tool versions, **When** the metadata is generated, **Then** each published feature family is attributed to the invocation that actually produced it, with that invocation's versions and effective configuration.
4. **Given** a recording whose feature computation failed, **When** the record is inspected, **Then** the failure is recorded against that recording and family with a reason, and is distinguishable from a deliberate exclusion.

---

### User Story 3 - A release cannot ship with a gap (Priority: P3)

Before anything is uploaded, the engineer runs a completeness check. It either passes or it names, in one report, every fact that is missing, empty, inconsistent, contradicted by the data, or unaccounted for. Nothing partial is written. The same check can be pointed at an already-published release to audit it without rebuilding.

**Why this priority**: This is the request's explicit success condition — a missing fact must fail loudly. It is what makes Story 1's output trustworthy, and it is independently valuable immediately: pointed at 3.0.0 it must find the defects listed above, which today's `ai_ready_score.json` reports as fully satisfied.

**Independent Test**: Run the check against the published 3.0.0 metadata; it must report each catalogued defect. Run it against freshly generated metadata for a build; it must pass. Remove one recorded fact and re-run; it must fail and name that fact.

**Acceptance Scenarios**:

1. **Given** metadata with any empty, placeholder, or self-inconsistent value, **When** the check runs, **Then** it exits unsuccessfully, names every violation with the item concerned, and no metadata artifact is written or overwritten.
2. **Given** the already-published 3.0.0 metadata and its dataset inventory, **When** the check runs against them, **Then** it reports the duplicated identifiers, the mismatched names, the unit-inconsistent total size, the impossible computation date, and the undescribed files.
3. **Given** a dataset whose data dictionary names a column the published file does not contain, **When** the check runs, **Then** it fails and names the column and the file.
4. **Given** a complete and consistent release, **When** the check runs twice with no change to the dataset, **Then** it passes both times and the generated metadata is byte-identical between runs.

---

### User Story 4 - Both published datasets described, and their relationship stated as measured (Priority: P4)

The engineer generates metadata for the deidentified audio dataset hosted on Sage/Synapse as well, and receives a reconciliation of the two published datasets for this release: how many participants and recordings are in both, in the feature dataset only, and in the audio dataset only, and for recordings present in the feature dataset, which feature families were published and which withheld. The audio dataset's own access tier, licence, and host facts are recorded separately from PhysioNet's, and a DOI that has not yet been minted is represented as pending rather than as an empty value or a guess.

**Why this priority**: The participant-set difference between the two datasets originates in separate RedCap exports per access tier, upstream of the pipeline, and the observed subset relationship is not guaranteed and may differ between the adult and pediatric cohorts. Publishing that relationship as a measured count rather than an assumption is what makes the two datasets citable together. It follows Stories 1-3 because it needs the same generation and checking machinery.

**Independent Test**: Run against a matched pair of built datasets for one release and check the reconciliation counts against direct counts taken from both datasets. Delivers the Synapse-side metadata package and a defensible statement of coverage.

**Acceptance Scenarios**:

1. **Given** the feature dataset and the deidentified audio dataset for one release, **When** metadata is generated, **Then** each is described as its own published dataset with its own host, access conditions, licence, and identifier, and neither inherits the other's.
2. **Given** the same pair, **When** the reconciliation is inspected, **Then** it states observed participant and recording overlap counts in all three directions, and asserts no rule about which is a subset of which.
3. **Given** a recording present in the feature dataset with some families withheld, **When** the reconciliation is inspected, **Then** the withheld families are named for that recording.
4. **Given** an audio dataset whose DOI has not been minted, **When** metadata is generated, **Then** the identifier is marked pending with no fabricated value, and the completeness check treats it as a known-pending fact rather than a gap.

---

### User Story 5 - Catalogue submission from the same facts (Priority: P5)

The engineer produces the C2M2 submission for the CFDE catalogue for both datasets from the same release record, with no separate manual sequence and no values re-entered. Any fact that appears in both the RO-Crate and the C2M2 submission is identical in both.

**Why this priority**: It removes the last manual metadata path, but the catalogue submission is amendable and downstream of the dataset descriptions, so it is the least urgent.

**Independent Test**: Generate both the RO-Crate and the C2M2 submission for one release and cross-compare every shared fact; all must agree. Delivers a catalogue submission with no manual step.

**Acceptance Scenarios**:

1. **Given** a release record, **When** both the RO-Crate and the C2M2 submission are generated, **Then** every fact present in both is identical, and the file inventory in both covers the same set of files.
2. **Given** a release with two published datasets, **When** the C2M2 submission is generated, **Then** each dataset appears with its own access tier and host, derived rather than typed.

---

### Edge Cases

- A feature family produces zero non-missing rows across the whole release, so no published file exists for it — the metadata must say the family was attempted and empty, not omit it (today `commands.py:532-544` and `:569-581` record this as `skipped(no-data)` and then discard it).
- A recording's audio fails to load, so no feature file exists at all (`prepare.py:157-160` returns `None`), versus a recording whose feature file exists but whose family value is all-missing (`bundle_data.py:149-151`), versus a recording deliberately excluded for sensitivity — three different reasons that must remain distinguishable.
- A file in the source tree does not follow the BIDS naming convention and is silently dropped from every traversal (`bids.py:71-72`); it must be accounted for rather than vanish.
- A recording's sidecar metadata file is missing, so deidentification skips it (`dataset.py:2157-2159`).
- A release's feature files were produced across several invocations under different installed versions, and one family was recomputed while others were preserved (`prepare.py:189-286`).
- The same release is generated for the adult and pediatric cohorts, where the relationship between the two published participant sets may differ.
- Metadata is regenerated for a dataset already uploaded to PhysioNet, where the dataset cannot be changed — the difference must be visible as a diff against what was published.
- A published fact is legitimately not yet available (an unminted Synapse DOI) and must be distinguishable from a fact that is missing by mistake.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: A single invocation MUST produce the complete publication metadata for a named release and a named published dataset, taking as input only the built dataset and the pipeline's own recorded history of the runs that produced it. The resulting artifact MUST contain no value authored by a human for this release. The accumulated run record is an internal input, not a published artifact: its facts appear in the RO-Crate and C2M2 outputs, but the record itself is never uploaded to PhysioNet or Synapse, since it carries operator identities, cluster job identifiers, and working paths that no consumer needs and that PhysioNet could not later correct.
- **FR-002**: Every file present in the published dataset MUST be described exactly once, each with its own byte size, content digest, media type, and role. The set of described locations MUST equal the set of files present, with no file undescribed and no described file absent.
- **FR-003**: No described item MUST carry an empty string, empty collection, null, or placeholder value for a property it declares. A property that does not apply MUST be absent rather than present-and-empty.
- **FR-004**: Within one metadata artifact, every identifier MUST be unique, and every item's declared name MUST agree with the location it points to.
- **FR-005**: Every aggregate figure (total size, participant count, session count, recording count, per-family row count) MUST equal the corresponding sum or cardinality over the described items, and MUST be expressed in a machine-comparable unit rather than prose.
- **FR-006**: All dates MUST be expressed in a single unambiguous calendar format, and no recorded date MUST fall outside the interval bounded by the run that produced the thing it describes and the release's publication.
- **FR-007**: Each data dictionary shipped with a published dataset MUST name exactly the columns present in the file it documents, and MUST declare each column's actual type.
- **FR-008**: The provenance record MUST name every software component that contributed to a published value — the release pipeline, the feature-extraction library (`senselab`), and each feature family's backend — each with the version actually installed during the run that produced that value, and each addressable by a durable identifier that resolves to that exact version rather than to a mutable branch, tag, or repository location.
- **FR-009**: Every model loaded during a run MUST be recorded with an immutable revision. No model MUST be recorded by a mutable reference.
- **FR-010**: The configuration effectively used to compute each published feature family MUST be recorded and attributed to the invocation that produced that family's values, including when a release's feature files are accumulated across multiple invocations. Every release recomputes its features from scratch, so no value is carried forward from a prior release and unknown provenance does not arise; the accumulation that must be handled is within one release, where a failed run is resumed with `--update`. Metadata generation MUST therefore verify that the invocations contributing to a release agree on software versions and effective configuration, and MUST refuse when it finds a mixture it cannot attribute to specific published values.
- **FR-011**: Per-recording records exist to make absence accountable, not to attribute versions — attribution is satisfied by FR-010's per-invocation record. For every combination of source recording and published feature family that yields no published value, the pipeline MUST record a reason, drawn from a closed set that distinguishes at least: participant not in this dataset's access tier, participant excluded, recording removed as sensitive, feature family withheld for this recording, task not selected for release, computation failed, and value entirely missing. The count of published values plus the count of recorded reasons MUST equal the total number of recording-by-family combinations in the source.
- **FR-012**: Each published feature family MUST state its own participant, session, and recording counts, because these differ between families within one release.
- **FR-013**: Each published dataset MUST be described with its own host, access conditions, licence, citation, and identifier, and MUST NOT inherit any of these from the other dataset in the same release.
- **FR-014**: For a release publishing both datasets, the metadata MUST state the observed participant and recording overlap between them in all three directions (both, feature-only, audio-only) as counted values, and MUST NOT assert a containment relationship as a rule.
- **FR-015**: A fact that is legitimately not yet available MUST be representable as explicitly pending, distinguishable from a fact that is missing in error, and MUST NOT be represented by an empty or invented value.
- **FR-016**: The catalogue (C2M2) submission and the RO-Crate for the same release MUST be derived from the same recorded facts, and every fact appearing in both MUST be identical in both.
- **FR-017**: Where a published value requires a term from a controlled vocabulary, an absent mapping MUST fail the release, naming the unmapped source value. No item MUST be dropped, and no vocabulary field left empty, because a mapping is missing.
- **FR-018**: Every named contributor and organisation MUST be recorded in a field capable of holding a resolvable identifier, populated wherever one is known, and the metadata MUST state how many contributors lack one rather than leaving the distinction invisible. Every award or grant identifier MUST be recorded as a structured value that can be checked, not as free text.
- **FR-019**: Generating metadata twice from an unchanged dataset and an unchanged run history MUST produce identical output, and re-running any generation step MUST NOT duplicate or accumulate content — so that a difference against what was published is a reliable correction signal for an immutable host.
- **FR-020**: The metadata MUST be sufficient to verify an uploaded copy of a published dataset against the release record file by file, and a verification run that finds any missing file or digest mismatch MUST report failure.
- **FR-021**: Metadata generation MUST fail with a non-zero result when any of FR-002 through FR-020 is unsatisfied, MUST report every unsatisfied item in one pass naming the item and the artifact concerned, and MUST NOT write or overwrite any metadata artifact on failure.
- **FR-022**: The completeness check MUST be runnable against an already-published release's metadata and file inventory without regenerating either.
- **FR-023**: Run provenance MUST be recorded per invocation rather than per recording, and concurrent array tasks MUST be able to record without coordinating with one another.
- **FR-024**: The documented release procedure (`RELEASE.md`) MUST include the metadata generation and completeness steps, and MUST agree with the scripts actually used to build a release — today it documents `generate-audio-features --update` and a `tiny` transcription model, while the v3.1 scripts pass no `--update` and use `large-v3-turbo`. Where doc and script disagree, the script is the reference.
- **FR-025**: No run record or published metadata artifact may contain participant identifiers in the source space, free-text response values, or transcribed speech. Records describing a deidentified distribution MUST use only deidentified identifiers and MUST NOT inherit records written in the pre-deidentification identifier space. Coverage differences between access tiers MUST be expressed as cohort-level counts, never as a per-participant statement of which consent applies to whom.
- **FR-026**: Files mapping deidentified identifiers back to source identifiers MUST remain outside every published dataset, and their absence MUST be verified before a dataset is uploaded.
- **FR-027**: Introducing run records and metadata artifacts MUST NOT change the behaviour or output of any existing pipeline step, validator, or upload tool, and MUST NOT change the address of any published file.
- **FR-028**: Where an existing process enumerates files indiscriminately, the new files MUST be handled deliberately — either described with a correct type or excluded explicitly — and the choice MUST be recorded rather than left implicit.

### Key Entities

- **Release**: A named version published to one or more hosts, with a cohort (adult or pediatric) and an access tier.
- **Published Dataset**: One artifact set delivered to one host — the feature-only dataset on PhysioNet, or the deidentified audio dataset on Sage/Synapse. Carries its own host, mutability, access conditions, licence, citation, and identifier.
- **Published File**: One file inside a published dataset, with a location, size, digest, media type, role, and the computation that produced it.
- **Feature Family**: A group of derived values computed by one backend (openSMILE, Praat/parselmouth, torchaudio, torchaudio-SQUIM, SPARC, PPGS) and published as one file or one column group, with its own coverage counts and its own effective configuration.
- **Run**: One invocation of a pipeline step, with its start and end, its inputs, the software and model versions in effect, its effective configuration, and the items it produced or skipped. A release accumulates many.
- **Software Component**: A named piece of software that contributed to a published value, with an observed version and a durable identifier resolving to that version.
- **Model**: A named model loaded during a run, with an immutable revision.
- **Exclusion**: A recording-and-family combination that yielded no published value, with a reason from a closed set and the step that decided it.
- **Coverage**: Counted participants, sessions, and recordings for a published dataset or feature family, and the measured overlap between the two published datasets of one release.
- **Data Dictionary**: The column-level description shipped alongside a published table, with each column's name, type, meaning, and the software provenance of its values.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every value in a release's publication metadata is traceable to the dataset or to a recorded pipeline run. Baseline: the notebook derives three things from the release (file sizes, column names, directory listings); the DOI, dates, licence and access URLs, publisher, funder, participant count, total size, 117-name author list, every entity identifier, and the whole provenance graph are typed into its cells.
- **SC-002**: 100% of the files in a published dataset are described with an individual size and digest. Baseline for 3.0.0: 15 of ~98 files described and 11 digested — and the generator itself computes no digests, its checksum helper being commented out at all eleven call sites; the 11 digests present come from a downstream tool.
- **SC-003**: Zero empty, placeholder, or self-contradictory values in a release's metadata. Baseline for 3.0.0: 286 of 1434 property values empty, affecting 72 of 75 entities; 4 of 15 described datasets pointing at no file; 6 of 15 linked to no schema; 2 duplicated identifiers; 2 filename/location mismatches; 1 aggregate size in a mismatched unit; 1 computation dated after the publication it produced; and the placeholder strings `"A Dataset description"` and `"a datafile description"` shipped on all 15.
- **SC-004**: Every data dictionary shipped in a release names exactly the columns of the file it documents. Baseline: 7 of 9 feature dictionaries agree; `ppgs.json` and `torchaudio_spectrogram.json` name a column that does not exist.
- **SC-005**: Every feature family published in a release is attributable in the provenance record to a named software component with an observed installed version and a durable identifier. Baseline: 1 software entity for the whole release, with a hand-typed version and a mutable repository URL; six backends and four models named only in prose; nine dictionary provenance blocks carrying frozen versions, `extraction_date: null`, and one identifier used for two different papers.
- **SC-006**: 100% of source recordings are accounted for in every published feature family, as either a published value or a recorded reason for absence. Baseline: unaccounted; skips are logged as warnings whose text guesses the reason.
- **SC-007**: A release with any metadata defect never reaches a publisher: the completeness check detects every defect catalogued for 3.0.0. Baseline: the current quality signal (`ai_ready_score.json`) reports all 13 dimensions satisfied for 3.0.0 despite every defect above.
- **SC-008**: The number of manual steps between a built dataset and a submittable metadata package for both hosts is one. Baseline: a 72-cell notebook in a separate repository, plus six C2M2 phases of roughly ten operator-supplied commands with a hand fix-up round, plus a three-step Sage sequence whose SLURM step references a job script that does not exist — none of it referenced in `RELEASE.md`.
- **SC-009**: Every contributor identifier present in a release's metadata resolves, the number of contributors without one is stated explicitly, and every award identifier is a structured value that validates. Baseline: 117 bare name strings, zero ORCIDs anywhere in the notebook or any published artifact, and one free-text funder string carrying a malformed award identifier.
- **SC-010**: Zero published items are dropped or left with an empty vocabulary term because a mapping is missing. Baseline in the C2M2 path: every `.wav` and `.parquet` row untyped, the whole pediatric cohort contributing no disease rows, seven adult conditions dropped with a print, and unmapped task names yielding files with no biosample link.
- **SC-011**: Regenerating a release's metadata without changing the dataset produces no difference, making a diff against the published version a reliable correction signal for an immutable host. Baseline: not reproducible; regeneration is a notebook execution.
- **SC-012**: For a release publishing both datasets, the participant and recording overlap between them is stated as counted values for every cohort published. Baseline: not stated in any published artifact.
- **SC-013**: An automated scan finds zero participant identifiers in the source space, free-text response values, transcripts, or per-participant consent statements in any run record or published artifact. Baseline: not scanned today.
- **SC-014**: Existing pipeline steps, the bundle validator, and the Sage upload verifier produce identical results before and after run records are introduced. Baseline: unverified.

## Assumptions

- The already-published 3.0.0 metadata is a validation target only. It is not regenerated, and no already-published file is moved, renamed, or altered.
- Each release recomputes every feature from scratch into a fresh folder; nothing is carried forward from a previous release. A run that fails partway is resumed with `--update`, which is the only way one release's feature files can span more than one invocation.
- Feature extraction is a SLURM array over subject-split files — 34 tasks for the v3.1 adult cohort, each processing its list sequentially — not a per-recording fan-out. Per-invocation records are therefore a handful of items per release.
- Where `RELEASE.md` and the release scripts under `post_3.0/v3.1/scripts/` disagree, the scripts describe what actually runs.
- Which features are computed, and which participants and recordings appear in each published dataset, do not change. Recording *why* an item is absent does not change *whether* it is absent.
- The participant-set difference between the feature dataset and the audio dataset originates in the separate RedCap export per access tier, upstream of the pipeline. The pipeline describes and reconciles that difference; it does not create or adjudicate it.
- The observed subset relationship between the two published participant sets is treated as an observation to be measured per release and cohort, not an invariant to be enforced.
- The pipeline can observe the versions of its own dependencies at run time, and can obtain an immutable revision for each model it loads. Where the feature-extraction library does not currently surface these (`senselab` exposes no version, returns no effective parameters, and discards the resolved model commit hash), obtaining them is in scope for this feature, whether by observing them in `b2aiprep` or by having `senselab` surface them.
- The C2M2 and Sage upload scripts under `external_scripts/` remain the reference for what the destinations require; this feature changes where those values come from, not what the destinations accept.
- A release is built once and published to hosts with different mutability. PhysioNet corrections require a new version; Synapse can be amended. Metadata generation is therefore expected to run before upload, and to be re-runnable for comparison afterwards.
- The existing controlled-vocabulary mappings (task-to-OBI, condition-to-DOID, file format terms) are assumed correct as far as they go. This feature assumes their gaps become visible failures rather than silent omissions; it does not assume responsibility for filling them.
- Deidentification and bundling are assumed to run on the same built dataset that is uploaded. The Sage upload currently takes the deidentified BIDS tree while PhysioNet takes the bundle, and nothing today cross-checks that the two came from the same run; this feature assumes both are described from one release record.

### Out of scope

Beyond the exclusions given in the request, the following are deliberately excluded:

- Correcting the substance of the narrative fields in the published metadata (bias, limitations, use cases, collection protocol, ethical review). These are human-authored prose that no pipeline run can derive; this feature requires only that they be sourced from a single reviewed place per release rather than retyped, and that their presence be checked.
- Minting DOIs, uploading to PhysioNet or Synapse, and submitting to the CFDE catalogue. Generation and verification are in scope; transport is not.
- Changing the deidentification rules, the exclusion configuration files, or the task selection lists. This feature records the effect of those rules; it does not alter them.
- Reconstructing provenance for releases built before this feature exists. It does not arise going forward, because each release recomputes its features from scratch rather than carrying values across releases; already-published releases are audit targets only.
- Re-deriving the participant-level clinical or demographic content of the phenotype tables.
- Expanding the controlled vocabularies themselves — adding EDAM terms for WAV and Parquet, DOID terms for the seven unmapped adult conditions and the empty pediatric map, or OBI terms for unmapped tasks. Making their absence fail the release is in scope (FR-017); curating the terms is separate domain work.
- Fixing the Sage upload and verification scripts' own operational defects (the missing SLURM job script, verification exiting successfully on a digest mismatch), except insofar as FR-020 requires verification against the release record to report failure.
