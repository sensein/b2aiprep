# Creating C2M2 documentation

Generates a single C2M2 datapackage combining four Bridge2AI-Voice datasets:
adult/pediatric × registered/controlled access.

## Project structure

```
voice
├── adult                 (owns adult subjects + biosamples; tier-neutral)
│   ├── adult_registered  (registered-access adult files: features/metadata/phenotype)
│   └── adult_controlled  (controlled-access adult files: raw audio)
└── peds                  (owns pediatric subjects + biosamples; tier-neutral)
    ├── peds_registered   (registered-access pediatric files)
    └── peds_controlled   (controlled-access pediatric files: raw audio)
```

Subjects and biosamples are independent of access tier, so they live on the
`adult`/`peds` parent projects. Files carry the access tier via their leaf project.
A controlled raw-audio file links to its biosample through `file_describes_biosample`;
its public identifiers (`local_id`, `filename`) are built from the anonymous
biosample id and `persistent_id`/`access_url` are left empty, so no public path
connects a subject to a disease-bearing biosample.

The scripts are directory-agnostic — paths are CLI arguments. If a bundle's
internal layout changes substantially (different `features/`/`phenotype/`
structure, or new file types in the controlled deid bundle) the code may need
updating; routine path/version changes do not.

## Requirements

- The [`cfde-c2m2`](https://github.com/MaayanLab/cfde-c2m2) CLI (for `init`,
  `prepare`, `validate`, `package`).
- A Python environment with `pandas`, `pyarrow` (reads the registered feature
  parquets), and `synapseclient` (for `synapse_manifest.py` only).
- `SAGE_PAT` set in the environment for `synapse_manifest.py` (a Synapse personal
  access token); the other scripts need no credentials.
- **Run the scripts from this `c2m2/` directory** (or put it on `PYTHONPATH`) — they
  import `constants` / `c2m2_mappings` / `bundle_to_c2m2` as top-level modules, so the
  working directory must be where these files live. `--c2m2_id_map`, the output
  directory, and the manifests can be anywhere.

## Setup

Using the following [guide](https://github.com/MaayanLab/cfde-c2m2), create the
folder you want to generate C2M2 in, then run:

```
cfde-c2m2 init
python finalize.py /path/to/c2m2_output/
```

`finalize.py` copies the static files (id_namespace, project, project_in_project, dcc).

## Build order

Run all steps against the **same** `--c2m2_id_map` file — it is the persistent
`(participant_id, session_id) -> anon_id` crosswalk shared across every run. Run
each cohort's **registered** step before its **controlled** step, because the
registered run creates the biosamples the controlled files link to.

### 1. Registered access (subjects, biosamples, derived files)

```
# Adult
python fill_subject_files.py /adult/registered/bundle/ /path/to/c2m2_output/
python bundle_to_c2m2.py     /adult/registered/bundle/ /path/to/c2m2_output/ --c2m2_id_map /path/to/id_map.csv --physionet_version x.x.x

# Pediatric
python fill_subject_files.py /peds/registered/bundle/ /path/to/c2m2_output/ --peds
python bundle_to_c2m2.py     /peds/registered/bundle/ /path/to/c2m2_output/ --c2m2_id_map /path/to/id_map.csv --peds --physionet_version x.x.x
```

### 2. Controlled access (raw audio + dataset/phenotype files)

C2M2 requires at least one of sha256/md5 per file. Rather than hash ~100k audio
files locally, pull `md5`/`size` (and the relative `path` for non-recording files)
from the citation-versioned Synapse fileviews. `synapse_manifest.py` does this
read-only (it never modifies the views). It needs the Sage token, so run it in a
shell where `SAGE_PAT` is set (e.g. with a leading `!` in Claude Code):

```
python synapse_manifest.py --out_dir /path/to/manifests/
# writes synapse_manifest_adult.csv and synapse_manifest_peds.csv
# (columns: id, name, path, dataFileMD5Hex, dataFileSizeBytes)
```

The citation fileview synIds are baked in as read-only defaults (override with
`--adult_view`/`--peds_view`). Use `--create-temp-views` only if no citation view
exists — it creates and deletes a temporary view scoped to the cohort folder.

Then add the controlled files to the package. Each recording becomes an
anonymously-named file linked to its biosample; non-recording files (README,
dataset_description, phenotype TSV/JSON) are described with their real path, and
`phenotype/task/*` files are linked to biosamples exactly as the registered run
does (read locally from the bundle). `--crosswalk_out` emits a PRIVATE synId ↔ anon
mapping — never add it to the package.

```
# Adult
python controlled_to_c2m2.py /adult/controlled/bids/ /path/to/c2m2_output/ \
    --c2m2_id_map /path/to/id_map.csv \
    --manifest /path/to/manifests/synapse_manifest_adult.csv \
    --crosswalk_out /path/to/adult_controlled_crosswalk.csv

# Pediatric
python controlled_to_c2m2.py /peds/controlled/bids/ /path/to/c2m2_output/ --peds \
    --c2m2_id_map /path/to/id_map.csv \
    --manifest /path/to/manifests/synapse_manifest_peds.csv \
    --crosswalk_out /path/to/peds_controlled_crosswalk.csv
```

Manifest columns are auto-detected (`id`/`name`/`path`/`dataFileMD5Hex`/`dataFileSizeBytes`)
or overridable via `--name_col/--path_col/--md5_col/--size_col/--synid_col`. Without
`--manifest` the script walks the bundle (size via `stat`; `--hash` for md5, slow);
without a checksum the rows fail validation.

## Finalize the C2M2 packaging

Run the following within the C2M2 output directory:

```
cfde-c2m2 prepare
cfde-c2m2 validate
```

Fix any validation errors, then package:

```
cfde-c2m2 package -o /output/path/for/zip/2026_03_C2M2.zip
```

# Post Metadata Generation

- Upload to [data.cfde.cloud](https://data.cfde.cloud)
- Keep the `--c2m2_id_map` file and any `--crosswalk_out` files in a safe place for
  future runs. These are the only links back to real participant ids / Synapse
  objects and must never be included in the uploaded package.
