# Creating C2M2 documentation

Generates a single C2M2 datapackage combining four Bridge2AI-Voice datasets:
adult/pediatric × registered/controlled access.

## Project structure

```
root
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

### 2. Controlled access (raw audio files)

Provide a Sage manifest CSV so `md5` and `size_in_bytes` come from Synapse (C2M2
requires at least one of sha256/md5 per file; hashing ~100k audio files locally is
avoided). The manifest needs a path/name column plus md5/size columns; Synapse
ids and BIDS entities are auto-detected, or pass `--path_col/--md5_col/--size_col/--synid_col`.
Use `--crosswalk_out` to emit a PRIVATE synId ↔ anon mapping (never add it to the package).

```
# Adult
python controlled_to_c2m2.py /adult/controlled/bids/ /path/to/c2m2_output/ \
    --c2m2_id_map /path/to/id_map.csv \
    --manifest /path/to/adult_manifest.csv \
    --crosswalk_out /path/to/adult_controlled_crosswalk.csv

# Pediatric
python controlled_to_c2m2.py /peds/controlled/bids/ /path/to/c2m2_output/ --peds \
    --c2m2_id_map /path/to/id_map.csv \
    --manifest /path/to/peds_manifest.csv \
    --crosswalk_out /path/to/peds_controlled_crosswalk.csv
```

Without `--manifest`, the script walks the BIDS tree (size via `stat`); add `--hash`
to compute md5 by reading every file (slow). Without a checksum the rows fail validation.

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
