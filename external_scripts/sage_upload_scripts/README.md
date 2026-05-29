# Sage Upload Process

This file will detail the processes that take place to upload data to Sage Synapse repository.

## 0. Prerequisites

As a prerequisite we are assuming that the Sage projects have been generated. The adult and pediatric data now live in separate projects, each with a destination folder that the data is synced *into* (this folder is nested inside the project, not necessarily at its top level). The project IDs, folder IDs, and expected folder names are configured in `sage_config.py`; the scripts validate that they resolved the expected folder by name before touching any data.

Additionally, to run any of these codes, one needs to setup a personal access token with Sage. To do so, go to Account Settings on Sage and scroll down to Personal Access Tokens (PATs) and generate one. This token ideally would have all 3 permissions (View, Download, Modify). The code currently assumes that this PAT is saved as an environment variable called `SAGE_PAT`. There are additionally options for setting up a Sage profile as to not have to pass the environment variable every time but this is currently not configured.

Finally, run `pip install synapseclient` for the required python API.

## 1. Generate Sage manifest file

In order to do bulk uploads of data to the Sage Synapse repository, we use the functions `synapseutils.generate_sync_manifest` and `synapseutils.syncToSynapse`. We start by generating a Sage manifest using the command below:

```
python sage_generate_manifest.py \ 
    --bids_folder path/to/bids/folder \
    --manifest_file path/to/save/manifest.tsv \
    --adult
```

`--bids_folder` gives the path to a BIDS folder we want to upload. This should be the output of the deidentify command in `RELEASE.md` but without any features. 

`--manifest_file` describes where to save the generated manifest file and what to name it. 

`--adult` is a flag for whether this is the adult or the pediatric data which will determine the folder on Synapse to upload to. Leave off the flag for generating the pediatric data manifest.

## 2. Upload the data using the generated manifest

Taking the generated manifest file from step 1, we can then upload it using the below command. Since the manifest is already specific to the adult vs. the pediatric data, the upload command is the same for both/there is no flag to specify which dataset.

```
python sage_upload_manifest.py \
    --manifest_file path/to/manifest.tsv
```

`--manifest_file` should match the file generated from running the command in step one. With no other flags this uploads the entire manifest at once, which can take quite a long time.

The script can restrict which rows it uploads, which is how we parallelize:

- `--subject sub-XXXX` uploads only one subject's subtree. This is the preferred way to parallelize (see step 2b) because two subjects never write to the same Synapse folder.
- `--toplevel` uploads only the dataset-level files that live outside any `sub-*` directory (`dataset_description.json`, `README.md`, `audio_quality_metrics.*`, `phenotype/`, ...).
- `--start` / `--end` slice the (post-filter) manifest by row index.
- `--dry_run` validates the (filtered) manifest without uploading. This is safe and does **not** touch the local upload cache (it returns before any store), so it is a good pre-check.

Files whose MD5 already matches what is on Sage are skipped, so re-running an upload is safe and is the recommended way to recover from transient errors.

### 2b. Parallel per-subject upload (SLURM)

The old approach of splitting the manifest by `--start`/`--end` row ranges tended to fail with concurrent-upload errors because rows for the *same* folder landed in different jobs, which then raced to create that folder. Uploading **one subject per job** avoids this: the only shared parent is the dataset folder, which already exists.

`sage_upload_array.sbatch` is a SLURM array job that uploads one subject per array task. The partition is set to `pi_satra` (up to 48h); `ou_bcs_normal` (up to 24h) is an alternative. The per-task `--time` is generous for a single subject. Output goes to the repo `logs/` directory.

```
# 1. Generate the manifest once (step 1 above), e.g. to manifest.tsv
# 2. Build the subject list from the BIDS folder
ls -d path/to/bids/folder/sub-* | xargs -n1 basename > subjects.txt

# 3. (optional) validate one subject without uploading
python sage_upload_manifest.py --manifest_file manifest.tsv \
    --subject "$(head -1 subjects.txt)" --dry_run

# 4. Submit the array, sized to the subject count and capped at 20 concurrent (%20)
sbatch --array=0-$(($(wc -l < subjects.txt) - 1))%20 \
    sage_upload_array.sbatch manifest.tsv subjects.txt

# 5. After the array finishes, upload the dataset-level files once
python sage_upload_manifest.py --manifest_file manifest.tsv --toplevel
```

The `%20` caps how many tasks run at once so the slow uplink is not overwhelmed; tune it as needed. Because matching-MD5 files are skipped, failed array indices can simply be resubmitted.

Note: the array tasks `cd` into this script directory and rely on `load_dotenv()`, so a `.env` file containing `SAGE_PAT=...` must be present here (or `SAGE_PAT` must otherwise be available to the job).

## 3. Verify uploaded data

This command can run both as a check to verify the data uploaded properly as well as a verification between the data on Sage and a local folder, especially to make sure future changes to the code only affected specified files. This verification will check that the folder structures are equivalent and whether any extra are on Sage compared to locally or if any are missing on Sage. It also checks against files, mainly checking that filenames are the same and if they are that the contents (through and md5 hash) are the same.

```
python verify_sage_contents.py \
    --bids_folder path/to/bids/folder \
    --adult \
    --get_md5 \
    --sync
```

`--bids_folder` specifies the BIDS folder that we want to compare the current data on Sage against. This would typically be the folder that was used for the generation of the manifest file, but might also be a new BIDS folder to check that only certain data and structures have changed.

`--adult` is a flag to specify whether to check that the Sage folder to compare against is for the adult data. Leave off the flag for checking the pediatric data.

`--get_md5` is a flag to specify whether to generate MD5 hashes of the files in the BIDS folder. If specified, it will generate the hash for the local files and compare it against the corresponding Sage file's MD5 hash.

`--sync` controls whether to actually act on the differences. Without it (the default) the script does a **dry run**: it reports the differences between Sage and the local folder and, in particular, **calls out the files/folders that are on Sage but no longer present locally** so they can be manually reviewed before anything is removed. This acts on individual files and folders, not a bulk synchronize like the upload step.

Behavior for entities present on Sage but missing locally:

- **Dry run (default):** they are listed as "deletion candidates" only. Nothing is changed on Sage. Review the list, then re-run with `--sync` once you're satisfied.
- **`--sync`:** each such file/folder is **deleted** (moved to the project Trash). This is recoverable; the underlying content is not fully removed until the **Trash is purged**, which is done manually (e.g. for PII the file must be deleted here and then purged from the Trash).

Note: deleting a file does **not** scrub its metadata (name, md5, path) from any DOI'd file-view snapshot that already included it — that metadata remains in the published snapshot. The file content itself is removed once the Trash is purged. PII screening should therefore happen **before** a snapshot is taken / a DOI is minted.

This script does not re-upload file content. Files that changed locally (md5 mismatch) or that exist only locally are reported; re-run the upload manifest flow (steps 1–2) to push them to Sage.