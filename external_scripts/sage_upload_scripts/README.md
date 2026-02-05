# Sage Upload Process

This file will detail the processes that take place to upload data to Sage Synapse repository.

## 0. Prerequisites

As a prerequisite we are assuming that the main Sage project has been generated, that both the pediatric and the adult data are using the same project, and those folders have been generated and the proper synapse IDs are in each of the scripts.

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
    --manifest_file path/to/manifest.tsv \
    --start 0 \
    --end 1000
```

`--manifest_file` should match the file generated from running the command in step one. 

This script is setup for parallel uploads by providing a start and end of the data in the manifest to upload. If not specified this will upload all of the data at once which might take quite a long time.

Note: I have found this file will fail with a variety of errors, typically related to doing concurrent uploads. For that reason I typically run it multiple time because reportedly under the hood it will cache the MD5 of the files uploaded and shouldn't reupload if they match.

# 3. Verify uploaded data

This command can run both as a check to verify the data uploaded properly as well as a verification between the data on Sage and a local folder, especially to make sure future changes to the code only affected specified files. This verification will check that the folder structures are equivalent and whether any extra are on Sage compared to locally or if any are missing on Sage. It also checks against files, mainly checking that filenames are the same and if they are that the contents (through and md5 hash) are the same.

```
python verify_sage_contents.py \
    --bids_folder path/to/bids/folder \
    --adult \
    --get_md5 \
    --execute
```

`--bids_folder` specifies the BIDS folder that we want to compare the current data on Sage against. This would typically be the folder that was used for the generation of the manifest file, but might also be a new BIDS folder to check that only certain data and structures have changed.

`--adult` is a flag to specify whether to check that the Sage folder to compare against is for the adult data. Leave off the flag for checking the pediatric data.

`--get_md5` is a flag to specify whether to generate MD5 hashes of the files in the BIDS folder. If specified, it will generate the hash for the local files and compare it against the corresponding Sage file's MD5 hash.

`--sync` is an in progress command for determining whether to syncronize Sage based on the local folder. This would not be a bulk syncronize like the upload previously but do so for individual files and folders. Currently it is not configured to do anything and defaults to just doing a dry run check of the difference between the data datasets.