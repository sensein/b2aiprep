"""Shared Synapse project/folder configuration for the Sage upload scripts.

Each dataset is uploaded into a single destination folder that lives inside its
own Synapse project. ``folder`` is the sync target -- everything is uploaded
*inside* it -- and ``folder_name`` is the expected name of that folder, used to
sanity-check that we resolved the right destination before touching any data.

Note: the adult and pediatric data now live in separate Synapse projects.
"""

ADULT = {
    "project": "syn72370534",
    "folder": "syn72493850",
    "folder_name": "Dataset - Adult Data",
}

PEDS = {
    "project": "syn73617068",
    "folder": "syn72493849",
    "folder_name": "Dataset - Pediatric Data",
}


def get_dataset_config(adult):
    """Return the project/folder config for the adult or pediatric dataset."""
    return ADULT if adult else PEDS


def validate_sync_folder(folder, config):
    """Raise if the resolved Synapse folder is not the expected sync target."""
    if folder.name != config["folder_name"]:
        raise ValueError(
            f"Expected sync folder named {config['folder_name']!r} "
            f"(id {config['folder']}) but resolved {folder.name!r}. "
            "Refusing to proceed against the wrong folder."
        )
