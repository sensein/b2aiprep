#!/bin/bash

python3 redcap_import.py /path/to/redcap/csv path/to/uuid/output/path --load_uuid_map path/to/preloaded/redcap/csv  columns_to_remove.json column_remap.json instrument_mapping.json value_transform.json free_text_col.json