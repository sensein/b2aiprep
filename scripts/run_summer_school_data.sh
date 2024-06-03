#!/bin/bash
python3 b2aiprep/scripts/summer_school_data.py \
    --redcap_file_path /Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/bridge2ai-voice-corpus-1/bridge2ai_voice_data.csv\
    --audio_dir_path /Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/audio-data\
    --tar_file_path ./summer_school_data_test_bundle.tar\
    --bids_files_path /Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data-bids-like 