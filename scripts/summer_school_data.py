"""
Organizes data, extracts features, and bundles everything together in an easily distributable
format for the Bridge2AI Summer School.
"""

from b2aiprep.process import (
    embed_speaker,
    verify_speaker, #maybe exclude?
    verify_speaker_from_files, #maybe exclude?
    specgram,
    melfilterbank,
    MFCC,
    resample_iir,
    extract_opensmile,
    VoiceConversion,
    SpeechToText
)

def retrieve_data():
    # TODO implement
    return


def organize_data():
    # TODO implement
    return


def extract_features():
    # TODO implement
    """
    - Current: 
         - opensmile (one vector per audio), 
         - spectrograms, 
         - mel filerbanks, 
         - mfccs, 
         - text transcripts
    - Can we detect if certain segments of voice contain identifiable information: https://github.com/sensein/b2aiprep/issues/51
    - Quality metrics (SNR, intelligibility)
    - Praat extraction
    - Some items from Nick feature table
    """
    return


def bundle_data(data, save_path):
    # TODO implement
    return


def main():
    retrieve_data()
    organize_data()
    extract_features()
    bundle_data()


if __name__ == "__main__":
    main()
