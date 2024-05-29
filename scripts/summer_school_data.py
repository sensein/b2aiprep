"""
Organizes data, extracts features, and bundles everything together in an easily distributable
format for the Bridge2AI Summer School.
"""

import os
import argparse

from b2aiprep.process import (
    #embed_speaker,
    specgram,
    melfilterbank,
    MFCC,
    extract_opensmile,
    Audio,
)


def retrieve_data():
    # TODO implement
    return


def organize_data():
    # TODO implement
    return


def wav_to_features(wav_path):
    """
    Extracts features from a single audio file at specified path.
    """
    audio = Audio.from_file(str(wav_path))
    audio = audio.to_16khz()
    features = {}
    # features["speaker_embedding"] = embed_speaker(audio, model=) model TBD
    features["specgram"] = specgram(audio)
    features["melfilterbank"] = melfilterbank(features["specgram"])
    features["mfcc"] = MFCC(features["melfilterbank"])
    features["sample_rate"] = audio.sample_rate
    features["opensmile"] = extract_opensmile(audio)
    return features


def extract_subject_features(subject_path):
    """
    Extracts features for every activity audio file.
    Args:
    Returns:
        features_pt: extracted features
    """

    # for wav file in subject dir
        # extract audio features
        # determine save path
        # save features
        
    #data structure
    # if subject is not None:
    #     if task is not None:
    #         prefix = f"sub-{subject}_task-{task}_md5-{md5sum}"
    #     else:
    #         prefix = f"sub-{subject}_md5-{md5sum}"
    # else:
    #     prefix = Path(filename).stem
    # if outdir is None:
    #     outdir = Path(os.getcwd())
    # outfile = outdir / f"{prefix}_features.pt"
    # torch.save(features, outfile)
    pass


def bundle_data(data, save_path):
    # TODO implement
    return


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process audio data for multiple subjects.")
    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help='Path to data in BIDS-like format')
    return parser.parse_args()


def main():
    args = parse_arguments()
    for dir in os.listdir(args.data_path):
        # check if dir is a subject
        retrieve_data(args.data_path)
        organize_data()
        extract_subject_features()
        bundle_data()


if __name__ == "__main__":
    main()
