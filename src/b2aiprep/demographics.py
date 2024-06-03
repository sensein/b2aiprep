from pathlib import Path
from typing import Dict, Optional
import pandas as pd

import torch

def load_features_for_recordings(df: pd.DataFrame, data_path: Path, feature: Optional[str] = None) -> Dict[str, torch.Tensor]:
    output = {}
    feature_options = (
        'specgram', 'melfilterbank', 'mfcc', 'opensmile',
        'sample_rate', 'checksum', 'transcription'
    )
    if feature is not None:
        if feature not in feature_options:
            raise ValueError(f'Unrecognized feature {feature}. Options: {feature_options}')

    for recording_id in df['recording_id'].unique():
        output[recording_id] = torch.load(data_path / f"{recording_id}_features.pt")

        # if requested, we subselect to the given feature
        if feature is not None:
            output[recording_id] = output[recording_id][feature]

    return output
