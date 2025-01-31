"""Implements functions for training AudioQC using a process similar to MRIQC."""

import pandas as pd


def get_features_df_with_site(features_csv_path, participants_tsv_path):
    features_df = pd.read_csv(features_csv_path)
    participants_df = pd.read_csv(participants_tsv_path, sep="\t")
    participant_to_site = dict(zip(participants_df["record_id"], participants_df["session_site"]))
    features_df["site"] = features_df["participant"].map(participant_to_site)
    # features_only_df = features_df.drop(columns=['site', 'participant', 'task'])  # Exclude non-feature columns
    return features_df


def site_wise_normalization(features_df, features, site_column="site", mode="both"):
    """
    Perform site-wise normalization using median and interquartile range (IQR).
    
    Args:
        features_df (pd.DataFrame): The dataset including site information.
        features (pd.DataFrame): The feature columns to normalize.
        site_column (str): The column representing site labels.
        mode (str): 'center', 'scale', or 'both' (default).
    
    Returns:
        pd.DataFrame: Normalized feature DataFrame.
    """
    normalized_features = features.copy()

    for site in features_df[site_column].unique():
        site_mask = features_df[site_column] == site
        site_data = features[site_mask]

        median = site_data.median()
        iqr = site_data.quantile(0.75) - site_data.quantile(0.25)  # Interquartile Range (IQR)

        if mode == "center":
            normalized_features.loc[site_mask] = site_data - median
        elif mode == "scale":
            normalized_features.loc[site_mask] = site_data / iqr
        elif mode == "both":
            normalized_features.loc[site_mask] = (site_data - median) / iqr

    return normalized_features


def site_predictive_dimensionality_reduction():
    return


def winnow():
    return


def add_labels(labels_csv=None):
    return


def svm_train():
    return


def rfc_train():
    return


def grid_search():
    return


def cross_validate():
    return


def inner_loop():
    return


def outer_loop():
    features_csv_path = "/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/bridge2ai-voice-corpus-3/derived/static_features.csv"
    participants_tsv_path = "/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/bridge2ai-voice-corpus-3/bids/bids/participants.tsv"
    get_features_df_with_site(
        features_csv_path=features_csv_path, participants_tsv_path=participants_tsv_path
    )

    return


"""
best_inner_loop_models = []
for sites:
    hold a site out
    best_fold_model = inner()
    best_inner_loop_models.append(best_fold_model)
best_best_inner_model = best(best_inner_loop_models)
final_model = cross_validation_across_sites(best_best_inner_model)
external_dataset_evaluation(final_model)

"""
