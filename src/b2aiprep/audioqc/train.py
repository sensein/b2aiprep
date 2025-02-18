"""Implements functions for training AudioQC using a process similar to MRIQC."""

import argparse
import json
import logging
import os
from datetime import datetime
from itertools import combinations, permutations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from b2aiprep.audioqc.save import save_model

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

MIN_FEATURES_TO_KEEP = 64  # in feature selection
DEBUG_MODE = True


def get_features_df_with_site(features_csv_path, participants_tsv_path):
    features_df = pd.read_csv(features_csv_path)
    participants_df = pd.read_csv(participants_tsv_path, sep="\t")
    participant_to_site = dict(zip(participants_df["record_id"], participants_df["session_site"]))
    features_df["site"] = features_df["participant"].map(participant_to_site)
    # features_only_df = features_df.drop(columns=['site', 'participant', 'task'])  # Exclude non-feature columns
    return features_df


def site_wise_normalization(features_df, site_column="site", mode="both"):
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
    non_feature_cols = ["label", "site", "participant", "task"]
    features_only = features_df.drop(columns=non_feature_cols)
    normalized_features = features_only.copy()

    for site in features_df[site_column].unique():
        site_mask = features_df[site_column] == site
        site_data = features_only[site_mask]

        median = site_data.median()
        iqr = site_data.quantile(0.75) - site_data.quantile(0.25)  # Interquartile Range (IQR)

        if mode == "center":
            normalized_features.loc[site_mask] = site_data - median
        elif mode == "scale":
            normalized_features.loc[site_mask] = site_data / iqr
        elif mode == "both":
            normalized_features.loc[site_mask] = (site_data - median) / iqr

    normalized_features[non_feature_cols] = features_df[non_feature_cols]

    return normalized_features


def site_predictability_feature_elimination(
    features_df, max_features_to_remove=100, min_features_to_keep=MIN_FEATURES_TO_KEEP
):
    """
    Implements Site-Predictability-Based Feature Elimination using ExtraTreesClassifier.

    This function iteratively removes the most predictive feature for site classification
    until either the classifier's performance is near chance level, a predefined
    maximum number of features is removed, or the minimum number of features to keep is reached.

    Args:
        features_df (pd.DataFrame): DataFrame containing features along with 'site', 'participant', and 'task' columns.
        max_features_to_remove (int, optional): Maximum number of features to remove. Default is 67.
        min_features_to_keep (int, optional): Minimum number of features to retain. Default is 64.

    Returns:
        tuple:
            - list: Removed features in order of elimination.
            - float: Final site prediction accuracy.
            - pd.DataFrame: DataFrame with remaining features after elimination.
    """
    # Drop non-AQM columns (site labels are the target variable)
    X = features_df.drop(columns=["site", "participant", "task", "label"])
    y = features_df["site"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train initial ExtraTreesClassifier to predict site labels
    site_predictor = ExtraTreesClassifier(n_estimators=100, random_state=42)
    site_predictor.fit(X_train, y_train)

    # Get initial accuracy (baseline)
    initial_accuracy = site_predictor.score(X_test, y_test)
    logger.info(f"Initial site prediction accuracy: {initial_accuracy:.2f}")

    # Define stopping criteria
    num_sites = len(np.unique(y))  # Number of unique sites
    chance_level = 1 / num_sites  # Random guessing accuracy

    # Start feature elimination loop
    features_to_remove = []
    iteration = 0
    new_accuracy = 0
    while iteration < max_features_to_remove and len(X_train.columns) > min_features_to_keep:
        # Get feature importances
        feature_importances = site_predictor.feature_importances_

        # Identify the most predictive feature
        most_predictive_feature = X_train.columns[np.argmax(feature_importances)]
        features_to_remove.append(most_predictive_feature)

        # Remove the most predictive feature from the dataset
        X_train = X_train.drop(columns=[most_predictive_feature])
        X_test = X_test.drop(columns=[most_predictive_feature])

        # Retrain the classifier without the removed feature
        site_predictor = ExtraTreesClassifier(n_estimators=100, random_state=42)
        site_predictor.fit(X_train, y_train)

        # Get new accuracy
        new_accuracy = site_predictor.score(X_test, y_test)
        logger.info(
            f"Iteration {iteration + 1}: Removed '{most_predictive_feature}', New Accuracy: {new_accuracy:.2f}"
        )

        # Check stopping conditions
        if new_accuracy <= chance_level:
            logger.info("Stopping: Site prediction accuracy is near chance level.")
            break

        iteration += 1

    logger.info(f"Final removed features: {features_to_remove}")
    logger.info(f"Final site prediction accuracy: {new_accuracy:.2f}")

    return features_df.drop(columns=features_to_remove)


def winnow_feature_selection(
    features_df, snr_threshold=1.0, min_features_to_keep=MIN_FEATURES_TO_KEEP
):
    """
    Perform Winnow-based feature selection using ExtraTreesClassifier.

    Args:
        features_df (pd.DataFrame): DataFrame containing features along with 'site', 'participant', and 'task' columns.
        snr_threshold (float): Minimum SNR threshold for feature retention.
        min_features_to_keep (int): Minimum number of features to retain.

    Returns:
        pd.DataFrame: DataFrame with retained features and original metadata columns.
    """
    # Preserve metadata columns
    metadata_columns = ["site", "participant", "task", "label"]

    # Extract feature matrix and target labels
    X = features_df.drop(columns=metadata_columns)
    y = features_df["site"]

    # Generate a synthetic random feature (noise)
    np.random.seed(42)
    X["random_noise"] = np.random.normal(0, 1, size=len(X))

    # Train ExtraTreesClassifier to measure feature importance
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # Get feature importances
    feature_importances = clf.feature_importances_
    feature_names = X.columns

    # Identify the importance of the synthetic feature (random noise)
    random_feature_importance = feature_importances[X.columns.get_loc("random_noise")]

    # Compute SNR for each feature (ratio of feature importance to random noise importance)
    snr = feature_importances / random_feature_importance

    # Select features where SNR exceeds the threshold
    features_to_keep = feature_names[snr > snr_threshold].tolist()

    # Avoid including the synthetic "random_noise" feature
    if "random_noise" in features_to_keep:
        features_to_keep.remove("random_noise")

    # Ensure at least min_features_to_keep are retained
    if len(features_to_keep) < min_features_to_keep:
        # Sort features by their SNR in descending order
        sorted_features = [
            f for f, s in sorted(zip(feature_names, snr), key=lambda x: x[1], reverse=True)
        ]
        # Exclude the "random_noise" feature from the sorted list
        sorted_features = [f for f in sorted_features if f != "random_noise"]
        # Select the top features to meet the minimum requirement
        features_to_keep = sorted_features[:min_features_to_keep]

    logger.info(
        f"Retained {len(features_to_keep)} features after applying SNR threshold and minimum feature constraint."
    )

    # Return the original DataFrame with only selected features and metadata columns
    return features_df[metadata_columns + features_to_keep]


def add_random_labels(
    df, column_name="label", labels=["accept", "exclude", "unsure"], random_seed=None
):
    """
    Adds a column with randomly assigned labels to the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to modify.
    - column_name (str): Name of the new column (default: "label").
    - labels (list): List of labels to sample from (default: ["accept", "exclude", "unsure"]).
    - random_seed (int or None): Random seed for reproducibility (default: None).

    Returns:
    - pd.DataFrame: Modified DataFrame with the new column.
    """
    if random_seed is not None:
        np.random.seed(random_seed)  # Set random seed for reproducibility

    df[column_name] = np.random.choice(labels, size=len(df))
    return df


def preprocess_data(features_df, preprocessing_steps, label_column="label"):
    """
    Preprocesses data by handling missing values, standardizing features,
    and applying specified preprocessing steps.

    Args:
        features_df (pd.DataFrame): DataFrame containing features and labels.
        preprocessing_steps (tuple): A tuple containing:
            - A set of preprocessing steps to apply ({"normalize", "eliminate", "winnow"}).
            - A string representing the normalization mode ("center", "scale", "both", or None).
        label_column (str): Name of the column containing classification labels (default: "label").

    Returns:
        tuple:
            - X_scaled (np.array): Scaled feature matrix.
            - y (pd.Series): Labels.
            - selected_features (list): List of selected feature names.
    """
    if label_column not in features_df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame.")

    if DEBUG_MODE:
        transformed_data = features_df.groupby("site").head(10).copy().reset_index(drop=True)
    else:
        transformed_data = features_df.copy().reset_index(drop=True)

    steps, mode = preprocessing_steps  # Extract preprocessing steps and mode
    for step in steps:
        if step == "normalize" and mode:
            transformed_data = site_wise_normalization(transformed_data, mode=mode)
        elif step == "eliminate":
            transformed_data = site_predictability_feature_elimination(transformed_data)
        elif step == "winnow":
            transformed_data = winnow_feature_selection(transformed_data)

    # Identify columns that are entirely NaN and replace only those with zeroes
    nan_columns = transformed_data.columns[transformed_data.isna().all()]
    transformed_data[nan_columns] = 0

    # Extract selected features after preprocessing
    selected_features = list(
        transformed_data.drop(columns=[label_column, "site", "participant", "task"]).columns
    )

    X = transformed_data[selected_features]  # Keep only selected features
    y = transformed_data[label_column]

    logger.info(f"Selected {len(selected_features)} features for training.")

    # Handle missing values
    if X.isna().sum().sum() > 0:
        logger.info("Warning: NaN values detected. Imputing missing values...")
        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(
            imputer.fit_transform(X), columns=selected_features
        )  # Assign correct columns

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, selected_features


def svm_train(X, y, cv_folds=5):
    """
    Trains and evaluates Support Vector Machines (SVM) using cross-validation.

    Args:
        X (pd.DataFrame or np.array): Feature matrix.
        y (pd.Series or np.array): Labels.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        dict: Contains the best SVM model and its cross-validation score.
    """
    if DEBUG_MODE:
        # Minimal search space for speed
        param_grid_svm = {
            "C": [1e-3, 1],
            "kernel": ["linear"],  # Only one kernel
            "gamma": ["scale"],  # Single gamma setting
        }
    else:
        # Larger search space
        param_grid_svm = {
            "C": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1, 1, 10],
            "degree": [2, 3, 4, 5, 6],
            "coef0": [0.0, 0.1, 0.5, 1.0],
            "class_weight": [None, "balanced"],
        }

    grid_search = GridSearchCV(SVC(), param_grid_svm, cv=cv_folds, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X, y)

    best_svc = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    logger.info(f"Best SVM {best_params}, CV Accuracy: {best_score:.4f}")

    return {
        "best_svc": best_svc,
        "best_params": best_params,
        "cv_accuracy": best_score,
    }


def rfc_train(X, y, cv_folds=5):
    """
    Trains and evaluates a Random Forest Classifier (RFC) using cross-validation.

    Args:
        X (pd.DataFrame or np.array): Feature matrix.
        y (pd.Series or np.array): Labels.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        dict: Contains the best RFC model and its cross-validation score.
    """
    if DEBUG_MODE:
        param_grid_rfc = {
            "n_estimators": [10],
            "max_depth": [5],
            "min_samples_split": [2],
            "min_samples_leaf": [1],
            "max_features": ["sqrt"],
            "bootstrap": [True],
            "criterion": ["gini"],
        }
    else:
        param_grid_rfc = {
            "n_estimators": [100, 200, 500, 1000],
            "max_depth": [None, 10, 20, 30, 40, 50],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["auto", "sqrt", "log2", None],
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"],
            "class_weight": [None, "balanced", "balanced_subsample"],
        }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rfc,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X, y)

    best_rfc = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    logger.info(f"Best Random Forest {best_params}, CV Accuracy: {best_score:.4f}")

    return {
        "best_rfc": best_rfc,
        "best_params": best_params,
        "cv_accuracy": best_score,
    }


def train_final_model(
    features_df, best_inner_model, best_preprocessing_steps, label_column="label"
):
    """
    Applies the best preprocessing steps to the entire dataset in the exact order
    and trains a new model on all data using the best hyperparameters.

    Args:
        features_df (pd.DataFrame): The complete dataset with features and labels.
        best_inner_model (sklearn estimator): The best-performing model from the inner loop.
        best_preprocessing_steps (tuple): The preprocessing steps used by the best model.
        label_column (str): Name of the classification label column.

    Returns:
        final_model: The newly trained model using the best hyperparameters on all data.
    """
    X_all, y_all, selected_features = preprocess_data(
        features_df, best_preprocessing_steps, label_column=label_column
    )

    best_model_class = type(best_inner_model)
    best_model_params = best_inner_model.get_params()
    final_model = best_model_class(**best_model_params)
    final_model.fit(X_all, y_all)
    return final_model, selected_features


def inner_loop(features_df, label_column="label", cv_folds=5, output_dir="training_results"):
    """Performs an inner-loop search over feature preprocessing configurations,
    trains SVM and RFC models, and selects the best-performing model.

    Args:
        features_df (pd.DataFrame): DataFrame containing features and labels.
        label_column (str): Name of the classification label column.
        cv_folds (int): Number of cross-validation folds.
        output_dir (str): Directory where model results are saved.

    Returns:
        tuple:
            - best_model (sklearn estimator): The best-performing model.
            - best_score (float): The best cross-validation accuracy score.
            - best_steps (tuple): The preprocessing steps that yielded the best-performing model.
            - selected_features (list): List of selected feature column names.
    """
    best_model = None
    best_score = 0
    best_steps = None
    selected_features = None  # To store feature names for best model

    preprocessing_steps = ["normalize", "eliminate", "winnow"]
    preprocessing_permutations = [
        list(permutation)
        for r in range(1, len(preprocessing_steps) + 1)
        for combination in combinations(preprocessing_steps, r)
        for permutation in permutations(combination)
    ]

    # if DEBUG_MODE:
    #     preprocessing_permutations = [["normalize"]]

    for preprocessing_permutation in preprocessing_permutations:
        normalize_modes = (
            ["center", "scale", "both"] if "normalize" in preprocessing_permutation else [None]
        )

        for mode in normalize_modes:
            transformed_data = features_df.copy().reset_index(drop=True)

            X, y, selected_feature_names = preprocess_data(
                transformed_data, (preprocessing_permutation, mode), label_column
            )

            svm_results = svm_train(X, y, cv_folds=cv_folds)
            rfc_results = rfc_train(X, y, cv_folds=cv_folds)

            for model_type, results in [
                ("SVM", svm_results),
                ("RandomForest", rfc_results),
            ]:
                model = results["best_svc"] if model_type == "SVM" else results["best_rfc"]
                score = results["cv_accuracy"]

                # Save model for this preprocessing step
                step_dir = os.path.join(
                    output_dir, "inner_loop", "_".join(preprocessing_permutation)
                )
                os.makedirs(step_dir, exist_ok=True)

                # Save selected features
                feature_file = os.path.join(step_dir, "selected_features.json")
                with open(feature_file, "w") as f:
                    json.dump(selected_features, f, indent=4)

                save_model(
                    os.path.join(step_dir, model_type),
                    model,
                    metadata={
                        "preprocessing_steps": preprocessing_permutation,
                        "mode": mode,
                        "cv_score": score,
                        "model_type": type(model).__name__,
                        "hyperparameters": model.get_params(),
                    },
                )

                if score > best_score:
                    best_model = model
                    best_score = score
                    best_steps = (preprocessing_permutation, mode)
                    selected_features = (
                        selected_feature_names  # Update selected features for best model
                    )

    return best_model, best_score, best_steps, selected_features


def train_qc_classifier(
    features_csv_path,
    participants_tsv_path,
    label_column="label",
    cv_folds=5,
    base_output_dir="training_results",
    n_jobs=-1,  # Enables parallel processing with automatic CPU allocation
):
    """Performs an outer-loop cross-validation process using a leave-one-site-out (LoSo) approach.
    Trains models using the inner loop, selects the best-performing model across all site folds,
    and then trains a new model with the best hyperparameters on all data.

    Args:
        features_csv_path (str): Path to the features CSV file.
        participants_tsv_path (str): Path to the participants TSV file.
        label_column (str): Name of the classification label column.
        cv_folds (int): Number of cross-validation folds.
        base_output_dir (str): Output directory for saved models.
        n_jobs (int): Number of parallel jobs (-1 = all available CPUs).

    Returns:
        final_model: The trained model using the best hyperparameters.
    """
    # Create a unique directory for this training run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_output_dir, f"training_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save training metadata
    training_run_metadata = {
        "timestamp": timestamp,
        "min_features_to_keep": MIN_FEATURES_TO_KEEP,
        "debug_mode": DEBUG_MODE,
        "cv_folds": cv_folds,
        "label_column": label_column,
        "features_csv_path": features_csv_path,
        "participants_tsv_path": participants_tsv_path,
        "n_jobs": n_jobs,
    }
    with open(os.path.join(run_dir, "training_run_metadata.json"), "w") as f:
        json.dump(training_run_metadata, f, indent=4, ensure_ascii=False)

    # Load and process data
    features_df = get_features_df_with_site(features_csv_path, participants_tsv_path)
    features_df = add_random_labels(features_df)
    unique_sites = features_df["site"].unique()

    # Define function for parallel execution
    def process_site(site):
        """Handles inner loop training for a single site."""
        logger.info(f"Processing site: {site}")

        train_df = features_df[features_df["site"] != site].copy().reset_index(drop=True)
        test_df = features_df[features_df["site"] == site].copy().reset_index(drop=True)
        site_dir = os.path.join(run_dir, f"site_{site}")
        os.makedirs(site_dir, exist_ok=True)

        # Run inner loop training
        best_fold_model, best_fold_score, best_fold_steps, selected_features = inner_loop(
            train_df, label_column, cv_folds, site_dir
        )

        # Apply best preprocessing to test set
        test_steps = (
            (["normalize"], best_fold_steps[1]) if "normalize" in best_fold_steps[0] else ([], None)
        )
        test_df = test_df[selected_features + [label_column, "site", "participant", "task"]]
        X_test, y_test, _ = preprocess_data(test_df, test_steps, label_column)
        best_model_score = best_fold_model.score(X_test, y_test)

        logger.info(
            f"Best model for site {site}: {best_fold_model} with test score {best_model_score:.4f}"
        )

        # Save best model for site
        save_model(
            site_dir,
            best_fold_model,
            metadata={
                "site": site,
                "best_preprocessing_steps": best_fold_steps,
                "test_score": best_model_score,
                "model_type": type(best_fold_model).__name__,
                "hyperparameters": best_fold_model.get_params(),
                "selected_features": selected_features,
            },
        )

        return best_fold_model, best_fold_score, best_fold_steps

    # Run inner loop training in parallel across sites
    best_inner_loop_models = Parallel(n_jobs=n_jobs)(
        delayed(process_site)(site) for site in unique_sites
    )

    # Select the best model from all sites
    best_inner_model, best_model_score, best_preprocessing_steps = max(
        best_inner_loop_models, key=lambda x: x[1]
    )
    logger.info(f"Best model selected: {best_inner_model} with score {best_model_score:.4f}")

    # Train final model with best parameters
    final_model, selected_features = train_final_model(
        features_df, best_inner_model, best_preprocessing_steps, label_column
    )

    # Save final model
    final_model_dir = os.path.join(run_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    save_model(
        final_model_dir,
        final_model,
        metadata={
            "best_preprocessing_steps": best_preprocessing_steps,
            "best_model_score": best_model_score,
            "model_type": type(final_model).__name__,
            "hyperparameters": final_model.get_params(),
            "selected_features": selected_features,
        },
    )

    logger.info(f"Training run saved at: {run_dir}")
    return final_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AudioQC model using MRIQC-like pipeline.")
    parser.add_argument("--features_csv", required=True, help="Path to the features CSV file.")
    parser.add_argument(
        "--participants_tsv", required=True, help="Path to the participants TSV file."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Base directory for saving training results."
    )

    args = parser.parse_args()

    train_qc_classifier(
        features_csv_path=args.features_csv,
        participants_tsv_path=args.participants_tsv,
        base_output_dir=args.output_dir,
    )
