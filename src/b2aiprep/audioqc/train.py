"""Implements functions for training AudioQC using a process similar to MRIQC."""

import logging
from itertools import combinations, permutations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

MIN_FEATURES_TO_KEEP = 64  # in feature selection


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


def preprocess_data(features_df, label_column="label"):
    """
    Preprocesses data by handling missing values and standardizing features.

    Args:
        features_df (pd.DataFrame): DataFrame containing features and labels.
        label_column (str): Name of the column containing classification labels (default: "label").

    Returns:
        X_scaled (np.array): Scaled feature matrix.
        y (pd.Series): Labels.
        scaler (StandardScaler): Scaler used for transformations.
    """
    if label_column not in features_df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame.")

    # Separate features and labels
    X = features_df.drop(columns=[label_column, "site", "participant", "task"])
    X = X[:15]  # TODO REMOVE
    logger.info(X.shape)
    y = features_df[label_column]
    y = y[:15]  # TODO REMOVE
    logger.info(y.shape)

    # Handle missing values
    if X.isna().sum().sum() > 0:
        logger.info("Warning: NaN values detected. Imputing missing values...")
        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def grid_search_cross_val(model, param_grid, X, y, cv=5):
    """
    Performs Grid Search with Cross-Validation for hyperparameter tuning.

    Args:
        model (sklearn estimator): The machine learning model to tune.
        param_grid (dict): Dictionary with hyperparameter options.
        X (pd.DataFrame or np.array): Feature matrix.
        y (pd.Series or np.array): Labels.
        cv (int, optional): Number of cross-validation folds (default: 5).

    Returns:
        best_model: The best model found during grid search.
        best_params (dict): Best hyperparameters found.
        best_score (float): Best cross-validation accuracy score.
    """
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def svm_train(X, y, cv_folds=10):
    """Trains and evaluates Support Vector Machines (SVM) using cross-validation.

    Args:
        X (pd.DataFrame or np.array): Feature matrix.
        y (pd.Series or np.array): Labels.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        dict: Contains best models (linear and RBF) and their cross-validation scores.
    """
    param_grid_lin = {"C": [0.1, 1, 10, 100]}
    param_grid_rbf = {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1]}

    best_svc_lin, best_params_lin, best_score_lin = grid_search_cross_val(
        SVC(kernel="linear"), param_grid_lin, X, y, cv=cv_folds
    )
    best_svc_rbf, best_params_rbf, best_score_rbf = grid_search_cross_val(
        SVC(kernel="rbf"), param_grid_rbf, X, y, cv=cv_folds
    )

    logger.info(f"Best Linear SVC {best_params_lin}, CV Accuracy: {best_score_lin:.4f}")
    logger.info(f"Best RBF SVC {best_params_rbf}, CV Accuracy: {best_score_rbf:.4f}")

    return {
        "best_linear_svc": best_svc_lin,
        "best_params_lin": best_params_lin,
        "best_rbf_svc": best_svc_rbf,
        "best_params_rbf": best_params_rbf,
        "cv_accuracy_linear": best_score_lin,
        "cv_accuracy_rbf": best_score_rbf,
    }


def rfc_train(X, y, cv_folds=10):
    """Trains and evaluates a Random Forest Classifier (RFC) using cross-validation.

    Args:
        X (pd.DataFrame or np.array): Feature matrix.
        y (pd.Series or np.array): Labels.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        dict: Contains the best RFC model and its cross-validation score.
    """
    param_grid_rfc = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    best_rfc, best_params_rfc, best_score_rfc = grid_search_cross_val(
        RandomForestClassifier(random_state=42), param_grid_rfc, X, y, cv=cv_folds
    )

    logger.info(f"Best Random Forest {best_params_rfc}, CV Accuracy: {best_score_rfc:.4f}")

    return {
        "best_rfc": best_rfc,
        "best_params_rfc": best_params_rfc,
        "cv_accuracy_rfc": best_score_rfc,
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
    best_perm, best_mode = best_preprocessing_steps
    features_transformed = features_df.copy()

    for step in best_perm:
        if step == "normalize" and best_mode:
            features_transformed = site_wise_normalization(features_transformed, mode=best_mode)
        elif step == "eliminate":
            features_transformed = site_predictability_feature_elimination(features_transformed)
        elif step == "winnow":
            features_transformed = winnow_feature_selection(features_transformed)

    # Preprocess and train a new model on all data using the best hyperparameters
    best_model_class = type(best_inner_model)
    best_model_params = best_inner_model.get_params()
    final_model = best_model_class(**best_model_params)

    X_all, y_all = preprocess_data(features_transformed, label_column=label_column)
    final_model.fit(X_all, y_all)

    return final_model


def inner_loop(features_df, label_column="label", cv_folds=10):
    """Performs an inner-loop search over feature preprocessing configurations,
    trains SVM and RFC models, and selects the best-performing model.

    Args:
        features_df (pd.DataFrame): DataFrame containing features and labels.
        label_column (str): Name of the classification label column.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        tuple:
            - best_model (sklearn estimator): The best-performing model.
            - best_score (float): The best cross-validation accuracy score.
            - best_steps (tuple): The preprocessing steps that yielded the best-performing model.
    """
    best_model = None
    best_score = 0
    best_steps = None

    preprocessing_steps = ["normalize", "eliminate", "winnow"]
    preprocessing_permutations = [
        list(permutation)
        for r in range(1, len(preprocessing_steps) + 1)
        for combination in combinations(preprocessing_steps, r)
        for permutation in permutations(combination)
    ]
    # Example override: only "normalize" for demonstration
    preprocessing_permutations = [set(["normalize"])]

    for preprocessing_permutation in preprocessing_permutations:
        features_transformed = features_df.copy()

        # Apply normalization if selected
        normalize_modes = (
            ["center", "scale", "both"] if "normalize" in preprocessing_permutation else [None]
        )
        # Example override: only "center" for demonstration
        normalize_modes = ["center"]

        for mode in normalize_modes:
            transformed_data = features_transformed.copy()
            if mode:
                transformed_data = site_wise_normalization(transformed_data, mode=mode)
            if "eliminate" in preprocessing_permutation:
                transformed_data = site_predictability_feature_elimination(transformed_data)
            if "winnow" in preprocessing_permutation:
                transformed_data = winnow_feature_selection(transformed_data)

            # Preprocess once, then pass X, y to trainers
            X, y = preprocess_data(transformed_data, label_column=label_column)

            svm_results = svm_train(X, y, cv_folds=cv_folds)
            best_svm_model, best_svm_score = (
                svm_results["best_linear_svc"]
                if svm_results["cv_accuracy_linear"] > svm_results["cv_accuracy_rbf"]
                else svm_results["best_rbf_svc"]
            ), max(svm_results["cv_accuracy_linear"], svm_results["cv_accuracy_rbf"])

            rfc_results = rfc_train(X, y, cv_folds=cv_folds)
            best_rfc_model, best_rfc_score = rfc_results["best_rfc"], rfc_results["cv_accuracy_rfc"]

            # Track the best model and store the preprocessing steps
            for model, score in [
                (best_svm_model, best_svm_score),
                (best_rfc_model, best_rfc_score),
            ]:
                if score > best_score:
                    best_model = model
                    best_score = score
                    # Store both the set of steps and the normalization mode used
                    best_steps = (preprocessing_permutation, mode)

    return best_model, best_score, best_steps


def outer_loop(features_csv_path, participants_tsv_path, label_column="label", cv_folds=5):
    """Performs an outer-loop cross-validation process using a leave-one-site-out (LoSo) approach.
    Trains models using the inner loop, selects the best-performing model across all site folds,
    and then trains a new model with the best hyperparameters on all data, applying the same
    preprocessing steps that led to the best model in the exact order.

    Args:
        features_csv_path (str): Path to the features CSV file.
        participants_tsv_path (str): Path to the participants TSV file.
        label_column (str): Name of the classification label column.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        tuple:
            - best_final_model: The newly trained model using the best hyperparameters on all data.
            - best_preprocessing_steps (tuple): The preprocessing steps used by the best model.
    """
    # Load dataset with site labels
    features_df = get_features_df_with_site(
        features_csv_path=features_csv_path, participants_tsv_path=participants_tsv_path
    )

    features_df = add_random_labels(features_df)

    unique_sites = features_df["site"].unique()
    best_inner_loop_models = []

    # Leave-One-Site-Out (LoSo) Cross-Validation
    for site in tqdm(unique_sites):
        logger.info(f"Processing site: {site}")

        # Hold out one site as the test set
        train_df = features_df[features_df["site"] != site].copy()
        test_df = features_df[features_df["site"] == site].copy()

        # Train model using the inner loop (now also returning best steps)
        best_fold_model, best_fold_score, best_fold_steps = inner_loop(
            train_df, label_column, cv_folds
        )

        # Evaluate the best model on the test set
        X_test, y_test = preprocess_data(test_df, label_column=label_column)
        best_model_score = best_fold_model.score(X_test, y_test)

        print(
            f"Best model for site {site}: {best_fold_model} "
            f"with test score {best_model_score:.4f} "
            f"and preprocessing steps {best_fold_steps}"
        )
        best_inner_loop_models.append((best_fold_model, best_fold_score, best_fold_steps))

    # Select the best model and its score across all sites
    best_inner_model, best_model_score, best_preprocessing_steps = max(
        best_inner_loop_models, key=lambda x: x[1]
    )

    print(
        f"Best model selected from inner loop: {best_inner_model} "
        f"with a score of {best_model_score:.4f} "
        f"and preprocessing steps {best_preprocessing_steps}"
    )

    final_model = train_final_model(
        features_df, best_inner_model, best_preprocessing_steps, label_column
    )

    return final_model


if __name__ == "__main__":
    features_csv_path = "/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/bridge2ai-voice-corpus-3/derived/static_features.csv"
    participants_tsv_path = "/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/bridge2ai-voice-corpus-3/bids/bids/participants.tsv"
    outer_loop(features_csv_path=features_csv_path, participants_tsv_path=participants_tsv_path)
