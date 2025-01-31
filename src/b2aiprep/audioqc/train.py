"""Implements functions for training AudioQC using a process similar to MRIQC."""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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


def site_predictability_feature_elimination(features_df, max_features_to_remove=67):
    """
    Implements Site-Predictability-Based Feature Elimination using ExtraTreesClassifier.

    This function iteratively removes the most predictive feature for site classification
    until either the classifier's performance is near chance level or a predefined
    maximum number of features is removed.

    Args:
        features_df (pd.DataFrame): DataFrame containing features along with 'site', 'participant', and 'task' columns.
        max_features_to_remove (int, optional): Maximum number of features to remove. Default is 67 (131 - 64).

    Returns:
        tuple:
            - list: Removed features in order of elimination.
            - float: Final site prediction accuracy.
            - pd.DataFrame: DataFrame with remaining features after elimination.
    """
    # Drop non-AQM columns (site labels are the target variable)
    X = features_df.drop(columns=["site", "participant", "task"])
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
    print(f"Initial site prediction accuracy: {initial_accuracy:.2f}")

    # Define stopping criteria
    num_sites = len(np.unique(y))  # Number of unique sites
    chance_level = 1 / num_sites  # Random guessing accuracy

    # Start feature elimination loop
    features_to_remove = []
    iteration = 0

    while iteration < max_features_to_remove:
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
        print(
            f"Iteration {iteration + 1}: Removed '{most_predictive_feature}', New Accuracy: {new_accuracy:.2f}"
        )

        # Check stopping conditions
        if new_accuracy <= chance_level:
            print("Stopping: Site prediction accuracy is near chance level.")
            break

        iteration += 1

    print(f"Final removed features: {features_to_remove}")
    print(f"Final site prediction accuracy: {new_accuracy:.2f}")

    return features_to_remove, new_accuracy, X_train


def winnow_feature_selection(X, y, snr_threshold=1.0):
    """
    Perform Winnow-based feature selection using ExtraTreesClassifier.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels (e.g., site).
        snr_threshold (float): Minimum SNR threshold for feature retention.

    Returns:
        pd.DataFrame: Reduced feature set with low SNR features removed.
    """
    X = X.copy()

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
    selected_features = feature_names[snr > snr_threshold].tolist()

    # Avoid ValueError: Only remove "random_noise" if it exists
    if "random_noise" in selected_features:
        selected_features.remove("random_noise")

    print(f"Removed {X.shape[1] - len(selected_features)} low-SNR features.")

    return X[selected_features]


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
    X = features_df.drop(columns=[label_column])
    y = features_df[label_column]

    # Handle missing values
    if X.isna().sum().sum() > 0:
        print("Warning: NaN values detected. Imputing missing values...")
        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


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


def svm_train(features_df, label_column="label", cv_folds=5):
    """
    Trains and evaluates Support Vector Machines (SVM) using cross-validation.

    Args:
        features_df (pd.DataFrame): DataFrame containing features and label column.
        label_column (str): Name of the classification label column.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        dict: Contains best models and cross-validation scores.
    """
    X, y, _ = preprocess_data(features_df, label_column)

    # Define hyperparameter grids for linear SVC and RBF SVC
    param_grid_lin = {"C": [0.1, 1, 10, 100]}
    param_grid_rbf = {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1]}

    # Grid search for Linear SVC
    best_svc_lin, best_params_lin, best_score_lin = grid_search_cross_val(
        SVC(kernel="linear"), param_grid_lin, X, y, cv=cv_folds
    )

    # Grid search for RBF SVC
    best_svc_rbf, best_params_rbf, best_score_rbf = grid_search_cross_val(
        SVC(kernel="rbf"), param_grid_rbf, X, y, cv=cv_folds
    )

    print(f"Best Linear SVC {best_params_lin}, CV Accuracy: {best_score_lin:.4f}")
    print(f"Best RBF SVC {best_params_rbf}, CV Accuracy: {best_score_rbf:.4f}")

    return {
        "best_linear_svc": best_svc_lin,
        "best_rbf_svc": best_svc_rbf,
        "cv_accuracy_linear": best_score_lin,
        "cv_accuracy_rbf": best_score_rbf,
    }


def rfc_train(features_df, label_column="label", cv_folds=5):
    """
    Trains and evaluates a Random Forest Classifier (RFC) using cross-validation.

    Args:
        features_df (pd.DataFrame): DataFrame containing features and label column.
        label_column (str): Name of the classification label column.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        dict: Contains best RFC model and its cross-validation score.
    """
    X, y, _ = preprocess_data(features_df, label_column)

    # Define hyperparameter grid for Random Forest
    param_grid_rfc = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Grid search for Random Forest
    best_rfc, best_params_rfc, best_score_rfc = grid_search_cross_val(
        RandomForestClassifier(random_state=42), param_grid_rfc, X, y, cv=cv_folds
    )

    print(f"Best Random Forest {best_params_rfc}, CV Accuracy: {best_score_rfc:.4f}")

    return {"best_rfc": best_rfc, "cv_accuracy_rfc": best_score_rfc}


def inner_loop(features_df, label_column="label", cv_folds=5):
    """
    Performs an inner-loop search over feature preprocessing configurations,
    trains SVM and RFC models, and selects the best-performing model.

    Args:
        features_df (pd.DataFrame): DataFrame containing features and labels.
        label_column (str): Name of the classification label column.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        tuple:
            - best_model (sklearn estimator): The best-performing model.
            - best_score (float): The best cross-validation accuracy score.
    """
    best_model = None
    best_score = 0
    feature_elimination_steps = ["normalize", "eliminate", "winnow"]
    feature_elimination_combos = [
        set(combo)
        for i in range(len(feature_elimination_steps) + 1)
        for combo in combinations(feature_elimination_steps, i)
    ]

    for feature_elimination_combo in feature_elimination_combos:
        # Generate feature-transformed dataset
        features_transformed = features_df.copy()

        # Apply normalization if selected
        normalize_modes = (
            ["center", "scale", "both"] if "normalize" in feature_elimination_combo else [None]
        )
        for mode in normalize_modes:
            transformed_data = features_transformed.copy()

            if mode:
                transformed_data = site_wise_normalization(transformed_data, mode=mode)
            if "eliminate" in feature_elimination_combo:
                transformed_data = site_predictability_feature_elimination(transformed_data)
            if "winnow" in feature_elimination_combo:
                transformed_data = winnow_feature_selection(transformed_data)

            # Train and evaluate SVM models
            svm_results = svm_train(transformed_data, label_column=label_column, cv_folds=cv_folds)
            best_svm_model, best_svm_score = (
                svm_results["best_linear_svc"]
                if svm_results["cv_accuracy_linear"] > svm_results["cv_accuracy_rbf"]
                else svm_results["best_rbf_svc"]
            ), max(svm_results["cv_accuracy_linear"], svm_results["cv_accuracy_rbf"])

            # Train and evaluate RFC model
            rfc_results = rfc_train(transformed_data, label_column=label_column, cv_folds=cv_folds)
            best_rfc_model, best_rfc_score = rfc_results["best_rfc"], rfc_results["cv_accuracy_rfc"]

            # Select the best model
            for model, score in [
                (best_svm_model, best_svm_score),
                (best_rfc_model, best_rfc_score),
            ]:
                if score > best_score:
                    best_model, best_score = model, score


def outer_loop(features_csv_path, participants_tsv_path, label_column="label", cv_folds=5):
    """
    Performs an outer-loop cross-validation process using a leave-one-site-out (LoSo) approach.
    Trains models using the inner loop, selects the best-performing model across all sites,
    and performs cross-validation across all sites with the best model.

    Args:
        features_csv_path (str): Path to the features CSV file.
        participants_tsv_path (str): Path to the participants TSV file.
        label_column (str): Name of the classification label column.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        best_final_model: The best-trained model after evaluation.
    """
    # Load dataset with site labels
    features_df = get_features_df_with_site(
        features_csv_path=features_csv_path, participants_tsv_path=participants_tsv_path
    )

    unique_sites = features_df["site"].unique()
    best_inner_loop_models = []

    # Leave-One-Site-Out (LoSo) Cross-Validation
    for site in unique_sites:
        print(f"Processing site: {site}")

        # Hold out one site as the test set
        train_df = features_df[features_df["site"] != site].copy()
        test_df = features_df[features_df["site"] == site].copy()

        # Train model using the inner loop
        best_fold_model, best_fold_score = inner_loop(train_df, label_column, cv_folds)

        print(f"Best model for site {site}: {best_fold_model} with CV score {best_fold_score:.4f}")
        best_inner_loop_models.append((best_fold_model, best_fold_score))

    # Select the best model across all sites
    best_inner_model = max(best_inner_loop_models, key=lambda x: x[1])[0]
    print(f"Best model selected from inner loop: {best_inner_model}")

    # Perform cross-validation across all sites with the best model
    final_model = cross_validation_across_sites(
        features_df, best_inner_model, label_column, cv_folds
    )

    # Evaluate the final model on an external dataset
    external_dataset_evaluation(final_model)

    return final_model
