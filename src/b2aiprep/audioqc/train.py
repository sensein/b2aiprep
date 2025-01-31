"""Implements functions for training AudioQC using a process similar to MRIQC."""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


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


def svm_train(features_df, label_column="label", test_size=0.2, random_state=42):
    """
    Trains Support Vector Machines (SVM) with hyperparameter tuning to predict labels ('accept', 'exclude', 'unsure').

    Parameters:
    - features_df (pd.DataFrame): DataFrame containing features and the label column.
    - label_column (str): Name of the column containing classification labels (default: "label").
    - test_size (float): Fraction of data to use for testing (default: 0.2).
    - random_state (int): Random seed for reproducibility.

    Returns:
    - dict: Contains best models and their test accuracies.
    """

    # Ensure label column exists
    if label_column not in features_df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame.")

    # Separate features and labels
    X = features_df.drop(columns=[label_column])  # Features
    y = features_df[label_column]  # Labels

    # Check for NaN values
    if X.isna().sum().sum() > 0:
        print("Warning: NaN values detected in features. Imputing missing values...")
        imputer = SimpleImputer(strategy="mean")  # Fill NaN with mean of the column
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Standardize features (SVM performs better with standardized inputs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define hyperparameter grids for linear SVC and RBF SVC
    param_grid_lin = {'C': [0.1, 1, 10, 100]}
    param_grid_rbf = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}

    # Perform Grid Search for Linear SVC
    grid_search_lin = GridSearchCV(SVC(kernel="linear"), param_grid_lin, cv=5, scoring="accuracy")
    grid_search_lin.fit(X_train_scaled, y_train)
    best_svc_lin = grid_search_lin.best_estimator_

    # Perform Grid Search for RBF SVC
    grid_search_rbf = GridSearchCV(SVC(kernel="rbf"), param_grid_rbf, cv=5, scoring="accuracy")
    grid_search_rbf.fit(X_train_scaled, y_train)
    best_svc_rbf = grid_search_rbf.best_estimator_

    # Evaluate the best models on test set
    y_pred_lin = best_svc_lin.predict(X_test_scaled)
    y_pred_rbf = best_svc_rbf.predict(X_test_scaled)

    acc_lin = accuracy_score(y_test, y_pred_lin)
    acc_rbf = accuracy_score(y_test, y_pred_rbf)

    # Print results
    print(f"Best Linear SVC (C={grid_search_lin.best_params_['C']}), Test Accuracy: {acc_lin:.4f}")
    print(f"Best RBF SVC (C={grid_search_rbf.best_params_['C']}, gamma={grid_search_rbf.best_params_['gamma']}), Test Accuracy: {acc_rbf:.4f}")

    # Return best models and accuracies
    return {
        "best_linear_svc": best_svc_lin,
        "best_rbf_svc": best_svc_rbf,
        "accuracy_linear": acc_lin,
        "accuracy_rbf": acc_rbf
    }


def train_random_forest(features_df, label_column="label", test_size=0.2, random_state=42):
    """
    Trains a Random Forest Classifier (RFC) with hyperparameter tuning using GridSearchCV.

    Parameters:
    - features_df (pd.DataFrame): DataFrame containing features and the label column.
    - label_column (str): Name of the column containing classification labels (default: "label").
    - test_size (float): Fraction of data to use for testing (default: 0.2).
    - random_state (int): Random seed for reproducibility.

    Returns:
    - dict: Contains best model and its test accuracy.
    """

    # Ensure label column exists
    if label_column not in features_df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame.")

    # Separate features and labels
    X = features_df.drop(columns=[label_column])  # Features
    y = features_df[label_column]  # Labels

    # Check for NaN values and impute missing data
    if X.isna().sum().sum() > 0:
        print("Warning: NaN values detected in features. Imputing missing values...")
        imputer = SimpleImputer(strategy="mean")  # Fill NaN with mean of the column
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Standardize features (optional for tree-based models, but helps when combined with other models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define hyperparameter grid for Random Forest
    param_grid = {
        "n_estimators": [50, 100, 200],  # Number of trees
        "max_depth": [None, 10, 20],  # Maximum depth of trees
        "min_samples_split": [2, 5, 10],  # Minimum samples required to split an internal node
        "min_samples_leaf": [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    }

    # Perform Grid Search for Random Forest
    grid_search = GridSearchCV(RandomForestClassifier(random_state=random_state), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_rfc = grid_search.best_estimator_

    # Evaluate the best model on test set
    y_pred = best_rfc.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    # Print results
    print(f"Best Random Forest Model: {grid_search.best_params_}, Test Accuracy: {acc:.4f}")

    # Return best model and accuracy
    return {
        "best_rfc": best_rfc,
        "accuracy": acc
    }




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
