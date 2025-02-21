import json
import os
from datetime import datetime

import joblib
import pandas as pd


def save_model(output_dir, model, metadata, feature_importance=None, removed_features=None):
    """Saves a model and associated metadata."""
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    joblib.dump(model, os.path.join(output_dir, "model.joblib"))

    # Save metadata with model type and hyperparameters
    metadata.update({"model_type": type(model).__name__, "hyperparameters": model.get_params()})

    with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    # Save feature importance (if available)
    if feature_importance is not None:
        feature_importance.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    # Save removed features (if applicable)
    if removed_features is not None:
        pd.DataFrame(removed_features, columns=["removed_features"]).to_csv(
            os.path.join(output_dir, "removed_features.csv"), index=False
        )


def save_training_results(base_output_dir, final_model, best_preprocessing_steps, best_model_score):
    """Creates structured output directories for all models and metadata."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_output_dir, f"training_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save final model
    save_model(
        os.path.join(run_dir, "final_model"),
        final_model,
        metadata={
            "best_preprocessing_steps": best_preprocessing_steps,
            "best_model_score": best_model_score,
        },
    )

    return run_dir
