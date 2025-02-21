

"""
For each site (Outer Loop - Leave-One-Site-Out Cross-Validation):
    Hold out one site as the test set
    Train on the remaining N-1 sites

    For each hyperparameter configuration (Inner Loop - Grid/Random Search):
        Select a combination of:
            - Preprocessing method (normalization + feature selection)
            - Model type (SVC-lin, SVC-rbf, RFC)
            - Classifier hyperparameters (C, gamma, tree depth, etc.)

        For each fold in cross-validation (Inner Loop - Cross-Validation):
            Split the training data into train/validation folds
            Train the model on the training fold
            Evaluate on the validation fold

        Select the best preprocessing/model configuration based on average validation performance

    Train the final model with the best hyperparameters & preprocessing on all N-1 training sites
    Evaluate on the held-out site
512 models

for each site:
    hold out one site
    training data = remaining sites data
    for preprocessing method in [site-wise normalization, filter site origin, winnow]
        for center, scale, or both if site-wise normalization:
            for model in (SVC-lin, SVC-rbf, RFC)
                for model-specific hyperparameters in grid:
                    run cross-validation on the training data
    test the models on the held-out site
pick the best model across all the possible configurations




## Overall Training Process:
```
For each site (Outer Loop - Leave-One-Site-Out Cross-Validation):
    Hold out one site as the test set
    Train on the remaining N-1 sites

    For each hyperparameter configuration (Inner Loop - Grid/Random Search):
        Select a combination of:
            - Preprocessing method (normalization + feature selection)
            - Model type (SVC-lin, SVC-rbf, RFC)
            - Classifier hyperparameters (C, gamma, tree depth, etc.)

        For each fold in cross-validation (Inner Loop - Cross-Validation):
            Split the training data into train/validation folds
            Train the model on the training fold
            Evaluate on the validation fold

        Select the best preprocessing/model configuration based on average validation performance

    Train the final model with the best hyperparameters & preprocessing on all N-1 training sites
    Evaluate on the held-out site

```


best_inner_loop_models = []
for sites:
    hold a site out
    best_fold_model = inner()
    best_inner_loop_models.append(best_fold_model)
best_best_inner_model = best(best_inner_loop_models)
final_model = cross_validation_across_sites(best_best_inner_model)
external_dataset_evaluation(final_model)
"""




sites = []
for test_site in sites:
    train_sites = sites - test_site
    for configuration in configurations:



def check_quality(features_df):
    """adds a column with label accept, unsure, or exclude"""
    # load model
    # classify
    # return updated features_df
    return features_df
