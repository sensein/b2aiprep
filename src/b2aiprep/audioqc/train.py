"""Implements functions for training AudioQC using a process similar to MRIQC."""


def get_features_df_with_site(features_path, participants_path):
    return


def site_wise_normalization():
    return


def site_predictive_dimensionality_reduction():
    return


def winnow():
    return


def add_labels(labels_csv = None):
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