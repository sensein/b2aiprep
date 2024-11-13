"""Updates the synthetic data to use the coded header (variable names) rather than the descriptions.

Also inserts the additional columns necessary to align the dataset with bridge2ai voice 3rd corpus, used
to generate v1.0 released data."""
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    # get path to this patch, which has the data etc
    current_path = Path(__file__).parent
    df_synthetic = pd.read_csv(current_path.joinpath("sdv_redcap_synthetic_data_1000_rows.csv"))

    # patch the dataset
    initial_columns = [
        'record_id',
        'redcap_repeat_instrument',
        'redcap_repeat_instance',
        'selected_language',
        'consent_status',
        'is_feasibility_participant',
        'enrollment_institution',
        'age',
        'eligible_studies___1',
        'eligible_studies___2',
        'eligible_studies___3',
        'eligible_studies___4',
    ]

    n_cols = len(initial_columns)
    synthetic_columns = list(df_synthetic.columns)
    df_synthetic.columns = initial_columns + synthetic_columns[n_cols:]


    add_columns_in_the_middle = [
        'eligible_studies___age_2_4',
        'eligible_studies___age_4_6',
        'eligible_studies___age_6_10',
        'eligible_studies___age_10_plus',
    ]
    for col in add_columns_in_the_middle:
        df_synthetic[col] = np.nan

    df_synthetic = df_synthetic[initial_columns + add_columns_in_the_middle + synthetic_columns[n_cols:]]

    add_columns_at_the_end = [
        'mbd_diagnosed_bm',
        'mbd_ever_seen_heard',
        'mbd_had_manic_episode',
        'mbd_active_psy_med_problems___adhd',
        'mbd_active_psy_med_problems___anxiety',
        'mbd_active_psy_med_problems___ocd',
        'mbd_active_psy_med_problems___stroke',
        'mbd_active_psy_med_problems___epilepsy',
        'mbd_active_psy_med_problems___laryngeal_cancer',
        'mbd_active_psy_med_problems___seasonal_allergies',
        'mbd_active_psy_med_problems___other',
        'mbd_hist_psy_med_problems___adhd',
        'mbd_hist_psy_med_problems___anxiety',
        'mbd_hist_psy_med_problems___ocd',
        'mbd_hist_psy_med_problems___stroke',
        'mbd_hist_psy_med_problems___epilepsy',
        'mbd_hist_psy_med_problems___laryngeal_cancer',
        'mbd_hist_psy_med_problems___seasonal_allergies',
        'mbd_hist_psy_med_problems___other',
        'mbd_many_depressive_episodes',
        'mbd_many_manic_episodes',
        'dmdd_diagnosed_dd',
        'dmdd_active_psy_med_problems___adhd',
        'dmdd_active_psy_med_problems___anxiety',
        'dmdd_active_psy_med_problems___ocd',
        'dmdd_active_psy_med_problems___stroke',
        'dmdd_active_psy_med_problems___epilepsy',
        'dmdd_active_psy_med_problems___laryngeal_cancer',
        'dmdd_active_psy_med_problems___seasonal_allergies',
        'dmdd_active_psy_med_problems___other',
        'dmdd_hist_psy_med_problems___adhd',
        'dmdd_hist_psy_med_problems___anxiety',
        'dmdd_hist_psy_med_problems___ocd',
        'dmdd_hist_psy_med_problems___stroke',
        'dmdd_hist_psy_med_problems___epilepsy',
        'dmdd_hist_psy_med_problems___laryngeal_cancer',
        'dmdd_hist_psy_med_problems___seasonal_allergies',
        'dmdd_hist_psy_med_problems___other',
        'dmdd_how_many_depressive_episodes',
        'dmdd_prescribed_medication',
        'dmdd_see_mental_health_professional',
        'diagnosis_diagnosed_ad',
        'diagnosis_ad_active_psy_med_problems___adhd',
        'diagnosis_ad_active_psy_med_problems___anxiety',
        'diagnosis_ad_active_psy_med_problems___ocd',
        'diagnosis_ad_active_psy_med_problems___stroke',
        'diagnosis_ad_active_psy_med_problems___epilepsy',
        'diagnosis_ad_active_psy_med_problems___laryngeal_cancer',
        'diagnosis_ad_active_psy_med_problems___seasonal_allergies',
        'diagnosis_ad_active_psy_med_problems___other',
        'diagnosis_ad_hist_psy_med_problems___adhd',
        'diagnosis_ad_hist_psy_med_problems___anxiety',
        'diagnosis_ad_hist_psy_med_problems___ocd',
        'diagnosis_ad_hist_psy_med_problems___stroke',
        'diagnosis_ad_hist_psy_med_problems___epilepsy',
        'diagnosis_ad_hist_psy_med_problems___laryngeal_cancer',
        'diagnosis_ad_hist_psy_med_problems___seasonal_allergies',
        'diagnosis_ad_hist_psy_med_problems___other',
        'diagnosis_ad_prescribed_medication',
        'diagnosis_ad_see_mental_health_professional',
    ]

    for col in add_columns_at_the_end:
        df_synthetic[col] = np.nan

    # load in the column data
    with open(current_path.joinpath('corpus_3_column_coded_names.txt'), 'r') as fp:
        updated_column_names = fp.read().split('\n')

    df_synthetic.columns = updated_column_names
    print('Patched! New shape:')
    print(df_synthetic.shape)

    df_synthetic.to_csv(current_path.joinpath('sdv_redcap_synthetic_data_1000_rows.csv'), index=False)

if __name__ == '__main__':
    main()