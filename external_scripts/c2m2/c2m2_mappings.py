peds_race_map = {
        'White': 'cfde_subject_race:3',
        'Black or African American': 'cfde_subject_race:2',
        'Asian': 'cfde_subject_race:5',
        'American Indian or Alaska Native': 'cfde_subject_race:0',
        'Native Hawaiian or Other Pacific Islander': 'cfde_subject_race:6',
        'Other': 'cfde_subject_race:4',
        'Canadian Indigenous or Aboriginal': 'cfde_subject_race:4',
        'Prefer not to answer': ''
        }


peds_sex_map = {
  '': 'cfde_subject_sex:0',
  'Female gender identity':'cfde_subject_sex:1',
  'Male gender identity': 'cfde_subject_sex:2',
  'Other': 'cfde_subject_sex:0'
}




race_map = {
  'race___1': 'cfde_subject_race:0',
  'race___2': 'cfde_subject_race:5',
  'race___3': 'cfde_subject_race:2',
  'race___4': 'cfde_subject_race:6',
  'race___5': 'cfde_subject_race:3',
  'race___6': 'cfde_subject_race:4',
  'race___7': 'cfde_subject_race:4',
}

sex_map = {
  '': 'cfde_subject_sex:0',
  'Female':'cfde_subject_sex:1',
  'Male': 'cfde_subject_sex:2',
}

ethnicity_map = {
  'Not Hispanic or Latino': 'cfde_subject_ethnicity:1',
  'Hispanic or Latino': 'cfde_subject_ethnicity:0',
  'Prefer not to answer': ''
}


# Maps task_name values (from parquet files) to OBI sample_prep_method terms.
# OBI:0003501 = sustained phonation, OBI:0003504 = reading passage,
# OBI:0003505 = picture description, OBI:0003529 = diadochokinesis
#
# Lookup logic (in bundle_to_c2m2.py): exact match first; if not found, strip a
# trailing -<digits> suffix and try again. This handles numbered repetitions
# (e.g. maximum-phonation-time-3 -> maximum-phonation-time) without substring
# matching, so caterpillar-passage will NOT accidentally match passage.
task_to_OBI = {
    'harvard-sentences-list': 'OBI:0003505',
    'animal-fluency': 'OBI:0003529',
    'breath-sounds': 'OBI:0003501',
    'cape-v-sentences': 'OBI:0003505',
    'cape-v-sentences-v2': 'OBI:0003505',
    'caterpillar-passage': 'OBI:0003505',
    'cinderella-story': 'OBI:0003505',
    'diadochokinesis-buttercup': 'OBI:0003501',
    'diadochokinesis-ka': 'OBI:0003501',
    'diadochokinesis-pa': 'OBI:0003501',
    'diadochokinesis-pataka': 'OBI:0003501',
    'diadochokinesis-ta': 'OBI:0003501',
    'diadochokinesis-v2-buttercup': 'OBI:0003501',
    'diadochokinesis-v2-kuh': 'OBI:0003501',
    'diadochokinesis-v2-puh': 'OBI:0003501',
    'diadochokinesis-v2-puhtuhkuh': 'OBI:0003501',
    'diadochokinesis-v2-tuh': 'OBI:0003501',
    'free-speech': 'OBI:0003529',
    'free-speech-v2': 'OBI:0003529',
    'glides-high-to-low': 'OBI:0003501',
    'glides-low-to-high': 'OBI:0003501',
    'high-to-low': 'OBI:0003501',
    'loudness': 'OBI:0003501',
    'loudness-v2': 'OBI:0003501',
    'maximum-phonation-time': 'OBI:0003501',
    'maximum-phonation-time-v2': 'OBI:0003501',
    'open-response-questions': 'OBI:0003529',
    'picture-description': 'OBI:0003529',
    'picture-description-option1': 'OBI:0003529',
    'picture-description-option2': 'OBI:0003529',
    'productive-vocabulary': 'OBI:0003529',
    'prolonged-vowel': 'OBI:0003501',
    'rainbow-passage': 'OBI:0003505',
    'rainbowpassage': 'OBI:0003505',
    'random-item-generation': 'OBI:0003529',
    'random-item-generation-v2': 'OBI:0003529',
    'respiration-and-cough-breath': 'OBI:0003501',
    'respiration-and-cough-cough': 'OBI:0003504',
    'respiration-and-cough-fivebreaths': 'OBI:0003501',
    'respiration-and-cough-threequickbreaths': 'OBI:0003501',
    'respiration-and-cough-v2-breath': 'OBI:0003501',
    'respiration-and-cough-v2-hardcough': 'OBI:0003504',
    'respiration-and-cough-v2-threebreaths': 'OBI:0003501',
    'respiration-and-cough-v2-threebreathsmouth': 'OBI:0003501',
    'respiration-and-cough-v2-threebreathsnose': 'OBI:0003501',
    'story-recall': 'OBI:0003529',
    'story-recall-v2': 'OBI:0003529',
    'voluntary-cough': 'OBI:0003504',
    'word-color-stroop': 'OBI:0003529',
    '123s': 'OBI:0003529',
    'days': 'OBI:0003529',
    'favorite-food': 'OBI:0003529',
    'favorite-show-movie-game': 'OBI:0003529',
    'long-sounds': 'OBI:0003501',
    'months': 'OBI:0003529',
    'naming-animals': 'OBI:0003529',
    'naming-food': 'OBI:0003529',
    'noisy-sounds': 'OBI:0003501',
    'outside-of-school': 'OBI:0003529',
    'passage': 'OBI:0003505',
    'picture': 'OBI:0003529',
    'picture-description': 'OBI:0003529',
    'ready-for-school': 'OBI:0003529',
    'repeat-words': 'OBI:0003529',
    'role-namings': 'OBI:0003529',
    'sentence': 'OBI:0003529',
    'silly-sounds': 'OBI:0003501',
}

# Adult diagnosis file stem → list of DOID terms
# Each participant with a positive gold-standard diagnosis gets linked to these DOID(s).
# Use empty list [] to skip a condition until a DOID is confirmed.
# copd_and_asthma is handled specially in bundle_to_c2m2.py via diagnosis_ca_copd_asthma column.
adult_diagnosis_to_DOID = {
    'airway_stenosis': ['DOID:11527'],
    'amyotrophic_lateral_sclerosis': ['DOID:332'],
    'anxiety': ['DOID:2030'],
    'benign_lesions': [],
    'bipolar_disorder': ['DOID:3312'],
    'cognitive_impairment': ['DOID:10652'],
    'copd_and_asthma': [],  # handled via diagnosis_ca_copd_asthma; COPD=DOID:3083, asthma=DOID:2841
    'depression': ['DOID:1470'],
    'glottic_insufficiency': [],
    'laryngeal_cancer': ['DOID:2600'],
    'laryngeal_dystonia': ['DOID:0050835'],
    'laryngitis': ['DOID:3437'],
    'muscle_tension_dysphonia': [],
    'parkinsons_disease': ['DOID:14330'],
    'precancerous_lesions': [],
    'unexplained_chronic_cough': [],
    'unilateral_vocal_fold_paralysis': [],
}

# Pediatric condition column → list of DOID terms
# Column names are from phenotype/pediatric/pediatric_medical_conditions.tsv.
# Positive values expected to be 'Yes' / '1' / 'True' (case-insensitive).
# TODO: populate with correct column names and DOID terms.
peds_condition_to_DOID = {
    # Example: 'peds_adhd': ['DOID:????'],
}

file_format_map = {
    '.tsv': 'format:3475',
    '.json': 'format:3464',
    '.parquet': '',
    '.txt': 'format:1964'
}

data_type_map = {
    '.tsv': 'data:3671',
    '.json': 'data:3671',
    '.txt': 'data:3671',
    '.parquet': ''
    }

mime_type_map = {
    '.tsv': 'text/tab-separated-values',
    '.json':' application/json',
    '.txt': 'text/plain',
    '.parquet': 'application/octet-stream'
}
