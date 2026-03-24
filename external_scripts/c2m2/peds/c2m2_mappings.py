
race_map = {
        'White': 'cfde_subject_race:3',
        'Black or African American': 'cfde_subject_race:2',
        'Asian': 'cfde_subject_race:5',
        'American Indian or Alaska Native': 'cfde_subject_race:0',
        'Native Hawaiian or Other Pacific Islander': 'cfde_subject_race:6',
        'Other': 'cfde_subject_race:4',
        'Prefer not to answer': ''
        }


sex_map = {
  '': 'cfde_subject_sex:0',
  'Female gender identity':'cfde_subject_sex:1',
  'Male gender identity': 'cfde_subject_sex:2',
  'Other': 'cfde_subject_sex:0'
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
