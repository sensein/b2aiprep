
SUBJECT_GRANULARITY = 'cfde_subject_granularity:0'
ID_NAMESPACE = 'bridge2ai_voice'
PROJECT_ID_NAMESPACE = 'bridge2ai_voice'
SUBJECT_ID_NAMESPACE = 'bridge2ai_voice'
PEDS_SUBJECT_ID_NAMESPACE = 'bridge2ai_voice'

# Tier-neutral parent projects own subjects and biosamples. Subjects and
# biosamples are independent of access tier (a recording is the same biosample
# whether reached via registered features or controlled raw audio), so they live
# on the cohort parent rather than being pinned to the registered leaf.
PROJECT_LOCAL_ID = 'adult'
PEDS_PROJECT_LOCAL_ID = 'peds'

# Leaf projects own files, partitioned by access tier.
ADULT_REGISTERED_PROJECT_LOCAL_ID = 'adult_registered'
ADULT_CONTROLLED_PROJECT_LOCAL_ID = 'adult_controlled'
PEDS_REGISTERED_PROJECT_LOCAL_ID = 'peds_registered'
PEDS_CONTROLLED_PROJECT_LOCAL_ID = 'peds_controlled'
