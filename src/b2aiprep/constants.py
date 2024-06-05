from enum import Enum
from importlib.resources import files
import json
import typing as t

GENERAL_QUESTIONNAIRES = [
    "participant",
    "eligibility",
    "enrollment",
    "vocalFoldParalysis",
    "laryngealdystonia",
    "precancerouseLesions",
    "laryngealcancer",
    "benignLesion",
    "bipolar",
    "depression",
    "airwaystenosis",
    "alzheimers",
    "parkinsons",
    "als"
]

VALIDATED_QUESTIONNAIRES = [
    "confounders",
    "demographics",
    "gad7",
    "phq9",
    "voicePerception",
    "vhi10",
    "panas",
    "customAffectScale",
    "dsm5",
    "ptsd",
    "adhd",
    "dyspnea",
    "leicester",
    "voiceSeverity",
    "winograd",
    "stroop",
    "vocab",
    "random"

]
AUDIO_TASKS = (
    'Animal fluency', 'Audio Check', 'Breath Sounds',
    'Cape V sentences', 'Caterpillar Passage', 'Cinderella Story',
    'Diadochokinesis', 'Free Speech', 'Glides', 'Loudness',
    'Maximum phonation time', 'Picture description',
    'Productive Vocabulary', 'Prolonged vowel', 'Rainbow Passage',
    'Random Item Generation', 'Respiration and cough', 'Story recall',
    'Voluntary Cough', 'Word-color Stroop',
)



class RepeatInstrument(Enum):
    SESSION = 'Session'
    ACOUSTIC_TASK = 'Acoustic Task'
    RECORDING = 'Recording'
    PARTICIPANT = 'Participant'
    GENERIC_DEMOGRAPHICS = 'Q Generic Demographics'
    GENERIC_CONFOUNDERS = 'Q Generic Confounders'
    GENERIC_VOICE_PERCEPTION = 'Q Generic Voice Perception'
    GENERIC_VOICE_HANDICAP = 'Q Generic Voice Handicap Index Vhi10'
    GENERIC_PHQ9_DEPRESSION = 'Q Generic Patient Health Questionnaire9'
    GENERIC_GAD7_ANXIETY = 'Q - Generic - Gad7 Anxiety'
    RESP_DYSPNEA_INDEX = 'Q - Resp - Dyspnea Index Di'
    RESP_LEICESTER_COUGH = 'Q - Resp - Leicester Cough Questionnaire Lcq'
    VOICE_VOICE_PROBLEM_SEVERITY = 'Q - Voice - Voice Problem Severity'
    MOOD_PANAS = 'Q - Mood - Panas'
    MOOD_CUSTOM_AFFECT = 'Q - Mood - Custom Affect Scale'
    MOOD_DSM5_ADULT = 'Q - Mood - Dsm5 Adult'
    MOOD_PTSD_ADULT = 'Q - Mood - Ptsd Adult'
    MOOD_ADHD_ADULT = 'Q - Mood - Adhd Adult'
    NEURO_WINOGRAD_SCHEMAS = 'Q - Neuro Winograd Schemas'
    NEURO_WORDCOLOR_STROOP = 'Q - Neuro - Wordcolor Stroop'
    NEURO_PRODUCTIVE_VOCABULARY = 'Q - Neuro - Productive Vocabulary'
    NEURO_RANDOM_ITEM_GENERATION = 'Q - Neuro - Random Item Generation'


REPEAT_INSTRUMENT_COLUMNS = {
    RepeatInstrument.PARTICIPANT: 'participant',
    RepeatInstrument.GENERIC_DEMOGRAPHICS: 'demographics',
    RepeatInstrument.GENERIC_CONFOUNDERS: 'confounders',
    RepeatInstrument.GENERIC_PHQ9_DEPRESSION: 'phq9',
    RepeatInstrument.GENERIC_GAD7_ANXIETY: 'gad7',
    RepeatInstrument.GENERIC_VOICE_PERCEPTION: 'voicePerception',
    RepeatInstrument.GENERIC_VOICE_HANDICAP: 'vhi10',
    RepeatInstrument.MOOD_PANAS: 'panas',
    RepeatInstrument.MOOD_CUSTOM_AFFECT: 'customAffectScale',
    RepeatInstrument.MOOD_DSM5_ADULT: 'dsm5',
    RepeatInstrument.MOOD_PTSD_ADULT: 'ptsd',
    RepeatInstrument.MOOD_ADHD_ADULT: 'adhd',
    RepeatInstrument.RESP_DYSPNEA_INDEX: 'dyspnea',
    RepeatInstrument.RESP_LEICESTER_COUGH: 'leicester',
    RepeatInstrument.VOICE_VOICE_PROBLEM_SEVERITY: 'voiceSeverity',
    RepeatInstrument.NEURO_WINOGRAD_SCHEMAS: 'winograd',
    RepeatInstrument.NEURO_WORDCOLOR_STROOP: 'stroop',
    RepeatInstrument.NEURO_PRODUCTIVE_VOCABULARY : 'vocab',
    RepeatInstrument.NEURO_RANDOM_ITEM_GENERATION : 'random',   
}

REPEAT_INSTRUMENT_SESSION_ID = {
    RepeatInstrument.PARTICIPANT: 'participant_session_id',
    RepeatInstrument.GENERIC_DEMOGRAPHICS: 'demographics_session_id',
    RepeatInstrument.GENERIC_CONFOUNDERS: 'confounders_session_id',
    RepeatInstrument.GENERIC_PHQ9_DEPRESSION: 'phq_9_session_id',
    RepeatInstrument.GENERIC_GAD7_ANXIETY: 'gad_7_session_id',
    RepeatInstrument.GENERIC_VOICE_PERCEPTION: 'voice_perception_session_id',
    RepeatInstrument.GENERIC_VOICE_HANDICAP: 'vhi_session_id',
    RepeatInstrument.MOOD_PANAS: 'panas_session_id',
    RepeatInstrument.MOOD_CUSTOM_AFFECT: 'custom_affect_scale_session_id',
    RepeatInstrument.MOOD_DSM5_ADULT: 'dsm_5_session_id',
    RepeatInstrument.MOOD_PTSD_ADULT: 'ptsd_session_id',
    RepeatInstrument.MOOD_ADHD_ADULT: 'adhd_session_id',
    RepeatInstrument.RESP_DYSPNEA_INDEX: 'dyspnea_index_session_id',
    RepeatInstrument.RESP_LEICESTER_COUGH: 'leicester_cough_session_id',
    RepeatInstrument.VOICE_VOICE_PROBLEM_SEVERITY: 'voice_severity_session_id',
    RepeatInstrument.NEURO_WINOGRAD_SCHEMAS: 'winograd_session_id',
    RepeatInstrument.NEURO_WORDCOLOR_STROOP: 'stroop_session_id',
    RepeatInstrument.NEURO_PRODUCTIVE_VOCABULARY: 'vocabulary_session_id',
    RepeatInstrument.NEURO_RANDOM_ITEM_GENERATION: 'random_session_id',
}

REPEAT_INSTRUMENT_PREFIX_MAPPING = {
    RepeatInstrument.SESSION: 'session_',
    RepeatInstrument.ACOUSTIC_TASK: 'acoustic_task_',
    RepeatInstrument.RECORDING: 'recording_',
}


def _load_data_columns() -> t.Dict[str, t.List[str]]:
    """Load the data columns for the Bridge2AI voice data.
    
    Returns
    -------
    dict
        A dictionary with the columns for each questionnaire.
    """
    b2ai_resources = files("b2aiprep").joinpath("resources")
    data_columns = {
        "columns": json.loads(b2ai_resources.joinpath("all_columns.json").read_text())
    }

    b2ai_resources = b2ai_resources.joinpath("instrument_columns")
    for questionnaire_name in VALIDATED_QUESTIONNAIRES:
        data_columns[questionnaire_name] = json.loads(
            b2ai_resources.joinpath(f"{questionnaire_name}.json").read_text()
        )

    return data_columns

DATA_COLUMNS = _load_data_columns()

def _load_questionnaire_mapping() -> dict:
    """Loads a fixed map from the string questionnaire "name" to the schema name.
     
    {"sessions": "sessionschema", ...}
    
    Returns
    -------
    dict
        The mapping from questionnaire name to schema name.
    """
    b2ai_resources = files("b2aiprep")
    return json.loads(b2ai_resources.joinpath("resources", "questionnaire2schema.json").read_text())

QUESTIONNAIRE_MAPPING = _load_questionnaire_mapping()
