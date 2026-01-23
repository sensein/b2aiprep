import json
import typing as t
from enum import Enum
from importlib.resources import files

from pydantic import BaseModel

FEATURE_EXTRACTION_SPEECH_RATE = (
    "speaking_rate",
    "articulation_rate",
    "phonation_ratio",
    "pause_rate",
    "mean_pause_duration",
)
FEATURE_EXTRACTION_DURATION = "duration"
FEATURE_EXTRACTION_PITCH_AND_INTENSITY = (
    "mean_f0_hertz",
    "std_f0_hertz",
    "mean_intensity_db",
    "std_intensity_db",
    "range_ratio_intensity_db",
)
FEATURE_EXTRACTION_HARMONIC_DESCRIPTORS = (
    "mean_hnr_db",
    "std_hnr_db",
    "spectral_slope",
    "spectral_tilt",
    "cepstral_peak_prominence_mean",
    "cepstral_peak_prominence_std",
)
FEATURE_EXTRACTION_FORMANTS = (
    "mean_f1_loc",
    "std_f1_loc",
    "mean_b1_loc",
    "std_b1_loc",
    "mean_f2_loc",
    "std_f2_loc",
    "mean_b2_loc",
    "std_b2_loc",
)
FEATURE_EXTRACTION_SPECTRAL_MOMENTS = (
    "spectral_gravity",
    "spectral_std_dev",
    "spectral_skewness",
    "spectral_kurtosis",
)
FEATURE_EXTRACTION_JITTER = (
    "local_jitter",
    "localabsolute_jitter",
    "rap_jitter",
    "ppq5_jitter",
    "ddp_jitter",
)
FEATURE_EXTRACTION_SHIMMER = (
    "local_shimmer",
    "localDB_shimmer",
    "apq3_shimmer",
    "apq5_shimmer",
    "apq11_shimmer",
    "dda_shimmer",
)

GENERAL_QUESTIONNAIRES = [
    "participant",
    "eligibility",
    "enrollment",
    "vocalFoldParalysis",
    "laryngealDystonia",
    "glotticinsufficiency",
    "precancerousLesions",
    "laryngealCancer",
    "benignLesion",
    "bipolar",
    "depression",
    "airwaystenosis",
    "alzheimers",
    "huntingtons",
    "parkinsons",
    "als",
    "chronicCough",
    "muscleTension",
    "laryngitis",
    "anxiety"
]

AUDIO_TASKS = (
    "Animal fluency",
    "Audio Check",
    "Breath Sounds",
    "Cape V sentences",
    "Caterpillar Passage",
    "Cinderella Story",
    "Diadochokinesis",
    "Free Speech",
    "Glides",
    "Loudness",
    "Maximum phonation time",
    "Picture description",
    "Productive Vocabulary",
    "Prolonged vowel",
    "Rainbow Passage",
    "Random Item Generation",
    "Respiration and cough",
    "Story recall",
    "Voluntary Cough",
    "Word-color Stroop",
)

SPEECH_TASKS = (
    "Animal fluency",
    "Cape V sentences",
    "Caterpillar Passage",
    "Cinderella Story",
    "Diadochokinesis",
    "Free Speech",
    "Picture description",
    "Productive Vocabulary",
    "Prolonged vowel",
    "Rainbow Passage",
    "Random Item Generation",
    "Story recall",
    "Word-color Stroop",
)

class Instrument(BaseModel):
    """Instruments are associated with fixed sets of columns and a string
    value to subselect RedCap CSV rows."""

    session_id: str
    name: str
    text: str
    schema_name_clobbered: str
    schema_name: str
    schema_name_pretty: str
    _columns: t.List[str] = []

    @property
    def columns(self):
        if len(self._columns) == 0:
            self._columns = self._load_instrument_columns()
        return self._columns

    def _load_instrument_columns(self) -> t.List[str]:
        """Load the data columns for the Bridge2AI voice data.

        Parameters
        ----------
        repeat_instrument : RepeatInstrument
            The repeat instrument for which to load the columns.

        Returns
        -------
        list
            The columns for the repeat instrument.
        """
        b2ai_resources = (
            files("b2aiprep").joinpath("prepare").joinpath("resources", "instrument_columns")
        )
        return json.loads(b2ai_resources.joinpath(f"{self.name}.json").read_text())

    def get_columns(self, add_record_id: bool = True) -> t.List[str]:
        """Get the set of string column names associated with an instrument.

        Parameters
        ----------
        add_record_id : bool, optional
            Whether to add the record_id column, by default True

        Returns
        -------
        List[str]
            The column names.
        """
        columns = self.columns
        if add_record_id:
            if "record_id" not in columns:
                columns.insert(0, "record_id")
        return columns

    def __init__(self, **data):
        super().__init__(**data)
        # trigger the loading of the columns with a call to the property
        self.columns


class RepeatInstrument(Enum):
    # instruments which use the general row of the redcap CSV export
    PARTICIPANT = Instrument(
        session_id="record_id",
        name="participant",
        text="Participant",
        schema_name_clobbered="subjectparticipantbasicinformationschema",
        schema_name="subject_participant_basic_information_schema",
        schema_name_pretty="Participant",
    )
    ELIGIBILITY = Instrument(
        session_id="record_id",
        name="eligibility",
        text="Participant",
        schema_name_clobbered="subjectparticipanteligiblestudiesschema",
        schema_name="subject_participant_eligible_studies_schema",
        schema_name_pretty="Participant Eligibility",
    )
    ENROLLMENT = Instrument(
        session_id="record_id",
        name="enrollment",
        text="Participant",
        schema_name_clobbered="enrollmentformschema",
        schema_name="enrollment_form_schema",
        schema_name_pretty="Participant Enrollment",
    )
    VOCAL_FOLD_PARALYSIS = Instrument(
        session_id="record_id",
        name="vocalFoldParalysis",
        text="Participant",
        schema_name_clobbered="dvoicevocalfoldparalysisschema",
        schema_name="d_voice_vocal_fold_paralysis_schema",
        schema_name_pretty="D Voice Vocal Fold Paralysis",
    )
    LARYNGEAL_DYSTONIA = Instrument(
        session_id="record_id",
        name="laryngealDystonia",
        text="Participant",
        schema_name_clobbered="dvoicelaryngealdystoniaschema",
        schema_name="d_voice_laryngeal_dystonia_schema",
        schema_name_pretty="D Voice Laryngeal Dystonia",
    )
    GLOTTIC_INSUFFICIENCY = Instrument(
        session_id="record_id",
        name="glotticinsufficiency",
        text="Participant",
        schema_name_clobbered="dvoiceglotticinsufficiencypresbyphoniaschema",
        schema_name="d_voice_glottic_insufficiency_presbyphonia_schema",
        schema_name_pretty="D Voice Glottic Insufficiency Presbyphonia",
    )
    PRECANCEROUS_LESIONS = Instrument(
        session_id="record_id",
        name="precancerousLesions",
        text="Participant",
        schema_name_clobbered="dvoiceprecancerouslesionsschema",
        schema_name="d_voice_precancerous_lesions_schema",
        schema_name_pretty="D Voice Precancerous Lesions",
    )
    LARYNGEAL_CANCER = Instrument(
        session_id="record_id",
        name="laryngealCancer",
        text="Participant",
        schema_name_clobbered="dvoicelaryngealcancerschema",
        schema_name="d_voice_laryngeal_cancer_schema",
        schema_name_pretty="D Voice Laryngeal Cancer",
    )
    BENIGN_LESION = Instrument(
        session_id="record_id",
        name="benignLesion",
        text="Participant",
        schema_name_clobbered="dvoicebenignlesionsschema",
        schema_name="d_voice_benign_lesions_schema",
        schema_name_pretty="D Voice Benign Lesions",
    )
    BIPOLAR = Instrument(
        session_id="record_id",
        name="bipolar",
        text="Participant",
        schema_name_clobbered="dmoodbipolardisorderschema",
        schema_name="d_mood_bipolar_disorder_schema",
        schema_name_pretty="D Mood Bipolar Disorder",
    )
    ANXIETY = Instrument(
        session_id="record_id",
        name="anxiety",
        text="Participant",
        schema_name_clobbered="dmoodanxietydisorderschema",
        schema_name="d_mood_anxiety_disorder_schema",
        schema_name_pretty="D Mood Anxiety Disorder",
    )
    DEPRESSION = Instrument(
        session_id="record_id",
        name="depression",
        text="Participant",
        schema_name_clobbered="dmooddepressionormajordepressivedisorderschema",
        schema_name="d_mood_depression_or_major_depressive_disorder_schema",
        schema_name_pretty="D Mood Depression or Major Depressive Disorder",
    )
    AIRWAY_STENOSIS = Instrument(
        session_id="record_id",
        name="airwaystenosis",
        text="Participant",
        schema_name_clobbered="drespairwaystenosisschema",
        schema_name="d_resp_airway_stenosis_schema",
        schema_name_pretty="D Resp Airway Stenosis",
    )
    ALZHEIMERS = Instrument(
        session_id="record_id",
        name="alzheimers",
        text="Participant",
        schema_name_clobbered="dneuroalzheimersdiseasemildcognitiveimpairmeschema",
        schema_name="d_neuro_alzheimers_disease_mild_cognitive_impairment_schema",
        schema_name_pretty="D Neuro Alzheimers Disease Mild Cognitive Impairment",
    )
    HUNTINGTONS = Instrument(
        session_id="record_id",
        name="huntingtons",
        text="Participant",
        schema_name_clobbered="dneurohuntingtonsdiseaseschema",
        schema_name="d_neuro_huntingtons_disease_schema",
        schema_name_pretty="D Neuro Huntingtons Disease",
    )
    PARKINSONS = Instrument(
        session_id="record_id",
        name="parkinsons",
        text="Participant",
        schema_name_clobbered="dneuroparkinsonsdiseaseschema",
        schema_name="d_neuro_parkinsons_disease_schema",
        schema_name_pretty="D Neuro Parkinsons Disease",
    )
    ALS = Instrument(
        session_id="record_id",
        name="als",
        text="Participant",
        schema_name_clobbered="dneuroamyotrophiclateralsclerosisalsschema",
        schema_name="d_neuro_amyotrophic_lateral_sclerosis_als_schema",
        schema_name_pretty="D Neuro Amyotrophic Lateral Sclerosis ALS",
    )

    CHRONIC_COUGH = Instrument(
        session_id="record_id",
        name="chronicCough",
        text="Participant",
        schema_name_clobbered="drespunexplainedchroniccoughschema",
        schema_name="d_resp_unexplained_chronic_cough_schema",
        schema_name_pretty="D Resp Unexplained Chronic Cough",
    )

    MUSCLE_TENSION = Instrument(
        session_id="record_id",
        name="muscleTension",
        text="Participant",
        schema_name_clobbered="dvoicemuscletensiondysphoniamtdschema",
        schema_name="d_voice_muscle_tension_dysphonia_mtd_schema",
        schema_name_pretty="D Voice Muscle Tension Dysphonia MTD",
    )

    LARYNGITIS = Instrument(
        session_id="record_id",
        name="laryngitis",
        text="Participant",
        schema_name_clobbered="dvoicelaryngitisschema",
        schema_name="d_voice_laryngitis_schema",
        schema_name_pretty="D Voice Laryngitis",
    )

    MEDICAL_CONDITIONS = Instrument(
        session_id="record_id",
        name="medical_conditions",
        text="medical_conditions",
        schema_name_clobbered="medical_conditions",
        schema_name="medical_conditions",
        schema_name_pretty="Medical Conditions",
    )

    PEDS_VHI10 = Instrument(
        session_id="record_id",
        name="pedsvhi10",
        text="pedsvhi10",
        schema_name_clobbered="pedsvhi10",
        schema_name="peds_vhi_10_schema",
        schema_name_pretty="Peds VHI-10",
    )

    QOL = Instrument(
        session_id="record_id",
        name="peds_voice_related_qol",
        text="peds_voice_related_qol",
        schema_name_clobbered="peds_voice_related_qol",
        schema_name="peds_voice_related_qol_schema",
        schema_name_pretty="Peds Voice Related QOL",
    )

    PHQA = Instrument(
        session_id="record_id",
        name="phqa_schema",
        text="phqa_schema",
        schema_name_clobbered="phqa_schema",
        schema_name="phqa_schema",
        schema_name_pretty="PHQA",
    )

    PEDS_DEMOGRAPHICS = Instrument(
        session_id="record_id",
        name="q_generic_demographics",
        text="q_generic_demographics",
        schema_name_clobbered="q_generic_demographics",
        schema_name="q_generic_demographics_schema",
        schema_name_pretty="Q - Generic - Demographics",
    )

    PEDS_BASIC_INFO = Instrument(
        session_id="record_id",
        name="subjectparticipant_basic_information",
        text="Participant",
        schema_name_clobbered="subjectparticipant_basic_information",
        schema_name="subject_participant_basic_information_schema",
        schema_name_pretty="Participant Basic Information",
    )

    PEDS_CONTACT_INFO = Instrument(
        session_id="record_id",
        name="subjectparticipant_contact_information",
        text="subjectparticipant_contact_information",
        schema_name_clobbered="subjectparticipant_contact_information",
        schema_name="subject_participant_contact_information_schema",
        schema_name_pretty="Participant Contact Information",
    )

    PEDS_OUTCOME_SURVEY = Instrument(
        session_id="record_id",
        name="peds_voice_outcome_survey",
        text="peds_voice_outcome_survey",
        schema_name_clobbered="peds_voice_outcome_survey",
        schema_name="peds_voice_outcome_survey_schema",
        schema_name_pretty="Peds Voice Outcome Survey",
    )

    # data where the row has a specific repeat instrument, filtered to by the text argument
    SESSION = Instrument(
        session_id="session_id", name="sessions", text="Session", schema_name_clobbered="sessionschema", schema_name="session_schema", schema_name_pretty="Session",
    )
    ACOUSTIC_TASK = Instrument(
        session_id="acoustic_task_id",
        name="acoustic_tasks",
        text="Acoustic Task",
        schema_name_clobbered="acoustictaskschema",
        schema_name="acoustic_task_schema",
        schema_name_pretty="Acoustic Task",
    )
    RECORDING = Instrument(
        session_id="recording_id",
        name="recordings",
        text="Recording",
        schema_name_clobbered="recordingschema",
        schema_name="recording_schema",
        schema_name_pretty="Recording",
    )
    GENERIC_DEMOGRAPHICS = Instrument(
        session_id="demographics_session_id",
        name="demographics",
        text="Q Generic Demographics",
        schema_name_clobbered="qgenericdemographicsschema",
        schema_name="q_generic_demographics_schema",
        schema_name_pretty="Q - Generic - Demographics",
    )
    GENERIC_CONFOUNDERS = Instrument(
        session_id="confounders_session_id",
        name="confounders",
        text="Q Generic Confounders",
        schema_name_clobbered="qgenericconfoundersschema",
        schema_name="q_generic_confounders_schema",
        schema_name_pretty="Q - Generic - Confounders",
    )
    GENERIC_VOICE_PERCEPTION = Instrument(
        session_id="voice_perception_session_id",
        name="voicePerception",
        text="Q Generic Voice Perception",
        schema_name_clobbered="qgenericvoiceperceptionschema",
        schema_name="q_generic_voice_perception_schema",
        schema_name_pretty="Q - Generic - Voice Perception",
    )
    GENERIC_VOICE_HANDICAP = Instrument(
        session_id="vhi_session_id",
        name="vhi10",
        text="Q Generic Voice Handicap Index Vhi10",
        schema_name_clobbered="qgenericvoicehandicapindexvhi10schema",
        schema_name="q_generic_voice_handicap_index_vhi_10_schema",
        schema_name_pretty="Q - Generic - VHI-10",
    )
    GENERIC_PHQ9_DEPRESSION = Instrument(
        session_id="phq_9_session_id",
        name="phq9",
        text="Q Generic Patient Health Questionnaire9",
        schema_name_clobbered="qgenericpatienthealthquestionnaire9schema",
        schema_name="q_generic_patient_health_questionnaire_9_schema",
        schema_name_pretty="Q - Generic - PHQ-9",
    )
    GENERIC_GAD7_ANXIETY = Instrument(
        session_id="gad_7_session_id",
        name="gad7",
        text="Q - Generic - Gad7 Anxiety",
        schema_name_clobbered="qgenericgad7anxietyschema",
        schema_name="q_generic_gad_7_anxiety_schema",
        schema_name_pretty="Q - Generic - GAD-7 Anxiety",
    )
    RESP_DYSPNEA_INDEX = Instrument(
        session_id="dyspnea_index_session_id",
        name="dyspnea",
        text="Q - Resp - Dyspnea Index Di",
        schema_name_clobbered="qrespdyspneaindexdischema",
        schema_name="q_resp_dyspnea_index_di_schema",
        schema_name_pretty="Q - Resp - Dyspnea Index (DI)",
    )
    RESP_LEICESTER_COUGH = Instrument(
        session_id="leicester_cough_session_id",
        name="leicester",
        text="Q - Resp - Leicester Cough Questionnaire Lcq",
        schema_name_clobbered="qrespleicestercoughquestionnairelcqschema",
        schema_name="q_resp_leicester_cough_questionnaire_lcq_schema",
        schema_name_pretty="Q - Resp - Leicester Cough Questionnaire (LCQ)",
    )
    VOICE_VOICE_PROBLEM_SEVERITY = Instrument(
        session_id="voice_severity_session_id",
        name="voiceSeverity",
        text="Q - Voice - Voice Problem Severity",
        schema_name_clobbered="qvoicevoiceproblemseverityschema",
        schema_name="q_voice_voice_problem_severity_schema",
        schema_name_pretty="Q - Voice - Voice Problem Severity",
    )
    MOOD_PANAS = Instrument(
        session_id="panas_session_id",
        name="panas",
        text="Q - Mood - Panas",
        schema_name_clobbered="qmoodpanasschema",
        schema_name="q_mood_panas_schema",
        schema_name_pretty="Q - Mood - PANAS",
    )
    MOOD_PARTICIPANT_HISTORY = Instrument(
        session_id="mph_session_id",
        name="mood_participant_history",
        text="Q - Mood - Participant History",
        schema_name_clobbered="qmoodparticipanthistoryschema",
        schema_name="q_mood_participant_history_schema",
        schema_name_pretty="Q - Mood - Participant History",
    )
    MOOD_CUSTOM_AFFECT = Instrument(
        session_id="custom_affect_scale_session_id",
        name="customAffectScale",
        text="Q - Mood - Custom Affect Scale",
        schema_name_clobbered="qmoodcustomaffectscaleschema",
        schema_name="q_mood_custom_affect_scale_schema",
        schema_name_pretty="Q - Mood - Custom Affect Scale",
    )
    MOOD_DSM5_ADULT = Instrument(
        session_id="dsm_5_session_id",
        name="dsm5",
        text="Q - Mood - Dsm5 Adult",
        schema_name_clobbered="qmooddsm5adultschema",
        schema_name="q_mood_dsm_5_adult_schema",
        schema_name_pretty="Q - Mood - DSM-5 Adult",
    )
    MOOD_PTSD_ADULT = Instrument(
        session_id="ptsd_session_id",
        name="ptsd",
        text="Q - Mood - Ptsd Adult",
        schema_name_clobbered="qmoodptsdadultschema",
        schema_name="q_mood_ptsd_adult_schema",
        schema_name_pretty="Q - Mood - PTSD Adult",
    )
    MOOD_ADHD_ADULT = Instrument(
        session_id="adhd_session_id",
        name="adhd",
        text="Q - Mood - Adhd Adult",
        schema_name_clobbered="qmoodadhdadultschema",
        schema_name="q_mood_adhd_adult_schema",
        schema_name_pretty="Q - Mood - ADHD Adult",
    )
    NEURO_WINOGRAD_SCHEMAS = Instrument(
        session_id="winograd_session_id",
        name="winograd",
        text="Q - Neuro Winograd Schemas",
        schema_name_clobbered="qneurowinogradschemasschema",
        schema_name="q_neuro_winograd_schemas_schema",
        schema_name_pretty="Q - Neuro - Winograd Schemas",
    )
    NEURO_WORDCOLOR_STROOP = Instrument(
        session_id="stroop_session_id",
        name="stroop",
        text="Q - Neuro - Wordcolor Stroop",
        schema_name_clobbered="qneurowordcolorstroopschema",
        schema_name="q_neuro_word_color_stroop_schema",
        schema_name_pretty="Q - Neuro - Wordcolor Stroop",
    )
    NEURO_PRODUCTIVE_VOCABULARY = Instrument(
        session_id="vocabulary_session_id",
        name="vocab",
        text="Q - Neuro - Productive Vocabulary",
        schema_name_clobbered="qneuroproductivevocabularyschema",
        schema_name="q_neuro_productive_vocabulary_schema",
        schema_name_pretty="Q - Neuro - Productive Vocabulary",
    )
    NEURO_RANDOM_ITEM_GENERATION = Instrument(
        session_id="random_session_id",
        name="random",
        text="Q - Neuro - Random Item Generation",
        schema_name_clobbered="qneurorandomitemgenerationschema",
        schema_name="q_neuro_random_item_generation_schema",
        schema_name_pretty="Q - Neuro - Random Item Generation",
    )


VALIDATED_QUESTIONNAIRES = [
    RepeatInstrument.GENERIC_DEMOGRAPHICS,
    RepeatInstrument.GENERIC_CONFOUNDERS,
    RepeatInstrument.GENERIC_VOICE_PERCEPTION,
    RepeatInstrument.GENERIC_VOICE_HANDICAP,
    RepeatInstrument.GENERIC_PHQ9_DEPRESSION,
    RepeatInstrument.GENERIC_GAD7_ANXIETY,
    RepeatInstrument.RESP_DYSPNEA_INDEX,
    RepeatInstrument.RESP_LEICESTER_COUGH,
    RepeatInstrument.VOICE_VOICE_PROBLEM_SEVERITY,
    RepeatInstrument.MOOD_PANAS,
    RepeatInstrument.MOOD_CUSTOM_AFFECT,
    RepeatInstrument.MOOD_DSM5_ADULT,
    RepeatInstrument.MOOD_PTSD_ADULT,
    RepeatInstrument.MOOD_ADHD_ADULT,
    RepeatInstrument.NEURO_WINOGRAD_SCHEMAS,
    RepeatInstrument.NEURO_WORDCOLOR_STROOP,
    RepeatInstrument.NEURO_PRODUCTIVE_VOCABULARY,
    RepeatInstrument.NEURO_RANDOM_ITEM_GENERATION,
    RepeatInstrument.HUNTINGTONS,
    RepeatInstrument.GLOTTIC_INSUFFICIENCY,
    RepeatInstrument.MUSCLE_TENSION,
    RepeatInstrument.MEDICAL_CONDITIONS,
    RepeatInstrument.PEDS_VHI10,
    RepeatInstrument.QOL,
    RepeatInstrument.PHQA,
    RepeatInstrument.PEDS_DEMOGRAPHICS,
    RepeatInstrument.PEDS_BASIC_INFO,
    RepeatInstrument.PEDS_CONTACT_INFO,
    RepeatInstrument.PEDS_OUTCOME_SURVEY,
]

REPEAT_INSTRUMENT_PREFIX_MAPPING = {
    RepeatInstrument.SESSION: "session_",
    RepeatInstrument.ACOUSTIC_TASK: "acoustic_task_",
    RepeatInstrument.RECORDING: "recording_",
}
