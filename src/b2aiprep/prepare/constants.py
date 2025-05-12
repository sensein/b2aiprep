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

def _load_participant_exclusions() -> t.List[str]:
    """Load the participant IDs to exclude from the dataset.

    Returns
    -------
    list
        The participant IDs to exclude.
    """
    b2ai_resources = files("b2aiprep").joinpath("prepare").joinpath("resources")
    if b2ai_resources.joinpath("participant_id_to_exclude_v2.json").exists():
        return json.loads(b2ai_resources.joinpath("participant_id_to_exclude_v2.json").read_text())
    return json.loads(b2ai_resources.joinpath("participant_id_to_exclude.json").read_text())

PARTICIPANT_ID_TO_REMOVE: t.List[str] = _load_participant_exclusions()

def _load_audio_filestem_exclusions() -> t.List[str]:
    """Load the audio filestems to exclude from the dataset.

    Returns
    -------
    list
        The audio filestems to exclude.
    """
    b2ai_resources = files("b2aiprep").joinpath("prepare").joinpath("resources")
    if b2ai_resources.joinpath("audio_filestem_to_exclude.json").exists():
        return json.loads(b2ai_resources.joinpath("audio_filestem_to_exclude.json").read_text())
    return []

AUDIO_FILESTEMS_TO_REMOVE: t.List[str] = _load_audio_filestem_exclusions()

class Instrument(BaseModel):
    """Instruments are associated with fixed sets of columns and a string
    value to subselect RedCap CSV rows."""

    session_id: str
    name: str
    text: str
    schema_name: str
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
        schema_name="subjectparticipantbasicinformationschema",
    )
    ELIGIBILITY = Instrument(
        session_id="record_id",
        name="eligibility",
        text="Participant",
        schema_name="subjectparticipanteligiblestudiesschema",
    )
    ENROLLMENT = Instrument(
        session_id="record_id",
        name="enrollment",
        text="Participant",
        schema_name="enrollmentformschema",
    )
    VOCAL_FOLD_PARALYSIS = Instrument(
        session_id="record_id",
        name="vocalFoldParalysis",
        text="Participant",
        schema_name="dvoicevocalfoldparalysisschema",
    )
    LARYNGEAL_DYSTONIA = Instrument(
        session_id="record_id",
        name="laryngealDystonia",
        text="Participant",
        schema_name="dvoicelaryngealdystoniaschema",
    )
    GLOTTIC_INSUFFICIENCY = Instrument(
        session_id="record_id",
        name="glotticinsufficiency",
        text="Participant",
        schema_name="dvoiceglotticinsufficiencypresbyphoniaschema",
    )
    PRECANCEROUS_LESIONS = Instrument(
        session_id="record_id",
        name="precancerousLesions",
        text="Participant",
        schema_name="dvoiceprecancerouslesionsschema",
    )
    LARYNGEAL_CANCER = Instrument(
        session_id="record_id",
        name="laryngealCancer",
        text="Participant",
        schema_name="dvoicelaryngealcancerschema",
    )
    BENIGN_LESION = Instrument(
        session_id="record_id",
        name="benignLesion",
        text="Participant",
        schema_name="dvoicebenignlesionsschema",
    )
    BIPOLAR = Instrument(
        session_id="record_id",
        name="bipolar",
        text="Participant",
        schema_name="dmoodbipolardisorderschema",
    )
    ANXIETY = Instrument(
        session_id="record_id",
        name="anxiety",
        text="Participant",
        schema_name="dmoodanxietydisorderschema",
    )
    DEPRESSION = Instrument(
        session_id="record_id",
        name="depression",
        text="Participant",
        schema_name="dmooddepressionormajordepressivedisorderschema",
    )
    AIRWAY_STENOSIS = Instrument(
        session_id="record_id",
        name="airwaystenosis",
        text="Participant",
        schema_name="drespairwaystenosisschema",
    )
    ALZHEIMERS = Instrument(
        session_id="record_id",
        name="alzheimers",
        text="Participant",
        schema_name="dneuroalzheimersdiseasemildcognitiveimpairmeschema",
    )
    HUNTINGTONS = Instrument(
        session_id="record_id",
        name="huntingtons",
        text="Participant",
        schema_name="dneurohuntingtonsdiseaseschema",
    )
    PARKINSONS = Instrument(
        session_id="record_id",
        name="parkinsons",
        text="Participant",
        schema_name="dneuroparkinsonsdiseaseschema",
    )
    ALS = Instrument(
        session_id="record_id",
        name="als",
        text="Participant",
        schema_name="dneuroamyotrophiclateralsclerosisalsschema",
    )

    CHRONIC_COUGH = Instrument(
        session_id="record_id",
        name="chronicCough",
        text="Participant",
        schema_name="drespunexplainedchroniccoughschema",
    )

    MUSCLE_TENSION = Instrument(
        session_id="record_id",
        name="muscleTension",
        text="Participant",
        schema_name="dvoicemuscletensiondysphoniamtdschema",
    )

    LARYNGITIS = Instrument(
        session_id="record_id",
        name="laryngitis",
        text="Participant",
        schema_name="dvoicelaryngitisschema",
    )

    MEDICAL_CONDITIONS = Instrument(
        session_id="medical_conditions_schema_session_id",
        name="medical_conditions",
        text="medical_conditions",
        schema_name="medical_conditions",
    )

    PEDS_VHI10 = Instrument(
        session_id="peds_vhi_10_schema_session_id",
        name="pedsvhi10",
        text="pedsvhi10",
        schema_name="pedsvhi10",
    )

    QOL = Instrument(
        session_id="peds_voice_related_qol_schema_session_id",
        name="peds_voice_related_qol",
        text="peds_voice_related_qol",
        schema_name="peds_voice_related_qol",
    )

    PHQA = Instrument(
        session_id="phqa_schema_session_id",
        name="phqa_schema",
        text="phqa_schema",
        schema_name="phqa_schema",
    )

    PEDS_DEMOGRAPHICS = Instrument(
        session_id="record_id",
        name="q_generic_demographics",
        text="q_generic_demographics",
        schema_name="q_generic_demographics",
    )

    PEDS_BASIC_INFO = Instrument(
        session_id="record_id",
        name="subjectparticipant_basic_information",
        text="Participant",
        schema_name="subjectparticipant_basic_information",
    )

    PEDS_CONTACT_INFO = Instrument(
        session_id="record_id",
        name="subjectparticipant_contact_information",
        text="subjectparticipant_contact_information",
        schema_name="subjectparticipant_contact_information",
    )

    PEDS_OUTCOME_SURVEY = Instrument(
        session_id="record_id",
        name="peds_voice_outcome_survey",
        text="peds_voice_outcome_survey",
        schema_name="peds_voice_outcome_survey",
    )

    # data where the row has a specific repeat instrument, filtered to by the text argument
    SESSION = Instrument(
        session_id="session_id", name="sessions", text="Session", schema_name="sessionschema"
    )
    ACOUSTIC_TASK = Instrument(
        session_id="acoustic_task_id",
        name="acoustic_tasks",
        text="Acoustic Task",
        schema_name="acoustictaskschema",
    )
    RECORDING = Instrument(
        session_id="recording_id",
        name="recordings",
        text="Recording",
        schema_name="recordingschema",
    )
    GENERIC_DEMOGRAPHICS = Instrument(
        session_id="demographics_session_id",
        name="demographics",
        text="Q Generic Demographics",
        schema_name="qgenericdemographicsschema",
    )
    GENERIC_CONFOUNDERS = Instrument(
        session_id="confounders_session_id",
        name="confounders",
        text="Q Generic Confounders",
        schema_name="qgenericconfoundersschema",
    )
    GENERIC_VOICE_PERCEPTION = Instrument(
        session_id="voice_perception_session_id",
        name="voicePerception",
        text="Q Generic Voice Perception",
        schema_name="qgenericvoiceperceptionschema",
    )
    GENERIC_VOICE_HANDICAP = Instrument(
        session_id="vhi_session_id",
        name="vhi10",
        text="Q Generic Voice Handicap Index Vhi10",
        schema_name="qgenericvoicehandicapindexvhi10schema",
    )
    GENERIC_PHQ9_DEPRESSION = Instrument(
        session_id="phq_9_session_id",
        name="phq9",
        text="Q Generic Patient Health Questionnaire9",
        schema_name="qgenericpatienthealthquestionnaire9schema",
    )
    GENERIC_GAD7_ANXIETY = Instrument(
        session_id="gad_7_session_id",
        name="gad7",
        text="Q - Generic - Gad7 Anxiety",
        schema_name="qgenericgad7anxietyschema",
    )
    RESP_DYSPNEA_INDEX = Instrument(
        session_id="dyspnea_index_session_id",
        name="dyspnea",
        text="Q - Resp - Dyspnea Index Di",
        schema_name="qrespdyspneaindexdischema",
    )
    RESP_LEICESTER_COUGH = Instrument(
        session_id="leicester_cough_session_id",
        name="leicester",
        text="Q - Resp - Leicester Cough Questionnaire Lcq",
        schema_name="qrespleicestercoughquestionnairelcqschema",
    )
    VOICE_VOICE_PROBLEM_SEVERITY = Instrument(
        session_id="voice_severity_session_id",
        name="voiceSeverity",
        text="Q - Voice - Voice Problem Severity",
        schema_name="qvoicevoiceproblemseverityschema",
    )
    MOOD_PANAS = Instrument(
        session_id="panas_session_id",
        name="panas",
        text="Q - Mood - Panas",
        schema_name="qmoodpanasschema",
    )
    MOOD_PARTICIPANT_HISTORY = Instrument(
        session_id="mph_session_id",
        name="mood_participant_history",
        text="Q - Mood - Participant History",
        schema_name="qmoodparticipanthistoryschema",
    )
    MOOD_CUSTOM_AFFECT = Instrument(
        session_id="custom_affect_scale_session_id",
        name="customAffectScale",
        text="Q - Mood - Custom Affect Scale",
        schema_name="qmoodcustomaffectscaleschema",
    )
    MOOD_DSM5_ADULT = Instrument(
        session_id="dsm_5_session_id",
        name="dsm5",
        text="Q - Mood - Dsm5 Adult",
        schema_name="qmooddsm5adultschema",
    )
    MOOD_PTSD_ADULT = Instrument(
        session_id="ptsd_session_id",
        name="ptsd",
        text="Q - Mood - Ptsd Adult",
        schema_name="qmoodptsdadultschema",
    )
    MOOD_ADHD_ADULT = Instrument(
        session_id="adhd_session_id",
        name="adhd",
        text="Q - Mood - Adhd Adult",
        schema_name="qmoodadhdadultschema",
    )
    NEURO_WINOGRAD_SCHEMAS = Instrument(
        session_id="winograd_session_id",
        name="winograd",
        text="Q - Neuro Winograd Schemas",
        schema_name="qneurowinogradschemasschema",
    )
    NEURO_WORDCOLOR_STROOP = Instrument(
        session_id="stroop_session_id",
        name="stroop",
        text="Q - Neuro - Wordcolor Stroop",
        schema_name="qneurowordcolorstroopschema",
    )
    NEURO_PRODUCTIVE_VOCABULARY = Instrument(
        session_id="vocabulary_session_id",
        name="vocab",
        text="Q - Neuro - Productive Vocabulary",
        schema_name="qneuroproductivevocabularyschema",
    )
    NEURO_RANDOM_ITEM_GENERATION = Instrument(
        session_id="random_session_id",
        name="random",
        text="Q - Neuro - Random Item Generation",
        schema_name="qneurorandomitemgenerationschema",
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
