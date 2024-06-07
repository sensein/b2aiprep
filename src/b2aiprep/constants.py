from enum import Enum
from importlib.resources import files
import json
import typing as t

from pydantic import BaseModel, model_validator, root_validator

GENERAL_QUESTIONNAIRES = [
    "participant",
    "eligibility",
    "enrollment",
    "vocalFoldParalysis",
    "laryngealDystonia",
    "precancerousLesions",
    "laryngealCancer",
    "benignLesion",
    "bipolar",
    "depression",
    "airwaystenosis",
    "alzheimers",
    "parkinsons",
    "als"
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

class Instrument(BaseModel):
    """Instruments are associated with fixed sets of columns and a string
    value to subselect RedCap CSV rows."""
    session_id: str
    name: str
    text: str
    schema: str
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
        b2ai_resources = files("b2aiprep").joinpath("resources", "instrument_columns")
        return json.loads(b2ai_resources.joinpath(f"{self.name}.json").read_text())

    def get_columns(self, add_record_id: bool=True) -> t.List[str]:
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
            if 'record_id' not in columns:
                columns.insert(0, 'record_id')
        return columns

    def __init__(self, **data):
        super().__init__(**data)
        # trigger the loading of the columns with a call to the property
        self.columns

class RepeatInstrument(Enum):
    # instruments which use the general row of the redcap CSV export
    PARTICIPANT = Instrument(session_id='record_id', name='participant', text='Participant', schema='subjectparticipantbasicinformationschema')
    ELIGIBILITY = Instrument(session_id='record_id', name='eligibility', text='Participant', schema='subjectparticipanteligiblestudiesschema')
    ENROLLMENT = Instrument(session_id='record_id', name='enrollment', text='Participant', schema='enrollmentformschema')
    VOCAL_FOLD_PARALYSIS = Instrument(session_id='record_id', name='vocalFoldParalysis', text='Participant', schema='dvoicevocalfoldparalysisschema')
    LARYNGEAL_DYSTONIA = Instrument(session_id='record_id', name='laryngealDystonia', text='Participant', schema='dvoicelaryngealdystoniaschema')
    PRECANCEROUS_LESIONS = Instrument(session_id='record_id', name='precancerousLesions', text='Participant', schema='dvoiceprecancerouslesionsschema')
    LARYNGEAL_CANCER = Instrument(session_id='record_id', name='laryngealCancer', text='Participant', schema='dvoicelaryngealcancerschema')
    BENIGN_LESION = Instrument(session_id='record_id', name='benignLesion', text='Participant', schema='dvoicebenignlesionsschema')
    BIPOLAR = Instrument(session_id='record_id', name='bipolar', text='Participant', schema='dmoodbipolardisorderschema')
    DEPRESSION = Instrument(session_id='record_id', name='depression', text='Participant', schema='dmooddepressionormajordepressivedisorderschema')
    AIRWAY_STENOSIS = Instrument(session_id='record_id', name='airwaystenosis', text='Participant', schema='drespairwaystenosisschema')
    ALZHEIMERS = Instrument(session_id='record_id', name='alzheimers', text='Participant', schema='dneuroalzheimersdiseasemildcognitiveimpairmeschema')
    PARKINSONS = Instrument(session_id='record_id', name='parkinsons', text='Participant', schema='dneuroparkinsonsdiseaseschema')
    ALS = Instrument(session_id='record_id', name='als', text='Participant', schema='dneuroamyotrophiclateralsclerosisalsschema')
    # data where the row has a specific repeat instrument, filtered to by the text argument
    SESSION = Instrument(session_id='session_id', name='sessions', text='Session', schema='sessionschema')
    ACOUSTIC_TASK = Instrument(session_id='acoustic_task_id', name='acoustic_tasks', text='Acoustic Task', schema='acoustictaskschema')
    RECORDING = Instrument(session_id='recording_id', name='recordings', text='Recording', schema='recordingschema')
    GENERIC_DEMOGRAPHICS = Instrument(session_id='demographics_session_id', name='demographics', text='Q Generic Demographics', schema='qgenericdemographicsschema')
    GENERIC_CONFOUNDERS = Instrument(session_id='confounders_session_id', name='confounders', text='Q Generic Confounders', schema='qgenericconfoundersschema')
    GENERIC_VOICE_PERCEPTION = Instrument(session_id='voice_perception_session_id', name='voicePerception', text='Q Generic Voice Perception', schema='qgenericvoiceperceptionschema')
    GENERIC_VOICE_HANDICAP = Instrument(session_id='vhi_session_id', name='vhi10', text='Q Generic Voice Handicap Index Vhi10', schema='qgenericvoicehandicapindexvhi10schema')
    GENERIC_PHQ9_DEPRESSION = Instrument(session_id='phq_9_session_id', name='phq9', text='Q Generic Patient Health Questionnaire9', schema='qgenericpatienthealthquestionnaire9schema')
    GENERIC_GAD7_ANXIETY = Instrument(session_id='gad_7_session_id', name='gad7', text='Q - Generic - Gad7 Anxiety', schema='qgenericgad7anxietyschema')
    RESP_DYSPNEA_INDEX = Instrument(session_id='dyspnea_index_session_id', name='dyspnea', text='Q - Resp - Dyspnea Index Di', schema='qrespdyspneaindexdischema')
    RESP_LEICESTER_COUGH = Instrument(session_id='leicester_cough_session_id', name='leicester', text='Q - Resp - Leicester Cough Questionnaire Lcq', schema='qrespleicestercoughquestionnairelcqschema')
    VOICE_VOICE_PROBLEM_SEVERITY = Instrument(session_id='voice_severity_session_id', name='voiceSeverity', text='Q - Voice - Voice Problem Severity', schema='qvoicevoiceproblemseverityschema')
    MOOD_PANAS = Instrument(session_id='panas_session_id', name='panas', text='Q - Mood - Panas', schema='qmoodpanasschema')
    MOOD_CUSTOM_AFFECT = Instrument(session_id='custom_affect_scale_session_id', name='customAffectScale', text='Q - Mood - Custom Affect Scale', schema='qmoodcustomaffectscaleschema')
    MOOD_DSM5_ADULT = Instrument(session_id='dsm_5_session_id', name='dsm5', text='Q - Mood - Dsm5 Adult', schema='qmooddsm5adultschema')
    MOOD_PTSD_ADULT = Instrument(session_id='ptsd_session_id', name='ptsd', text='Q - Mood - Ptsd Adult', schema='qmoodptsdadultschema')
    MOOD_ADHD_ADULT = Instrument(session_id='adhd_session_id', name='adhd', text='Q - Mood - Adhd Adult', schema='qmoodadhdadultschema')
    NEURO_WINOGRAD_SCHEMAS = Instrument(session_id='winograd_session_id', name='winograd', text='Q - Neuro Winograd Schemas', schema='qneurowinogradschemasschema')
    NEURO_WORDCOLOR_STROOP = Instrument(session_id='stroop_session_id', name='stroop', text='Q - Neuro - Wordcolor Stroop', schema='qneurowordcolorstroopschema')
    NEURO_PRODUCTIVE_VOCABULARY = Instrument(session_id='vocabulary_session_id', name='vocab', text='Q - Neuro - Productive Vocabulary', schema='qneuroproductivevocabularyschema')
    NEURO_RANDOM_ITEM_GENERATION = Instrument(session_id='random_session_id', name='random', text='Q - Neuro - Random Item Generation', schema='qneurorandomitemgenerationschema')

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
]

REPEAT_INSTRUMENT_PREFIX_MAPPING = {
    RepeatInstrument.SESSION: 'session_',
    RepeatInstrument.ACOUSTIC_TASK: 'acoustic_task_',
    RepeatInstrument.RECORDING: 'recording_',
}
