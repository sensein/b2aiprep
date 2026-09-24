"""
Utilities for extracting subsets of data from a BIDS-like formatted dataset.

The BIDS-like format assumed is in the following structure. Let's assume we have
a participant (p1) and they have multiple sessios (s1, s2, s3). Then the BIDS-like
structure is:

sub-p1/
    ses-s1/
        beh/
            sub-p1_ses-s1_session-questionnaire-a.json
            sub-p1_ses-s1_session-questionnaire-b.json
        sub-p1_ses-s1_sessionschema.json
    sub-p1_subject-questionnaire-a.json
    sub-p1_subject-questionnaire-b.json
    ...
"""

from copy import copy, deepcopy
from functools import partial
import datetime
import logging
import os
import re
import shutil
import enum
import typing as t
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from importlib.resources import files
from importlib import resources
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from fhir.resources.questionnaireresponse import QuestionnaireResponse
from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from soundfile import LibsndfileError
from tqdm import tqdm

from b2aiprep.prepare.constants import RepeatInstrument, Instrument
from b2aiprep.prepare.date_shift import shift_dates
from b2aiprep.prepare.update import build_activity_payload
from b2aiprep.prepare.utils import (
    copy_package_resource,
    get_commit_sha,
    canonical_task_entity,
    normalize_task_label,
    sanitize_task_entity_in_bids_stem,
    AUDIO_CHECK_LABEL,
    is_audio_check,
    TaskMatcher,
)
from b2aiprep.prepare.fhir_utils import convert_response_to_bids_metadata, _population_from_cohort, _is_present, _language_from_selected
from b2aiprep.prepare.prepare import (
    get_value_from_metadata,
    remap_id, 
    update_metadata_record_and_session_id,
    reduce_id_length
)
from b2aiprep.prepare.bids import get_paths
from pydantic import BaseModel
from b2aiprep.prepare.redcap import RedCapDataset, _dropped_source_columns

_LOGGER = logging.getLogger(__name__)


class DispositionLevel(enum.Enum):
    """Threshold for which disposition levels survive column dropping.

    The hierarchy is RELEASE < REVIEW < INTERNAL.  Passing a level keeps
    columns at that level and below, dropping everything above.
    """
    RELEASE = "release"
    REVIEW = "review"
    INTERNAL = "internal"


DEFAULT_RESAMPLE_RATE = 16000
DEFAULT_BIT_DEPTH = 16


def _guard_resample_overshoot(resampled_audio, in_peak: float):
    """Prevent the 16-bit save from silently hard-clamping resample overshoot.

    Resampling can push samples past full-scale (|x| > 1); writing those to PCM
    would clamp them destructively with no error. If that happened, rescale the
    resampled waveform down to the input's peak so nothing exceeds [-1, 1] --
    preserving gain and waveform shape and introducing no clipping. No-op when the
    resampled peak is already in range. Returns (audio, scale) with scale=None when
    unchanged."""
    wf = resampled_audio.waveform
    if wf.numel() == 0:
        return resampled_audio, None
    out_peak = float(wf.abs().max())
    if out_peak <= 1.0:
        return resampled_audio, None
    scale = min(1.0, float(in_peak)) / out_peak
    return (
        Audio(waveform=wf * scale, sampling_rate=resampled_audio.sampling_rate,
              metadata=resampled_audio.metadata),
        scale,
    )

# Sensitive audio feature content that must not be present for sensitive tasks.
#
# This is a grouped spec because the per-record feature `.pt` files are nested dicts
# (e.g., `features["torchaudio"]["mfcc"]`), while some sensitive artifacts live at
# the top-level (e.g., `features["transcription"]`).
# A full wav header is 44 bytes and even a 1-second 16kHz mono clip is ~32 KB.
# Anything under this threshold is not a usable recording (the v4 exports contain
# 259 sources at exactly 4096 bytes -- collection-side non-recordings).
_MIN_AUDIO_BYTES = 8192

_SENSITIVE_FEATURES_REMOVED_FROM_BUNDLE: t.Mapping[str, t.FrozenSet[str]] = {
    "torchaudio": frozenset({"mel_filter_bank", "mfcc", "mel_spectrogram", "spectrogram"}),
    "": frozenset({"ppgs", "transcription"}),
    "sparc": frozenset({"ema"}),
}


def _note_entity_collision(
    seen: t.Dict[str, t.Any], entity: str, identifier: t.Any
) -> t.Tuple[bool, t.Any]:
    """Record `identifier` under `entity`; report whether a different record already claimed it.

    Returns (collided, prior_identifier). `collided` is True when `entity` has been
    seen before for a record that is not provably the same one.

    Membership is tested with `in`, never with a truthiness or `is not None` check on
    the stored value: an id that is missing (None/NaN/"") must still mark the entity as
    seen. Testing the value instead would read "seen, with a missing id" as "not seen
    yet" and silently skip the warning for a real collision -- the exact failure these
    guards exist to catch. Two ids are treated as the same record only when both are
    present and equal, so NaN != NaN cannot manufacture a false collision either.
    """
    if entity not in seen:
        seen[entity] = identifier
        return False, None
    prior = seen[entity]
    if _is_present(prior) and _is_present(identifier) and prior == identifier:
        return False, prior
    return True, prior


def _remove_sensitive_features_from_feature_payload(
    features: t.MutableMapping[str, t.Any],
    spec: t.Mapping[str, t.AbstractSet[str]] = _SENSITIVE_FEATURES_REMOVED_FROM_BUNDLE,
) -> None:
    """Remove sensitive feature content from an in-memory feature payload in-place."""

    for group, keys_to_remove in spec.items():
        if group == "":
            for key in keys_to_remove:
                features.pop(key, None)
            continue

        grouped_payload = features.get(group)
        if isinstance(grouped_payload, dict):
            for key in keys_to_remove:
                grouped_payload.pop(key, None)

def _copy_audio_files_parallel(copy_tasks: t.List[t.Tuple[Path, Path]], max_workers: int = 16, sanitize_audio_format: bool = False):
    """Copy audio files in parallel using ThreadPoolExecutor.
    
    Args:
        copy_tasks: List of (source_path, dest_path) tuples
        max_workers: Number of parallel worker threads
    """
    
    def copy_one_file(src: Path, dst: Path) -> t.Optional[str]:
        """Copy a single file, return error message if failed."""
        try:
            if sanitize_audio_format:
                src_audio = Audio(filepath=src)
                downmixed_audio = downmix_audios_to_mono([src_audio])[0]
                audio_16k = resample_audios([downmixed_audio], DEFAULT_RESAMPLE_RATE)[0]
                # Guard against the 16-bit save silently clamping resample overshoot.
                in_peak = float(downmixed_audio.waveform.abs().max())
                audio_16k, scale = _guard_resample_overshoot(audio_16k, in_peak)
                if scale is not None:
                    _LOGGER.warning(
                        "Resample overshoot for %s; rescaled x%.5f to input peak %.4f "
                        "to avoid destructive 16-bit clamp.", src, scale, in_peak,
                    )
                audio_16k.save_to_file(dst,bits_per_sample=DEFAULT_BIT_DEPTH)
            else:
                shutil.copyfile(src, dst)
            return None
        except Exception as e:
            return f"Failed to copy {src} -> {dst}: {e}"
    
    if not copy_tasks:
        return
    
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(copy_one_file, src, dst): (src, dst) for src, dst in copy_tasks}
        
        for future in as_completed(futures):
            error = future.result()
            if error:
                errors.append(error)
                _LOGGER.error(error)
    
    if errors:
        failed_files = [err.split("->")[0].replace("Failed to copy", "").strip() for err in errors]
        total_files = len(copy_tasks)
        unique_error_types = set()
        for err in errors:
            if ":" in err:
                unique_error_types.add(err.split(":")[-1].strip())
        _LOGGER.warning(
            "Encountered %d errors out of %d files during parallel audio copying. "
            "Error types: %s",
            len(errors), total_files, list(unique_error_types),
        )
        _LOGGER.warning(
            "All %d failed audio files (QA -- verify these are expected; "
            "truncated/corrupt sources should be caught by the pre-scan instead): %s",
            len(failed_files), failed_files,
        )


class BIDSDataset:
    def __init__(self, data_path: t.Union[Path, str, os.PathLike]):
        self.data_path = Path(data_path).resolve()
    
    @classmethod
    def from_redcap(
        cls,
        redcap_dataset: RedCapDataset,
        outdir: t.Union[str, Path],
        audiodir: t.Optional[t.Union[str, Path]] = None,
        max_audio_workers: int = 16,
        sanitize_audio_format: bool = False,
        drop_audio_check: bool = True,
        date_shift_anchor: t.Optional[datetime.date] = None,
        date_shift_log: t.Optional[t.Union[str, Path]] = None,
        skip_audio_copy: bool = False,
    ) -> 'BIDSDataset':
        """
        Create a BIDSDataset by converting a RedCapDataset to BIDS format.
        
        Args:
            redcap_dataset: The RedCapDataset to convert
            outdir: Output directory for BIDS structure
            audiodir: Optional directory containing audio files
            max_audio_workers: Number of parallel threads for audio copying (default: 16)
            sanitize_audio_format: Whether to standardize the audio to 16KHz and mono-channel
            drop_audio_check: Exclude the session microphone check from every output (default
                True). Pass False to keep it for internal quality review -- it must never
                reach a release.
            date_shift_anchor: Date each participant's earliest session is shifted to (within
                three days). With None, every ``date_shift=YES`` column is blanked.
            skip_audio_copy: Resolve source audio as usual (so sidecars, sessions.tsv and
                recording.tsv are exactly what a full build writes) but copy no audio files.
                For metadata-only builds; not for a release.
            date_shift_log: Optionally also write the date-shift report (anchor, counts, and the
                internal IDs left unshifted) as JSON. Must be outside ``outdir``. A summary is
                always logged.

        Returns:
            BIDSDataset instance pointing to the created BIDS directory
        """
        if date_shift_log is not None:
            BIDSDataset._check_date_shift_log(date_shift_log, outdir)
        outdir = Path(outdir).as_posix()
        BIDSDataset._initialize_data_directory(outdir)

        if drop_audio_check:
            redcap_dataset = copy(redcap_dataset)
            redcap_dataset.df = BIDSDataset._drop_audio_check_rows(redcap_dataset.df)

        redcap_dataset = BIDSDataset._apply_field_map_at_ingest(
            redcap_dataset, date_shift_anchor, date_shift_log, outdir
        )

        _LOGGER.info("Converting RedCap dataset to BIDS phenotype files.")
        # Subselect the RedCap dataframe and output components to individual files in the phenotype directory
        BIDSDataset._construct_phenotype_from_reproschema(
            df=redcap_dataset.df,
            output_dir=os.path.join(outdir, "phenotype"),
            dropped_at_ingest=redcap_dataset.metadata.get("dropped_at_ingest", ()),
        )

        if audiodir is None:
            # Return a new BIDSDataset instance pointing to the created directory
            return cls(outdir)

        _LOGGER.info("Processing audio files into BIDS format.")
        # We have two remaining tasks: (1) copy the audio files, and (2) create sidecar .json files.
        # First we prepare the metadata necessary for the .json files.
        participants_df = redcap_dataset.get_df_of_repeat_instrument(RepeatInstrument.PARTICIPANT.value)
        _LOGGER.info(f"Number of {RepeatInstrument.PARTICIPANT.name} entries: {len(participants_df)}")
        sessions_df = redcap_dataset.get_df_of_repeat_instrument(RepeatInstrument.SESSION.value)
        _LOGGER.info(f"Number of {RepeatInstrument.SESSION.name} entries: {len(sessions_df)}")
        acoustic_tasks_df = redcap_dataset.get_df_of_repeat_instrument(RepeatInstrument.ACOUSTIC_TASK.value)
        _LOGGER.info(f"Number of {RepeatInstrument.ACOUSTIC_TASK.name} entries: {len(acoustic_tasks_df)}")
        recordings_df = redcap_dataset.get_df_of_repeat_instrument(RepeatInstrument.RECORDING.value)
        _LOGGER.info(f"Number of {RepeatInstrument.RECORDING.name} entries: {len(recordings_df)}")

        sessions_by_participant = defaultdict(list)
        for session in sessions_df.to_dict("records"):
            sessions_by_participant[session["record_id"]].append(session)

        tasks_by_session = defaultdict(list)
        for task in acoustic_tasks_df.to_dict("records"):
            tasks_by_session[task["acoustic_task_session_id"]].append(task)

        recordings_by_task = defaultdict(list)
        for recording in recordings_df.to_dict("records"):
            recordings_by_task[recording["recording_acoustic_task_id"]].append(recording)

        # Per-participant stimulus for a few tasks (vocab words, random category,
        # stroop colors) is stored in linked questionnaires, not in any static
        # descriptions file. Build a lookup keyed by (instrument, acoustic_task_id)
        # so the metadata resolver can populate `stimulus_text` for those tasks.
        questionnaire_lookup: t.Dict[tuple, dict] = {}
        for instrument_key, repeat_instrument, join_column in (
            ("vocab", RepeatInstrument.NEURO_PRODUCTIVE_VOCABULARY, "vocabulary_recording_acoustic_task_id"),
            ("random", RepeatInstrument.NEURO_RANDOM_ITEM_GENERATION, "random_recording_acoustic_task_id"),
            ("stroop", RepeatInstrument.NEURO_WORDCOLOR_STROOP, "stroop_recording_acoustic_task_id"),
        ):
            try:
                instrument_df = redcap_dataset.get_df_of_repeat_instrument(repeat_instrument.value)
            except Exception as exc:  # instrument absent for this cohort/export
                _LOGGER.warning(f"Skipping {instrument_key} questionnaire join: {exc}")
                continue
            for row in instrument_df.to_dict("records"):
                task_id = row.get(join_column)
                if not _is_present(task_id):
                    continue
                key = (instrument_key, task_id)
                if key in questionnaire_lookup:
                    _LOGGER.warning(
                        f"Multiple {instrument_key} questionnaire rows for "
                        f"acoustic_task_id {task_id}; keeping the first, ignoring duplicate."
                    )
                    continue
                questionnaire_lookup[key] = row

        participants = []
        for participant in participants_df.to_dict("records"):
            participants.append(participant)
            participant["sessions"] = sessions_by_participant.get(participant["record_id"], [])

            for session in participant["sessions"]:
                session_id = session["session_id"]
                session["acoustic_tasks"] = tasks_by_session.get(session_id, [])
                
                for task in session["acoustic_tasks"]:
                    task["recordings"] = recordings_by_task.get(task["acoustic_task_id"], [])

        # Output participant data to FHIR format
        audio_files: t.List[Path] = []
        if audiodir is not None and Path(audiodir).exists():
            audio_files = list(Path(audiodir).rglob("*.wav"))
        
        audio_mappings_path = files("b2aiprep.prepare.resources").joinpath("audio_task_descriptions.json")
        with open(audio_mappings_path, 'r') as file_object:
            audio_descriptor_dict = json.load(file_object, object_pairs_hook=OrderedDict)
        # create an index of recording_id: audio_file for later use
        # ASSUMES that audio files are named with the recording_id in the filename
        # we use a defensive regex to grab uuid-like IDs from the stem just in case
        p_uuid = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')
        audio_files_by_recording: t.Optional[t.Dict[str, Path]] = None
        if audiodir is not None:
            audio_files_by_recording = {}
        for audio_file in audio_files:
            match = p_uuid.search(audio_file.stem)
            if not match:
                continue
            uuid = match.group(0)
            if uuid in audio_files_by_recording:
                _LOGGER.warning(
                    f"Multiple audio files found for recording UUID {uuid}: "
                    f"{audio_files_by_recording[uuid]} and {audio_file}. "
                    "Only the last one will be retained."
                )
            audio_files_by_recording[uuid] = audio_file

        participants_with_audio = set()
        all_recording_ids_with_sidecar: t.Set[str] = set()
        for participant in tqdm(participants, desc="Writing participant data to file"):
            had_audio, rec_ids_with_sidecar = cls._output_participant_data_to_metadata_file(
                participant,
                Path(outdir),
                audio_files_by_recording=audio_files_by_recording,
                max_audio_workers=max_audio_workers,
                sanitize_audio_format=sanitize_audio_format,
                audio_descriptor_dict=audio_descriptor_dict,
                questionnaire_lookup=questionnaire_lookup,
                skip_audio_copy=skip_audio_copy,
            )
            if had_audio:
                participants_with_audio.add(participant["record_id"])
                all_recording_ids_with_sidecar.update(rec_ids_with_sidecar)

        # Filter recording.tsv to only recordings that produced a sidecar, then
        # filter acoustic_task.tsv to only tasks that still have at least one
        # recording. Recordings whose source was missing or truncated were
        # skipped; their rows would otherwise reference nonexistent files.
        if audio_files_by_recording is not None and all_recording_ids_with_sidecar:
            phenotype_dir = os.path.join(outdir, "phenotype")
            for tsv_name, id_col in [("task/recording.tsv", "recording_id")]:
                fp = os.path.join(phenotype_dir, tsv_name)
                if not os.path.isfile(fp):
                    continue
                df_tsv = pd.read_csv(fp, sep="\t", dtype=str)
                if id_col not in df_tsv.columns:
                    continue
                before = len(df_tsv)
                df_tsv = df_tsv.loc[df_tsv[id_col].isin(all_recording_ids_with_sidecar)]
                after = len(df_tsv)
                if before != after:
                    df_tsv.to_csv(fp, sep="\t", index=False)
                    _LOGGER.info(
                        "phenotype/%s: %d -> %d rows after removing recordings/tasks "
                        "without a sidecar on disk.",
                        tsv_name, before, after,
                    )

            # acoustic_task.tsv rows whose recordings were all filtered out above
            # are orphans — no file on disk references their acoustic_task_id.
            recording_fp = os.path.join(phenotype_dir, "task/recording.tsv")
            acoustic_task_fp = os.path.join(phenotype_dir, "task/acoustic_task.tsv")
            if os.path.isfile(recording_fp) and os.path.isfile(acoustic_task_fp):
                df_rec = pd.read_csv(recording_fp, sep="\t", dtype=str)
                df_at = pd.read_csv(acoustic_task_fp, sep="\t", dtype=str)
                if (
                    "recording_acoustic_task_id" in df_rec.columns
                    and "acoustic_task_id" in df_at.columns
                ):
                    surviving_task_ids = set(df_rec["recording_acoustic_task_id"].dropna())
                    before_at = len(df_at)
                    df_at = df_at.loc[df_at["acoustic_task_id"].isin(surviving_task_ids)]
                    after_at = len(df_at)
                    if before_at != after_at:
                        df_at.to_csv(acoustic_task_fp, sep="\t", index=False)
                        _LOGGER.info(
                            "phenotype/task/acoustic_task.tsv: %d -> %d rows after "
                            "removing tasks with no surviving recordings.",
                            before_at, after_at,
                        )

        # QA report: participants with no distributed audio
        participants_without_audio = {p["record_id"] for p in participants} - participants_with_audio
        if participants_without_audio:
            _LOGGER.warning(
                "%d of %d participant(s) produced no audio files and were excluded from "
                "the BIDS tree (no sub-*/ directory, no rows in phenotype). Their records "
                "exist in the REDCap export but had no recordings with a locatable source "
                "file after filtering. QA: verify these are expected (enrollment-only, "
                "audio-check-only, or missing source audio). IDs: %s",
                len(participants_without_audio),
                len(participants),
                ", ".join(sorted(participants_without_audio)),
            )

        # Filter phenotype tables to only participants with audio
        if participants_without_audio:
            phenotype_dir = os.path.join(outdir, "phenotype")
            if os.path.isdir(phenotype_dir):
                _LOGGER.info(
                    "Filtering phenotype tables to %d participants with audio "
                    "(removing %d without).",
                    len(participants_with_audio),
                    len(participants_without_audio),
                )
                BIDSDataset._filter_phenotype_to_participants(
                    phenotype_dir, participants_with_audio
                )

        # Return a new BIDSDataset instance pointing to the created directory
        return cls(outdir)


    @staticmethod
    def _initialize_data_directory(bids_dir_path: str) -> None:
        """Initializes the data directory using the template.

        Args:
            bids_dir_path (str): The path to the BIDS directory where the data should be initialized.

        Returns:
            None
        """
        if not os.path.exists(bids_dir_path):
            os.makedirs(bids_dir_path)
            _LOGGER.info(f"Created directory: {bids_dir_path}")

        template_package = "b2aiprep.template"
        copy_package_resource(template_package, "CHANGELOG.md", bids_dir_path)
        copy_package_resource(template_package, "README.md", bids_dir_path)
        copy_package_resource(template_package, "dataset_description.json", bids_dir_path)
        copy_package_resource(template_package, "phenotype", bids_dir_path)

    def find_questionnaires(self, questionnaire_name: str) -> t.List[Path]:
        """
        Find all the questionnaires with a given suffix.

        Parameters
        ----------
        questionnaire_name : str
            The name of the questionnaire.

        Returns
        -------
        List[Path]
            A list of questionnaires which have the given questionnaire suffix.
        """
        questionnaires = []
        
        # Handle special cases for the new data structure
        if questionnaire_name == "recordingschema":
            # Find all audio metadata JSON files which contain recordingschema data
            for audio_json in self.data_path.rglob("*/audio/*.json"):
                questionnaires.append(audio_json)
        else:
            # Try the original approach first for backward compatibility
            for questionnaire in self.data_path.rglob(f"sub-*_{questionnaire_name}.json"):
                questionnaires.append(questionnaire)
        
        return questionnaires

    def find_subject_questionnaires(self, subject_id: str) -> t.List[Path]:
        """
        Find all the questionnaires for a given subject.

        Parameters
        ----------
        subject_id : str
            The subject identifier.

        Returns
        -------
        List[Path]
            A list of questionnaires for the specific subject.
        """
        subject_path = self.data_path / f"sub-{subject_id}"
        questionnaires = []
        for questionnaire in subject_path.glob("sub-*.json"):
            questionnaires.append(questionnaire)
        return questionnaires

    def find_session_questionnaires(self, subject_id: str, session_id: str) -> t.List[Path]:
        """
        Find all the questionnaires for a given subject and session.

        Parameters
        ----------
        subject_id : str
            The subject identifier.
        session_id : str
            The session identifier.

        Returns
        -------
        List[Path]
            A list of questionnaires for the specific subject and session.
        """
        session_path = self.data_path / f"sub-{subject_id}" / f"ses-{session_id}"
        questionnaires = []
        for questionnaire in session_path.glob("sub-*.json"):
            questionnaires.append(questionnaire)
        return questionnaires

    def find_subjects(self) -> t.List[Path]:
        """
        Find all the subjects in the dataset.

        Returns
        -------
        List[Path]
            A list of subject paths.
        """
        subjects = []
        for subject in self.data_path.glob("sub-*"):
            subjects.append(subject)
        return subjects

    def find_sessions(self, subject_id: str) -> t.List[Path]:
        """
        Find all the sessions for a given subject.

        Parameters
        ----------
        subject_id : str
            The subject identifier.

        Returns
        -------
        List[Path]
            A list of session paths.
        """
        subject_path = self.data_path / f"sub-{subject_id}"
        sessions = []
        for session in subject_path.glob("ses-*"):
            sessions.append(session)
        return sessions

    def find_tasks(self, subject_id: str, session_id: str) -> t.Dict[str, Path]:
        """
        Find all the tasks for a given subject and session.

        Parameters
        ----------
        subject_id : str
            The subject identifier.
        session_id : str
            The session identifier.

        Returns
        -------
        Dict[str, Path]
            A dictionary of tasks for the subject and session.
        """
        session_path = self.data_path / f"sub-{subject_id}" / f"ses-{session_id}"
        tasks = {}
        prefix_length = len(f"sub-{subject_id}_ses-{session_id}_task-")
        for task in session_path.glob(f"sub-{subject_id}_ses-{session_id}_task-*"):
            task_id = task.stem[prefix_length:-4]
            tasks[task_id] = task.stem
        return tasks

    def list_questionnaire_types(self, subject_only: bool = False) -> t.List[str]:
        """
        List all the questionnaire types in the dataset.

        Returns
        -------
        List[str]
            A list of questionnaire types.
        """
        questionnaire_types = set()
        for subject_path in self.data_path.glob("sub-*"):
            # subject-wide resources
            for questionnaire in subject_path.glob("sub-*.json"):
                questionnaire_types.add(questionnaire.stem.split("_")[-1])
            if subject_only:
                continue
            # session-wide resources
            for session_path in subject_path.glob("ses-*"):
                beh_path = session_path.joinpath("beh")
                if beh_path.exists():
                    for questionnaire in beh_path.glob("sub-*.json"):
                        questionnaire_types.add(questionnaire.stem.split("_")[-1])
        return sorted(list(questionnaire_types))

    def load_questionnaire(self, questionnaire_path: Path) -> QuestionnaireResponse:
        """
        Load a questionnaire from a given path.

        Parameters
        ----------
        questionnaire_path : Path
            The path to the questionnaire.

        Returns
        -------
        pd.DataFrame
            The questionnaire data.
        """
        return QuestionnaireResponse.parse_raw(questionnaire_path.read_text())

    def load_subject_questionnaires(self, subject_id: str) -> t.List[QuestionnaireResponse]:
        """
        Load all the questionnaires for a given subject.

        Parameters
        ----------
        subject_id : str
            The subject identifier.

        Returns
        -------
        List[QuestionnaireResponse]
            A list of questionnaires for the specific subject. Each element is a FHIR
            QuestionnaireResponse object, which inherits from Pydantic.
        """
        questionnaires = self.find_subject_questionnaires(subject_id)
        return [self.load_questionnaire(path) for path in questionnaires]

    def load_questionnaires(self, questionnaire_name: str) -> t.List[QuestionnaireResponse]:
        """
        Load all the questionnaires with a given name.

        Parameters
        ----------
        questionnaire_name : str
            The name of the questionnaire.

        Returns
        -------
        List[QuestionnaireResponse]
            A list of questionnaires for the specific subject. Each element is a FHIR
            QuestionnaireResponse object, which inherits from Pydantic.
        """
        questionnaires = self.find_questionnaires(questionnaire_name)
        return [self.load_questionnaire(path) for path in questionnaires]

    def questionnaire_to_dataframe(self, questionnaire: QuestionnaireResponse) -> pd.DataFrame:
        """
        Convert a questionnaire to a pandas DataFrame.

        Parameters
        ----------
        questionnaire : pd.DataFrame
            The questionnaire data.

        Returns
        -------
        pd.DataFrame
            The questionnaire data as a DataFrame. The dataframe is in a "long" format
            with a column for "linkId" and multiple value columns (e.g. "valueString")
        """
        questionnaire_dict = questionnaire.dict()
        items = questionnaire_dict["item"]
        has_multiple_answers = False
        for item in items:
            if ("answer" in item) and (len(item["answer"]) > 1):
                has_multiple_answers = True
                break
        if has_multiple_answers:
            raise NotImplementedError("Questionnaire has multiple answers per question.")

        items = []
        for item in questionnaire_dict["item"]:
            # Rename record_id to participant_id
            link_id = item["linkId"]
            if link_id == "record_id":
                link_id = "participant_id"
            if "answer" in item:
                items.append(
                    OrderedDict(
                        linkId=link_id,
                        **item["answer"][0],
                    )
                )
            else:
                items.append(
                    OrderedDict(
                        linkId=link_id,
                        valueString=None,
                    )
                )
        # unroll based on the possible value options
        return pd.DataFrame(items)

    def find_audio(self, subject_id: str, session_id: str) -> t.List[Path]:
        """
        Find all the audio recordings for a given subject and session.

        Parameters
        ----------
        subject_id : str
            The subject identifier.
        session_id : str
            The session identifier.

        Returns
        -------
        List[Path]
            A list of audio recordings.
        """
        session_path = self.data_path / f"sub-{subject_id}" / f"ses-{session_id}" / "audio"
        audio = []
        for audio_file in session_path.glob("*.wav"):
            audio.append(audio_file)
        return audio

    def find_audio_features(self, subject_id: str, session_id: str) -> t.List[Path]:
        """
        Find all the audio features for a given subject and session.

        Parameters
        ----------
        subject_id : str
            The subject identifier.
        session_id : str
            The session identifier.

        Returns
        -------
        List[Path]
            A list of audio features.
        """
        session_path = self.data_path / f"sub-{subject_id}" / f"ses-{session_id}" / "audio"
        features = []
        for feature_file in session_path.glob("*.pt"):
            features.append(feature_file)
        return features

    def find_audio_transcripts(self, subject_id: str, session_id: str) -> t.List[Path]:
        """
        Find all the audio transcripts for a given subject and session.

        Parameters
        ----------
        subject_id : str
            The subject identifier.
        session_id : str
            The session identifier.

        Returns
        -------
        List[Path]
            A list of audio transcripts.
        """
        session_path = (
            self.data_path / f"sub-{subject_id}" / f"ses-{session_id}" / "audio_transcripts"
        )
        transcripts = []
        for transcript_file in session_path.glob("*.json"):
            transcripts.append(transcript_file)
        return transcripts

    @staticmethod
    def _df_to_dict(df: pd.DataFrame, index_col: str) -> t.Dict[str, t.Any]:
        """Convert a DataFrame to a dictionary of dictionaries, with the given column as the index.

        Retains the index column within the dictionary.

        Args:
            df: DataFrame to convert.
            index_col: Column to use as the index.

        Returns:
            Dictionary of dictionaries.

        Raises:
            ValueError: If index column not found in DataFrame or non-unique values found.
        """
        if index_col not in df.columns:
            raise ValueError(f"Index column {index_col} not found in DataFrame.")
        if df[index_col].isnull().any():
            _LOGGER.warning(
                f"Found {df[index_col].isnull().sum()} null value(s) for {index_col}. Removing."
            )
            df = df.dropna(subset=[index_col])

        non_unique = df[df[index_col].duplicated(keep=False)]
        if df[index_col].nunique() < df.shape[0]:
            raise ValueError(f"Non-unique {index_col} values found. {non_unique}")
        

        # *copy* the given column into the index, preserving the original column
        # so that it is output in the later call to to_dict()
        df.index = df[index_col]

        return df.to_dict("index")

    @staticmethod
    def _dataframe_to_tsv(df: pd.DataFrame, tsv_path: str) -> None:
        """Construct a TSV file from a DataFrame.

        Args:
            df: DataFrame containing the data.
            tsv_path: Path to the output TSV file.
        """
        # Save the combined DataFrame to a TSV file
        df.to_csv(tsv_path, sep="\t", index=False)
        _LOGGER.info(f"TSV file with {df.shape[0]} rows created and saved to: {tsv_path}")

    @staticmethod
    def _load_reproschema(
        reproschema_file: Path,
        reproschema_folder: Path,
    ) -> t.Dict[str, t.Dict[str, t.Any]]:
        with reproschema_file.open("r", encoding="utf-8") as fp:
            schema_json = json.load(fp)

        protocol_order = schema_json.get("ui", {}).get("order")

        commit_sha = get_commit_sha(reproschema_folder)

        activities = {}
        for rel_path in protocol_order:
            # activities are relative to the reproschema schema file itself
            activity_path = reproschema_file.parent.joinpath(rel_path).resolve()
            if not activity_path.exists():
                _LOGGER.warning("Skipping missing activity %s", rel_path)
                continue

            activity_json = json.loads(activity_path.read_text())
            activity_id = activity_json.get("id")

            payload = build_activity_payload(
                activity_json=activity_json,
                activity_path=activity_path,
                reproschema_folder=reproschema_folder,
                commit_sha=commit_sha,
            )

            activities[activity_id] = payload

        return activities

    @staticmethod
    def _check_date_shift_log(date_shift_log: t.Union[str, Path], outdir: t.Union[str, Path]) -> Path:
        """The report names internal record and session IDs, so it must not land in the tree."""
        log_path = Path(date_shift_log).resolve()
        if log_path.is_relative_to(Path(outdir).resolve()):
            raise ValueError(f"date_shift_log must be outside the BIDS output: {log_path}")
        return log_path

    @staticmethod
    def _apply_field_map_at_ingest(
        redcap_dataset: RedCapDataset,
        date_shift_anchor: t.Optional[datetime.date],
        date_shift_log: t.Optional[t.Union[str, Path]],
        outdir: t.Union[str, Path],
    ) -> RedCapDataset:
        """Shift ``date_shift=YES`` dates, then remove every ``disposition=drop`` column.

        Runs before any output is written, so real dates and dropped columns never reach the
        BIDS tree. Dates are shifted first: the shift reads ``enrollment_institution``, the
        ``*_via`` columns and the participant's postal code / state, and those must still be
        present even if a future field map drops them.
        """
        field_map = BIDSDataset._load_reorganization_file(exclude_dropped=False)
        date_columns = field_map.loc[
            field_map["date_shift"].astype(str).str.upper() == "YES", "column_name_source"
        ].tolist()
        df, report = shift_dates(redcap_dataset.df, date_columns, date_shift_anchor)

        # A source column is removed only when no row keeps it (shared with instrument selection).
        dropped = _dropped_source_columns()
        present = [c for c in df.columns if c in dropped]
        df = df.drop(columns=present)
        _LOGGER.info("Removed %d disposition=drop column(s) at ingest.", len(present))

        # The anchor is logged with the run so the shifts can be reproduced; the log lives with
        # the job output, outside the BIDS tree.
        _LOGGER.info(
            "Date shift: anchor=%s, %d of %d participant(s) shifted, session time zones %s.",
            date_shift_anchor.isoformat() if date_shift_anchor else None,
            report["participants_shifted"],
            report["participants"],
            report["session_timezone_sources"],
        )
        if report["participants_without_offset"]:
            # IDs are listed for QA; this log lives with the job output, outside the BIDS tree.
            _LOGGER.info(
                "Participants without a date offset, by reason: %s",
                dict(Counter(report["participants_without_offset"].values())),
            )
            _LOGGER.info("Participants without a date offset: %s", report["participants_without_offset"])
        if report["sessions_without_timezone"]:
            _LOGGER.info("Sessions without a time zone: %s", report["sessions_without_timezone"])

        if date_shift_log is not None:
            log_path = BIDSDataset._check_date_shift_log(date_shift_log, outdir)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as fp:
                json.dump(
                    {
                        "anchor": date_shift_anchor.isoformat() if date_shift_anchor else None,
                        "source": redcap_dataset.metadata.get("source_file"),
                        "outdir": str(Path(outdir).resolve()),
                        "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        **report,
                    },
                    fp,
                    indent=2,
                )
            _LOGGER.info("Date-shift report written to %s", log_path)

        shifted = copy(redcap_dataset)
        shifted.df = df
        shifted.metadata = {**redcap_dataset.metadata, "dropped_at_ingest": present}
        return shifted

    @staticmethod
    def _load_reorganization_file(exclude_dropped: bool = True) -> pd.DataFrame:
        """Load the field map (bids_field_organization.csv).

        Args:
            exclude_dropped: Leave out ``disposition=drop`` rows. The CSV's ``delete`` column is
                kept as a historical record only and is not read.

        Returns:
            DataFrame containing the reorganization data.
        """
        reorganization_file = files("b2aiprep.prepare.resources").joinpath("bids_field_organization.csv")
        df = pd.read_csv(reorganization_file, sep=',', header=0)
        if exclude_dropped:
            df = df.loc[df['disposition'] != 'drop']
        return df

    @staticmethod
    def _find_redcap_checkbox_columns(
        df: pd.DataFrame,
    ) -> t.Dict[str, t.List[str]]:
        """Find RedCap checkbox columns in the DataFrame.

        Args:
            df: DataFrame containing the data.

        Returns:
            List of base column names for checkbox fields.
        """
        checkbox_columns = defaultdict(list)
        pattern = re.compile(r'^(?P<base>.+?)___(?P<suffix>.+)$')
        for col in df.columns:
            match = pattern.match(col)
            if match:
                base_col = match.group('base')
                checkbox_columns[base_col].append(col)
        return checkbox_columns

    @staticmethod
    def _fix_disjoint_demographic_rows(
        df: pd.DataFrame,
        id_col: str,
        demographic_col: str,
        schema_name: str = 'demographics'
    ) -> None:
        """Fix disjoint demographic rows by propagating known values.

        Args:
            df: DataFrame containing the data.
            id_col: Column name for participant ID.
            demographic_col: Column name for the demographic variable to fix.
        """
        # carry-forward age column values within each participant
        # use groupby-apply-transform to ensure we do not mix values across participants
        df[demographic_col] = df.groupby(id_col)[demographic_col].transform(
            lambda x: x.ffill().bfill()
        )

        # drop rows where the *only* non-null value is id_col + age
        # this avoids dropping rows where other data is present
        def is_only_id_and_demo(row: pd.Series) -> bool:
            non_null_cols = row.dropna().index.tolist()
            if len(non_null_cols) == 2 and id_col in non_null_cols and demographic_col in non_null_cols:
                return True
            return False
        mask_only_id_and_demo = df.apply(is_only_id_and_demo, axis=1)
        num_dropped = mask_only_id_and_demo.sum()
        if num_dropped > 0:
            _LOGGER.info((f"Fixing {schema_name} by dropping {num_dropped}/{df.shape[0]} rows with only "
                         f"{id_col} and {demographic_col} present."))
            df.drop(index=df[mask_only_id_and_demo].index, inplace=True)
        
    @staticmethod
    def _load_participant_allowlist(
        config_dir: Path, input_tree_participants: t.AbstractSet[str]
    ) -> t.Set[str]:
        """Load the participant allowlist for deidentification.

        Prefers ``participants_to_include.json`` (explicit allowlist).  When that
        file is absent, falls back to inverting ``participants_to_remove.json``
        against *input_tree_participants* so existing configs keep working.
        """
        include_path = config_dir / "participants_to_include.json"
        if include_path.exists():
            with open(include_path, "r") as f:
                allowlist = set(json.load(f))
            _LOGGER.info("Loaded participant allowlist with %d entries from %s.", len(allowlist), include_path)
        else:
            remove_path = config_dir / "participants_to_remove.json"
            if not remove_path.exists():
                raise FileNotFoundError(
                    f"Neither participants_to_include.json nor participants_to_remove.json "
                    f"found in {config_dir}."
                )
            with open(remove_path, "r") as f:
                to_remove = set(json.load(f))
            allowlist = set(input_tree_participants) - to_remove
            _LOGGER.info(
                "No participants_to_include.json; derived allowlist by inverting "
                "participants_to_remove.json (%d removed) against %d input participants → %d allowed.",
                len(to_remove), len(input_tree_participants), len(allowlist),
            )

        if not allowlist:
            raise ValueError(
                "Participant allowlist is empty — no participants would be included in the "
                "deidentified output. Check the config directory."
            )

        unmatched = allowlist - set(input_tree_participants)
        if unmatched:
            _LOGGER.warning(
                "%d allowlist entries do not match any participant in the input tree: %s",
                len(unmatched), sorted(unmatched),
            )
            allowlist = allowlist & set(input_tree_participants)
            if not allowlist:
                raise ValueError(
                    "Participant allowlist is empty after removing entries that do not "
                    "match any participant in the input tree."
                )

        return allowlist

    @staticmethod
    def _build_session_id_mapping(
        data_path: Path, participant_allowlist: t.AbstractSet[str]
    ) -> t.Dict[str, str]:
        """Build a mapping from original session UUIDs to shortened IDs.

        For each participant on the allowlist, reads their ``sessions.tsv`` and
        assigns ordinals from ``session_index`` (if present) or falls back to
        the legacy truncated-UUID behavior (8-char prefix, 16-char on collision).
        Returns a single flat dict covering all participants.
        """
        from b2aiprep.prepare.prepare import reduce_id_length

        mapping: t.Dict[str, str] = {}
        for pid in sorted(participant_allowlist):
            participant_dir = data_path / f"sub-{pid}"
            sessions_path = participant_dir / "sessions.tsv"
            if not sessions_path.exists():
                sessions_path = participant_dir / f"sub-{pid}_sessions.tsv"
            if not sessions_path.exists():
                _LOGGER.warning("No sessions.tsv found for participant %s; skipping session mapping.", pid)
                continue

            df = pd.read_csv(sessions_path, sep="\t", dtype=str)
            if "session_id" not in df.columns:
                _LOGGER.warning("sessions.tsv for participant %s has no session_id column; skipping.", pid)
                continue

            df = df.drop_duplicates(subset=["session_id"])
            if "session_index" in df.columns:
                df = df.sort_values("session_index", key=lambda s: pd.to_numeric(s, errors="coerce"))
                for ordinal, (_, row) in enumerate(df.iterrows(), start=1):
                    mapping[row["session_id"]] = f"{ordinal:02d}"
            else:
                _LOGGER.info("No session_index for participant %s; using truncated UUID fallback.", pid)
                for _, row in df.iterrows():
                    mapping[row["session_id"]] = reduce_id_length(row["session_id"])

        _LOGGER.info("Built session ID mapping: %d sessions across %d participants.",
                      len(mapping), len(participant_allowlist))
        return mapping

    _cached_field_map_df: t.ClassVar[t.Optional[pd.DataFrame]] = None

    # Identifier columns the pipeline adds to every table; not field-map rows of each table.
    _PIPELINE_ID_COLUMNS = frozenset({"participant_id", "record_id"})
    # Field-map table that describes the per-participant sessions.tsv columns.
    _SESSIONS_SCHEMA = "session"

    @staticmethod
    def _drop_columns_by_disposition(
        df: pd.DataFrame,
        field_map_df: t.Optional[pd.DataFrame] = None,
        level: DispositionLevel = DispositionLevel.RELEASE,
        schema_name: t.Optional[str] = None,
        keep_date_shifted: bool = False,
    ) -> t.Tuple[pd.DataFrame, t.List[str]]:
        """Drop columns above *level* in the disposition hierarchy.

        With *schema_name* (a phenotype table), only that table's field-map rows are consulted:
        output names are unique within a table but not across tables (e.g. ``self_reported_*`` is
        released in ``eligibility`` and internal in ``enrollment``).

        With *keep_date_shifted*, ``date_shift=YES`` columns are kept whatever the level: for
        builds that show the shifted dates alongside a release-like tree.

        The hierarchy is RELEASE < REVIEW < INTERNAL.  At the default
        ``RELEASE`` level, both ``internal`` and ``review`` columns are
        dropped.  At ``REVIEW``, only ``internal`` columns are dropped
        (``review`` columns are kept for per-value processing).  At
        ``INTERNAL``, nothing is dropped.

        Columns not in the field map are kept (pipeline-authored).
        """
        if field_map_df is None:
            if BIDSDataset._cached_field_map_df is None:
                BIDSDataset._cached_field_map_df = BIDSDataset._load_reorganization_file(exclude_dropped=False)
            field_map_df = BIDSDataset._cached_field_map_df

        if "disposition" not in field_map_df.columns:
            raise ValueError("Field map is missing the 'disposition' column.")
        if schema_name is not None and "schema_name" in field_map_df.columns:
            field_map_df = field_map_df.loc[field_map_df["schema_name"] == schema_name]
            if field_map_df.empty:
                # Without the table's rules nothing would be dropped: refuse instead.
                raise ValueError(
                    f"Table {schema_name!r} has no rows in the field map; cannot apply dispositions. "
                    "Was the tree built with a different bids_field_organization.csv?"
                )

        if level == DispositionLevel.INTERNAL:
            return df, []

        hierarchy = {"release": 0, "review": 1, "internal": 2}
        threshold = hierarchy[level.value]
        drop_dispositions = [d for d, rank in hierarchy.items() if rank > threshold]
        drop_rows = field_map_df["disposition"].isin(drop_dispositions)
        if keep_date_shifted and "date_shift" in field_map_df.columns:
            drop_rows &= field_map_df["date_shift"].astype(str).str.upper() != "YES"
        to_drop_names = set(field_map_df.loc[drop_rows, "column_name"].dropna())

        present = [c for c in df.columns if c in to_drop_names]
        if present:
            for col in present:
                _LOGGER.info("Dropping column '%s' (disposition-based).", col)
            df = df.drop(columns=present)

        all_field_map_names = set(field_map_df["column_name"].dropna())
        unknown = [
            c for c in df.columns
            if c not in all_field_map_names and c not in BIDSDataset._PIPELINE_ID_COLUMNS
        ]
        if unknown:
            _LOGGER.warning(
                "Columns not in field map (kept as pipeline-authored): %s",
                ", ".join(sorted(unknown)),
            )

        return df, present

    @staticmethod
    def _load_column_value_reviews(
        config_dir: Path,
        field_map_df: t.Optional[pd.DataFrame] = None,
    ) -> t.Dict[t.Tuple[str, str], str]:
        """Load the column value review manifest from the config directory.

        Entries may use either ``column_name`` (the output/BIDS name) or
        ``source_column_name`` (the original REDCap name).  Source names are
        normalized to output names via the field map so that lookups in
        ``_apply_column_value_reviews`` always use output names.
        """
        manifest_path = config_dir / "column_value_reviews.json"
        if not manifest_path.exists():
            return {}

        if field_map_df is None:
            if BIDSDataset._cached_field_map_df is None:
                BIDSDataset._cached_field_map_df = BIDSDataset._load_reorganization_file(exclude_dropped=False)
            field_map_df = BIDSDataset._cached_field_map_df

        source_to_output: t.Dict[str, str] = {}
        if "column_name_source" in field_map_df.columns and "column_name" in field_map_df.columns:
            for _, row in field_map_df.iterrows():
                src = row.get("column_name_source")
                out = row.get("column_name")
                if pd.notna(src) and pd.notna(out) and str(src) != str(out):
                    source_to_output[str(src)] = str(out)

        with open(manifest_path, "r") as f:
            data = json.load(f)
        verdicts_list = data.get("verdicts", [])
        lookup: t.Dict[t.Tuple[str, str], str] = {}
        normalized_count = 0
        for entry in verdicts_list:
            col = entry.get("column_name") or entry.get("source_column_name", "")
            if col in source_to_output:
                col = source_to_output[col]
                normalized_count += 1
            key = (entry["participant_id"], col)
            if key in lookup:
                _LOGGER.warning(
                    "Duplicate column value review for %s/%s; last entry wins.",
                    key[0], key[1],
                )
            lookup[key] = entry["verdict"].strip().lower()
        _LOGGER.info("Loaded %d column value review verdicts from %s.", len(lookup), manifest_path)
        if normalized_count:
            _LOGGER.info("Normalized %d verdicts from source to output column names.", normalized_count)
        return lookup

    @staticmethod
    def _get_review_column_names(
        field_map_df: t.Optional[pd.DataFrame] = None,
        schema_name: t.Optional[str] = None,
    ) -> t.Set[str]:
        """Return the column names with disposition=review, within *schema_name* when given."""
        if field_map_df is None:
            if BIDSDataset._cached_field_map_df is None:
                BIDSDataset._cached_field_map_df = BIDSDataset._load_reorganization_file(exclude_dropped=False)
            field_map_df = BIDSDataset._cached_field_map_df
        if "disposition" not in field_map_df.columns:
            return set()
        if schema_name is not None and "schema_name" in field_map_df.columns:
            field_map_df = field_map_df.loc[field_map_df["schema_name"] == schema_name]
        return set(
            field_map_df.loc[
                field_map_df["disposition"] == "review", "column_name"
            ].dropna()
        )

    @staticmethod
    def _apply_column_value_reviews(
        df: pd.DataFrame,
        review_columns: t.AbstractSet[str],
        verdicts: t.Dict[t.Tuple[str, str], str],
    ) -> t.Tuple[pd.DataFrame, t.List[str]]:
        """Apply per-cell verdicts to review-disposition columns.

        Columns with zero verdicts are dropped entirely (backward compat).
        Cells with no verdict default to null (fail-safe).
        """
        id_col = "participant_id" if "participant_id" in df.columns else "record_id"
        if id_col not in df.columns:
            return df, []

        fully_dropped = []
        for col in sorted(review_columns & set(df.columns)):
            col_verdicts = {
                pid: v for (pid, cname), v in verdicts.items() if cname == col
            }
            if not col_verdicts:
                df = df.drop(columns=[col])
                fully_dropped.append(col)
                continue
            for idx, row in df.iterrows():
                pid = row[id_col]
                verdict = col_verdicts.get(pid)
                if verdict == "safe":
                    pass
                elif verdict == "redact":
                    df.at[idx, col] = "[REDACTED]"
                else:
                    df.at[idx, col] = pd.NA
        return df, fully_dropped

    @staticmethod
    def _filter_phenotype_to_participants(
        phenotype_dir: str, keep_ids: t.AbstractSet[str]
    ) -> None:
        """Remove rows for participants without audio from every phenotype TSV.

        Called after the participant loop when some participants were skipped entirely
        because they had no locatable source audio. Without this, the phenotype tables
        would list participants whose sub-*/ directory does not exist, and recording.tsv
        would reference files that were never written.

        Logs a per-file summary so the output is auditable.
        """
        for dp, _, fs in os.walk(phenotype_dir):
            for fn in sorted(fs):
                if not fn.endswith(".tsv"):
                    continue
                fp = os.path.join(dp, fn)
                df = pd.read_csv(fp, sep="\t", dtype=str)
                id_col = None
                for candidate in ("participant_id", "record_id"):
                    if candidate in df.columns:
                        id_col = candidate
                        break
                if id_col is None:
                    continue
                before = len(df)
                df = df.loc[df[id_col].isin(keep_ids)]
                after = len(df)
                if before != after:
                    df.to_csv(fp, sep="\t", index=False)
                    rel = os.path.relpath(fp, phenotype_dir)
                    _LOGGER.info(
                        "phenotype/%s: %d -> %d rows after removing participants without audio.",
                        rel, before, after,
                    )
                    # Update the companion JSON if it exists (it carries data-element metadata,
                    # not row counts, so it does not need rewriting -- but we log for completeness).

    @staticmethod
    def _drop_audio_check_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Remove every row describing the session's microphone check.

        Applied once, to the RedCap dataframe, before anything reads it -- so a single filter
        covers every artifact at once: the `recording`/`acoustic_task` phenotype tables, the
        per-recording and per-acoustic-task sidecars, and the audio copy list. Filtering later
        would mean repeating the rule per artifact and finding a new place to repeat it every
        time one is added, which is how the three existing copies in deidentify came about.

        The audio check is collection apparatus, not research data: it is absent from the task
        registry (so every one of them logged a "no task match" warning -- 1,668 in the v4
        pediatric build alone), it is stripped again at deidentify, and it has never shipped in
        a release. Measured on the v4 exports: 7,970 of 70,520 adult recordings and 1,383 of
        99,724 pediatric.
        """
        if "redcap_repeat_instrument" not in df.columns:
            return df
        drop = pd.Series(False, index=df.index)
        for instrument, column in (("Recording", "recording_name"),
                                   ("Acoustic Task", "acoustic_task_name")):
            if column not in df.columns:
                continue
            is_row = df["redcap_repeat_instrument"].eq(instrument)
            drop |= is_row & df[column].apply(is_audio_check)
        if not drop.any():
            return df
        counts = df.loc[drop, "redcap_repeat_instrument"].value_counts().to_dict()
        _LOGGER.info(
            "Dropping %d audio-check row(s) at ingest (%s); they are excluded from the "
            "phenotype tables, sidecars and audio copies.",
            int(drop.sum()),
            ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())),
        )
        return df.loc[~drop]

    _FORM_COMPLETE_VALUES = frozenset({"Complete", "2"})

    @staticmethod
    def _drop_rows_without_substantive_data(
        df: pd.DataFrame,
        id_col: str,
        csv_only_columns: t.AbstractSet[str],
        calculated_columns: t.AbstractSet[str] = frozenset(),
        schema_name: str = "",
        schema_group: str = "",
    ) -> pd.DataFrame:
        """Drop rows that contain no participant-entered data.

        Two classes of column are excluded from the emptiness test:

        * ``csv_only_columns``: columns with no ReproSchema definition --
          ``<form>_complete``, ``<form>_timestamp``, and other RedCap-generated
          infrastructure.
        * ``calculated_columns``: ``is_redcap_calculation=YES`` in the field map.
          RedCap evaluates these for every record regardless of form completion,
          so a participant who never had an ALS assessment can still have
          ``gsd_calculation = 0``.

        Internal columns count as data here, so the pre-deidentification tree keeps rows
        holding only internal data; ``_drop_rows_emptied_by_deidentify`` re-runs this test
        once deidentify has removed them.

        For **diagnosis** forms (``schema_group == "diagnosis"``), any row where
        ``<form>_complete`` is not ``Complete`` is dropped regardless of data
        content.  An incomplete diagnosis form has not been clinician-verified and
        must not be disseminated.  Different scenarios are logged at appropriate
        levels so QA can triage them.

        When every column is bookkeeping there is nothing to test against, so
        the table is emptied — a table with no participant-entered data has no
        research value.
        """
        non_substantive = set(csv_only_columns) | set(calculated_columns)
        substantive = [c for c in df.columns if c != id_col and c not in non_substantive]
        if not substantive:
            _LOGGER.warning(
                "%s: no substantive columns — only bookkeeping. Returning empty table.",
                schema_name or "unknown",
            )
            return df.iloc[0:0]

        complete_col = None
        for col in csv_only_columns:
            if col.endswith("_complete") and col in df.columns:
                complete_col = col
                break

        all_substantive_null = df[substantive].isna().all(axis=1)
        calc_cols_present = [c for c in calculated_columns if c in df.columns]
        has_calc_data = (
            df[calc_cols_present].notna().any(axis=1)
            if calc_cols_present
            else pd.Series(False, index=df.index)
        )

        if complete_col is not None:
            form_complete = df[complete_col].isin(BIDSDataset._FORM_COMPLETE_VALUES)
        else:
            form_complete = pd.Series(True, index=df.index)

        is_diagnosis = schema_group == "diagnosis"

        if is_diagnosis and complete_col is not None:
            drop_mask = ~form_complete
            dropped = df[drop_mask]
            if not dropped.empty:
                inc_no_data = dropped[
                    all_substantive_null[drop_mask] & ~has_calc_data[drop_mask]
                ]
                inc_calc_only = dropped[
                    all_substantive_null[drop_mask] & has_calc_data[drop_mask]
                ]
                inc_with_data = dropped[~all_substantive_null[drop_mask]]

                if len(inc_no_data):
                    _LOGGER.debug(
                        "%s: dropping %d incomplete rows with no data",
                        schema_name, len(inc_no_data),
                    )
                if len(inc_calc_only):
                    _LOGGER.info(
                        "%s: dropping %d incomplete rows whose only data is "
                        "auto-calculated fields",
                        schema_name, len(inc_calc_only),
                    )
                if len(inc_with_data):
                    _LOGGER.warning(
                        "%s: dropping %d incomplete rows that contain "
                        "participant-entered data (form not clinician-verified)",
                        schema_name, len(inc_with_data),
                    )
        else:
            drop_mask = all_substantive_null
            dropped = df[drop_mask]
            if not dropped.empty and complete_col is not None:
                inc_dropped = dropped[~form_complete[drop_mask]]
                comp_dropped = dropped[form_complete[drop_mask]]
                if len(inc_dropped):
                    _LOGGER.debug(
                        "%s: dropping %d rows with no substantive data "
                        "(form incomplete)",
                        schema_name, len(inc_dropped),
                    )
                if len(comp_dropped):
                    _LOGGER.warning(
                        "%s: dropping %d rows with no substantive data "
                        "despite form marked Complete (data quality anomaly)",
                        schema_name, len(comp_dropped),
                    )

        return df.loc[~drop_mask]

    @staticmethod
    def _drop_rows_emptied_by_deidentify(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
        """Re-run the ingest emptiness test on a deidentified phenotype table.

        Ingest keeps rows whose only data is internal (they are useful before deidentification).
        Once deidentify has removed internal and unreviewed columns, such rows hold only the
        participant id and bookkeeping, and are dropped here with the same rule as at ingest.
        """
        if "participant_id" not in df.columns or df.empty:
            return df
        if BIDSDataset._cached_field_map_df is None:
            BIDSDataset._cached_field_map_df = BIDSDataset._load_reorganization_file(exclude_dropped=False)
        rows = BIDSDataset._cached_field_map_df
        rows = rows.loc[rows["schema_name"] == schema_name]
        # Shifted dates are timing metadata, not data: kept for discussion in some builds, but a
        # row holding only a date has nothing to publish.
        bookkeeping = set(
            rows.loc[
                rows["source"].isin(["redcap_generated", "pipeline"])
                | (rows.get("date_shift", pd.Series("", index=rows.index)).astype(str).str.upper() == "YES"),
                "column_name",
            ].dropna()
        )
        calculated = set(
            rows.loc[rows["is_redcap_calculation"].astype(str).str.upper() == "YES", "column_name"].dropna()
        )
        group = rows["group"].dropna().iloc[0] if rows["group"].notna().any() else ""
        before = len(df)
        df = BIDSDataset._drop_rows_without_substantive_data(
            df, "participant_id", bookkeeping, calculated, schema_name=schema_name, schema_group=group,
        )
        if len(df) < before:
            _LOGGER.info(
                "phenotype/%s: dropped %d row(s) left with no publishable data after deidentify.",
                schema_name, before - len(df),
            )
        return df

    @staticmethod
    def _synthetic_data_element(
        column: str,
        updated_data: t.Mapping[str, t.Any],
        column_choice: t.Optional[str] = None,
        clean_phenotype_data: bool = True,
    ) -> t.Dict[str, t.Any]:
        """Build a minimal data dictionary entry for a column with no ReproSchema definition.

        ReproSchema is generated from the RedCap data dictionary, so it only describes columns
        that someone authored as a field. It legitimately has no entry for the columns RedCap
        synthesizes per instrument (`<form>_complete`, `<form>_timestamp`), for RedCap's own
        structural columns, or for values b2aiprep computes itself. For those the reorganization
        CSV is the only description that exists, so it becomes the description of record.

        The result deliberately carries no `termURL`, no `choices` and no `question`: there is no
        authored definition to cite, and minting an ontology reference here would put an unsourced
        term into a published data dictionary. Consumers can therefore identify CSV-described
        columns by the absence of `termURL`. This matches the shape already used for the
        synthetic `participant_id` element.

        Args:
            column: the source (RedCap) column name.
            updated_data: the row from bids_field_organization.csv describing this column.
            column_choice: the option code when `column` is a `___`-suffixed checkbox option.
            clean_phenotype_data: when True, checkbox options are emitted as 0/1 integers.
        """
        _raw_desc = updated_data.get("description")
        description = str(_raw_desc).strip() if pd.notna(_raw_desc) else ""
        if not description:
            # Never leave a published dictionary entry without a description; say plainly that
            # the field map did not supply one rather than emitting an empty string.
            description = (
                f"No description available for {column}; this column has no ReproSchema "
                "definition and bids_field_organization.csv does not describe it."
            )
        value_type = (
            ["xsd:integer"] if (column_choice is not None and clean_phenotype_data) else ["xsd:string"]
        )
        return {"description": description, "valueType": value_type}

    def _construct_phenotype_from_reproschema(
        df: pd.DataFrame,
        output_dir: str,
        clean_phenotype_data: bool = True,
        dropped_at_ingest: t.Iterable[str] = (),
    ) -> None:
        """Construct TSV/JSON files from a source ReproSchema folder.

        Args:
            df: DataFrame containing the data.
            output_dir: Directory where the TSV files will be saved.
            clean_phenotype_data: Whether to clean the phenotype data (default: True).
            dropped_at_ingest: ``disposition=drop`` columns already removed from *df*; counted
                in the column report as deleted intentionally.
        """

        # We will ignore data dictionary columns when there are corresponding columns
        # in the dataframe with the suffix "___[text]". These are multi-checkbox columns.
        # Only the option responses are kept, the main column does not exist in the data,
        # only in the data dictionary.
        checkbox_columns = BIDSDataset._find_redcap_checkbox_columns(df)

        # Rename repeat instruments to their standard names
        instrument_name_to_schema_name = {
            repeat_instrument.value.schema_name_pretty: repeat_instrument.value.schema_name
            for repeat_instrument in RepeatInstrument.__iter__()
        }
        df['schema_name_source'] = df['redcap_repeat_instrument'].map(instrument_name_to_schema_name)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Load reproschema file

        source_path = files("b2aiprep.redcap2rs")
        if source_path.joinpath('b2ai-redcap2rs_schema').is_file():
            resolved_schema_file = source_path.joinpath('b2ai-redcap2rs_schema')
            reproschema_folder = Path(resolved_schema_file).parent.resolve()
        elif (source_path / 'b2ai-redcap2rs' / 'b2ai-redcap2rs_schema').is_file():
            resolved_schema_file = source_path / 'b2ai-redcap2rs' / 'b2ai-redcap2rs_schema'
            reproschema_folder = Path(resolved_schema_file).parent.parent.resolve()
        else:
            raise FileNotFoundError(
                f"Could not find 'b2ai-redcap2rs_schema' in source directory: {source_path}"
            )

        schemas = BIDSDataset._load_reproschema(resolved_schema_file, reproschema_folder)

        # create an index for data_element -> schema
        element_to_schema = {}
        for schema_name, data in schemas.items():
            for element_name in data['data_elements'].keys():
                element_to_schema[element_name] = schema_name

                # add in the checkbox columns which are not natively listed as individual
                # questions in the reproschema activities
                if element_name in checkbox_columns:
                    for checkbox_column in checkbox_columns[element_name]:
                        element_to_schema[checkbox_column] = schema_name

        # with all the reproschema activities defined, we now parse our manual reorganization
        # this is a CSV with:
        #   schema_name_source, column_name_source
        # ... used to identify the source reproschema element, and:
        #   schema_name, column_name, group, description
        # ... used to arrange & describe the output in the phenotype/ folder.
        # the overall goal is to remap from schema_name_source -> schema_name
        #   (the schema_name becomes the filename)
        # and to map from column_name_source -> column_name
        #   (the column_name becomes the column in the TSV file)
        df_reorg = BIDSDataset._load_reorganization_file(exclude_dropped=False)

        # Track inclusion/exclusion for a final report.
        # Note: columns are tracked using their *source* names (i.e., RedCap/df column names).
        df_deleted = df_reorg.loc[df_reorg['disposition'] == 'drop']
        df_reorg_active = df_reorg.loc[df_reorg['disposition'] != 'drop']

        _norm = lambda c: str(c)
        df_cols = {_norm(c) for c in df.columns}

        reorg_all_cols = {_norm(c) for c in df_reorg['column_name_source'].dropna().tolist()}
        cols_for_deletion = {_norm(c) for c in df_deleted['column_name_source'].dropna().tolist()}
        cols_for_adding = {_norm(c) for c in df_reorg_active['column_name_source'].dropna().tolist()}

        # the rest we will track as we go
        included_cols: t.Set[str] = set()
        missing_in_df_cols: t.Set[str] = set()
        pipeline_cols: t.Set[str] = set()
        redcap_group_cols: t.Set[str] = set()
        # Columns described by bids_field_organization.csv alone, with no ReproSchema element.
        synthesized_cols: t.Set[str] = set()

        df_reorg = df_reorg_active

        element_used = {} # keep track of whether we have used an element, for logging later.
        payload = {
            "description": "",
            "data_elements": {},
            # this variable will track the name of the original columns in RedCap
            "columns_for_indexing": [],
            # this variable will track the name of the *new* columns in the output df
            "columns_for_output": [],
            # group is popped before saving
            "group": "",
            # source schema name is used to filter rows
            "schema_name_source": [],
            # output columns with no ReproSchema definition (RedCap form status/timestamps).
            # Tracked so they can be excluded from the "is this row empty?" test below.
            "columns_csv_only": [],
            # columns flagged is_redcap_calculation=YES in the field map.
            # Tracked separately from csv_only because they DO have ReproSchema definitions
            # but should not count as substantive participant-entered data.
            "columns_calculated": [],
        }
        updated_schemas = defaultdict(lambda: deepcopy(payload))
        for updated_schema_name, group in df_reorg.groupby('schema_name'):
            column_mapping = group.set_index('column_name_source').to_dict(orient='index')
            for column, updated_data in column_mapping.items():
                col_norm = _norm(column)
                if col_norm not in df_cols:
                    if col_norm in checkbox_columns:
                        # skip the main checkbox column if we have the ___ option columns
                        _LOGGER.debug(
                            f'Skipping RedCap checkbox main column "{column}" as option columns found.'
                        )
                        redcap_group_cols.add(col_norm)
                        continue
                    if str(updated_data.get("source", "")).lower() == "pipeline":
                        _LOGGER.debug(f'Pipeline-computed column "{column}" not in RedCap source.')
                        pipeline_cols.add(col_norm)
                        continue
                    _LOGGER.warning(f'Requested output for "{column}", but this column was not found in the source df.')
                    missing_in_df_cols.add(col_norm)
                    continue
                included_cols.add(col_norm)
                
                if '___' in column:
                    column_base, column_choice = column.rsplit('___', maxsplit=1)
                else:
                    column_base = column
                    column_choice = None

                # the source schema is defined based on the element itself;
                # we do not need the schema_name_source column, but it is kept for ease of reading the CSV.
                schema_to_use = element_to_schema.get(column)
                source_elements = schemas[schema_to_use]["data_elements"] if schema_to_use else {}
                lookup_key = column_base if column_base in checkbox_columns else column
                csv_only_column = False
                if lookup_key not in source_elements:
                    # No ReproSchema definition exists for this column, so there is nothing to look
                    # up. RedCap emits columns that were never authored as fields -- <form>_complete,
                    # <form>_timestamp, its own structural columns -- and ReproSchema is generated
                    # from the data dictionary, which does not describe them. Derived columns that
                    # b2aiprep computes itself land here too. Synthesize a minimal element from the
                    # reorganization CSV instead of raising KeyError, and record the name so the run
                    # reports exactly which columns are described by the CSV alone.
                    data_element = BIDSDataset._synthetic_data_element(
                        column, updated_data, column_choice, clean_phenotype_data
                    )
                    synthesized_cols.add(col_norm)
                    csv_only_column = True
                elif column_base in checkbox_columns:
                    data_element = copy(source_elements[column_base])
                    # reduce the choices to just the choice for this checkbox
                    data_element['choices'] = [
                        choice for choice in data_element.get('choices', [])
                        if str(choice.get('value')) == str(column_choice)
                    ]
                    if clean_phenotype_data:
                        # for checkbox columns, we convert to integer 0/1
                        data_element['valueType'] = ['xsd:integer']
                else:
                    # populate the detailed metadata for this column
                    data_element = copy(source_elements[column])
                if "description" in updated_data and pd.notna(updated_data["description"]) and str(updated_data["description"]).strip():
                    description = updated_data["description"]
                elif "description" in data_element and pd.notna(data_element["description"]) and str(data_element["description"]).strip():
                    description = data_element["description"]
                else:
                    description = data_element.get("question", {}).get("en", "")
                data_element["description"] = description
                new_element_name = updated_data["column_name"]

                updated_schema_name = updated_data["schema_name"]
                updated_schemas[updated_schema_name]['columns_for_indexing'].append(column)
                updated_schemas[updated_schema_name]['columns_for_output'].append(new_element_name)
                if csv_only_column:
                    updated_schemas[updated_schema_name]['columns_csv_only'].append(new_element_name)
                if str(updated_data.get("is_redcap_calculation", "")).upper() == "YES":
                    updated_schemas[updated_schema_name]['columns_calculated'].append(new_element_name)
                updated_schemas[updated_schema_name]['schema_name_source'].append(updated_data["schema_name_source"])

                # update the payload so we have a reproschema json for this df
                updated_schemas[updated_schema_name]["data_elements"][new_element_name] = data_element
                updated_schemas[updated_schema_name]["group"] = updated_data["group"]
                element_used[column] = True

        # with our full updated_schemas dict prepared, we can iterate through the *new* schema names
        # and output a single folder for each
        for schema_name, payload in updated_schemas.items():
            columns_for_indexing = payload.pop("columns_for_indexing")
            columns_for_output = payload.pop("columns_for_output")
            schema_name_sources = set(payload.pop("schema_name_source"))
            csv_only_columns = set(payload.pop("columns_csv_only"))
            calculated_columns = set(payload.pop("columns_calculated"))
            group = payload.pop("group")
            if "record_id" not in columns_for_indexing:
                columns_for_indexing = ["record_id"] + columns_for_indexing
                columns_for_output = ["participant_id"] + columns_for_output

                # record_id is implicitly included in outputs when missing from the reorg mapping.
                if "record_id" in df_cols:
                    included_cols.add("record_id")

                # add record_id to the data elements if missing, add it at the beginning
                payload["data_elements"] = {
                    "participant_id": {
                        "description": "A unique identifier for the participant.",
                        "valueType": [
                            "xsd:string"
                        ]
                    },
                    **payload["data_elements"],
                }
            # we now extract the sub-dataframe from our source redcap data and output it to tsv
            # filter df to only rows with the relevant repeat_instruments
            selected_df = df.loc[:, columns_for_indexing]
            # next up, we subselect the rows used for this set of columns ("repeat instrument").
            # but first, verify that we have all of the source schema names in our df
            # this is defensive in case we are missing an entire repeat_instrument from the source df.
            # in that case we will use all rows and drop those with all NaN later.
            if schema_name_sources:
                have_all_schema_names = all(
                    source_name in df['schema_name_source'].unique()
                    for source_name in schema_name_sources
                )
                if have_all_schema_names:
                    selected_df = selected_df.loc[
                        df['schema_name_source'].isin(schema_name_sources)
                    ]

            # Rename columns based on the reorganization mapping.
            if len(columns_for_indexing) != len(columns_for_output):
                raise ValueError(
                    f"Schema {schema_name}: columns_for_indexing ({len(columns_for_indexing)}) and "
                    f"columns_for_output ({len(columns_for_output)}) length mismatch."
                )

            rename_map = dict(zip(columns_for_indexing, columns_for_output))
            if rename_map.get("record_id") not in (None, "participant_id"):
                _LOGGER.warning(
                    f"Schema {schema_name}: overriding record_id rename target "
                    f"{rename_map.get('record_id')!r} -> 'participant_id'."
                )
                rename_map["record_id"] = "participant_id"

            # Protect against accidentally collapsing columns.
            output_cols = list(rename_map.values())
            dupes = sorted({c for c in output_cols if output_cols.count(c) > 1})
            if dupes:
                raise ValueError(
                    f"Schema {schema_name}: duplicate output column names after mapping: {dupes}. "
                    "Check bids_field_organization.csv for collisions."
                )

            selected_df = selected_df.rename(columns=rename_map)
            # Keep output column order stable and aligned to metadata.
            selected_df = selected_df[output_cols]
            id_col = "participant_id"

            updated_schema = {schema_name: payload}
            if clean_phenotype_data:
                selected_df, updated_schema = BIDSDataset._clean_phenotype_data(selected_df, updated_schema)

            # Remove rows where the only non-null value is record_id/participant_id.
            # Columns with no ReproSchema definition are excluded from this test: RedCap emits a
            # <form>_complete for every record whether or not the form was ever filled in, so
            # counting it as content would keep a row for every participant in every table -- e.g.
            # every diagnosis table would list the whole cohort instead of the diagnosed subset.
            selected_df = BIDSDataset._drop_rows_without_substantive_data(
                selected_df, id_col, csv_only_columns, calculated_columns,
                schema_name=schema_name, schema_group=group,
            )
            if selected_df.empty:
                _LOGGER.warning(f"No data remaining after dropping empty rows for {schema_name}")
                continue

            if 'age' in selected_df.columns:
                BIDSDataset._fix_disjoint_demographic_rows(selected_df, id_col, 'age', schema_name=schema_name)
                selected_df = BIDSDataset._drop_rows_without_substantive_data(
                    selected_df, id_col, csv_only_columns, calculated_columns,
                    schema_name=schema_name, schema_group=group,
                )
                if selected_df.empty:
                    _LOGGER.warning(f"No data remaining after dropping empty rows for {schema_name}")
                    continue

            # Output to a TSV/JSON file.
            filename = f'{schema_name}.json'
            if group != "":
                output_dir_grouped = os.path.join(output_dir, group)
                os.makedirs(output_dir_grouped, exist_ok=True)
            else:
                output_dir_grouped = output_dir
            BIDSDataset._dataframe_to_tsv(
                selected_df,
                os.path.join(output_dir_grouped, filename.replace(".json", ".tsv"))
            )
            with open(os.path.join(output_dir_grouped, filename), "w") as f:
                json.dump(updated_schema, f, indent=2)

        # Final report: included/excluded columns (by source column name).
        # our later loops only go through reorg columns, so calculate what we're missing now
        excluded_in_df_not_in_reorg = df_cols - reorg_all_cols

        _LOGGER.info(
            "RedCap Dataframe column report: total=%d, included=%d, deleted_intentionally=%d, only_in_df=%d",
            len(df_cols),
            len(included_cols.intersection(df_cols)),
            len(cols_for_deletion.intersection(df_cols | set(dropped_at_ingest))),
            len(excluded_in_df_not_in_reorg),
        )
        _LOGGER.info(
            "RedCap ReproSchema expected column report: total=%d, included=%d, "
            "redcap_group_general_q=%d, deleted_intentionally=%d, pipeline=%d, missing_in_df=%d",
            df_reorg_active.shape[0] + df_deleted.shape[0],
            len(included_cols),
            len(redcap_group_cols),
            len(cols_for_deletion),
            len(pipeline_cols),
            len(missing_in_df_cols),
        )
        if synthesized_cols:
            # Surfaced at INFO, not debug: these columns reach the output with a data dictionary
            # entry carrying only a description and a valueType. They have no termURL, no choices
            # and no question text, because no authored definition exists to cite -- inventing an
            # ontology reference for them would put an unsourced term into a published dictionary.
            _LOGGER.info(
                "%d column(s) had no ReproSchema element and were described from "
                "bids_field_organization.csv alone (no termURL/choices in the data dictionary): %s",
                len(synthesized_cols),
                ", ".join(sorted(synthesized_cols)[:10])
                + (", ..." if len(synthesized_cols) > 10 else ""),
            )
            _LOGGER.debug(f"Synthesized (CSV-only) columns: {sorted(synthesized_cols)}")
        _LOGGER.debug(f"Included columns: {sorted(included_cols)}")
        _LOGGER.debug(f"Excluded (missing in df) columns: {sorted(missing_in_df_cols)}")
        _LOGGER.debug(f"Excluded (redcap group) columns: {sorted(redcap_group_cols)}")
        _LOGGER.debug(f"Excluded (deleted) columns: {sorted(cols_for_deletion)}")
        _LOGGER.debug(f"Excluded (in df but not in reorg) columns: {sorted(excluded_in_df_not_in_reorg)}")

    @staticmethod
    def _get_instrument_for_name(name: str) -> Instrument:
        """Get instrument for a given name.

        Args:
            name: The instrument name.

        Returns:
            The instrument object.

        Raises:
            ValueError: If no instrument found for the given name.
        """
        for repeat_instrument in RepeatInstrument:
            instrument = repeat_instrument.value
            if instrument.name == name:
                return instrument
        raise ValueError(f"No instrument found for value {name}")

    @staticmethod
    def _write_pydantic_model_to_bids_file(
        output_path: Path,
        data: dict,
        schema_name: str,
        subject_id: str,
        session_id: t.Optional[str] = None,
        task_name: t.Optional[str] = None,
        recording_name: t.Optional[str] = None,
        task_entity: t.Optional[str] = None,
    ):
        """Write a Pydantic model (presumably a FHIR resource) to a JSON file.

        Follows the BIDS file name conventions.

        Args:
            output_path: The path to write the file to.
            data: The data to write.
            schema_name: The name of the schema.
            subject_id: The subject ID.
            session_id: The session ID.
            task_name: The task name.
            recording_name: The recording name.
            task_entity: Pre-computed BIDS ``task-`` entity. Callers that also name an audio
                file pass the entity they used, so the sidecar and the audio cannot end up
                with different names. When omitted it is derived from recording_name or
                task_name.
        """
        # sub-<participant_id>_ses-<session_id>_task-<task_name>_run-_metadata.json
        filename = f"sub-{subject_id}"
        if pd.notna(session_id):
            session_id = str(session_id).replace(" ", "-").replace("_", "-")
            filename += f"_ses-{session_id}"
        if task_entity:
            filename += f"_task-{task_entity}"
        elif pd.notna(task_name):
            # The task entity is normalized at ingest (lowercase, non-alphanumerics -> "-",
            # curated aliases) so the internal and published trees share one naming and no
            # parentheses or spaces reach a file name. The raw RedCap name stays in the
            # sidecar contents and the phenotype tables.
            label = recording_name if pd.notna(recording_name) else task_name
            filename += f"_task-{canonical_task_entity(label)}"

        schema_name = schema_name.replace(" ", "-").replace("schema", "").replace("_", "-")
        schema_name = schema_name + "-metadata"
        filename += f"_{schema_name}.json"

        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=False)
        with open(output_path / filename, "w") as f:
            f.write(json.dumps(data, indent=2))

    @staticmethod
    def _output_participant_data_to_metadata_file(
        participant: dict, outdir: Path, audio_files_by_recording: t.Optional[t.Dict[str, Path]] = None,
        max_audio_workers: int = 16, sanitize_audio_format: bool = False, audio_descriptor_dict:OrderedDict = {},
        questionnaire_lookup: t.Optional[t.Dict[tuple, dict]] = None,
        skip_audio_copy: bool = False,
    ) -> t.Tuple[bool, t.Set[str]]:
        """Output participant data to FHIR format.

        Args:
            participant: The participant data dictionary.
            outdir: The output directory path.
            audio_files_by_recording: Dictionary mapping recording IDs to audio file paths (optional).
            max_audio_workers: Number of parallel threads for audio copying (default: 16).
            sanitize_audio_format: Standardize to 16KHz mono-audio
        """
        participant_id = participant["record_id"]
        subject_path = outdir / f"sub-{participant_id}"
        # Administration language is recorded once per participant (the RedCap
        # base row's `selected_language`); every recording of this participant
        # inherits it (there are no mixed-language sessions).
        participant_language = _language_from_selected(participant.get("selected_language"))

        # TODO: prepare a Patient resource to use as the reference for each questionnaire
        # patient = create_fhir_patient(participant)

        session_instrument = BIDSDataset._get_instrument_for_name("sessions")
        recording_instrument = BIDSDataset._get_instrument_for_name("recordings")

        # Collect all audio copy tasks for parallel execution
        audio_copy_tasks = []

        # Pre-scan: determine which recordings have a locatable source file.
        # Only those get sidecars and copy tasks; the rest are logged for QA.
        recordings_with_source: t.Set[str] = set()
        sessions_with_source: t.Set[str] = set()
        recordings_without_source: t.List[t.Tuple[str, str, str]] = []  # (rec_id, rec_name, session_id)
        if audio_files_by_recording is not None:
            for session in participant.get("sessions", []):
                for task in session.get("acoustic_tasks", []):
                    if task is None:
                        continue
                    # The writing loop below skips tasks with no name; count them the same way so
                    # a metadata-only build keeps exactly the sessions a full build keeps.
                    _task_name = task.get("acoustic_task_name")
                    if not _task_name or pd.isna(_task_name):
                        continue
                    for recording in task.get("recordings", []):
                        rec_id = recording.get("recording_id", "")
                        if not rec_id:
                            continue
                        _raw_name = recording.get("recording_name")
                        if not (pd.notna(_raw_name) and str(_raw_name).strip()):
                            recordings_without_source.append(
                                (rec_id, "", session.get("session_id", ""))
                            )
                            continue
                        audio_path = audio_files_by_recording.get(rec_id)
                        if audio_path is not None:
                            try:
                                sz = audio_path.stat().st_size
                            except OSError:
                                sz = 0
                            if sz >= _MIN_AUDIO_BYTES:
                                recordings_with_source.add(rec_id)
                                sessions_with_source.add(session.get("session_id", ""))
                            else:
                                recordings_without_source.append(
                                    (rec_id, recording.get("recording_name", ""),
                                     session.get("session_id", ""))
                                )
                                continue
                        else:
                            recordings_without_source.append(
                                (rec_id, recording.get("recording_name", ""), session.get("session_id", ""))
                            )

        if audio_files_by_recording is not None and not recordings_with_source:
            # No audio at all for this participant -- skip entirely.
            # The caller aggregates these and logs a single QA warning.
            return False, set()

        if recordings_without_source:
            _LOGGER.info(
                "Participant %s: %d of %d recording(s) have no source audio file and will "
                "not receive a sidecar or audio copy. QA: verify these are expected "
                "(truncated source, missing from Wasabi, collection error). "
                "recording_ids: %s",
                participant_id,
                len(recordings_without_source),
                len(recordings_with_source) + len(recordings_without_source),
                ", ".join(r[0] for r in recordings_without_source),
            )


        # validated questionnaires are asked per session
        sessions_rows = []
        
        for session in participant["sessions"]:
            # Columns removed at ingest (disposition=drop) are absent from the session dict.
            sessions_row = {key: session[key] for key in session_instrument.columns if key in session}
            sessions_rows.append(sessions_row)
            session_id = session["session_id"]
            # TODO: prepare a session resource to use as the encounter reference for
            # each session questionnaire
            session_path = subject_path / f"ses-{session_id}"
            audio_output_path = session_path / "audio"
            if not audio_output_path.exists():
                audio_output_path.mkdir(parents=True, exist_ok=True)

            # Detect recording_name collisions within a session: two distinct
            # recordings that map to the same BIDS task entity would silently
            # overwrite each other's sidecar and drop one audio file (the copy is
            # skipped when the destination already exists). Track the normalized
            # entity -> recording_id and warn on a clash.
            seen_recording_entities: t.Dict[str, str] = {}
            # Two acoustic tasks in one session whose names differ only in case
            # (e.g. "Free speech" and "Free Speech") share one BIDS entity. Recordings
            # under each task keep their own names and both tasks remain in
            # phenotype/task/acoustic_task.tsv.
            seen_task_entities: t.Dict[str, str] = {}

            # multiple acoustic tasks are asked per session
            for task in session["acoustic_tasks"]:
                if task is None:
                    continue
                
                acoustic_task_name = task.get("acoustic_task_name")
                if not acoustic_task_name or pd.isna(acoustic_task_name):
                    _LOGGER.warning(f"Skipping task with missing acoustic_task_name for participant {participant_id}, session {session_id}")
                    continue
                
                acoustic_task_name = acoustic_task_name.replace(" ", "-").replace("_", "-")
                _task_entity = canonical_task_entity(acoustic_task_name)
                _task_id = task.get("acoustic_task_id")
                _task_collided, _prior_task = _note_entity_collision(
                    seen_task_entities, _task_entity, _task_id
                )
                if _task_collided:
                    _LOGGER.warning(
                        "acoustic_task_name collision: %r maps to the same BIDS task entity "
                        "for participant %s session %s (acoustic_task_id %s and %s); "
                        "recordings are unaffected and both tasks remain in "
                        "phenotype/task/acoustic_task.tsv",
                        acoustic_task_name, participant_id, session_id, _prior_task, _task_id,
                    )
                # Skip tasks with no source audio — no recordings to process.
                if audio_files_by_recording is not None:
                    _task_has_audio = any(
                        recording.get("recording_id", "") in recordings_with_source
                        for recording in task.get("recordings", [])
                    )
                    if not _task_has_audio:
                        continue

                # Population (from the acoustic task's cohort) disambiguates the
                # few families that exist in both peds and adult (picture-description).
                task_population = _population_from_cohort(task.get("acoustic_task_cohort"))

                # prefix is used to name audio files, if they are copied over
                prefix = f"sub-{participant_id}_ses-{session_id}"
                # there may be more than one recording per acoustic task
                for recording in task["recordings"]:
                    # collision check: normalized recording_name is the BIDS task
                    # entity; a clash between two recording_ids overwrites files.
                    _raw_rec_name = recording.get("recording_name")
                    _rec_name = str(_raw_rec_name).strip() if pd.notna(_raw_rec_name) else ""
                    if not _rec_name:
                        # Mirrors the acoustic_task_name guard above: without a name there is
                        # no task entity, and emitting one anyway produced a "task-nan" audio
                        # file whose sidecar was named after the acoustic task instead.
                        _LOGGER.warning(
                            "Skipping recording with missing recording_name for participant %s, "
                            "session %s (recording_id %s)",
                            participant_id, session_id, recording.get("recording_id"),
                        )
                        continue
                    _rec_entity = canonical_task_entity(_rec_name)
                    _rec_id = recording.get("recording_id")
                    _rec_collided, _prior = _note_entity_collision(
                        seen_recording_entities, _rec_entity, _rec_id
                    )
                    if _rec_collided:
                        _LOGGER.warning(
                            "recording_name collision: %r maps to the same BIDS task "
                            "entity for participant %s session %s (recording_id %s and "
                            "%s); the later sidecar overwrites the earlier and one audio "
                            "file is dropped",
                            _rec_name, participant_id, session_id, _prior, _rec_id,
                        )
                    # Skip recordings whose source audio is missing -- no sidecar, no copy.
                    # The per-participant log above already named them for QA.
                    if audio_files_by_recording is not None and _rec_id not in recordings_with_source:
                        continue

                    meta_data = convert_response_to_bids_metadata(
                        recording,
                        questionnaire_name=recording_instrument.name,
                        mapping_name=recording_instrument.schema_name_clobbered,
                        columns=recording_instrument.columns,
                        audio_task_descriptions=audio_descriptor_dict,
                        questionnaire_lookup=questionnaire_lookup,
                        population=task_population,
                        language=participant_language,
                    )
                    BIDSDataset._write_pydantic_model_to_bids_file(
                        audio_output_path,
                        meta_data,
                        schema_name=recording_instrument.schema_name_clobbered,
                        subject_id=participant_id,
                        session_id=session_id,
                        task_name=acoustic_task_name,
                        recording_name=_rec_name,
                        task_entity=_rec_entity,
                    )
                    if audio_files_by_recording is None:
                        continue
                    audio_file = audio_files_by_recording.get(recording["recording_id"], None)

                    if not audio_file:
                        continue

                    # Schedule audio copy (to be executed in parallel later)
                    ext = audio_file.suffix
                    audio_file_destination = (
                        audio_output_path / f"{prefix}_task-{_rec_entity}{ext}"
                    )
                    
                    if not skip_audio_copy and not audio_file_destination.exists():
                        audio_copy_tasks.append((audio_file, audio_file_destination))

        # Execute all audio copies in parallel
        if audio_copy_tasks:
            _copy_audio_files_parallel(audio_copy_tasks, max_workers=max_audio_workers, sanitize_audio_format=sanitize_audio_format)

        # Remove session directories that have no audio files. This can
        # happen when every recording in a session had no source file.
        removed_sessions: t.Set[str] = set()
        for session in participant["sessions"]:
            session_id = session["session_id"]
            session_audio = subject_path / f"ses-{session_id}" / "audio"
            if not session_audio.is_dir():
                continue
            if skip_audio_copy:
                has_audio = session_id in sessions_with_source
            else:
                has_audio = any(f.suffix == ".wav" for f in session_audio.iterdir())
            if not has_audio:
                shutil.rmtree(session_audio)
                removed_sessions.add(session_id)
                session_dir = session_audio.parent
                if session_dir.is_dir() and not any(session_dir.iterdir()):
                    session_dir.rmdir()

        # Save sessions.tsv, excluding sessions whose directories were removed
        sessions_rows = [r for r in sessions_rows if r["session_id"] not in removed_sessions]
        sessions_df = pd.DataFrame(sessions_rows)
        if not os.path.exists(subject_path):
            os.mkdir(subject_path)
        sessions_tsv_path = subject_path / "sessions.tsv"
        sessions_df.to_csv(sessions_tsv_path, sep="\t", index=False)

        return True, recordings_with_source
    @staticmethod
    def load_phenotype_file(
        phenotype_filepath: Path,
    ) -> t.Tuple[pd.DataFrame, str, t.Dict[str, t.Any], t.Dict[str, t.Any]]:
        """Load a phenotype TSV and its JSON sidecar, keeping the sidecar's schema wrapper.

        ``redcap2bids`` writes sidecars as ``{schema_name: {description, ..., data_elements}}``.
        Older trees may carry a flat ``{column: element}`` dictionary instead; both are accepted.

        Returns:
            Tuple of (DataFrame, schema_name, header, data_elements) where ``header`` holds every
            schema-level key other than ``data_elements`` so a writer can rebuild the wrapper with
            ``{schema_name: {**header, "data_elements": ...}}``.
        """
        df = pd.read_csv(phenotype_filepath.with_suffix(".tsv"), sep="\t")
        with open(phenotype_filepath.with_suffix(".json"), "r") as f:
            raw = json.load(f)

        wrapped_keys = [
            key for key, value in raw.items()
            if isinstance(value, dict) and "data_elements" in value
        ]
        if wrapped_keys:
            schema_name = wrapped_keys[0]
            header = {k: v for k, v in raw[schema_name].items() if k != "data_elements"}
            data_elements: t.Dict[str, t.Any] = {}
            for key in wrapped_keys:
                data_elements.update(raw[key]["data_elements"])
        else:
            schema_name = phenotype_filepath.stem
            header = {"description": ""}
            data_elements = raw

        return df, schema_name, header, data_elements

    @staticmethod
    def load_phenotype_data(phenotype_filepath: Path) -> t.Tuple[pd.DataFrame, t.Dict[str, t.Any]]:
        """Load phenotype data from TSV and JSON files.

        Convenience wrapper around :meth:`load_phenotype_file` that discards the schema wrapper.
        """
        phenotype_name = phenotype_filepath.stem
        df, _, _, phenotype = BIDSDataset.load_phenotype_file(phenotype_filepath)

        # Add record_id to phenotype if missing
        if df.shape[1] > 0:
            phenotype_has_id = 'record_id' in list(phenotype.keys()) or 'participant_id' in list(phenotype.keys())
            df_has_id = 'record_id' in df.columns or 'participant_id' in df.columns
            if not phenotype_has_id and not df_has_id:
                phenotype = BIDSDataset._add_record_id_to_phenotype(phenotype)

        # Validate column count
        if len(phenotype) != df.shape[1]:
            _LOGGER.warning(
                f"Phenotype {phenotype_name} has {len(phenotype)} columns, but the data has {df.shape[1]} columns."
            )

        return df, phenotype

    @staticmethod
    def _map_series(series: pd.Series, mapping: dict) -> pd.Series:
        """
        Map values in a pandas Series using a provided mapping dictionary. Log any issues arising.
        
        Args:
            series: The pandas Series to map.
            mapping: The mapping dictionary.
        Returns:
            The mapped pandas Series.
        """
        ids_before = series.copy()
        series = series.map(mapping).fillna(series)
        idxUnchanged = ids_before == series
        n_unchanged = idxUnchanged.sum()
        proportion = ((len(ids_before) - n_unchanged) / len(ids_before)) if len(ids_before) > 0 else 0
        _LOGGER.debug(f"Remapped {proportion:3.1%} ({len(ids_before) - n_unchanged}/{len(ids_before)}) of IDs for '{series.name}'")
        if n_unchanged > 0:
            _LOGGER.warning((
                f"A subset of IDs ({1-proportion:3.1%}, {n_unchanged}) are missing "
                f"remapping for '{series.name}': {set(ids_before[idxUnchanged])}"
            ))
        return series

    @staticmethod
    def _deidentify_phenotype(df: pd.DataFrame, phenotype: dict, participant_ids_to_remove: t.List[str] = [], participant_ids_to_remap: dict = {}, participant_session_id_to_remap: dict = {}) -> t.Tuple[pd.DataFrame, dict]:
        """
        Apply deidentification operations to phenotype data.
        
        Args:
            df: DataFrame containing the phenotype data
            phenotype: Dictionary containing the phenotype metadata
            
        Returns:
            Tuple of (deidentified_df, deidentified_phenotype_dict)
        """
        # Rename record_id to participant_id
        if "record_id" in df.columns:
            df, phenotype = BIDSDataset._rename_record_id_to_participant_id(df, phenotype)

        # Remove participants
        idx = df["participant_id"].isin(participant_ids_to_remove)
        if idx.any():
            _LOGGER.info(f"Removing {idx.sum()} participants from phenotype data.")
            df = df.loc[~idx]

        # Remap IDs
        if participant_ids_to_remap and "participant_id" in df.columns:
            remap_partial = partial(remap_id, id_mapping=participant_ids_to_remap)
            df.loc[:, "participant_id"] = BIDSDataset._map_series(df["participant_id"], remap_partial)

        if participant_session_id_to_remap:
            remap_partial = partial(remap_id, id_mapping=participant_session_id_to_remap, id_type="session")
            for col in df.columns:
                if "session_id" in col:
                    df.loc[:, col] = BIDSDataset._map_series(df[col], remap_partial)

        # Sanitize task labels in task phenotype tables (e.g., phenotype/task/acoustic_task.tsv)
        if "acoustic_task_name" in df.columns:
            df.loc[:, "acoustic_task_name"] = df["acoustic_task_name"].apply(
                lambda v: normalize_task_label(v) if pd.notna(v) else v
            )

        return df, phenotype

    @staticmethod
    def _is_redcap_group(column: str) -> bool:
        """Check if a column is a RedCap group column."""
        pattern = re.compile(r'^(?P<base>.+?)___(?P<suffix>.+)$')
        match = pattern.match(column)
        return match is not None

    @staticmethod
    def _clean_phenotype_data(df: pd.DataFrame, phenotype: dict) -> t.Tuple[pd.DataFrame, dict]:
        """
        Apply data cleaning operations to phenotype data.
        
        Args:
            df: DataFrame containing the phenotype data
            phenotype: Dictionary containing the phenotype metadata
            
        Returns:
            Tuple of (cleaned_df, cleaned_phenotype_dict)
        """
        # Fix alcohol column date values
        df = BIDSDataset._fix_alcohol_column(df)

        for c in df.columns:
            if BIDSDataset._is_redcap_group(c):
                # convert from whatever the text is to 1
                if df[c].dropna().nunique() > 1:
                    _LOGGER.warning(
                        f"RedCap checkbox column {c} has unexpected non-binary values: "
                        f"{df[c].dropna().unique().tolist()}. Converting to binary."
                    )
                df[c] = df[c].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != "" else np.nan).astype(pd.Int8Dtype())
                # TODO: phenotype changes for checkbox columns?
        # Warn about empty columns - but keep them as some are expected
        BIDSDataset._warn_about_empty_columns(df, phenotype)
        
        # Add derived columns
        if ("gender_identity" in df.columns) and ("specify_gender_identity" in df.columns):
            df, phenotype = BIDSDataset._add_sex_at_birth_column(df, phenotype)

        return df, phenotype

    @staticmethod
    def _fix_alcohol_column(df: pd.DataFrame) -> pd.DataFrame:
        """Fix known date values in the alcohol_amt column."""
        if "alcohol_amt" in df:
            date_fix_map = {
                "4-Mar": "3 - 4",
                "6-May": "5 - 6",
                "9-Jul": "7 - 9",
            }
            df["alcohol_amt"] = df["alcohol_amt"].apply(
                lambda x: date_fix_map[x] if x in date_fix_map else x
            )
        return df

    @staticmethod
    def _warn_about_empty_columns(df: pd.DataFrame, phenotype: dict) -> None:
        """Warn about columns that are empty."""
        empty_columns = []
        for column in df.columns:
            if df[column].isnull().all() and BIDSDataset._is_redcap_group(column) is False:
                empty_columns.append(column)
        
        if empty_columns:
            _LOGGER.warning(f"Found {len(empty_columns)} empty columns: {empty_columns}")

    @staticmethod
    def _remove_sensitive_columns(df: pd.DataFrame, phenotype: dict) -> t.Tuple[pd.DataFrame, dict]:
        """Remove columns with sensitive data. Deprecated: use disposition column in field map."""

        # TODO: Revisit this list. The deidentification is now implicitly applied by the
        # use of bids_field_reorganization.csv to select only desired columns.
        # This is kept here for now as an extra layer of safety, particularly for pediatrics
        # which has not had the RedCap dataset imported/tested yet. In the future, this code may
        # remove useful non-PHI columns, so care should be taken.
        columns_to_drop = [
            "state_province",
            "zipcode",
            "other_edu_level",
            "others_household_specify",
            "diagnosis_alz_dementia_mci_ca_rudas_score",
            "diagnosis_alz_dementia_mci_ca_mmse_score",
            "diagnosis_alz_dementia_mci_ca_moca_score",
            "diagnosis_alz_dementia_mci_ca_adas_cog_score",
            "diagnosis_alz_dementia_mci_ca_other",
            "diagnosis_alz_dementia_mci_ca_other_score",
            "diagnosis_parkinsons_ma_updrs_part_i_score",
            "diagnosis_parkinsons_ma_updrs_part_ii_score",
            "diagnosis_parkinsons_ma_updrs_part_iii_score",
            "diagnosis_parkinsons_ma_updrs_part_iv_score",
            "diagnosis_parkinsons_non_motor_symptoms_yes",
            "traumatic_event",
            # pediatric columns
            "city",
            "state_province",
            "peds_zipcode",
            "peds_other_race_specify",
            "peds_other_primary_language",
            "peds_mc_conditions_other_specified",
            "peds_mc_chronic_medical_condition_specified",
            "peds_mc_genetic_syndromes_specified",
            "peds_mc_hospitalized_specified",
            "peds_mc_allergies_specified",
            "peds_mc_dif_swallowing_specified",
            "peds_mc_ear_inf_ant_py_specified",
            "peds_mc_etp_procedure",
            "peds_mc_eval_voice_swal_c_specified",
            "peds_mc_ft_specified",
            "peds_mc_reflux_specified",
            "peds_mc_hl_specified",
            "peds_mc_hw_voice_2_w_specified",
            "peds_mc_no_surgeries_procedure",
            "peds_mc_stridor_specified",
            "peds_mc_ox_sup_specified",
            "peds_mc_meds_specified",
            "peds_mc_v_dis_specified",
            "peds_mc_a_therapy_specified",
            "peds_mc_surgery_t_vc_a_specified",
            "peds_mc_fr_inf_tons_specified",
            "peds_mc_fr_v_f_specified",
            # below could be considered for inclusion in the future
            "peds_mc_tonsillectomy_date",
            "peds_mc_adenoidectomy_date",
            "peds_mc_neck_mass_branchial_cleft_cyst_surgery_date",
            "peds_mc_etp_procedure_date",
            "peds_mc_neck_mass_dermoid_cyst_surgery_date",
            "peds_mc_neck_mass_enlarged_lymph_node_surgery_date",
            "peds_mc_lingual_tonsillectomy_date",
            "peds_mc_no_surgeries_procedure_date",
            "peds_mc_neck_mass_thyroglossal_duct_cyst_surgery_date",
            "peds_mc_neck_mass_hyroid_nodule_or_cancer_surgery_date",
            "peds_mc_v_dis_specified"
        ]
        df, phenotype = BIDSDataset._drop_columns_from_df_and_data_dict(
            df, phenotype, columns_to_drop, "Removing PHI containing columns"
        )

        if 'gender_identity' in df.columns:
            if 'sex_at_birth' in df.columns:
                _LOGGER.info(f"sex_at_birth value_counts: {df['sex_at_birth'].value_counts(dropna=False).to_dict()}")
            _LOGGER.info(f"gender_identity value_counts: {df['gender_identity'].value_counts(dropna=False).to_dict()}")
            df, phenotype = BIDSDataset._drop_columns_from_df_and_data_dict(
                df, phenotype, ["gender_identity"], "Remove sensitive demographic columns"
            )
        
        return df, phenotype

    @staticmethod
    def _drop_columns_from_df_and_data_dict(
        df: pd.DataFrame, phenotype: dict, columns_to_drop: t.List[str], message: str
    ) -> t.Tuple[pd.DataFrame, dict]:
        """Drop columns from the DataFrame and phenotype dictionary."""
        columns_to_drop_in_df = [col for col in columns_to_drop if col in df.columns]
        
        if columns_to_drop_in_df:
            _LOGGER.info(f"{message}: {columns_to_drop_in_df}")
            df = df.drop(columns=columns_to_drop_in_df)
            phenotype = {k: v for k, v in phenotype.items() if k not in columns_to_drop_in_df}
        
        return df, phenotype

    @staticmethod
    def _add_record_id_to_phenotype(phenotype: dict) -> dict:
        """Add record_id to phenotype metadata if missing."""
        if 'record_id' in phenotype:
            return phenotype

        phenotype_updated = {
            'record_id': {
                "description": "Unique identifier for each participant."
            }
        }
        phenotype_updated.update(phenotype)
        return phenotype_updated

    @staticmethod
    def _rename_record_id_to_participant_id(df: pd.DataFrame, phenotype: dict) -> t.Tuple[pd.DataFrame, dict]:
        """Rename record_id column to participant_id."""
        phenotype_updated = {}
        for name, value in phenotype.items():
            if name == 'record_id':
                phenotype_updated['participant_id'] = value
                continue
            phenotype_updated[name] = value

        # Only rename if record_id exists and participant_id doesn't exist
        if 'record_id' in df.columns and 'participant_id' not in df.columns:
            df = df.rename(columns={"record_id": "participant_id"})
        elif 'record_id' in df.columns and 'participant_id' in df.columns:
            # If both exist, drop record_id since participant_id takes precedence
            df = df.drop(columns=['record_id'])
            # Remove record_id from phenotype_updated if it exists
            phenotype_updated = {k: v for k, v in phenotype_updated.items() if k != 'record_id'}
        
        return df, phenotype_updated

    @staticmethod
    def _add_sex_at_birth_column(df: pd.DataFrame, phenotype: dict) -> t.Tuple[pd.DataFrame, dict]:
        """Add sex_at_birth column derived from gender_identity and specify_gender_identity."""
        df["sex_at_birth"] = None
        for sex_at_birth in ["Male", "Female"]:
            idx = (
                df["gender_identity"].str.contains(sex_at_birth)
                & df["specify_gender_identity"].notnull()
            )
            df.loc[idx, "sex_at_birth"] = sex_at_birth

        # Re-order columns to place sex_at_birth after gender_identity
        phenotype_reordered = deepcopy(phenotype)
        first_key = next(iter(phenotype))
        # reset data elements for reordering
        phenotype_reordered[first_key]["data_elements"] = {}
        data_elements_updated = phenotype_reordered[first_key]["data_elements"]
        columns = []
        for c in df.columns:
            if c == "specify_gender_identity":
                # this continue implicitly removes this column from the output
                continue
            elif c == "gender_identity":
                columns.append(c)
                columns.append("sex_at_birth")
                data_elements_updated[c] = phenotype[first_key]["data_elements"][c]
                data_elements_updated["sex_at_birth"] = {
                    "description": "The sex at birth for the individual."
                }
            elif c == "sex_at_birth":
                continue
            else:
                columns.append(c)
                if c in phenotype[first_key]["data_elements"]:
                    data_elements_updated[c] = phenotype[first_key]["data_elements"][c]

        df = df[columns]
        return df, phenotype_reordered

    @staticmethod
    def _reduce_id_length(df: pd.DataFrame, id_name: str) -> pd.DataFrame:
        """Reduce the length of ID columns to 8 characters."""
        if id_name in df.columns:
            df = df.copy()  # Avoid SettingWithCopyWarning
            df[id_name] = df[id_name].apply(reduce_id_length)
        return df

    @staticmethod
    def load_remap_id_list(publish_config_dir: Path) -> t.Dict[str, str]:
        audio_to_remap_path = publish_config_dir / "id_remapping.json"
        if not audio_to_remap_path.exists():
            raise FileNotFoundError(f"ID remapping file {audio_to_remap_path} does not exist.")

        with open(audio_to_remap_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"ID remapping file {audio_to_remap_path} should contain a dict of participant_id:new_id.")

        return data
    
    @staticmethod
    def load_remap_session_id_list(publish_config_dir: Path) -> t.Dict[str, str]:
        session_id_to_remap_path = publish_config_dir / "session_id_remapping.json"
        if not session_id_to_remap_path.exists():
            raise FileNotFoundError(f"ID remapping file {session_id_to_remap_path} does not exist.")

        with open(session_id_to_remap_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"ID remapping file {session_id_to_remap_path} should contain a dict of session_id:new_id.")

        return data

    @staticmethod
    def map_sequential_session_ids(folder_path: Path, sequential: bool = False) ->  t.Dict[str, str]:
        """Map session UUIDs to shortened IDs. Deprecated: use _build_session_id_mapping instead."""
        folder = Path(folder_path)
        session_files = list(folder.rglob("sessions.tsv"))
        
        # add in the session.tsv file from the phenotype folder
        # this *should* have all possible session ids, so the scan of sessions.tsv
        # is redundant / for safety only

        phenotype_session_files = list(folder.joinpath("phenotype").rglob("session.tsv"))
        session_files = session_files + phenotype_session_files

        if not session_files:
            _LOGGER.warning((
                f"No 'sessions.tsv' files found under {folder_path}"
                f" - cannot map session IDs."
            ))
            return {}

        df_list = []
        for session_file in session_files:
            try:
                df_session = pd.read_csv(session_file, sep="\t", dtype=str)
                if 'record_id' in df_session.columns and 'session_id' in df_session.columns:
                    df_list.append(df_session[['record_id', 'session_id']])
                elif 'participant_id' in df_session.columns and 'session_id' in df_session.columns:
                    df_session = df_session.rename(columns={"participant_id": "record_id"})
                    df_list.append(df_session[['record_id', 'session_id']])
                else:
                    _LOGGER.warning(f"'sessions.tsv' file {session_file} is missing required columns.")
            except Exception as e:
                _LOGGER.error(f"Error reading 'sessions.tsv' file {session_file}: {e}")
        
        if not df_list:
            _LOGGER.warning((
                f"No valid 'sessions.tsv' files with required columns found under {folder_path}"
                f" - cannot map session IDs."
            ))
            return {}
        combined = pd.concat(df_list).drop_duplicates().reset_index(drop=True)
        # Sort so all rows of a record_id appear together
        combined.sort_values(by=['record_id', 'session_id'], inplace=True)
        
        if sequential:
            # we create a sequential integer using the original session_id to sort
            combined['mapped_id'] = combined.groupby('record_id').cumcount() + 1
        else:
            # we assume the original session_id is a uuid, and we trim it to 8 characters
            # we do this in a way that ensures uniqueness within each participant
            combined['mapped_id'] = combined['session_id'].map(reduce_id_length)
            
            # check for duplicates *across* participants - this is not an issue, but may confuse
            # users, so we log a warning and restore the original session IDs up to 16 char
            # probability of collision for 8 char, 10,000 IDs is ~0.0116
            # probability of collision for 16 char, 1,000,000 IDs is ~2.7105e-8
            duplicates = combined.duplicated(subset=['mapped_id'], keep=False)
            if duplicates.any():
                # use 16 character session IDs for participants with duplicates
                duplicate_mapped_id = combined.loc[duplicates, 'mapped_id'].unique()
                for mapped_id in duplicate_mapped_id:
                    idx = combined['mapped_id'] == mapped_id
                    combined.loc[idx, 'mapped_id'] = combined.loc[idx, 'session_id'].map(
                        lambda x: reduce_id_length(x, length=16)
                    )
                
                n_dupe = duplicate_mapped_id.shape[0]
                _LOGGER.warning((
                    f"Session ID remapping resulted in duplicate session_id for {n_dupe} sessions. "
                    f"Using original session IDs for these cases."
                ))
        
        # enforce lower case as it is convention in BIDS
        combined['mapped_id'] = combined['mapped_id'].astype(str).str.lower()
        session_id_dict = combined.set_index('session_id')['mapped_id'].to_dict()
        _LOGGER.info(f"Created a session_id mapping of {len(session_id_dict):,} IDs.")
        return session_id_dict

    @staticmethod
    def load_participant_ids_to_remove(publish_config_dir: Path) -> t.List[str]:
        """Load list of participant IDs to remove from JSON file."""
        participant_to_remove_path = publish_config_dir / "participants_to_remove.json"
        if not participant_to_remove_path.exists():
            # If file doesn't exist, raise an error
            raise FileNotFoundError(f"Participant IDs to remove file {participant_to_remove_path} does not exist.")

        with open(participant_to_remove_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Participant IDs to remove file {participant_to_remove_path} should contain a list of participant IDs.")
        
        _LOGGER.info(f"Loaded {len(data)} participant IDs to remove: {data}.")
        return data
    
    @staticmethod
    def _check_phenotype_tables_in_field_map(phenotype_dir: Path) -> None:
        """Stop before writing if any phenotype table (or sessions.tsv's table) is unknown.

        redcap2bids names every table after a field-map ``schema_name``, so an unknown name means
        the tree was built with a different field map, and its dispositions cannot be trusted.
        """
        if BIDSDataset._cached_field_map_df is None:
            BIDSDataset._cached_field_map_df = BIDSDataset._load_reorganization_file(exclude_dropped=False)
        known = set(BIDSDataset._cached_field_map_df["schema_name"].dropna())
        unknown = []
        if BIDSDataset._SESSIONS_SCHEMA not in known:
            unknown.append(f"{BIDSDataset._SESSIONS_SCHEMA} (sessions.tsv)")
        if phenotype_dir.exists():
            for tsv in sorted(phenotype_dir.rglob("*.tsv")):
                _, schema_name, _, _ = BIDSDataset.load_phenotype_file(tsv)
                if schema_name not in known:
                    unknown.append(f"{schema_name} ({tsv.relative_to(phenotype_dir)})")
        if unknown:
            raise ValueError(
                "Phenotype tables not in the field map (was the tree built with a different "
                f"bids_field_organization.csv?): {unknown}"
            )

    @staticmethod
    def _filestems_for_recording_ids(
        bids_path: Path,
        participant_ids: t.Iterable[str],
        recording_ids: t.AbstractSet[str],
        max_workers: int = 16,
    ) -> t.List[str]:
        """Filestems in *bids_path* of the recordings whose sidecar ``recording_id`` is listed.

        Only the given participants' sidecars are read, and only when *recording_ids* is
        non-empty. Each stem is cut at the task entity, matching how filestem exclusions compare.
        """
        if not recording_ids:
            return []
        suffix = "_recording-metadata.json"

        def _scan(pid: str) -> t.List[str]:
            found = []
            for sidecar in (bids_path / f"sub-{pid}").glob(f"ses-*/audio/*{suffix}"):
                try:
                    rid = json.loads(sidecar.read_text()).get("recording_id", "")
                except (OSError, json.JSONDecodeError):
                    continue
                if str(rid).strip().lower() in recording_ids:
                    stem = sidecar.name[: -len(suffix)]
                    parts = stem.split("_")
                    task_idx = next((i for i, p in enumerate(parts) if p.startswith("task-")), None)
                    found.append("_".join(parts[: task_idx + 1]) if task_idx is not None else stem)
            return found

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            stems = [s for chunk in executor.map(_scan, sorted(participant_ids)) for s in chunk]
        _LOGGER.info(
            "audio_recording_ids_to_remove: %d ID(s) resolved to %d filestem(s) in this tree.",
            len(recording_ids), len(stems),
        )
        return stems

    @staticmethod
    def _report_exclusion_coverage(
        bids_path: Path,
        filestems: t.Iterable[str],
        recording_ids: t.AbstractSet[str],
        input_tree_participants: t.AbstractSet[str],
    ) -> None:
        """Warn when an exclusion list cannot match this tree, instead of silently removing nothing."""
        stems = list(filestems)
        if stems:
            subjects = {Path(s).stem.split("_")[0][len("sub-"):] for s in stems}
            matched = subjects & set(input_tree_participants)
            level = logging.WARNING if not matched else logging.INFO
            _LOGGER.log(
                level,
                "audio_filestems_to_remove: %d stem(s) over %d subject(s); %d of those subjects are in "
                "this tree.%s",
                len(stems), len(subjects), len(matched),
                " None match: the list uses IDs from another registration and removes nothing here "
                "(use audio_recording_ids_to_remove.json)." if not matched else "",
            )
        if recording_ids:
            recording_tsv = bids_path / "phenotype" / "task" / "recording.tsv"
            if recording_tsv.exists():
                present = set(
                    pd.read_csv(recording_tsv, sep="\t", usecols=["recording_id"], dtype=str)
                    ["recording_id"].dropna().str.lower()
                )
                _LOGGER.info(
                    "audio_recording_ids_to_remove: %d ID(s), %d present in this tree.",
                    len(recording_ids), len(recording_ids & present),
                )

    @staticmethod
    def load_audio_recording_ids_to_remove(publish_config_dir: Path) -> t.Set[str]:
        """Recording IDs to remove (optional ``audio_recording_ids_to_remove.json``), lowercased.

        Preferred over filestems: ``recording_id`` is stable, while filestems embed participant
        IDs and task labels that change between registrations (the pediatric filestem list uses
        pre-RedCap subject IDs and old task names, and matches nothing in a v4 tree).
        """
        path = publish_config_dir / "audio_recording_ids_to_remove.json"
        if not path.exists():
            return set()
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} should contain a list of recording IDs.")
        return {str(r).strip().lower() for r in data if str(r).strip()}

    @staticmethod
    def load_audio_filestems_to_remove(publish_config_dir: Path) -> t.List[str]:
        """Load list of audio file stems to remove from JSON file."""
        audio_to_remove_path = publish_config_dir / "audio_filestems_to_remove.json"
        if not audio_to_remove_path.exists():
            # If file doesn't exist, raise an error
            raise FileNotFoundError(f"Audio filestems to remove file {audio_to_remove_path} does not exist.")

        with open(audio_to_remove_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Audio filestems to remove file {audio_to_remove_path} should contain a list of audio file stems.")
        
        return data

    @staticmethod
    def load_audio_tasks_to_include(deidentify_config_dir: Path) -> t.List[str]:
        """Load list of audio tasks that are to be included."""
        audio_tasks_to_include_path = deidentify_config_dir / "audio_tasks_to_include.json"
        if not audio_tasks_to_include_path.exists():
            # If file doesn't exist, raise an error
            raise FileNotFoundError(f"Inclusion audio tasks file {audio_tasks_to_include_path} does not exist.")

        with open(audio_tasks_to_include_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Inclusion audio tasks file {audio_tasks_to_include_path} should contain a list of audio task names.")
        
        return data

    def deidentify(
        self,
        outdir: t.Union[str, Path],
        deidentify_config_dir: Path,
        skip_audio: bool = False,
        skip_audio_features: bool = False,
        max_workers: int = 16,
        disposition_level: t.Optional[DispositionLevel] = None,
        keep_shifted_dates: bool = False,
    ) -> 'BIDSDataset':
        """Create a deidentified version of the BIDS dataset.

        Uses per-participant parallelization: each participant directory is
        processed as an independent unit (audio, features, sidecars).  Phenotype
        tables and quality metrics are processed globally afterward, filtered to
        only the participants that actually produced output.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=False)

        # --- config loading ---
        participant_ids_to_remap = BIDSDataset.load_remap_id_list(deidentify_config_dir)
        audio_filestems_to_remove = BIDSDataset.load_audio_filestems_to_remove(deidentify_config_dir)
        audio_tasks_to_include = BIDSDataset.load_audio_tasks_to_include(deidentify_config_dir)

        # Allowlist replaces exclusion list
        input_tree_participants = {
            d.name[4:] for d in self.data_path.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        }
        participant_allowlist = BIDSDataset._load_participant_allowlist(
            deidentify_config_dir, input_tree_participants
        )

        # Session mapping from allowlist participants
        participant_session_id_to_remap = BIDSDataset._build_session_id_mapping(
            self.data_path, participant_allowlist
        )

        configured_filestems = list(audio_filestems_to_remove)
        # Recording IDs are resolved to this tree's filestems, so audio, sidecars, features and
        # quality metrics are all filtered by the one filestem mechanism below.
        recording_ids_to_remove = BIDSDataset.load_audio_recording_ids_to_remove(deidentify_config_dir)
        audio_filestems_to_remove = list(audio_filestems_to_remove) + BIDSDataset._filestems_for_recording_ids(
            self.data_path, participant_allowlist, recording_ids_to_remove, max_workers=max_workers
        )
        audio_filestems_to_remove = BIDSDataset._expand_filestems_for_deidentification(
            audio_filestems_to_remove,
            participant_ids_to_remap=participant_ids_to_remap,
            participant_session_id_to_remap=participant_session_id_to_remap,
        )

        # --- per-participant processing ---
        participant_dirs = sorted(
            self.data_path / f"sub-{pid}"
            for pid in participant_allowlist
            if (self.data_path / f"sub-{pid}").is_dir()
        )
        _LOGGER.info("Deidentifying %d participants (max_workers=%d).", len(participant_dirs), max_workers)

        # Exact labels, globs ("identifying-pictures-*") and regexes ("re:...").
        normalized_include_tasks = TaskMatcher(audio_tasks_to_include)
        canonical_exclusions = {
            sanitize_task_entity_in_bids_stem(Path(s).stem)
            for s in audio_filestems_to_remove
        }
        BIDSDataset._report_exclusion_coverage(
            self.data_path, configured_filestems, recording_ids_to_remove, input_tree_participants
        )

        def _process_one(pdir: Path) -> t.Optional[str]:
            pid = pdir.name[4:]
            new_pid = participant_ids_to_remap.get(pid, pid)
            try:
                had_output = BIDSDataset._deidentify_participant_files(
                    pdir, outdir, participant_ids_to_remap,
                    participant_session_id_to_remap,
                    audio_filestems_to_remove, audio_tasks_to_include,
                    skip_audio, skip_audio_features,
                    _normalized_include_tasks=normalized_include_tasks,
                    _canonical_exclusions=canonical_exclusions,
                    disposition_level=disposition_level or DispositionLevel.RELEASE,
                    keep_shifted_dates=keep_shifted_dates,
                )
                if had_output:
                    return pid
                _LOGGER.info("Participant %s excluded: no audio after task filtering.", pid)
                return None
            except Exception:
                _LOGGER.exception("Failed to process participant %s.", pid)
                out_participant = outdir / f"sub-{new_pid}"
                if out_participant.exists():
                    shutil.rmtree(out_participant)
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(_process_one, participant_dirs),
                total=len(participant_dirs),
                desc="Deidentifying participants",
            ))

        # With skip_audio the per-participant pass still walks every sidecar, so its results say
        # exactly who has output; no override is needed.
        participants_with_output = {pid for pid in results if pid is not None}
        _LOGGER.info(
            "Per-participant processing complete: %d of %d produced output.",
            len(participants_with_output), len(participant_dirs),
        )
        if not participants_with_output and participant_dirs:
            raise RuntimeError(
                f"Deidentify produced zero output from {len(participant_dirs)} participants. "
                "Check the logs above for per-participant errors."
            )

        # Derive exclusion set for phenotype/QM (everyone NOT in output set)
        participant_ids_to_exclude = list(input_tree_participants - participants_with_output)

        # --- phenotype ---
        column_value_verdicts = BIDSDataset._load_column_value_reviews(deidentify_config_dir)
        has_review_verdicts = bool(column_value_verdicts)
        # Default: keep review columns only when a verdict manifest checked them. An explicit
        # level is for QA builds (e.g. INTERNAL keeps everything, REVIEW passes unreviewed
        # columns through unchecked) and must never be used for a release.
        phenotype_level = disposition_level or (
            DispositionLevel.REVIEW if has_review_verdicts else DispositionLevel.RELEASE
        )
        if disposition_level is not None or keep_shifted_dates:
            _LOGGER.warning(
                "QA deidentify: disposition level %s, keep shifted dates=%s. Not a release build.",
                phenotype_level.value, keep_shifted_dates,
            )

        # Warm the field-map cache before parallel phenotype processing
        BIDSDataset._drop_columns_by_disposition(pd.DataFrame())

        phenotype_base_path = self.data_path.joinpath("phenotype")
        BIDSDataset._check_phenotype_tables_in_field_map(phenotype_base_path)
        if phenotype_base_path.exists():
            _LOGGER.info("Processing phenotype data for deidentification.")
            phenotype_output_path = outdir.joinpath("phenotype")
            phenotype_output_path.mkdir(parents=True, exist_ok=True)

            phenotype_files = sorted(phenotype_base_path.rglob("*.tsv"))

            def _process_one_phenotype(phenotype_filepath: Path) -> str:
                df_pheno, schema_name, header, phenotype_dict = BIDSDataset.load_phenotype_file(phenotype_filepath)

                # Apply per-value verdicts BEFORE ID remapping (verdicts use original IDs). An
                # explicit QA level shows the values unchecked, so verdicts are not applied.
                if has_review_verdicts and disposition_level is None:
                    df_pheno, review_dropped = BIDSDataset._apply_column_value_reviews(
                        df_pheno, BIDSDataset._get_review_column_names(schema_name=schema_name),
                        column_value_verdicts,
                    )
                    for col in review_dropped:
                        phenotype_dict.pop(col, None)

                df_pheno, phenotype_dict = BIDSDataset._deidentify_phenotype(
                    df_pheno, phenotype_dict,
                    participant_ids_to_exclude,
                    participant_ids_to_remap,
                    participant_session_id_to_remap,
                )

                # Drop internal columns (and review if no manifest)
                df_pheno, dropped_cols = BIDSDataset._drop_columns_by_disposition(
                    df_pheno,
                    level=phenotype_level,
                    schema_name=schema_name,
                    keep_date_shifted=keep_shifted_dates,
                )
                for col in dropped_cols:
                    phenotype_dict.pop(col, None)

                # Rows kept at ingest for their internal columns may now hold nothing publishable.
                df_pheno = BIDSDataset._drop_rows_emptied_by_deidentify(df_pheno, schema_name)
                if df_pheno.empty:
                    # Same as ingest: a table with no rows is not written.
                    _LOGGER.info("phenotype/%s: no rows left after deidentify; not written.", phenotype_filepath.stem)
                    return phenotype_filepath.stem

                if has_review_verdicts and disposition_level is None:
                    dropped_cols.extend(review_dropped)

                if dropped_cols:
                    _LOGGER.info(
                        "phenotype/%s: dropped %d columns by disposition: %s",
                        phenotype_filepath.stem, len(dropped_cols), ", ".join(sorted(dropped_cols)),
                    )

                phenotype_subdir = phenotype_output_path.joinpath(
                    phenotype_filepath.parent.relative_to(phenotype_base_path)
                )
                phenotype_subdir.mkdir(parents=True, exist_ok=True)
                df_pheno.to_csv(
                    phenotype_subdir.joinpath(f"{phenotype_filepath.stem}.tsv"),
                    sep="\t", index=False,
                )
                with open(phenotype_subdir.joinpath(f"{phenotype_filepath.stem}.json"), "w") as f:
                    json.dump({schema_name: {**header, "data_elements": phenotype_dict}}, f, indent=2)
                return phenotype_filepath.stem

            n_pheno_workers = max(1, min(max_workers, len(phenotype_files)))
            with ThreadPoolExecutor(max_workers=n_pheno_workers) as executor:
                list(tqdm(
                    executor.map(_process_one_phenotype, phenotype_files),
                    total=len(phenotype_files),
                    desc="Deidentifying phenotype",
                ))
            _LOGGER.info("Finished processing phenotype data (%d files).", len(phenotype_files))

        # --- quality metrics ---
        _LOGGER.info("Processing quality metrics for deidentification.")
        BIDSDataset._deidentify_quality_metrics(
            self.data_path,
            outdir,
            participant_ids_to_exclude,
            audio_filestems_to_remove,
            audio_tasks_to_include,
            participant_ids_to_remap,
            participant_session_id_to_remap,
        )
        _LOGGER.info("Finished processing quality metrics.")

        # session_id_mapping.json generation is disabled pending a decision on
        # whether to ship it (contains original session UUIDs) and whether the
        # mapping should be generated at ingest rather than deidentify.
        # _build_session_id_mapping is still used internally for the renaming.

        # --- template files ---
        for template_file in ["README.md", "CHANGES.md", "dataset_description.json"]:
            template_path = self.data_path.joinpath(template_file)
            if template_path.exists():
                shutil.copy(template_path, outdir)

        _LOGGER.info("Deidentification completed.")
        return BIDSDataset(outdir)

    @staticmethod
    def _collect_paths(
        data_path: Path,
        file_extension: str
    ) -> t.List[Path]:
        """Collect all file paths with the given extension from the dataset."""
        paths = get_paths(data_path, file_extension=file_extension)
        paths = [x["path"] for x in paths]
        
        # Sort audio paths for consistent processing
        paths = sorted(
            paths,
            # sort first by subject, then by task
            key=lambda x: (x.stem.split("_")[0], x.stem.split("_")[2]),
        )
        return paths

    @staticmethod
    def _apply_exclusion_list_to_filepaths(
        paths: t.List[Path],
        exclusion_list: t.List[str],
        exclusion_type: str = 'participant'
    ) -> t.List[Path]:
        """Remove filepaths based on overlap with a specified exclusion list."""
        n = len(paths)
        exclusion = set(exclusion_list)
        if len(exclusion) == 0:
            return paths
        def _canonical_recording_stem(stem: str) -> str:
            """Canonicalize a BIDS-like recording stem for matching.

            Many artifacts share a common "recording" identity, but append additional
            entities (e.g. "_features", "_run-1", "_rec-foo"). For exclusion matching,
            we treat the canonical identifier as everything up through the `task-...`
            entity, inclusive.

            The task entity is normalized so an exclusion entry written against an
            older tree ("task-Animal-fluency") still matches a tree built after
            redcap2bids began normalizing at ingest ("task-animal-fluency"). Only the
            task entity is normalized; the subject and session entities are compared
            verbatim, so this cannot make an entry match a different participant.
            """

            parts = stem.split("_")
            try:
                task_idx = next(i for i, part in enumerate(parts) if part.startswith("task-"))
            except StopIteration:
                return stem

            # Join up to and including the task entity (e.g. sub-..._ses-..._task-...)
            canonical = "_".join(parts[: task_idx + 1])
            return sanitize_task_entity_in_bids_stem(canonical)

        if exclusion_type == 'participant':
            paths = [
                x for x in paths
                if all(f"sub-{pid}" not in str(x) for pid in exclusion)
            ]
        elif exclusion_type == 'filename':
            # Normalize exclusions to recording filestems (strip extensions and known suffixes).
            canonical_exclusion = {
                _canonical_recording_stem(Path(excl).stem) for excl in exclusion
            }
            # One canonical stem per path, reused by both the unmatched-report and the
            # filter below; recomputing it per path twice is pure waste on a full tree.
            canonical_paths = [(a, _canonical_recording_stem(a.stem)) for a in paths]
            unmatched = canonical_exclusion - {stem for _, stem in canonical_paths}
            if unmatched:
                # An exclusion list that matches nothing is indistinguishable from one that
                # was applied, so say so rather than reporting a silent "removed 0 records".
                # Stale identifiers are the expected cause: participant/session ids change
                # between collection platforms, and a list built for one tree does not
                # transfer to another.
                _LOGGER.warning(
                    "%d of %d filename exclusions matched no file; those recordings were not "
                    "removed because nothing in this tree carries their identity. Examples: %s",
                    len(unmatched), len(canonical_exclusion), sorted(unmatched)[:5],
                )
            paths = [a for a, stem in canonical_paths if stem not in canonical_exclusion]
        elif exclusion_type == 'filestem_contains':
            paths = [
                a for a in paths
                if all(
                    normalize_task_label(excl) not in normalize_task_label(a.stem)
                    for excl in exclusion
                )
            ]
            # for better logging, add the list of exclusions to the exclusion type
            exclusion_type += f" ({', '.join(exclusion)})"
        else:
            raise ValueError(f"Unknown exclusion_type: {exclusion_type}")
        if len(paths) < n:
            _LOGGER.info(
                f"Removed {n - len(paths)} records due to exclusion: {exclusion_type}."
            )
        return paths

    @staticmethod
    def _grab_filepaths_list_to_include(
        paths: t.List[Path],
        inclusion_list: t.List[str],
    ) -> t.List[Path]:
        """Grabs filepaths based on overlap with a specified inclusion list."""
        n = len(paths)
        inclusion = set(inclusion_list)
        if len(inclusion) == 0:
            return []

        new_paths = []
        for file in paths:
            for incl in inclusion_list:
                if normalize_task_label(incl) in normalize_task_label(file.stem):
                    new_paths.append(file)
                    break

        if len(new_paths) < n:
            _LOGGER.info(
                f"Removed {n - len(new_paths)} records due to not being part of inclusion list: {inclusion_list}."
            )
        return new_paths

    @staticmethod
    def _expand_filestems_for_deidentification(
        filestems: t.Iterable[str],
        *,
        participant_ids_to_remap: t.Mapping[str, str],
        participant_session_id_to_remap: t.Mapping[str, str],
    ) -> t.List[str]:
        """Expand configured recording filestems to match both original and remapped IDs.

        Users may provide filestems referencing either:
        - the source dataset naming (e.g. sub-001js_ses-<uuid>_task-foo)
        - the deidentified naming (e.g. sub-512342_ses-<mapped>_task-foo)

        This returns a de-duplicated list containing both forms.
        """

        # Build reverse maps so we can support both directions when possible.
        reverse_participant_map = {v: k for k, v in participant_ids_to_remap.items()}
        reverse_session_map = {v: k for k, v in participant_session_id_to_remap.items()}

        expanded: t.Set[str] = set()
        pattern = re.compile(r"^sub-(?P<sub>[^_]+)(?:_ses-(?P<ses>[^_]+))?(?P<rest>.*)$")

        for raw in filestems:
            stem = Path(raw).stem  # tolerate accidental extensions
            expanded.add(stem)

            m = pattern.match(stem)
            if not m:
                continue

            sub = m.group("sub")
            ses = m.group("ses")
            rest = m.group("rest")

            # Forward-map into deidentified space
            sub_fwd = participant_ids_to_remap.get(sub, sub)
            ses_fwd = participant_session_id_to_remap.get(ses, ses) if ses is not None else None
            if ses_fwd is None:
                expanded.add(f"sub-{sub_fwd}{rest}")
            else:
                expanded.add(f"sub-{sub_fwd}_ses-{ses_fwd}{rest}")

            # Reverse-map back into original space (best-effort)
            sub_rev = reverse_participant_map.get(sub, sub)
            ses_rev = reverse_session_map.get(ses, ses) if ses is not None else None
            if ses_rev is None:
                expanded.add(f"sub-{sub_rev}{rest}")
            else:
                expanded.add(f"sub-{sub_rev}_ses-{ses_rev}{rest}")

        return sorted(expanded)

    @staticmethod
    def _extract_participant_id_from_path(path: t.Union[str, Path]) -> str:
        """Extract participant ID from the path, preferring directory parts.
        Falls back to regex on filestem if needed."""
        # Prefer directory parts
        path = Path(path)
        for part in Path(path).parts:
            if part.startswith("sub-"):
                return part[4:]

        # Fallback to filestem regex
        m = re.search(r"sub-([A-Za-z0-9\-]+)", path.stem)
        if m:
            return m.group(1)

        raise ValueError(f"Could not extract participant ID from path: {path}")

    @staticmethod
    def _extract_session_id_from_path(path: t.Union[str, Path]) -> str:
        """Extract session ID from the path, preferring directory parts.
        Falls back to regex on filestem using ses-(.+?)_ if needed."""
        # Prefer directory parts
        path = Path(path)
        for part in path.parts:
            if part.startswith("ses-"):
                return part[4:]

        # Fallback to filestem regex: ses-(.+?)_
        m = re.search(r"ses-(.+?)_", path.stem)
        if m:
            return m.group(1)

        raise ValueError(f"Could not extract session ID from path: {path}")

    @staticmethod
    def _extract_task_name_from_path(path: t.Union[str, Path]) -> str:
        """Extract the task name from the stem of the path.
        
        Tasks are optional components of filenames. They must follow the `task-<label>` pattern."""
        path = Path(path)
        m = re.search(r"task-(.+?)(_|$)", path.stem)
        if m:
            return m.group(1)

        raise ValueError(f"Could not extract task name from path: {path}")

    @staticmethod
    def _deidentify_participant_files(
        participant_dir: Path,
        outdir: Path,
        participant_ids_to_remap: t.Dict[str, str],
        participant_session_id_to_remap: t.Dict[str, str],
        audio_filestems_to_remove: t.List[str],
        audio_tasks_to_include: t.List[str],
        skip_audio: bool = False,
        skip_audio_features: bool = False,
        _normalized_include_tasks: t.Optional[TaskMatcher] = None,
        _canonical_exclusions: t.Optional[t.Set[str]] = None,
        disposition_level: DispositionLevel = DispositionLevel.RELEASE,
        keep_shifted_dates: bool = False,
    ) -> bool:
        """Process one participant directory for deidentification.

        Returns True if at least one audio file was written, False otherwise.
        When False, any partially-created output directory is cleaned up.
        """
        pid = participant_dir.name[4:]
        new_pid = participant_ids_to_remap.get(pid, pid)

        if _normalized_include_tasks is not None:
            normalized_include_tasks = _normalized_include_tasks
        else:
            normalized_include_tasks = TaskMatcher(audio_tasks_to_include)
        if _canonical_exclusions is not None:
            canonical_exclusions = _canonical_exclusions
        else:
            canonical_exclusions = {
                sanitize_task_entity_in_bids_stem(Path(s).stem)
                for s in audio_filestems_to_remove
            }

        n_audio_written = 0
        n_features_written = 0
        n_skipped = 0

        # --- Audio files and their sidecars ---
        # With skip_audio the recordings are walked through their sidecars instead, so a
        # metadata-only tree (redcap2bids --skip-audio-copy) still gets deidentified sidecars
        # and sessions.tsv, with no audio copied.
        if skip_audio:
            suffix = "_recording-metadata.json"
            recordings = [
                p.with_name(p.name[: -len(suffix)] + ".wav")
                for p in sorted(participant_dir.rglob(f"*{suffix}"))
            ]
        else:
            recordings = sorted(participant_dir.rglob("*.wav"))
        for wav_path in recordings:
            # Audio-check safety net
            task_match = re.search(r"task-(.+?)(_|$)", wav_path.stem)
            if task_match and is_audio_check(task_match.group(1)):
                n_skipped += 1
                continue

            # Filestem exclusion
            canonical_stem = sanitize_task_entity_in_bids_stem(wav_path.stem)
            parts = canonical_stem.split("_")
            try:
                task_idx = next(i for i, p in enumerate(parts) if p.startswith("task-"))
                recording_stem = "_".join(parts[:task_idx + 1])
            except StopIteration:
                recording_stem = canonical_stem
            if recording_stem in canonical_exclusions:
                n_skipped += 1
                _LOGGER.debug("Skipping excluded filestem: %s", wav_path.name)
                continue

            # Task inclusion (empty list = publish nothing, matching old behavior)
            if task_match:
                if not normalized_include_tasks:
                    n_skipped += 1
                    continue
                task_label = normalize_task_label(task_match.group(1))
                if task_label not in normalized_include_tasks:
                    n_skipped += 1
                    continue

            # Remap IDs in path
            session_id_raw = BIDSDataset._extract_session_id_from_path(wav_path)
            new_session_id = remap_id(session_id_raw, participant_session_id_to_remap, id_type="session")

            sanitized_stem = sanitize_task_entity_in_bids_stem(wav_path.stem)
            stem_ending = "-".join(sanitized_stem.split("_")[2:])

            # Sidecar must exist before we copy the wav (match old behavior)
            json_path = wav_path.parent / f"{wav_path.stem}_recording-metadata.json"
            if not json_path.exists():
                _LOGGER.warning("Missing sidecar for %s; skipping.", wav_path.name)
                n_skipped += 1
                continue

            out_wav = outdir / f"sub-{new_pid}" / f"ses-{new_session_id}" / "audio" / (
                f"sub-{new_pid}_ses-{new_session_id}_{stem_ending}.wav"
            )
            metadata = json.loads(json_path.read_text())
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            update_metadata_record_and_session_id(
                metadata, participant_ids_to_remap, participant_session_id_to_remap
            )
            out_json = out_wav.with_suffix(".json")
            with open(out_json, "w") as f:
                json.dump(metadata, f, indent=2)
            if not skip_audio:
                shutil.copyfile(wav_path, out_wav)
            n_audio_written += 1

        # --- Feature files ---
        if not skip_audio_features:
            for feat_path in sorted(participant_dir.rglob("*.pt")):
                # Audio-check safety net
                task_match = re.search(r"task-(.+?)(_|$)", feat_path.stem)
                if task_match and is_audio_check(task_match.group(1)):
                    n_skipped += 1
                    continue

                # Filestem exclusion (strip _features suffix first)
                base_stem = feat_path.stem.replace("_features", "")
                canonical_stem = sanitize_task_entity_in_bids_stem(base_stem)
                parts = canonical_stem.split("_")
                try:
                    task_idx = next(i for i, p in enumerate(parts) if p.startswith("task-"))
                    recording_stem = "_".join(parts[:task_idx + 1])
                except StopIteration:
                    recording_stem = canonical_stem
                if recording_stem in canonical_exclusions:
                    n_skipped += 1
                    continue

                # Remap IDs in path
                session_id_raw = BIDSDataset._extract_session_id_from_path(feat_path)
                new_session_id = remap_id(session_id_raw, participant_session_id_to_remap, id_type="session")

                sanitized_base = sanitize_task_entity_in_bids_stem(base_stem)
                stem_ending = "-".join(sanitized_base.split("_")[2:]) + "_features"

                out_feat = outdir / f"sub-{new_pid}" / f"ses-{new_session_id}" / "audio" / (
                    f"sub-{new_pid}_ses-{new_session_id}_{stem_ending}{feat_path.suffix}"
                )
                out_feat.parent.mkdir(parents=True, exist_ok=True)

                task_match_feat = re.search(r"task-(.+?)(_|$)", feat_path.stem)
                task_is_included = (
                    task_match_feat
                    and normalize_task_label(task_match_feat.group(1)) in normalized_include_tasks
                )
                if task_is_included:
                    shutil.copyfile(feat_path, out_feat)
                else:
                    features = torch.load(feat_path, weights_only=False, map_location=torch.device("cpu"))
                    _remove_sensitive_features_from_feature_payload(features)
                    torch.save(features, out_feat)
                n_features_written += 1

        # --- Sessions metadata carry-forward ---
        out_participant = outdir / f"sub-{new_pid}"
        if out_participant.exists():
            sessions_path = participant_dir / "sessions.tsv"
            if not sessions_path.exists():
                sessions_path = participant_dir / f"sub-{pid}_sessions.tsv"
            if sessions_path.exists():
                df_ses = pd.read_csv(sessions_path, sep="\t", dtype=str)
                # Drop columns by disposition
                df_ses, dropped_cols = BIDSDataset._drop_columns_by_disposition(
                    df_ses, level=disposition_level, keep_date_shifted=keep_shifted_dates,
                    schema_name=BIDSDataset._SESSIONS_SCHEMA,
                )
                if dropped_cols:
                    _LOGGER.debug("Participant %s sessions.tsv: dropped %d columns (%s).",
                                  pid, len(dropped_cols), ", ".join(dropped_cols))
                # Remap record_id → participant_id
                if "record_id" in df_ses.columns:
                    df_ses = df_ses.rename(columns={"record_id": "participant_id"})
                if "participant_id" in df_ses.columns:
                    remap_partial = partial(remap_id, id_mapping=participant_ids_to_remap)
                    df_ses["participant_id"] = BIDSDataset._map_series(df_ses["participant_id"], remap_partial)
                # Remap session_id
                if "session_id" in df_ses.columns:
                    remap_ses = partial(remap_id, id_mapping=participant_session_id_to_remap, id_type="session")
                    df_ses["session_id"] = BIDSDataset._map_series(df_ses["session_id"], remap_ses)
                    # Keep only sessions that were written: every recording of the others was
                    # filtered out (task list, filestem or recording-ID exclusions).
                    written = {d.name[len("ses-"):] for d in out_participant.glob("ses-*") if d.is_dir()}
                    kept = df_ses["session_id"].astype(str).isin(written)
                    if not kept.all():
                        _LOGGER.info(
                            "Participant %s sessions.tsv: dropped %d session(s) with no output: %s",
                            pid, int((~kept).sum()), ", ".join(df_ses.loc[~kept, "session_id"].astype(str)),
                        )
                    df_ses = df_ses.loc[kept]
                # Write with BIDS-compliant naming
                out_sessions = out_participant / f"sub-{new_pid}_sessions.tsv"
                df_ses.to_csv(out_sessions, sep="\t", index=False)
            else:
                _LOGGER.warning("No sessions.tsv found for participant %s; skipping carry-forward.", pid)

        # --- Cleanup if no output ---
        out_participant = outdir / f"sub-{new_pid}"
        # Feature files count: a participant whose recordings are all from tasks whose audio is
        # not released still has (stripped) features to publish.
        has_output = n_audio_written > 0 or n_features_written > 0 or (skip_audio and out_participant.exists())
        if not has_output:
            if out_participant.exists():
                shutil.rmtree(out_participant)
            _LOGGER.info(
                "Participant %s: no audio or feature files after filtering (%d skipped). Removed output dir.",
                pid, n_skipped,
            )
            return False

        _LOGGER.debug(
            "Participant %s: %d audio, %d features, %d skipped.",
            pid, n_audio_written, n_features_written, n_skipped,
        )
        return True

    @staticmethod
    def _deidentify_quality_metrics(
        data_path: Path,
        outdir: Path,
        exclude_participant_ids: t.List[str] = [],
        exclude_audio_filestems: t.List[str] = [],
        audio_tasks_to_include_list: t.List[str] = [],
        participant_ids_to_remap: t.Dict[str, str] = {},
        participant_session_id_to_remap: t.Dict[str, str] = {},
    ) -> None:
        """
        Deidentify and copy audio quality metrics to the output directory.

        If no quality metrics file exists this method is a no-op, since the file
        is optional.  When present, the same participant exclusion, task filtering,
        and ID remapping that is applied to audio files is applied here so the
        released metrics are consistent with the rest of the deidentified dataset.

        Args:
            data_path: Path to the source BIDS dataset
            outdir: Output directory for the deidentified dataset
            exclude_participant_ids: list of participant IDs to exclude
            exclude_audio_filestems: list of audio file stems to exclude
            audio_tasks_to_include_list: list of audio tasks to include
            participant_ids_to_remap: mapping from old to new participant IDs
            participant_session_id_to_remap: mapping from old to new session IDs
        """
        quality_metrics_path = data_path / "audio_quality_metrics.tsv"
        if not quality_metrics_path.exists():
            _LOGGER.info("No audio_quality_metrics.tsv found; skipping quality metrics deidentification.")
            return

        df = pd.read_csv(quality_metrics_path, sep="\t", dtype=str)

        # Remove excluded participants
        if exclude_participant_ids and "participant_id" in df.columns:
            idx = df["participant_id"].isin(exclude_participant_ids)
            if idx.any():
                _LOGGER.info(f"Removing {idx.sum()} quality metric rows for excluded participants.")
                df = df.loc[~idx]

        # Remove audio-check task rows. Uses the same predicate as every other audio-check
        # filter; on a tree built with drop_audio_check=True there is nothing left to remove,
        # but older trees and internal-review builds still pass through here.
        if "task_name" in df.columns:
            df = df.loc[~df["task_name"].apply(is_audio_check)]

        # Doesn't filter to only included audio tasks as these are similar to non-identifiable features
        # # Filter to only included audio tasks
        # if audio_tasks_to_include_list and "task_name" in df.columns:
        #     normalized_tasks = {normalize_task_label(task) for task in audio_tasks_to_include_list}
        #     df = df.loc[df["task_name"].apply(lambda task: normalize_task_label(task) in normalized_tasks)]

        # Remove rows for excluded audio file stems
        if exclude_audio_filestems and {"participant_id", "session_id", "task_name"}.issubset(df.columns):
            def row_is_excluded(row):
                stem = f"sub-{row['participant_id']}_ses-{row['session_id']}_task-{row['task_name']}"
                return any(normalize_task_label(excl) in normalize_task_label(stem) for excl in exclude_audio_filestems)
            df = df.loc[~df.apply(row_is_excluded, axis=1)]

        # Remap participant IDs
        if participant_ids_to_remap and "participant_id" in df.columns:
            remap_partial = partial(remap_id, id_mapping=participant_ids_to_remap)
            df["participant_id"] = BIDSDataset._map_series(df["participant_id"], remap_partial)

        # Remap session IDs
        if participant_session_id_to_remap and "session_id" in df.columns:
            remap_partial = partial(remap_id, id_mapping=participant_session_id_to_remap, id_type="session")
            df["session_id"] = BIDSDataset._map_series(df["session_id"], remap_partial)

        df.to_csv(outdir / "audio_quality_metrics.tsv", sep="\t", index=False)

        quality_json_path = data_path / "audio_quality_metrics.json"
        if quality_json_path.exists():
            shutil.copy(quality_json_path, outdir)
        else:
            _LOGGER.warning(
                "audio_quality_metrics.json not found in BIDS dataset; "
                "copying schema from package resources."
            )
            qc_json_resource = resources.files("b2aiprep").joinpath(
                "prepare", "resources", "audio_quality_metrics.json"
            )
            with qc_json_resource.open() as src, open(outdir / "audio_quality_metrics.json", "w") as dst:
                dst.write(src.read())


class VBAIDataset(BIDSDataset):
    """Extension of BIDS format dataset implementing helper functions for data specific
    to the Bridge2AI Voice as a Biomarker of Health project.
    """

    def __init__(self, data_path: t.Union[Path, str, os.PathLike]):
        super().__init__(data_path)

    def _merge_columns_with_underscores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merges columns which are exported by RedCap in a one-hot encoding manner, i.e.
        they correspond to a single category but are split into multiple yes/no columns.

        Modifies the dataframe in place.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to modify.

        Returns
        -------
        pd.DataFrame
            The modified dataframe.
        """

        # identify all the columns which end with __1, __2, etc.
        # extract the prefix for these columns only
        columns_with_underscores = sorted(
            list(
                set(
                    [
                        col[: col.rindex("__") - 1]
                        for col in df.columns
                        if re.search("__[0-9]+$", col) is not None
                    ]
                )
            )
        )

        # iterate through each prefix and merge together data into this prefix
        for col in columns_with_underscores:
            columns_to_merge = df.filter(like=f"{col}__").columns
            df[col] = df[columns_to_merge].apply(
                lambda x: next((i for i in x if i is not None), None), axis=1
            )
            df.drop(columns=columns_to_merge, inplace=True)
        return df

    def load_and_pivot_questionnaire(self, questionnaire_name: str) -> pd.DataFrame:
        """
        Loads all data for a questionnaire and pivots on the appropriate identifier column.

        Parameters
        ----------
        questionnaire_name : str
            The name of the questionnaire to load.

        Returns
        -------
        pd.DataFrame
            A "wide" format dataframe with questionnaire data. Returns empty DataFrame 
            if columns are not found in participants.tsv.
        """
        participants_file = self.data_path / "participants.tsv"
        if not participants_file.exists():
            _LOGGER.warning(f"participants.tsv file not found at {participants_file}")
            return pd.DataFrame()

        participants_df = pd.read_csv(participants_file, sep="\t")
        # Load questionnaire columns from instrument_columns resources
        instrument_columns_path = files("b2aiprep").joinpath("prepare").joinpath("resources").joinpath("instrument_columns")
        questionnaire_file = instrument_columns_path.joinpath(f"{questionnaire_name}.json")
        
        if not questionnaire_file.exists():
            _LOGGER.warning(f"Questionnaire JSON file not found: {questionnaire_name}.json")
            return pd.DataFrame()
            
        questionnaire_columns = json.loads(questionnaire_file.read_text())

        # Filter to only include columns that exist in participants.tsv
        available_columns = [col for col in questionnaire_columns if col in participants_df.columns]
        
        if not available_columns:
            _LOGGER.warning(f"No columns from '{questionnaire_name}' questionnaire found in participants.tsv")
            return pd.DataFrame()

        # Select the available columns
        questionnaire_df = participants_df[available_columns].copy()
        
        return questionnaire_df

    def load_participants(self) -> pd.DataFrame:
        """
        Loads the participants.tsv file and returns a dataframe with participant data.

        Returns
        -------
        pd.DataFrame
            A dataframe with participant data.
        """
        participants_file = self.data_path / "participants.tsv"
        if not participants_file.exists():
            _LOGGER.warning("participants.tsv file not found")
            return pd.DataFrame()

        try:
            df = pd.read_csv(participants_file, sep='\t')
            # Rename record_id to participant_id for consistency
            if 'record_id' in df.columns:
                df.rename(columns={'record_id': 'participant_id'}, inplace=True)
            return df
        except Exception as e:
            _LOGGER.error(f"Error loading participants data: {e}")
            return pd.DataFrame()

    def _load_session_schema_from_participants_tsv(self) -> pd.DataFrame:
        """
        Load session schema data from the participants.tsv file.
        
        Returns
        -------
        pd.DataFrame
            Session schema data with participant_id and session information.
        """
        participants_file = self.data_path / "participants.tsv"
        if not participants_file.exists():
            _LOGGER.warning("participants.tsv file not found")
            return pd.DataFrame()
        
        try:
            # Read the participants file
            df = pd.read_csv(participants_file, sep='\t')
            
            # Select session-related columns
            session_columns = [
                'participant_id', 'session_id', 'session_status', 
                'session_is_control_participant', 'session_duration'
            ]
            
            # Only include columns that exist in the file
            available_columns = [col for col in session_columns if col in df.columns]
            
            if not available_columns:
                _LOGGER.warning("No session-related columns found in participants.tsv")
                return pd.DataFrame()
            
            # Filter to rows that have session data (session_id is not null)
            session_df = df[available_columns].copy()
            if 'session_id' in session_df.columns:
                session_df = session_df.dropna(subset=['session_id'])
            
            return session_df
            
        except Exception as e:
            _LOGGER.error(f"Error loading session data from participants.tsv: {e}")
            return pd.DataFrame()

    def _load_recording_and_acoustic_task_df(self) -> pd.DataFrame:
        """Loads recording schema dataframe with the acoustic task name.

        Returns
        -------
        pd.DataFrame
            The recordings dataframe with the additional "acoustic_task_name" column.
        """
        recording_df = self.load_and_pivot_questionnaire("recordingschema")
        task_df = self.load_and_pivot_questionnaire("acoustictaskschema")

        recording_df = recording_df.merge(
            task_df[["acoustic_task_id", "acoustic_task_name"]],
            how="inner",
            left_on="recording_acoustic_task_id",
            right_on="acoustic_task_id",
        )
        return recording_df

    def load_recording(self, recording_id: str) -> Audio:
        """Checks for and loads in the given recording_id.

        Parameters
        ----------
        recording_id : str
            The recording identifier.

        Returns
        -------
        Audio
            The loaded audio.
        """
        # verify the recording_id is in the recording_df
        recording_df = self._load_recording_and_acoustic_task_df()
        idx = recording_df["recording_id"] == recording_id
        if not idx.any():
            raise ValueError(
                f"Recording ID '{recording_id}' not found in \
                             recordings dataframe."
            )

        row = recording_df.loc[idx].iloc[0]

        # Use participant_id or fall back to record_id for backward compatibility
        subject_id = row.get("participant_id", row.get("record_id"))
        session_id = row["recording_session_id"]
        task = row["acoustic_task_name"].replace(" ", "-")
        name = row["recording_name"].replace(" ", "-")

        audio_file = self.data_path.joinpath(
            f"sub-{subject_id}",
            f"ses-{session_id}",
            "audio",
            f"sub-{subject_id}_ses-{session_id}_{task}_rec-{name}.wav",
        )
        return Audio(filepath=str(audio_file))

    def load_recordings(self) -> t.List[Audio]:
        """Loads all audio recordings in the dataset.

        Returns
        -------
        List[Audio]
            The loaded audio recordings.
        """
        recording_df = self._load_recording_and_acoustic_task_df()
        audio_data = []
        missed_files = []
        for _, row in tqdm(
            recording_df.iterrows(), total=recording_df.shape[0], desc="Loading audio"
        ):
            # Use participant_id or fall back to record_id for backward compatibility
            subject_id = row.get("participant_id", row.get("record_id"))
            session_id = row["recording_session_id"]
            task = row["acoustic_task_name"].replace(" ", "-")
            name = row["recording_name"].replace(" ", "-")
            audio_file = self.data_path.joinpath(
                f"sub-{subject_id}",
                f"ses-{session_id}",
                "audio",
                f"sub-{subject_id}_ses-{session_id}_{task}_rec-{name}.wav",
            )
            try:
                audio_data.append(Audio(filepath=str(audio_file)))
            except (LibsndfileError, FileNotFoundError):
                # assuming lbsnd file error is a file not found, usually it is
                missed_files.append(audio_file)
                continue

        if len(missed_files) > 0:
            _LOGGER.warning(
                f"Could not find {len(missed_files)} / {recording_df.shape[0]} audio files."
            )

        return audio_data

    def load_spectrograms(self) -> t.List[np.array]:
        """Loads all audio recordings in the dataset."""
        recording_df = self._load_recording_and_acoustic_task_df()
        audio_data = []
        missed_files = []
        for _, row in tqdm(
            recording_df.iterrows(), total=recording_df.shape[0], desc="Loading audio"
        ):
            # Use participant_id or fall back to record_id for backward compatibility
            subject_id = row.get("participant_id", row.get("record_id"))
            session_id = row["recording_session_id"]
            task = row["acoustic_task_name"].replace(" ", "-")
            name = row["recording_name"].replace(" ", "-")
            audio_file = self.data_path.joinpath(
                f"sub-{subject_id}",
                f"ses-{session_id}",
                "audio",
                f"sub-{subject_id}_ses-{session_id}_{task}_rec-{name}.pt",
            )
            try:
                device = 'cpu' # not checking for cuda because optimization would be minimal if any
                features = torch.load(str(audio_file), weights_only=False, map_location=torch.device(device))
                audio_data.append(features["specgram"])
            except FileNotFoundError:
                # assuming lbsnd file error is a file not found, usually it is
                missed_files.append(audio_file)
                continue

        if len(missed_files) > 0:
            _LOGGER.warning(
                f"Could not find {len(missed_files)} / {recording_df.shape[0]} feature files."
            )

        return audio_data

    def validate_audio_files_exist(self) -> bool:
        """
        Validates that the audio recordings for all sessions are present.

        Parameters
        ----------
        subject_id : str
            The subject identifier.
        session_id : str
            The session identifier.

        Returns
        -------
        bool
            Whether the audio files are present.
        """
        missing_audio_files = []
        # iterate over all of the audio tasks in beh subfolder
        subjects = self.find_subjects()
        for subject in subjects:
            sessions = self.find_sessions(subject)
            for session in sessions:
                tasks = self.find_tasks(subject, session)
                for task_name, task_filename in tasks.items():
                    # check if the audio file is present
                    if "_rec-" not in task_name:
                        continue
                    if not task_name.endswith("_recordingschema.json"):
                        continue

                    suffix_len = len("_recordingschema")
                    audio_filename = (
                        Path(task_filename.parent).joinpath("..", "audio"),
                        f"{task_filename.stem[:suffix_len]}",
                    )
                    if not audio_filename.exists():
                        missing_audio_files.append(audio_filename)

        _LOGGER.debug(f"Missing audio files: {missing_audio_files}")
        return len(missing_audio_files) == 0
