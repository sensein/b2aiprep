import argparse
import sys
from importlib.resources import files

import altair as alt
import streamlit as st

from b2aiprep.prepare.dataset import VBAIDataset

st.set_page_config(page_title="Subject questionnaires", page_icon="📊")

@st.cache_data
def get_available_questionnaires():
    """Get list of available questionnaire names from instrument_columns directory."""
    try:
        instrument_columns_path = files("b2aiprep").joinpath("prepare").joinpath("resources").joinpath("instrument_columns")
        questionnaire_files = []
        for file_path in instrument_columns_path.iterdir():
            if file_path.name.endswith('.json'):
                questionnaire_files.append(file_path.name[:-5])  # Remove .json extension
        return sorted(questionnaire_files)
    except Exception as e:
        st.error(f"Error loading questionnaire list: {e}")
        return []

@st.cache_data
def get_questionnaire_dataframe(_dataset: VBAIDataset, questionnaire_name: str):
    return _dataset.load_and_pivot_questionnaire(questionnaire_name)


def parse_args(args):
    parser = argparse.ArgumentParser("Audio processing for BIDS data.")
    parser.add_argument("bids_dir", help="Folder with the BIDS data")
    return parser.parse_args(args)


args = parse_args(sys.argv[1:])
st.session_state.bids_dir = args.bids_dir

dataset = VBAIDataset(st.session_state.bids_dir)


# st.markdown("# Disease prevalence")

# for questionnaire_name in GENERAL_QUESTIONNAIRES[4:]:
#     df = load_questionnaire_into_dataframe(dataset, questionnaire_name)
#     st.write(f"* {df.shape[0]} individuals have responses for
# the {questionnaire_name} questionnaire.")

st.markdown("# Subject questionnaires")

# Get available questionnaires
available_questionnaires = get_available_questionnaires()
if not available_questionnaires:
    st.error("No questionnaires found in instrument_columns directory.")
    st.stop()

questionnaire_name = st.selectbox(
    "Which questionnaire would you like to review?", available_questionnaires
)
df = get_questionnaire_dataframe(dataset, questionnaire_name)

# Check if dataframe is empty and stop if so
if df.empty:
    st.warning(f"No data found for questionnaire '{questionnaire_name}' in participants.tsv file.")
    st.stop()

session_id_column = [col for col in df.columns if col.endswith("session_id")]
if session_id_column:
    session_id_column = session_id_column[0]

    # filter out rows w/ null session id as they have null data for the rest too
    idxNullSession = df[session_id_column].isnull()
    idxNullRow = df.isnull().all(axis=1)

    idx = idxNullSession | idxNullRow
    st.info(f"Filtering out {idx.sum()} rows with null session_id or all empty values.")
    df = df[~idx].copy()
else:
    st.warning("No session_id column found in the questionnaire data. All rows retained.")
# there are always a lot of binary questions,
# so we have a bar chart which includes all checked/unchecked or yes/no questions

binary_questions = []
for column in df.columns:
    unique_values = set(df[column].dropna().unique())  # Drop NaN values
    if len(unique_values) == 0:
        continue
    # verify that the values are one of "Yes", "No", "Checked", or "Unchecked"
    if unique_values.issubset({"Yes", "No", "Checked", "Unchecked"}):
        binary_questions.append(column)

st.write(
    f"""
    * {df.shape[0]} individuals have responses for the {questionnaire_name} questionnaire.
    * {df.shape[1]} questions in this questionnaire.
    * {len(binary_questions)} binary questions in this questionnaire.
    """
)

# create a single bar char with these questions
if len(binary_questions) > 0:
    st.markdown("# Binary questions")
    grp = df[binary_questions].copy().map(lambda x: 1 if x in ("Checked", "Yes") else 0)
    grp = grp.mean().reset_index(name="count")
    # convert to percentage
    grp["percentage"] = (grp["count"] * 100).round(1)
    grp.index.name = "linkId"

    # create horizontal bar chart
    chart = (
        alt.Chart(grp)
        .mark_bar()
        .encode(x="percentage:Q", y=alt.Y("linkId:N", sort="-x"), tooltip=["count", "percentage"])
        .properties(width=600, height=400)
    )

    st.altair_chart(chart, use_container_width=True)


st.markdown("# Bar plot")
questions = df.columns.tolist()
if questions:
    question_name = st.selectbox(
        "Which question would you like to display?",
        questions,
        # skip the typically null session_id
        index=1 if len(questions) > 1 else None
    )
    
    grp = df.groupby(question_name).size().reset_index(name="count")
    st.bar_chart(grp, x=question_name, y="count")
else:
    st.info("No questions available for bar plot.")

st.markdown("# Review the data")
st.write(df)
