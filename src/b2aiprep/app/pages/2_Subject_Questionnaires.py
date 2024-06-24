import streamlit as st
import pandas as pd
import altair as alt

from b2aiprep.constants import GENERAL_QUESTIONNAIRES
from b2aiprep.dataset import VBAIDataset

st.set_page_config(page_title="Subject questionnaires", page_icon="📊")


@st.cache_data
def get_questionnaire_dataframe(_dataset: VBAIDataset, questionnaire_name: str):
    return _dataset.load_and_pivot_questionnaire(questionnaire_name)

dataset = VBAIDataset(st.session_state.bids_dir)

# st.markdown("# Disease prevalence")

# for questionnaire_name in GENERAL_QUESTIONNAIRES[4:]:
#     df = load_questionnaire_into_dataframe(dataset, questionnaire_name)
#     st.write(f"* {df.shape[0]} individuals have responses for the {questionnaire_name} questionnaire.")

st.markdown("# Subject questionnaires")

questionnaire_name = st.selectbox(
    'Which questionnaire would you like to review?',
    GENERAL_QUESTIONNAIRES
)
df = get_questionnaire_dataframe(dataset, questionnaire_name)

# there are always a lot of binary questions,
# so we have a bar chart which includes all checked/unchecked or yes/no questions

binary_questions = []
for column in df.columns:
    unique_values = set(df[column].unique())
    if len(unique_values) == 0:
        continue
    # verify that the values are one of "Yes", "No", "Checked", or "Unchecked"
    if unique_values.issubset({"Yes", "No", "Checked", "Unchecked"}):
        binary_questions.append(column)

c = df.columns[1]
st.write(
    f"""
    * {df.shape[0]} individuals have responses for the {questionnaire_name} questionnaire.
    * {df[c].notnull().sum()} individuals have a response for the first question ({c}).
    * {df.shape[1]} questions in this questionnaire.
    * {len(binary_questions)} binary questions in this questionnaire.
    """
)

# create a single bar char with these questions
if len(binary_questions) > 0:
    st.markdown("# Binary questions")
    grp = df[binary_questions].copy().map(lambda x: 1 if x in ("Checked", "Yes") else 0)
    grp = grp.mean().reset_index(name='count')
    # convert to percentage
    grp['percentage'] = (grp['count'] * 100).round(1)
    grp.index.name = 'linkId'
    
    # create horizontal bar chart
    chart = alt.Chart(grp).mark_bar().encode(
        x='percentage:Q',
        y=alt.Y('linkId:N', sort='-x'),
        tooltip=['count', 'percentage']
    ).properties(
        width=600,
        height=400
    )

    st.altair_chart(chart, use_container_width=True)


st.markdown("# Bar plot")
questions = df.columns.tolist()[1:]
question_name = st.selectbox(
    'Which question would you like to display?',
    questions
)

grp = df.groupby(question_name).size().reset_index(name='count')
st.bar_chart(grp, x=question_name, y='count')

st.markdown("# Review the data")
st.write(df)