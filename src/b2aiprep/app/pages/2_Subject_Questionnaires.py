import streamlit as st
import pandas as pd
import altair as alt

from b2aiprep.dataset import VBAIDataset
from b2aiprep.constants import GENERAL_QUESTIONNAIRES

st.set_page_config(page_title="Subject questionnaires", page_icon="📊")

@st.cache_data
def get_bids_data():
    # TODO: allow user to specify input folder input
    dataset = VBAIDataset('output')
    return dataset

dataset = get_bids_data()

# st.markdown("# Disease prevalence")

# for questionnaire_name in GENERAL_QUESTIONNAIRES[4:]:
#     df = load_questionnaire_into_dataframe(dataset, questionnaire_name)
#     st.write(f"* {df.shape[0]} individuals have responses for the {questionnaire_name} questionnaire.")

st.markdown("# Subject questionnaires")


questionnaire_name = st.selectbox(
    'Which questionnaire would you like to review?',
    GENERAL_QUESTIONNAIRES
)
df = dataset.load_and_pivot_questionnaire(questionnaire_name)

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
    grp = grp.mean().reset_index(name='count')
    # convert to percentage
    grp['percentage'] = (grp['count'] * 100).round(1)
    grp.index.name = 'linkId'
    
    # create bar chart in altair
    # add a tick for each bar to make sure the labels are on the plot for linkId
    chart = (
        alt.Chart(grp)
        .mark_bar()
        .encode(
            x=alt.X('linkId:O', axis=alt.Axis(title='Question', tickCount=grp.shape[0])),
            y=alt.Y('percentage:Q', axis=alt.Axis(title='Percentage')),
            tooltip=['linkId', 'percentage']
        )
    )

    st.altair_chart(chart, use_container_width=True)


st.markdown("# Bar plot")
questions = df.columns.tolist()
question_name = st.selectbox(
    'Which question would you like to display?',
    questions
)

grp = df.groupby(question_name).size().reset_index(name='count')
st.bar_chart(grp, x=question_name, y='count')

st.markdown("# Review the data")
st.write(df)