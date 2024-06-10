import streamlit as st
import pandas as pd
import altair as alt

from b2aiprep.dataset import VBAIDataset

st.set_page_config(page_title="Demographics", page_icon="📈")

st.markdown("# Demographics")
st.sidebar.header("Demographics")
st.write(
    """This page overviews the demographics of the dataset."""
)

def get_bids_data():
    # TODO: allow user to specify input folder input
    dataset = VBAIDataset('output')
    return dataset

schema_name = 'qgenericdemographicsschema'
dataset = get_bids_data()
df = dataset.load_and_pivot_questionnaire(schema_name)
st.write(df)

st.markdown("# Subsets")
st.write(df.groupby('gender_identity').size().sort_values(ascending=False))

income = df[['household_income_usa', 'household_income_ca']].copy()
income['household_income'] = None
idx = income['household_income_usa'].notnull()
income.loc[idx, 'household_income'] = 'USD ' + income.loc[idx, 'household_income_usa']

idx = income['household_income_ca'].notnull()
income.loc[idx, 'household_income'] = 'CAD ' + income.loc[idx, 'household_income_ca']
st.write(income.groupby('household_income').size().sort_values(ascending=False))