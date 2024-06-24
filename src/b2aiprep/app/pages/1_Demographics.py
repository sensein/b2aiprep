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
    dataset = VBAIDataset(st.session_state.bids_dir)
    return dataset

schema_name = 'qgenericdemographicsschema'
dataset = get_bids_data()
df = dataset.load_and_pivot_questionnaire(schema_name)

# st.markdown("## Age Distribution")
# st.write(df['age'].describe())


st.markdown("## Gender Identity")
gender_counts = df['gender_identity'].value_counts()
st.bar_chart(gender_counts)


st.markdown("## Sexual Orientation")
orientation_counts = df['sexual_orientation'].value_counts()
st.bar_chart(orientation_counts)


st.markdown("## Race")
race_columns = [col for col in df.columns if 'race___' in col]
race_counts = df[race_columns].sum()
st.bar_chart(race_counts)

st.markdown("## Ethnicity")
ethnicity_counts = df['ethnicity'].value_counts()
st.bar_chart(ethnicity_counts)

st.markdown("## Marital Status")
marital_status_columns = [col for col in df.columns if 'marital_status___' in col]
marital_status_counts = df[marital_status_columns].sum()
st.bar_chart(marital_status_counts)


st.markdown("## Employment Status")
employ_status_columns = [col for col in df.columns if 'employ_status___' in col]
employ_status_counts = df[employ_status_columns].sum()
st.bar_chart(employ_status_counts)

# we need to do some harmonization of the USA / CA salaries
st.markdown("## Household Income")
income = df[['household_income_usa', 'household_income_ca']].copy()
income_cols = list(income.columns)
# get the upper range of their income, or if they only have one, the upper limit
for col in income_cols:
    # need to extract the *last* instance of this pattern
    income[f'{col}_lower'] = income[col].str.extract(r'\$(\d+,\d+)\s*$')
    income[f'{col}_lower'] = income[f'{col}_lower'].str.replace(',', '')
    income[f'{col}_lower'] = pd.to_numeric(income[f'{col}_lower'], errors='coerce')

    # now create an integer which is higher if the value is higher
    income[f'{col}_seq_num'] = income[f'{col}_lower'].rank(ascending=True, method='dense')
    income[f'{col}_seq_num'] = income[f'{col}_seq_num'].fillna(-1).astype(int)
    

    idxNan = income[col].str.contains('Prefer not to answer').fillna(False)
    income.loc[idxNan, f'{col}_seq_num'] = 0

income['seq_num'] = income[['household_income_usa_seq_num', 'household_income_ca_seq_num']].max(axis=1)
# get our look-up dict for each
income_lookups = {}
for col in income_cols:
    income_lookups[col] = income[
        [col, f'{col}_seq_num']
    ].drop_duplicates().set_index(f'{col}_seq_num').to_dict()[col]

income['country'] = 'Missing'
idx = income['household_income_usa'].notnull()
income.loc[idx, 'country'] = 'USA'
idx = income['household_income_ca'].notnull()
income.loc[idx, 'country'] = 'Canada'

income_grouped = pd.crosstab(income['seq_num'], income['country'])
# as it turns out, both countries have the same values for income brackets
# so we can just use one of the mapping tables
n_missing = (income['seq_num'] == -1).sum()
income_grouped.index = income_grouped.index.map(income_lookups[col])
income_grouped = income_grouped[['USA', 'Canada']]
income_grouped.index.name = 'Household Income (CAD or USD)'
# st.write(income_grouped)

# grouped barchart
income_grouped = income_grouped.reset_index()
income_grouped = income_grouped.melt(id_vars='Household Income (CAD or USD)', var_name='Country', value_name='Count')
chart = (
    alt.Chart(income_grouped)
    .mark_bar()
    .encode(
        x=alt.X('Household Income (CAD or USD):O', axis=alt.Axis(title='Income')),
        y=alt.Y('Count:Q', axis=alt.Axis(title='Count')),
        color='Country:N',
        tooltip=['Household Income (CAD or USD)', 'Count', 'Country']
    )
)
st.altair_chart(chart, use_container_width=True)
st.write(f"{n_missing} missing a household income.")


st.markdown("## Full dataframe")
st.write(df)
