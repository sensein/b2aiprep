import argparse
from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st

from b2aiprep.dashboard.utils import load_dataset
from b2aiprep.prepare.dataset import VBAIDataset

st.set_page_config(page_title="Demographics", page_icon="📈")

if "bids_dir" not in st.session_state:
    st.session_state.bids_dir = ""

bids_dir = st.sidebar.text_input(
    "Path to BIDS-like formatted data folder",
    key="bids_dir",
    value=st.session_state.get("bids_dir", ""),
    placeholder="e.g. /path/to/bids/data",
    help="Path to the BIDS-like formatted data folder.",
)

if not bids_dir:
    st.stop()

bids_dir = Path(bids_dir).resolve()
dataset: VBAIDataset = load_dataset(bids_dir)

st.markdown("# Demographics")
st.write("""This page overviews the demographics of the dataset.""")

@st.cache_data(ttl=3600, show_spinner=True)
def load_participants():
    """Load participants data from the dataset."""
    return dataset.load_participants()

st.write("## Session demographics")

df = load_participants()
if df is None:
    st.error("No participants data found. Please check the BIDS directory: {}".format(bids_dir))
    st.stop()

st.write("### Participants")
st.dataframe(df)

st.markdown("## Age & Sex")
sex_counts = df["sex_at_birth"].value_counts()

container = st.container()
col1, col2 = container.columns(2)

with col1:
    age_hist = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("age:Q", bin=alt.Bin(maxbins=30), title="Age"),
            y=alt.Y("count()", title="Count"),
            tooltip=["age:Q", "count()"],
        )
        .configure_axis(labelFontSize=12, titleFontSize=14)
    )

    st.altair_chart(age_hist)

with col2:
    sex_chart = (
        alt.Chart(sex_counts.reset_index())
        .mark_arc(innerRadius=50, outerRadius=120)
        .encode(
            theta=alt.Theta("count:Q", title="Count"),
            color=alt.Color(
                "sex_at_birth:N", 
                title="Sex at Birth",
                scale=alt.Scale(scheme="category10")
            ),
            tooltip=["sex_at_birth:N", "count:Q"]
        )
        .resolve_scale(color="independent")
        .properties(width=300, height=300)
    )

    st.altair_chart(sex_chart)

def extract_race_df(df):
    """Extract race data from df and re-shape to be easy to work with."""
    r_columns = [col for col in df.columns if "race___" in col]
    df_r = df[r_columns].copy()

    rename_dict = {}
    for c in df_r.columns:
        # rename the column using the unique values
        unique_values = df_r[c].dropna().unique()
        if len(unique_values) == 1:
            rename_dict[c] = unique_values[0]
        else:
            st.warning(f"Column {c} has multiple unique values: {unique_values}. Using last value.")
            rename_dict[c] = unique_values[-1]
    df_r.rename(columns=rename_dict, inplace=True)

    # keep it one-hot encoded
    for c in df_r.columns:
        df_r[c] = (df_r[c] == c).astype(int)
    
    return df_r

st.write("## Race & Ethnicity")

container = st.container()
col1, col2 = container.columns(2)

with col1:
    race_columns = [col for col in df.columns if "race___" in col]
    if not race_columns:
        st.warning("Race data not found in the dataset.")
    else:
        # race-columns are in a weird format: each column name is race___#, but the values
        # are the text version of the race. we need to make a new dataframe
        df_r = extract_race_df(df)
        # now we have a simple one-hot encoded dataframe
        # we can make a bar chart of the counts
        st.bar_chart(df_r.sum().sort_values(ascending=True))

with col2:
    ethnicity_counts = df["ethnicity"].value_counts()
    st.bar_chart(ethnicity_counts)

st.markdown("## Marital Status")
marital_status_columns = [col for col in df.columns if "marital_status___" in col]
marital_status_counts = df[marital_status_columns].sum()
st.bar_chart(marital_status_counts)


st.markdown("## Employment Status")
employ_status_columns = [col for col in df.columns if "employ_status___" in col]
employ_status_counts = df[employ_status_columns].sum()
st.bar_chart(employ_status_counts)

# we need to do some harmonization of the USA / CA salaries
st.markdown("## Household Income")
income = df[["household_income_usa", "household_income_ca"]].copy()
income_cols = list(income.columns)
# get the upper range of their income, or if they only have one, the upper limit
for col in income_cols:
    # need to extract the *last* instance of this pattern
    income[f"{col}_lower"] = income[col].str.extract(r"\$(\d+,\d+)\s*$")
    income[f"{col}_lower"] = income[f"{col}_lower"].str.replace(",", "")
    income[f"{col}_lower"] = pd.to_numeric(income[f"{col}_lower"], errors="coerce")

    # now create an integer which is higher if the value is higher
    income[f"{col}_seq_num"] = income[f"{col}_lower"].rank(ascending=True, method="dense")
    income[f"{col}_seq_num"] = income[f"{col}_seq_num"].fillna(-1).astype(int)

    idxNan = income[col].str.contains("Prefer not to answer").fillna(False)
    income.loc[idxNan, f"{col}_seq_num"] = 0

income["seq_num"] = income[["household_income_usa_seq_num", "household_income_ca_seq_num"]].max(
    axis=1
)
# get our look-up dict for each
income_lookups = {}
for col in income_cols:
    income_lookups[col] = (
        income[[col, f"{col}_seq_num"]].drop_duplicates().set_index(f"{col}_seq_num").to_dict()[col]
    )

income["country"] = "Missing"
idx = income["household_income_usa"].notnull()
income.loc[idx, "country"] = "USA"
idx = income["household_income_ca"].notnull()
income.loc[idx, "country"] = "Canada"

income_grouped = pd.crosstab(income["seq_num"], income["country"])
# as it turns out, both countries have the same values for income brackets
# so we can just use one of the mapping tables
n_missing = (income["seq_num"] == -1).sum()
income_grouped.index = income_grouped.index.map(income_lookups[col])
income_grouped = income_grouped[["USA", "Canada"]]
income_grouped.index.name = "Household Income (CAD or USD)"
# st.write(income_grouped)

# grouped barchart
income_grouped = income_grouped.reset_index()
income_grouped = income_grouped.melt(
    id_vars="Household Income (CAD or USD)", var_name="Country", value_name="Count"
)
chart = (
    alt.Chart(income_grouped)
    .mark_bar()
    .encode(
        x=alt.X("Household Income (CAD or USD):O", axis=alt.Axis(title="Income")),
        y=alt.Y("Count:Q", axis=alt.Axis(title="Count")),
        color="Country:N",
        tooltip=["Household Income (CAD or USD)", "Count", "Country"],
    )
)
st.altair_chart(chart, use_container_width=True)
st.write(f"{n_missing} missing a household income.")


st.markdown("## Full dataframe")
st.write(df)
