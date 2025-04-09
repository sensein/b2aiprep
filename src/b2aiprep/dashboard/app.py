import streamlit as st

summary_page = st.Page(
    "pages/summary.py",
    title="Summary of dataset",
)

demographics_page = st.Page(
    "pages/1_Demographics.py",
    title="Demographics",
)

questionnaires_page = st.Page(
    "pages/2_Subject_Questionnaires.py",
    title="Questionnaires",
)

audio_page = st.Page(
    "pages/3_Audio.py",
    title="Audio",
)

search_page = st.Page(
    "pages/4_Search.py",
    title="Search",
)

validate_page = st.Page(
    "pages/5_Validate.py",
    title="Validate",
)

pg = st.navigation(
    pages=[
        summary_page,
        demographics_page,
        questionnaires_page,
        audio_page,
        search_page,
        validate_page,
    ],
    title="Dashboard",
    sidebar_title="Navigation",
)

pg.run()