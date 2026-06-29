import streamlit as st

from alice_texts import TEXT_OPTIONS


selected_text_label = st.radio(
    "Sélectionnez la version du texte",
    options=list(TEXT_OPTIONS),
)
selected_text = TEXT_OPTIONS[selected_text_label]

st.markdown(f"## {selected_text.title}")
st.markdown(selected_text.read().replace("\n", "  \n"))
