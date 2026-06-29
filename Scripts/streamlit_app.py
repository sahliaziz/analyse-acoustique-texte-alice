import streamlit as st

main_page = st.Page("main.py", title="Extraction des mesures", icon="🎤")
about_page = st.Page("about.py", title="À propos", icon="ℹ️")
text_page = st.Page("text.py", title="Le texte standardisé", icon="📖")

pg = st.navigation([main_page, about_page, text_page])

pg.run()