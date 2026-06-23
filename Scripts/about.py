import streamlit as st

st.markdown("""
            
    **"Le voyage d'Alice"** est un texte standardisé créé pour l'évaluation de la parole et de la voix en français.
    Il permet d'analyser :

    - **L'articulation** des sons de la parole (dysarthrie, apraxie)
    - Les **variations prosodiques** et le **comportement phonatoire** (dysphonie, harmonisation vocale)
    - La **fluence / disfluences** (bégaiement, bredouillement)

    chez les locuteurs âgés d'au moins 12 ans.

    Ce pipeline automatise l'extraction de **mesures acoustiques** à partir d'un enregistrement audio de la lecture
    de ce texte. Les mesures sont organisées en **quatre catégories** :

    ---

    ##### 1) Mesures de qualité vocale
    Extraites sur la phrase voisée : *« mais la brise légère et l'air iodé de la mer les ravivent »*

    - **CPPs** (Cepstral Peak Prominence Smoothed) — Proéminence du pic cepstral lissé. Plus le pic est prononcé,
    plus le spectre est périodique (voix saine). Une voix dysphonique donne un pic moins discernable.
    - **Pente** (LTAS slope) — Pente du spectre moyen à long terme (rapport d'énergie basses / hautes fréquences).
    Une pente accentuée peut indiquer une voix soufflée ou hypofonctionnelle.
    - **Tilt** (LTAS tilt) — Inclinaison de la droite de régression à travers le LTAS (différence d'énergie 0-1 kHz vs 1-10 kHz).

    ---

    ##### 2) Mesures vocaliques
    - Extraction des **deux premiers formants (F1, F2)** des voyelles cardinales.
    - Tracage du **triangle vocalique** et calcul de son **aire** relative à un triangle de référence (VSA).
    Le VSA est sensible aux différences d'intelligibilité et permet de détecter une centralisation des voyelles.

    ---

    ##### 3) Mesures consonantiques
    Calcul des **quatre moments spectraux** sur les consonnes en contexte /aCa/ :

    - **CoG** (Center of Gravity, en Hz) — Fréquence qui divise le spectre en deux moitiés d'énergie égale.
    - **SD** (Standard Deviation, en Hz) — Dispersion du noyau spectral autour du CoG.
    - **SKEW** (Skewness) — Asymétrie de la distribution spectrale. Valeur positive = inclinaison négative
    (concentration dans les basses fréquences).
    - **Kurtosis** — Acuité du pic spectral. Plus la valeur est élevée, plus le pic est défini.

    ---

    ##### 4) Mesures semi-consonantiques
    - Pente des **trois premiers formants (F1, F2, F3)** dans les semi-consonnes.
    La pente de F2 est un indicateur de la **vitesse des mouvements articulatoires** ;
    un ralentissement peut réduire l'intelligibilité.

    ---

    ##### 5) Autres mesures
    - **F0 moyenne** et **écart-type** — Fréquence fondamentale moyenne et sa variabilité.
    - **Débit de parole** — Nombre de syllabes par seconde (temps de parole total).
    - **Vitesse d'articulation** — Nombre de syllabes par seconde (phonation seule, sans les silences).
    - **Durée moyenne des silences** — Pause moyenne entre les segments de parole.

""")