# Le voyage d’Alice

Pipeline d'extraction des mesures acoustiques de la parole et de la voix sur une lecture à voix haute du texte standardisé "Le voyage d'Alice"


## Prerequis

### Logiciels

- Python `>=3.11`
- [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)
- [praat](https://praat.org/) accessible en ligne de commande
- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) accessible avec la commande `mfa`
- [uv](https://docs.astral.sh/uv/)

## Installation

Cloner le dépôt, créer l'environnement `micromamba`, installer les dépendances Python, puis préparer MFA:

```bash
git clone https://github.com/sahliaziz/analyse-acoustique-texte-alice.git
cd analyse-acoustique-texte-alice

micromamba create -n alice-voix python=3.11 -c conda-forge
micromamba activate alice-voix

# Installer les dépendances du projet dans l'environnement actif
uv sync --active --frozen

# Installer MFA dans le même environnement
micromamba install -c conda-forge montreal-forced-aligner

# Récupérer le modèle acoustique français utilisé par l'alignement
mfa model download acoustic french_mfa
```

Le dictionnaire MFA utilisé par l'application est déjà inclus dans le dépôt sous `alice_mfa.dict`.

Vérifier également que `praat` est executable depuis le terminal :

```bash
praat --version
```

Vérifier aussi que MFA est bien disponible :

```bash
mfa --help
```

## Lancer l'interface

Une fois l'environnement configuré, lancer l'interface Streamlit avec :

```bash
streamlit run Scripts/streamlit_app.py
```

L'interface permet de téléverser un ou plusieurs fichiers `.wav`, de choisir le genre du locuteur et la version du texte, puis de lancer l'extraction des mesures acoustiques.

Il faut compter environ 5 minutes par audio. Une fois que ceux-ci sont traités, les résultats se trouveront dans le dossier "result" sous la forme de trois tableurs correspondant respectivement aux mesures acoustiques générales, aux mesures acoustiques liées aux consonnes et aux mesures acoustiques liées aux semi-voyelles.

## Fonctionnement

Les mesures présentées dans la thèse de Timothy Pommée (2021) sont effectuées par ses scripts (adaptés à une utilisation automatisée) ainsi que quelques rajouts. Elles sont ensuite formatées dans des `.csv` ne contenant que les mesures qui se sont avérées pertinentes pour discriminer des défauts de prononciation.
