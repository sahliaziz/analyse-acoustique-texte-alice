# Le voyage d’Alice

Pipeline d'extraction des mesures acoustiques de la parole et de la voix sur une lecture à voix haute du texte standardisé "Le voyage d'Alice"


## Prerequis

### Logiciels

- Python `>=3.11`
- [praat](https://praat.org/) accessible en ligne de commande
- [uv](https://docs.astral.sh/uv/)

## Installation

Cloner le dépot, installer l'environnement Python et les dépendances:

```bash
git clone https://github.com/sahliaziz/analyse-acoustique-texte-alice.git
uv venv
source .venv/bin/activate
uv sync
```

Si `uv` n'est pas utilisé dans votre environnement, un environnement virtuel Python classique convient aussi, à condition d'installer les dépendances définies dans `pyproject.toml`.

Vérifier également que `praat` est executable depuis le terminal :

```bash
praat --version
```

Il faut compter environ 5 minutes par audio. Une fois que ceux-ci sont traités, les résultats se trouveront dans le dossier "result" sous la forme de trois tableurs correspondant respectivement aux mesures acoustiques générales, aux mesures acoustiques liées aux consonnes et aux mesures acoustiques liées aux semi-voyelles.

## Fonctionnement

Les mesures présentées dans la thèse de Timothy Pommée (2021) sont effectuées par ses scripts (adaptés à une utilisation automatisée) ainsi que quelques rajouts. Elles sont ensuite formatées dans des `.csv` ne contenant que les mesures qui se sont avérées pertinentes pour discriminer des défauts de prononciation.