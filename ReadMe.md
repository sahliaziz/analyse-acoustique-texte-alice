# ReadMe

**Guide d'utilisation**

Le script est fait pour être utilisé de la sorte : lancer "install.sh" afin que celui-ci installe tous les outils
nécessaires, ensuite lancer "script.sh" afin que les mesures soient effectuées.

Une fois le script lancé, il vous sera demandé de copier les fichiers à analyser avec les lignes suivantes :

```bash
Placez vos enregistrements de sujet masculin au format WAV dans le dossier "1_wav_originaux/FR/Hommes"
Placez vos enregistrements de sujet masculin au format WAV dans le dossier "1_wav_originaux/FR/Femmes"
```

Ainsi que les alignements forcés qui leur sont liés (s'il y en a) :

```bash
Placez les alignements correspondants aux audios dans le dossier "Paty_alignment"
```

Il faut compter environ 5 minutes par audio. Une fois que ceux-ci sont traités, les résultats se trouveront dans le dossier "Analyzed_results" sous la forme de trois tableurs correspondant respectivement aux mesures acoustiques générales, aux mesures acoustiques liées aux consonnes et aux mesures acoustiques
liées aux semi-voyelles.

**Fonctionnement**

Les mesures présentées dans la thèse de Timothy Pommée sont effectuées par ses scripts (adaptés à une utilisation automatisée) ainsi que quelques rajouts. Elles sont ensuite formatées dans des .csv ne contenant que les mesures qui se sont avérées pertinentes pour discriminer des défauts de prononciation.

**Axes d'amélioration**

- [ ] Refaire les scripts faits en python 2 en python 3 pour ne plus dépendre de conda, et ainsi utiliser une venv
      pouvant être générée à partir d'un script pour assurer la pérennité du projet ainsi que la facilité de fonctionnement.
- [ ] Garder la précision de l'alignement fait sur Paty en ayant une version "offline" du modèle "Lucile - phonème adulte" pour pouvoir avoir quelque chose de totalement utilisable par une personne ne faisant pas partie de l'IRIT (un praticien par exemple) et ce de manière automatisée.
- [ ] Faire marcher "9_spectral_moments.py" même si l'alignement est fait avec BAUS et non Paty, ce qui n'est actuellement pas le cas (possible évolution de l'API depuis la thèse de Monsieur Pommée ?). Actuellement BAUS renvoie seulement 23 consonnes à la place des 29 qui devraient être renvoyées (et qui le sont par Paty).
