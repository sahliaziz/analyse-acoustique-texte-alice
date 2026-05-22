#! /bin/bash
######################################################################
# Initialisation
######################################################################

DIR1='1_wav_originaux'
DIR2='2_wav_traites'
DIR3='3_alignement_force'
DIR4='4_qualite_vocale'
DIR5='5_consonnes'
DIR6='6_voyelles'
DIR7='7_semi_consonnes'
DIR8='8_Segments'
DIR9='Pitch_F0'
DIR10='Paty_alignment'
DIR11='Analyzed_results'
DIRS="$DIR1 $DIR2 $DIR3 $DIR4 $DIR5 $DIR6 $DIR7 $DIR8 $DIR9 $DIR10 $DIR11 $DIR1/FR $DIR1/FR/Hommes $DIR1/FR/Femmes $DIR4/only_voiced $DIR5/txt $DIR5/wav $DIR4/result $DIR5/result $DIR6/result $DIR7/result $DIR9/Reaper_results"
FILELOG="$0_$(date +%Y%m%d_%H%M%S).log"

TXTFULL='Lundi matin, Alice et son Papa vont à Malibou.
Là-bas, ils rejoignent Papy après un voyage sans soucis.
Il fait chaud, mais la brise légère et l’air iodé de la mer les ravivent.
Vers midi, Alice s’exclame :
J’ai vraiment très très faim !
Papy les guide alors vite vers un café luxueux au bord de l’eau :
Le Bigorneau Salé.
Mardi, ils vont à la plage.
Il n’y a pas un nuage dans le ciel.
Papa s’interroge :
Avons-nous pris la crème solaire ?
Bien sûr !
répond Alice.
Mercredi, Papa et Papy se baladent en bavardant.
Pendant ce temps, Alice se détend en lisant un roman et mange un bonbon à l’ananas.
Jeudi, elle va faire un jogging.
Papa lui crie :
Nous partons faire quelques achats !
Au magasin, Papy achète des noix de macadamia.
Vendredi, ils visitent un musée d’art abstrait.
Papa s’extasie devant un splendide tableau et demande :
Qui a donc créé cette œuvre ?
Samedi matin, Alice s’entraîne pour la soirée karaoké en répétant rapidement :
pataka pataka pataka ».
Samedi soir, ils fêtent leur départ en dansant la java sous le lilas.
Comme à l’arrivée, il fait chaud, mais la brise légère et l’air iodé de la mer les ravivent.
Dimanche, Alice, Papa et Papy quittent Malibou.
Ils rentrent affamés.
À table, il y a de la pizza garnie et des lasagnes aux champignons.
Rassasiés, ils s’exclament
Quel séjour extraordinaire !'
TXTEXTRAIT='Lundi matin, Alice et son Papa vont à Malibou.
Là-bas, ils rejoignent Papy après un voyage sans soucis.
Il fait chaud, mais la brise légère et l’air iodé de la mer les ravivent.
Vers midi, Alice s’exclame :
J’ai vraiment très très faim !
Papy les guide alors vite vers un café luxueux au bord de l’eau :
Le Bigorneau Salé.
Mardi, ils vont à la plage.'

echo "Suppression des répertoires anciennement utilisés"
rm -rf $(echo "$DIRS") 2>/dev/null
rm *.log 2>/dev/null

for DIR in $DIRS; do
    echo "Création du répertoire $DIR"
    mkdir "$DIR"
done

# TODO remove temp command
rm *.log 2>/dev/null
touch "$FILELOG"

# echo "DOSSIER DE DEPART" "$(cd -P "$DIR1" && pwd)" "DOSSIER ARRIVEE" "$(cd -P "$DIR2" && pwd)" "========="

# TODO install dep ffmpeg

######################################################################
# 1.
# Conversion de l'audio original en monocanal,
# échantillonnage à 16khZ
######################################################################

echo "Placez vos enregistrements de sujet masculin au format WAV dans le dossier \"$DIR1/FR/Hommes\""
echo "Placez vos enregistrements de sujet féminin au format WAV dans le dossier \"$DIR1/FR/Femmes\""
echo "Placez les alignements correspondants aux audios dans le dossier \"Paty_alignment\""
read -p "Appuyer sur ENTREE pour continuer"


echo "Les enregistrements correspondent-ils à une lecture du texte entier ?"
select inputChoice in "Oui" "Non"; do
    case $inputChoice in
    "Oui")
        TXT="$TXTFULL"
        break
        ;;
    "Non")
        TXT="$TXTEXTRAIT"
        break
        ;;
    *)
        echo "Réponse incorrect. Veuillez réessayer"
        ;;
    esac
done
for FILE in "$DIR1"/FR/*/*.wav; do
    echo "Traitement du fichier $FILE"
    ffmpeg -i "$FILE" -ac 1 -ar 16000 -acodec pcm_s16le "$DIR2"/$(basename "$FILE") >>"$FILELOG" 2>&1
done

######################################################################
# 2.
# Creer un fichier TXT correspondant au fichier WAV
######################################################################
for FILE in "$DIR2"/*.wav; do
    FILENAME=$(basename "$FILE" .wav)
    FILETXT="$DIR2/$FILENAME.txt"
    FILETEXTGRID="$DIR3/$FILENAME.TextGrid"
    TEMPXML="res.xml"

    echo "$TXT" >"$FILETXT"

    ######################################################################
    # 3.
    # Alignement forcé en ligne
    ######################################################################

    echo "Alignement forcé du fichier audio $FILE par BAS WebServices"
    curl -v -H 'content-type: multipart/form-data' -F SIGNAL=@"$FILE" LANGUAGE=fra-FR -F OUTFORMAT=TextGrid -F TEXT=@"$FILETXT" 'https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runMAUSBasic' 2>>"$FILELOG" >"$TEMPXML"
    echo $(cat "$TEMPXML") >> "$FILELOG"

    SUCCESS=$(sed -nre 's:^.*<success>(.*)</success>.*$:\1:p' "$TEMPXML")
    if [ "$SUCCESS" '=' 'true' ]; then
        echo "Téléchargement du fichier TextGrid"
        DOWNLOADLINK=$(sed -nre 's:^.*<downloadLink>(.*)</downloadLink>.*$:\1:p' "$TEMPXML")
        wget "$DOWNLOADLINK" -O "$FILETEXTGRID" -a "$FILELOG" >>"$FILELOG"
    else
        echo "Erreur API BAUS concernant le fichier \"$FILENAME\"" >&2
        sed -nre 's:^.*<output>(.*)</output>.*$:\1:p' "$TEMPXML" >&2
    fi
    rm "$TEMPXML"
done

# TODO install praat

uv run 'Scripts/BAUS_to_Paty.py'
praat --run 'Scripts/4_textgrid_to_table.praat' $(cd -P "$DIR3" && pwd)

uv run 'Scripts/5_extract_consonants_from_textgridtable.py'
#source ../venv/bin/activate 'Scripts/5_extract_consonants_from_textgridtable.py'

######################################################################
# 4.
# Mesures de qualité vocale
######################################################################

echo 'Début des prises de mesures'
echo 'Mesures de qualité vocale'
praat --run "Scripts/6_qualite_vocale.praat" $(pwd)
######################################################################
# 5.
# Mesures consonantiques : moments spectraux
######################################################################

for FILE in "$DIR2"/*.wav; do
    FILENAME=$(basename "$FILE" .wav)
    #echo "FILENAME = $(basename "$FILE" .wav)"
    FILELENGTH=$(wc -l <"$DIR3/${FILENAME}.TextGrid")
    MAXLEN='5000'
    FILEDIVERG="$DIR5/txt/${FILENAME}_diverg.txt"
    #echo "FILEDIVERG = $DIR5/txt/${FILENAME}_diverg.txt"
    FILESRC="$DIR5/wav/${FILENAME}.wav"
    #echo "FILESRC = $DIR5/wav/${FILENAME}.wav"
    cp "$FILE" "$FILESRC"
    echo "Création de $FILEDIVERG"
    echo "Traitement de $FILEDIVERG" >> "$FILELOG"
    if [ "$FILELENGTH" -ge "$MAXLEN" ]; then
        uv run 'Scripts/diverg.py' -i "$FILESRC" -b "$FILEDIVERG" -v >>"$FILELOG"
    fi
done

uv run 'Scripts/8_txt2textgrid.py'
#source ../venv/bin/activate 'Scripts/8_txt2textgrid.py'

echo 'Mesures des moments spectraux des consonnes'
uv run 'Scripts/9_spectral_moments.py'

######################################################################
# 6.
# Mesures vocaliques : triangle vocalique
######################################################################

INPUTFILE="input.csv"

echo 'Title;Speaker;File;Language;Log;Plotfile' >"$DIR6/$INPUTFILE"

for FILE in "$DIR1"/FR/Hommes/*.wav; do
    if test -f "$FILE"; then
        FILENAME=$(basename "$FILE" .wav)
        FILESRC=$(cd -P "$DIR2" && pwd)/"$FILENAME".wav
        FILEPLOT=$(cd -P "$DIR6/result" && pwd)/"$FILENAME".png
        echo "$FILENAME;M;$FILESRC;FR;voweltriangle_praat.txt;$FILEPLOT" >>"$DIR6/$INPUTFILE"
    fi
done

for FILE in "$DIR1"/FR/Femmes/*.wav; do
    if test -f "$FILE"; then
        FILENAME=$(basename "$FILE" .wav)
        FILESRC=$(cd -P "$DIR2" && pwd)/"$FILENAME".wav
        FILEPLOT=$(cd -P "$DIR6/result" && pwd)/"$FILENAME".png
        echo "$FILENAME;F;$FILESRC;FR;voweltriangle_praat.txt;$FILEPLOT" >>"$DIR6/$INPUTFILE"
    fi
done

echo 'Mesures des triangles vocaliques des voyelles'
echo $(cd -P "$DIR6" && echo $(pwd)/$INPUTFILE)
praat --run 'Scripts/10_VowelTriangle.praat' $(cd -P "$DIR6" && echo $(pwd)/$INPUTFILE)

######################################################################
# 7.
# Mesures semi-consonantiques : F2 slope
######################################################################

echo 'Mesures de F2 des semi-consonnes'
# The next part is made to avoid Praat's limitations of loaded files per instance
mkdir "./semivowel_environment/"

for file in ./2_wav_traites/*.wav; do
    filename=${file:16}
    filename=${filename::-4}
    mkdir "./semivowel_environment/"$filename
    mkdir "./semivowel_environment/"$filename"/7_semi_consonnes/"
    mkdir "./semivowel_environment/"$filename"/7_semi_consonnes/result/"
    mkdir "./semivowel_environment/"$filename"/2_wav_traites/"
    mkdir "./semivowel_environment/"$filename"/3_alignement_force/"
    cp "./2_wav_traites/"$filename* "./semivowel_environment/"$filename"/2_wav_traites/"
    cp "./3_alignement_force/"$filename* "./semivowel_environment/"$filename"/3_alignement_force/"
    praat --run 'Scripts/11_formantTrans_glides.praat' $(pwd)"/semivowel_environment/"$filename
    cp "./semivowel_environment/"$filename"/7_semi_consonnes/result/"$filename".csv" "./7_semi_consonnes/result/"
done

rm -r "./semivowel_environment/"

######################################################################
# 9.
# Mesures pitch : f0
######################################################################

echo 'Mesures pitch : f0'
for wavefile in ./2_wav_traites/*.wav; do
    f0=${wavefile:16}
    f0=${f0::-4}
    f0="./Pitch_F0/Reaper_results/"$f0".f0"
    reaper -i "$wavefile" -f $f0 -a -m 50 -x 400
done

######################################################################
# 8.
# Data analysis
######################################################################

echo 'Analyse et formalisation des données obtenues'
uv run 'Scripts/analysis/mesures_acoustiques.py'
uv run 'Scripts/analysis/mesures_acoustiques_consonnes.py'
uv run 'Scripts/analysis/mesures_acoustiques_semivoyelles.py'

echo 'Fin des analyses, vous pouvez fermer ce terminal'
exit 0
