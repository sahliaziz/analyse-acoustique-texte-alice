import pandas as pd
import requests
from pathlib import Path
from pydub import AudioSegment
import spectral_moments
import traitement_textgrid
import diverg
import subprocess
import analysis.mesures_acoustiques as mesures
import streamlit as st
import unicodedata


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULT_DIR = PROJECT_ROOT / "result"
RESULT_DIR.mkdir(exist_ok=True)



def sanitize_filename(name: str) -> str:
    normalized = unicodedata.normalize("NFD", name)
    ascii_only = normalized.encode("ascii", errors="ignore").decode("ascii")
    return ascii_only


def forced_alignment(audio_file: Path, text_file: Path) -> str | None:
    url = "https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runMAUSBasic"

    files = {
        "SIGNAL": open(audio_file, "rb"),
        "TEXT": open(text_file, "rb"),
    }

    data = {
        "LANGUAGE": "fra-FR",
        "OUTFORMAT": "TextGrid",
    }

    response = requests.post(url, files=files, data=data)
    
    if response.status_code != 200:
        print(f"Erreur lors de l'alignement forcé: {response.text}")
        return None
    
    if "<success>true</success>" in response.text:
        download_url = response.text.split("<downloadLink>")[1].split("</downloadLink>")[0]
        textgrid_response = requests.get(download_url)
        if textgrid_response.status_code == 200:
            return textgrid_response.text
        else:
            print(f"Erreur lors du téléchargement du TextGrid: {textgrid_response.text}")
            return None


def process_audio(audio_file: Path) -> AudioSegment:
    audio = AudioSegment.from_wav(audio_file)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    return audio

st.title("Analyse acoustique d'un enregistrement vocal")

audio_files = st.file_uploader(
    "Téléchargez un ou plusieurs fichiers audio au format .wav",
    type=["wav"],
    accept_multiple_files=True
)

speaker_gender = st.radio("Sélectionnez le genre du locuteur", options=["M", "F"])

if audio_files is not None:
    for audio_file in audio_files:
        st.divider()

        audio_file.name = sanitize_filename(audio_file.name)

        audio_file_path = PROJECT_ROOT / audio_file.name
        with open(audio_file_path, "wb") as f:
            f.write(audio_file.getvalue())

        output_stem = audio_file_path.stem

        # Per-file result directories
        file_result_dir = RESULT_DIR / output_stem
        file_result_dir.mkdir(exist_ok=True)
        (file_result_dir / "pictures").mkdir(exist_ok=True)
        (file_result_dir / "only_voiced").mkdir(exist_ok=True)

        # All output paths scoped to this file's result dir
        processed_audio_path = PROJECT_ROOT / f"{output_stem}.wav"
        tg_output_path        = PROJECT_ROOT / f"{output_stem}.TextGrid"
        fichier_texte         = PROJECT_ROOT / "texte_entier.txt"
        diverg_output_path    = file_result_dir / "divergences.csv"
        spectral_debug_path   = file_result_dir / "script_debug.txt"
        input_csv_path        = (PROJECT_ROOT / "input_triangle_voc.csv").resolve()
        spectral_moments_output_path = file_result_dir / "spectralmoments.csv"
        formants_output_path  = file_result_dir / "formants_glides.csv"
        mesure_vocale1_path   = file_result_dir / "Measures_sent1.txt"
        mesure_vocale2_path   = file_result_dir / "Measures_sent2.txt"
        voweltriangle_path    = file_result_dir / "voweltriangle.txt"

        # Progress UI
        total_steps = 9
        progress_text = st.empty()
        progress_text.markdown(f"**Traitement de {audio_file.name}**")
        progress_bar = st.progress(0)

        def set_progress(step: int, message: str) -> None:
            progress_text.markdown(f"**Traitement de {audio_file.name}**\n\n{message}")
            progress_bar.progress(int(step / total_steps * 100))

        set_progress(1, "(1/9) Traitement de l'audio...")
        processed_audio = process_audio(audio_file_path)
        processed_audio.export(processed_audio_path, format="wav")

        set_progress(2, "(2/9) Alignement forcé...")
        textgrid_content = forced_alignment(processed_audio_path, fichier_texte)

        if textgrid_content:
            with open(tg_output_path, "w", encoding="utf-8") as f:
                f.write(textgrid_content)

        df_tg = traitement_textgrid.tier_to_df(tg_output_path, 2)
        consonnes = traitement_textgrid.extract_consonants(df_tg)

        set_progress(3, "(3/9) Extraction des mesures acoustiques...")
        subprocess.call([
            "praat", "--run", SCRIPT_DIR / "6_qualite_vocale.praat", file_result_dir, processed_audio_path, tg_output_path
        ])

        fe = AudioSegment.from_wav(processed_audio_path).frame_rate
        data = AudioSegment.from_wav(processed_audio_path).get_array_of_samples()
        order = 16

        set_progress(4, "(4/9) Détection des frontières...")
        frontieres, _ = diverg.segment(
            data,
            fe,
            ordre=order,
            with_backward=True,
            seuil_vois=None,
            withTrace=False
        )

        diverg_df = pd.DataFrame(frontieres, columns=["time", "metric"])
        diverg_df["time"] = diverg_df["time"] / fe

        print(f"Ordre {order} : {len(frontieres)} Frontières")

        diverg_df.to_csv(diverg_output_path, index=False)

        set_progress(5, "(5/9) Extraction des moments spectraux...")
        spectral_moments.extract_moments(consonnes, diverg_df, spectral_debug_path, processed_audio_path)

        with open(input_csv_path, "w") as f:
            f.write("Title;Speaker;File;Language;Log;Plotfile\n")
            f.write(
                f"{output_stem};{speaker_gender};{processed_audio_path.name};FR;{voweltriangle_path.resolve()};{(file_result_dir / 'pictures' / (output_stem + '_plot.png')).resolve()}\n"
            )
        set_progress(6, "(6/9) Création du triangle vocalique...")
        subprocess.call(["praat", "--run", SCRIPT_DIR / "10_VowelTriangle.praat", input_csv_path])

        set_progress(7, "(7/9) Extraction des formants pour les glides...")
        subprocess.call([
            "praat", "--run", SCRIPT_DIR / "11_formantTrans_glides.praat", processed_audio_path, tg_output_path, formants_output_path
        ])

        set_progress(8, "(8/9) Extraction des mesures vocales...")
        f0_df=mesures.measure_pitch(processed_audio_path)


        set_progress(9, "(9/9) Compilation des mesures acoustiques...")
        mesures_df = mesures.mesures_acoustiques(
            qualite_vocale=mesure_vocale1_path,
            voweltriangle_path=voweltriangle_path,
            f0_df=f0_df,
            tg_content=textgrid_content, # type: ignore
            audio_file=processed_audio_path,
        )

        st.divider()


        st.subheader("1) Mesures de qualité vocale")

        st.image(file_result_dir / "pictures" / f"{output_stem}_sent1.png", caption="Analyses acoustiques de la première phrase")
        st.image(file_result_dir / "pictures" / f"{output_stem}_sent2.png", caption="Analyses acoustiques de la deuxième phrase")

        st.subheader("2) Mesures vocaliques")
        st.write(mesures_df, "Mesures vocaliques extraites de l'enregistrement")
        st.image(file_result_dir / "pictures" / f"{output_stem}_plot.png", caption="Triangle vocalique")

        st.subheader("3) Mesures consonantiques")
        mesures_cons_df = mesures.mesures_acoustiques_consonnes(spectral_moments_output_path)
        st.write(mesures_cons_df, "Mesures consonantiques extraites de l'enregistrement")

        st.subheader("4) Mesure semi-consonantique")
        mesures_semivoyelles_df = mesures.mesures_acoustiques_semivoyelles(formants_output_path)
        st.write(mesures_semivoyelles_df, "Mesures semi-voyelles extraites de l'enregistrement")


        mesures_df.to_csv(file_result_dir / "mesures_acoustiques.csv", index=False)
        mesures_cons_df.to_csv(file_result_dir / "mesures_acoustiques_consonnes.csv", index=False)
        mesures_semivoyelles_df.to_csv(file_result_dir / "mesures_acoustiques_semivoyelles.csv", index=False)

        progress_text.markdown(f"**Traitement de {audio_file_path.name}**\n\nTerminé")
        progress_bar.progress(100)
        st.write("Analyse acoustique terminée. Résultats enregistrés dans le dossier 'result'.")
