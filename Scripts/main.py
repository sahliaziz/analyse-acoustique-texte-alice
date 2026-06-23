import io
import subprocess
import unicodedata
import zipfile
from pathlib import Path

import diverg_opt as diverg
import mesures_acoustiques as mesures
import pandas as pd
import plotly.express as px
import requests
import spectral_moments
import streamlit as st
import traitement_textgrid
from pydub import AudioSegment

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULT_DIR = PROJECT_ROOT / "result"
RESULT_DIR.mkdir(exist_ok=True)


# =====================================================
# SESSION STATE
# =====================================================

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "file_results" not in st.session_state:
    st.session_state.file_results = {}

if "all_mesures_df" not in st.session_state:
    st.session_state.all_mesures_df = pd.DataFrame()

if "all_mesures_cons_df" not in st.session_state:
    st.session_state.all_mesures_cons_df = pd.DataFrame()

if "all_mesures_semivoyelles_df" not in st.session_state:
    st.session_state.all_mesures_semivoyelles_df = pd.DataFrame()

if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False


# =====================================================
# FUNCTIONS
# =====================================================


def forced_alignment_MAUS(audio_file: Path, text_file: Path) -> str | None:
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
        download_url = response.text.split("<downloadLink>")[1].split(
            "</downloadLink>"
        )[0]
        textgrid_response = requests.get(download_url)
        if textgrid_response.status_code == 200:
            return textgrid_response.text
        else:
            print(
                f"Erreur lors du téléchargement du TextGrid: {textgrid_response.text}"
            )
            return None


def forced_alignment_MFA(
    audio_file: Path, transcript: Path, dictionary: str, output_dir: Path
) -> None:
    cmd = [
        "mfa",
        "align_one",
        "--single_speaker",
        "--output_format",
        "long_textgrid",
        audio_file,
        transcript,
        dictionary,
        "french_mfa",
        output_dir,
    ]
    subprocess.run(cmd, check=True)


def sanitize_filename(name: str) -> str:
    normalized = unicodedata.normalize("NFD", name)
    ascii_only = normalized.encode("ascii", errors="ignore").decode("ascii")
    return ascii_only


def process_audio(audio_file: Path) -> AudioSegment:
    audio = AudioSegment.from_wav(audio_file)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    return audio


def zip_result_dir(result_dir: Path) -> bytes:
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as zip_file:
        for file_path in result_dir.rglob("*"):
            if file_path.is_file():
                zip_file.write(
                    file_path,
                    file_path.relative_to(result_dir.parent),
                )

    return zip_buffer.getvalue()


# =====================================================
# UI
# =====================================================

st.title('"Le voyage d\'Alice" — Extraction de mesures acoustiques')

audio_files = st.file_uploader(
    "Téléchargez un ou plusieurs fichiers audio",
    type=["wav"],
    accept_multiple_files=True,
)

controls_disabled = not audio_files or st.session_state.analysis_started

gender_selector = st.radio(
    "Sélectionnez le genre du locuteur",
    options=["Homme", "Femme"],
    disabled=controls_disabled,
)

speaker_gender = "M" if gender_selector == "Homme" else "F"

transcription_checkbox = st.checkbox(
    "Transcrire le texte lu pour vérifier la lecture", disabled=controls_disabled
)


# =====================================================
# START ANALYSIS
# =====================================================

if st.button(
    "Démarrer l'analyse acoustique",
    disabled=controls_disabled,
):
    st.session_state.analysis_started = True
    st.rerun()

st.divider()

# =====================================================
# PROCESS FILES ONLY ONCE
# =====================================================

if st.session_state.analysis_started and audio_files:
    total_steps = 11 if transcription_checkbox else 9

    for audio_file in audio_files:
        audio_file.name = sanitize_filename(audio_file.name)

        # -------------------------------------
        # already processed
        # -------------------------------------

        if audio_file.name in st.session_state.processed_files:
            continue

        progress_text = st.empty()
        progress_text.markdown(f"**Traitement de {audio_file.name}**")
        progress_bar = st.progress(0)

        def set_progress(step: int, message: str) -> None:
            progress_text.markdown(f"**Traitement de {audio_file.name}**\n\n{message}")
            progress_bar.progress(int(step / total_steps * 100))

        # -------------------------------------
        # SAVE FILE
        # -------------------------------------

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
        tg_output_path = PROJECT_ROOT / f"{output_stem}.TextGrid"
        fichier_texte = PROJECT_ROOT / "texte_entier.txt"
        diverg_output_path = file_result_dir / "divergences.csv"
        spectral_debug_path = file_result_dir / "script_debug.txt"
        input_csv_path = (PROJECT_ROOT / "input_triangle_voc.csv").resolve()
        spectral_moments_output_path = file_result_dir / "spectralmoments.csv"
        formants_output_path = file_result_dir / "formants_glides.csv"
        mesure_vocale1_path = file_result_dir / "Measures_phrase1.txt"
        mesure_vocale2_path = file_result_dir / "Measures_phrase2.txt"
        voweltriangle_path = file_result_dir / "voweltriangle.txt"
        transcript_path = file_result_dir / f"{output_stem}_transcript.txt"

        set_progress(1, "Traitement de l'audio...")
        processed_audio = process_audio(audio_file_path)
        processed_audio.export(processed_audio_path, format="wav")

        set_progress(2, "Alignement forcé...")
        forced_alignment_MFA(processed_audio_path, fichier_texte, "alice", PROJECT_ROOT)

        textgrid_content = tg_output_path.read_text()
        df_tg = traitement_textgrid.tier_to_df(tg_output_path, 1)
        consonnes = traitement_textgrid.extract_consonants(df_tg)

        set_progress(3, "Extraction des mesures acoustiques...")
        subprocess.call(
            [
                "praat",
                "--run",
                SCRIPT_DIR / "qualite_vocale.praat",
                file_result_dir,
                processed_audio_path,
                tg_output_path,
            ]
        )

        fe = AudioSegment.from_wav(processed_audio_path).frame_rate
        data = AudioSegment.from_wav(processed_audio_path).get_array_of_samples()
        order = 16

        set_progress(4, "Détection des frontières...")
        frontieres = diverg.segment(data, fe, ordre=order, with_backward=True)

        diverg_df = pd.DataFrame(frontieres, columns=["time", "metric"])
        diverg_df["time"] = diverg_df["time"] / fe

        print(f"Ordre {order} : {len(frontieres)} Frontières")

        diverg_df.to_csv(diverg_output_path, index=False)

        set_progress(5, "Extraction des moments spectraux...")
        try:
            spectral_moments.extract_moments(
                consonnes, diverg_df, spectral_debug_path, processed_audio_path
            )
        except KeyError:
            st.error(
                "Erreur lors de l'extraction des moments spectraux, assurez vous que le texte lu correspond bien au texte standardisé."
            )
            continue

        with open(input_csv_path, "w") as f:
            f.write("Title;Speaker;File;Language;Log;Plotfile\n")
            f.write(
                f"{output_stem};{speaker_gender};{processed_audio_path.name};FR;{voweltriangle_path.resolve()};{(file_result_dir / 'pictures' / (output_stem + '_plot.png')).resolve()}\n"
            )
        set_progress(6, "Création du triangle vocalique...")
        subprocess.call(
            ["praat", "--run", SCRIPT_DIR / "vowel_triangle.praat", input_csv_path]
        )

        set_progress(7, "Extraction des formants pour les glides...")
        subprocess.call(
            [
                "praat",
                "--run",
                SCRIPT_DIR / "formantTrans_glides.praat",
                processed_audio_path,
                tg_output_path,
                formants_output_path,
            ]
        )

        set_progress(8, "Extraction des mesures vocales...")
        f0_df = mesures.measure_pitch(processed_audio_path)

        set_progress(9, "Compilation des mesures acoustiques...")
        mesures_df = mesures.mesures_acoustiques(
            qualite_vocale=mesure_vocale1_path,
            voweltriangle_path=voweltriangle_path,
            f0_df=f0_df,
            tg_content=textgrid_content,  # type: ignore
            audio_file=processed_audio_path,
        )
        mesures_cons_df = mesures.mesures_acoustiques_consonnes(
            spectral_moments_output_path
        )
        mesures_semivoyelles_df = mesures.mesures_acoustiques_semivoyelles(
            formants_output_path
        )

        mesures_df.to_csv(file_result_dir / "mesures_acoustiques.csv", index=False)
        mesures_cons_df.to_csv(
            file_result_dir / "mesures_acoustiques_consonnes.csv", index=False
        )
        mesures_semivoyelles_df.to_csv(
            file_result_dir / "mesures_acoustiques_semivoyelles.csv", index=False
        )

        if transcription_checkbox:
            import torch
            from qwen_asr import Qwen3ASRModel
            import dtw

            set_progress(10, "Transcription du texte lu...")

            model = Qwen3ASRModel.from_pretrained(
                pretrained_model_name_or_path="Qwen/Qwen3-ASR-0.6B",
                dtype=torch.bfloat16,
                device_map="cuda:0",
                max_inference_batch_size=32,
                max_new_tokens=512,
            )

            results = model.transcribe(
                audio=str(processed_audio_path),
                language="French",
            )

            transcript_text = results[0].text

            with open(transcript_path, "w") as f:
                f.write(transcript_text)

            set_progress(11, "Alignement forcé...")
            forced_alignment_MFA(
                processed_audio_path, transcript_path, "french_mfa", PROJECT_ROOT
            )

            textgrid_content = tg_output_path.read_text()
            transcript_df_tg = traitement_textgrid.tier_to_df(tg_output_path, 1)

            all_phonemes = sorted(
                set(df_tg["text"].tolist() + transcript_df_tg["text"].tolist())
            )
            phoneme_to_id = {p: i for i, p in enumerate(all_phonemes)}

            ref_features = dtw.df_to_feature_matrix(df_tg, phoneme_to_id)
            trans_features = dtw.df_to_feature_matrix(transcript_df_tg, phoneme_to_id)

            raw_dist, norm_dist, path_len = dtw.dtw_distance(
                ref_features, trans_features
            )

        # -------------------------------------------------
        # SAVE GLOBAL DATA
        # -------------------------------------------------

        st.session_state.all_mesures_df = pd.concat(
            [
                st.session_state.all_mesures_df,
                mesures_df,
            ],
            ignore_index=True,
        )

        st.session_state.all_mesures_cons_df = pd.concat(
            [
                st.session_state.all_mesures_cons_df,
                mesures_cons_df,
            ],
            ignore_index=True,
        )

        st.session_state.all_mesures_semivoyelles_df = pd.concat(
            [
                st.session_state.all_mesures_semivoyelles_df,
                mesures_semivoyelles_df,
            ],
            ignore_index=True,
        )

        # -------------------------------------------------
        # SAVE FILE RESULTS
        # -------------------------------------------------

        st.session_state.file_results[audio_file.name] = {
            "output_stem": output_stem,
            "mesures_df": mesures_df,
            "mesures_cons_df": mesures_cons_df,
            "mesures_semivoyelles_df": mesures_semivoyelles_df,
            "phrase1_img": file_result_dir / "pictures" / f"{output_stem}_phrase1.png",
            "phrase2_img": file_result_dir / "pictures" / f"{output_stem}_phrase2.png",
            "triangle_img": file_result_dir / "pictures" / f"{output_stem}_plot.png",
            "zip_data": zip_result_dir(file_result_dir),
            "raw_dtw_distance": raw_dist if transcription_checkbox else None,
            "norm_dtw_distance": norm_dist if transcription_checkbox else None,
        }

        st.session_state.processed_files.add(audio_file.name)

        progress_text.empty()
        progress_bar.empty()


# =====================================================
# DISPLAY RESULTS
# =====================================================

if st.session_state.file_results:
    st.header("Résultats")

    for (
        filename,
        result,
    ) in st.session_state.file_results.items():
        with st.expander(
            f"Résultats pour {filename}",
            expanded=True,
        ):
            st.subheader("1) Mesures de qualité vocale")

            st.image(result["phrase1_img"])

            st.image(result["phrase2_img"])
            
            if result['raw_dtw_distance'] is not None and result['norm_dtw_distance'] is not None:
                st.write(
                    f"Distance DTW brute : {result['raw_dtw_distance']:.2f}, Distance DTW normalisée : {result['norm_dtw_distance']:.2f}"
                )

            st.subheader("2) Mesures vocaliques")

            st.write(result["mesures_df"].drop(columns=["Fichier"]))

            st.image(result["triangle_img"])

            st.subheader("3) Mesures consonantiques")

            st.write(result["mesures_cons_df"].drop(columns=["Fichier"]))

            st.subheader("4) Mesures semi-consonantiques")

            st.write(result["mesures_semivoyelles_df"].drop(columns=["Fichier"]))
            st.download_button(
                "Télécharger les résultats",
                data=result["zip_data"],
                file_name=(f"{result['output_stem']}_resultats.zip"),
                mime="application/zip",
                on_click="ignore",
            )


# =====================================================
# COMPARISON SECTION
# =====================================================

if not st.session_state.all_mesures_df.empty:
    all_mesures_df = st.session_state.all_mesures_df
    all_mesures_cons_df = st.session_state.all_mesures_cons_df
    all_mesures_semivoyelles_df = st.session_state.all_mesures_semivoyelles_df

    with st.expander(
        "Comparer les enregistrements",
        expanded=True,
    ):
        category = st.selectbox(
            "Sélectionnez une catégorie de mesures à comparer",
            options=["Mesures vocales", "Mesures consonantiques"],
        )

        if category == "Mesures vocales":
            selection = st.selectbox(
                "Sélectionnez une mesure à comparer",
                options=st.session_state.all_mesures_df.drop(
                    columns=["Fichier"]
                ).columns,
            )
            st.bar_chart(all_mesures_df.set_index("Fichier")[selection])

        elif category == "Mesures consonantiques":
            selection = st.selectbox(
                "Sélectionnez une mesure à comparer",
                options=st.session_state.all_mesures_cons_df.drop(
                    columns=["Fichier", "Phonème"]
                ).columns,
            )
            if selection in all_mesures_cons_df.columns:
                fig = px.bar(
                    all_mesures_cons_df,
                    x="Fichier",
                    y=selection,
                    color="Phonème",
                    barmode="group",
                )
                st.caption("Survolez le graphique pour afficher les options de téléchargement.")
                st.plotly_chart(fig)
