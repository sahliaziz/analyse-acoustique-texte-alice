import pandas as pd
import requests
from pathlib import Path
from pydub import AudioSegment
import spectral_moments
import traitement_textgrid
import diverg
import subprocess
import parselmouth

"""
DIR1 = '1_wav_originaux'
DIR2 = '2_wav_traites'
DIR3 = '3_alignement_force'
DIR4 = '4_qualite_vocale'
DIR5 = '5_consonnes'
DIR6 = '6_voyelles'
DIR7 = '7_semi_consonnes'
DIR8 = '8_Segments'
DIR9 = 'Pitch_F0'
DIR10 = 'Paty_alignment'
DIR11 = 'Analyzed_results'
"""

TEXTE = open('../texte_entier.txt', 'r').read()
TEXTE_EXTRAIT = TEXTE[:350]

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


def measure_pitch(audio_file: Path) -> pd.DataFrame:
    snd = parselmouth.Sound(audio_file)
    pitch = snd.to_pitch(time_step=0.005, pitch_floor=50.0, pitch_ceiling=400.0)

    df = pd.DataFrame({
        "time": pitch.xs(),
        "f0": pitch.selected_array["frequency"]
    })
    return df


def main():
    fichier_audio = Path("../../exemple/JP_2.wav")
    fichier_texte = Path("../texte_entier.txt")
    """
    audio_traite = process_audio(fichier_audio)
    audio_traite.export("../../JP_2.wav", format="wav")
    textgrid_content = forced_alignment(Path("../../JP_2.wav"), fichier_texte)
    """
    tg_output_path = Path("../../JP_2.TextGrid")
    """
    if textgrid_content:
        with open(tg_output_path, "w") as f:
            f.write(textgrid_content)
    """
    df_tg = traitement_textgrid.tier_to_df(tg_output_path, 2)
    consonnes = traitement_textgrid.extract_consonants(df_tg)

    subprocess.call([
        "praat", "--run", "6_qualite_vocale.praat", "../../JP_2.wav", tg_output_path
    ])

    fe = AudioSegment.from_wav("../../JP_2.wav").frame_rate
    data = AudioSegment.from_wav("../../JP_2.wav").get_array_of_samples()
    order = 16

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
    
    diverg_df.to_csv("../../result/divergences.csv", index=False)

    spectral_moments.extract_moments(consonnes, diverg_df, "../../result/script_debug.txt", "../../JP_2.wav")


    subprocess.call(["praat", "--run", "11_formantTrans_glides.praat", "../../JP_2.wav", tg_output_path, "../../result/formants_glides.csv"])

import os;print(os.getcwd())

if __name__ == "__main__":
    main()