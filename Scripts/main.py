import argparse
import pandas as pd
import requests
from pathlib import Path
from pydub import AudioSegment
import spectral_moments
import traitement_textgrid
import diverg
import subprocess
import analysis.mesures_acoustiques as mesures 


TEXTE = open('../texte_entier.txt', 'r').read()
TEXTE_EXTRAIT = TEXTE[:350]
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULT_DIR = PROJECT_ROOT / "result"
RESULT_DIR.mkdir(exist_ok=True)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", type=Path, help="Path to the input wav file")
    parser.add_argument(
        "speaker_gender",
        choices=("male", "female"),
        help="Speaker gender used by the vowel triangle script",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    audio_file_path = args.wav_path.resolve()
    fichier_texte = (SCRIPT_DIR.parent / "texte_entier.txt").resolve()
    speaker_code = "M" if args.speaker_gender == "male" else "F"
    output_stem = audio_file_path.stem
    processed_audio_path = PROJECT_ROOT / f"{output_stem}.wav"
    tg_output_path = PROJECT_ROOT / f"{output_stem}.TextGrid"
    diverg_output_path = RESULT_DIR / "divergences.csv"
    spectral_debug_path = RESULT_DIR / "script_debug.txt"
    input_csv_path = (PROJECT_ROOT / "input_triangle_voc.csv").resolve()
    formants_output_path = RESULT_DIR / "formants_glides.csv"
    mesure_vocale1_path = RESULT_DIR / "Measures_sent1.txt"
    mesure_vocale2_path = RESULT_DIR / "Measures_sent2.txt"
    voweltriangle_path = RESULT_DIR / "voweltriangle.txt"


    processed_audio = process_audio(audio_file_path)
    processed_audio.export(processed_audio_path, format="wav")

    textgrid_content = forced_alignment(processed_audio_path, fichier_texte)
    
    if textgrid_content:
        with open(tg_output_path, "w") as f:
            f.write(textgrid_content)

    df_tg = traitement_textgrid.tier_to_df(tg_output_path, 2)
    consonnes = traitement_textgrid.extract_consonants(df_tg)

    subprocess.call([
        "praat", "--run", "6_qualite_vocale.praat", processed_audio_path, tg_output_path
    ])

    fe = AudioSegment.from_wav(processed_audio_path).frame_rate
    data = AudioSegment.from_wav(processed_audio_path).get_array_of_samples()
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
    
    diverg_df.to_csv(diverg_output_path, index=False)

    spectral_moments.extract_moments(consonnes, diverg_df, spectral_debug_path, processed_audio_path)

    with open(input_csv_path, "w") as f:
        f.write("Title;Speaker;File;Language;Log;Plotfile\n")
        f.write(
            f"{output_stem};{speaker_code};{processed_audio_path.name};FR;{voweltriangle_path.resolve()};{(RESULT_DIR / 'pictures' / (output_stem + '_plot.png')).resolve()}\n"
        )

    subprocess.call(["praat", "--run", "10_VowelTriangle.praat", input_csv_path])

    subprocess.call([
        "praat", "--run", "11_formantTrans_glides.praat", processed_audio_path, tg_output_path, formants_output_path
    ])

    f0_df=mesures.measure_pitch(processed_audio_path)

    mesures_df = mesures.mesures_acoustiques(
        qualite_vocale=mesure_vocale1_path,
        voweltriangle_path=voweltriangle_path,
        f0_df=f0_df,
        tg_content=textgrid_content, # type: ignore
        audio_file=processed_audio_path,
    )

    print(mesures_df)
    mesures_df.to_csv(RESULT_DIR / "measures_acoustiques.csv")


if __name__ == "__main__":
    main()
