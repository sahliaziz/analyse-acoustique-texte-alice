"""
Author: Pierre PINCON
Team: SAMoVA
Date: 2022
"""

import os
from pathlib import Path
import re
from pydub import AudioSegment
import parselmouth

import numpy as np
import pandas as pd
from segment import Segment, SegmentType
from silero_vad import get_speech_timestamps, load_silero_vad


path_vocal_quality = "./4_qualite_vocale/Measures_sent1.txt"
path_vowel = "./6_voyelles/voweltriangle_praat.txt"
speech_directory = "./2_wav_traites"
tab_to_csv = [
    [
        "sujet",
        "genre",
        "repetition",
        "cpps",
        "slope",
        "tilt",
        "aire triangle s_2",
        "mean F0",
        "std F0",
        "speech rate",
        "articulation rate",
        "mean silence duration",
    ]
]


def read_wav(path : Path | str) -> tuple[bytes, int, float]:
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate, duration).
    """
    audio = AudioSegment.from_wav(path)

    assert audio.channels == 1
    assert audio.sample_width == 2
    assert audio.frame_rate in (8000, 16000, 32000, 48000)

    pcm_data = audio.raw_data
    sample_rate = audio.frame_rate
    duration = len(audio) / 1000.0

    return pcm_data, sample_rate, duration


def make_segments(audio_tensor, sample_rate):
    model = load_silero_vad()

    speech_timestamps = get_speech_timestamps(
        audio_tensor, model, sampling_rate=sample_rate
    )

    segments = []

    prev_end = 0.0

    for ts in speech_timestamps:
        start = ts["start"] / sample_rate
        end = ts["end"] / sample_rate

        # SILENCE before speech
        if start > prev_end:
            segments.append(
                Segment(prev_end, start, start - prev_end, SegmentType.SILENCE)
            )

        # VOICE segment
        segments.append(Segment(start, end, end - start, SegmentType.VOICE))

        prev_end = end

    # trailing silence
    total_duration = len(audio_tensor) / sample_rate
    if prev_end < total_duration:
        segments.append(
            Segment(
                prev_end, total_duration, total_duration - prev_end, SegmentType.SILENCE
            )
        )

    return segments


def vowel_counter(tg_content: str) -> int:
    """Count the number of vowels (thus syllables) in a Praat TextGrid file.
    Takes the content of a TextGrid file as a string, and returns the number of syllables.
    """
    vowels = ("@", "a~", "E", "9~", "e~", "O", "a", "e", "i", "o", "o~", "u", "y")

    tier3_content = tg_content.partition("item [3]")[0] + "item [3]"
    labels = re.findall(r'text = "(.*?)"', tier3_content)

    return sum(1 for label in labels if label.strip() in vowels)


def segments_list(path):
    """Reads a .wav file.

    Takes the path, and returns a list of segments.
    """
    pcm_data, sample_rate, duration = read_wav(path)  # sample rate should be 16000
    audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
    segments = make_segments(audio, sample_rate)
    return segments


def voice_length(path: Path | str, option: str) -> float:
    """Reads a .wav file.

    Takes an audio path and an option (articulation/speech), and returns the speech duration.
    """
    segments = segments_list(path)
    if option == "articulation":
        vla = 0.0
        for seg in segments:
            if seg.type == SegmentType.VOICE:
                vla += seg.duration
        return vla
    elif option == "speech":
        voice_segments = [seg for seg in segments if seg.type == SegmentType.VOICE]
        if not voice_segments:
            return 0.0
        return voice_segments[-1].end - voice_segments[0].start
    else:
        raise ValueError("Option must be 'articulation' or 'speech'")


def speech_rate(audio_file: Path | str, tg_content: str) -> float:
    """Reads a .wav file.

    Takes an audio path, and returns the speech rate.
    """
    vowel_count = vowel_counter(tg_content)
    speech_duration = voice_length(audio_file, "speech")
    if speech_duration == 0:
        return 0.0
    return vowel_count / speech_duration
    


def articulation_rate(audio_file: Path | str, tg_content: str) -> float:
    """Reads a .wav file.

    Takes an audio path, and returns the articulation rate.
    """
    vowel_count = vowel_counter(tg_content)
    articulation_duration = voice_length(audio_file, "articulation")
    if articulation_duration == 0:
        return 0.0
    return vowel_count / articulation_duration


def mean_silence_file(path: Path | str):
    """Reads a .wav file.

    Takes a .wav file, and returns the average duration of a silence.
    """
    segments = segments_list(path)
    silence_list = []
    for seg in segments:
        if seg.type == SegmentType.SILENCE:
            silence_list.append(seg.duration)
    return np.mean(silence_list)


def vocal_quality_analysis(path: Path | str) -> pd.DataFrame:
    """Reads a .csv file.
    Takes a .csv file created by "6_qualite_vocale.praat", and returns a dataframe of relevant measures.
    """
    df = pd.read_csv(path, header=None, sep=";")

    return pd.DataFrame({
        "sujet":      df.iloc[:, 0].str[:-9],
        "genre":      df.iloc[:, 0].str[-6],
        "repetition": df.iloc[:, 0].str[-8],
        "cpps":       df.iloc[:, 1],
        "slope":      df.iloc[:, 2],
        "tilt":       df.iloc[:, 3],
    })


def vowel_analysis(path: Path | str) -> pd.DataFrame:
    """Reads a .csv file.
    Takes a .csv file created by "10_vowel_triangle.praat", and returns a dataframe of relevant measures.
    """
    df = pd.read_csv(path, header=0, index_col=0, sep="\t")
    df = df.reindex(sorted(df.index), axis=0)
    return df


def measure_pitch(audio_file: Path) -> pd.DataFrame:
    snd = parselmouth.Sound(audio_file)
    pitch = snd.to_pitch(time_step=0.005, pitch_floor=50.0, pitch_ceiling=400.0)

    df = pd.DataFrame({
        "time": pitch.xs(),
        "f0": pitch.selected_array["frequency"]
    })
    return df


def pitch_mean(f0_df : pd.DataFrame) -> float:
    f0_list = f0_df.iloc[:, 1]
    return f0_list[f0_list > 0.0].mean()


def pitch_std(f0_df : pd.DataFrame) -> float:
    f0_list = f0_df.iloc[:, 1]
    return f0_list[f0_list > 0.0].std()


def main():
    # The following lines concatenate all measures in a single .csv for analysis.
    # tab = np.array(vocal_quality_analysis(path_vocal_quality))
    # tab = np.concatenate((tab, np.array(vowel_analysis(path_vowel))), axis=1)
    # tab = np.concatenate((tab, np.array(pitch_mean(directory_f0))), axis=1)
    # tab = np.concatenate((tab, np.array(pitch_std(directory_f0))), axis=1)
    # tab = np.concatenate((tab, np.array(speech_rate(speech_directory))), axis=1)
    # tab = np.concatenate((tab, np.array(articulation_rate(speech_directory))), axis=1)
    # tab = np.concatenate((tab, np.array(mean_silence_file(speech_directory))), axis=1)
    # tab_formalized = tab.tolist()
    # tab_to_csv.extend(tab_formalized)
    # df_export = pd.DataFrame(tab_to_csv)




if __name__ == "__main__":
    main()
