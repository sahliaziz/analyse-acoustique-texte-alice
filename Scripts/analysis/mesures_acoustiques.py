"""
Author: Pierre PINCON
Team: SAMoVA
Date: 2022
"""

from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import re
from pydub import AudioSegment
import parselmouth
import numpy as np
import pandas as pd
from silero_vad import get_speech_timestamps, load_silero_vad



class SegmentType(Enum):
    """Represents the segment type of the segment. It can be SILENCE (1) or VOICE (2)."""
    SILENCE = 1
    VOICE = 2


@dataclass
class Segment:
    """Represents a "segment" of audio frames by the start timecode, the end timecode, the duration / length of the
    segment and its type (SILENCE or VOICE)."""
    start: float
    end: float
    duration: float
    type: SegmentType

    def __str__(self) -> str:
        return f"{self.type} : {self.start} -> {self.end} [{self.duration}]"


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
        "fichier":    df.iloc[:, 0],
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


def mesures_acoustiques_consonnes(path: Path | str) -> pd.DataFrame:
    """Format spectral-moment output for a single file.

    Takes the path to `spectralmoments.txt` generated for one recording and
    returns only the relevant consonant measures.
    """
    df = pd.read_csv(path, sep="\t", header=None)
    rows: list[list[str]] = []

    for _, row in df.iloc[1:].iterrows():
        if any(row.iloc[index] == "--undefined--" for index in range(2, 6)):
            continue

        rows.append(
            [
                row.iloc[0],
                row.iloc[1],
                row.iloc[2],
                row.iloc[3],
                row.iloc[4],
                row.iloc[5]
            ]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "fichier",
            "phoneme",
            "cog",
            "sd",
            "skew",
            "kurt"
        ]
    )


def mesures_acoustiques_semivoyelles(path: Path) -> pd.DataFrame:
    """Format semivowel formant-transition output for a single file.

    Takes the path to one glide CSV generated by `11_formantTrans_glides.praat`
    and returns the relevant semivowel acoustic measures.
    """
    df = pd.read_csv(path, header=0)

    rows = [
        [
            row["interval"],
            float(row["f1_slope"]),
            float(row["f2_slope"]),
            float(row["f3_slope"]),
        ]
        for _, row in df.iterrows()
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "phoneme",
            "f1_slope",
            "f2_slope",
            "f3_slope",
        ],
    )


def measure_pitch(audio_file: Path) -> pd.DataFrame:
    snd = parselmouth.Sound(str(audio_file))
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


def mesures_acoustiques(
        qualite_vocale: Path, voweltriangle_path: Path, f0_df: pd.DataFrame, tg_content: str, audio_file: Path
    ) -> pd.DataFrame:
    """Combines all acoustic measures into a single dataframe.
    Takes paths to vocal quality and vowel triangle CSVs, a pitch dataframe, 
    a TextGrid string, and an audio file path, and returns a dataframe with all measures.
    """
    df_vq = vocal_quality_analysis(qualite_vocale)
    df_vt = vowel_analysis(voweltriangle_path)

    df = df_vq.copy()
    df["aire triangle s_2"] = df_vt["Area2"].values
    df["mean F0"] = pitch_mean(f0_df)
    df["std F0"] = pitch_std(f0_df)
    df["speech rate"] = speech_rate(audio_file, tg_content)
    df["articulation rate"] = articulation_rate(audio_file, tg_content)
    df["mean silence duration"] = mean_silence_file(audio_file)

    return df
