"""
Author: Pierre PINCON
Team: SAMoVA
Date: 2022
"""

import contextlib
import os
import time
import wave

import Frame
import librosa
import numpy as np
import pandas as pd
import webrtcvad
from Segment import Segment
from SegmentType import SegmentType

# Global variables
frame_duration = 10  # ms
aggressiveness = 2

start = time.perf_counter()

path_vocal_quality = "./4_qualite_vocale/Measures_sent1.txt"
path_vowel = "./6_voyelles/voweltriangle_praat.txt"
directory_f0 = "./Pitch_F0/Reaper_results"
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


def read_wave(path):
    """Reads a .wav file in bytes.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    try:
        with contextlib.closing(wave.open(path, "rb")) as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000, 48000)
            pcm_data = wf.readframes(wf.getnframes())
            duration = wf.getnframes() / float(sample_rate)

            return pcm_data, sample_rate, duration
    except:
        raise


def make_segments(aggressiveness, frames, sample_rate):
    """Makes simple segments using silence or voice frames.

    Takes the vad object, the audio frames, the sample rate of the audio record and returns list of tuples each
    containing start and end timecodes of the segment.
    """
    voice_trigger = False
    silence_trigger = False
    segments = []

    vad = webrtcvad.Vad(aggressiveness)

    # iterating frames, create voice or silence segment
    for frame in frames:
        # frame is silence type
        if vad.is_speech(frame.data, sample_rate) is False:
            # voice segment being created
            if voice_trigger is True:
                segments[-1].end = frame.timestamp
                segments[-1].duration = frame.timestamp - segments[-1].start
                voice_trigger = False

            # no silence segment being created
            if silence_trigger is False:
                segments.append(Segment(frame.timestamp, -1, -1, SegmentType.SILENCE))
                silence_trigger = True

        # frame is voice type
        else:
            # silence segment being created
            if silence_trigger is True:
                segments[-1].end = frame.timestamp
                segments[-1].duration = frame.timestamp - segments[-1].start
                silence_trigger = False

            # no voice segment being created
            if voice_trigger is False:
                segments.append(Segment(frame.timestamp, -1, -1, SegmentType.VOICE))
                voice_trigger = True
    else:
        # case of the last segment if it is not finished
        if voice_trigger is True or silence_trigger is True:
            segments[-1].end = frames[-1].timestamp
            segments[-1].duration = segments[-1].end - segments[-1].start

    return segments


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yield Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame.Frame(audio[offset : offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vowel_counter(path):
    """Count the number of vowels (thus syllables) in a Paty file.

    Takes the path of a TextGrid file, and returns the number of syllables.
    """
    vowel = ("@", "a~", "E", "9~", "e~", "O", "a", "e", "i", "o", "o~", "u", "y")
    vowel_count = 0
    file = open(path, "r")
    file_data = file.read()
    file_data = "".join(file_data.partition("item [3]")[:2])
    for vow in vowel:
        vowel_count += file_data.count(vow)
    file.close()
    return vowel_count


def segments_list(path):
    """Reads a .wav file.

    Takes the path, and returns a list of segments.
    """
    audio, sample_rate, duration = read_wave(path)  # sample rate should be 16000
    frames = frame_generator(frame_duration, audio, sample_rate)
    frames = list(frames)
    segments = make_segments(aggressiveness, frames, sample_rate)
    return segments


def voice_length(path, option):
    """Reads a .wav file.

    Takes an audio path and an option (articulation/speech), and returns the speech duration.
    """
    segments = segments_list(path)
    if option == "articulation":
        vla = 0
        for seg in segments:
            if seg.type == SegmentType.VOICE:
                vla += seg.duration
        return vla
    elif option == "speech":
        vls = librosa.get_duration(filename=path)
        number_seg = 0
        first_sil = len(segments)

        for seg in segments:
            if seg.type == SegmentType.SILENCE:
                if number_seg < first_sil:
                    first_sil = number_seg
                last_sil = number_seg
            number_seg += 1
        vls = vls - segments[last_sil].duration - segments[first_sil].duration
        return vls


def speech_rate(directory):
    """Reads all .wav files in a directory.

    Takes all .wav files in a directory, and returns a list of speech rates corresponding to all .wav files
    alphabetically.
    """
    tab = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".wav"):
            path = os.path.join(directory, filename)
            csv_path = os.path.join("./3_alignement_force/", path[16:-4] + ".csv")
            vowel_count = vowel_counter(csv_path)
            tab.append([vowel_count / voice_length(path, "speech")])
    return tab


def articulation_rate(directory):
    """Reads all .wav files in a directory.

    Takes all .wav files in a directory, and returns a list of articulation rates corresponding to all .wav files
    alphabetically.
    """
    tab = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".wav"):
            path = os.path.join(directory, filename)
            csv_path = os.path.join("./3_alignement_force/", path[16:-4] + ".csv")
            vowel_count = vowel_counter(csv_path)
            tab.append([vowel_count / voice_length(path, "articulation")])
    return tab


def mean_silence_file(path):
    """Reads a .wav file.

    Takes a .wav file, and returns the average duration of a silence.
    """
    segments = segments_list(path)
    silence_list = []
    for seg in segments:
        if seg.type == SegmentType.SILENCE:
            silence_list.append(seg.duration)
    return np.mean(silence_list)


def mean_silence_directory(directory):
    """Reads a .wav file.

    Takes all .wav files in a directory, and returns the average duration of a silence for each file. Iterates
    mean_silence_file in each file of a directory.
    """
    tab = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".wav"):
            path = os.path.join(directory, filename)
            csv_path = os.path.join("./3_alignement_force/", path[16:-4] + ".csv")
            vowel_count = vowel_counter(csv_path)
            tab.append([mean_silence_file(path)])
    return tab


def vocal_quality_analysis(path):
    """Reads a .csv file.

    Takes a .csv file created by "6_qualite_vocale.praat", and returns a list of relevant measures.
    """
    tab = []
    df = pd.read_csv(path, header=None, sep=";")
    for i in range(len(df.axes[0])):
        temporary_tab = []
        temporary_tab.append(df.iloc[i][0][:-9])
        temporary_tab.append(df.iloc[i][0][-6])
        temporary_tab.append(df.iloc[i][0][-8])
        temporary_tab.append(df.iloc[i][1])
        temporary_tab.append(df.iloc[i][2])
        temporary_tab.append(df.iloc[i][3])
        tab.append(temporary_tab)
    return tab


def vowel_analysis(path):
    """Reads a .csv file.

    Takes a .csv file created by "10_vowel_triangle.praat", and returns a list of relevant measures.
    """
    tab = []
    df = pd.read_csv(path, header=0, index_col=0, sep="\t")
    df = df.reindex(sorted(df.index), axis=0)
    for i in range(len(df.axes[0])):
        tab.append([df.iloc[i][2]])
    return tab


def pitch_mean(directory):
    """Reads all .csv files in a directory.

    Takes a .csv file created by Reaper, and returns a list of means corresponding to the pitch for each audio
    alphabetically.
    """
    tab = []
    for filename in sorted(os.listdir(directory)):
        path = os.path.join(directory, filename)
        df = pd.read_csv(path, sep=" ", skiprows=[i for i in range(1, 7)])
        list_f0 = df.values.tolist()
        sublist = []
        for i in range(len(list_f0)):
            if list_f0[i][1] != -1:
                sublist.append(list_f0[i][1])
        tab.append([np.mean(sublist)])
    return tab


def pitch_std(directory):
    """Reads all .csv files in a directory.

    Takes a .csv file created by Reaper, and returns a list of standard deviations corresponding to the pitch for
    each audio alphabetically.
    """
    tab = []
    for filename in sorted(os.listdir(directory)):
        path = os.path.join(directory, filename)
        df = pd.read_csv(path, sep=" ", skiprows=[i for i in range(1, 7)])
        list_f0 = df.values.tolist()
        sublist = []
        for i in range(len(list_f0)):
            if list_f0[i][1] != -1:
                sublist.append(list_f0[i][1])
        tab.append([np.std(sublist)])
    return tab


# The following lines concatene all measures in a single .csv for analsis


tab = np.array(vocal_quality_analysis(path_vocal_quality))
tab = np.concatenate((tab, np.array(vowel_analysis(path_vowel))), axis=1)
tab = np.concatenate((tab, np.array(pitch_mean(directory_f0))), axis=1)
tab = np.concatenate((tab, np.array(pitch_std(directory_f0))), axis=1)
tab = np.concatenate((tab, np.array(speech_rate(speech_directory))), axis=1)
tab = np.concatenate((tab, np.array(articulation_rate(speech_directory))), axis=1)
tab = np.concatenate((tab, np.array(mean_silence_directory(speech_directory))), axis=1)
tab_formalized = tab.tolist()
tab_to_csv.extend(tab_formalized)
df_export = pd.DataFrame(tab_to_csv)
df_export.to_csv("./Analyzed_results/mesures_acoustiques.csv", header=None, index=None)

print("Time elapsed during mesures_acoustiques.py = ", time.perf_counter() - start)
