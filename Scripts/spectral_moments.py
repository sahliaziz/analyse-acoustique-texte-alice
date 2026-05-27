#!/usr/bin/env python
# coding: utf-8
# this script is part of Timothy Pommée's PhD thesis (2021)

# Selection of burst analysis window in plosives and of stable segment in fricatives

# Consonant segments must have previously been detected by forced alignment
# Forced alignment txt format:
# left border time	k	right border time
# left border time	b	right border time
# ...etc.
# !! consonant order: p1, b, p2, p3, p4, v1, p5, S, g1, k1, d, p6, t1, k2, p7, t2, k3, p8, t3, k4, Z, v2, p9, f, t4, g2, z, s
# otherwise, adapt the indexes in "fa_df.iloc[i].split"

# Ruptures in the speech signal must have previously been detected using the diverg method
# Ruptures txt format:
# rupture1 time	forward/backward
# rupture2 time	forward/backward
# etc.

import math
import subprocess
import sys
from pathlib import Path

import librosa
import numpy
import parselmouth
import pandas as pd
from pydub import AudioSegment

SCRIPT_DIR = Path(__file__).resolve().parent
PRAAT_SCRIPT = SCRIPT_DIR / "script_spectral_moments_python.praat"
PLOSIVE_FRAME_SIZE = 0.005
FRICATIVE_FRAME_SIZE = 0.01
PRAAT_LABEL_ORDER = (
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "p8",
    "p9",
    "t1",
    "t2",
    "t3",
    "t4",
    "k1",
    "k2",
    "k3",
    "k4",
    "b",
    "d",
    "g1",
    "g2",
    "f",
    "s",
    "ch",
    "v1",
    "v2",
    "z",
    "j",
)
DEBUG_WINDOW_GROUPS = (
    ("p", ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9")),
    ("t", ("t1", "t2", "t3", "t4")),
    ("k", ("k1", "k2", "k3", "k4")),
    ("b", ("b",)),
    ("d", ("d",)),
    ("g", ("g1", "g2")),
    ("f", ("f",)),
    ("s", ("s",)),
    ("ch", ("ch",)),
    ("v", ("v1", "v2")),
    ("z", ("z",)),
    ("j", ("j",)),
)


def _praat_result_dir(path: Path) -> str:
    return f"{path.resolve().as_posix().rstrip('/')}/"


def _default_window(ruptures, segment_end: float, frame_size: float) -> float:
    return segment_end - frame_size if not ruptures else 0.0


def _build_praat_command(
    praat_id: str,
    praat_wav_dir: Path,
    labels: dict[str, str],
    windows: dict[str, float],
    praat_result_dir: str,
) -> list[str]:
    return [
        "praat",
        "--run",
        str(PRAAT_SCRIPT),
        praat_id,
        str(praat_wav_dir),
        *[labels[name] for name in PRAAT_LABEL_ORDER],
        *[str(windows[name]) for name in PRAAT_LABEL_ORDER],
        praat_result_dir,
    ]


def _write_debug_windows(result_file_path: Path, windows: dict[str, float]) -> None:
    with open(result_file_path, "a") as result_file:
        for prefix, names in DEBUG_WINDOW_GROUPS:
            values = "".join(
                f"mom_win_{name} = {round(windows[name], 6):<20}" for name in names
            )
            result_file.writelines(f"{values}\n" if prefix != "p" else f"\n{values}\n")


def extract_moments(
    fa_df: pd.DataFrame,
    diverg_df: pd.DataFrame,
    resFile: str | Path,
    soundfile: Path,
) -> None:
    soundfile = Path(soundfile).resolve()
    result_file_path = Path(resFile).resolve()
    result_file_path.parent.mkdir(parents=True, exist_ok=True)
    praat_id = soundfile.stem
    praat_wav_dir = soundfile.parent
    praat_result_dir = _praat_result_dir(result_file_path.parent)

    plos_p1 = fa_df.iloc[0]
    plos_p2 = fa_df.iloc[2]
    plos_p3 = fa_df.iloc[3]
    plos_p4 = fa_df.iloc[4]
    plos_p5 = fa_df.iloc[6]
    plos_p6 = fa_df.iloc[11]
    plos_p7 = fa_df.iloc[14]
    plos_p8 = fa_df.iloc[17]
    plos_p9 = fa_df.iloc[23]
    plos_t1 = fa_df.iloc[12]
    plos_t2 = fa_df.iloc[15]
    plos_t3 = fa_df.iloc[18]
    plos_t4 = fa_df.iloc[25]
    plos_k1 = fa_df.iloc[9]
    plos_k2 = fa_df.iloc[13]
    plos_k3 = fa_df.iloc[16]
    plos_k4 = fa_df.iloc[19]
    plos_b = fa_df.iloc[1]
    plos_d = fa_df.iloc[10]
    plos_g1 = fa_df.iloc[8]
    plos_g2 = fa_df.iloc[26]
    fric_f = fa_df.iloc[24]
    fric_s = fa_df.iloc[28]
    fric_ch = fa_df.iloc[7]
    fric_v1 = fa_df.iloc[5]
    fric_v2 = fa_df.iloc[22]
    fric_z = fa_df.iloc[27]
    fric_j = fa_df.iloc[21]

    plos_p1_beg = float(plos_p1.iloc[0])
    plos_p1_end = float(plos_p1.iloc[2])
    plos_p2_beg = float(plos_p2.iloc[0])
    plos_p2_end = float(plos_p2.iloc[2])
    plos_p3_beg = float(plos_p3.iloc[0])
    plos_p3_end = float(plos_p3.iloc[2])
    plos_p4_beg = float(plos_p4.iloc[0])
    plos_p4_end = float(plos_p4.iloc[2])
    plos_p5_beg = float(plos_p5.iloc[0])
    plos_p5_end = float(plos_p5.iloc[2])
    plos_p6_beg = float(plos_p6.iloc[0])
    plos_p6_end = float(plos_p6.iloc[2])
    plos_p7_beg = float(plos_p7.iloc[0])
    plos_p7_end = float(plos_p7.iloc[2])
    plos_p8_beg = float(plos_p8.iloc[0])
    plos_p8_end = float(plos_p8.iloc[2])
    plos_p9_beg = float(plos_p9.iloc[0])
    plos_p9_end = float(plos_p9.iloc[2])

    plos_t1_beg = float(plos_t1.iloc[0])
    plos_t1_end = float(plos_t1.iloc[2])
    plos_t2_beg = float(plos_t2.iloc[0])
    plos_t2_end = float(plos_t2.iloc[2])
    plos_t3_beg = float(plos_t3.iloc[0])
    plos_t3_end = float(plos_t3.iloc[2])
    plos_t4_beg = float(plos_t4.iloc[0])
    plos_t4_end = float(plos_t4.iloc[2])

    plos_k1_beg = float(plos_k1.iloc[0])
    plos_k1_end = float(plos_k1.iloc[2])
    plos_k2_beg = float(plos_k2.iloc[0])
    plos_k2_end = float(plos_k2.iloc[2])
    plos_k3_beg = float(plos_k3.iloc[0])
    plos_k3_end = float(plos_k3.iloc[2])
    plos_k4_beg = float(plos_k4.iloc[0])
    plos_k4_end = float(plos_k4.iloc[2])

    plos_b_beg = float(plos_b.iloc[0])
    plos_b_end = float(plos_b.iloc[2])
    plos_b_end_inf = float(plos_b_end) - 0.02
    plos_b_end_sup = float(plos_b_end) + 0.02

    plos_d_beg = float(plos_d.iloc[0])
    plos_d_end = float(plos_d.iloc[2])
    plos_d_end_inf = float(plos_d_end) - 0.02
    plos_d_end_sup = float(plos_d_end) + 0.02

    plos_g1_beg = float(plos_g1.iloc[0])
    plos_g1_end = float(plos_g1.iloc[2])
    plos_g1_end_inf = float(plos_g1_end) - 0.02
    plos_g1_end_sup = float(plos_g1_end) + 0.02
    plos_g2_beg = float(plos_g2.iloc[0])
    plos_g2_end = float(plos_g2.iloc[2])
    plos_g2_end_inf = float(plos_g2_end) - 0.02
    plos_g2_end_sup = float(plos_g2_end) + 0.02

    fric_f_beg = float(fric_f.iloc[0]) - 0.01
    fric_f_end = float(fric_f.iloc[2]) + 0.01

    fric_s_beg = float(fric_s.iloc[0]) - 0.01
    fric_s_end = float(fric_s.iloc[2]) + 0.01

    fric_ch_beg = float(fric_ch.iloc[0]) - 0.01
    fric_ch_end = float(fric_ch.iloc[2]) + 0.01

    fric_v1_beg = float(fric_v1.iloc[0]) - 0.01
    fric_v1_end = float(fric_v1.iloc[2]) + 0.01
    fric_v2_beg = float(fric_v2.iloc[0]) - 0.01
    fric_v2_end = float(fric_v2.iloc[2]) + 0.01

    fric_z_beg = float(fric_z.iloc[0]) - 0.01
    fric_z_end = float(fric_z.iloc[2]) + 0.01

    fric_j_beg = float(fric_j.iloc[0]) - 0.01
    fric_j_end = float(fric_j.iloc[2]) + 0.01

    p1_str = "p"
    p2_str = "p"
    p3_str = "p"
    p4_str = "p"
    p5_str = "p"
    p6_str = "p"
    p7_str = "p"
    p8_str = "p"
    p9_str = "p"
    t1_str = "t"
    t2_str = "t"
    t3_str = "t"
    t4_str = "t"
    k1_str = "k"
    k2_str = "k"
    k3_str = "k"
    k4_str = "k"
    b_str = "b"
    d_str = "d"
    g1_str = "g"
    g2_str = "g"
    f_str = "f"
    s_str = "s"
    ch_str = "ch"
    v1_str = "v"
    v2_str = "v"
    z_str = "z"
    j_str = "j"

    # extract ruptures inside of these consonant segments from diverg
    ruptures = diverg_df.iloc[:, 0]
    p1_rupt = [x for x in ruptures if x >= plos_p1_beg and x <= plos_p1_end]
    p2_rupt = [x for x in ruptures if x >= plos_p2_beg and x <= plos_p2_end]
    p3_rupt = [x for x in ruptures if x >= plos_p3_beg and x <= plos_p3_end]
    p4_rupt = [x for x in ruptures if x >= plos_p4_beg and x <= plos_p4_end]
    p5_rupt = [x for x in ruptures if x >= plos_p5_beg and x <= plos_p5_end]
    p6_rupt = [x for x in ruptures if x >= plos_p6_beg and x <= plos_p6_end]
    p7_rupt = [x for x in ruptures if x >= plos_p7_beg and x <= plos_p7_end]
    p8_rupt = [x for x in ruptures if x >= plos_p8_beg and x <= plos_p8_end]
    p9_rupt = [x for x in ruptures if x >= plos_p9_beg and x <= plos_p9_end]

    t1_rupt = [x for x in ruptures if x >= plos_t1_beg and x <= plos_t1_end]
    t2_rupt = [x for x in ruptures if x >= plos_t2_beg and x <= plos_t2_end]
    t3_rupt = [x for x in ruptures if x >= plos_t3_beg and x <= plos_t3_end]
    t4_rupt = [x for x in ruptures if x >= plos_t4_beg and x <= plos_t4_end]

    k1_rupt = [x for x in ruptures if x >= plos_k1_beg and x <= plos_k1_end]
    k2_rupt = [x for x in ruptures if x >= plos_k2_beg and x <= plos_k2_end]
    k3_rupt = [x for x in ruptures if x >= plos_k3_beg and x <= plos_k3_end]
    k4_rupt = [x for x in ruptures if x >= plos_k4_beg and x <= plos_k4_end]

    b_rupt = [x for x in ruptures if x >= plos_b_beg and x <= plos_b_end]
    b_rupt_end = [
        x for x in ruptures if x >= plos_b_end_inf and x <= plos_b_end_sup
    ]
    b_rupt_end_inf = [
        x for x in ruptures if x >= plos_b_end_inf and x <= plos_b_end
    ]
    b_rupt_end_sup = [
        x for x in ruptures if x > plos_b_end and x <= plos_b_end_sup
    ]

    d_rupt = [x for x in ruptures if x >= plos_d_beg and x <= plos_d_end]
    d_rupt_end = [
        x for x in ruptures if x >= plos_d_end_inf and x <= plos_d_end_sup
    ]
    d_rupt_end_inf = [
        x for x in ruptures if x >= plos_d_end_inf and x <= plos_d_end
    ]
    d_rupt_end_sup = [
        x for x in ruptures if x > plos_d_end and x <= plos_d_end_sup
    ]

    g1_rupt = [x for x in ruptures if x >= plos_g1_beg and x <= plos_g1_end]
    g1_rupt_end = [
        x for x in ruptures if x >= plos_g1_end_inf and x <= plos_g1_end_sup
    ]
    g1_rupt_end_inf = [
        x for x in ruptures if x >= plos_g1_end_inf and x <= plos_g1_end
    ]
    g1_rupt_end_sup = [
        x for x in ruptures if x > plos_g1_end and x <= plos_g1_end_sup
    ]
    g2_rupt = [x for x in ruptures if x >= plos_g2_beg and x <= plos_g2_end]
    g2_rupt_end = [
        x for x in ruptures if x >= plos_g2_end_inf and x <= plos_g2_end_sup
    ]
    g2_rupt_end_inf = [
        x for x in ruptures if x >= plos_g2_end_inf and x <= plos_g2_end
    ]
    g2_rupt_end_sup = [
        x for x in ruptures if x > plos_g2_end and x <= plos_g2_end_sup
    ]

    f_rupt = [x for x in ruptures if x >= fric_f_beg and x <= fric_f_end]

    s_rupt = [x for x in ruptures if x >= fric_s_beg and x <= fric_s_end]

    ch_rupt = [x for x in ruptures if x >= fric_ch_beg and x <= fric_ch_end]

    v1_rupt = [x for x in ruptures if x >= fric_v1_beg and x <= fric_v1_end]
    v2_rupt = [x for x in ruptures if x >= fric_v2_beg and x <= fric_v2_end]

    z_rupt = [x for x in ruptures if x >= fric_z_beg and x <= fric_z_end]

    j_rupt = [x for x in ruptures if x >= fric_j_beg and x <= fric_j_end]

    frame_size_plos = PLOSIVE_FRAME_SIZE
    frame_size_fric = FRICATIVE_FRAME_SIZE

    # Default windows for unvoiced plosives fall back to segment end - frame size.
    mom_win_p1 = _default_window(p1_rupt, plos_p1_end, frame_size_plos)
    mom_win_p2 = _default_window(p2_rupt, plos_p2_end, frame_size_plos)
    mom_win_p3 = _default_window(p3_rupt, plos_p3_end, frame_size_plos)
    mom_win_p4 = _default_window(p4_rupt, plos_p4_end, frame_size_plos)
    mom_win_p5 = _default_window(p5_rupt, plos_p5_end, frame_size_plos)
    mom_win_p6 = _default_window(p6_rupt, plos_p6_end, frame_size_plos)
    mom_win_p7 = _default_window(p7_rupt, plos_p7_end, frame_size_plos)
    mom_win_p8 = _default_window(p8_rupt, plos_p8_end, frame_size_plos)
    mom_win_p9 = _default_window(p9_rupt, plos_p9_end, frame_size_plos)
    mom_win_t1 = _default_window(t1_rupt, plos_t1_end, frame_size_plos)
    mom_win_t2 = _default_window(t2_rupt, plos_t2_end, frame_size_plos)
    mom_win_t3 = _default_window(t3_rupt, plos_t3_end, frame_size_plos)
    mom_win_t4 = _default_window(t4_rupt, plos_t4_end, frame_size_plos)
    mom_win_k1 = _default_window(k1_rupt, plos_k1_end, frame_size_plos)
    mom_win_k2 = _default_window(k2_rupt, plos_k2_end, frame_size_plos)
    mom_win_k3 = _default_window(k3_rupt, plos_k3_end, frame_size_plos)
    mom_win_k4 = _default_window(k4_rupt, plos_k4_end, frame_size_plos)

    # read sound file and get sample rate to adapt the number of samples per frame size accordingly
    snd = AudioSegment.from_wav(soundfile)
    samp_freq = snd.frame_rate
    # calculate total intensity of central 20ms of the forced alignment segment (supposedly silence), averaged and normalized

    y, _ = librosa.load(soundfile, sr=samp_freq)

    parsound = parselmouth.Sound(str(soundfile))

    p1_mid_silence_beg = int(
        ((plos_p1_beg + ((plos_p1_end - plos_p1_beg) / 2)) - 0.01) * samp_freq
    )
    p1_mid_silence_end = int(
        ((plos_p1_beg + ((plos_p1_end - plos_p1_beg) / 2)) + 0.01) * samp_freq
    )
    silence_part = parsound.extract_part(
        from_time=p1_mid_silence_beg / samp_freq,
        to_time=p1_mid_silence_end / samp_freq,
        preserve_times=True,
    )
    mean_e_silence_p1 = silence_part.get_intensity()
    sum_e = 0
    for i in range(p1_mid_silence_beg, p1_mid_silence_end):
        sample = parsound.extract_part(
            from_time=i / samp_freq,
            to_time=(i + 1) / samp_freq,
            preserve_times=True,
        )
        samp_e = sample.get_intensity()
        sum_e = (samp_e - mean_e_silence_p1) ** 2 + sum_e
    silence_e_p1 = sum_e
    silence_e_norm_p1 = silence_e_p1 / (samp_freq * 0.02)
    if math.isnan(silence_e_norm_p1) == True:
        silence_e_norm_p1 = 0

    p2_mid_silence_beg = int(
        ((plos_p2_beg + ((plos_p2_end - plos_p2_beg) / 2)) - 0.01) * samp_freq
    )
    p2_mid_silence_end = int(
        ((plos_p2_beg + ((plos_p2_end - plos_p2_beg) / 2)) + 0.01) * samp_freq
    )
    silence_part = parsound.extract_part(
        from_time=p2_mid_silence_beg / samp_freq,
        to_time=p2_mid_silence_end / samp_freq,
        preserve_times=True,
    )
    mean_e_silence_p2 = silence_part.get_intensity()
    sum_e = 0
    for i in range(p2_mid_silence_beg, p2_mid_silence_end):
        sample = parsound.extract_part(
            from_time=i / samp_freq,
            to_time=(i + 1) / samp_freq,
            preserve_times=True,
        )
        samp_e = sample.get_intensity()
        sum_e = (samp_e - mean_e_silence_p2) ** 2 + sum_e
    silence_e_p2 = sum_e
    silence_e_norm_p2 = silence_e_p2 / (samp_freq * 0.02)
    if math.isnan(silence_e_norm_p2) == True:
        silence_e_norm_p2 = 0

    p3_mid_silence_beg = int(
        ((plos_p3_beg + ((plos_p3_end - plos_p3_beg) / 2)) - 0.01) * samp_freq
    )
    p3_mid_silence_end = int(
        ((plos_p3_beg + ((plos_p3_end - plos_p3_beg) / 2)) + 0.01) * samp_freq
    )
    silence_part = parsound.extract_part(
        from_time=p3_mid_silence_beg / samp_freq,
        to_time=p3_mid_silence_end / samp_freq,
        preserve_times=True,
    )
    mean_e_silence_p3 = silence_part.get_intensity()
    sum_e = 0
    for i in range(p3_mid_silence_beg, p3_mid_silence_end):
        sample = parsound.extract_part(
            from_time=i / samp_freq,
            to_time=(i + 1) / samp_freq,
            preserve_times=True,
        )
        samp_e = sample.get_intensity()
        sum_e = (samp_e - mean_e_silence_p3) ** 2 + sum_e
    silence_e_p3 = sum_e
    silence_e_norm_p3 = silence_e_p3 / (samp_freq * 0.02)
    if math.isnan(silence_e_norm_p3) == True:
        silence_e_norm_p3 = 0

    p4_mid_silence_beg = int(
        ((plos_p4_beg + ((plos_p4_end - plos_p4_beg) / 2)) - 0.01) * samp_freq
    )
    p4_mid_silence_end = int(
        ((plos_p4_beg + ((plos_p4_end - plos_p4_beg) / 2)) + 0.01) * samp_freq
    )
    silence_part = parsound.extract_part(
        from_time=p4_mid_silence_beg / samp_freq,
        to_time=p4_mid_silence_end / samp_freq,
        preserve_times=True,
    )
    mean_e_silence_p4 = silence_part.get_intensity()
    sum_e = 0
    for i in range(p4_mid_silence_beg, p4_mid_silence_end):
        sample = parsound.extract_part(
            from_time=i / samp_freq,
            to_time=(i + 1) / samp_freq,
            preserve_times=True,
        )
        samp_e = sample.get_intensity()
        sum_e = (samp_e - mean_e_silence_p4) ** 2 + sum_e
    silence_e_p4 = sum_e
    silence_e_norm_p4 = silence_e_p4 / (samp_freq * 0.02)
    if math.isnan(silence_e_norm_p4) == True:
        silence_e_norm_p4 = 0

    p5_mid_silence_beg = int(
        ((plos_p5_beg + ((plos_p5_end - plos_p5_beg) / 2)) - 0.01) * samp_freq
    )
    p5_mid_silence_end = int(
        ((plos_p5_beg + ((plos_p5_end - plos_p5_beg) / 2)) + 0.01) * samp_freq
    )
    silence_part = parsound.extract_part(
        from_time=p5_mid_silence_beg / samp_freq,
        to_time=p5_mid_silence_end / samp_freq,
        preserve_times=True,
    )
    mean_e_silence_p5 = silence_part.get_intensity()
    sum_e = 0
    for i in range(p5_mid_silence_beg, p5_mid_silence_end):
        sample = parsound.extract_part(
            from_time=i / samp_freq,
            to_time=(i + 1) / samp_freq,
            preserve_times=True,
        )
        samp_e = sample.get_intensity()
        sum_e = (samp_e - mean_e_silence_p5) ** 2 + sum_e
    silence_e_p5 = sum_e
    silence_e_norm_p5 = silence_e_p5 / (samp_freq * 0.02)
    if math.isnan(silence_e_norm_p5) == True:
        silence_e_norm_p5 = 0

    p6_mid_silence_beg = int(
        ((plos_p6_beg + ((plos_p6_end - plos_p6_beg) / 2)) - 0.01) * samp_freq
    )
    p6_mid_silence_end = int(
        ((plos_p6_beg + ((plos_p6_end - plos_p6_beg) / 2)) + 0.01) * samp_freq
    )
    silence_part = parsound.extract_part(
        from_time=p6_mid_silence_beg / samp_freq,
        to_time=p6_mid_silence_end / samp_freq,
        preserve_times=True,
    )
    mean_e_silence_p6 = silence_part.get_intensity()
    sum_e = 0
    for i in range(p6_mid_silence_beg, p6_mid_silence_end):
        sample = parsound.extract_part(
            from_time=i / samp_freq,
            to_time=(i + 1) / samp_freq,
            preserve_times=True,
        )
        samp_e = sample.get_intensity()
        sum_e = (samp_e - mean_e_silence_p6) ** 2 + sum_e
    silence_e_p6 = sum_e
    silence_e_norm_p6 = silence_e_p6 / (samp_freq * 0.02)
    if math.isnan(silence_e_norm_p6) == True:
        silence_e_norm_p6 = 0

    p7_mid_silence_beg = int(
        ((plos_p7_beg + ((plos_p7_end - plos_p7_beg) / 2)) - 0.01) * samp_freq
    )
    p7_mid_silence_end = int(
        ((plos_p7_beg + ((plos_p7_end - plos_p7_beg) / 2)) + 0.01) * samp_freq
    )
    silence_part = parsound.extract_part(
        from_time=p7_mid_silence_beg / samp_freq,
        to_time=p7_mid_silence_end / samp_freq,
        preserve_times=True,
    )
    mean_e_silence_p7 = silence_part.get_intensity()
    sum_e = 0
    for i in range(p7_mid_silence_beg, p7_mid_silence_end):
        sample = parsound.extract_part(
            from_time=i / samp_freq,
            to_time=(i + 1) / samp_freq,
            preserve_times=True,
        )
        samp_e = sample.get_intensity()
        sum_e = (samp_e - mean_e_silence_p7) ** 2 + sum_e
    silence_e_p7 = sum_e
    silence_e_norm_p7 = silence_e_p7 / (samp_freq * 0.02)
    if math.isnan(silence_e_norm_p7) == True:
        silence_e_norm_p7 = 0

    p8_mid_silence_beg = int(
        ((plos_p8_beg + ((plos_p8_end - plos_p8_beg) / 2)) - 0.01) * samp_freq
    )
    p8_mid_silence_end = int(
        ((plos_p8_beg + ((plos_p8_end - plos_p8_beg) / 2)) + 0.01) * samp_freq
    )
    silence_part = parsound.extract_part(
        from_time=p8_mid_silence_beg / samp_freq,
        to_time=p8_mid_silence_end / samp_freq,
        preserve_times=True,
    )
    mean_e_silence_p8 = silence_part.get_intensity()
    sum_e = 0
    for i in range(p8_mid_silence_beg, p8_mid_silence_end):
        sample = parsound.extract_part(
            from_time=i / samp_freq,
            to_time=(i + 1) / samp_freq,
            preserve_times=True,
        )
        samp_e = sample.get_intensity()
        sum_e = (samp_e - mean_e_silence_p8) ** 2 + sum_e
    silence_e_p8 = sum_e
    silence_e_norm_p8 = silence_e_p8 / (samp_freq * 0.02)
    if math.isnan(silence_e_norm_p8) == True:
        silence_e_norm_p8 = 0

    p9_mid_silence_beg = int(
        ((plos_p9_beg + ((plos_p9_end - plos_p9_beg) / 2)) - 0.01) * samp_freq
    )
    p9_mid_silence_end = int(
        ((plos_p9_beg + ((plos_p9_end - plos_p9_beg) / 2)) + 0.01) * samp_freq
    )
    silence_part = parsound.extract_part(
        from_time=p9_mid_silence_beg / samp_freq,
        to_time=p9_mid_silence_end / samp_freq,
        preserve_times=True,
    )
    mean_e_silence_p9 = silence_part.get_intensity()
    sum_e = 0
    for i in range(p9_mid_silence_beg, p9_mid_silence_end):
        sample = parsound.extract_part(
            from_time=i / samp_freq,
            to_time=(i + 1) / samp_freq,
            preserve_times=True,
        )
        samp_e = sample.get_intensity()
        sum_e = (samp_e - mean_e_silence_p9) ** 2 + sum_e
    silence_e_p9 = sum_e
    silence_e_norm_p9 = silence_e_p9 / (samp_freq * 0.02)
    if math.isnan(silence_e_norm_p9) == True:
        silence_e_norm_p9 = 0

    t1_mid_silence_beg = int(
        ((plos_t1_beg + ((plos_t1_end - plos_t1_beg) / 2)) - 0.01) * samp_freq
    )
    t1_mid_silence_end = int(
        ((plos_t1_beg + ((plos_t1_end - plos_t1_beg) / 2)) + 0.01) * samp_freq
    )
    silence_part = parsound.extract_part(
        from_time=t1_mid_silence_beg / samp_freq,
        to_time=t1_mid_silence_end / samp_freq,
        preserve_times=True,
    )
    mean_e_silence_t1 = silence_part.get_intensity()
    sum_e = 0
    for i in range(t1_mid_silence_beg, t1_mid_silence_end):
        sample = parsound.extract_part(
            from_time=i / samp_freq,
            to_time=(i + 1) / samp_freq,
            preserve_times=True,
        )
        samp_e = sample.get_intensity()
        sum_e = (samp_e - mean_e_silence_t1) ** 2 + sum_e
    silence_e_t1 = sum_e
    silence_e_norm_t1 = silence_e_t1 / (samp_freq * 0.02)
    if math.isnan(silence_e_norm_t1) == True:
        silence_e_norm_t1 = 0

    t2_mid_silence_beg = int(
        ((plos_t2_beg + ((plos_t2_end - plos_t2_beg) / 2)) - 0.01) * samp_freq
    )
    t2_mid_silence_end = int(
        ((plos_t2_beg + ((plos_t2_end - plos_t2_beg) / 2)) + 0.01) * samp_freq
    )
    silence_part = parsound.extract_part(
        from_time=t2_mid_silence_beg / samp_freq,
        to_time=t2_mid_silence_end / samp_freq,
        preserve_times=True,
    )
    mean_e_silence_t2 = silence_part.get_intensity()
    sum_e = 0
    for i in range(t2_mid_silence_beg, t2_mid_silence_end):
        sample = parsound.extract_part(
            from_time=i / samp_freq,
            to_time=(i + 1) / samp_freq,
            preserve_times=True,
        )
        samp_e = sample.get_intensity()
        sum_e = (samp_e - mean_e_silence_t2) ** 2 + sum_e
    silence_e_t2 = sum_e
    silence_e_norm_t2 = silence_e_t2 / (samp_freq * 0.02)
    if math.isnan(silence_e_norm_t2) == True:
        silence_e_norm_t2 = 0

    t3_mid_silence_beg = int(
        ((plos_t3_beg + ((plos_t3_end - plos_t3_beg) / 2)) - 0.01) * samp_freq
    )
    t3_mid_silence_end = int(
        ((plos_t3_beg + ((plos_t3_end - plos_t3_beg) / 2)) + 0.01) * samp_freq
    )
    silence_part = parsound.extract_part(
        from_time=t3_mid_silence_beg / samp_freq,
        to_time=t3_mid_silence_end / samp_freq,
        preserve_times=True,
    )
    mean_e_silence_t3 = silence_part.get_intensity()
    sum_e = 0
    for i in range(t3_mid_silence_beg, t3_mid_silence_end):
        sample = parsound.extract_part(
            from_time=i / samp_freq,
            to_time=(i + 1) / samp_freq,
            preserve_times=True,
        )
        samp_e = sample.get_intensity()
        sum_e = (samp_e - mean_e_silence_t3) ** 2 + sum_e
    silence_e_t3 = sum_e
    silence_e_norm_t3 = silence_e_t3 / (samp_freq * 0.02)
    if math.isnan(silence_e_norm_t3) == True:
        silence_e_norm_t3 = 0

    t4_mid_silence_beg = int(
        ((plos_t4_beg + ((plos_t4_end - plos_t4_beg) / 2)) - 0.01) * samp_freq
    )
    t4_mid_silence_end = int(
        ((plos_t4_beg + ((plos_t4_end - plos_t4_beg) / 2)) + 0.01) * samp_freq
    )
    silence_part = parsound.extract_part(
        from_time=t4_mid_silence_beg / samp_freq,
        to_time=t4_mid_silence_end / samp_freq,
        preserve_times=True,
    )
    mean_e_silence_t4 = silence_part.get_intensity()
    sum_e = 0
    for i in range(t4_mid_silence_beg, t4_mid_silence_end):
        sample = parsound.extract_part(
            from_time=i / samp_freq,
            to_time=(i + 1) / samp_freq,
            preserve_times=True,
        )
        samp_e = sample.get_intensity()
        sum_e = (samp_e - mean_e_silence_t4) ** 2 + sum_e
    silence_e_t4 = sum_e
    silence_e_norm_t4 = silence_e_t4 / (samp_freq * 0.02)
    if math.isnan(silence_e_norm_t4) == True:
        silence_e_norm_t4 = 0

    k1_mid_silence_beg = int(
        ((plos_k1_beg + ((plos_k1_end - plos_k1_beg) / 2)) - 0.01) * samp_freq
    )
    k1_mid_silence_end = int(
        ((plos_k1_beg + ((plos_k1_end - plos_k1_beg) / 2)) + 0.01) * samp_freq
    )
    silence_part = parsound.extract_part(
        from_time=k1_mid_silence_beg / samp_freq,
        to_time=k1_mid_silence_end / samp_freq,
        preserve_times=True,
    )
    mean_e_silence_k1 = silence_part.get_intensity()
    sum_e = 0
    for i in range(k1_mid_silence_beg, k1_mid_silence_end):
        sample = parsound.extract_part(
            from_time=i / samp_freq,
            to_time=(i + 1) / samp_freq,
            preserve_times=True,
        )
        samp_e = sample.get_intensity()
        sum_e = (samp_e - mean_e_silence_k1) ** 2 + sum_e
    silence_e_k1 = sum_e
    silence_e_norm_k1 = silence_e_k1 / (samp_freq * 0.02)
    if math.isnan(silence_e_norm_k1) == True:
        silence_e_norm_k1 = 0

    k2_mid_silence_beg = int(
        ((plos_k2_beg + ((plos_k2_end - plos_k2_beg) / 2)) - 0.01) * samp_freq
    )
    k2_mid_silence_end = int(
        ((plos_k2_beg + ((plos_k2_end - plos_k2_beg) / 2)) + 0.01) * samp_freq
    )
    silence_part = parsound.extract_part(
        from_time=k2_mid_silence_beg / samp_freq,
        to_time=k2_mid_silence_end / samp_freq,
        preserve_times=True,
    )
    mean_e_silence_k2 = silence_part.get_intensity()
    sum_e = 0
    for i in range(k2_mid_silence_beg, k2_mid_silence_end):
        sample = parsound.extract_part(
            from_time=i / samp_freq,
            to_time=(i + 1) / samp_freq,
            preserve_times=True,
        )
        samp_e = sample.get_intensity()
        sum_e = (samp_e - mean_e_silence_k2) ** 2 + sum_e
    silence_e_k2 = sum_e
    silence_e_norm_k2 = silence_e_k2 / (samp_freq * 0.02)
    if math.isnan(silence_e_norm_k2) == True:
        silence_e_norm_k2 = 0

    k3_mid_silence_beg = int(
        ((plos_k3_beg + ((plos_k3_end - plos_k3_beg) / 2)) - 0.01) * samp_freq
    )
    k3_mid_silence_end = int(
        ((plos_k3_beg + ((plos_k3_end - plos_k3_beg) / 2)) + 0.01) * samp_freq
    )
    silence_part = parsound.extract_part(
        from_time=k3_mid_silence_beg / samp_freq,
        to_time=k3_mid_silence_end / samp_freq,
        preserve_times=True,
    )
    mean_e_silence_k3 = silence_part.get_intensity()
    sum_e = 0
    for i in range(k3_mid_silence_beg, k3_mid_silence_end):
        sample = parsound.extract_part(
            from_time=i / samp_freq,
            to_time=(i + 1) / samp_freq,
            preserve_times=True,
        )
        samp_e = sample.get_intensity()
        sum_e = (samp_e - mean_e_silence_k3) ** 2 + sum_e
    silence_e_k3 = sum_e
    silence_e_norm_k3 = silence_e_k3 / (samp_freq * 0.02)
    if math.isnan(silence_e_norm_k3) == True:
        silence_e_norm_k3 = 0

    k4_mid_silence_beg = int(
        ((plos_k4_beg + ((plos_k4_end - plos_k4_beg) / 2)) - 0.01) * samp_freq
    )
    k4_mid_silence_end = int(
        ((plos_k4_beg + ((plos_k4_end - plos_k4_beg) / 2)) + 0.01) * samp_freq
    )
    silence_part = parsound.extract_part(
        from_time=k4_mid_silence_beg / samp_freq,
        to_time=k4_mid_silence_end / samp_freq,
        preserve_times=True,
    )
    mean_e_silence_k4 = silence_part.get_intensity()
    sum_e = 0
    for i in range(k4_mid_silence_beg, k4_mid_silence_end):
        sample = parsound.extract_part(
            from_time=i / samp_freq,
            to_time=(i + 1) / samp_freq,
            preserve_times=True,
        )
        samp_e = sample.get_intensity()
        sum_e = (samp_e - mean_e_silence_k4) ** 2 + sum_e
    silence_e_k4 = sum_e
    silence_e_norm_k4 = silence_e_k4 / (samp_freq * 0.02)
    if math.isnan(silence_e_norm_k4) == True:
        silence_e_norm_k4 = 0

    # calculate total intensity of [framesize] ms after each rupture following the midpoint, averaged and normalized
    # the cut-off score is set to 2000

    p1_rupt_am = [
        x
        for x in p1_rupt
        if x >= (p1_mid_silence_end / samp_freq) and x <= plos_p1_end
    ]

    # check if first rupture intensity is > than 2000
    if not p1_rupt_am:
        mom_win_p1 = plos_p1_end - frame_size_plos
    else:
        p1_e_rupt1_beg = int(p1_rupt_am[0] * samp_freq)
        p1_e_rupt1_end = int((p1_rupt_am[0] + frame_size_plos) * samp_freq)
        rupt_part = parsound.extract_part(
            from_time=p1_e_rupt1_beg / samp_freq,
            to_time=p1_e_rupt1_end / samp_freq,
            preserve_times=True,
        )
        mean_e_rupt1_p1 = rupt_part.get_intensity()
        sum_e = 0
        for i in range(p1_e_rupt1_beg, p1_e_rupt1_end):
            sample = parsound.extract_part(
                from_time=i / samp_freq,
                to_time=(i + 1) / samp_freq,
                preserve_times=True,
            )
            samp_e = sample.get_energy()
            sum_e = (samp_e - mean_e_rupt1_p1) ** 2 + sum_e
        e_rupt1_p1 = sum_e
        e_rupt1_norm_p1 = e_rupt1_p1 / (samp_freq * 0.005)
        # if yes, then the first rupture after the median silence is chosen as the analysis window
        if e_rupt1_norm_p1 >= 2000:
            mom_win_p1 = p1_rupt_am[0]

        # if not, then we check for the next ruptures, starting at 60ms before the end of the plosive segment and ending at 5ms before the end of the plosive
        # if no rupture is detected in this part, take the end of the plosive segment - framesize as analysis window
        # the intensity of the ruptures must be > than the intensity of the central 20ms, otherwise take the end of the plosive segment - framesize as analysis window

        else:
            end_ruptures_p1 = [
                x
                for x in p1_rupt
                if (
                    x >= (plos_p1_end - 0.06)
                    and x > (p1_rupt_am[0])
                    and x < (plos_p1_end - 0.005)
                )
            ]
            # if no rupture is detected in this part, take the end of the plosive segment - framesize as analysis window
            if not end_ruptures_p1:
                mom_win_p1 = plos_p1_end - frame_size_plos
            else:
                for i in end_ruptures_p1:
                    p1_e_rupt_beg = int(i * samp_freq)
                    p1_e_rupt_end = int((i + frame_size_plos) * samp_freq)
                    rupt_part = parsound.extract_part(
                        from_time=p1_e_rupt_beg / samp_freq,
                        to_time=p1_e_rupt_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt_p1 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p1_e_rupt_beg, p1_e_rupt_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt_p1) ** 2 + sum_e
                    e_rupt_p1 = sum_e
                    e_rupt_norm_p1 = e_rupt_p1 / (samp_freq * 0.005)
                    # if a rupture is detected that has a higher intensity than the first rupture and the middle silence, and that is smaller than 3000, choose this analysis window and break
                    if (
                        e_rupt_norm_p1 >= e_rupt1_norm_p1
                        and e_rupt_norm_p1 >= silence_e_norm_p1
                    ) and e_rupt_norm_p1 < 3000:
                        mom_win_p1 = p1_e_rupt_beg / samp_freq
                        break
                # if no such rupture is detected, check if rupt1 is between 60ms and 5ms before the plosive end and higher than 1500 ; if yes, choose this analysis window
                if mom_win_p1 == 0 and len(end_ruptures_p1) > 0:
                    p1_e_rupt2_beg = int(end_ruptures_p1[0] * samp_freq)
                    p1_e_rupt2_end = int(
                        (end_ruptures_p1[0] + frame_size_plos) * samp_freq
                    )
                    rupt_part = parsound.extract_part(
                        from_time=p1_e_rupt2_beg / samp_freq,
                        to_time=p1_e_rupt2_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt2_p1 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p1_e_rupt2_beg, p1_e_rupt2_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt2_p1) ** 2 + sum_e
                    e_rupt2_p1 = sum_e
                    e_rupt2_norm_p1 = e_rupt_p1 / (samp_freq * 0.005)

                    if e_rupt1_norm_p1 >= 1500 and (
                        (p1_e_rupt1_beg / samp_freq) >= (plos_p1_end - 0.06)
                        and (p1_e_rupt1_beg / samp_freq) < (plos_p1_end - 0.005)
                    ):
                        mom_win_p = p1_rupt_am[0]
                    # if not, but the following rupture is smaller in energy, take rupt1 as analysis window
                    elif e_rupt2_norm_p1 < e_rupt1_norm_p1:
                        mom_win_p1 = p1_rupt_am[0]
                    # otherwise, take the end of the plosive segment - framesize as analysis window
                    else:
                        mom_win_p1 = plos_p1_end - frame_size_plos
                else:
                    mom_win_p1 = plos_p1_end - frame_size_plos

    p2_rupt_am = [
        x
        for x in p2_rupt
        if x >= (p2_mid_silence_end / samp_freq) and x <= plos_p2_end
    ]

    if not p2_rupt_am:
        mom_win_p2 = plos_p2_end - frame_size_plos
    else:
        p2_e_rupt1_beg = int(p2_rupt_am[0] * samp_freq)
        p2_e_rupt1_end = int((p2_rupt_am[0] + frame_size_plos) * samp_freq)
        rupt_part = parsound.extract_part(
            from_time=p2_e_rupt1_beg / samp_freq,
            to_time=p2_e_rupt1_end / samp_freq,
            preserve_times=True,
        )
        mean_e_rupt1_p2 = rupt_part.get_intensity()
        sum_e = 0
        for i in range(p2_e_rupt1_beg, p2_e_rupt1_end):
            sample = parsound.extract_part(
                from_time=i / samp_freq,
                to_time=(i + 1) / samp_freq,
                preserve_times=True,
            )
            samp_e = sample.get_energy()
            sum_e = (samp_e - mean_e_rupt1_p2) ** 2 + sum_e
        e_rupt1_p2 = sum_e
        e_rupt1_norm_p2 = e_rupt1_p2 / (samp_freq * 0.005)
        if e_rupt1_norm_p2 >= 2000:
            mom_win_p2 = p2_rupt_am[0]

        else:
            end_ruptures_p2 = [
                x
                for x in p2_rupt
                if (
                    x >= (plos_p2_end - 0.06)
                    and x > (p2_rupt_am[0])
                    and x < (plos_p2_end - 0.005)
                )
            ]
            if not end_ruptures_p2:
                mom_win_p2 = plos_p2_end - frame_size_plos
            else:
                for i in end_ruptures_p2:
                    p2_e_rupt_beg = int(i * samp_freq)
                    p2_e_rupt_end = int((i + frame_size_plos) * samp_freq)
                    rupt_part = parsound.extract_part(
                        from_time=p2_e_rupt_beg / samp_freq,
                        to_time=p2_e_rupt_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt_p2 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p2_e_rupt_beg, p2_e_rupt_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt_p2) ** 2 + sum_e
                    e_rupt_p2 = sum_e
                    e_rupt_norm_p2 = e_rupt_p2 / (samp_freq * 0.005)
                    if (
                        e_rupt_norm_p2 >= e_rupt1_norm_p2
                        and e_rupt_norm_p2 >= silence_e_norm_p2
                    ) and e_rupt_norm_p2 < 3000:
                        mom_win_p2 = p2_e_rupt_beg / samp_freq
                        break
                if mom_win_p2 == 0 and len(end_ruptures_p2) > 0:
                    p2_e_rupt2_beg = int(end_ruptures_p2[0] * samp_freq)
                    p2_e_rupt2_end = int(
                        (end_ruptures_p2[0] + frame_size_plos) * samp_freq
                    )
                    rupt_part = parsound.extract_part(
                        from_time=p2_e_rupt2_beg / samp_freq,
                        to_time=p2_e_rupt2_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt2_p2 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p2_e_rupt2_beg, p2_e_rupt2_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt2_p2) ** 2 + sum_e
                    e_rupt2_p2 = sum_e
                    e_rupt2_norm_p2 = e_rupt_p2 / (samp_freq * 0.005)

                    if e_rupt1_norm_p2 >= 1500 and (
                        (p2_e_rupt1_beg / samp_freq) >= (plos_p2_end - 0.06)
                        and (p2_e_rupt1_beg / samp_freq) < (plos_p2_end - 0.005)
                    ):
                        mom_win_p2 = p2_rupt_am[0]
                    elif e_rupt2_norm_p2 < e_rupt1_norm_p2:
                        mom_win_p2 = p2_rupt_am[0]
                    else:
                        mom_win_p2 = plos_p2_end - frame_size_plos
                else:
                    mom_win_p2 = plos_p2_end - frame_size_plos

    p3_rupt_am = [
        x
        for x in p3_rupt
        if x >= (p3_mid_silence_end / samp_freq) and x <= plos_p3_end
    ]

    if not p3_rupt_am:
        mom_win_p3 = plos_p3_end - frame_size_plos
    else:
        p3_e_rupt1_beg = int(p3_rupt_am[0] * samp_freq)
        p3_e_rupt1_end = int((p3_rupt_am[0] + frame_size_plos) * samp_freq)
        rupt_part = parsound.extract_part(
            from_time=p3_e_rupt1_beg / samp_freq,
            to_time=p3_e_rupt1_end / samp_freq,
            preserve_times=True,
        )
        mean_e_rupt1_p3 = rupt_part.get_intensity()
        sum_e = 0
        for i in range(p3_e_rupt1_beg, p3_e_rupt1_end):
            sample = parsound.extract_part(
                from_time=i / samp_freq,
                to_time=(i + 1) / samp_freq,
                preserve_times=True,
            )
            samp_e = sample.get_energy()
            sum_e = (samp_e - mean_e_rupt1_p3) ** 2 + sum_e
        e_rupt1_p3 = sum_e
        e_rupt1_norm_p3 = e_rupt1_p3 / (samp_freq * 0.005)
        if e_rupt1_norm_p3 >= 2000:
            mom_win_p3 = p3_rupt_am[0]

        else:
            end_ruptures_p3 = [
                x
                for x in p3_rupt
                if (
                    x >= (plos_p3_end - 0.06)
                    and x > (p3_rupt_am[0])
                    and x < (plos_p3_end - 0.005)
                )
            ]
            if not end_ruptures_p3:
                mom_win_p3 = plos_p3_end - frame_size_plos
            else:
                for i in end_ruptures_p3:
                    p3_e_rupt_beg = int(i * samp_freq)
                    p3_e_rupt_end = int((i + frame_size_plos) * samp_freq)
                    rupt_part = parsound.extract_part(
                        from_time=p3_e_rupt_beg / samp_freq,
                        to_time=p3_e_rupt_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt_p3 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p3_e_rupt_beg, p3_e_rupt_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt_p3) ** 2 + sum_e
                    e_rupt_p3 = sum_e
                    e_rupt_norm_p3 = e_rupt_p3 / (samp_freq * 0.005)
                    if (
                        e_rupt_norm_p3 >= e_rupt1_norm_p3
                        and e_rupt_norm_p3 >= silence_e_norm_p3
                    ) and e_rupt_norm_p3 < 3000:
                        mom_win_p3 = p3_e_rupt_beg / samp_freq
                        break
                if mom_win_p3 == 0 and len(end_ruptures_p3) > 0:
                    p3_e_rupt2_beg = int(end_ruptures_p3[0] * samp_freq)
                    p3_e_rupt2_end = int(
                        (end_ruptures_p3[0] + frame_size_plos) * samp_freq
                    )
                    rupt_part = parsound.extract_part(
                        from_time=p3_e_rupt2_beg / samp_freq,
                        to_time=p3_e_rupt2_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt2_p3 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p3_e_rupt2_beg, p3_e_rupt2_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt2_p3) ** 2 + sum_e
                    e_rupt2_p3 = sum_e
                    e_rupt2_norm_p3 = e_rupt_p3 / (samp_freq * 0.005)

                    if e_rupt1_norm_p3 >= 1500 and (
                        (p3_e_rupt1_beg / samp_freq) >= (plos_p3_end - 0.06)
                        and (p3_e_rupt1_beg / samp_freq) < (plos_p3_end - 0.005)
                    ):
                        mom_win_p3 = p3_rupt_am[0]
                    elif e_rupt2_norm_p3 < e_rupt1_norm_p3:
                        mom_win_p3 = p3_rupt_am[0]
                    else:
                        mom_win_p3 = plos_p3_end - frame_size_plos
                else:
                    mom_win_p3 = plos_p3_end - frame_size_plos

    p4_rupt_am = [
        x
        for x in p4_rupt
        if x >= (p4_mid_silence_end / samp_freq) and x <= plos_p4_end
    ]

    if not p4_rupt_am:
        mom_win_p4 = plos_p4_end - frame_size_plos
    else:
        p4_e_rupt1_beg = int(p4_rupt_am[0] * samp_freq)
        p4_e_rupt1_end = int((p4_rupt_am[0] + frame_size_plos) * samp_freq)
        rupt_part = parsound.extract_part(
            from_time=p4_e_rupt1_beg / samp_freq,
            to_time=p4_e_rupt1_end / samp_freq,
            preserve_times=True,
        )
        mean_e_rupt1_p4 = rupt_part.get_intensity()
        sum_e = 0
        for i in range(p4_e_rupt1_beg, p4_e_rupt1_end):
            sample = parsound.extract_part(
                from_time=i / samp_freq,
                to_time=(i + 1) / samp_freq,
                preserve_times=True,
            )
            samp_e = sample.get_energy()
            sum_e = (samp_e - mean_e_rupt1_p4) ** 2 + sum_e
        e_rupt1_p4 = sum_e
        e_rupt1_norm_p4 = e_rupt1_p4 / (samp_freq * 0.005)
        if e_rupt1_norm_p4 >= 2000:
            mom_win_p4 = p4_rupt_am[0]

        else:
            end_ruptures_p4 = [
                x
                for x in p4_rupt
                if (
                    x >= (plos_p4_end - 0.06)
                    and x > (p4_rupt_am[0])
                    and x < (plos_p4_end - 0.005)
                )
            ]
            if not end_ruptures_p4:
                mom_win_p4 = plos_p4_end - frame_size_plos
            else:
                for i in end_ruptures_p4:
                    p4_e_rupt_beg = int(i * samp_freq)
                    p4_e_rupt_end = int((i + frame_size_plos) * samp_freq)
                    rupt_part = parsound.extract_part(
                        from_time=p4_e_rupt_beg / samp_freq,
                        to_time=p4_e_rupt_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt_p4 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p4_e_rupt_beg, p4_e_rupt_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt_p4) ** 2 + sum_e
                    e_rupt_p4 = sum_e
                    e_rupt_norm_p4 = e_rupt_p4 / (samp_freq * 0.005)
                    if (
                        e_rupt_norm_p4 >= e_rupt1_norm_p4
                        and e_rupt_norm_p4 >= silence_e_norm_p4
                    ) and e_rupt_norm_p4 < 3000:
                        mom_win_p4 = p4_e_rupt_beg / samp_freq
                        break
                if mom_win_p4 == 0 and len(end_ruptures_p4) > 0:
                    p4_e_rupt2_beg = int(end_ruptures_p4[0] * samp_freq)
                    p4_e_rupt2_end = int(
                        (end_ruptures_p4[0] + frame_size_plos) * samp_freq
                    )
                    rupt_part = parsound.extract_part(
                        from_time=p4_e_rupt2_beg / samp_freq,
                        to_time=p4_e_rupt2_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt2_p4 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p4_e_rupt2_beg, p4_e_rupt2_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt2_p4) ** 2 + sum_e
                    e_rupt2_p4 = sum_e
                    e_rupt2_norm_p4 = e_rupt_p4 / (samp_freq * 0.005)

                    if e_rupt1_norm_p4 >= 1500 and (
                        (p4_e_rupt1_beg / samp_freq) >= (plos_p4_end - 0.06)
                        and (p4_e_rupt1_beg / samp_freq) < (plos_p4_end - 0.005)
                    ):
                        mom_win_p4 = p4_rupt_am[0]
                    elif e_rupt2_norm_p4 < e_rupt1_norm_p4:
                        mom_win_p4 = p4_rupt_am[0]
                    else:
                        mom_win_p4 = plos_p4_end - frame_size_plos
                else:
                    mom_win_p4 = plos_p4_end - frame_size_plos

    p5_rupt_am = [
        x
        for x in p5_rupt
        if x >= (p5_mid_silence_end / samp_freq) and x <= plos_p5_end
    ]

    if not p5_rupt_am:
        mom_win_p5 = plos_p5_end - frame_size_plos
    else:
        p5_e_rupt1_beg = int(p5_rupt_am[0] * samp_freq)
        p5_e_rupt1_end = int((p5_rupt_am[0] + frame_size_plos) * samp_freq)
        rupt_part = parsound.extract_part(
            from_time=p5_e_rupt1_beg / samp_freq,
            to_time=p5_e_rupt1_end / samp_freq,
            preserve_times=True,
        )
        mean_e_rupt1_p5 = rupt_part.get_intensity()
        sum_e = 0
        for i in range(p5_e_rupt1_beg, p5_e_rupt1_end):
            sample = parsound.extract_part(
                from_time=i / samp_freq,
                to_time=(i + 1) / samp_freq,
                preserve_times=True,
            )
            samp_e = sample.get_energy()
            sum_e = (samp_e - mean_e_rupt1_p5) ** 2 + sum_e
        e_rupt1_p5 = sum_e
        e_rupt1_norm_p5 = e_rupt1_p5 / (samp_freq * 0.005)
        if e_rupt1_norm_p5 >= 2000:
            mom_win_p5 = p5_rupt_am[0]

        else:
            end_ruptures_p5 = [
                x
                for x in p5_rupt
                if (
                    x >= (plos_p5_end - 0.06)
                    and x > (p5_rupt_am[0])
                    and x < (plos_p5_end - 0.005)
                )
            ]
            if not end_ruptures_p5:
                mom_win_p5 = plos_p5_end - frame_size_plos
            else:
                for i in end_ruptures_p5:
                    p5_e_rupt_beg = int(i * samp_freq)
                    p5_e_rupt_end = int((i + frame_size_plos) * samp_freq)
                    rupt_part = parsound.extract_part(
                        from_time=p5_e_rupt_beg / samp_freq,
                        to_time=p5_e_rupt_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt_p5 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p5_e_rupt_beg, p5_e_rupt_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt_p5) ** 2 + sum_e
                    e_rupt_p5 = sum_e
                    e_rupt_norm_p5 = e_rupt_p5 / (samp_freq * 0.005)
                    if (
                        e_rupt_norm_p5 >= e_rupt1_norm_p5
                        and e_rupt_norm_p5 >= silence_e_norm_p5
                    ) and e_rupt_norm_p5 < 3000:
                        mom_win_p5 = p5_e_rupt_beg / samp_freq
                        break
                if mom_win_p5 == 0 and len(end_ruptures_p5) > 0:
                    p5_e_rupt2_beg = int(end_ruptures_p5[0] * samp_freq)
                    p5_e_rupt2_end = int(
                        (end_ruptures_p5[0] + frame_size_plos) * samp_freq
                    )
                    rupt_part = parsound.extract_part(
                        from_time=p5_e_rupt2_beg / samp_freq,
                        to_time=p5_e_rupt2_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt2_p5 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p5_e_rupt2_beg, p5_e_rupt2_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt2_p5) ** 2 + sum_e
                    e_rupt2_p5 = sum_e
                    e_rupt2_norm_p5 = e_rupt_p5 / (samp_freq * 0.005)

                    if e_rupt1_norm_p5 >= 1500 and (
                        (p5_e_rupt1_beg / samp_freq) >= (plos_p5_end - 0.06)
                        and (p5_e_rupt1_beg / samp_freq) < (plos_p5_end - 0.005)
                    ):
                        mom_win_p5 = p5_rupt_am[0]
                    elif e_rupt2_norm_p5 < e_rupt1_norm_p5:
                        mom_win_p5 = p5_rupt_am[0]
                    else:
                        mom_win_p5 = plos_p5_end - frame_size_plos
                else:
                    mom_win_p5 = plos_p5_end - frame_size_plos

    p6_rupt_am = [
        x
        for x in p6_rupt
        if x >= (p6_mid_silence_end / samp_freq) and x <= plos_p6_end
    ]

    if not p6_rupt_am:
        mom_win_p6 = plos_p6_end - frame_size_plos
    else:
        p6_e_rupt1_beg = int(p6_rupt_am[0] * samp_freq)
        p6_e_rupt1_end = int((p6_rupt_am[0] + frame_size_plos) * samp_freq)
        rupt_part = parsound.extract_part(
            from_time=p6_e_rupt1_beg / samp_freq,
            to_time=p6_e_rupt1_end / samp_freq,
            preserve_times=True,
        )
        mean_e_rupt1_p6 = rupt_part.get_intensity()
        sum_e = 0
        for i in range(p6_e_rupt1_beg, p6_e_rupt1_end):
            sample = parsound.extract_part(
                from_time=i / samp_freq,
                to_time=(i + 1) / samp_freq,
                preserve_times=True,
            )
            samp_e = sample.get_energy()
            sum_e = (samp_e - mean_e_rupt1_p6) ** 2 + sum_e
        e_rupt1_p6 = sum_e
        e_rupt1_norm_p6 = e_rupt1_p6 / (samp_freq * 0.005)
        if e_rupt1_norm_p6 >= 2000:
            mom_win_p6 = p6_rupt_am[0]

        else:
            end_ruptures_p6 = [
                x
                for x in p6_rupt
                if (
                    x >= (plos_p6_end - 0.06)
                    and x > (p6_rupt_am[0])
                    and x < (plos_p6_end - 0.005)
                )
            ]
            if not end_ruptures_p6:
                mom_win_p6 = plos_p6_end - frame_size_plos
            else:
                for i in end_ruptures_p6:
                    p6_e_rupt_beg = int(i * samp_freq)
                    p6_e_rupt_end = int((i + frame_size_plos) * samp_freq)
                    rupt_part = parsound.extract_part(
                        from_time=p6_e_rupt_beg / samp_freq,
                        to_time=p6_e_rupt_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt_p6 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p6_e_rupt_beg, p6_e_rupt_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt_p6) ** 2 + sum_e
                    e_rupt_p6 = sum_e
                    e_rupt_norm_p6 = e_rupt_p6 / (samp_freq * 0.005)
                    if (
                        e_rupt_norm_p6 >= e_rupt1_norm_p6
                        and e_rupt_norm_p6 >= silence_e_norm_p6
                    ) and e_rupt_norm_p6 < 3000:
                        mom_win_p6 = p6_e_rupt_beg / samp_freq
                        break
                if mom_win_p6 == 0 and len(end_ruptures_p6) > 0:
                    p6_e_rupt2_beg = int(end_ruptures_p6[0] * samp_freq)
                    p6_e_rupt2_end = int(
                        (end_ruptures_p6[0] + frame_size_plos) * samp_freq
                    )
                    rupt_part = parsound.extract_part(
                        from_time=p6_e_rupt2_beg / samp_freq,
                        to_time=p6_e_rupt2_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt2_p6 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p6_e_rupt2_beg, p6_e_rupt2_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt2_p6) ** 2 + sum_e
                    e_rupt2_p6 = sum_e
                    e_rupt2_norm_p6 = e_rupt_p6 / (samp_freq * 0.005)

                    if e_rupt1_norm_p6 >= 1500 and (
                        (p6_e_rupt1_beg / samp_freq) >= (plos_p6_end - 0.06)
                        and (p6_e_rupt1_beg / samp_freq) < (plos_p6_end - 0.005)
                    ):
                        mom_win_p6 = p6_rupt_am[0]
                    elif e_rupt2_norm_p6 < e_rupt1_norm_p6:
                        mom_win_p6 = p6_rupt_am[0]
                    else:
                        mom_win_p6 = plos_p6_end - frame_size_plos
                else:
                    mom_win_p6 = plos_p6_end - frame_size_plos

    p7_rupt_am = [
        x
        for x in p7_rupt
        if x >= (p7_mid_silence_end / samp_freq) and x <= plos_p7_end
    ]

    if not p7_rupt_am:
        mom_win_p7 = plos_p7_end - frame_size_plos
    else:
        p7_e_rupt1_beg = int(p7_rupt_am[0] * samp_freq)
        p7_e_rupt1_end = int((p7_rupt_am[0] + frame_size_plos) * samp_freq)
        rupt_part = parsound.extract_part(
            from_time=p7_e_rupt1_beg / samp_freq,
            to_time=p7_e_rupt1_end / samp_freq,
            preserve_times=True,
        )
        mean_e_rupt1_p7 = rupt_part.get_intensity()
        sum_e = 0
        for i in range(p7_e_rupt1_beg, p7_e_rupt1_end):
            sample = parsound.extract_part(
                from_time=i / samp_freq,
                to_time=(i + 1) / samp_freq,
                preserve_times=True,
            )
            samp_e = sample.get_energy()
            sum_e = (samp_e - mean_e_rupt1_p7) ** 2 + sum_e
        e_rupt1_p7 = sum_e
        e_rupt1_norm_p7 = e_rupt1_p7 / (samp_freq * 0.005)
        if e_rupt1_norm_p7 >= 2000:
            mom_win_p7 = p7_rupt_am[0]

        else:
            end_ruptures_p7 = [
                x
                for x in p7_rupt
                if (
                    x >= (plos_p7_end - 0.06)
                    and x > (p7_rupt_am[0])
                    and x < (plos_p7_end - 0.005)
                )
            ]
            if not end_ruptures_p7:
                mom_win_p7 = plos_p7_end - frame_size_plos
            else:
                for i in end_ruptures_p7:
                    p7_e_rupt_beg = int(i * samp_freq)
                    p7_e_rupt_end = int((i + frame_size_plos) * samp_freq)
                    rupt_part = parsound.extract_part(
                        from_time=p7_e_rupt_beg / samp_freq,
                        to_time=p7_e_rupt_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt_p7 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p7_e_rupt_beg, p7_e_rupt_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt_p7) ** 2 + sum_e
                    e_rupt_p7 = sum_e
                    e_rupt_norm_p7 = e_rupt_p7 / (samp_freq * 0.005)
                    if (
                        e_rupt_norm_p7 >= e_rupt1_norm_p7
                        and e_rupt_norm_p7 >= silence_e_norm_p7
                    ) and e_rupt_norm_p7 < 3000:
                        mom_win_p7 = p7_e_rupt_beg / samp_freq
                        break
                if mom_win_p7 == 0 and len(end_ruptures_p7) > 0:
                    p7_e_rupt2_beg = int(end_ruptures_p7[0] * samp_freq)
                    p7_e_rupt2_end = int(
                        (end_ruptures_p7[0] + frame_size_plos) * samp_freq
                    )
                    rupt_part = parsound.extract_part(
                        from_time=p7_e_rupt2_beg / samp_freq,
                        to_time=p7_e_rupt2_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt2_p7 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p7_e_rupt2_beg, p7_e_rupt2_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt2_p7) ** 2 + sum_e
                    e_rupt2_p7 = sum_e
                    e_rupt2_norm_p7 = e_rupt_p7 / (samp_freq * 0.005)

                    if e_rupt1_norm_p7 >= 1500 and (
                        (p7_e_rupt1_beg / samp_freq) >= (plos_p7_end - 0.06)
                        and (p7_e_rupt1_beg / samp_freq) < (plos_p7_end - 0.005)
                    ):
                        mom_win_p7 = p7_rupt_am[0]
                    elif e_rupt2_norm_p7 < e_rupt1_norm_p7:
                        mom_win_p7 = p7_rupt_am[0]
                    else:
                        mom_win_p7 = plos_p7_end - frame_size_plos
                else:
                    mom_win_p7 = plos_p7_end - frame_size_plos

    p8_rupt_am = [
        x
        for x in p8_rupt
        if x >= (p8_mid_silence_end / samp_freq) and x <= plos_p8_end
    ]

    if not p8_rupt_am:
        mom_win_p8 = plos_p8_end - frame_size_plos
    else:
        p8_e_rupt1_beg = int(p8_rupt_am[0] * samp_freq)
        p8_e_rupt1_end = int((p8_rupt_am[0] + frame_size_plos) * samp_freq)
        rupt_part = parsound.extract_part(
            from_time=p8_e_rupt1_beg / samp_freq,
            to_time=p8_e_rupt1_end / samp_freq,
            preserve_times=True,
        )
        mean_e_rupt1_p8 = rupt_part.get_intensity()
        sum_e = 0
        for i in range(p8_e_rupt1_beg, p8_e_rupt1_end):
            sample = parsound.extract_part(
                from_time=i / samp_freq,
                to_time=(i + 1) / samp_freq,
                preserve_times=True,
            )
            samp_e = sample.get_energy()
            sum_e = (samp_e - mean_e_rupt1_p8) ** 2 + sum_e
        e_rupt1_p8 = sum_e
        e_rupt1_norm_p8 = e_rupt1_p8 / (samp_freq * 0.005)
        if e_rupt1_norm_p8 >= 2000:
            mom_win_p8 = p8_rupt_am[0]

        else:
            end_ruptures_p8 = [
                x
                for x in p8_rupt
                if (
                    x >= (plos_p8_end - 0.06)
                    and x > (p8_rupt_am[0])
                    and x < (plos_p8_end - 0.005)
                )
            ]
            if not end_ruptures_p8:
                mom_win_p8 = plos_p8_end - frame_size_plos
            else:
                for i in end_ruptures_p8:
                    p8_e_rupt_beg = int(i * samp_freq)
                    p8_e_rupt_end = int((i + frame_size_plos) * samp_freq)
                    rupt_part = parsound.extract_part(
                        from_time=p8_e_rupt_beg / samp_freq,
                        to_time=p8_e_rupt_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt_p8 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p8_e_rupt_beg, p8_e_rupt_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt_p8) ** 2 + sum_e
                    e_rupt_p8 = sum_e
                    e_rupt_norm_p8 = e_rupt_p8 / (samp_freq * 0.005)
                    if (
                        e_rupt_norm_p8 >= e_rupt1_norm_p8
                        and e_rupt_norm_p8 >= silence_e_norm_p8
                    ) and e_rupt_norm_p8 < 3000:
                        mom_win_p8 = p8_e_rupt_beg / samp_freq
                        break
                if mom_win_p8 == 0 and len(end_ruptures_p8) > 0:
                    p8_e_rupt2_beg = int(end_ruptures_p8[0] * samp_freq)
                    p8_e_rupt2_end = int(
                        (end_ruptures_p8[0] + frame_size_plos) * samp_freq
                    )
                    rupt_part = parsound.extract_part(
                        from_time=p8_e_rupt2_beg / samp_freq,
                        to_time=p8_e_rupt2_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt2_p8 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p8_e_rupt2_beg, p8_e_rupt2_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt2_p8) ** 2 + sum_e
                    e_rupt2_p8 = sum_e
                    e_rupt2_norm_p8 = e_rupt_p8 / (samp_freq * 0.005)

                    if e_rupt1_norm_p8 >= 1500 and (
                        (p8_e_rupt1_beg / samp_freq) >= (plos_p8_end - 0.06)
                        and (p8_e_rupt1_beg / samp_freq) < (plos_p8_end - 0.005)
                    ):
                        mom_win_p8 = p8_rupt_am[0]
                    elif e_rupt2_norm_p8 < e_rupt1_norm_p8:
                        mom_win_p8 = p8_rupt_am[0]
                    else:
                        mom_win_p8 = plos_p8_end - frame_size_plos
                else:
                    mom_win_p8 = plos_p8_end - frame_size_plos

    p9_rupt_am = [
        x
        for x in p9_rupt
        if x >= (p9_mid_silence_end / samp_freq) and x <= plos_p9_end
    ]

    if not p9_rupt_am:
        mom_win_p9 = plos_p9_end - frame_size_plos
    else:
        p9_e_rupt1_beg = int(p9_rupt_am[0] * samp_freq)
        p9_e_rupt1_end = int((p9_rupt_am[0] + frame_size_plos) * samp_freq)
        rupt_part = parsound.extract_part(
            from_time=p9_e_rupt1_beg / samp_freq,
            to_time=p9_e_rupt1_end / samp_freq,
            preserve_times=True,
        )
        mean_e_rupt1_p9 = rupt_part.get_intensity()
        sum_e = 0
        for i in range(p9_e_rupt1_beg, p9_e_rupt1_end):
            sample = parsound.extract_part(
                from_time=i / samp_freq,
                to_time=(i + 1) / samp_freq,
                preserve_times=True,
            )
            samp_e = sample.get_energy()
            sum_e = (samp_e - mean_e_rupt1_p9) ** 2 + sum_e
        e_rupt1_p9 = sum_e
        e_rupt1_norm_p9 = e_rupt1_p9 / (samp_freq * 0.005)
        if e_rupt1_norm_p9 >= 2000:
            mom_win_p9 = p9_rupt_am[0]

        else:
            end_ruptures_p9 = [
                x
                for x in p9_rupt
                if (
                    x >= (plos_p9_end - 0.06)
                    and x > (p9_rupt_am[0])
                    and x < (plos_p9_end - 0.005)
                )
            ]
            if not end_ruptures_p9:
                mom_win_p9 = plos_p9_end - frame_size_plos
            else:
                for i in end_ruptures_p9:
                    p9_e_rupt_beg = int(i * samp_freq)
                    p9_e_rupt_end = int((i + frame_size_plos) * samp_freq)
                    rupt_part = parsound.extract_part(
                        from_time=p9_e_rupt_beg / samp_freq,
                        to_time=p9_e_rupt_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt_p9 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p9_e_rupt_beg, p9_e_rupt_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt_p9) ** 2 + sum_e
                    e_rupt_p9 = sum_e
                    e_rupt_norm_p9 = e_rupt_p9 / (samp_freq * 0.005)
                    if (
                        e_rupt_norm_p9 >= e_rupt1_norm_p9
                        and e_rupt_norm_p9 >= silence_e_norm_p9
                    ) and e_rupt_norm_p9 < 3000:
                        mom_win_p9 = p9_e_rupt_beg / samp_freq
                        break
                if mom_win_p9 == 0 and len(end_ruptures_p9) > 0:
                    p9_e_rupt2_beg = int(end_ruptures_p9[0] * samp_freq)
                    p9_e_rupt2_end = int(
                        (end_ruptures_p9[0] + frame_size_plos) * samp_freq
                    )
                    rupt_part = parsound.extract_part(
                        from_time=p9_e_rupt2_beg / samp_freq,
                        to_time=p9_e_rupt2_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt2_p9 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(p9_e_rupt2_beg, p9_e_rupt2_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt2_p9) ** 2 + sum_e
                    e_rupt2_p9 = sum_e
                    e_rupt2_norm_p9 = e_rupt_p9 / (samp_freq * 0.005)

                    if e_rupt1_norm_p9 >= 1500 and (
                        (p9_e_rupt1_beg / samp_freq) >= (plos_p9_end - 0.06)
                        and (p9_e_rupt1_beg / samp_freq) < (plos_p9_end - 0.005)
                    ):
                        mom_win_p9 = p9_rupt_am[0]
                    elif e_rupt2_norm_p9 < e_rupt1_norm_p9:
                        mom_win_p9 = p9_rupt_am[0]
                    else:
                        mom_win_p9 = plos_p9_end - frame_size_plos
                else:
                    mom_win_p9 = plos_p9_end - frame_size_plos

    t1_rupt_am = [
        x
        for x in t1_rupt
        if x >= (t1_mid_silence_end / samp_freq) and x <= plos_t1_end
    ]

    if not t1_rupt_am:
        mom_win_t1 = plos_t1_end - frame_size_plos
    else:
        t1_e_rupt1_beg = int(t1_rupt_am[0] * samp_freq)
        t1_e_rupt1_end = int((t1_rupt_am[0] + frame_size_plos) * samp_freq)
        rupt_part = parsound.extract_part(
            from_time=t1_e_rupt1_beg / samp_freq,
            to_time=t1_e_rupt1_end / samp_freq,
            preserve_times=True,
        )
        mean_e_rupt1_t1 = rupt_part.get_intensity()
        sum_e = 0
        for i in range(t1_e_rupt1_beg, t1_e_rupt1_end):
            sample = parsound.extract_part(
                from_time=i / samp_freq,
                to_time=(i + 1) / samp_freq,
                preserve_times=True,
            )
            samp_e = sample.get_energy()
            sum_e = (samp_e - mean_e_rupt1_t1) ** 2 + sum_e
        e_rupt1_t1 = sum_e
        e_rupt1_norm_t1 = e_rupt1_t1 / (samp_freq * 0.005)
        if e_rupt1_norm_t1 >= 2000:
            mom_win_t1 = t1_rupt_am[0]

        else:
            end_ruptures_t1 = [
                x
                for x in t1_rupt
                if (
                    x >= (plos_t1_end - 0.06)
                    and x > (t1_rupt_am[0])
                    and x < (plos_t1_end - 0.005)
                )
            ]
            if not end_ruptures_t1:
                mom_win_t1 = plos_t1_end - frame_size_plos
            else:
                for i in end_ruptures_t1:
                    t1_e_rupt_beg = int(i * samp_freq)
                    t1_e_rupt_end = int((i + frame_size_plos) * samp_freq)
                    rupt_part = parsound.extract_part(
                        from_time=t1_e_rupt_beg / samp_freq,
                        to_time=t1_e_rupt_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt_t1 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(t1_e_rupt_beg, t1_e_rupt_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt_t1) ** 2 + sum_e
                    e_rupt_t1 = sum_e
                    e_rupt_norm_t1 = e_rupt_t1 / (samp_freq * 0.005)
                    if (
                        e_rupt_norm_t1 >= e_rupt1_norm_t1
                        and e_rupt_norm_t1 >= silence_e_norm_t1
                    ) and e_rupt_norm_t1 < 3000:
                        mom_win_t1 = t1_e_rupt_beg / samp_freq
                        break
                if mom_win_t1 == 0 and len(end_ruptures_t1) > 0:
                    t1_e_rupt2_beg = int(end_ruptures_t1[0] * samp_freq)
                    t1_e_rupt2_end = int(
                        (end_ruptures_t1[0] + frame_size_plos) * samp_freq
                    )
                    rupt_part = parsound.extract_part(
                        from_time=t1_e_rupt2_beg / samp_freq,
                        to_time=t1_e_rupt2_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt2_t1 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(t1_e_rupt2_beg, t1_e_rupt2_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt2_t1) ** 2 + sum_e
                    e_rupt2_t1 = sum_e
                    e_rupt2_norm_t1 = e_rupt_t1 / (samp_freq * 0.005)

                    if e_rupt1_norm_t1 >= 1500 and (
                        (t1_e_rupt1_beg / samp_freq) >= (plos_t1_end - 0.06)
                        and (t1_e_rupt1_beg / samp_freq) < (plos_t1_end - 0.005)
                    ):
                        mom_win_t1 = t1_rupt_am[0]
                    elif e_rupt2_norm_t1 < e_rupt1_norm_t1:
                        mom_win_t1 = t1_rupt_am[0]
                    else:
                        mom_win_t1 = plos_t1_end - frame_size_plos
                else:
                    mom_win_t1 = plos_t1_end - frame_size_plos

    t2_rupt_am = [
        x
        for x in t2_rupt
        if x >= (t2_mid_silence_end / samp_freq) and x <= plos_t2_end
    ]

    if not t2_rupt_am:
        mom_win_t2 = plos_t2_end - frame_size_plos
    else:
        t2_e_rupt1_beg = int(t2_rupt_am[0] * samp_freq)
        t2_e_rupt1_end = int((t2_rupt_am[0] + frame_size_plos) * samp_freq)
        rupt_part = parsound.extract_part(
            from_time=t2_e_rupt1_beg / samp_freq,
            to_time=t2_e_rupt1_end / samp_freq,
            preserve_times=True,
        )
        mean_e_rupt1_t2 = rupt_part.get_intensity()
        sum_e = 0
        for i in range(t2_e_rupt1_beg, t2_e_rupt1_end):
            sample = parsound.extract_part(
                from_time=i / samp_freq,
                to_time=(i + 1) / samp_freq,
                preserve_times=True,
            )
            samp_e = sample.get_energy()
            sum_e = (samp_e - mean_e_rupt1_t2) ** 2 + sum_e
        e_rupt1_t2 = sum_e
        e_rupt1_norm_t2 = e_rupt1_t2 / (samp_freq * 0.005)
        if e_rupt1_norm_t2 >= 2000:
            mom_win_t2 = t2_rupt_am[0]

        else:
            end_ruptures_t2 = [
                x
                for x in t2_rupt
                if (
                    x >= (plos_t2_end - 0.06)
                    and x > (t2_rupt_am[0])
                    and x < (plos_t2_end - 0.005)
                )
            ]
            if not end_ruptures_t2:
                mom_win_t2 = plos_t2_end - frame_size_plos
            else:
                for i in end_ruptures_t2:
                    t2_e_rupt_beg = int(i * samp_freq)
                    t2_e_rupt_end = int((i + frame_size_plos) * samp_freq)
                    rupt_part = parsound.extract_part(
                        from_time=t2_e_rupt_beg / samp_freq,
                        to_time=t2_e_rupt_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt_t2 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(t2_e_rupt_beg, t2_e_rupt_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt_t2) ** 2 + sum_e
                    e_rupt_t2 = sum_e
                    e_rupt_norm_t2 = e_rupt_t2 / (samp_freq * 0.005)
                    if (
                        e_rupt_norm_t2 >= e_rupt1_norm_t2
                        and e_rupt_norm_t2 >= silence_e_norm_t2
                    ) and e_rupt_norm_t2 < 3000:
                        mom_win_t2 = t2_e_rupt_beg / samp_freq
                        break
                if mom_win_t2 == 0 and len(end_ruptures_t2) > 0:
                    t2_e_rupt2_beg = int(end_ruptures_t2[0] * samp_freq)
                    t2_e_rupt2_end = int(
                        (end_ruptures_t2[0] + frame_size_plos) * samp_freq
                    )
                    rupt_part = parsound.extract_part(
                        from_time=t2_e_rupt2_beg / samp_freq,
                        to_time=t2_e_rupt2_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt2_t2 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(t2_e_rupt2_beg, t2_e_rupt2_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt2_t2) ** 2 + sum_e
                    e_rupt2_t2 = sum_e
                    e_rupt2_norm_t2 = e_rupt_t2 / (samp_freq * 0.005)

                    if e_rupt1_norm_t2 >= 1500 and (
                        (t2_e_rupt1_beg / samp_freq) >= (plos_t2_end - 0.06)
                        and (t2_e_rupt1_beg / samp_freq) < (plos_t2_end - 0.005)
                    ):
                        mom_win_t2 = t2_rupt_am[0]
                    elif e_rupt2_norm_t2 < e_rupt1_norm_t2:
                        mom_win_t2 = t2_rupt_am[0]
                    else:
                        mom_win_t2 = plos_t2_end - frame_size_plos
                else:
                    mom_win_t2 = plos_t2_end - frame_size_plos

    t3_rupt_am = [
        x
        for x in t3_rupt
        if x >= (t3_mid_silence_end / samp_freq) and x <= plos_t3_end
    ]

    if not t3_rupt_am:
        mom_win_t3 = plos_t3_end - frame_size_plos
    else:
        t3_e_rupt1_beg = int(t3_rupt_am[0] * samp_freq)
        t3_e_rupt1_end = int((t3_rupt_am[0] + frame_size_plos) * samp_freq)
        rupt_part = parsound.extract_part(
            from_time=t3_e_rupt1_beg / samp_freq,
            to_time=t3_e_rupt1_end / samp_freq,
            preserve_times=True,
        )
        mean_e_rupt1_t3 = rupt_part.get_intensity()
        sum_e = 0
        for i in range(t3_e_rupt1_beg, t3_e_rupt1_end):
            sample = parsound.extract_part(
                from_time=i / samp_freq,
                to_time=(i + 1) / samp_freq,
                preserve_times=True,
            )
            samp_e = sample.get_energy()
            sum_e = (samp_e - mean_e_rupt1_t3) ** 2 + sum_e
        e_rupt1_t3 = sum_e
        e_rupt1_norm_t3 = e_rupt1_t3 / (samp_freq * 0.005)
        if e_rupt1_norm_t3 >= 2000:
            mom_win_t3 = t3_rupt_am[0]

        else:
            end_ruptures_t3 = [
                x
                for x in t3_rupt
                if (
                    x >= (plos_t3_end - 0.06)
                    and x > (t3_rupt_am[0])
                    and x < (plos_t3_end - 0.005)
                )
            ]
            if not end_ruptures_t3:
                mom_win_t3 = plos_t3_end - frame_size_plos
            else:
                for i in end_ruptures_t3:
                    t3_e_rupt_beg = int(i * samp_freq)
                    t3_e_rupt_end = int((i + frame_size_plos) * samp_freq)
                    rupt_part = parsound.extract_part(
                        from_time=t3_e_rupt_beg / samp_freq,
                        to_time=t3_e_rupt_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt_t3 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(t3_e_rupt_beg, t3_e_rupt_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt_t3) ** 2 + sum_e
                    e_rupt_t3 = sum_e
                    e_rupt_norm_t3 = e_rupt_t3 / (samp_freq * 0.005)
                    if (
                        e_rupt_norm_t3 >= e_rupt1_norm_t3
                        and e_rupt_norm_t3 >= silence_e_norm_t3
                    ) and e_rupt_norm_t3 < 3000:
                        mom_win_t3 = t3_e_rupt_beg / samp_freq
                        break
                if mom_win_t3 == 0 and len(end_ruptures_t3) > 0:
                    t3_e_rupt2_beg = int(end_ruptures_t3[0] * samp_freq)
                    t3_e_rupt2_end = int(
                        (end_ruptures_t3[0] + frame_size_plos) * samp_freq
                    )
                    rupt_part = parsound.extract_part(
                        from_time=t3_e_rupt2_beg / samp_freq,
                        to_time=t3_e_rupt2_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt2_t3 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(t3_e_rupt2_beg, t3_e_rupt2_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt2_t3) ** 2 + sum_e
                    e_rupt2_t3 = sum_e
                    e_rupt2_norm_t3 = e_rupt_t3 / (samp_freq * 0.005)

                    if e_rupt1_norm_t3 >= 1500 and (
                        (t3_e_rupt1_beg / samp_freq) >= (plos_t3_end - 0.06)
                        and (t3_e_rupt1_beg / samp_freq) < (plos_t3_end - 0.005)
                    ):
                        mom_win_t3 = t3_rupt_am[0]
                    elif e_rupt2_norm_t3 < e_rupt1_norm_t3:
                        mom_win_t3 = t3_rupt_am[0]
                    else:
                        mom_win_t3 = plos_t3_end - frame_size_plos
                else:
                    mom_win_t3 = plos_t3_end - frame_size_plos

    t4_rupt_am = [
        x
        for x in t4_rupt
        if x >= (t4_mid_silence_end / samp_freq) and x <= plos_t4_end
    ]

    if not t4_rupt_am:
        mom_win_t4 = plos_t4_end - frame_size_plos
    else:
        t4_e_rupt1_beg = int(t4_rupt_am[0] * samp_freq)
        t4_e_rupt1_end = int((t4_rupt_am[0] + frame_size_plos) * samp_freq)
        rupt_part = parsound.extract_part(
            from_time=t4_e_rupt1_beg / samp_freq,
            to_time=t4_e_rupt1_end / samp_freq,
            preserve_times=True,
        )
        mean_e_rupt1_t4 = rupt_part.get_intensity()
        sum_e = 0
        for i in range(t4_e_rupt1_beg, t4_e_rupt1_end):
            sample = parsound.extract_part(
                from_time=i / samp_freq,
                to_time=(i + 1) / samp_freq,
                preserve_times=True,
            )
            samp_e = sample.get_energy()
            sum_e = (samp_e - mean_e_rupt1_t4) ** 2 + sum_e
        e_rupt1_t4 = sum_e
        e_rupt1_norm_t4 = e_rupt1_t4 / (samp_freq * 0.005)
        if e_rupt1_norm_t4 >= 2000:
            mom_win_t4 = t4_rupt_am[0]

        else:
            end_ruptures_t4 = [
                x
                for x in t4_rupt
                if (
                    x >= (plos_t4_end - 0.06)
                    and x > (t4_rupt_am[0])
                    and x < (plos_t4_end - 0.005)
                )
            ]
            if not end_ruptures_t4:
                mom_win_t4 = plos_t4_end - frame_size_plos
            else:
                for i in end_ruptures_t4:
                    t4_e_rupt_beg = int(i * samp_freq)
                    t4_e_rupt_end = int((i + frame_size_plos) * samp_freq)
                    rupt_part = parsound.extract_part(
                        from_time=t4_e_rupt_beg / samp_freq,
                        to_time=t4_e_rupt_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt_t4 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(t4_e_rupt_beg, t4_e_rupt_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt_t4) ** 2 + sum_e
                    e_rupt_t4 = sum_e
                    e_rupt_norm_t4 = e_rupt_t4 / (samp_freq * 0.005)
                    if (
                        e_rupt_norm_t4 >= e_rupt1_norm_t4
                        and e_rupt_norm_t4 >= silence_e_norm_t4
                    ) and e_rupt_norm_t4 < 3000:
                        mom_win_t4 = t4_e_rupt_beg / samp_freq
                        break
                if mom_win_t4 == 0 and len(end_ruptures_t4) > 0:
                    t4_e_rupt2_beg = int(end_ruptures_t4[0] * samp_freq)
                    t4_e_rupt2_end = int(
                        (end_ruptures_t4[0] + frame_size_plos) * samp_freq
                    )
                    rupt_part = parsound.extract_part(
                        from_time=t4_e_rupt2_beg / samp_freq,
                        to_time=t4_e_rupt2_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt2_t4 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(t4_e_rupt2_beg, t4_e_rupt2_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt2_t4) ** 2 + sum_e
                    e_rupt2_t4 = sum_e
                    e_rupt2_norm_t4 = e_rupt_t4 / (samp_freq * 0.005)

                    if e_rupt1_norm_t4 >= 1500 and (
                        (t4_e_rupt1_beg / samp_freq) >= (plos_t4_end - 0.06)
                        and (t4_e_rupt1_beg / samp_freq) < (plos_t4_end - 0.005)
                    ):
                        mom_win_t4 = t4_rupt_am[0]
                    elif e_rupt2_norm_t4 < e_rupt1_norm_t4:
                        mom_win_t4 = t4_rupt_am[0]
                    else:
                        mom_win_t4 = plos_t4_end - frame_size_plos
                else:
                    mom_win_t4 = plos_t4_end - frame_size_plos

    k1_rupt_am = [
        x
        for x in k1_rupt
        if x >= (k1_mid_silence_end / samp_freq) and x <= plos_k1_end
    ]

    if not k1_rupt_am:
        mom_win_k1 = plos_k1_end - frame_size_plos
    else:
        k1_e_rupt1_beg = int(k1_rupt_am[0] * samp_freq)
        k1_e_rupt1_end = int((k1_rupt_am[0] + frame_size_plos) * samp_freq)
        rupt_part = parsound.extract_part(
            from_time=k1_e_rupt1_beg / samp_freq,
            to_time=k1_e_rupt1_end / samp_freq,
            preserve_times=True,
        )
        mean_e_rupt1_k1 = rupt_part.get_intensity()
        sum_e = 0
        for i in range(k1_e_rupt1_beg, k1_e_rupt1_end):
            sample = parsound.extract_part(
                from_time=i / samp_freq,
                to_time=(i + 1) / samp_freq,
                preserve_times=True,
            )
            samp_e = sample.get_energy()
            sum_e = (samp_e - mean_e_rupt1_k1) ** 2 + sum_e
        e_rupt1_k1 = sum_e
        e_rupt1_norm_k1 = e_rupt1_k1 / (samp_freq * 0.005)
        if e_rupt1_norm_k1 >= 2000:
            mom_win_k1 = k1_rupt_am[0]

        else:
            end_ruptures_k1 = [
                x
                for x in k1_rupt
                if (
                    x >= (plos_k1_end - 0.06)
                    and x > (k1_rupt_am[0])
                    and x < (plos_k1_end - 0.005)
                )
            ]
            if not end_ruptures_k1:
                mom_win_k1 = plos_k1_end - frame_size_plos
            else:
                for i in end_ruptures_k1:
                    k1_e_rupt_beg = int(i * samp_freq)
                    k1_e_rupt_end = int((i + frame_size_plos) * samp_freq)
                    rupt_part = parsound.extract_part(
                        from_time=k1_e_rupt_beg / samp_freq,
                        to_time=k1_e_rupt_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt_k1 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(k1_e_rupt_beg, k1_e_rupt_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt_k1) ** 2 + sum_e
                    e_rupt_k1 = sum_e
                    e_rupt_norm_k1 = e_rupt_k1 / (samp_freq * 0.005)
                    if (
                        e_rupt_norm_k1 >= e_rupt1_norm_k1
                        and e_rupt_norm_k1 >= silence_e_norm_k1
                    ) and e_rupt_norm_k1 < 3000:
                        mom_win_k1 = k1_e_rupt_beg / samp_freq
                        break
                if mom_win_k1 == 0 and len(end_ruptures_k1) > 0:
                    k1_e_rupt2_beg = int(end_ruptures_k1[0] * samp_freq)
                    k1_e_rupt2_end = int(
                        (end_ruptures_k1[0] + frame_size_plos) * samp_freq
                    )
                    rupt_part = parsound.extract_part(
                        from_time=k1_e_rupt2_beg / samp_freq,
                        to_time=k1_e_rupt2_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt2_k1 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(k1_e_rupt2_beg, k1_e_rupt2_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt2_k1) ** 2 + sum_e
                    e_rupt2_k1 = sum_e
                    e_rupt2_norm_k1 = e_rupt_k1 / (samp_freq * 0.005)

                    if e_rupt1_norm_k1 >= 1500 and (
                        (k1_e_rupt1_beg / samp_freq) >= (plos_k1_end - 0.06)
                        and (k1_e_rupt1_beg / samp_freq) < (plos_k1_end - 0.005)
                    ):
                        mom_win_k1 = k1_rupt_am[0]
                    elif e_rupt2_norm_k1 < e_rupt1_norm_k1:
                        mom_win_k1 = k1_rupt_am[0]
                    else:
                        mom_win_k1 = plos_k1_end - frame_size_plos
                else:
                    mom_win_k1 = plos_k1_end - frame_size_plos

    k2_rupt_am = [
        x
        for x in k2_rupt
        if x >= (k2_mid_silence_end / samp_freq) and x <= plos_k2_end
    ]

    if not k2_rupt_am:
        mom_win_k2 = plos_k2_end - frame_size_plos
    else:
        k2_e_rupt1_beg = int(k2_rupt_am[0] * samp_freq)
        k2_e_rupt1_end = int((k2_rupt_am[0] + frame_size_plos) * samp_freq)
        rupt_part = parsound.extract_part(
            from_time=k2_e_rupt1_beg / samp_freq,
            to_time=k2_e_rupt1_end / samp_freq,
            preserve_times=True,
        )
        mean_e_rupt1_k2 = rupt_part.get_intensity()
        sum_e = 0
        for i in range(k2_e_rupt1_beg, k2_e_rupt1_end):
            sample = parsound.extract_part(
                from_time=i / samp_freq,
                to_time=(i + 1) / samp_freq,
                preserve_times=True,
            )
            samp_e = sample.get_energy()
            sum_e = (samp_e - mean_e_rupt1_k2) ** 2 + sum_e
        e_rupt1_k2 = sum_e
        e_rupt1_norm_k2 = e_rupt1_k2 / (samp_freq * 0.005)
        if e_rupt1_norm_k2 >= 2000:
            mom_win_k2 = k2_rupt_am[0]

        else:
            end_ruptures_k2 = [
                x
                for x in k2_rupt
                if (
                    x >= (plos_k2_end - 0.06)
                    and x > (k2_rupt_am[0])
                    and x < (plos_k2_end - 0.005)
                )
            ]
            if not end_ruptures_k2:
                mom_win_k2 = plos_k2_end - frame_size_plos
            else:
                for i in end_ruptures_k2:
                    k2_e_rupt_beg = int(i * samp_freq)
                    k2_e_rupt_end = int((i + frame_size_plos) * samp_freq)
                    rupt_part = parsound.extract_part(
                        from_time=k2_e_rupt_beg / samp_freq,
                        to_time=k2_e_rupt_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt_k2 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(k2_e_rupt_beg, k2_e_rupt_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt_k2) ** 2 + sum_e
                    e_rupt_k2 = sum_e
                    e_rupt_norm_k2 = e_rupt_k2 / (samp_freq * 0.005)
                    if (
                        e_rupt_norm_k2 >= e_rupt1_norm_k2
                        and e_rupt_norm_k2 >= silence_e_norm_k2
                    ) and e_rupt_norm_k2 < 3000:
                        mom_win_k2 = k2_e_rupt_beg / samp_freq
                        break
                if mom_win_k2 == 0 and len(end_ruptures_k2) > 0:
                    k2_e_rupt2_beg = int(end_ruptures_k2[0] * samp_freq)
                    k2_e_rupt2_end = int(
                        (end_ruptures_k2[0] + frame_size_plos) * samp_freq
                    )
                    rupt_part = parsound.extract_part(
                        from_time=k2_e_rupt2_beg / samp_freq,
                        to_time=k2_e_rupt2_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt2_k2 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(k2_e_rupt2_beg, k2_e_rupt2_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt2_k2) ** 2 + sum_e
                    e_rupt2_k2 = sum_e
                    e_rupt2_norm_k2 = e_rupt_k2 / (samp_freq * 0.005)

                    if e_rupt1_norm_k2 >= 1500 and (
                        (k2_e_rupt1_beg / samp_freq) >= (plos_k2_end - 0.06)
                        and (k2_e_rupt1_beg / samp_freq) < (plos_k2_end - 0.005)
                    ):
                        mom_win_k2 = k2_rupt_am[0]
                    elif e_rupt2_norm_k2 < e_rupt1_norm_k2:
                        mom_win_k2 = k2_rupt_am[0]
                    else:
                        mom_win_k2 = plos_k2_end - frame_size_plos
                else:
                    mom_win_k2 = plos_k2_end - frame_size_plos

    k3_rupt_am = [
        x
        for x in k3_rupt
        if x >= (k3_mid_silence_end / samp_freq) and x <= plos_k3_end
    ]

    if not k3_rupt_am:
        mom_win_k3 = plos_k3_end - frame_size_plos
    else:
        k3_e_rupt1_beg = int(k3_rupt_am[0] * samp_freq)
        k3_e_rupt1_end = int((k3_rupt_am[0] + frame_size_plos) * samp_freq)
        rupt_part = parsound.extract_part(
            from_time=k3_e_rupt1_beg / samp_freq,
            to_time=k3_e_rupt1_end / samp_freq,
            preserve_times=True,
        )
        mean_e_rupt1_k3 = rupt_part.get_intensity()
        sum_e = 0
        for i in range(k3_e_rupt1_beg, k3_e_rupt1_end):
            sample = parsound.extract_part(
                from_time=i / samp_freq,
                to_time=(i + 1) / samp_freq,
                preserve_times=True,
            )
            samp_e = sample.get_energy()
            sum_e = (samp_e - mean_e_rupt1_k3) ** 2 + sum_e
        e_rupt1_k3 = sum_e
        e_rupt1_norm_k3 = e_rupt1_k3 / (samp_freq * 0.005)
        if e_rupt1_norm_k3 >= 2000:
            mom_win_k3 = k3_rupt_am[0]

        else:
            end_ruptures_k3 = [
                x
                for x in k3_rupt
                if (
                    x >= (plos_k3_end - 0.06)
                    and x > (k3_rupt_am[0])
                    and x < (plos_k3_end - 0.005)
                )
            ]
            if not end_ruptures_k3:
                mom_win_k3 = plos_k3_end - frame_size_plos
            else:
                for i in end_ruptures_k3:
                    k3_e_rupt_beg = int(i * samp_freq)
                    k3_e_rupt_end = int((i + frame_size_plos) * samp_freq)
                    rupt_part = parsound.extract_part(
                        from_time=k3_e_rupt_beg / samp_freq,
                        to_time=k3_e_rupt_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt_k3 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(k3_e_rupt_beg, k3_e_rupt_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt_k3) ** 2 + sum_e
                    e_rupt_k3 = sum_e
                    e_rupt_norm_k3 = e_rupt_k3 / (samp_freq * 0.005)
                    if (
                        e_rupt_norm_k3 >= e_rupt1_norm_k3
                        and e_rupt_norm_k3 >= silence_e_norm_k3
                    ) and e_rupt_norm_k3 < 3000:
                        mom_win_k3 = k3_e_rupt_beg / samp_freq
                        break
                if mom_win_k3 == 0 and len(end_ruptures_k3) > 0:
                    k3_e_rupt2_beg = int(end_ruptures_k3[0] * samp_freq)
                    k3_e_rupt2_end = int(
                        (end_ruptures_k3[0] + frame_size_plos) * samp_freq
                    )
                    rupt_part = parsound.extract_part(
                        from_time=k3_e_rupt2_beg / samp_freq,
                        to_time=k3_e_rupt2_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt2_k3 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(k3_e_rupt2_beg, k3_e_rupt2_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt2_k3) ** 2 + sum_e
                    e_rupt2_k3 = sum_e
                    e_rupt2_norm_k3 = e_rupt_k3 / (samp_freq * 0.005)

                    if e_rupt1_norm_k3 >= 1500 and (
                        (k3_e_rupt1_beg / samp_freq) >= (plos_k3_end - 0.06)
                        and (k3_e_rupt1_beg / samp_freq) < (plos_k3_end - 0.005)
                    ):
                        mom_win_k3 = k3_rupt_am[0]
                    elif e_rupt2_norm_k3 < e_rupt1_norm_k3:
                        mom_win_k3 = k3_rupt_am[0]
                    else:
                        mom_win_k3 = plos_k3_end - frame_size_plos
                else:
                    mom_win_k3 = plos_k3_end - frame_size_plos

    k4_rupt_am = [
        x
        for x in k4_rupt
        if x >= (k4_mid_silence_end / samp_freq) and x <= plos_k4_end
    ]

    if not k4_rupt_am:
        mom_win_k4 = plos_k4_end - frame_size_plos
    else:
        k4_e_rupt1_beg = int(k4_rupt_am[0] * samp_freq)
        k4_e_rupt1_end = int((k4_rupt_am[0] + frame_size_plos) * samp_freq)
        rupt_part = parsound.extract_part(
            from_time=k4_e_rupt1_beg / samp_freq,
            to_time=k4_e_rupt1_end / samp_freq,
            preserve_times=True,
        )
        mean_e_rupt1_k4 = rupt_part.get_intensity()
        sum_e = 0
        for i in range(k4_e_rupt1_beg, k4_e_rupt1_end):
            sample = parsound.extract_part(
                from_time=i / samp_freq,
                to_time=(i + 1) / samp_freq,
                preserve_times=True,
            )
            samp_e = sample.get_energy()
            sum_e = (samp_e - mean_e_rupt1_k4) ** 2 + sum_e
        e_rupt1_k4 = sum_e
        e_rupt1_norm_k4 = e_rupt1_k4 / (samp_freq * 0.005)
        if e_rupt1_norm_k4 >= 2000:
            mom_win_k4 = k4_rupt_am[0]

        else:
            end_ruptures_k4 = [
                x
                for x in k4_rupt
                if (
                    x >= (plos_k4_end - 0.06)
                    and x > (k4_rupt_am[0])
                    and x < (plos_k4_end - 0.005)
                )
            ]
            if not end_ruptures_k4:
                mom_win_k4 = plos_k4_end - frame_size_plos
            else:
                for i in end_ruptures_k4:
                    k4_e_rupt_beg = int(i * samp_freq)
                    k4_e_rupt_end = int((i + frame_size_plos) * samp_freq)
                    rupt_part = parsound.extract_part(
                        from_time=k4_e_rupt_beg / samp_freq,
                        to_time=k4_e_rupt_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt_k4 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(k4_e_rupt_beg, k4_e_rupt_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt_k4) ** 2 + sum_e
                    e_rupt_k4 = sum_e
                    e_rupt_norm_k4 = e_rupt_k4 / (samp_freq * 0.005)
                    if (
                        e_rupt_norm_k4 >= e_rupt1_norm_k4
                        and e_rupt_norm_k4 >= silence_e_norm_k4
                    ) and e_rupt_norm_k4 < 3000:
                        mom_win_k4 = k4_e_rupt_beg / samp_freq
                        break
                if mom_win_k4 == 0 and len(end_ruptures_k4) > 0:
                    k4_e_rupt2_beg = int(end_ruptures_k4[0] * samp_freq)
                    k4_e_rupt2_end = int(
                        (end_ruptures_k4[0] + frame_size_plos) * samp_freq
                    )
                    rupt_part = parsound.extract_part(
                        from_time=k4_e_rupt2_beg / samp_freq,
                        to_time=k4_e_rupt2_end / samp_freq,
                        preserve_times=True,
                    )
                    mean_e_rupt2_k4 = rupt_part.get_intensity()
                    sum_e = 0
                    for i in range(k4_e_rupt2_beg, k4_e_rupt2_end):
                        sample = parsound.extract_part(
                            from_time=i / samp_freq,
                            to_time=(i + 1) / samp_freq,
                            preserve_times=True,
                        )
                        samp_e = sample.get_energy()
                        sum_e = (samp_e - mean_e_rupt2_k4) ** 2 + sum_e
                    e_rupt2_k4 = sum_e
                    e_rupt2_norm_k4 = e_rupt_k4 / (samp_freq * 0.005)

                    if e_rupt1_norm_k4 >= 1500 and (
                        (k4_e_rupt1_beg / samp_freq) >= (plos_k4_end - 0.06)
                        and (k4_e_rupt1_beg / samp_freq) < (plos_k4_end - 0.005)
                    ):
                        mom_win_k4 = k4_rupt_am[0]
                    elif e_rupt2_norm_k4 < e_rupt1_norm_k4:
                        mom_win_k4 = k4_rupt_am[0]
                    else:
                        mom_win_k4 = plos_k4_end - frame_size_plos
                else:
                    mom_win_k4 = plos_k4_end - frame_size_plos

    # FOR VOICED PLOSIVES
    mom_win_b = 0
    mom_win_d = 0
    mom_win_g1 = 0
    mom_win_g2 = 0

    # take the nearest rupture at 20ms prior to the right border of the forced alignment segment (if the two last ruptures are closer than framesize, take the earliest one)
    # if there is no rupture in the 20ms prior to the right border, take the nearest border at 20ms after the border

    if not b_rupt_end:
        with open(resFile, "a") as result_file:
            print(
                "\nWARNING: There is no rupture 20ms prior or after the right border of the forced alignment segment for [b].\nTook end of consonant segment - window size ("
                + str(frame_size_plos)
                + "ms) as analysis window.",
                file=result_file,
            )
        mom_win_b = plos_b_end - frame_size_plos
    elif len(b_rupt_end_inf) == 1:
        mom_win_b = b_rupt_end_inf[-1]
    elif len(b_rupt_end_inf) > 1:
        if b_rupt_end_inf[-1] - b_rupt_end_inf[-2] > frame_size_plos * 2:
            mom_win_b = b_rupt_end_inf[-1]
        elif b_rupt_end_inf[-1] - b_rupt_end_inf[-2] <= frame_size_plos * 2:
            mom_win_b = b_rupt_end_inf[-2]
    elif len(b_rupt_end_inf) == 0 and len(b_rupt_end_sup) > 0:
        mom_win_b = b_rupt_end_sup[0]

    if not d_rupt_end:
        with open(resFile, "w") as result_file:
            print(
                "\nWARNING: There is no rupture 20ms prior or after the right border of the forced alignment segment for [d].\nTook end of consonant segment - window size ("
                + str(frame_size_plos)
                + "ms) as analysis window.",
                file=result_file,
            )
        mom_win_d = plos_d_end - frame_size_plos
    elif len(d_rupt_end_inf) == 1:
        mom_win_d = d_rupt_end_inf[-1]
    elif len(d_rupt_end_inf) > 1:
        if d_rupt_end_inf[-1] - d_rupt_end_inf[-2] > frame_size_plos * 2:
            mom_win_d = d_rupt_end_inf[-1]
        elif d_rupt_end_inf[-1] - d_rupt_end_inf[-2] <= frame_size_plos * 2:
            mom_win_d = d_rupt_end_inf[-2]
    elif len(d_rupt_end_inf) == 0 and len(d_rupt_end_sup) > 0:
        mom_win_d = d_rupt_end_sup[0]

    if not g1_rupt_end:
        with open(resFile, "a") as result_file:
            print(
                "\nWARNING: There is no rupture 20ms prior or after the right border of the forced alignment segment for [g1].\nTook end of consonant segment - window size ("
                + str(frame_size_plos)
                + "ms) as analysis window.",
                file=result_file,
            )
        mom_win_g1 = plos_g1_end - frame_size_plos
    elif len(g1_rupt_end_inf) == 1:
        mom_win_g1 = g1_rupt_end_inf[-1]
    elif len(g1_rupt_end_inf) > 1:
        if g1_rupt_end_inf[-1] - g1_rupt_end_inf[-2] > frame_size_plos * 2:
            mom_win_g1 = g1_rupt_end_inf[-1]
        elif g1_rupt_end_inf[-1] - g1_rupt_end_inf[-2] <= frame_size_plos * 2:
            mom_win_g1 = g1_rupt_end_inf[-2]
    elif len(g1_rupt_end_inf) == 0 and len(g1_rupt_end_sup) > 0:
        mom_win_g1 = g1_rupt_end_sup[0]

    if not g2_rupt_end:
        with open(resFile, "a") as result_file:
            print(
                "\nWARNING: There is no rupture 20ms prior or after the right border of the forced alignment segment for [g2].\nTook end of consonant segment - window size ("
                + str(frame_size_plos)
                + "ms) as analysis window.",
                file=result_file,
            )
        mom_win_g2 = plos_g2_end - frame_size_plos
    elif len(g2_rupt_end_inf) == 1:
        mom_win_g2 = g2_rupt_end_inf[-1]
    elif len(g2_rupt_end_inf) > 1:
        if g2_rupt_end_inf[-1] - g2_rupt_end_inf[-2] > frame_size_plos * 2:
            mom_win_g2 = g2_rupt_end_inf[-1]
        elif g2_rupt_end_inf[-1] - g2_rupt_end_inf[-2] <= frame_size_plos * 2:
            mom_win_g2 = g2_rupt_end_inf[-2]
    elif len(g2_rupt_end_inf) == 0 and len(g2_rupt_end_sup) > 0:
        mom_win_g2 = g2_rupt_end_sup[0]

    # FOR FRICATIVES

    mom_win_f = 0
    mom_win_s = 0
    mom_win_ch = 0
    mom_win_v1 = 0
    mom_win_v2 = 0
    mom_win_z = 0
    mom_win_j = 0

    # if there are not at least 2 ruptures inside of the consonant segment, then the midpoint of the segment is the analysis window location
    # FOR UNVOICED FRICATIVES
    # if there are >=2 ruptures,take the longest segment inside of the forced alignment segment + tolerance window of 10ms
    # calculate the distances between ruptures, keep the highest distance
    # the analysis window will then be placed at midpoint of this segment
    # FOR VOICED FRICATIVES
    # if there are >=2 ruptures,search for the segment that is at least 20ms long and has the lowest normalized energy inside of the forced alignment segment + tolerance window of 10ms
    # the analysis window will then be placed at midpoint of this segment

    if len(f_rupt) < 2:
        mom_win_f = (
            float(fric_f[0])
            + ((float(fric_f[2]) - float(fric_f[0])) / 2)
            - frame_size_fric / 2
        )
        with open(resFile, "a") as result_file:
            print(
                "\nWARNING: There are not at least two ruptures inside of the [f] segment.\nTook the midpoint of the segment as analysis window location",
                file=result_file,
            )
    elif (
        len(f_rupt) == 2
        and (float(f_rupt[1]) - float(f_rupt[0])) < frame_size_fric
    ):
        mom_win_f = (
            float(fric_f[0])
            + ((float(fric_f[2]) - float(fric_f[0])) / 2)
            - frame_size_fric / 2
        )
        with open(resFile, "a") as result_file:
            print(
                "\nWARNING: There are two ruptures inside of the [f] segment, but they are too close to be considered as separate ruptures (< frame size).\nTook the midpoint of the segment as analysis window location",
                file=result_file,
            )
    else:
        i = 0
        l = len(f_rupt)
        max_seg_f = 0
        while i < l - 1:
            if float(f_rupt[i + 1]) - float(f_rupt[i]) > max_seg_f:
                max_seg_f = float(f_rupt[i + 1]) - float(f_rupt[i])
                mom_win_f = (
                    float(f_rupt[i])
                    + ((float(f_rupt[i + 1]) - float(f_rupt[i])) / 2)
                    - frame_size_fric / 2
                )
            i = i + 1

    if len(s_rupt) < 2:
        mom_win_s = (
            float(fric_s[0])
            + ((float(fric_s[2]) - float(fric_s[0])) / 2)
            - frame_size_fric / 2
        )
        with open(resFile, "a") as result_file:
            print(
                "\nWARNING: There are not at least two ruptures inside of the [s] segment.\nTook the midpoint of the segment as analysis window location",
                file=result_file,
            )
    elif (
        len(s_rupt) == 2
        and (float(s_rupt[1]) - float(s_rupt[0])) < frame_size_fric
    ):
        mom_win_s = (
            float(fric_s[0])
            + ((float(fric_s[2]) - float(fric_s[0])) / 2)
            - frame_size_fric / 2
        )
        with open(resFile, "a") as result_file:
            print(
                "\nWARNING: There are two ruptures inside of the [s] segment, but they are too close to be considered as separate ruptures (< frame size).\nTook the midpoint of the segment as analysis window location",
                file=result_file,
            )
    else:
        i = 0
        l = len(s_rupt)
        max_seg_s = 0
        while i < l - 1:
            if float(s_rupt[i + 1]) - float(s_rupt[i]) > max_seg_s:
                max_seg_s = float(s_rupt[i + 1]) - float(s_rupt[i])
                mom_win_s = (
                    float(s_rupt[i])
                    + ((float(s_rupt[i + 1]) - float(s_rupt[i])) / 2)
                    - frame_size_fric / 2
                )
            i = i + 1

    if len(ch_rupt) < 2:
        mom_win_ch = (
            float(fric_ch[0])
            + ((float(fric_ch[2]) - float(fric_ch[0])) / 2)
            - frame_size_fric / 2
        )
        with open(resFile, "a") as result_file:
            print(
                "\nWARNING: There are not at least two ruptures inside of the [ch] segment.\nTook the midpoint of the segment as analysis window location",
                file=result_file,
            )
    elif (
        len(ch_rupt) == 2
        and (float(ch_rupt[1]) - float(ch_rupt[0])) < frame_size_fric
    ):
        mom_win_ch = (
            float(fric_ch[0])
            + ((float(fric_ch[2]) - float(fric_ch[0])) / 2)
            - frame_size_fric / 2
        )
        with open(resFile, "a") as result_file:
            print(
                "\nWARNING: There are two ruptures inside of the [ch] segment, but they are too close to be considered as separate ruptures (< frame size).\nTook the midpoint of the segment as analysis window location",
                file=result_file,
            )
    else:
        i = 0
        l = len(ch_rupt)
        max_seg_ch = 0
        while i < l - 1:
            if float(ch_rupt[i + 1]) - float(ch_rupt[i]) > max_seg_ch:
                max_seg_ch = float(ch_rupt[i + 1]) - float(ch_rupt[i])
                mom_win_ch = (
                    float(ch_rupt[i])
                    + ((float(ch_rupt[i + 1]) - float(ch_rupt[i])) / 2)
                    - frame_size_fric / 2
                )
            i = i + 1

    if len(v1_rupt) < 2:
        mom_win_v1 = (
            float(fric_v1.iloc[0])
            + ((float(fric_v1.iloc[2]) - float(fric_v1.iloc[0])) / 2)
            - frame_size_fric / 2
        )
        with open(resFile, "a") as result_file:
            print(
                "\nWARNING: There are not at least two ruptures inside of the [v1] segment.\nTook the midpoint of the segment as analysis window location",
                file=result_file,
            )
    elif (
        len(v1_rupt) == 2
        and (float(v1_rupt[1]) - float(v1_rupt[0])) < frame_size_fric
    ):
        mom_win_v1 = (
            float(fric_v1.iloc[0])
            + ((float(fric_v1.iloc[2]) - float(fric_v1.iloc[0])) / 2)
            - frame_size_fric / 2
        )
        with open(resFile, "a") as result_file:
            print(
                "\nWARNING: There are two ruptures inside of the [v1] segment, but they are too close to be considered as separate ruptures (< frame size).\nTook the midpoint of the segment as analysis window location",
                file=result_file,
            )
    elif (
        len(v1_rupt) == 2
        and (float(v1_rupt[1]) - float(v1_rupt[0])) > frame_size_fric
    ):
        min_e_seg = sys.maxsize
        if (float(v1_rupt[1]) - float(v1_rupt[0])) >= 0.02:
            rupt_beg = v1_rupt[0] * samp_freq
            rupt_end = v1_rupt[1] * samp_freq
            mean_e_rupt = 0
            mean_e_rupt = numpy.mean(y[int(rupt_beg) : int(rupt_end) + 1])
            sum_e = 0
            for j in range(int(rupt_beg), int(rupt_end)):
                sum_e = (y[j] - mean_e_rupt) ** 2 + sum_e
            sum_e_norm = sum_e / (len(range(int(rupt_beg), int(rupt_end))))
            if sum_e_norm < min_e_seg:
                min_e_seg = sum_e_norm
                mom_win_v1 = (
                    v1_rupt[0]
                    + ((v1_rupt[1] - v1_rupt[0]) / 2)
                    - frame_size_fric / 2
                )
        elif (float(v1_rupt[1]) - float(v1_rupt[0])) < 0.02:
            with open(resFile, "a") as result_file:
                print(
                    "\nWARNING: The longest stable part inside of the [v1] segment is shorter than 20ms.",
                    file=result_file,
                )
            rupt_beg = v1_rupt[0] * samp_freq
            rupt_end = v1_rupt[1] * samp_freq
            mean_e_rupt = 0
            mean_e_rupt = numpy.mean(y[int(rupt_beg) : int(rupt_end) + 1])
            sum_e = 0
            for j in range(int(rupt_beg), int(rupt_end)):
                sum_e = (y[j] - mean_e_rupt) ** 2 + sum_e
            sum_e_norm = sum_e / (len(range(int(rupt_beg), int(rupt_end))))
            if sum_e_norm < min_e_seg:
                min_e_seg = sum_e_norm
                mom_win_v1 = (
                    v1_rupt[0]
                    + ((v1_rupt[1] - v1_rupt[0]) / 2)
                    - frame_size_fric / 2
                )
    else:
        l = len(v1_rupt)
        i = 0
        min_e_seg = sys.maxsize
        while i < l - 1:
            if (float(v1_rupt[i + 1]) - float(v1_rupt[i])) >= 0.02:
                rupt_beg = v1_rupt[i] * samp_freq
                rupt_end = v1_rupt[i + 1] * samp_freq
                mean_e_rupt = 0
                mean_e_rupt = numpy.mean(y[int(rupt_beg) : int(rupt_end) + 1])
                sum_e = 0
                for j in range(int(rupt_beg), int(rupt_end)):
                    sum_e = (y[j] - mean_e_rupt) ** 2 + sum_e
                sum_e_norm = sum_e / (len(range(int(rupt_beg), int(rupt_end))))
                if sum_e_norm < min_e_seg:
                    min_e_seg = sum_e_norm
                    mom_win_v1 = (
                        v1_rupt[i]
                        + ((v1_rupt[i + 1] - v1_rupt[i]) / 2)
                        - frame_size_fric / 2
                    )
            i = i + 1

    if len(v2_rupt) < 2:
        mom_win_v2 = (
            float(fric_v2.iloc[0])
            + ((float(fric_v2.iloc[2]) - float(fric_v2.iloc[0])) / 2)
            - frame_size_fric / 2
        )
        with open(resFile, "a") as result_file:
            print(
                "\nWARNING: There are not at least two ruptures inside of the [v2] segment.\nTook the midpoint of the segment as analysis window location",
                file=result_file,
            )
    elif (
        len(v2_rupt) == 2
        and (float(v2_rupt[1]) - float(v2_rupt[0])) < frame_size_fric
    ):
        mom_win_v2 = (
            float(fric_v2[0])
            + ((float(fric_v2[2]) - float(fric_v2[0])) / 2)
            - frame_size_fric / 2
        )
        with open(resFile, "a") as result_file:
            print(
                "\nWARNING: There are two ruptures inside of the [v2] segment, but they are too close to be considered as separate ruptures (< frame size).\nTook the midpoint of the segment as analysis window location",
                file=result_file,
            )

    elif (
        len(v2_rupt) == 2
        and (float(v2_rupt[1]) - float(v2_rupt[0])) > frame_size_fric
    ):
        min_e_seg = sys.maxsize
        if (float(v2_rupt[1]) - float(v2_rupt[0])) >= 0.02:
            rupt_beg = v2_rupt[0] * samp_freq
            rupt_end = v2_rupt[1] * samp_freq
            mean_e_rupt = 0
            mean_e_rupt = numpy.mean(y[int(rupt_beg) : int(rupt_end) + 1])
            sum_e = 0
            for j in range(int(rupt_beg), int(rupt_end)):
                sum_e = (y[j] - mean_e_rupt) ** 2 + sum_e
            sum_e_norm = sum_e / (len(range(int(rupt_beg), int(rupt_end))))
            if sum_e_norm < min_e_seg:
                min_e_seg = sum_e_norm
                mom_win_v2 = (
                    v2_rupt[0]
                    + ((v2_rupt[1] - v2_rupt[0]) / 2)
                    - frame_size_fric / 2
                )
        elif (float(v2_rupt[1]) - float(v2_rupt[0])) < 0.02:
            with open(resFile, "a") as result_file:
                print(
                    "\nWARNING: The longest stable part inside of the [v2] segment is shorter than 20ms.",
                    file=result_file,
                )
            rupt_beg = v2_rupt[0] * samp_freq
            rupt_end = v2_rupt[1] * samp_freq
            mean_e_rupt = 0
            mean_e_rupt = numpy.mean(y[int(rupt_beg) : int(rupt_end) + 1])
            sum_e = 0
            for j in range(int(rupt_beg), int(rupt_end)):
                sum_e = (y[j] - mean_e_rupt) ** 2 + sum_e
            sum_e_norm = sum_e / (len(range(int(rupt_beg), int(rupt_end))))
            if sum_e_norm < min_e_seg:
                min_e_seg = sum_e_norm
                mom_win_v2 = (
                    v2_rupt[0]
                    + ((v2_rupt[1] - v2_rupt[0]) / 2)
                    - frame_size_fric / 2
                )
    else:
        l = len(v2_rupt)
        i = 0
        min_e_seg = sys.maxsize
        while i < l - 1:
            if (float(v2_rupt[i + 1]) - float(v2_rupt[i])) >= 0.02:
                rupt_beg = v2_rupt[i] * samp_freq
                rupt_end = v2_rupt[i + 1] * samp_freq
                mean_e_rupt = 0
                mean_e_rupt = numpy.mean(y[int(rupt_beg) : int(rupt_end) + 1])
                sum_e = 0
                for j in range(int(rupt_beg), int(rupt_end)):
                    sum_e = (y[j] - mean_e_rupt) ** 2 + sum_e
                sum_e_norm = sum_e / (len(range(int(rupt_beg), int(rupt_end))))
                if sum_e_norm < min_e_seg:
                    min_e_seg = sum_e_norm
                    mom_win_v2 = (
                        v2_rupt[i]
                        + ((v2_rupt[i + 1] - v2_rupt[i]) / 2)
                        - frame_size_fric / 2
                    )
            i = i + 1

    if len(z_rupt) < 2:
        mom_win_z = (
            float(fric_z[0])
            + ((float(fric_z[2]) - float(fric_z[0])) / 2)
            - frame_size_fric / 2
        )
        with open(resFile, "a") as result_file:
            print(
                "\nWARNING: There are not at least two ruptures inside of the [z] segment.\nTook the midpoint of the segment as analysis window location",
                file=result_file,
            )

    elif (
        len(z_rupt) == 2
        and (float(z_rupt[1]) - float(z_rupt[0])) < frame_size_fric
    ):
        mom_win_z = (
            float(fric_z[0])
            + ((float(fric_z[2]) - float(fric_z[0])) / 2)
            - frame_size_fric / 2
        )
        with open(resFile, "a") as result_file:
            print(
                "\nWARNING: There are two ruptures inside of the [z] segment, but they are too close to be considered as separate ruptures (< frame size).\nTook the midpoint of the segment as analysis window location",
                file=result_file,
            )

    elif (
        len(z_rupt) == 2
        and (float(z_rupt[1]) - float(z_rupt[0])) > frame_size_fric
    ):
        min_e_seg = sys.maxsize
        if (float(z_rupt[1]) - float(z_rupt[0])) >= 0.02:
            rupt_beg = z_rupt[0] * samp_freq
            rupt_end = z_rupt[1] * samp_freq
            mean_e_rupt = 0
            mean_e_rupt = numpy.mean(y[int(rupt_beg) : int(rupt_end) + 1])
            sum_e = 0
            for j in range(int(rupt_beg), int(rupt_end)):
                sum_e = (y[j] - mean_e_rupt) ** 2 + sum_e
            sum_e_norm = sum_e / (len(range(int(rupt_beg), int(rupt_end))))
            if sum_e_norm < min_e_seg:
                min_e_seg = sum_e_norm
                mom_win_z = (
                    z_rupt[0]
                    + ((z_rupt[1] - z_rupt[0]) / 2)
                    - frame_size_fric / 2
                )
        elif (float(z_rupt[1]) - float(z_rupt[0])) < 0.02:
            with open(resFile, "a") as result_file:
                print(
                    "\nWARNING: The longest stable part inside of the [z] segment is shorter than 20ms.",
                    file=result_file,
                )
            rupt_beg = z_rupt[0] * samp_freq
            rupt_end = z_rupt[1] * samp_freq
            mean_e_rupt = 0
            mean_e_rupt = numpy.mean(y[int(rupt_beg) : int(rupt_end) + 1])
            sum_e = 0
            for j in range(int(rupt_beg), int(rupt_end)):
                sum_e = (y[j] - mean_e_rupt) ** 2 + sum_e
            sum_e_norm = sum_e / (len(range(int(rupt_beg), int(rupt_end))))
            if sum_e_norm < min_e_seg:
                min_e_seg = sum_e_norm
                mom_win_z = (
                    z_rupt[0]
                    + ((z_rupt[1] - z_rupt[0]) / 2)
                    - frame_size_fric / 2
                )
    else:
        l = len(z_rupt)
        i = 0
        min_e_seg = sys.maxsize
        while i < l - 1:
            if (float(z_rupt[i + 1]) - float(z_rupt[i])) >= 0.02:
                rupt_beg = z_rupt[i] * samp_freq
                rupt_end = z_rupt[i + 1] * samp_freq
                mean_e_rupt = 0
                mean_e_rupt = numpy.mean(y[int(rupt_beg) : int(rupt_end) + 1])
                sum_e = 0
                for j in range(int(rupt_beg), int(rupt_end)):
                    sum_e = (y[j] - mean_e_rupt) ** 2 + sum_e
                sum_e_norm = sum_e / (len(range(int(rupt_beg), int(rupt_end))))
                if sum_e_norm < min_e_seg:
                    min_e_seg = sum_e_norm
                    mom_win_z = (
                        z_rupt[i]
                        + ((z_rupt[i + 1] - z_rupt[i]) / 2)
                        - frame_size_fric / 2
                    )
            i = i + 1

    if len(j_rupt) < 2:
        mom_win_j = (
            float(fric_j[0])
            + ((float(fric_j[2]) - float(fric_j[0])) / 2)
            - frame_size_fric / 2
        )
        with open(resFile, "a") as result_file:
            print(
                "\nWARNING: There are not at least two ruptures inside of the [j] segment.\nTook the midpoint of the segment as analysis window location",
                file=result_file,
            )
    elif (
        len(j_rupt) == 2
        and (float(j_rupt[1]) - float(j_rupt[0])) < frame_size_fric
    ):
        mom_win_j = (
            float(fric_j[0])
            + ((float(fric_j[2]) - float(fric_j[0])) / 2)
            - frame_size_fric / 2
        )
        with open(resFile, "a") as result_file:
            print(
                "\nWARNING: There are two ruptures inside of the [j] segment, but they are too close to be considered as separate ruptures (< frame size).\nTook the midpoint of the segment as analysis window location",
                file=result_file,
            )

    elif (
        len(j_rupt) == 2
        and (float(j_rupt[1]) - float(j_rupt[0])) > frame_size_fric
    ):
        min_e_seg = sys.maxsize
        if (float(j_rupt[1]) - float(j_rupt[0])) >= 0.02:
            rupt_beg = j_rupt[0] * samp_freq
            rupt_end = j_rupt[1] * samp_freq
            mean_e_rupt = 0
            mean_e_rupt = numpy.mean(y[int(rupt_beg) : int(rupt_end) + 1])
            sum_e = 0
            for j in range(int(rupt_beg), int(rupt_end)):
                sum_e = (y[j] - mean_e_rupt) ** 2 + sum_e
            sum_e_norm = sum_e / (len(range(int(rupt_beg), int(rupt_end))))
            if sum_e_norm < min_e_seg:
                min_e_seg = sum_e_norm
                mom_win_j = (
                    j_rupt[0]
                    + ((j_rupt[1] - j_rupt[0]) / 2)
                    - frame_size_fric / 2
                )
        elif (float(j_rupt[1]) - float(j_rupt[0])) < 0.02:
            with open(resFile, "a") as result_file:
                print(
                    "\nWARNING: The longest stable part inside of the [j] segment is shorter than 20ms.",
                    file=result_file,
                )
            rupt_beg = j_rupt[0] * samp_freq
            rupt_end = j_rupt[1] * samp_freq
            mean_e_rupt = 0
            mean_e_rupt = numpy.mean(y[int(rupt_beg) : int(rupt_end) + 1])
            sum_e = 0
            for j in range(int(rupt_beg), int(rupt_end)):
                sum_e = (y[j] - mean_e_rupt) ** 2 + sum_e
            sum_e_norm = sum_e / (len(range(int(rupt_beg), int(rupt_end))))
            if sum_e_norm < min_e_seg:
                min_e_seg = sum_e_norm
                mom_win_j = (
                    j_rupt[0]
                    + ((j_rupt[1] - j_rupt[0]) / 2)
                    - frame_size_fric / 2
                )

    else:
        l = len(j_rupt)
        i = 0
        min_e_seg = sys.maxsize
        while i < l - 1:
            if (float(j_rupt[i + 1]) - float(j_rupt[i])) >= 0.02:
                rupt_beg = j_rupt[i] * samp_freq
                rupt_end = j_rupt[i + 1] * samp_freq
                mean_e_rupt = 0
                mean_e_rupt = numpy.mean(y[int(rupt_beg) : int(rupt_end) + 1])
                sum_e = 0
                for j in range(int(rupt_beg), int(rupt_end)):
                    sum_e = (y[j] - mean_e_rupt) ** 2 + sum_e
                sum_e_norm = sum_e / (len(range(int(rupt_beg), int(rupt_end))))
                if sum_e_norm < min_e_seg:
                    min_e_seg = sum_e_norm
                    mom_win_j = (
                        j_rupt[i]
                        + ((j_rupt[i + 1] - j_rupt[i]) / 2)
                        - frame_size_fric / 2
                    )
            i = i + 1

    labels = {
        "p1": p1_str,
        "p2": p2_str,
        "p3": p3_str,
        "p4": p4_str,
        "p5": p5_str,
        "p6": p6_str,
        "p7": p7_str,
        "p8": p8_str,
        "p9": p9_str,
        "t1": t1_str,
        "t2": t2_str,
        "t3": t3_str,
        "t4": t4_str,
        "k1": k1_str,
        "k2": k2_str,
        "k3": k3_str,
        "k4": k4_str,
        "b": b_str,
        "d": d_str,
        "g1": g1_str,
        "g2": g2_str,
        "f": f_str,
        "s": s_str,
        "ch": ch_str,
        "v1": v1_str,
        "v2": v2_str,
        "z": z_str,
        "j": j_str,
    }
    windows = {
        "p1": mom_win_p1,
        "p2": mom_win_p2,
        "p3": mom_win_p3,
        "p4": mom_win_p4,
        "p5": mom_win_p5,
        "p6": mom_win_p6,
        "p7": mom_win_p7,
        "p8": mom_win_p8,
        "p9": mom_win_p9,
        "t1": mom_win_t1,
        "t2": mom_win_t2,
        "t3": mom_win_t3,
        "t4": mom_win_t4,
        "k1": mom_win_k1,
        "k2": mom_win_k2,
        "k3": mom_win_k3,
        "k4": mom_win_k4,
        "b": mom_win_b,
        "d": mom_win_d,
        "g1": mom_win_g1,
        "g2": mom_win_g2,
        "f": mom_win_f,
        "s": mom_win_s,
        "ch": mom_win_ch,
        "v1": mom_win_v1,
        "v2": mom_win_v2,
        "z": mom_win_z,
        "j": mom_win_j,
    }

    subprocess.call(
        _build_praat_command(
            praat_id,
            praat_wav_dir,
            labels,
            windows,
            praat_result_dir,
        )
    )
    _write_debug_windows(result_file_path, windows)
