#! /usr/bin/env python
# coding: utf-8

import os
import wave
import librosa
import datetime
from pathlib import Path

import numpy as np
import librosa.display
import matplotlib.pyplot as plt

scriptdir = '/home/rmihaja/OneDrive/Study/L2/Internship/src/Scripts'
d = '/home/rmihaja/OneDrive/Study/L2/Internship/src'
# wav sound files location
wavDir = d + "/2_wav_traites/"
# python result file location
resDir = d + "/8_prosodie/"
resFile = str(datetime.datetime.now()) + ".txt"

for filename in sorted(os.listdir(wavDir)):
    if filename.endswith(".wav"):
        print(filename)
        soundfile = wavDir + filename
        # ID = filename.replace(".wav", "")
        # with open(resDir + resFile, "a") as result_file:
        # result_file.write(ID)

        # read sound file and get sample rate to adapt the number of samples per frame size accordingly
        snd = wave.open(soundfile, 'rb')
        samp_freq = float(snd.getframerate())
        snd.close()

        # compute F0 with librosa

        y, sr = librosa.load(soundfile, sr=samp_freq)

        # prosody value with probabilistic yin
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz(
            'C2'), fmax=librosa.note_to_hz('C7'), sr=sr)

        # prosody value with regular yin
        f0 = librosa.yin(y, fmin=librosa.note_to_hz(
            'C2'), fmax=librosa.note_to_hz('C7'), sr=sr)

        times = librosa.times_like(f0)

        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
        ax.set(title='pYIN fundamental frequency estimation')
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
        ax.legend(loc='upper right')
