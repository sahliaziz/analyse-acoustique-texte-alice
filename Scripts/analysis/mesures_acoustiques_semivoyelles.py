"""
Author: Pierre PINCON
Team: SAMoVA
Date: 2022

This script formats the data created by "11_formantTrans_glides.praat" so only relevant measures are kept.
"""

import os
import time

import pandas as pd

start = time.perf_counter()
path = './7_semi_consonnes/result/'

list_files = os.listdir(path)

tab = [['Sujet', 'Genre', 'Repetition', 'phoneme', 'f1_slope',
        'f2_slope', 'f3_slope']]

for file in range(len(list_files)):
    df = pd.read_csv(path + list_files[file], sep='\t', header=None)
    name = list_files[file][:-8]
    genre = list_files[file][-5]
    rep = list_files[file][-7]

    for lines in range(1, len(df.axes[0])):
        temporary_list = [name, genre, rep, df.iloc[lines][0],
                          float(df.iloc[lines][9]), float(df.iloc[lines][10]), float(df.iloc[lines][11])]

        tab.append(temporary_list[:])

df_mean = pd.DataFrame(tab)
df_mean.to_csv('./Analyzed_results/mesures_acoustiques_semivoyelles.csv',
               index=False, header=False)

print('Time elapsed during semi_vowels_analysis.py =', time.perf_counter()-start)
