"""
Author: Pierre PINCON
Team: SAMoVA
Date: 2022

This script formats the data created by "9_spectral_moments.py" so only relevant measures are kept.
"""

import time
import pandas as pd

start = time.perf_counter()

path = "./5_consonnes/result/spectralmoments_python.txt"
df = pd.read_csv(path, sep="\t", header=None)
tab_to_csv = [["sujet", "genre", "repetition", "phoneme", "cog", "sd", "skew", "kurt"]]
for i in range(1, len(df.axes[0])):
    if (
        df.iloc[i][2] != "--undefined--"
        and df.iloc[i][3] != "--undefined--"
        and df.iloc[i][4] != "--undefined--"
        and df.iloc[i][5] != "--undefined--"
    ):
        temporary_tab = []
        temporary_tab.append(df.iloc[i][0][:-4])
        temporary_tab.append(df.iloc[i][0][-1])
        temporary_tab.append(df.iloc[i][0][-3])
        temporary_tab.append(df.iloc[i][1])
        temporary_tab.append(df.iloc[i][2])
        temporary_tab.append(df.iloc[i][3])
        temporary_tab.append(df.iloc[i][4])
        temporary_tab.append(df.iloc[i][5])
        tab_to_csv.append(temporary_tab[:])
        df_export = pd.DataFrame(tab_to_csv)
        df_export.to_csv("./Analyzed_results/mesures_acoustiques_consonnes.csv", header=False, index=False)

print(
    "Time elapsed during mesures_acoustiques_consonnes.py =",
    time.perf_counter() - start,
)
