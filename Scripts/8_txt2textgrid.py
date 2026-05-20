# coding: utf-8
# this script is part of Timothy Pommée's PhD thesis (2021)

import os
from pathlib import Path

# The location of entry data files, should be modified according to  execution context
scriptdir = os.path.abspath(os.path.dirname(__file__))
d = str(Path(__file__).resolve().parents[1])
# text files (.txt) location
sylDir = d + "/5_consonnes/txt/"
# resulting textgrid files location where results will be stored
txtgrDir = sylDir  # + "/ruptures_textgrid/"
for txtFile in sorted(os.listdir(sylDir)):
    if txtFile.endswith("_diverg.txt"):
        fichier_texte = os.path.splitext(txtFile)[0]
        filepath = sylDir + "/" + txtFile
        f = open(filepath, "r")

        lignes = f.readlines()
        tier1 = []
        for l in lignes:
            tier1.append(l.split()[0])
        f.close()
        outputpath = txtgrDir + fichier_texte + ".TextGrid"
        w = open(outputpath, "w")
        w.write('File type= "ooTextFile"\n')
        w.write('Object class = "TextGrid"\n')
        w.write("\n")
        w.write("xmin = 0 \n")
        w.write("xmax = " + tier1[-1] + "\n")
        w.write("tiers? <exists> \n")
        w.write("size = 1 \n")
        w.write("item []: \n")
        w.write("    item [1]:\n")
        w.write('        class = "IntervalTier" \n')
        w.write('        name = "burstboundaries" \n')
        w.write("        xmin = 0 \n")
        w.write("        xmax = " + tier1[-1] + "\n")
        p = str(len(tier1) - 1)
        w.write("        intervals: size = " + p + "\n")
        i = 1
        d_preced = tier1[0]
        for d in tier1[1::]:
            # print (d)
            w.write("        intervals [" + str(i) + "]:\n")
            w.write("            xmin = " + str(float(d_preced)) + "\n")
            w.write("            xmax = " + str(float(d)) + "\n")
            w.write('            text = ""' + "\n")
            d_preced = d
            i = i + 1
        w.close()
