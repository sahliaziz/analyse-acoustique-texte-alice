#!/usr/bin/env python
# coding: utf-8
# this script is part of Timothy Pommée's PhD thesis (2021)

import os
import sys
from pathlib import Path

scriptdir = os.path.abspath(os.path.dirname(__file__))
d = str(Path(__file__).resolve().parents[1])
directory = d + "/3_alignement_force/"
result_directory = d + "/5_consonnes/txt/"
consonants = ("p", "t", "k", "b", "d", "g", "f", "s", "S", "v", "z", "Z")

for filename in sorted(os.listdir(directory)):
    if filename.endswith(".csv"):
        with open(os.path.join(directory, filename), "r") as inputfile:
            print(filename)
            lines = inputfile.readlines()
            lines_to_write = []
            length = len(lines) - 2
            if length >= 658:
                for i in range(2, length):
                    tmin, text, tmax = lines[i].split()
                    text_before = lines[i - 1].split()[1]
                    text_bebefore = lines[i - 2].split()[1]
                    text_after = lines[i + 1].split()[1]
                    text_afterter = lines[i + 2].split()[1]
                    if text in consonants:
                        if (
                            (text_before == "a" and text_after == "a")
                            or (
                                text_bebefore == "a"
                                and text_before == "<p:>"
                                and text_after == "<p:>"
                                and text_afterter == "a"
                            )
                            or (
                                text_bebefore == "a"
                                and text_before == "<p:>"
                                and text_after == "a"
                            )
                            or (
                                text_before == "a"
                                and text_after == "<p:>"
                                and text_afterter == "a"
                            )
                        ):
                            lines_to_write.append(lines[i])

                            file_to_write = result_directory + filename
                            with open(file_to_write, "w") as outputfile:
                                for line in lines_to_write:
                                    outputfile.write(line)

folder = result_directory
for filename in sorted(os.listdir(folder)):
    infilename = os.path.join(folder, filename)
    if not os.path.isfile(infilename):
        continue
    oldbase = os.path.splitext(filename)
    newname = infilename.replace(".csv", ".txt")
    output = os.rename(infilename, newname)
