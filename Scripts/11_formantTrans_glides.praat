# Formant transitions for glides
# Calculates start and end values and slopes of F1, F2, F3 in one WAV/TextGrid pair.
# This script is part of Timothy Pommee's PhD thesis (2021), adapted for Python-driven single-file use.

form Variables
    sentence wavfile input.wav
    sentence textgridfile input.TextGrid
    sentence outputfile formant_trans_glides.csv
endform

tier = 3
glides$ = "wHj"
intervalHead$ = "interval "
cols$ = intervalHead$ + "tbeg tend f1_beg f1_end f2_beg f2_end f3_beg f3_end f1_slope f2_slope f3_slope"

formantsTable = Create Table with column names: "formants", 0, cols$
sound = Read from file: wavfile$
textgrid = Read from file: textgridfile$

selectObject: textgrid
numberOfIntervals = Get number of intervals: tier

row = 1
for interval from 1 to numberOfIntervals
    selectObject: textgrid
    int$ = Get label of interval: tier, interval
    glidebeg = Get start time of interval: tier, interval
    glideend = Get end time of interval: tier, interval

    if index(glides$, int$)
        selectObject: sound
        soundPart = Extract part: glidebeg, glideend, "rectangular", 1, "yes"
        formant = To Formant (burg): 0, 5, 5500, 0.005, 50
        formantTable = Down to Table: "no", "yes", 6, "no", 3, "no", 3, "no"
        nRows = Get number of rows

        if nRows >= 5
            tbeg = Get value: 1, "time(s)"
            tend = Get value: nRows, "time(s)"

            f1_1 = Get value: 1, "F1(Hz)"
            f1_2 = Get value: 2, "F1(Hz)"
            f1_3 = Get value: 3, "F1(Hz)"
            f1_4 = Get value: 4, "F1(Hz)"
            f1_5 = Get value: 5, "F1(Hz)"
            f1_6 = Get value: nRows, "F1(Hz)"
            f1_7 = Get value: nRows - 1, "F1(Hz)"
            f1_8 = Get value: nRows - 2, "F1(Hz)"
            f1_9 = Get value: nRows - 3, "F1(Hz)"
            f1_10 = Get value: nRows - 4, "F1(Hz)"

            f2_1 = Get value: 1, "F2(Hz)"
            f2_2 = Get value: 2, "F2(Hz)"
            f2_3 = Get value: 3, "F2(Hz)"
            f2_4 = Get value: 4, "F2(Hz)"
            f2_5 = Get value: 5, "F2(Hz)"
            f2_6 = Get value: nRows, "F2(Hz)"
            f2_7 = Get value: nRows - 1, "F2(Hz)"
            f2_8 = Get value: nRows - 2, "F2(Hz)"
            f2_9 = Get value: nRows - 3, "F2(Hz)"
            f2_10 = Get value: nRows - 4, "F2(Hz)"

            f3_1 = Get value: 1, "F3(Hz)"
            f3_2 = Get value: 2, "F3(Hz)"
            f3_3 = Get value: 3, "F3(Hz)"
            f3_4 = Get value: 4, "F3(Hz)"
            f3_5 = Get value: 5, "F3(Hz)"
            f3_6 = Get value: nRows, "F3(Hz)"
            f3_7 = Get value: nRows - 1, "F3(Hz)"
            f3_8 = Get value: nRows - 2, "F3(Hz)"
            f3_9 = Get value: nRows - 3, "F3(Hz)"
            f3_10 = Get value: nRows - 4, "F3(Hz)"

            f1_beg = (f1_1 + f1_2 + f1_3 + f1_4 + f1_5) / 5
            f1_end = (f1_6 + f1_7 + f1_8 + f1_9 + f1_10) / 5
            f2_beg = (f2_1 + f2_2 + f2_3 + f2_4 + f2_5) / 5
            f2_end = (f2_6 + f2_7 + f2_8 + f2_9 + f2_10) / 5
            f3_beg = (f3_1 + f3_2 + f3_3 + f3_4 + f3_5) / 5
            f3_end = (f3_6 + f3_7 + f3_8 + f3_9 + f3_10) / 5

            f1_mov = f1_end - f1_beg
            f2_mov = f2_end - f2_beg
            f3_mov = f3_end - f3_beg
            duration = (tend - tbeg) * 1000
            f1_slope = f1_mov / duration
            f2_slope = f2_mov / duration
            f3_slope = f3_mov / duration

            selectObject: formantsTable
            Append row
            Set string value: row, "interval", int$
            Set numeric value: row, "tbeg", tbeg
            Set numeric value: row, "tend", tend
            Set numeric value: row, "f1_beg", f1_beg
            Set numeric value: row, "f1_end", f1_end
            Set numeric value: row, "f2_beg", f2_beg
            Set numeric value: row, "f2_end", f2_end
            Set numeric value: row, "f3_beg", f3_beg
            Set numeric value: row, "f3_end", f3_end
            Set numeric value: row, "f1_slope", f1_slope
            Set numeric value: row, "f2_slope", f2_slope
            Set numeric value: row, "f3_slope", f3_slope
            row = row + 1
        endif

        selectObject: formantTable
        plusObject: formant
        plusObject: soundPart
        Remove
    endif
endfor

selectObject: formantsTable
Save as comma-separated file: outputfile$

selectObject: formantsTable
plusObject: sound
plusObject: textgrid
Remove
