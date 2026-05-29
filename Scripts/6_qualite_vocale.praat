# TITLE OF THE SCRIPT: Analyses acoustiques parole continue
# This script is part of Timothy Pommée's PhD thesis (2021) and has been adapted from Maryn, Corthals & Barsties (Acoustic Voice Quality Index 03.01)

form Select input files
	sentence resultdir 
	sentence wavfile input.wav
	sentence textgridfile input.TextGrid
endform

dir_pictures$ = resultdir$ + "/pictures/"
dir_onlyvoiced$ = resultdir$ + "/only_voiced/"


Erase all
Select inner viewport... 0.5 7.5 0.5 4.5
Axes... 0 1 0 1
Black
Text special... 0.5 centre 0.6 half Helvetica 12 0 Please wait an instant. Depending on the duration and/or the sample rate of the recorded
Text special... 0.5 centre 0.4 half Helvetica 12 0 sound files, this script takes more or less time to process the sound and search for the AVQI.


tier = 1
fileName$ = replace_regex$(wavfile$, ".*[/\\]", "", 0)
fileName_raw$ = replace_regex$(fileName$, "(?i)\.wav$", "", 0)

Read from file: textgridfile$
numberOfIntervals = Get number of intervals... tier

notempty = Count intervals where... 1 "is not equal to"
for interval from 1 to numberOfIntervals
	int$ = ""
	int$ = Get label of interval... tier interval
	if int$ == "plage"
		end_first_half_text = interval
		beg_second_half_text = interval+1
	endif
endfor

# FIRST SENTENCE

	# Default values for variables
	sent_start1 = 0
	sent_end1 = 0
	for interval from 1 to end_first_half_text
		selectObject: "TextGrid " + fileName_raw$
		intname$ = ""
		intname$ = Get label of interval... tier interval
		if intname$ == "mais"
			sent_start1 = Get start time of interval... tier interval
		endif
		if intname$ == "ravivent"
			sent_end1 = Get end time of interval... tier interval
		endif
	endfor

	Read from file: wavfile$
	Extract part... sent_start1 sent_end1 rectangular 1.0 no
	Rename... cs
# --------------------------------------------------------------------------------------------
# PART 0:
# HIGH-PASS FILTERING OF THE SOUND FILES.
# --------------------------------------------------------------------------------------------
	select Sound cs
	Filter (stop Hann band)... 0 34 0.1
	Rename... cs2
# --------------------------------------------------------------------------------------------
# PART 1:
# DETECTION, EXTRACTION AND CONCATENATION OF
# THE VOICED SEGMENTS IN THE RECORDING
# OF CONTINUOUS SPEECH.
# --------------------------------------------------------------------------------------------
	select Sound cs2
	Copy... original
	samplingRate = Get sampling frequency
	intermediateSamples = Get sampling period
	Create Sound... onlyVoice 0 0.001 'samplingRate' 0
	select Sound original
	To TextGrid (silences)... 50 0.003 -25 0.1 0.1 silence sounding
	select Sound original
	plus TextGrid original
	Extract intervals where... 1 no "does not contain" silence
	Concatenate
	select Sound chain
	Rename... onlyLoud
	globalPower = Get power in air
	select TextGrid original
	Remove

	select Sound onlyLoud
	signalEnd = Get end time
	windowBorderLeft = Get start time
	windowWidth = 0.03
	windowBorderRight = windowBorderLeft + windowWidth
	globalPower = Get power in air
	voicelessThreshold = globalPower*(30/100)

	select Sound onlyLoud
	extremeRight = signalEnd - windowWidth
	while windowBorderRight < extremeRight
		Extract part... 'windowBorderLeft' 'windowBorderRight' Rectangular 1.0 no
		select Sound onlyLoud_part
		partialPower = Get power in air
		if partialPower > voicelessThreshold
			call checkZeros 0
			if (zeroCrossingRate <> undefined) and (zeroCrossingRate < 3000)
				select Sound onlyVoice
				plus Sound onlyLoud_part
				Concatenate
				Rename... onlyVoiceNew
				select Sound onlyVoice
				Remove
				select Sound onlyVoiceNew
				Rename... onlyVoice
			endif
		endif
		select Sound onlyLoud_part
		Remove
		windowBorderLeft = windowBorderLeft + 0.03
		windowBorderRight = windowBorderLeft + 0.03
		select Sound onlyLoud
	endwhile
	select Sound onlyVoice

	procedure checkZeros zeroCrossingRate

		start = 0.0025
		startZero = Get nearest zero crossing... start
		findStart = startZero
		findStartZeroPlusOne = startZero + intermediateSamples
		startZeroPlusOne = Get nearest zero crossing... 'findStartZeroPlusOne'
		zeroCrossings = 0
		strips = 0

		while (findStart < 0.0275) and (findStart <> undefined)
			while startZeroPlusOne = findStart
				findStartZeroPlusOne = findStartZeroPlusOne + intermediateSamples
				startZeroPlusOne = Get nearest zero crossing... 'findStartZeroPlusOne'
			endwhile
			afstand = startZeroPlusOne - startZero
			strips = strips +1
			zeroCrossings = zeroCrossings +1
			findStart = startZeroPlusOne
		endwhile
		zeroCrossingRate = zeroCrossings/afstand
	endproc

# --------------------------------------------------------------------------------------------
# PART 2:
# DETERMINATION OF THE THREE ACOUSTIC MEASURES.
# --------------------------------------------------------------------------------------------

	select Sound onlyVoice
	durationOnlyVoice = Get total duration
	Rename... cs
	durationAll = Get total duration
	minimumSPL = Get minimum... 0 0 None
	maximumSPL = Get maximum... 0 0 None

# Narrow-band spectrogram and LTAS

	To Spectrogram... 0.03 4000 0.002 20 Gaussian
	select Sound cs
	To Ltas... 1
	minimumSpectrum = Get minimum... 0 4000 None
	maximumSpectrum = Get maximum... 0 4000 None

# Power-cepstrogram, Cepstral peak prominence and Smoothed cepstral peak prominence

	select Sound cs
	To PowerCepstrogram... 60 0.002 5000 50
	cpps = Get CPPS... no 0.01 0.001 60 330 0.05 Parabolic 0.001 0 Straight Robust
	To PowerCepstrum (slice)... 0.1
	Rename... cs_0_100
	maximumCepstrum = Get peak... 60 330 None

# Slope of the long-term average spectrum

	select Sound cs
	To Ltas... 1
	slope = Get slope... 0 1000 1000 10000 energy

# Tilt of trendline through the long-term average spectrum

	select Ltas cs
	Compute trend line... 1 10000
	tilt = Get slope... 0 1000 1000 10000 energy


# --------------------------------------------------------------------------------------------
# PART 3:
# DRAWING ALL THE INFORMATION AND THE GRAPHS.
# --------------------------------------------------------------------------------------------

# Title and file information

	Erase all
	Solid line
	Line width... 1
	Black
	Helvetica
	Select inner viewport... 0 8 0 0.5
	Font size... 1
	Select inner viewport... 0.5 7.5 0.1 0.15
	Axes... 0 1 0 1
	Font size... 12
	Select inner viewport... 0.5 7.5 0 0.5
	Axes... 0 1 0 1
	Text... 0 Left 0.5 Half ##Analyses acoustiques parole continue phrase voisée#
	Font size... 8
	Select inner viewport... 0.5 7.5 0 0.5
	Axes... 0 1 0 3
	Text... 1 Right 2.3 Half 'fileName_raw$' sent_1

	# Oscillogram

	Font size... 7
	Select inner viewport... 0.5 5 0.5 2.0
	select Sound cs
	Draw... 0 0 0 0 no Curve
	Draw inner box
	One mark left... minimumSPL no yes no 'minimumSPL:2'
	One mark left... maximumSPL no yes no 'maximumSPL:2'
	Text left... no Sound pressure level (Pa)
	One mark bottom... 0 no yes no 0.00
	One mark bottom... durationOnlyVoice no no yes
	One mark bottom... durationAll no yes no 'durationAll:2'
	Text bottom... no Time (s)

	# Narrow-band spectrogram

	Select inner viewport... 0.5 5 2.3 3.8
	select Spectrogram cs
	Paint... 0 0 0 4000 100 yes 50 6 0 no
	Draw inner box
	One mark left... 0 no yes no 0
	One mark left... 4000 no yes no 4000
	Text left... no Frequency (Hz)
	One mark bottom... 0 no yes no 0.00
	One mark bottom... durationOnlyVoice no no yes
	One mark bottom... durationAll no yes no 'durationAll:2'
	Text bottom... no Time (s)

	# LTAS

	Select inner viewport... 5.4 7.5 2.3 3.8
	select Ltas cs
	Draw... 0 4000 minimumSpectrum maximumSpectrum no Curve
	Draw inner box
	One mark left... minimumSpectrum no yes no 'minimumSpectrum:2'
	One mark left... maximumSpectrum no yes no 'maximumSpectrum:2'
	Text left... no Sound pressure level (dB/Hz)
	One mark bottom... 0 no yes no 0
	One mark bottom... 4000 no yes no 4000
	Text bottom... no Frequency (Hz)

	# Power-cepstrogram

	Select inner viewport... 0.5 5 4.1 5.6
	select PowerCepstrogram cs
	Paint: 0, 0, 0, 0, 80, "no", 30, 0, "no"
	Draw inner box
	One mark left... 0.00303 no yes no 0.003
	One mark left... 0.01667 no yes no 0.017
	Text left... no Quefrency (s)
	One mark bottom... 0 no yes no 0.00
	One mark bottom... durationOnlyVoice no no yes
	One mark bottom... durationAll no yes no 'durationAll:2'
	Text bottom... no Time (s)

	# Power-cepstrum

	Select inner viewport... 5.4 7.5 4.1 5.6
	select PowerCepstrum cs_0_100
	Draw... 0.00303 0.01667 0 0 no
	Draw tilt line... 0.00303 0.01667 0 0 0.00303 0.01667 Straight Robust
	Draw inner box
	One mark left... maximumCepstrum no yes no 'maximumCepstrum:2'
	Text left... no Amplitude (dB)
	One mark bottom... 0.00303 no yes no 0.003
	One mark bottom... 0.01667 no yes no 0.017
	Text bottom... no Quefrency (s)

	# Data

	Font size... 10
	Select inner viewport... 0.5 7.5 5.9 7.4
	Axes... 0 7 6 0
	Text... 0.05 Left 0.5 Half Smoothed cepstral peak prominence (CPPS): ##'cpps:2'#
	Text... 0.05 Left 1.5 Half Slope of LTAS: ##'slope:2' dB#
	Text... 0.05 Left 2.5 Half Tilt of trendline through LTAS: ##'tilt:2' dB#

	# Copy Praat picture
	Select inner viewport... 0.5 7.5 0 7.4
	Save as 600-dpi PNG file: dir_pictures$ + fileName_raw$ + "_sent1.png"

	# Copy data to file
	appendFileLine: resultdir$ + "/Measures_sent1.txt", fileName$, ",", fixed$ (cpps, 2), ",", fixed$ (slope, 2), ",", fixed$ (tilt, 2), ","

	# Save sound file with only voiced segments
	select Sound cs
	fileName_onlyvoiced1$ = fileName_raw$ + "_OnlyVoiced"
	Save as WAV file: dir_onlyvoiced$ + fileName_onlyvoiced1$ + "_sent1.wav"

# Remove intermediate objects
	select all
	Remove


#SECOND SENTENCE

	if notempty == 232

		# Default values for variables
		Read from file: textgridfile$
		sent_start2 = 0
		sent_end2 = 0
		for interval from beg_second_half_text to numberOfIntervals
			selectObject: "TextGrid " + fileName_raw$
			intname$ = ""
			intname$ = Get label of interval... tier interval
			if intname$ == "mais"
				sent_start2 = Get start time of interval... tier interval
			endif
			if intname$ == "ravivent"
				sent_end2 = Get end time of interval... tier interval
			endif
		endfor

		Read from file: wavfile$
		Extract part... sent_start2 sent_end2 rectangular 1.0 no
		Rename... cs
	# --------------------------------------------------------------------------------------------
	# PART 0:
	# HIGH-PASS FILTERING OF THE SOUND FILES.
	# --------------------------------------------------------------------------------------------
		select Sound cs
		Filter (stop Hann band)... 0 34 0.1
		Rename... cs2
	# --------------------------------------------------------------------------------------------
	# PART 1:
	# DETECTION, EXTRACTION AND CONCATENATION OF
	# THE VOICED SEGMENTS IN THE RECORDING
	# OF CONTINUOUS SPEECH.
	# --------------------------------------------------------------------------------------------
		select Sound cs2
		Copy... original
		samplingRate = Get sampling frequency
		intermediateSamples = Get sampling period
		Create Sound... onlyVoice 0 0.001 'samplingRate' 0
		select Sound original
		To TextGrid (silences)... 50 0.003 -25 0.1 0.1 silence sounding
		select Sound original
		plus TextGrid original
		Extract intervals where... 1 no "does not contain" silence
		Concatenate
		select Sound chain
		Rename... onlyLoud
		globalPower = Get power in air
		select TextGrid original
		Remove

		select Sound onlyLoud
		signalEnd = Get end time
		windowBorderLeft = Get start time
		windowWidth = 0.03
		windowBorderRight = windowBorderLeft + windowWidth
		globalPower = Get power in air
		voicelessThreshold = globalPower*(30/100)

		select Sound onlyLoud
		extremeRight = signalEnd - windowWidth
		while windowBorderRight < extremeRight
			Extract part... 'windowBorderLeft' 'windowBorderRight' Rectangular 1.0 no
			select Sound onlyLoud_part
			partialPower = Get power in air
			if partialPower > voicelessThreshold
				call checkZeros 0
				if (zeroCrossingRate <> undefined) and (zeroCrossingRate < 3000)
					select Sound onlyVoice
					plus Sound onlyLoud_part
					Concatenate
					Rename... onlyVoiceNew
					select Sound onlyVoice
					Remove
					select Sound onlyVoiceNew
					Rename... onlyVoice
				endif
			endif
			select Sound onlyLoud_part
			Remove
			windowBorderLeft = windowBorderLeft + 0.03
			windowBorderRight = windowBorderLeft + 0.03
			select Sound onlyLoud
		endwhile
		select Sound onlyVoice

		procedure checkZeros zeroCrossingRate

			start = 0.0025
			startZero = Get nearest zero crossing... 'start'
			findStart = startZero
			findStartZeroPlusOne = startZero + intermediateSamples
			startZeroPlusOne = Get nearest zero crossing... 'findStartZeroPlusOne'
			zeroCrossings = 0
			strips = 0

			while (findStart < 0.0275) and (findStart <> undefined)
				while startZeroPlusOne = findStart
					findStartZeroPlusOne = findStartZeroPlusOne + intermediateSamples
					startZeroPlusOne = Get nearest zero crossing... 'findStartZeroPlusOne'
				endwhile
				afstand = startZeroPlusOne - startZero
				strips = strips +1
				zeroCrossings = zeroCrossings +1
				findStart = startZeroPlusOne
			endwhile
			zeroCrossingRate = zeroCrossings/afstand
		endproc

	# --------------------------------------------------------------------------------------------
	# PART 2:
	# DETERMINATION OF THE THREE ACOUSTIC MEASURES.
	# --------------------------------------------------------------------------------------------

		select Sound onlyVoice
		durationOnlyVoice = Get total duration
		Rename... cs
		durationAll = Get total duration
		minimumSPL = Get minimum... 0 0 None
		maximumSPL = Get maximum... 0 0 None

	# Narrow-band spectrogram and LTAS

		To Spectrogram... 0.03 4000 0.002 20 Gaussian
		select Sound cs
		To Ltas... 1
		minimumSpectrum = Get minimum... 0 4000 None
		maximumSpectrum = Get maximum... 0 4000 None

	# Power-cepstrogram, Cepstral peak prominence and Smoothed cepstral peak prominence

		select Sound cs
		To PowerCepstrogram... 60 0.002 5000 50
		cpps = Get CPPS... no 0.01 0.001 60 330 0.05 Parabolic 0.001 0 Straight Robust
		To PowerCepstrum (slice)... 0.1
		Rename... cs_0_100
		maximumCepstrum = Get peak... 60 330 None

	# Slope of the long-term average spectrum

		select Sound cs
		To Ltas... 1
		slope = Get slope... 0 1000 1000 10000 energy

	# Tilt of trendline through the long-term average spectrum

		select Ltas cs
		Compute trend line... 1 10000
		tilt = Get slope... 0 1000 1000 10000 energy


	# --------------------------------------------------------------------------------------------
	# PART 3:
	# DRAWING ALL THE INFORMATION AND THE GRAPHS.
	# --------------------------------------------------------------------------------------------

	# Title and file information

		Erase all
		Solid line
		Line width... 1
		Black
		Helvetica
		Select inner viewport... 0 8 0 0.5
		Font size... 1
		Select inner viewport... 0.5 7.5 0.1 0.15
		Axes... 0 1 0 1
		Font size... 12
		Select inner viewport... 0.5 7.5 0 0.5
		Axes... 0 1 0 1
		Text... 0 Left 0.5 Half ##Analyses acoustiques parole continue phrase voisée#
		Font size... 8
		Select inner viewport... 0.5 7.5 0 0.5
		Axes... 0 1 0 3
		Text... 1 Right 2.3 Half 'fileName_raw$' sent_2

		# Oscillogram

		Font size... 7
		Select inner viewport... 0.5 5 0.5 2.0
		select Sound cs
		Draw... 0 0 0 0 no Curve
		Draw inner box
		One mark left... minimumSPL no yes no 'minimumSPL:2'
		One mark left... maximumSPL no yes no 'maximumSPL:2'
		Text left... no Sound pressure level (Pa)
		One mark bottom... 0 no yes no 0.00
		One mark bottom... durationOnlyVoice no no yes
		One mark bottom... durationAll no yes no 'durationAll:2'
		Text bottom... no Time (s)

		# Narrow-band spectrogram

		Select inner viewport... 0.5 5 2.3 3.8
		select Spectrogram cs
		Paint... 0 0 0 4000 100 yes 50 6 0 no
		Draw inner box
		One mark left... 0 no yes no 0
		One mark left... 4000 no yes no 4000
		Text left... no Frequency (Hz)
		One mark bottom... 0 no yes no 0.00
		One mark bottom... durationOnlyVoice no no yes
		One mark bottom... durationAll no yes no 'durationAll:2'
		Text bottom... no Time (s)

		# LTAS

		Select inner viewport... 5.4 7.5 2.3 3.8
		select Ltas cs
		Draw... 0 4000 minimumSpectrum maximumSpectrum no Curve
		Draw inner box
		One mark left... minimumSpectrum no yes no 'minimumSpectrum:2'
		One mark left... maximumSpectrum no yes no 'maximumSpectrum:2'
		Text left... no Sound pressure level (dB/Hz)
		One mark bottom... 0 no yes no 0
		One mark bottom... 4000 no yes no 4000
		Text bottom... no Frequency (Hz)

		# Power-cepstrogram

		Select inner viewport... 0.5 5 4.1 5.6
		select PowerCepstrogram cs
		Paint: 0, 0, 0, 0, 80, "no", 30, 0, "no"
		Draw inner box
		One mark left... 0.00303 no yes no 0.003
		One mark left... 0.01667 no yes no 0.017
		Text left... no Quefrency (s)
		One mark bottom... 0 no yes no 0.00
		One mark bottom... durationOnlyVoice no no yes
		One mark bottom... durationAll no yes no 'durationAll:2'
		Text bottom... no Time (s)

		# Power-cepstrum

		Select inner viewport... 5.4 7.5 4.1 5.6
		select PowerCepstrum cs_0_100
		Draw... 0.00303 0.01667 0 0 no
		Draw tilt line... 0.00303 0.01667 0 0 0.00303 0.01667 Straight Robust
		Draw inner box
		One mark left... maximumCepstrum no yes no 'maximumCepstrum:2'
		Text left... no Amplitude (dB)
		One mark bottom... 0.00303 no yes no 0.003
		One mark bottom... 0.01667 no yes no 0.017
		Text bottom... no Quefrency (s)

		# Data

		Font size... 10
		Select inner viewport... 0.5 7.5 5.9 7.4
		Axes... 0 7 6 0
		Text... 0.05 Left 0.5 Half Smoothed cepstral peak prominence (CPPS): ##'cpps:2'#
		Text... 0.05 Left 1.5 Half Slope of LTAS: ##'slope:2' dB#
		Text... 0.05 Left 2.5 Half Tilt of trendline through LTAS: ##'tilt:2' dB#

		# Copy Praat picture
		Select inner viewport... 0.5 7.5 0 7.4
		Save as 600-dpi PNG file: dir_pictures$ + fileName_raw$ + "_sent2.png"

		# Copy data to file
		appendFileLine: resultdir$ + "/Measures_sent2.txt", fileName$, ",", fixed$ (cpps, 2), ",", fixed$ (slope, 2), ",", fixed$ (tilt, 2), ","

		# Save sound file with only voiced segments
		select Sound cs
		fileName_onlyvoiced2$ = fileName_raw$ + "_OnlyVoiced"
		Save as WAV file: dir_onlyvoiced$ + fileName_onlyvoiced1$ + "_sent2.wav"

	endif

# Remove intermediate objects
select all
Remove
Erase all
