# Calculate spectral moments
# This script is part of Timothy Pommée's PhD thesis (2021) and has been adapted from Mayer (2011) : https://praatpfanne.lingphon.net/downloads/spectral_moments.txt

form Variables
    sentence id I love you
    sentence wavdir Oh do you
    sentence p1_str yes
    sentence p2_str yes
    sentence p3_str yes
    sentence p4_str yes
    sentence p5_str yes
    sentence p6_str yes
    sentence p7_str yes
    sentence p8_str yes
    sentence p9_str yes
    sentence t1_str yes
    sentence t2_str yes
    sentence t3_str yes
    sentence t4_str yes
    sentence k1_str yes
	sentence k2_str yes
	sentence k3_str yes
	sentence k4_str yes
    sentence b_str yes
    sentence d_str yes
    sentence g1_str yes
    sentence g2_str yes
    sentence f_str yes
    sentence s_str yes
    sentence ch_str yes
    sentence v1_str yes
    sentence v2_str yes
    sentence z_str yes
    sentence j_str yes
    real mom_win_p1 0
    real mom_win_p2 0
    real mom_win_p3 0
    real mom_win_p4 0
    real mom_win_p5 0
    real mom_win_p6 0
    real mom_win_p7 0
    real mom_win_p8 0
    real mom_win_p9 0
    real mom_win_t1 0
    real mom_win_t2 0
    real mom_win_t3 0
    real mom_win_t4 0
    real mom_win_k1 0
    real mom_win_k2 0
    real mom_win_k3 0
    real mom_win_k4 0
    real mom_win_b 0
    real mom_win_d 0
    real mom_win_g1 0
    real mom_win_g2 0
    real mom_win_f 0
    real mom_win_s 0
    real mom_win_ch 0
    real mom_win_v1 0
    real mom_win_v2 0
    real mom_win_z 0
    real mom_win_j 0
	sentence resdir Me too
endform


# Window length for plosives (5ms) and for fricatives (10ms); adapt to your requirements
windowF_plos = 0.005
windowF_fric = 0.01

clearinfo
csv_sep$ = ","

result_file$ = resdir$ + "spectralmoments.csv"
writeFileLine: result_file$, "fichier", csv_sep$, "phoneme", csv_sep$, "cog", csv_sep$, "sd", csv_sep$, "skew", csv_sep$, "kurt", csv_sep$

Read from file... 'wavdir$'/'id$'.wav
filename$ = id$ + ".wav"
	
	
# define analysis window (separate for fricatives and plosives)
# and call spectralMoments procedure

if p1_str$ == "p"
	select Sound 'id$'
	tx = mom_win_p1
	ty = mom_win_p1 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "p1", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_p1.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if p2_str$ == "p"
	select Sound 'id$'
	tx = mom_win_p2
	ty = mom_win_p2 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "p2", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_p2.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if p3_str$ == "p"
	select Sound 'id$'
	tx = mom_win_p3
	ty = mom_win_p3 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "p3", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_p3.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if p4_str$ == "p"
	select Sound 'id$'
	tx = mom_win_p4
	ty = mom_win_p4 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "p4", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_p4.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if p5_str$ == "p"
	select Sound 'id$'
	tx = mom_win_p5
	ty = mom_win_p5 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "p5", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_p5.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if p6_str$ == "p"
	select Sound 'id$'
	tx = mom_win_p6
	ty = mom_win_p6 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "p6", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_p6.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if p7_str$ == "p"
	select Sound 'id$'
	tx = mom_win_p7
	ty = mom_win_p7 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "p7", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_p7.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if p8_str$ == "p"
	select Sound 'id$'
	tx = mom_win_p8
	ty = mom_win_p8 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "p8", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_p8.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if p9_str$ == "p"
	select Sound 'id$'
	tx = mom_win_p9
	ty = mom_win_p9 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "p9", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_p9.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif


if t1_str$ == "t"
	select Sound 'id$'
	tx = mom_win_t1
	ty = mom_win_t1 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "t1", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_t1.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if t2_str$ == "t"
	select Sound 'id$'
	tx = mom_win_t2
	ty = mom_win_t2 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "t2", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_t2.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if t3_str$ == "t"
	select Sound 'id$'
	tx = mom_win_t3
	ty = mom_win_t3 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "t3", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_t3.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if t4_str$ == "t"
	select Sound 'id$'
	tx = mom_win_t4
	ty = mom_win_t4 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "t4", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_t4.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif


if k1_str$ == "k"
	select Sound 'id$'
	tx = mom_win_k1
	ty = mom_win_k1 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "k1", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_k1.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if k2_str$ == "k"
	select Sound 'id$'
	tx = mom_win_k2
	ty = mom_win_k2 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "k2", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_k2.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if k3_str$ == "k"
	select Sound 'id$'
	tx = mom_win_k3
	ty = mom_win_k3 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "k3", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_k3.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if k4_str$ == "k"
	select Sound 'id$'
	tx = mom_win_k4
	ty = mom_win_k4 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "k4", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_k4.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif
	
if b_str$ == "b"
	select Sound 'id$'
	tx = mom_win_b
	ty = mom_win_b + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "b", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_b.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if d_str$ == "d"
	select Sound 'id$'
	tx = mom_win_d
	ty = mom_win_d + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "d", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_d.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if g1_str$ == "g"
	select Sound 'id$'
	tx = mom_win_g1
	ty = mom_win_g1 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "g1", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_g1.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if g2_str$ == "g"
	select Sound 'id$'
	tx = mom_win_g2
	ty = mom_win_g2 + windowF_plos
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "g2", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_g2.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if f_str$ == "f"
	select Sound 'id$'
	tx = mom_win_f
	ty = mom_win_f + windowF_fric
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "f", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_f.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if s_str$ == "s"
	select Sound 'id$'
	tx = mom_win_s
	ty = mom_win_s + windowF_fric
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "s", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_s.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if ch_str$ == "ch"
	select Sound 'id$'
	tx = mom_win_ch
	ty = mom_win_ch + windowF_fric
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "ch", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_ch.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif
	
if v1_str$ == "v"
	select Sound 'id$'
	tx = mom_win_v1
	ty = mom_win_v1 + windowF_fric
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "v1", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_v1.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if v2_str$ == "v"
	select Sound 'id$'
	tx = mom_win_v2
	ty = mom_win_v2 + windowF_fric
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "v2", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_v2.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if z_str$ == "z"
	select Sound 'id$'
	tx = mom_win_z
	ty = mom_win_z + windowF_fric
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "z", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_z.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif

if j_str$ == "j"
	select Sound 'id$'
	tx = mom_win_j
	ty = mom_win_j + windowF_fric
	call spectralMoments 'tx' 'ty'
	appendFileLine: result_file$, filename$, csv_sep$, "j", csv_sep$, fixed$(grav,2), csv_sep$, fixed$(sdev,2), csv_sep$, fixed$(skew,4), csv_sep$, fixed$(kurt,4), csv_sep$
	intervalfile$ = resdir$ + id$ + "_j.wav"
	select Sound 'id$'
	Extract part... tx ty rectangular 1.0 no
	Save as WAV file... 'intervalfile$'
endif


# spectralMoments procedure
procedure spectralMoments .onset .offset
	# windowing
	Extract part... '.onset' '.offset' Hamming 1 no
	snd = selected ("Sound")
	# preemphasis filter (Nissen, 2003) and 1000Hz high-pass filter
	Filter (pre-emphasis)... 100
	Filter (pass Hann band)... 1000 0 100
	sndPre = selected ("Sound")
	# calculate spectrum
	To Spectrum... yes
	# extract spectral moments
	grav = Get centre of gravity... 2
	sdev = Get standard deviation... 2
	skew = Get skewness... 2
	kurt = Get kurtosis... 2
	plus snd
	plus sndPre
	Remove
endproc
