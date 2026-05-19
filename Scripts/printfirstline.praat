form Variables
	sentence resdir Smile if you read this
endform

result_file$ = resdir$ + "spectralmoments_python.txt"
appendFileLine: result_file$, "id", tab$, "phoneme", tab$, "cog", tab$,	"sd", tab$, "skew", tab$, "kurt", tab$