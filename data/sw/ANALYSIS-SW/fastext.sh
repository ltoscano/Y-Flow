IndriRunQuery qmodel/fastext -index=index/ -count=20 -trecFormat=true > result/result.first
python param/trim_local_result.py # generate result.file
rm result/result.first
