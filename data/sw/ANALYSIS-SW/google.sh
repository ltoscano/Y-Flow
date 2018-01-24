IndriRunQuery ./data/sw/ANALYSIS-SW/qmodel/google -index=./data/sw/ANALYSIS-SW/index/ -count=20 -trecFormat=true > ./data/sw/ANALYSIS-SW/result/result.first
python ./data/sw/ANALYSIS-SW/param/trim_result.py #result.file
