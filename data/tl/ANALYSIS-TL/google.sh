IndriRunQuery ./data/tl/ANALYSIS-TL/qmodel/google -index=./data/tl/ANALYSIS-TL/index/ -count=20 -trecFormat=true > ./data/tl/ANALYSIS-TL/result/result.first
python ./data/tl/ANALYSIS-TL/param/trim_result.py #result.file
