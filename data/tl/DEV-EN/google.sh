python ./data/sw/ANALYSIS-SW/src/google.py --src_lang en --tgt_lang sw --query ./data/sw/topics/google.t --tquery ./data/sw/ANALYSIS-SW/qmodel/google
IndriRunQuery ./data/sw/ANALYSIS-SW/qmodel/google -index=./data/sw/ANALYSIS-SW/index/ -count=1000 -trecFormat=true > ./data/sw/ANALYSIS-SW/result/result.first
python ./data/sw/ANALYSIS-SW/param/trim_result.py #result.file
