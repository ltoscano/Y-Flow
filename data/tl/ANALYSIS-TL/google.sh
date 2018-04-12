python ./data/tl/ANALYSIS-TL/src/google.py --src_lang en --tgt_lang tl --query ./data/tl/topics/google.t --tquery ./data/tl/ANALYSIS-TL/qmodel/google
IndriRunQuery ./data/tl/ANALYSIS-TL/qmodel/google -index=./data/tl/ANALYSIS-TL/index/ -count=1000 -trecFormat=true > ./data/tl/ANALYSIS-TL/result/result.first
python ./data/tl/ANALYSIS-TL/param/trim_result.py #result.file
