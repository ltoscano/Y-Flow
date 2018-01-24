IndriRunQuery ./data/tl/ANALYSIS-TL/qmodel/google -index=data/tl/ANALYSIS-TL/index/ -count=20 -trecFormat=true > data/tl/ANALYSIS-TL/result/result.first
python data/tl/ANALYSIS-TL/param/trim_result.py #result.file
#trec_eval -q data/sw/judg/rel.judg data/sw/ANALYSIS-SW/result/result.file > data/sw/ANALYSIS-SW/result/eval.result
