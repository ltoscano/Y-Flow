python ./data/tl/ANALYSIS-TL/src/wiktionary.py --src_lang en --tgt_lang tl --query en.t --tquery ./data/tl/ANALYSIS-TL/qmodel/wiktionary --dico_train ./data/dictionary/en-tl.txt --rank 5
IndriRunQuery ./data/tl/ANALYSIS-TL/qmodel/wiktionary -index=./data/tl/ANALYSIS-TL/index/ -count=20 -trecFormat=true > ./data/tl/ANALYSIS-TL/result/result.first
python ./data/tl/ANALYSIS-TL/param/trim_result.py #result.file
