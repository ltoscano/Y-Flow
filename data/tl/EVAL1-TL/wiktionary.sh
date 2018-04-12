python src/wiktionary.py --src_lang en --tgt_lang sw --query ../topics/en.t --tquery qmodel/wiktionary --dico_train ../../dictionary/en-sw.txt --rank 5
IndriRunQuery qmodel/wiktionary -index=index/ -count=20 -trecFormat=true > result/result.first
python param/trim_local_result.py #result.file
