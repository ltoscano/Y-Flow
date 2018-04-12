python src/wiktionary.py --src_lang en --tgt_lang tl --query ../topics/en.t --tquery qmodel/wiktionary --dico_train ../../dictionary/en-tl.txt --rank 5
IndriRunQuery qmodel/wiktionary -index=index/ -count=9 -trecFormat=true > result/result.first
python param/trim_local_result.py # generate result.file
#rm result/result.first
