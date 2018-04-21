python src/wiktionary.py --src_lang en --tgt_lang tl --query ../topics/QUERY1/query_list.tsv --tquery qmodel/wiktionary --dico_train ../../dictionary/en-tl.txt --rank 5
IndriRunQuery qmodel/wiktionary -index=index/ -count=20 -trecFormat=true > result/result.first
python param/trim_result.py #result.file
trec_eval -q -N 471 ../judg/rel.analysis result/result.file > wiktionary.out
