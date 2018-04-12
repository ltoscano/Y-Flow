python src/phrase.py --phrase True --src_lang en --tgt_lang sw --query ../topics/QUERY1/query_list.tsv --tquery qmodel/wiktionary --dico_train ../../dictionary/en-sw.txt --rank 5
IndriRunQuery qmodel/wiktionary -index=index/ -count=20 -trecFormat=true > result/result.first 
python param/trim_local_result.py #result.file
trec_eval -q -N 948 ../judg/rel.dev result/result.file > phrase.out
