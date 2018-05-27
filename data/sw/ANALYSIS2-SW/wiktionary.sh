CUTOFF=$1
python src/wiktionary.py --src_lang en --tgt_lang sw --query ../topics/QUERY1/query_list.tsv --tquery qmodel/wiktionary --dico_train ../../dictionary/en-sw.txt --rank 5
IndriRunQuery qmodel/wiktionary -index=index/ -count=$CUTOFF -trecFormat=true > result/result.first
python param/trim_result.py #result.file
trec_eval -q -N 471 ../judg/rel.analysis result/result.file > wiktionary.out
