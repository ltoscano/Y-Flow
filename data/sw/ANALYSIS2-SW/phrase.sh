#! /bin/sh

CUTOFF=$1

python src/phrase.py --phrase False --src_lang en --tgt_lang sw --query ../topics/QUERY1/query_list.tsv --tquery qmodel/phrase --dico_train ../../dictionary/en-sw.txt --rank 5
IndriRunQuery qmodel/phrase -index=index/ -count=$CUTOFF -trecFormat=true > result/result.first 
python param/trim_local_result.py #result.file
trec_eval -q -c -N 471 ../judg/rel.analysis result/result.file > phrase.out
#trec_eval -q -N 471 ../judg/rel.analysis result/result.file > phrase.out.cutoff.$CUTOFF
