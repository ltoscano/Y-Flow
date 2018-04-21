#! /bin/sh

CUTOFF=$1

python src/phrase.py --phrase True --src_lang en --tgt_lang tl --query ../topics/QUERY1/query_list.tsv --tquery qmodel/phrase --dico_train ../../dictionary/en-tl.txt --rank 5
IndriRunQuery qmodel/phrase -index=index/ -count=$CUTOFF -trecFormat=true > result/result.first 
python param/trim_local_result.py #result.file
