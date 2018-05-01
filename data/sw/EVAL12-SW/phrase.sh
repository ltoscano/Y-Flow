#! /bin/sh

CUTOFF=$1
QUERY='../topics/QUERY1/query_list.tsv'
if [ -n "$2" ]; then
  QUERY=$2
fi



python src/phrase.py --phrase True --src_lang en --tgt_lang sw --query $QUERY --tquery qmodel/phrase --dico_train ../../dictionary/en-sw.txt --rank 5
IndriRunQuery qmodel/phrase -index=index/ -count=$CUTOFF -trecFormat=true > result/result.first 
python param/trim_result.py #result.file
