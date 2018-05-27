#! /bin/sh
export TF_CPP_MIN_LOG_LEVEL=2

CUTOFF=$1
QUERY='../topics/QUERY1/query_list.tsv'


python src/phrase.py --phrase False --stemmer True --src_lang en --tgt_lang en --query $QUERY --tquery qmodel/phrase
IndriRunQuery qmodel/phrase -index=index.scripts.v2/ -count=$CUTOFF -trecFormat=true > result/result.first
python param/trim_result.py # generate result.file
rm result/result.first
#python src/morph.py --src_lang en --tgt_lang sw --query ../topics/QUERY2/query_list.tsv --query_morph ../topics/QUERY2/query_list_morph.txt
#cp result/result.file ../../../submission/20180511/result.file
trec_eval -c -q -N 666 ../judg/rel.dev result/result.file > phrase.out

