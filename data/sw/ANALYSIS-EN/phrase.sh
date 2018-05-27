#! /bin/sh

CUTOFF=$1
QUERY='../topics/QUERY1/query_list.tsv'
if [ -n "$2" ]; then
  QUERY=$2
fi

python3 src/phrase.py --phrase False --src_lang en --tgt_lang en --query $QUERY  --tquery qmodel/mono --model mono --query_coeff 0.0
IndriRunQuery qmodel/mono -index=index/ -count=$CUTOFF -trecFormat=true > result/result.first
python param/trim_result.py # generate result.file
#rm result/result.first
trec_eval -q -N 471 ../judg/rel.analysis result/result.file -c > phrase.out
