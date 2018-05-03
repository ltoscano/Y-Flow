#! /bin/sh

CUTOFF=$1
QUERY='../topics/QUERY2/query_list.tsv'
if [ -n "$2" ]; then
  QUERY=$2
fi


python src/phrase.py --phrase True --stemmer True --src_lang en --tgt_lang en --query $QUERY --tquery qmodel/phrase --model mono
IndriRunQuery qmodel/phrase -index=index.umd.smt/ -count=$CUTOFF -trecFormat=true > result/result.first
python param/trim_result.py # generate result.file
rm result/result.first
python src/morph.py --src_lang en --tgt_lang sw --query ../topics/QUERY2/query_list.tsv --query_morph ../topics/QUERY2/query_list_morph.txt
cp result/result.file ../../../submission/20180429/result.file.sw.dt.phrase100.cutoff$CUTOFF
trec_eval -q -N 666 ../judg/rel.dev result/result.file > phrase.out
#python indri_to_nist_result.py
#cut -f 8 tools/SCORING/* | grep -vE "(Beta|AQWV)" | awk 'NF'  > tools/SCORING/all.AQWVscores.tsv
#cut -f 8 tools/SCORING/* | grep -vE "(Beta|AQWV)" | awk 'NF'| awk '{T+= $NF} END { print T/NR }'
