#! /bin/sh

CUTOFF=$1
QUERY='../topics/QUERY1/query_list.tsv'
if [ -n "$2" ]; then
  QUERY=$2
fi


python src/mono.py --src_lang en --tgt_lang en --query $QUERY --tquery qmodel/mono --model mono
IndriRunQuery qmodel/mono -index=index/ -count=$CUTOFF -trecFormat=true > result/result.first
python param/trim_result.py # generate result.file
rm result/result.first
trec_eval -q -N 666 ../judg/rel.dev result/result.file > phrase.out.cutoff.$CUTOFF
#python indri_to_nist_result.py
#cut -f 8 tools/SCORING/* | grep -vE "(Beta|AQWV)" | awk 'NF'  > tools/SCORING/all.AQWVscores.tsv
#cut -f 8 tools/SCORING/* | grep -vE "(Beta|AQWV)" | awk 'NF'| awk '{T+= $NF} END { print T/NR }'