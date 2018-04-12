python src/phrase.py --phrase True --src_lang en --tgt_lang en --query ../topics/QUERY1/query_list.tsv --tquery qmodel/mono --model mono
IndriRunQuery qmodel/mono -index=index/ -count=5 -trecFormat=true > result/result.first
python param/trim_local_result.py # generate result.file
rm result/result.first
trec_eval -q -N 471 ../judg/rel.judg result/result.file > phrase.out
#python indri_to_nist_result.py
#cut -f 8 tools/SCORING/* | grep -vE "(Beta|AQWV)" | awk 'NF'  > tools/SCORING/all.AQWVscores.tsv
#cut -f 8 tools/SCORING/* | grep -vE "(Beta|AQWV)" | awk 'NF'| awk '{T+= $NF} END { print T/NR }'
