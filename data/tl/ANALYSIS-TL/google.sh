#rm tools/SCORING/*
#rm tools/REFERENCE/*
#rm tools/VALIDATION/*
IndriRunQuery qmodel/google -index=index/ -count=20 -trecFormat=true > result/result.first
python param/trim_result.py #result.file
trec_eval -q ../judg/rel.judg result/result.file > google.out
#python indri_to_nist_result.py
#cut -f 8 tools/SCORING/* | grep -vE "(Beta|AQWV)" | awk 'NF'  > tools/SCORING/all.AQWVscores.tsv
#cut -f 8 tools/SCORING/* | grep -vE "(Beta|AQWV)" | awk 'NF'| awk '{T+= $NF} END { print T/NR }'
