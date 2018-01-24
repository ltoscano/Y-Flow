#rm tools/SCORING/*
#rm tools/REFERENCE/*
#rm tools/VALIDATION/*
#python src/wiktionary.py --dico_train sw.csv --tquery qmodel/sw_google --query sw.csv --rank 5
python ./data/sw/ANALYSIS-SW/src/wiktionary.py --src_lang en --tgt_lang sw --query ./data/sw/topics/en.t --tquery ./data/sw/ANALYSIS-SW/qmodel/wiktionary --dico_train ./data/dictionary/en-sw.txt --rank 5
IndriRunQuery ./data/sw/ANALYSIS-SW/qmodel/wiktionary -index=./data/sw/ANALYSIS-SW/index/ -count=20 -trecFormat=true > ./data/sw/ANALYSIS-SW/result/result.first
python ./data/sw/ANALYSIS-SW/param/trim_result.py #result.file
#imv ./data/sw/ANALYSIS-SW/result/result.file result/result.wiktionary
#trec_eval -q ../judg/rel.judg result/result.wiktionary > wiktionary.out
#python indri_to_nist_result.py
#cut -f 8 tools/SCORING/* | grep -vE "(Beta|AQWV)" | awk 'NF'  > tools/SCORING/all.AQWVscores.tsv
#cut -f 8 tools/SCORING/* | grep -vE "(Beta|AQWV)" | awk 'NF'| awk '{T+= $NF} END { print T/NR }'
