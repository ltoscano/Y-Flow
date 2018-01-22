#rm tools/SCORING/*
#rm tools/REFERENCE/*
#rm tools/VALIDATION/*
python src/wiktionary.py --src_lang en --tgt_lang sw --query ../topics/en.t --tquery qmodel/wiktionary --dico_train ../../data/dictionary/en-sw.txt --rank 5
IndriRunQuery qmodel/wiktionary -index=index/ -count=20 -trecFormat=true > result/result.first
python param/trim_result.py #result.file
mv result/result.file result/result.wiktionary
trec_eval -q ../judg/rel.judg result/result.wiktionary > wiktionary.out
#python indri_to_nist_result.py
#cut -f 8 tools/SCORING/* | grep -vE "(Beta|AQWV)" | awk 'NF'  > tools/SCORING/all.AQWVscores.tsv
#cut -f 8 tools/SCORING/* | grep -vE "(Beta|AQWV)" | awk 'NF'| awk '{T+= $NF} END { print T/NR }'
