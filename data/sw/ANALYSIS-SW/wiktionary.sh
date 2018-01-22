rm tools/SCORING/*
rm tools/REFERENCE/*
rm tools/VALIDATION/*
python src/wiktionary.py --src_lang en --tgt_lang sw --query topics/toy.t --tquery qmodel/sw.wiktionary --dico_train data/crosslingual/dictionaries/en-sw.txt --rank 5
IndriRunQuery qmodel/sw.wiktionary -index=index/sw -count=100 -trecFormat=true > result/result.first
python param/trim_result.py #result.file
trec_eval -q judg/rel.judg result/result.file > wiktionary.out
python indri_to_nist_result.py
cut -f 8 tools/SCORING/* | grep -vE "(Beta|AQWV)" | awk 'NF'  > tools/SCORING/all.AQWVscores.tsv
cut -f 8 tools/SCORING/* | grep -vE "(Beta|AQWV)" | awk 'NF'| awk '{T+= $NF} END { print T/NR }'
