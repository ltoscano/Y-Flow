#rm tools/SCORING/*
#rm tools/REFERENCE/*
#rm tools/VALIDATION/*

# for generating qmodel/tl_model
#python src/wiktionary.py --dico_train tl.csv --tquery qmodel/tl_google --query tl.csv --rank 5
python src/wiktionary.py --src_lang en --tgt_lang tl --query ../topics/en.t --tquery qmodel/tl_google --dico_train ../../dictionary/en-tl.txt --rank 5
IndriRunQuery qmodel/tl_google -index=index/ -count=20 -trecFormat=true > result/result.first
python param/trim_result.py #result.file
mv result/result.file result/result.google
trec_eval -q ../judg/rel.judg result/result.wiktionary > wiktionary.out
#python indri_to_nist_result.py
#cut -f 8 tools/SCORING/* | grep -vE "(Beta|AQWV)" | awk 'NF'  > tools/SCORING/all.AQWVscores.tsv
#cut -f 8 tools/SCORING/* | grep -vE "(Beta|AQWV)" | awk 'NF'| awk '{T+= $NF} END { print T/NR }'
