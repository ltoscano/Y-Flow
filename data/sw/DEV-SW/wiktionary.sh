CUTOFF=$1
python src/wiktionary.py --src_lang en --tgt_lang sw --query ../topics/en.t --tquery qmodel/wiktionary --dico_train ../../dictionary/en-sw.txt --rank 5
IndriRunQuery qmodel/wiktionary -index=index/ -count=$CUTOFF -trecFormat=true > result/result.first 
python param/trim_local_result.py #result.file
trec_eval -q -N 666 ../judg/rel.dev result/result.file > wiktionary.out
