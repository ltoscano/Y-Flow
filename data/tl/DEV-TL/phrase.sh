#! /bin/sh

CUTOFF=$1

python src/phrase.py --phrase False --src_lang en --tgt_lang tl --query ../topics/QUERY1/query_list.tsv --tquery qmodel/phrase --dico_train ../../dictionary/en-tl-fast.txt --rank 5
IndriRunQuery qmodel/phrase -index=index/ -count=$CUTOFF -trecFormat=true > result/result.first 
python param/trim_local_result.py #result.file
trec_eval -c -q -N 704 ../judg/rel.dev result/result.file > phrase.out

echo 'generate ranking data...'
./generate_ranking_data.sh

CUDA_VISIBLE_DEVICES=0 python material.py
