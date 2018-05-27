#! /bin/sh
export TF_CPP_MIN_LOG_LEVEL=2

CUTOFF=$1
QUERY='../topics/QUERY1/query_list.tsv'
if [ -n "$2" ]; then
  QUERY=$2
fi

python src/phrase.py --phrase True --src_lang en --tgt_lang sw --query $QUERY --tquery qmodel/phrase --dico_train ../../dictionary/en-sw.txt --rank 5
IndriRunQuery qmodel/phrase -index=index/ -count=$CUTOFF -trecFormat=true > result/result.first 
python param/trim_result.py #result.file
#cp result/result.file ../../../submission/20180511/result.file
#python src/morph.py --src_lang en --tgt_lang sw --query ../topics/QUERY2/query_list_morph.txt
trec_eval -q -N 666 ../judg/rel.dev result/result.file > phrase.out
#trec_eval -q -N 666 ../judg/rel.dev result/result.file > phrase.out.cutoff.$CUTOFF


echo 'generate ranking data...'
./generate_ranking_data.sh

CUDA_VISIBLE_DEVICES=0 python material.py
