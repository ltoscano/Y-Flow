CUTOFF=$1
QUERY='../topics/QUERY2/query_list.tsv'
if [ -n "$2" ]; then
  QUERY=$2
fi


python src/phrase.py --phrase True --src_lang en --tgt_lang en --query $QUERY --tquery qmodel/phrase --model phrase
IndriRunQuery qmodel/phrase -index=index/ -count=$CUTOFF -trecFormat=true > result/result.first
python param/trim_result.py # generate result.file
rm result/result.first
