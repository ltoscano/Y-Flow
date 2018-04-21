CUTOFF=$1
QUERY='../topics/QUERY1/query_list.tsv'
if [ -n "$2" ]; then
  QUERY=$2
fi


python src/phrase.py --phrase True --src_lang en --tgt_lang en --query $QUERY --tquery qmodel/mono --model mono
IndriRunQuery qmodel/mono -index=index/ -count=5 -trecFormat=true > result/result.first
python param/trim_local_result.py # generate result.file
rm result/result.first
