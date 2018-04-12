python src/mono.py --src_lang en --tgt_lang en --query ../topics/en.t --tquery qmodel/mono --model mono 
IndriRunQuery qmodel/mono -index=index/ -count=1000 -trecFormat=true > result/result.first
python param/trim_local_result.py # generate result.file
#rm result/result.first
