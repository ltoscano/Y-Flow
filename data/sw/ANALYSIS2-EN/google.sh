python  src/google.py --src_lang en --tgt_lang sw --query ../topics/google.t --tquery qmodel/google
IndriRunQuery qmodel/google -index=index/ -count=100 -trecFormat=true > result/result.first
python param/trim_result.py #result.file
