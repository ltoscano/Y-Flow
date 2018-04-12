python google.py --src_lang en --tgt_lang tl --query ../topics/google.t --tquery qmodel/google
IndriRunQuery qmodel/google -index=index/ -count=20 -trecFormat=true > result/result.first
python param/trim_result.py #result.file
