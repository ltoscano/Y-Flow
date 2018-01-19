#!/bin/bash

# generate matchzoo data for ranking
multi_en="../../multi/wiki.en.vec"    # /   (root directory)
multi_tl="../../multi/wiki.tl.vec"    # /   (root directory)
if [ -f "$multi_en" ] 
then
	echo "loading existing multilingual word-embedding .."
	#echo "$multi_en found."
else
	echo "creating multilingual word-embedding .."
	python src/supervised.py --src_lang en --tgt_lang tl --src_emb ../../fastext/wiki.en.vec --tgt_emb ../../fastext/wiki.tl.vec --exp_path ../../multi --n_iter 10 --dico_train ../../dictionary/en-tl.txt
fi

python src/trans.py --src_lang en --tgt_lang tl --query ../topics/en.t --tquery qmodel/fastext --model multi --src_emb ../../multi/wiki.en.vec --tgt_emb ../../multi/wiki.tl.vec --dico_train ../../dictionary/en-tl.txt 
IndriRunQuery qmodel/fastext -index=index/ -count=20 -trecFormat=true > result/result.first
python param/trim_result.py # generate result.file
rm result/result.first

python sample.py
python test_preparation_for_ranking.py

# 1. download embedding 
#wget http://nlp.stanford.edu/data/glove.6B.zip
#unzip glove.6B.zip
#mv glove.6B.50d.txt ../../data/toy_example/ranking/

#wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
#mv wiki.en.vec ../../fastext/

# 2. map word embedding
python gen_w2v.py ../../multi/wiki.en.vec word_dict.txt embed_fastext_d300
python norm_embed.py embed_fastext_d300 embed_fastext_d300_norm

# 3. run to generate histogram for DRMM
python test_histogram_generator.py  'ranking'

# 4. run to generate tri-grams for DSSM or CDSSM
#python test_triletter_preprocess.py 'ranking'

# 5. run to generate binsum for aNMM
#python test_binsum_generator.py 'ranking'

