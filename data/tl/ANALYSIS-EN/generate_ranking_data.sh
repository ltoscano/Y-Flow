#!/bin/bash

# generate matchzoo data for ranking
sh run.sh
python sample.py
python test_preparation_for_ranking.py

# 1. download embedding 
#wget http://nlp.stanford.edu/data/glove.6B.zip
#unzip glove.6B.zip
#mv glove.6B.50d.txt ../../data/toy_example/ranking/

#wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
#mv wiki.en.vec ../../fastext/

# 2. map word embedding
python gen_w2v.py ../../fastext/wiki.en.vec word_dict.txt embed_fastext_d300
python norm_embed.py embed_fastext_d300 embed_fastext_d300_norm

# 3. run to generate histogram for DRMM
python test_histogram_generator.py  'ranking'
cat relation_train.txt relation_test.txt relation_valid.txt > relation_all.txt 
# 4. run to generate tri-grams for DSSM or CDSSM
#python test_triletter_preprocess.py 'ranking'

# 5. run to generate binsum for aNMM
#python test_binsum_generator.py 'ranking'

# 6. delete text2id.txt file
rm text2id.txt


