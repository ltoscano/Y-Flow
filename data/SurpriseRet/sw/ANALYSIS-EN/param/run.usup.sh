CUDA_VISIBLE_DEVICES=0  MP_NUM_THREADS=5  THEANO_FLAGS=device=cuda0 flags=-lopenblas python supervised.py --src_lang en --tgt_lang sw --src_emb data/fastext/wiki.en.vec --tgt_emb data/fastext/wiki.sw.vec --n_iter 5 --dico_train data/crosslingual/dictionaries/en-sw.txt

