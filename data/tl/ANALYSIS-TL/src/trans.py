import os
import json
import argparse
from collections import OrderedDict
import torch
from fastext import FastVector 
import re
from utils import bool_flag, initialize_exp
from utils import load_external_embeddings
from models import build_model
from trainer import Trainer
from word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary
from gensim.models import KeyedVectors
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

import numpy as np
VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'


# main
parser = argparse.ArgumentParser(description='Supervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=bool_flag, default=True, help="Export embeddings after training")
# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size")
# training refinement
parser.add_argument("--n_iters", type=int, default=5, help="Number of iterations")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_train", type=str, default="default", help="Path to training dictionary (default: use identical character strings)")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=10000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default='', help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Reload target embeddings")
parser.add_argument("--query", type=str, default='', help="Reloud source query file")
parser.add_argument("--tquery", type=str, default='', help="Reloud source query file")
parser.add_argument("--model", type=str, default='', help="[multi,fastext]")
parser.add_argument("--src_align", type=str, default='', help="Reload source alignments")
parser.add_argument("--tgt_align", type=str, default='', help="Reload target alignments")

parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")


# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert params.dico_train in ["identical_char", "default"] or os.path.isfile(params.dico_train)
assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size

assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)

if(params.model=="multi"):
    src_model = KeyedVectors.load_word2vec_format(params.src_emb)
    tgt_model = KeyedVectors.load_word2vec_format(params.tgt_emb)
elif(params.model=="fastext"):
    src_model = FastVector(vector_file=params.src_emb)
    tgt_model = FastVector(vector_file=params.tgt_emb)
    src_model.apply_transform(params.src_align)
    tgt_model.apply_transform(params.tgt_align)

ps = PorterStemmer()
dictionary = {}
with open(params.dico_train) as f:
    for line in tqdm(f):
        word, trans = line.rstrip().split(' ')
        word = ps.stem(word).encode('ascii', 'ignore')
        if(word in dictionary):
            dictionary[word].append(trans)
        else:
            dictionary[word] = [trans]
#print(dictionary[ps.stem('condolences').encode('ascii','ignore')])
#input('here')
out = open(params.tquery,'w')
out.write("<parameters>\n")
with open(params.query) as f:
    i=1
    for line in f:
        line = re.sub('[(){},;:"?]','',line)
        tokens = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",line.rstrip())
        qid = tokens[0]
        did = tokens[-1]
        words = tokens[1:len(tokens)-1]
        print(qid,did,words)
        #input('here') 
        out.write("<query>\n")
        out.write("<type>indri</type>\n")
        out.write("<number>{0}</number>\n".format(qid))
        out.write("<text>\n")
        out.write("#combine(\n")
        
        for word in words:
            out.write("#wsyn(\n")
            #for t in dictionary[word]:
            if params.model=="multi":
                if word in src_model:
                    #print('yes')
                    #input('here')
                    for similar_word in tgt_model.similar_by_vector(src_model[word]):
                        w_= re.sub('[<>\\(){},;:"?/\'#]',' ',similar_word[0])
                        out.write("{0:.2f} {1}\n".format(
                        similar_word[1],w_.encode('utf-8').strip()))
                else:
                    #print('no')
                    #input('here')
                    out.write("{0:.2f} {1}\n".format(1.0,word))
            else: #params.model=="fastext":
                stem = ps.stem(word).encode('ascii', 'ignore')
                if stem in dictionary and word in src_model:
                    norm =0.0
                    for t in dictionary[stem]:
                        if t in tgt_model:
                            norm += np.exp(FastVector.cosine_similarity(src_model[word],tgt_model[t]))

                    for t in dictionary[stem]:
                        if t not in tgt_model:
                            out.write("{0:.2f} {1}\n".format(0.0,t.strip()))
                            continue
                        out.write("{0:.2f} {1}\n".format(np.exp(FastVector.cosine_similarity(src_model[word],tgt_model[t]))/norm,t.strip()))
                else:
                    out.write("{0:.2f} {1}\n".format(1.0,word))

            out.write(")\n")
        out.write(")\n")
        out.write("</text>\n")
        out.write("</query>\n")
        i+=1

out.write("</parameters>")
out.close()
print('done')
"""
Learning loop for Procrustes Iterative Refinement
"""


