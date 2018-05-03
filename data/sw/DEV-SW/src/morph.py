import os
import json
import argparse
from collections import OrderedDict
#import torch
from fastext import FastVector 
import re
from utils import bool_flag, initialize_exp
from utils import load_external_embeddings
from models import build_model
from trainer import Trainer
from word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary
from gensim.models import KeyedVectors
import numpy as np
import itertools
VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'


# main
parser = argparse.ArgumentParser(description='Supervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=bool_flag, default=True, help="Export embeddings after training")
parser.add_argument("--phrase", type=bool_flag, default=False, help="Respect Phrases")
# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size")
# training refinement
parser.add_argument("--n_iters", type=int, default=50000, help="Number of iterations")
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
parser.add_argument("--model", type=str, default='', help="[multi,fastext]")
parser.add_argument("--src_align", type=str, default='', help="Reload source alignments")
parser.add_argument("--tgt_align", type=str, default='', help="Reload target alignments")
parser.add_argument("--rank", type=str, default='', help="top-rank translations")

parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")


# parse parameters
params = parser.parse_args()

# check parameters
#assert not params.cuda or torch.cuda.is_available()
assert params.dico_train in ["identical_char", "default"] or os.path.isfile(params.dico_train)
assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
print('generating Indri query format ..')

dictionary = {}
from nltk import PorterStemmer 
stemmer = PorterStemmer()

q2text={}
d2text={}
q2docs={}
q2docs2score={}

with open('result/result.file') as f:
    for line in f.readlines():
        tokens = line.split(' ')
        qid = tokens[0]
        did = tokens[2]
        score = tokens[4]
        if qid in q2docs:
            q2docs[qid].append(did)
            q2docs2score[qid].append((did,float(score)))
        else:
            q2docs[qid] = [did]
            q2docs2score[qid] = [(did,float(score))]

for filename in os.listdir('docs/'):
    if filename.endswith(".txt"): 
        with open(os.path.join('docs/', filename)) as f:
            data = f.read().replace('\r\n',' ').replace('\n',' ').replace('\t',' ')
            m = re.search('MATERIAL_BASE(.+?)\.', filename)
            if m:
                filename = 'MATERIAL_BASE'+m.group(1)
            d2text[filename] = str(data)#''.join(str(w) for w in data)
        f.close()




with open(params.query) as f:
    i = 0
    for line in f:
        line = line.strip().lower()
        print(line)
        if i == 0:
            i = 1
            continue## query first line unnecessary
        qid = tokens[0]
        words = tokens[1:-1] ## skip domain
        
        
        
        if i == params.n_iters:
            break
        i += 1

out.write("</parameters>")
out.close()
print('done')
"""
Learning loop for Procrustes Iterative Refinement
"""


