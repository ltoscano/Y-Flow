import os, io
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
parser.add_argument("--tquery", type=str, default='', help="Reloud source query file")
parser.add_argument("--model", type=str, default='', help="[multi,fastext]")
parser.add_argument("--src_align", type=str, default='', help="Reload source alignments")
parser.add_argument("--tgt_align", type=str, default='', help="Reload target alignments")

parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
parser.add_argument("--query_coeff", type=float, default=1.0, help="coefficient between query and muse expansion")
parser.add_argument("--stemmer", type=bool, default=False, help="Respect Phrases")
parser.add_argument("--expansion", type=bool_flag, default=False, help="Respect Phrases")


# parse parameters
params = parser.parse_args()

# check parameters
#assert not params.cuda or torch.cuda.is_available()
assert params.dico_train in ["identical_char", "default"] or os.path.isfile(params.dico_train)
assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
print('loading word vectors')
word_vectors = KeyedVectors.load_word2vec_format('/data/projects/material/eval/Y-Flow/data/fastext/wiki.en.vec', binary=False, limit=10)
vocabs = word_vectors.vocab.keys()
print('generating Indri query format ..')
out = open(params.tquery,'w')

from krovetzstemmer import Stemmer
stemmer = Stemmer()

out.write("<parameters>\n")
with open(params.query) as f:
    i = 0
    for line in f:
        line = line.strip().lower()
        line = line.replace('example_of', '')
        conceptual = True
        non_hyp = True
        if '+' in line:
            if not 'syn' in line:
                conceptual = False
        if 'hyp' in line:
            non_hyp = False
        if params.phrase:
            phrases=re.findall(r'\"(.+?)\"', line)
            phrases = [re.sub('[<>]', '', phrase) for phrase in phrases if '[' not in phrase]## only query 3637
            for phrase in phrases:
                line = line.replace(phrase, '')
        else:
            phrases = []
        line = re.sub('[(){};?<>]','',line)
        line = re.sub(',',' ',line)
        #print(line)
        line = re.sub('[A-Za-z]+:','',line) ## get rid of hyp, syn etc
        tokens = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",line)
        if i == 0:
            i = 1
            continue## query first line unnecessary
        qid = tokens[0]
        #did = tokens[-1]
        words = tokens[1:-1] ## skip domain
        #print(qid,did,words)
        #input('here') 
        out.write("<query>\n")
        out.write("<type>indri</type>\n")
        out.write("<number>{0}</number>\n".format(qid))
        out.write("<text>\n")
        #out.write("#combine(\n")
        for phrase in phrases:
            out.write("#1({})".format(phrase))
            out.write('\n')
        vocab_words = []
        for word in words:
            if word in vocabs:
                vocab_words.append(word)
        if vocab_words:
            word_sim = word_vectors.most_similar(positive=vocab_words)[:10]
            new_words = [x[0] for x in word_sim]
            weights = np.array([x[1] for x in word_sim])
            weights = softmax(weights)
        else:
            new_words = []
        for word in words:
            out.write(stemmer.stem(word)+"\n")
        #out.write('#combine(\n')
        if params.expansion:
            #if non_hyp and conceptual:
            j = 0
            out.write('#weight(\n')
            for word in words:
                out.write(str(params.query_coeff)+' ' + stemmer.stem(word)+'\n')
            for new_word in new_words:
               out.write(str(weights[j]*(1-params.query_coeff))+' '+stemmer.stem(str(new_word.encode('utf-8').strip()))+'\n')
               j+=1
            out.write(")\n")
        #out.write(")\n")
        #out.write(')\n')
        out.write("</text>\n")
        out.write("</query>\n")
        if i == params.n_iters:
            break
        i += 1

out.write("</parameters>")
out.close()
print('done')
"""
Learning loop for Procrustes Iterative Refinement
"""


