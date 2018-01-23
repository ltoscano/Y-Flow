# -*- coding: utf8 -*-
from __future__ import print_function
import os
import sys
import time
import json
import argparse
import random
random.seed(49999)
import numpy
numpy.random.seed(49999)
import tensorflow
import codecs
tensorflow.set_random_seed(49999)
from tqdm import tqdm

from collections import OrderedDict

import keras
import keras.backend as K
from keras.models import Sequential, Model

from yflow.utils import *
from subprocess import call
#import yflow.inputs
#import yflow.metrics
#from yflow.losses import *

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
sess = tensorflow.Session(config = config)

def load_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    if model_type == 'JSON':
        mo = Model.from_config(config['model'])
    elif model_type == 'PY':
        model_config = config['model']['setting']
        model_config.update(config['inputs']['share'])
        sys.path.insert(0, config['model']['model_path'])

        model = import_object(config['model']['model_py'], model_config)
        mo = model.build()
    return mo


 

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source','-src', default='en', help='output language [sw,tl,en]')
    parser.add_argument('--target','-tgt', default='sw', help='output language [sw,tl,en]')
    parser.add_argument('--collection','-c', default='en', help='language of documents [sw,tl,en]')
    parser.add_argument('--out','-o', default='en', help='output language [sw,tl,en]')
    parser.add_argument('--method','-m', default='mt', help='method [mt,google,wiktionary,fastext]')
    
    args = parser.parse_args()
    if args.source == 'en' and args.target == 'sw' and args.collection == 'en' and args.method == 'mt':
        #call["sh","duet.c=sw.q=en.d=en.sh"]
        for  i in {0.0,0.2,0.4,0.6,0.8}:
            call["python", "5-fold.py", "--phase", "train", "--split", str(i), "--model_file", "examples/sw/config/duet_ranking_en.config"]
            call["python", "yflow/main.py", "--phase", "train" ,"--model_file", "examples/sw/config/duet_ranking_en.config"] # _en,_tl,_sw
            call["python", "yflow/main.py" ,"--phase", "predict", "--model_file", "examples/sw/config/duet_ranking_en.config"] # _en, _tl,_sw
            call["mv", "predict.test.duet_ranking.txt","predict."+str(i)+".txt"]
	call("cat predict.0* > predict.test.duet_ranking.txt",shell=True)
	call("rm predict.0*",shell=True)
        call["python", "aggregate.py", "--phase", "predict", "--model_file", "examples/sw/config/duet_ranking_en.config"] # _en, _tl,_sw
    elif args.source == 'en' and args.target == 'tl' and args.collection == 'en' and args.method == 'mt':
        for  i in {0.0,0.2,0.4,0.6,0.8}:
            call["python", "5-fold.py", "--phase", "train", "--split", str(i), "--model_file", "examples/tl/config/duet_ranking_en.config"]
            call["python", "yflow/main.py", "--phase", "train" ,"--model_file", "examples/tl/config/duet_ranking_en.config"] # _en,_tl,_sw
            call["python", "yflow/main.py" ,"--phase", "predict", "--model_file", "examples/tl/config/duet_ranking_en.config"] # _en, _tl,_sw
            call["mv", "predict.test.duet_ranking.txt","predict."+str(i)+".txt"]
	call("cat predict.0* > predict.test.duet_ranking.txt",shell=True)
	call("rm predict.0*",shell=True)
        call["python", "aggregate.py", "--phase", "predict", "--model_file", "examples/tl/config/duet_ranking_en.config"] # _en, _tl,_sw
    elif args.source == 'en' and args.target == 'sw' and args.collection == 'sw' and args.method == 'fastext':
        for  i in tqdm({0.0,0.2,0.4,0.6,0.8}):
            call(["python","5-fold.py","--phase","train","--split",str(i), "--model_file", "examples/sw/config/duet_ranking_sw.config"])
            call(["python", "yflow/main.py", "--phase", "train" ,"--model_file", "examples/sw/config/duet_ranking_sw.config"]) # _en,_tl,_sw
            call(["python", "yflow/main.py" ,"--phase", "predict", "--model_file", "examples/sw/config/duet_ranking_sw.config"]) # _en, _tl,_sw
            call(["mv", "predict.test.duet_ranking.txt","predict."+str(i)+".txt"])
	call("cat predict.0* > predict.test.duet_ranking.txt",shell=True)
	call("rm predict.0*",shell=True)
        call(["python", "aggregate.py", "--phase", "predict", "--model_file", "examples/sw/config/duet_ranking_sw.config"]) # _en, _tl,_sw
    elif args.source == 'en' and args.target == 'tl' and args.collection == 'tl' and args.method == 'fastext':
        for  i in {0.0,0.2,0.4,0.6,0.8}:
            call(["python", "5-fold.py", "--phase", "train", "--split",str(i), "--model_file", "examples/tl/config/duet_ranking_tl.config"])
            call(["python", "yflow/main.py", "--phase", "train" ,"--model_file", "examples/tl/config/duet_ranking_tl.config"]) # _en,_tl,_sw
            call["python", "yflow/main.py" ,"--phase", "predict", "--model_file", "examples/tl/config/duet_ranking_tl.config"] # _en, _tl,_sw
            call["mv", "predict.test.duet_ranking.txt","predict."+str(i)+".txt"]
	call("cat predict.0* > predict.test.duet_ranking.txt",shell=True)
	call("rm predict.0*",shell=True)
        call["python", "aggregate.py", "--phase", "predict", "--model_file", "examples/tl/config/duet_ranking_tl.config"] # _en, _tl,_sw
    else:
        print(args.source,args.target,args.collection,args.method)
        print('Phase Error.', end='\n')
    return

if __name__=='__main__':
    main(sys.argv)
