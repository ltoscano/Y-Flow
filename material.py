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
    parser.add_argument('--lang', '-l',default='sw', help='document collection [sw,tl]')
    parser.add_argument('--out','-o', default='en', help='output language [sw,tl,en]')
    parser.add_argument('--method','-m', default='mt', help='method [mt,google,wiktionary]')
    parser.add_argument('--pipeline','-p', default='en2en', help='end-2-end pipeline [en2en,en2sw,en2tl]')
    
    args = parser.parse_args()
    if args.lang == 'sw' and args.method == 'mt' and args.pipeline == 'en2en':
        call["sh","duet.c=sw.q=en.d=en.sh"]

        for  i in {0.0,0.2,0.4,0.6,0.8}:
            call["python", "5-fold.py", "--phase", "train", "--split", i, "--model_file", "examples/sw/config/duet_ranking_en.config"]
            call["python", "yflow/main.py", "--phase", "train" ,"--model_file", "examples/sw/config/duet_ranking_en.config"] # _en,_tl,_sw
            call["python", "yflow/main.py" ,"--phase", "predict", "--model_file", "examples/sw/config/duet_ranking_en.config"] # _en, _tl,_sw
            call["mv", "predict.test.duet_ranking.txt" "predict."+i+".txt"]
	call["cat", "predict.0*", ">", "predict.test.duet_ranking.txt"]
	call["rm", "predict.0*"]
	call["python", "aggregate.py", "--phase", "predict", "--model_file", "examples/sw/config/duet_ranking_en.config"] # _en, _tl,_sw
    elif args.lang == 'tl' and args.method == 'mt' and args.pipeline == 'en2en':
        call["sh","duet.c=tl.q=en.d=en.sh"]
    else:
        print('Phase Error.', end='\n')
    return

if __name__=='__main__':
    main(sys.argv)
