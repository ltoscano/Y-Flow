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

def parse_line(self, line, delimiter='\t'):
        subs = line.split(delimiter)
        # print('subs: ', len(subs))
        if 3 != len(subs):
            raise ValueError('format of data file wrong, should be \'label,text1,text2\'.')
        else:
            return subs[0], subs[1], subs[2]


def crossval(config,split):
    #print(json.dumps(config, indent=2), end='\n')
    # read basic config
    rel_file = config['inputs']['all']['relation_file']
    relations = []
    with open(rel_file) as f:
        for line in f.readlines():
            line = line.rstrip()
            label, t1, t2 = line.split(' ')#parse_line(line,' ')
            relations.append((label, t1, t2))
    f.close()
    
    #random.shuffle(relations)
    total_rel = len(relations)
    num_train = int(total_rel * 0.8)
    num_test = int(total_rel * 0.2)
    pivot = int(float(split)* total_rel)
    test_end = num_train + num_test
    #pivot = int(split*total_rel)
    print('#######>',split,'<########')
    rel_test = relations[pivot:pivot+num_test]
    del relations[pivot:pivot+num_test]
    rel_train = relations
    '''if split+num_train < total_rel:
        rel_train = relations[split: split+num_train]
        rel_test = relations[split+num_train: split+num_train+num_test]
    else:
        rel_train = relations[split:]
        rel_train = rel_train+relations[:num_train-len(rel_train)]
        rel_test = relations[split-num_test: split]'''
    return rel_train, rel_test


 

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='0', help='Split point: split point to break training and test data')
    parser.add_argument('--phase', default='train', help='Phase: Can be train or predict, the default value is train.')
    parser.add_argument('--model_file', default='./models/arci.config', help='Model_file: Y-Flow model file for the chosen model.')
    args = parser.parse_args()
    model_file =  args.model_file
    with open(model_file, 'r') as f:
        config = json.load(f)
    phase = args.phase
    #print(args.split)
    #input('here')
    if args.phase == 'predict':
        judg_file = config['inputs']['predict']['judg_file']
        result_folder = config['inputs']['predict']['result_folder']
        with open("eval.sh", 'w') as f:
            f.write("trec_eval -q "+judg_file+" -m map -m P.5,10 predict.test.duet_ranking.txt > eval.test.duet_ranking.txt")
        f.close()
        call(["sh","eval.sh"])
        call(["cp","predict.test.duet_ranking.txt","eval.test.duet_ranking.txt",result_folder])
        os.remove("eval.sh")
        with open("eval.test.duet_ranking.txt", 'r') as f:
            for line in f.readlines():
                metric,qid,val  = line.rstrip().split('\t')
                if qid =='all':
                    print(metric+'='+val)

        f.close()
        #call(["trec_eval"," -q ",judg_file," predict.test.duet_ranking.txt "," > "," eval.test.duet_ranking.txt"])
    else:
        print('Phase Error.', end='\n')
    return

if __name__=='__main__':
    main(sys.argv)
