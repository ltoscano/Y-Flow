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


def crossval(config,q,split):
    rel_file = config['inputs']['all']['relation_file']
    relations = []
    with open(rel_file) as f:
        for line in f.readlines():
            line = line.rstrip()
            label, t1, t2 = line.split(' ')#parse_line(line,' ')
            relations.append((label, t1, t2))
    f.close()
    total_rel = len(relations)
    num_train = int(total_rel * 0.8)
    num_test = int(total_rel * 0.2)
    pivot = int(float(split)* total_rel)
    test_end = num_train + num_test
    rel_test = relations[pivot:pivot+num_test]
    del relations[pivot:pivot+num_test]
    rel_train = relations
    
        #rel_train, rel_test = crossval(config,args.split)
    rel_file = config['inputs']['train']['relation_file']
    with open(rel_file,'w') as f:
        for x in rel_train:
            for y in x:
                f.write(y+' ')
            f.write('\n')
    f.close()
    rel_file = config['inputs']['test']['relation_file']
    with open(rel_file,'w') as f:
        for x in rel_test:
            for y in x:
                f.write(y+' ')
            f.write('\n')
    f.close()
    return rel_train, rel_test

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source','-src', default='en', help='source language [sw,tl,en]')
    parser.add_argument('--query','-q', default='en', help='source language [sw,tl,en]')
    parser.add_argument('--target','-tgt', default='none', help='target language [sw,tl,en]')
    parser.add_argument('--collection','-c', default='en', help='language of documents [sw,tl,en]')
    parser.add_argument('--out','-o', default='en', help='output language [sw,tl,en]')
    parser.add_argument('--method','-m', default='none', help='method [mt,google,wiktionary,fastext]')
    parser.add_argument('--phase', default='train', help='Phase: Can be train or predict, the default value is train.')
    parser.add_argument('--type', '-type',default='speech', help='[speech,text]')

    args = parser.parse_args()

    if args.source == 'en' and args.target == 'tl' and args.collection == 'tl' and args.method == 'fastext':
        q = args.query
        for  i in {0.8}:
            model_file ="examples/tl/config/duet_ranking_tl_umd.config"
            with open(model_file, 'r') as f:
                config = json.load(f)
            
            with open('en.t','w') as f:
                f.write('query10000 '+q+' GOV')
            f.close()

            wiktionary_script = config['global']['wiktionary_script']
            result_folder = config['inputs']['predict']['result_folder']
            print(wiktionary_script)
            call(["sh",wiktionary_script])
            call(["cp",result_folder+'result.file','predict.test.duet_ranking.txt'])

            crossval(config,q,i)
            if args.phase == 'train':
                call(["python", "yflow/main.py", "--phase", "train" ,"--model_file", model_file, "--fold", str(i)]) # _en,_tl,_sw
            call(["python", "yflow/main.py" ,"--phase", "predict", "--model_file", model_file, "--fold", str(i)]) # _en, _tl,_sw
            call(["mv", "predict.test.duet_ranking.txt","predict."+str(i)+".txt"])
        #call("cat predict.0* > predict.test.duet_ranking.txt",shell=True)
        #call("rm predict.0*",shell=True)
    else:
        print('-src source language [sw,tl,en] -tgt target language [sw,tl,en] -c collection language [sw,tl,en] -m method [mt,google,wiktionary,fastext]')
        return
    
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

    result_name="results/"+'_'.join(sys.argv[1:])[1:]+'.txt'
    call(["cp","eval.test.duet_ranking.txt",result_name+"_eval"])
    call(["cp","predict.test.duet_ranking.txt",result_name+"_predict"])
    print('Result txt file saved in',result_name)
    # check result by ID
    return

if __name__=='__main__':
    main(sys.argv)
