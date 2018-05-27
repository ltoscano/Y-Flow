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


def crossval(config,split):
    origResult = {}
    pairSet = set()
    with open('result/result.file') as f:
        for line in f.readlines():
            tokens = line.rstrip().split(' ')
            if tokens[0] not in origResult:
                pairSet.add((tokens[0],tokens[2]))
                origResult[tokens[0]]=[(tokens[2],tokens[4])] # doc and score
            else:
                pairSet.add((tokens[0],tokens[2]))
                origResult[tokens[0]].append((tokens[2],tokens[4])) # doc and score

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
    #num_test = int(total_rel * 0.2)
    num_test = total_rel - num_train
    pivot = int(float(split)* total_rel)
    rel_test = relations[pivot:pivot+num_test]
    del relations[pivot:pivot+num_test]
    rel_train = relations

    print('************',total_rel,num_test,num_train,split,pivot, len(rel_test))
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
            if (x[1],x[2]) not in pairSet:#these examples have been added from judg file
                continue
            for y in x:
                f.write(y+' ')
            f.write('\n')
    f.close()
    return rel_train, rel_test

def trimResultDuet(duetFile,resultFile):
    duetResult = {}
    origResult = {}
    pairSet = set()
    from collections import OrderedDict
    with open(duetFile) as f:
        for line in f.readlines():
            tokens = line.rstrip().split('\t')
            if tokens[0] not in duetResult:
                duetResult[tokens[0]]=[(tokens[2],tokens[4])] # doc and score
            else:
                duetResult[tokens[0]].append((tokens[2],tokens[4])) # doc and score
    with open(resultFile) as f:
        for line in f.readlines():
            tokens = line.rstrip().split(' ')
            if tokens[0] not in origResult:
                pairSet.add((tokens[0],tokens[2]))
                origResult[tokens[0]]=[(tokens[2],tokens[4])] # doc and score
            else:
                pairSet.add((tokens[0],tokens[2]))
                origResult[tokens[0]].append((tokens[2],tokens[4])) # doc and score
    os.remove(resultFile)           
    uniqPairSet = set()
    with open(resultFile,'w') as f:
        for query in duetResult:
            i=1
            for doc,score in duetResult[query]: 
                if ((query,doc) in pairSet) and ((query,doc) not in uniqPairSet):
                    f.write(query+' Q0 '+doc+' '+str(i)+' '+score+' indri\n')
                    uniqPairSet.add((query,doc))
                    i+=1
    
if __name__ == '__main__':
    for i in [0.0,0.2,0.4,0.6,0.8]:
        model_file ="config/duet_ranking.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        crossval(config,i)
        call(["python", "yflow/main.py", "--phase", "train" ,"--model_file", model_file, "--fold", str(i)]) # _en,_tl,_sw
        call(["python", "yflow/main.py" ,"--phase", "predict", "--model_file", model_file, "--fold", str(i)]) # _en, _tl,_sw
        call(["mv", "result/predict.test.duet_ranking.txt","result/predict."+str(i)+".txt"])
    call("cat result/predict.0* > result/result.duet",shell=True)
    judg_file = config['inputs']['predict']['judg_file']
    trimResultDuet('result/result.duet','result/result.file')
    result_folder = config['inputs']['predict']['result_folder']
    with open("eval.sh", 'w') as f:
        f.write("trec_eval -q "+judg_file+" -m map -m P.5,10 -m aqwv result/result.file > duet.out")
    f.close()
    call(["sh","eval.sh"])
    os.remove("eval.sh")
    with open("duet.out", 'r') as f:
        for line in f.readlines():
            metric,qid,val  = line.rstrip().split('\t')
            if qid =='all':
                print(metric+'='+val)
    f.close()
