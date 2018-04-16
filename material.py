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

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
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
    parser.add_argument('--target','-tgt', default='none', help='target language [sw,tl,en]')
    parser.add_argument('--collection','-c', default='en', help='language of documents [analysis, dev, eval1,eval2, clef]')
    parser.add_argument('--out','-o', default='en', help='output language [sw,tl,en]')
    parser.add_argument('--method','-method', default='duet+', help='method [qt,dt,duet+]')
    parser.add_argument('--model','-model', default='duet+', help='method [googl,wiktionary,duet+]')
    parser.add_argument('--query_list', '-q', nargs='*',help='check query result [query Id list]')
    parser.add_argument('--phase', default='train', help='Phase: Can be train or predict, the default value is train.')

    args = parser.parse_args()

    current_dir = os.getcwd()
    print(current_dir)

    if args.source == 'en' and args.target == 'sw' and args.collection == 'analysis' and args.method == 'dt':
        model_file ="examples/sw/config/analysis_dt_ranking.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        phrase_script = config['global']['phrase_script']
        directory = config['global']['directory']
        result_folder = config['inputs']['predict']['result_folder']
        os.chdir(directory)
        call(["sh",phrase_script,"1000"])
        call(["cp",result_folder+'result.file',os.path.join(current_dir,'predict.test.duet_ranking.txt')])
        os.chdir(current_dir)
    elif args.source == 'en' and args.target == 'sw' and args.collection == 'eval1' and args.method == 'dt':
        model_file ="examples/sw/config/eval1_dt_ranking.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        phrase_script = config['global']['phrase_script']
        directory = config['global']['directory']
        result_folder = config['inputs']['predict']['result_folder']
        os.chdir(directory)
        call(["sh",phrase_script,"1000"])
        call(["cp",result_folder+'result.file',os.path.join(current_dir,'predict.test.duet_ranking.txt')])
        os.chdir(current_dir)
    elif args.source == 'en' and args.target == 'sw' and args.collection == 'eval2' and args.method == 'dt':
        model_file ="examples/sw/config/eval2_dt_ranking.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        phrase_script = config['global']['phrase_script']
        directory = config['global']['directory']
        result_folder = config['inputs']['predict']['result_folder']
        os.chdir(directory)
        call(["sh",phrase_script,"1000"])
        call(["cp",result_folder+'result.file',os.path.join(current_dir,'predict.test.duet_ranking.txt')])
        os.chdir(current_dir)
    elif args.source == 'en' and args.target == 'sw' and args.collection == 'dev' and args.method == 'dt':
        model_file ="examples/sw/config/dev_dt_ranking.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        phrase_script = config['global']['phrase_script']
        directory = config['global']['directory']
        result_folder = config['inputs']['predict']['result_folder']
        os.chdir(directory)
        call(["sh",phrase_script,"1000"])
        call(["cp",result_folder+'result.file',os.path.join(current_dir,'predict.test.duet_ranking.txt')])
        os.chdir(current_dir)
    elif args.source == 'en' and args.target == 'tl' and args.collection == 'analysis' and args.method == 'dt':
        model_file ="examples/tl/config/analysis_dt_ranking.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        phrase_script = config['global']['phrase_script']
        directory = config['global']['directory']
        result_folder = config['inputs']['predict']['result_folder']
        os.chdir(directory)
        call(["sh",phrase_script,"1000"])
        call(["cp",result_folder+'result.file',os.path.join(current_dir,'predict.test.duet_ranking.txt')])
        os.chdir(current_dir)
    elif args.source == 'en' and args.target == 'tl' and args.collection == 'eval1' and args.method == 'dt':
        model_file ="examples/tl/config/eval1_dt_ranking.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        phrase_script = config['global']['phrase_script']
        directory = config['global']['directory']
        result_folder = config['inputs']['predict']['result_folder']
        os.chdir(directory)
        call(["sh",phrase_script,"1000"])
        call(["cp",result_folder+'result.file',os.path.join(current_dir,'predict.test.duet_ranking.txt')])
        os.chdir(current_dir)
    elif args.source == 'en' and args.target == 'tl' and args.collection == 'eval2' and args.method == 'dt':
        model_file ="examples/tl/config/eval2_dt_ranking.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        phrase_script = config['global']['phrase_script']
        directory = config['global']['directory']
        result_folder = config['inputs']['predict']['result_folder']
        os.chdir(directory)
        call(["sh",phrase_script,"1000"])
        call(["cp",result_folder+'result.file',os.path.join(current_dir,'predict.test.duet_ranking.txt')])
        os.chdir(current_dir)
    elif args.source == 'en' and args.target == 'tl' and args.collection == 'dev' and args.method == 'dt':
        model_file ="examples/tl/config/dev_dt_ranking.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        phrase_script = config['global']['phrase_script']
        directory = config['global']['directory']
        os.chdir(directory)
        result_folder = config['inputs']['predict']['result_folder']
        call(["sh",phrase_script,"1000"])
        call(["cp",result_folder+'result.file',os.path.join(current_dir,'predict.test.duet_ranking.txt')])
        os.chdir(current_dir)
    elif args.source == 'en' and args.target == 'sw' and args.collection == 'analysis' and args.method=='qt' and args.model == 'google':
        model_file ="examples/sw/config/analysis_qt_google_ranking.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        google_script = config['global']['google_script']
        directory = config['global']['directory']
        os.chdir(directory)
        result_folder = config['inputs']['predict']['result_folder']
        call(["sh",google_script])
        call(["cp",result_folder+'result.file',os.path.join(current_dir,'predict.test.duet_ranking.txt')])
        os.chdir(current_dir)
    elif args.source == 'en' and args.target == 'sw' and args.collection == 'eval1' and args.method=='qt' and args.model == 'google':
        model_file ="examples/sw/config/eval1_qt_google_ranking.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        google_script = config['global']['google_script']
        directory = config['global']['directory']
        os.chdir(directory)
        result_folder = config['inputs']['predict']['result_folder']
        call(["sh",google_script])
        call(["cp",result_folder+'result.file',os.path.join(current_dir,'predict.test.duet_ranking.txt')])
        #call(["cp",result_folder+'result.file','predict.test.duet_ranking.txt'])
        os.chdir(current_dir)
    elif args.source == 'en' and args.target == 'sw' and args.collection == 'eval2' and args.method=='qt' and args.model == 'google':
        model_file ="examples/sw/config/eval2_qt_google_ranking.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        google_script = config['global']['google_script']
        directory = config['global']['directory']
        os.chdir(directory)
        result_folder = config['inputs']['predict']['result_folder']
        call(["sh",google_script])
        call(["cp",result_folder+'result.file',os.path.join(current_dir,'predict.test.duet_ranking.txt')])
        #call(["cp",result_folder+'result.file','predict.test.duet_ranking.txt'])
        os.chdir(current_dir)
    elif args.source == 'en' and args.target == 'sw' and args.collection == 'dev' and args.method=='qt' and args.model == 'google':
        model_file ="examples/sw/config/dev_qt_google_ranking.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        google_script = config['global']['google_script']
        directory = config['global']['directory']
        os.chdir(directory)
        result_folder = config['inputs']['predict']['result_folder']
        call(["sh",google_script])
        call(["cp",result_folder+'result.file',os.path.join(current_dir,'predict.test.duet_ranking.txt')])
        #call(["cp",result_folder+'result.file','predict.test.duet_ranking.txt'])
        os.chdir(current_dir)
    elif args.source == 'en' and args.target == 'tl' and args.collection == 'analysis' and args.method=='qt' and args.model == 'google':
        model_file ="examples/tl/config/analysis_qt_google_ranking.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        google_script = config['global']['google_script']
        directory = config['global']['directory']
        os.chdir(directory)
        result_folder = config['inputs']['predict']['result_folder']
        call(["sh",google_script])
        call(["cp",result_folder+'result.file',os.path.join(current_dir,'predict.test.duet_ranking.txt')])
        #call(["cp",result_folder+'result.file','predict.test.duet_ranking.txt'])
        os.chdir(current_dir)
    elif args.source == 'en' and args.target == 'tl' and args.collection == 'eval1' and args.method=='qt' and args.model == 'google':
        model_file ="examples/tl/config/eval1_qt_google_ranking.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        google_script = config['global']['google_script']
        directory = config['global']['directory']
        os.chdir(directory)
        result_folder = config['inputs']['predict']['result_folder']
        call(["sh",google_script])
        call(["cp",result_folder+'result.file',os.path.join(current_dir,'predict.test.duet_ranking.txt')])
        os.chdir(current_dir)
    elif args.source == 'en' and args.target == 'tl' and args.collection == 'eval2' and args.method=='qt' and args.model == 'google':
        model_file ="examples/tl/config/eval2_qt_google_ranking.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        google_script = config['global']['google_script']
        directory = config['global']['directory']
        os.chdir(directory)
        result_folder = config['inputs']['predict']['result_folder']
        call(["sh",google_script])
        call(["cp",result_folder+'result.file',os.path.join(current_dir,'predict.test.duet_ranking.txt')])
        os.chdir(current_dir)
    elif args.source == 'en' and args.target == 'tl' and args.collection == 'dev' and args.method=='qt' and args.model == 'google':
        model_file ="examples/tl/config/dev_qt_google_ranking.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        google_script = config['global']['google_script']
        directory = config['global']['directory']
        os.chdir(directory)
        result_folder = config['inputs']['predict']['result_folder']
        call(["sh",google_script])
        call(["cp",result_folder+'result.file',os.path.join(current_dir,'predict.test.duet_ranking.txt')])
        os.chdir(current_dir)
    elif args.source == 'en' and args.target == 'tl' and args.collection == 'sw' and args.method == 'wiktionary':
        model_file ="examples/sw/config/duet_ranking_wiktionary.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        wiktionary_script = config['global']['wiktionary_script']
        result_folder = config['inputs']['predict']['result_folder']
        call(["sh",wiktionary_script])
        call(["cp",result_folder+'result.file','predict.test.duet_ranking.txt'])
    elif args.source == 'en' and args.target == 'tl' and args.collection == 'tl' and args.method == 'wiktionary':
        model_file ="examples/tl/config/duet_ranking_wiktionary.config"
        with open(model_file, 'r') as f:
            config = json.load(f)
        wiktionary_script = config['global']['wiktionary_script']
        result_folder = config['inputs']['predict']['result_folder']
        call(["sh",wiktionary_script])
        call(["cp",result_folder+'result.file','predict.test.duet_ranking.txt'])

    else:
        print('-src source language [sw,tl,en] -tgt target language [sw,tl,en] -c collection language [sw,tl,en] -m method [mt,google,wiktionary,fastext]')
        return
    
    judg_file = config['inputs']['predict']['judg_file']
    result_folder = config['inputs']['predict']['result_folder']
    with open("eval.sh", 'w') as f:
        f.write("trec_eval -q "+judg_file+" -m map -m P.5,10 -m aqwv predict.test.duet_ranking.txt > eval.test.duet_ranking.txt")
    f.close()
    
    if args.collection!="eval1" and args.collection!="eval2":
        call(["sh","eval.sh"])
        os.remove("eval.sh")
        with open("eval.test.duet_ranking.txt", 'r') as f:
            for line in f.readlines():
                metric,qid,val  = line.rstrip().split('\t')
                if qid =='all':
                    print(metric+'='+val)

    result_name="results/"+'_'.join(sys.argv[1:])[1:]+'.txt'
    call(["cp","eval.test.duet_ranking.txt",result_name+"_eval"])
    call(["cp","predict.test.duet_ranking.txt",result_name+"_predict"])
    # check result by ID
    if args.query_list:
        call(["python", "single_query_result.py", "-q", ' '.join(args.query_list)])
    return

if __name__=='__main__':
    main(sys.argv)
