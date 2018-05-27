import subprocess
import os, re
import sys
from argparse import ArgumentParser 
import numpy as np
from collections import defaultdict

parser = ArgumentParser()
parser.add_argument('judge', help='judgment file')
#parser.add_argument('result_file', help='result file')
parser.add_argument("--hard_k", dest="hard_k", help="rnn architecutre", type = int, default = 1000)
#parser.add_argument("--beta", dest="beta", help="rnn architecutre", type = float, default = 0.1)
opts = parser.parse_args()

def tune_k(opts):
    max_score = -1000
    #for k in xrange(1, 51):
    for k in [1000]:
        score = top_k(opts, k)

def top_k(opts, k):
    command = 'python src/wiktionary.py --src_lang en --tgt_lang sw --query ../topics/en.t --tquery qmodel/wiktionary --dico_train ../../dictionary/en-sw.txt --rank 5'
    subprocess.check_call(command, shell=True)
    command = 'IndriRunQuery qmodel/wiktionary -index=index/ -count={} -trecFormat=true > result/result.jungo'.format(k)
    subprocess.check_call(command, shell=True)
    trim('result/result.jungo', 'result/result.full')
def trim(in_file, out_file):
    g = open(out_file,'w')
    with open(in_file) as f:
        for line in f.readlines():
            tokens = line.rstrip().split(' ')
            a =set()
            if 'query' not in tokens[0]:
                continue

            tokens[2] = tokens[2].split('/')[-1]
            m = re.search('MATERIAL_BASE(.+?)\.', tokens[2])
            if m:
                tokens[2] = 'MATERIAL_BASE'+m.group(1)
                if tokens[2] in a:
                    continue
                a.add(tokens[2])
            for t in tokens:
                g.write(t+' ')
            g.write('\n')
    f.close()
    g.close()

if __name__ == '__main__':
    tune_k(opts)
