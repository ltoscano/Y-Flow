import subprocess
import os, re
import sys
from argparse import ArgumentParser 
import numpy as np
from collections import defaultdict

parser = ArgumentParser()
parser.add_argument('--query_file', dest='query_file', metavar='N', help='new result file', default='../topics/QUERY1/query_list.tsv')
#parser.add_argument("--beta", dest="beta", help="rnn architecutre", type = float, default = 0.1)
opts = parser.parse_args()


def simple_phrase(opts):
    command = ''.format(opts.query_file)
    subprocess.check_call(command, shell=True)
    command = 'IndriRunQuery qmodel/wiktionary -index=index/ -count={} -trecFormat=true > result/result.jungo'.format(k)
    subprocess.check_call(command, shell=True)
    trim('result/result.jungo', 'result/result.jungo.eval')
    command  = 'trec_eval {} {} -N 471 -m aqwv'.format(opts.judge, 'result/result.jungo.eval')
    output = subprocess.check_output(command, shell=True).decode('utf-8')
    output = output.split()
    return float(output[2])
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
