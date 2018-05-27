import subprocess
import os, re
import sys
from argparse import ArgumentParser 
import numpy as np
from collections import defaultdict
from gensim.models import KeyedVectors

parser = ArgumentParser()
parser.add_argument('--judge', help='judgment file', default='../judg/rel.analysis')
parser.add_argument('--query', help='query file', default='../topics/QUERY2/query_list.tsv')
parser.add_argument('--cut_off', help='query file', type=int, default=5)
parser.add_argument("--phrase", type=bool, default=False, help="Respect Phrases")
opts = parser.parse_args()


def main(opts):
    new_q = opts.query+ '_expanded'
    query_ids = get_query_ids(opts.query)
    command = 'python src/phrase.py --phrase False --src_lang en --tgt_lang en --query {}  --tquery qmodel/mono --model mono'.format(opts.query)
    subprocess.check_call(command, shell=True)
    command = 'IndriRunQuery qmodel/mono -index=index.umd.smt/ -count={} -trecFormat=true'.format(opts.cut_off)
    output = subprocess.check_output(command, shell=True).decode('utf-8')
    missing_ids = get_missing_ids(output, query_ids)
    create_new_queries(missing_ids, opts.query, new_q)
    command = 'python src/phrase.py --phrase False --src_lang en --tgt_lang en --query {}  --tquery qmodel/mono --model mono'.format(new_q)
    subprocess.check_call(command, shell=True)
    command = 'IndriRunQuery qmodel/mono -index=index.umd.smt/ -count={} -trecFormat=true'.format(opts.cut_off)
    output_expanded = subprocess.check_output(command, shell=True).decode('utf-8')
    all_output = output + output_expanded
    #all_output = output_expanded
    with open('result/result.first', 'wt') as fout:
        fout.write(all_output)
    command = 'python param/trim_result.py'
    subprocess.check_call(command, shell=True)
    #command = 'rm result/result.first'
    #subprocess.check_call(command, shell=True)
    #command = 'trec_eval -c -q -N 471 ../judg/rel.analysis result/result.file -c > phrase.out'
    #subprocess.check_call(command, shell=True)
def get_missing_ids(output, query_ids):
    returned_ids = []
    for line in output.split('\n'):
        line = line.split()
        if len(line) >=1:
            query = line[0]
            if query[:5]=='query' and query!='query_id':
                returned_ids.append(int(query[5:]))
    return set(query_ids) - set(returned_ids)
def get_query_ids(query_file):
    query_ids = []
    with open(query_file) as fin:
        for line in fin:
            line = line.split()
            if len(line)>=1:
                query = line[0]
                if query[:5]=='query' and query!='query_id':
                    query_ids.append(int(query[5:]))
    return query_ids
def create_new_queries(missing_ids, old_q, new_q):
    word_vectors = KeyedVectors.load_word2vec_format('/data/projects/material/eval/Y-Flow/data/fastext/wiki.en.vec', binary=False, limit=50000)
    with open(new_q, 'wt') as fout:
        with open(old_q) as fin:
            for line in fin:
                line = line.strip()
                line = line.replace('EXAMPLE_OF', '')
                if line[:5] == 'query' and line[:8] != 'query_id':
                    qid = int(line.split()[0][5:])
                    if qid in missing_ids:
                
                        if opts.phrase:
                            phrases=re.findall(r'\"(.+?)\"', line)
                            phrases = [re.sub('[<>]', '', phrase) for phrase in phrases if '[' not in phrase]## only query 3637
                            for phrase in phrases:
                                line = line.replace(phrase, '') 
                        else:
                            phrases = []
                        line = re.sub('[(){};?<>]','',line)
                        line = re.sub(',',' ',line)
                        line = re.sub('[A-Za-z]+:','',line) ## get rid of hyp, syn etc
                        tokens = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",line)
                        words = tokens[1:-1] ## skip domain
                        new_words = []
                        for word in words:
                            try:
                                new_words.extend([x for x, sim in word_vectors.most_similar_cosmul(word.lower())[:10]])
                            except:
                                continue
                        #print(words)
                        #print(words)
                        #for phrase in phrases:
                        if len(new_words) >= 1:
                            line = '\t'.join([tokens[0]] + new_words + [tokens[-1]])
                            fout.write(line)
                            fout.write('\n')
                else:
                    fout.write(line)
                    fout.write('\n')


if __name__ == '__main__':
    main(opts)
