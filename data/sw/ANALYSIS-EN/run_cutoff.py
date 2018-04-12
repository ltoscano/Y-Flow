import subprocess
import os
import sys
from argparse import ArgumentParser 
import numpy as np
from collections import defaultdict

parser = ArgumentParser()
parser.add_argument('judge', help='judgment file')
parser.add_argument('result_file', help='result file')
parser.add_argument('--new', dest='new_result_file', metavar='N', help='new result file', default='result/result-cutoff.file')
#parser.add_argument("--hard_cutoff",  help="hard cutoff", action="store_true", default=True)
parser.add_argument("--hard_k", dest="hard_k", help="rnn architecutre", type = int, default = 1000)
#parser.add_argument("--beta", dest="beta", help="rnn architecutre", type = float, default = 0.1)
opts = parser.parse_args()

def run_cutoff(opts):
    scores = get_result_file(opts.result_file)
    betas = np.arange(0.01, 10.01, 0.01)
    #betas = [10]
    max_score = -1000
    max_beta = 0
    for beta in betas:
        beta = round(beta, 3) 
        picked = beta_pruning(scores, beta)
        output_new_results(picked, opts.new_result_file)
        overall_score = get_sorted_results(opts)
        if overall_score > max_score:
            max_score = overall_score
            max_beta = beta
    print(max_beta, max_score)
def output_new_results(picked, new_result_file):
    with open(new_result_file, 'wt') as fout:
        for query_no, scores_query in picked.items():
            for j, score in enumerate(scores_query):
                output_line = ['query'+str(query_no), 'Q0', score[0], str(j+1), str(score[1]), 'indri']
                output_line = ' '.join(output_line)
                fout.write(output_line)
                fout.write('\n')
def beta_pruning(scores, beta):
    picked = defaultdict(list)
    for query_no, scores_query in scores.items():
        max_score =  -1000
        for doc_id, score in scores_query.items():
            if score > max_score:
                max_score = score
        for doc_id, score in scores_query.items():
            if score >= max_score-beta:
                picked[query_no].append((doc_id, score))
        picked[query_no] = sorted(picked[query_no], key=lambda x: -x[1])
    return picked

def get_result_file(result_file):
    ## scores[query_number][doc_id] = Indri score
    scores = defaultdict(dict)
    with open(result_file) as fin:
        for line in fin:
            line = line.split()
            try:
                scores[int(line[0][5:])][line[2][:-16]] = float(line[4])
            except:
                pass
    return scores
def get_sorted_results(opts):
    #command  = 'trec_eval -q {} {} -m map'.format(opts.judge, opts.new_result_file)
    #command  = 'trec_eval -q {} {} -m map'.format(opts.judge, opts.new_result_file)
    command  = 'trec_eval -q {} {} -N 471 -m aqwv'.format(opts.judge, opts.new_result_file)
    output = subprocess.check_output(command, shell=True).decode('utf-8')
    scores = []
    for line in output.split('\n'):
        line = line.split()
        if len(line) == 3:
            if line[1] != 'all':
                line[1] = int(line[1][5:])
                scores.append(line)
            else:
                all_score = line
    scores = sorted(scores, key=lambda x: x[1])
    scores.append(all_score)
    overall_score = scores[-1]
    return float(overall_score[2])
    #for score in scores:
    #    score[1] = str(score[1])
    #    print('\t'.join(score))

if __name__ == '__main__':
    run_cutoff(opts)
