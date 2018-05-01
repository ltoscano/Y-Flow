import os
import json
import argparse
from collections import OrderedDict
from collections import defaultdict
import re
#from gensim.models import KeyedVectors
import numpy as np
from sklearn import svm
from subprocess import call
import subprocess
VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'


# main
parser = argparse.ArgumentParser(description='Supervised training')
parser.add_argument("--train", type=str, default='result.file', help="result file")
parser.add_argument("--judg", type=str, default='../../judg/rel.dev', help="judgment file")
parser.add_argument("--topic", type=str, default='../../topics/QUERY1/query_list.tsv', help="query to word file")

# parse parameters
params = parser.parse_args()

# check parameters
#X = [[0, 0], [1, 1],[0.5,0.1]]
#y = [0, 1,2]
q=defaultdict(list)
qid2cutoff={}
training_size=0
with open(params.train) as f:
    for line in f.readlines():
        training_size +=1
        result = []
        tokens = line.rstrip().split(" ")
        result.append(tokens[2])
        result.append(tokens[3])
        result.append(tokens[4]) # document, rank, score
        query = tokens[0]
        q[query].append(result)

# read in word embeddings for each word
# read in query to word mapping
query_to_word=defaultdict(list)
with open(params.topic) as f:
    for line in f.readlines():
        tokens = line.rstrip().replace(","," ").replace("[", " ").replace("]"," ").replace(">"," ").replace("<"," ").replace("+"," ").replace('"', " ").replace("EXAMPLE_OF("," ").replace(")"," ").split()
        # split().split(",").split("[").split("]")
        # tokens = re.findall(, line.rstrip())
        query = tokens[0]
        words = []
        for i in range(1, len(tokens) - 1):
            words.append(re.sub('[A-Za-z]+:','',tokens[i]))
        query_to_word[query] = words

n=10
cutoff = [i+1 for i in range(n)]

X = []
y=[]
clf = svm.SVC(decision_function_shape='ovo')
best_cutoff_map = {}
totall_sum_cutoff =0.0
total_sum_query = 0.0
for i in q:
    AQWV_scores = []
    
    for c in cutoff:
        out = open("query.tsv",'w')
        r = 0
        for doc in q[i]:
            if r > c:
                break
            out.write(i + " Q0 ")
            for j in range(len(doc)):
                out.write(doc[j]+" ")
            out.write("indri\n")
            r +=1
        out.close
        with open("eval.sh", 'w') as f:
            f.write("trec_eval -q -N 471 "+params.judg+" -m aqwv query.tsv > predict.tsv")
        f.close()
        call(["sh","eval.sh"])

        with open("predict.tsv") as f:
            for line in f.readlines(): #it's possible to have empty file
                aqwv = float(line.split("\t")[2])
                AQWV_scores.append(aqwv)
                break

        os.remove("eval.sh")
    best_cutoff =0
    if len(AQWV_scores)>0:
        best_cutoff = np.argmax(AQWV_scores)+1
        totall_sum_cutoff += best_cutoff
        total_sum_query += 1
    best_cutoff_map[i] = best_cutoff
    print(i,best_cutoff)
with open("cutoff.out", 'w') as f:
    for i in best_cutoff_map:
        #print(best_cutoff_map[i])
        f.write('(\'%s\''% i)
        f.write(',%s)' % best_cutoff_map[i])
        f.write("\n")

print('overall_cutoff==',(float)(totall_sum_cutoff/total_sum_query))
