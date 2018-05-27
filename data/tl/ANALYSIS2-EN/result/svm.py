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
parser.add_argument("--input", type=str, default=-1, help="result file")
parser.add_argument("--train", type=str, default=-1, help="result file")
parser.add_argument("--judg", type=str, default=-1, help="judgment file")
parser.add_argument("--emb", type=str, default=-1, help="word embeddings file")
parser.add_argument("--topic", type=str, default=-1, help="query to word file")

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
e=defaultdict(list)
m=50 #length of word embedding
with open(params.emb) as f:
    for line in f.readlines():
        result = []
        tokens = line.rstrip().split(" ")
        word = tokens[0]
        for i in range(0, m):
            result.append(float(tokens[i+1])) # store each value of the word embedding vector
        e[word] = result

# read in query to word mapping
query_to_word=defaultdict(list)
with open(params.topic) as f:
    for line in f.readlines():
        tokens = line.rstrip().replace(","," ").replace("[", " ").replace("]"," ").replace(">"," ").replace("<"," ").replace("+"," ").replace('"', " ").replace("EXAMPLE_OF("," ").replace(")"," ").split()
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
for i in q:
    AQWV_scores = []
    xi = np.zeros(m + n + 1)
    k=0
    for doc in q[i]:
        if k==n:
            break
        xij = doc[2] # indri score
        xi[k]=np.exp(float(xij))
        k+=1
    xi = xi/sum(xi)

    #add word embeddings
    terms = query_to_word[i]
    print i
    print terms
    all_emb = []
    emb = []
    len_terms = len(terms)
    for k in range(0, m):
        all_emb.append(0)
    for term in terms:
        term_emb = e[term.lower()]
        print term
        print term_emb
        if (len(term_emb) == m):
            for j in range(0, m):
                all_emb[j] += term_emb[j]
        else:
            len_terms -=1
    if (len_terms > 0):
        for j in range(0,m):
            emb.append(all_emb[j] / len_terms)
        
        j = 0
        for count in range(n, m+n):
            xi[count] = emb[j]
            j+=1
        xi[m+n]=training_size

        print "Xi: " + str(xi)
        X.append(xi)
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
        best_cutoff = np.argmax(AQWV_scores)+1
        y.append(best_cutoff)
clf.fit(X, y) # THE ERROR IS COMING FROM HERE
q_test_features=defaultdict(list)
q_test_docs=defaultdict(list)
with open(params.input) as f:
    for line in f.readlines():
        result = []
        tokens = line.rstrip().split(" ")
        result.append(tokens[2])
        result.append(tokens[3])
        result.append(tokens[4]) # document, rank, score
        query = tokens[0]
        q_test_features[query].append(np.exp(float(tokens[4])))
        q_test_docs[query].append(result)

out = open(params.input+".optimized",'w')
for query in q_test_features:
    # print "Query: " + str(query)
    xi=np.zeros(m+n+1)
    scores = q_test_features[query]/sum(q_test_features[query])
    for a in range(len(scores)):
        if a==n:
            break
        xi[a] = scores[a]

    #add word embeddings here
    terms = query_to_word[re.sub('[A-Za-z]+:','',query)]
    all_emb = []
    emb = []
    len_terms = len(terms)
    for i in range(0, m):
        all_emb.append(0)
    for term in terms:
        term_emb = e[term.lower()]
        if (len(term_emb) == m):
            for j in range(0, m):
                all_emb[j] += term_emb[j]
        else:
            len_terms -= 1

    if (len_terms > 0):
        for j in range(0,m):
            emb.append(all_emb[j] / len_terms)
        j = 0
        for count in range(n, m+n):
            xi[count] = emb[j]
            j+=1
        xi[m+n]=training_size
        
        best_cutof = clf.predict([xi])
        print(query,best_cutoff) 
        r = 0
        for doc in q_test_docs[query]:
            if r >= best_cutoff:
                break
            out.write(query + " Q0 ")
            for j in range(len(doc)):
                out.write(doc[j]+" ")
            out.write("indri\n")
            r +=1
            out.close

f.close()
