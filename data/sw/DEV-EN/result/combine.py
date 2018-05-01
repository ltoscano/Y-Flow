
import numpy as np
from nltk.tokenize import RegexpTokenizer
import itertools
from nltk.tokenize import word_tokenize
import re
import os
import nltk
from heapq import heappush, heappop
import sys
import operator
# query is .txt file of query terms
# text is .txt file of text
# distance searches for the words in query in text, then gives a score based on the total distance between
# the first and last word in a path.  All paths include all words in q in order
# ALL terms in query are ASSUMED to be present somewhere in text


# Load dictionary with query terms and relevant documents
# Format: queryID ? file_name rank score
q2text={}
d2text={}
q2docs1={}
q2docs2score1={}
q2docs2={}
q2docs2score2={}
query = set()
with open(sys.argv[1]) as f:
    for line in f.readlines():
        tokens = line.split(' ')
        qid = tokens[0]
        did = tokens[2]
        score = tokens[4]
        query.add(qid)
        if qid in q2docs1:
            q2docs1[qid].append(did)
            q2docs2score1[qid].append((did,float(score)))
        else:
            q2docs1[qid] = [did]
            q2docs2score1[qid] = [(did,float(score))]

with open(sys.argv[2]) as f:
    for line in f.readlines():
        tokens = line.split(' ')
        qid = tokens[0]
        did = tokens[2]
        score = tokens[4]
        query.add(qid)
        if qid in q2docs2:
            q2docs2[qid].append(did)
            q2docs2score2[qid].append((did,float(score)))
        else:
            q2docs2[qid] = [did]
            q2docs2score2[qid] = [(did,float(score))]

try:
    os.remove("result.combination")
except:
    pass

for qid in query:
    scores = {}
    if qid in q2docs2score2:
        for d1,s1 in q2docs2score2[qid]:
            scores[d1] = s1

    if qid in q2docs2score1:
        for d2,s2 in q2docs2score1[qid]:
            if d2 not in scores:
                scores[d2] = s2
            else:
                scores[d2] = 0.5*scores[d2]+0.5*s2

    sorted_doc_list = sorted(scores.items(), key=operator.itemgetter(1),reverse=True)
    with open("result.combination", 'a') as f:
        rank = 1
        for doc,score in sorted_doc_list:
            f.write(qid+" Q0 "+ doc + " "+str(rank)+" " + str(score)+ " indri\n")
            rank +=1

