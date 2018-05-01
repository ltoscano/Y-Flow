
import numpy as np
from nltk.tokenize import RegexpTokenizer
import itertools
from nltk.tokenize import word_tokenize
import re
import os
import nltk
from heapq import heappush, heappop

# query is .txt file of query terms
# text is .txt file of text
# distance searches for the words in query in text, then gives a score based on the total distance between
# the first and last word in a path.  All paths include all words in q in order
# ALL terms in query are ASSUMED to be present somewhere in text

def distance(query, text):

	# Tokenize text and query. Ignore punctuation and make lowercase
	tokenizer = RegexpTokenizer(r'\w+')
	q = tokenizer.tokenize(query)
	txt = tokenizer.tokenize(text)
	q = [item.lower() for item in q]
	txt = [item.lower() for item in txt]

	q = np.asarray(q)
	txt = np.asarray(txt)

	# Make lists of locations of each word in query
	lsts = [[] for i in range(len(q))]

	for i in range(len(q)):
		lsts[i] = list(np.where(txt == q[i])[0])

	# Initialize score to worst value
	tot_score = 0


	# Calculate score of eligible paths
        list_= []
        visited=set()
        for indices in itertools.product(*lsts):
		ind = list(indices)
                if(sorted(ind) == ind):
                    heappush(list_,(ind,ind[len(ind)-1]-ind[0]))
        while list_:
            ind , score= heappop(list_)
            is_visited = False
            for v in ind:
                if v in visited:
                    is_visited = True # this node has been involved in a path
            if is_visited:
                continue
            for i in range(len(ind)-1):
                tot_score += ind[i+1]-ind[i] # total sum of consequtive elements
            for v in ind:
                visited.add(v)
        print(np.log(len(txt)/((0.5+tot_score))))
        return np.log(len(txt)/(len(q)*(0.5+tot_score)))

#distance('hello world', 'hello this is world.')

######
# Create dict_old and dict_new for each query
# each entry holds a document name, a score, and the actual query words
# List of ordered documents for each query, indri in _old, updated in _new


# Load dictionary with query terms and relevant documents
# Format: queryID ? file_name rank score
q2text={}
d2text={}
q2docs={}
q2docs2score={}

with open('result/result.file') as f:
    for line in f.readlines():
        tokens = line.split(' ')
        qid = tokens[0]
        did = tokens[2]
        score = tokens[4]
        if qid in q2docs:
            q2docs[qid].append(did)
            q2docs2score[qid].append((did,float(score)))
        else:
            q2docs[qid] = [did]
            q2docs2score[qid] = [(did,float(score))]

for filename in os.listdir('docs/'):
    if filename.endswith(".txt"): 
        with open(os.path.join('docs/', filename)) as f:
            data = f.read().replace('\r\n',' ').replace('\n',' ').replace('\t',' ')
            m = re.search('MATERIAL_BASE(.+?)\.', filename)
            if m:
                filename = 'MATERIAL_BASE'+m.group(1)
            d2text[filename] = str(data)#''.join(str(w) for w in data)
        f.close()

with open("../topics/QUERY1/query_list.tsv") as f:
    i = 0
    for line in f:
        line = line.strip()
        phrases=re.findall(r'\"(.+?)\"', line)
        phrases = [re.sub('[<>]', '', phrase) for phrase in phrases if '[' not in phrase]## only query 3637
        for phrase in phrases:
            line = line.replace(phrase, '')
        line = re.sub('[(){};?<>]','',line)
        line = re.sub(',',' ',line)
        line = re.sub('[A-Za-z]+:','',line) ## get rid of hyp, syn etc
        tokens = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",line)
        if i == 0:
            i = 1
            continue## query first line unnecessary
        qid = tokens[0]
        words = tokens[1:-1] ## skip domain
        q_text = " ".join(words)
        for phrase in phrases:
            q_text += phrase
        q2text[qid] = q_text


		# put queryID into dict if not already there
		# put query terms into dict if not already there
		#file_quer = open("../topics/QUERY1/query_list.tsv")
		# append file_name into list under queryID
		# assumes ranking in file is best to worst

# For each query term in dict_old
	# For each element under that query (name)
		# Load the text file held in that element
		#file_txt = open("/data/projects/material/eval/Y-Flow/data/sw/ANALYSIS-EN/docs/"+name+"/")



#file_quer = open("../topics/QUERY1/query_list.tsv")
#print(file_quer.read())
try:
    os.remove("result/result.proximity")
except:
    pass

for qid in q2docs2score:
    doc_list = []
    for doc,score in q2docs2score[qid]:
            proximity = distance(q2text[qid], d2text[doc])
            new_score = 0.95*score + 0.03*proximity 
            doc_list.append((doc,new_score))
    sorted_doc_list = sorted(doc_list,key=lambda tup: -tup[1]) 
    with open("result/result.proximity", 'a') as f:
        rank = 1
        for doc,score in sorted_doc_list:
            f.write(qid+" Q0 "+ doc + " "+str(rank)+" " + str(score)+ " indri\n")
            rank +=1
            if rank > 20:
                break

