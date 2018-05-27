from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import re
import os
import nltk
nltk.download('punkt')
tokenizer = RegexpTokenizer(r'\w+')

query_tsv_path = '../topics/QUERY1/query_list.tsv'
rel_path = '../judg/rel.dev'
docs_path = 'docs.scripts.v2/'

w = open('sample.txt','w') 
q2d  = {}
docs = {}
q2text = {}
indri_q2d = {}

with open(query_tsv_path) as f:
    for line in f.readlines():
        line = re.sub('[(){};?<>]','',line)
        line = re.sub(',',' ',line)
        line = re.sub('[A-Za-z]+:','',line) ## get rid of hyp, syn etc
        tokens = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",line)
 
        #tokens = tokenizer.tokenize(line)
        Text = ' '.join(tokens[1:len(tokens)-1])
        print(tokens[0],Text)
        q2text[tokens[0]] =  Text #''.join(word_tokenize(f)) for f in tokens[1:len(tokens)-1]
f.close()

with open(rel_path) as f:
    for line in f.readlines():
        tokens = line.rstrip().split(' ')
        if tokens[0] in q2d:
            q2d[tokens[0]].append(tokens[2])
        else:
            q2d[tokens[0]] = [tokens[2]]
f.close()

with open('result/result.file') as f:
    for line in f.readlines():
        tokens = line.split(' ')
        qid = tokens[0]
        did = tokens[2]
        if qid in indri_q2d:
            indri_q2d[qid].append(did)
        else:
            indri_q2d[qid] = [did]
f.close() 
for filename in os.listdir(docs_path):
    if filename.endswith(".txt"): 
        with open(os.path.join(docs_path, filename)) as f:
            #data = f.read().replace('\r\n',' ')
            data = f.read().replace('\r\n',' ').replace('\n',' ').replace('\t',' ')
            m = re.search('MATERIAL_BASE(.+?)\.', filename)
            if m:
                filename = 'MATERIAL_BASE'+m.group(1)
            docs[filename] = str(data)#''.join(str(w) for w in data)
            #print(docs[filename])
        f.close()

text2id = {}

for q in q2d:
    if q not in q2text:
        continue
    for d in docs:
        if d in q2d[q]:
            w.write('1\t'+q2text[q]+'\t'+docs[d]+'\n')
        elif q in indri_q2d:
            if d in indri_q2d[q]:
                w.write('0\t'+q2text[q]+'\t'+docs[d]+'\n')
        else:
            w.write('0\t'+q2text[q]+'\t'+docs[d]+'\n')
        text2id[q] = q2text[q]
        text2id[d] = docs[d]

text2id_f = open('text2id.txt', 'w')
for text, newid in text2id.iteritems():
    text2id_f.write(text.strip().encode('utf-8'))
    text2id_f.write('\t')
    text2id_f.write(newid)
    text2id_f.write('\n')
print('Finished writing text2id.txt!')
text2id_f.close()
w.close()
