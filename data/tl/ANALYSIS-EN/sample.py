from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import re
import os
import nltk
nltk.download('punkt')
tokenizer = RegexpTokenizer(r'\w+')

w = open('sample.txt','w') 
q2d  = {}
docs = {}
q2text = {}
with open('../topics/en.t') as f:
    for line in f.readlines():
        tokens = tokenizer.tokenize(line)
        #print(a)
        #tokens = line.rstrip().split(' ')
        #Text = ''
        #for x in  tokens[1:len(tokens)-1]:
        #    Text = Text.join(' '+x)
        Text = ' '.join(tokens[1:len(tokens)-1])
        print(tokens[0],Text)
        q2text[tokens[0]] =  Text #''.join(word_tokenize(f)) for f in tokens[1:len(tokens)-1]
f.close()

with open('../judg/rel.judg') as f:
    for line in f.readlines():
        tokens = line.rstrip().split(' ')
        if tokens[0] in q2d:
            q2d[tokens[0]].append(tokens[2])
        else:
            q2d[tokens[0]] = [tokens[2]]
f.close()
for filename in os.listdir('docs/'):
    if filename.endswith(".txt"): 
        with open(os.path.join('docs/', filename)) as f:
            data = f.read().replace('\r\n',' ')
            m = re.search('MATERIAL_BASE(.+?)\.', filename)
            if m:
                filename = 'MATERIAL_BASE'+m.group(1)
            docs[filename] = str(data)#''.join(str(w) for w in data)
            #print(docs[filename])
        f.close()
for q in q2d:
    if q not in q2text:
        continue
    for d in docs:
        if d in q2d[q]:
            w.write('1\t'+q2text[q]+'\t'+docs[d]+'\n')
        else:
            w.write('0\t'+q2text[q]+'\t'+docs[d]+'\n')
w.close()
