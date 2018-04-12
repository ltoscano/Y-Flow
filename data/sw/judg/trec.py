g = open('judg.txt','w')
qids = {}
with open('rel.judg.dev') as f:
    for line in f.readlines():
        qids[line.rstrip().split("\t")[0]]=line.rstrip().split("\t")[1]
for a in qids:            
    g.write(a+" 0 "+qids[a]+" 1\n")
g.close()

