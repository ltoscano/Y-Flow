g = open('queries.txt','w')
qids = []
with open('tm') as f:
    for line in f.readlines():
        qids.append(line.rstrip())
with open('rel.judg') as f:
    for line in f.readlines():
        tokens = line.split(' ')
        if tokens[0] in qids:
            g.write(line)

g.close()

