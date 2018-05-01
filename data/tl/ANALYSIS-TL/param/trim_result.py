import re

g = open('./result/result.file','w')
q=[]
with open('../judg/rel.analysis') as f:
    for line in f.readlines():
        tokens = line.rstrip().split(' ')
        if 'query' not in tokens[0]:
            continue
        q.append(tokens[0]) 

with open('./result/result.first') as f:
    for line in f.readlines():
        tokens = line.rstrip().split(' ')
        a =set()
        if tokens[0] not in q:
            continue
        if 'query' not in tokens[0]:
            continue
        
        tokens[2] = tokens[2].split('/')[-1]
        m = re.search('MATERIAL_BASE(.+?)\.', tokens[2])
        if m:
            tokens[2] = 'MATERIAL_BASE'+m.group(1)
            if tokens[2] in a:
                continue
            a.add(tokens[2])
        for t in tokens:
            g.write(t+' ')
        g.write('\n')
f.close()
g.close()


        

    
