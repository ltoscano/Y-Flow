import re
import os
w = open('corpus.txt','w') 
for filename in os.listdir('docs/'):
    if filename.endswith(".txt"): 
        with open(os.path.join('docs/', filename)) as f:
            data = f.read().replace('\n',' ')
            m = re.search('MATERIAL_BASE(.+?)\.', filename)
            if m:
                filename = 'MATERIAL_BASE'+m.group(1)
            w.write(filename+' '+data+'\n')
        f.close()
w.close()
