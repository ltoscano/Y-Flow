a = {}
b=[]
file = open("sheet","w") 
 
with open("fr.t") as f:
    i=1
    for l in f.readlines():
        a[i] = l.rstrip()    
        i+=1
'''with open("map") as f:
    for l in f.readlines():
        a[l.split("\t")[0].rstrip()] = l.split("\t")[1].rstrip()    '''

with open("map") as f:
    for l in f.readlines():
            file.write(a[int(l.rstrip())]+"\n")   
   
'''with open("umd") as f:
    for l in f.readlines():
        if(l.rstrip() in a):
            file.write(a[l.rstrip()]+"\n")   
        else:
            file.write("0\n")'''
file.close()

