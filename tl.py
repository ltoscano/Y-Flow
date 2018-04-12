a = {}
b=[]
file = open("sheet","w") 
 
with open("en.t") as f:
    for l in f.readlines():
        print(l.split(" ")[0])
        a[l.split(" ")[0]] = l.rstrip().split(" ")[1:]

with open("map") as f:
    for l in f.readlines():
            print(a[l.split("\t")[0].rstrip()])
            file.write(str(a[l.split("\t")[0].rstrip()])+"\n")   
   
'''with open("umd") as f:
    for l in f.readlines():
        if(l.rstrip() in a):
            file.write(a[l.rstrip()]+"\n")   
        else:
            file.write("0\n")'''
file.close()

