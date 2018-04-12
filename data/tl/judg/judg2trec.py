f = open("query_annotation.tsv")
f_out = open("rel.dev",'w')

lines = f.readlines()
for line in lines[1:]:
  sp = line.strip().split()
  f_out.write(sp[0] + " 0 "+sp[1]+" 1\n")
f_out.close()
f.close()
