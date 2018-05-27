query = {}
queryList = []
query2AQWV = {}
with open('../topics/QUERY1/query_list.tsv') as f:
    for line in f.readlines():
        tokens = line.split('\t')
        query[tokens[0]] = tokens[1]

with open('phrase.aqwv.out') as f:
    for line in f.readlines():
        tokens = line.split('\t')
        query2AQWV[tokens[0]] = tokens[1]
        queryList.append(tokens[0])


with open('query.txt','w') as f:
    for q in queryList:
        if q in query:
            #f.write(q+' '+query[q]+'\n')
            f.write(query[q]+'\n')
