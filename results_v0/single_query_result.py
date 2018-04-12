import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--query_list', '-q', nargs='*',default='', help='check query result [query Id list or *')

# read in results
scores = {}
result=open('eval.test.duet_ranking.txt','r')
for line in result.readlines():
    line = line.strip().split('\t')
    id = line[1]
    type = line[0].strip()
    score = line[2]

    if id == 'all':
        if 'all' in scores.keys():
            d = scores['all']
        else:
            d = {}
        d[type] = score
        scores['all'] = d
    else:
        if id in scores.keys():
            d = scores[id]
        else:
            d = {}
        d[type] = score
        scores[id] = d

args = parser.parse_args()

query_list = args.query_list[0].split(' ')

if len(query_list) > 0:
    print 'Checking results for ', query_list
    # print the list
    for item in query_list:
        if item in scores.keys():
            print item,':', 'map',scores[item]['map'],';', 'P_5',scores[item]['P_5'],';', 'P_10',scores[item]['P_10'],';'
        else:
            print item, ':', ' not in collection!'

