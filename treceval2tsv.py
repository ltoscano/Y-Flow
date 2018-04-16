import sys
import os
import argparse
import numpy as np

def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--tsv_dir', default='./result_tsv', help='')
  parser.add_argument('--topic_file', default='', help='')
  parser.add_argument('--domain_file', default='', help='')
  parser.add_argument('--annotation_file', default='', help='')
  parser.add_argument('--result_file', default='', help='')
  args = parser.parse_args()

  if not os.path.exists(args.tsv_dir):
    os.mkdir(args.tsv_dir)

  id2domain = {}
  f = open(args.domain_file)
  for l in f.readlines():
    sp = l.strip().split('\t')
    domainid = sp[0]
    if domainid == "domain_id":
      print 'skip'
      continue
    domain = sp[1]
    id2domain[domainid] = domain

  id2query = {}
  f = open(args.topic_file)
  for l in f.readlines():
    sp = l.strip().split('\t')
    qid = sp[0]
    if qid == "query_id":
      print 'skip'
      continue
    query = sp[1]
    domainid = sp[2]
    id2query[qid] = query+":"+id2domain[domainid]
  f.close()

  id2annotation = {}
  f = open(args.annotation_file)
  for l in f.readlines():
    sp = l.strip().split('\t')
    qid = sp[0]
    if qid == "query_id":
      print 'skip'
      continue
    docid = sp[1]
    if qid not in id2annotation:
      id2annotation[qid] = []
    id2annotation[qid].append(docid)
  f.close()

  f_out = open('query_list.tsv','w')
  for qid in sorted(id2query.keys()):
    num_gold_doc = 0
    if qid in id2annotation:
      num_gold_doc = len(id2annotation[qid])
    #f_out.write(qid+'\t'+str(num_gold_doc)+'\t'+id2query[qid]+'\n')
    f_out.write(str(num_gold_doc)+'\n')
  f_out.close()


  if args.result_file != '':
    id2result = {}
    id2sum = {}
    f = open(args.result_file)
    for l in f.readlines():
      sp = l.strip().split()
      qid = sp[0]
      doc = sp[2]
      score = np.exp(float(sp[4]))
      if qid not in id2result:
        id2result[qid] = []
      if qid not in id2sum:
        id2sum[qid] = 0.
      id2result[qid].append((doc,score))
      id2sum[qid] += score
    f.close()
    
    for qid in id2query:
      if qid in id2result:
        result_list = id2result[qid]
      else:
        result_list = []
      f_out = open(os.path.join(args.tsv_dir,'q-%s.tsv'%qid),'w')

      f_out.write('%s\t%s\n' % (qid,id2query[qid]))
      for doc,score in result_list:
        f_out.write('%s\t%.3f\n' % (doc,score/id2sum[qid]))

      f_out.close()

if __name__=='__main__':
  main(sys.argv)
