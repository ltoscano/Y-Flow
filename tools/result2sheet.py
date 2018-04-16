import os

def get_id2query(query_file):
  id2query = {}
  f = open(query_file)
  for l in f.readlines():
    sp = l.strip().split('\t')
    qid = sp[0]
    if qid == "query_id":
      print 'skip'
      continue
    query = sp[1]
    domainid = sp[2]
    id2query[qid] = query
  f.close()
  return id2query

def get_metric(result_file):
  f = open(result_file)
  ap = {}
  p_5 = {}
  p_10 = {}
  aqwv = {}
  for line in f.readlines():
    sp = line.strip().split()
    metric = sp[0]
    qid = sp[1]
    try:
      value = float(sp[2])
    except:
      continue
    if metric == "map":
      ap[qid] = value
    if metric == "aqwv":
      aqwv[qid] = value
    if metric == "P_5":
      p_5[qid] = value
    if metric == "P_10":
      p_10[qid] = value
  f.close()
  return ap, p_5, p_10, aqwv

def get_result(data_dir):
  if 'sw' in data_dir:
    id2query = get_id2query('../data/sw/topics/QUERY1/query_list.tsv')
  elif 'tl' in data_dir:
    id2query = get_id2query('../data/tl/topics/QUERY1/query_list.tsv')

  ap,p_5,p_10,aqwv1 = get_metric(os.path.join(data_dir,'phrase.out.cutoff1'))
  ap,p_5,p_10,aqwv5 = get_metric(os.path.join(data_dir,'phrase.out.cutoff5'))
  ap,p_5,p_10,aqwv10 = get_metric(os.path.join(data_dir,'phrase.out.cutoff10'))
  ap,p_5,p_10,aqwv20 = get_metric(os.path.join(data_dir,'phrase.out.cutoff20'))

  return_str = ""

  qids = sorted([qid for qid in id2query.keys() if qid != 'all']) + ['all']
  #print qids
  for qid in qids:
    if qid in ap:
      q_ap = ap[qid]
    else:
      q_ap = 0
    return_str += "%.4f\t" % q_ap

    if qid in p_5:
      q_p5 = p_5[qid]
    else:
      q_p5 = 0
    return_str += "%.4f\t" % q_p5

    if qid in p_10:
      q_p10 = p_10[qid]
    else:
      q_p10 = 0
    return_str += "%.4f\t" % q_p10

    if qid in aqwv1:
      q_aqwv1 = aqwv1[qid]
    else:
      q_aqwv1 = 0
    return_str += "%.4f\t" % q_aqwv1

    if qid in aqwv5:
      q_aqwv5 = aqwv5[qid]
    else:
      q_aqwv5 = 0
    return_str += "%.4f\t" % q_aqwv5

    if qid in aqwv10:
      q_aqwv10 = aqwv10[qid]
    else:
      q_aqwv10 = 0
    return_str += "%.4f\t" % q_aqwv10

    if qid in aqwv20:
      q_aqwv20 = aqwv20[qid]
    else:
      q_aqwv20 = 0
    return_str += "%.4f\n" % q_aqwv20
  return return_str

file_names = ["ANALYSIS-SW-DT",
             "ANALYSIS-SW-QT",
             "DEV-SW-DT",
             "DEV-SW-QT",
             "ANALYSIS-TL-DT",
             "ANALYSIS-TL-QT",
             "DEV-TL-DT",
             "DEV-TL-QT"]

data_dirs = ["../data/sw/ANALYSIS-EN",
             "../data/sw/ANALYSIS-SW",
             "../data/sw/DEV-EN",
             "../data/sw/DEV-SW",
             "../data/tl/ANALYSIS-EN",
             "../data/tl/ANALYSIS-TL",
             "../data/tl/DEV-EN",
             "../data/tl/DEV-TL"]

for file_name, data_dir in zip(file_names, data_dirs):
  result = get_result(data_dir)
  f_out = open('%s.tsv' % file_name, 'w')
  f_out.write(result)
  f_out.close()
