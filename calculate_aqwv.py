import sys
import argparse

def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--predict', default='', help='')
  parser.add_argument('--judge', default='', help='')
  parser.add_argument('--total', type=int, help='')
  parser.add_argument('--language', default='sw', help='')
  parser.add_argument('--beta', type=float, default=20.0, help='')

  args = parser.parse_args()
  N_total = args.total

  query_type = {}
  query_topics = {}
  if args.language == 'sw':
    topic_file = open('data/sw/topics/en.t')
  else:
    topic_file = open('data/tl/topics/en.t')
  for line in topic_file.readlines():
    split_line = line.strip().split()
    query = split_line[0]
    if ',' in line:
      query_type[query] = 'conjunction'
    elif '"' in line:
      query_type[query] = 'phrase'
    else:
      query_type[query] = 'single'

  judge_file = open(args.judge)
  judge = {}
  for line in judge_file.readlines():
    line_split = line.split()
    query = line_split[0]
    doc = line_split[2]
    if query not in judge:
      judge[query] = []
    judge[query].append(doc)

  print >> sys.stderr, len(judge)

  predict_file = open(args.predict)
  predict = {}
  for line in predict_file.readlines():
    line_split = line.split()
    query = line_split[0]
    doc = line_split[2]
    if query not in predict:
      predict[query] = []
    predict[query].append(doc)

  print >> sys.stderr, len(predict)

  cutoffs = [1,3,5,10,20,-1]
  def get_optimal_aqwv(query,q_pred,q_judge):
    N_rel = len(q_judge)
    N_nonrel = N_total - N_rel
    optimal_aqwv = None
    optimal_cutoff = None
    for cutoff in cutoffs:
      if cutoff == -1:
        cutoff = len(q_pred)
      pred_cutoff = q_pred[:cutoff]
      N_correct = len(set(q_judge) & set(pred_cutoff))
      N_FA = len(pred_cutoff) - N_correct
      N_miss = N_rel - N_correct

      #if query == 'query1721':
      #  print cutoff
      #  print N_total,N_rel,N_nonrel
      #  print N_correct,N_FA,N_miss
      #  print

      if N_rel > 0:
        aqwv = 1 - (N_miss / float(N_rel) + args.beta*(N_FA / float(N_nonrel)))
      else:
        aqwv = 1 - args.beta*(N_FA / float(N_nonrel))
      if optimal_aqwv is None:
        optimal_aqwv = aqwv
        optimal_cutoff = cutoff
      elif optimal_aqwv < aqwv:
        optimal_aqwv = aqwv
        optimal_cutoff = cutoff
    return optimal_aqwv, optimal_cutoff

  optimal = {}
  for query in query_type.keys():
    if query not in predict:
      q_pred = []
    else:
      q_pred = predict[query]

    if query not in judge:
      q_judge = []
    else:
      q_judge = judge[query]

    optimal_aqwv,optimal_cutoff = get_optimal_aqwv(query,q_pred,q_judge)
    optimal[query] = (optimal_aqwv,optimal_cutoff)
    print query, "%.4f" % optimal_aqwv, optimal_cutoff

  avg_optimal = sum(optimal[q][0] for q in optimal) / len(optimal)
  avg_cutoff = sum(optimal[q][1] for q in optimal) / len(optimal)
  print >> sys.stderr, len(optimal)
  print >> sys.stderr, "Average Optimal: %.4f, %.2f" % (avg_optimal,avg_cutoff)

  optimal_single = [optimal[q][0] for q in optimal if query_type[q] is 'single']
  cutoff_single = [optimal[q][1] for q in optimal if query_type[q] is 'single']
  avg_optimal_single = sum(optimal_single) / len(optimal_single)
  avg_cutoff_single = sum(cutoff_single) / len(cutoff_single)
  print >> sys.stderr, len(optimal_single)
  print >> sys.stderr, "Average Optimal Single Word: %.4f, %.2f" % (avg_optimal_single,avg_cutoff_single)

  optimal_phrase = [optimal[q][0] for q in optimal if query_type[q] is 'phrase']
  cutoff_phrase = [optimal[q][1] for q in optimal if query_type[q] is 'phrase']
  avg_optimal_phrase = sum(optimal_phrase) / len(optimal_phrase)
  avg_cutoff_phrase = sum(cutoff_phrase) / len(cutoff_phrase)
  print >> sys.stderr, len(optimal_phrase)
  print >> sys.stderr, "Average Optimal Phrase: %.4f, %.2f" % (avg_optimal_phrase,avg_cutoff_phrase)

  optimal_conjunction = [optimal[q][0] for q in optimal if query_type[q] is 'conjunction']
  cutoff_conjunction = [optimal[q][1] for q in optimal if query_type[q] is 'conjunction']
  avg_optimal_conjunction = sum(optimal_conjunction) / len(optimal_conjunction)
  avg_cutoff_conjunction = sum(cutoff_conjunction) / len(cutoff_conjunction)
  print >> sys.stderr, len(optimal_conjunction)
  print >> sys.stderr, "Average Optimal Conjunction: %.4f, %.2f" % (avg_optimal_conjunction, avg_cutoff_conjunction)

if __name__=='__main__':
  main(sys.argv)
