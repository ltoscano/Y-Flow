import sys
import argparse

def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--predict', default='', help='')
  parser.add_argument('--judge', default='', help='')
  parser.add_argument('--total', type=int, help='')
  parser.add_argument('--beta', type=float, default=20.0, help='')

  args = parser.parse_args()
  N_total = args.total

  judge_file = open(args.judge)
  judge = {}
  for line in judge_file.readlines():
    line_split = line.split()
    query = line_split[0]
    doc = line_split[2]
    if query not in judge:
      judge[query] = []
    judge[query].append(doc)

  predict_file = open(args.predict)
  predict = {}
  for line in predict_file.readlines():
    line_split = line.split()
    query = line_split[0]
    doc = line_split[2]
    if query not in predict:
      predict[query] = []
    predict[query].append(doc)

  
  #assert set(predict.keys()) == set(judge.keys())
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

      aqwv = 1 - (N_miss / float(N_rel) + args.beta*(N_FA / float(N_nonrel)))
      if optimal_aqwv is None:
        optimal_aqwv = aqwv
        optimal_cutoff = cutoff
      elif optimal_aqwv < aqwv:
        optimal_aqwv = aqwv
        optimal_cutoff = cutoff
    return optimal_aqwv, optimal_cutoff

  optimal = {}
  for query in predict.keys():
    optimal_aqwv,optimal_cutoff = get_optimal_aqwv(query,predict[query],judge[query])
    optimal[query] = (optimal_aqwv,optimal_cutoff)
    print query, "%.4f" % optimal_aqwv, optimal_cutoff

  avg_optimal = sum(optimal[q][0] for q in optimal) / len(judge)
  print "Average Optimal: %.4f" % avg_optimal

if __name__=='__main__':
  main(sys.argv)
