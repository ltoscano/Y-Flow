import subprocess

for i in range(2, 2000):
    parse_command = 'python src/phrase.py --phrase True --src_lang en --tgt_lang en --query ../topics/query_list.tsv --tquery qmodel/mono --model mono --n_iters {}'.format(i)
    subprocess.check_call(parse_command, shell=True)
    indri_command = 'IndriRunQuery qmodel/mono -index=index/ -count=5 -trecFormat=true >| result/result.first'
    subprocess.check_call(indri_command, shell=True)
    print(i)
