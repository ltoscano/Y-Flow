import argparse
import os
import io
import re

def output_doc(text, filename):
    with open(filename, 'wt') as fout:
        for line in text:
            fout.write(line)

def trecs2trec(filename, output_dir, lowercase=False):
    with open(filename) as fin:
        print(filename)
        for line in fin:
            line.strip()
            if '<DOC>' in line: ## begin a document
                text = [line]
            elif '</DOC>' in line:
                text.append(line)
                output_doc(text, os.path.join(output_dir, doc_id))
            elif '<DOCID>' in line:
                #doc_id = line[7:-8]
                text.append(line)
                start_idx = re.search('<DOCID>', line).end()
                end_idx = re.search('</DOCID>', line).start()
                doc_id = line[start_idx:end_idx]
            else:
                text.append(line)
  
def main(base_dir, output_dir, lower):
    files = os.listdir(base_dir)
    #files = [x for x in files if x != 'sda.dtd' and 'xml' not in x]
    files = [x for x in files if x != 'sda.dtd']
    for filename in files:
        input_file = os.path.join(base_dir, filename)
        trecs2trec(input_file, output_dir, lower)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help = 'base directory for data')
    parser.add_argument('output_dir', help = 'output directory')
    parser.add_argument('--lower', action='store_true')
    args = parser.parse_args()
    main(args.base_dir, args.output_dir, args.lower)
