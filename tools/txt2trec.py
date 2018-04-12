import os
import sys
import argparse

def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', default='', help='')
  parser.add_argument('--output_dir', default='', help='')
  args = parser.parse_args()

  input_dir = args.input_dir
  output_dir = args.output_dir
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  print "Input dir", input_dir
  print "Output dir", input_dir

  input_txt_files = os.listdir(input_dir)
  print "Number of input files", len(input_txt_files)
  for f in input_txt_files:
    print f
    input_f = open(os.path.join(input_dir,f))
    input_txt = input_f.read()
    input_f.close()

    if f.endswith(".txt"):
      docno = f[:-4]
    else:
      docno = f

    output_txt = "<DOC>\n" 
    output_txt += "<DOCNO> "
    output_txt += docno
    output_txt += " </DOCNO>\n"
    output_txt += "<TEXT>\n"
    output_txt += input_txt
    output_txt += "</TEXT>\n"
    output_txt += "</DOC>\n" 

    output_f = os.path.join(output_dir,docno+'.txt')
    print output_f
    output_f = open(output_f,'w')
    output_f.write(output_txt)
    output_f.close()

if __name__=='__main__':
  main(sys.argv)
