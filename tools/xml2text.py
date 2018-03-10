import argparse
import os
from lxml import etree
import io

def output_doc(text, filename):
    with open(filename, 'wt') as fout:
        for line in text:
            fout.write(line)

def xml2text(filename, output_dir):
    #parser = etree.XMLParser(encoding='ISO-8859-1')
    parser = etree.XMLParser(encoding='utf-8')
    #with open(filename) as fin:
    xmltree = etree.parse(filename, parser)
    root = xmltree.getroot()
    for child in root:
        text = []
        for element in child.iter():
            tag = element.tag
            if tag == 'DOCID':
                doc_id = element.text
            elif tag == 'TEXT':
                if element.text is not None:
                    text.append(element.text)
        output_doc(text, os.path.join(output_dir, doc_id))
        
def add_root_clause(input_file, output_file):
## I really didn't want to do this, but since each file in our French collections is a naive concatenation of multiple xml files, I needed to hallucinate a new root node encapsulating all to get the xml parser to work.
    with open(output_file, 'wt') as fout:
        fout.write('<ALL>\n')
        with io.open(input_file, encoding='ISO-8859-1') as fin:
            for line in fin:
                fout.write(line)
        fout.write('</ALL>')
  
def main(base_dir, output_dir):
    files = os.listdir(base_dir)
    #files = files[:1]
    for filename in files:
        input_file = os.path.join(base_dir, filename)
        add_root_clause(input_file, input_file+'.xml')
        xml2text(input_file+'.xml', output_dir)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help = 'base directory for data')
    parser.add_argument('output_dir', help = 'output directory')
    args = parser.parse_args()
    main(args.base_dir, args.output_dir)
