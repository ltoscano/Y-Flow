#!/usr/bin/env python3
import unicodedata
import os, io
#s = 'Découvrez tous les logiciels à télécharger'
#s2 = unicodedata.normalize('NFD', s).encode('ascii', 'ignore')
#print(s2.decode("utf-8"))
if __name__ == '__main__':
    in_file = 'en-fr.txt'
    out_file = 'en-fr-normalized.txt'
    with open(in_file, 'rt') as fin:
        with open(out_file, 'wt') as fout:
            for line in fin:
                line = unicodedata.normalize('NFD', line).encode('ascii', 'ignore').decode("utf-8")
                fout.write(line)
                    
