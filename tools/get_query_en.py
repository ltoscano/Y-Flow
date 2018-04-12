import io
if __name__ == '__main__':
    #original = '/data/corpora/clef/ClefFrench/CLEF/QUERIES1-200/original-queries/tqueries.f.final' 
    original = '/data/corpora/clef/ClefFrench/CLEF/QUERIES1-200/original-queries/tqueries.e.final' 
    out = 'en.t'
    with io.open(original, encoding = "ISO-8859-1") as fin:
        with io.open(out, 'wt', encoding='utf-8') as fout:
            for line in fin:
                line = line.strip()
                q = line[line.find('(')+1:line.find(')')]
                q = q.strip()
                fout.write(q)
                fout.write('\n')

