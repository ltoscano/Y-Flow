import io
if __name__ == '__main__':
    #original = '/data/corpora/clef/ClefFrench/CLEF/QUERIES1-200/original-queries/tqueries.f.final' 
    original = '/data/corpora/clef/ClefFrench/CLEF/QUERIES1-200/original-queries/tqueries.f.final' 
    out = 'fr.t'
    with io.open(original, encoding = "ISO-8859-1") as fin:
        with io.open(out, 'wt', encoding='utf-8') as fout:
            for line in fin:
                line = line.strip()
                if line[-1] == '(': ## start a query
                    q = []
                elif line[0] == ')': ## end a query
                    fout.write(' '.join(q))
                    fout.write('\n')
                else:
                    pass
                    q.append(line.strip())

