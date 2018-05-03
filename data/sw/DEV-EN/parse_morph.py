import re, json
def parse_morph_file(filename):
    with open(filename) as fin:
        for line in fin:
            line.strip()
            line  = line[2:-3]
            dicts = [json.loads(x) for x in re.split(r'(\{.*?\})', line) if len(x)>1]
            for dictionary in dicts:
                print(dictionary)
if __name__ == '__main__':
    morph_file = '../topics/QUERY2/query_list_morph.txt'
    parse_morph_file(morph_file)
