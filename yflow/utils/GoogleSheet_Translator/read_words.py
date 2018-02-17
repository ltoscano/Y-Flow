from __future__ import print_function


def read_words(f_name):
    data = []
    f = open(f_name, 'r')
    for line in f:
        data.append(line.rstrip())
    return data
