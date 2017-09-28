from collections import Counter
import argparse
import re
import json
import gzip
import codecs


def main(args):

    dicts = load_dictionary(args.dictionary)
    load_data(args.file, [dicts])

def fopen(filename, mode='r', encoding='utf-8'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode=mode, encoding=encoding)
    return codecs.open(filename, mode=mode, encoding=encoding)

def load_data(path, dicts=(), n_words=-1, get_factors=False, factors=1,
              encoding='utf-8'):

    assert type(dicts) == list, 'provide dictionaries as a list'
    data = []

    unk_counter = 0
    unkless_lines = 0

    with fopen(path, mode='r', encoding=encoding) as f:
        for s in f:
            s = s.strip().split()  # tokens of form 'factor1|factor2|...' (num_dicts) or 'word' (no num_dicts)

            local = 0
            if get_factors:
                s= [[dicts[i][f] if f in dicts[i] else 1 for (i, f) in
                      enumerate(w.split('|')[:factors])]
                     for w in s]
            else:
                s_ = []
                for w in s:
                    if w not in dicts[0]:
                        unk_counter += 1
                        local += 1

            if float(local) / len(s) > 0.1 :
                unkless_lines = 0

            if len(s) == 0:
                continue

            # replace out-of-vocabulary words with the unknown word symbol (1)
            if n_words > 0:
                s = [[f if f < n_words else 1 for f in w] for w in s]


            data.append(s)

    print("Loaded {} sentences from {}".format(len(data), path))
    print("Detected {} UNK tokens".format(unk_counter))
    print("{} sentences have zero UNK tokens".format(unkless_lines))

def load_dictionary(path, max_words=0):
    """
    loads json-formatted vocabularies from disk
    :param path:
    :param max_words:
    :return:
    """
    # assert max_words > 0, 'you probably want to set max_words'  # TODO remove

    dictionary = load_json(path)

    if max_words > 0:
        for word, idx in list(dictionary.items()):
            if idx >= max_words:
                del dictionary[word]

    return dictionary

def load_json(filename):
    """
    json loader to load Nematus vocabularies
    :param filename:
    :return:
    """
    with open(filename, mode='rb') as f:
        # return unicode_to_utf8(json.load(f))
        return json.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Count words.')
    parser.add_argument('file', type=str, help='text to be counted')
    parser.add_argument('dictionary', type=str)
    args = parser.parse_args()
    main(args)
