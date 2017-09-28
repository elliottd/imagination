#!/usr/bin/python

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import argparse
import numpy as np
import json
from collections import OrderedDict
from io import open
import cPickle as pkl

def create_dictionary(filename, eos_unk=True,
                      unk_symbol='<UNK>', eos_symbol='</s>'):
    """
    Create dictionary for filename
    :param filename:
    :param eos_unk:
    :param pad:
    :param unk_symbol:
    :param eos_symbol:
    :param pad_symbol:
    :return:
    """
    print('Processing {}'.format(filename))
    word_freqs = OrderedDict()
    if filename.endswith('pkl'):
        data = pkl.load(open(filename, 'rb'))
        sentences = [x[0] for x in data]
        for line in sentences:
            words_in = line.strip().split(' ')
            for w in words_in:
                if w not in word_freqs:
                    word_freqs[w] = 0
                word_freqs[w] += 1
    else:
        with open(filename, mode='r', encoding='utf-8') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1
    words = word_freqs.keys()
    freqs = word_freqs.values()

    sorted_idx = np.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    worddict = OrderedDict()

    if eos_unk:
        worddict[eos_symbol] = 0
        worddict[unk_symbol] = 1

    added_symbols_count = 2 if eos_unk else 0

    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii + added_symbols_count

    with open('{}.json'.format(filename), 'w', encoding='utf8') as f:
        data = json.dumps(worddict, ensure_ascii=False, indent=2)
        f.write(unicode(data))  # unicode(data) auto-decodes data to unicode if str (workaround for bug)

    print('Done')


def parse_args():
    """
    Arguments parser.
    :return: args object with parsed arguments
    """
    ap = argparse.ArgumentParser('Build a dictionary for a file.')
    ap.add_argument('paths', metavar='FILE', nargs='+', type=str,
                    default='filename', help='input file(s)')
    ap.add_argument('--no-eos-unk', dest='eos_unk', action='store_false',
                    help='do not include EOS/UNK items')
    ap.add_argument('--unk-symbol', type=str,
                    help='use this value as unknown word symbol',
                    default='<UNK>')
    ap.add_argument('--eos-symbol', type=str,
                    help='use this value as end of sentence symbol',
                    default='</s>')

    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    for path in args.paths:
        create_dictionary(path, args.eos_unk, args.unk_symbol, args.eos_symbol)
