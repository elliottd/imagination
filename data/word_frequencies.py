from collections import Counter
import argparse
import re
import pprint


def main(args):

    words = re.findall('[^\s]+', open(args.file, encoding='utf-8').read())
    c = Counter(words)

    n_tokens = sum(c.values())
    n_types = len(list(c))

    print('Found %d word tokens' % n_tokens)
    print('Found %d word types' % n_types)

    count_low_freq_words = 0

    count_ones = 0

    for word in c:
        if c[word] == 1:
            count_ones += 1
        if c[word] < 5:
            count_low_freq_words += 1

    print('There were %d words with count < 5.' % count_low_freq_words)
    print('There were %d words with count = 1.'  % count_ones)

    print('Types - low freq words = %d' % (n_types - count_low_freq_words))

    c = list(c.items())
    c = sorted(c, key=lambda x: int(x[1]))

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(c)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Count words.')
    parser.add_argument('file', type=str, help='text to be counted')
    args = parser.parse_args()
    main(args)
