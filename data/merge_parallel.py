#!/usr/bin/python3

import argparse
import logging
import os
import subprocess
import uuid

parser = argparse.ArgumentParser(
    description="""

Merge parallel data to use it with e.g. fast_align

Input files: source sentence file, target sentence file
Output: source sentence 1 ||| target sentence 1, etc.

""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("source", type=str, help="Path to (tokenized) source data")
parser.add_argument("target", type=str, help="Path to (tokenized) target data")
parser.add_argument("output_file", type=str, help="Path to output file")


def merge_parallel(src_filename, trg_filename, merged_filename):
    with open(src_filename, mode='r', encoding='utf-8') as left:
        with open(trg_filename, mode='r', encoding='utf-8') as right:
            with open(merged_filename, mode='w', encoding='utf-8') as final:
                while True:
                    lline = left.readline()
                    rline = right.readline()
                    if (lline == '') or (rline == ''):
                        break
                    if (lline != '\n') and (rline != '\n'):
                        final.write(lline[:-1] + ' ||| ' + rline)


def main():

    # check if source and target files exist
    for path in [args.source, args.target]:
        if not os.path.exists(path):
            raise Exception("File does not exist: {}".format(path))

    # shuffle data sets n times
    merge_parallel(args.source, args.target, args.output_file)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('merger')

    args = parser.parse_args()
    main()
