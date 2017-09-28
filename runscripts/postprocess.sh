#!/bin/sh

IN=$1
L=$2

# Normalise the punctuation
perl tools/normalize-punctuation.perl -l $2 < $IN > $IN.norm

# Lowercase the data
perl tools/lowercase.perl < $IN.norm > $IN.norm.lc

# Detokenise the data
perl tools/detokenizer.perl -l $2 < $IN.norm.lc > $IN.norm.lc.det
