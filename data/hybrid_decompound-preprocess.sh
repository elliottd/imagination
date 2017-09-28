#!/bin/sh

if [ "$#" -ne 4 ]; then
      echo "Illegal number of parameters"
      echo "./decompounder-preprocess.sh DATA_PATH LANGUAGE TOOLS_PATH MODEL_FILE_PATH"
      echo "./decompounder-preprocess.sh data/wmt_task/ de ../tools/ data/wmt_task1/model.model"
      exit 1
fi

IN=$1
LANG=$2
# We'll only decompound if the language argument ($2) is de
DECOMPOUND_LANG=de
TOOLS=$3
# We'll decompound German using a model trained on the Multi30K data
MODEL_FILE=${4}

# Normalise the punctuation
perl ${TOOLS}/normalize-punctuation.perl -l $LANG < $IN.${LANG} > $IN.${LANG}.norm

# Tokenise the data
perl ${TOOLS}/tokenizer.perl -l $LANG < $IN.${LANG}.norm > $IN.${LANG}.norm.tok

# Lowercase the data
perl ${TOOLS}/lowercase.perl < $IN.${LANG}.norm.tok > $IN.${LANG}.norm.tok.lower

# Decompound the data if the language is German
if [ ${LANG} = "de" ]; then
  python2 ${TOOLS}/hybrid_compound_splitter.py -merge-filler -no-truecase -q -model ${MODEL_FILE} < $IN.${LANG}.norm.tok.lower > $IN.${LANG}.norm.tok.lower.decomp
fi
