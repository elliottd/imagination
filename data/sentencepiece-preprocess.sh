#!/bin/bash

echo "DataPath L1 L2 MosesToolsPath SentencePiecePath TargetVocab SentencePieceData D1Size D2Size"
echo "ASSUMES SentencePieceData is ordered as L1: D1, D2 followed by L2: D1 D2 in the same file"
echo "E.g. ./preprocess.sh bpe_task1 en de ../tools/ ../tools/subword-nmt 40000 data/joint.in 29000 239881"

PATH=$1

# source language (example: fr)
S=$2
# target language (example: en)
T=$3

TOOLS_PATH=$4
MPATH=$5
NUMWORDS=$6
SP_DATA=$7
D1SIZE=$8
D2SIZE=$9

## Normalise the punctuation
echo "Normalising punctuation..."
for split in "train" "dev"; do
  /usr/bin/perl ${TOOLS_PATH}/normalize-punctuation.perl -l $S < ${PATH}${split}.${S} > ${PATH}${split}.${S}.norm
  /usr/bin/perl ${TOOLS_PATH}/normalize-punctuation.perl -l $T < ${PATH}${split}.${T} > ${PATH}${split}.${T}.norm
done

### tokenize
echo "Tokenization..."
for split in "train" "dev"; do
  /usr/bin/perl ${TOOLS_PATH}/tokenizer.perl -threads 5 -l $S < ${PATH}${split}.${S}.norm > ${PATH}${split}.${S}.norm.tok
  /usr/bin/perl ${TOOLS_PATH}/tokenizer.perl -threads 5 -l $T < ${PATH}${split}.${T}.norm > ${PATH}${split}.${T}.norm.tok
done
#
### Lowercase the data
echo "Lowercasing..."
for split in "train" "dev"; do
  /usr/bin/perl ${TOOLS_PATH}/lowercase.perl < ${PATH}${split}.${S}.norm.tok > ${PATH}${split}.${S}.norm.tok.lower
  /usr/bin/perl ${TOOLS_PATH}/lowercase.perl < ${PATH}${split}.${T}.norm.tok > ${PATH}${split}.${T}.norm.tok.lower
done
#
## learn the SP model on the joint training data:
echo "Learning SentencePiece model on the joint data..."
/usr/local/bin/spm_train --input ${SP_DATA} --model_prefix ${MPATH}/joint --vocab_size ${NUMWORDS} --model_type unigram

echo "Applying SentencePiece model to the joint data..."
/usr/local/bin/spm_encode --model ${MPATH}/joint.model < ${SP_DATA} > ${SP_DATA}.sp

# OH THIS IS SO UGLY
echo "Creating the first input file for L1"
START=1
END=${D1SIZE}
echo ${START}
echo ${END}
/bin/sed -n -e "${START},${END}p" ${SP_DATA}.sp > d1_train.en.norm.tok.lower.sp
START=$((END+1))
END=$((END+D2SIZE))
echo ${START}
echo ${END}
/bin/sed -n -e "${START},${END}p" ${SP_DATA}.sp > d2_train.en.norm.tok.lower.sp
START=$((END+1))
END=$((END+D1SIZE))
echo ${START}
echo ${END}
/bin/sed -n -e "${START},${END}p" ${SP_DATA}.sp > d1_train.de.norm.tok.lower.sp
START=$((END+1))
END=$((END+D2SIZE))
echo ${START}
echo ${END}
/bin/sed -n -e "${START},${END}p" ${SP_DATA}.sp > d2_train.de.norm.tok.lower.sp

## build dictionaries
/bin/cat d1_train.en.norm.tok.lower.sp d2_train.en.norm.tok.lower.sp > en.norm.tok.lower.sp 
/usr/bin/python2 build_dictionary.py en.norm.tok.lower.sp
/bin/cat d1_train.de.norm.tok.lower.sp d2_train.de.norm.tok.lower.sp > de.norm.tok.lower.sp 
/usr/bin/python2 build_dictionary.py de.norm.tok.lower.sp

## frequency statistics
/usr/bin/python3 word_frequencies.py en.norm.tok.lower.sp
/usr/bin/python3 word_frequencies.py de.norm.tok.lower.sp
