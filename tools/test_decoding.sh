#!/bin/bash
export PYTHONPATH=.:/home/delliott/src/imaginet:/home/delliott/src/nmt

function join_by { 
  local IFS="$1";
  shift;
  #echo "$*";
}

IFS=","
INPUT=$1
MODELS=$2
CONFIGS=$3
SRCDICT=$4
TGTDICT=$5
OUTPUT=$6
BEAM=$7
REF=$8

declare globalvar arrMODELS=(${MODELS//,/ })
join_by ' ' "${arrMODELS[@]}"
#echo $arrMODELS
declare globalvar arrCONFIGS=(${CONFIGS//,/ })
join_by ' ' "${arrCONFIGS[@]}"
#echo $arrCONFIGS

THEANO_FLAGS=device=gpu1,floatX=float32 python2 -m nmt test --input $1 \
  --models $arrMODELS \
  --configs $arrCONFIGS \
  --src-dicts $SRCDICT \
  --trg-dict $TGTDICT \
  --output $OUTPUT \
  -k $BEAM \
  --suppress-unk \
