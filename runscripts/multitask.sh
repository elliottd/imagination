#!/bin/bash

# Multi30K Task 1
# Multitask NMT + Imaginet model
# Example usage: ./multitask.sh 5
# This will run five random seeds of the model
# --mtl --mtl-config --mtl-ratios control the multasking model
# The MTL parameters are saved as defined in --mtl-config

# Tell Python where the NMT and the Imaginet code can be found
export PYTHONPATH=.:/home/delliott/src/imaginet:/home/delliott/src/nmt
THEANO_FLAGS="device=gpu1,dnn.enabled=False,floatX=float32"
MODEL_DIR=${HOME}/src/nmt/models/ # This is where we'll save the NMT model parameters.

COUNTER="$1" # the first argument to the bash script.

until [ $COUNTER = 0 ]; do

  OUTPUT_DIR=${MODEL_DIR}/multitask_${COUNTER}/
  mkdir -p ${OUTPUT_DIR}

  cd $HOME/src
  THEANO_FLAGS=${THEANO_FLAGS} python -m nmt train \
    model \
    ${OUTPUT_DIR} \
    nmt/data/wmt_task1/train.en \
    nmt/data/wmt_task1/train.de \
    nmt/data/wmt_task1/dev.en \
    nmt/data/wmt_task1/dev.de \
    --src-dicts \
    nmt/data/wmt_task1/en_dict.json \
    --trg-dicts \
    nmt/data/wmt_task1/de_dict.json \
    --n-words-src 10214 \
    --n-words-trg 16022 \
    --dim-emb 620 \
    --dim-per-factor 620 \
    --factors 1 \
    --dim 1000 \
    --dim-att 2000 \
    --batch-size 80 \
    --sort-k-batches 3 \
    --maxlen 50 \
    --validation-frequency -1 \
    --exp_id run${COUNTER} \
    --bleu-val-burnin 7500 \
    --bleu-val-ref nmt/data/wmt_task1/dev.de.gold \
    --bleu-script nmt/nmt/multi-bleu.perl \
    --display-frequency 100 \
    --optimizer adam \
    --learning-rate 1e-4 \
    --decay-c 0. \
    --beam-size 12 \
    --max-epochs 100 \
    --dropout --dropout-src 0.0 --dropout-emb 0.2 --dropout-rec 0.2 --dropout-hid 0.0 \
    --at_replace \
    --early_stopping bleu \
    --patience 5 \
    --mtl --mtl-configs imaginet/config.yaml \
    --mtl-ratio 0.5 0.5  \
    &> ${OUTPUT_DIR}/training.log

    let COUNTER-=1
done
