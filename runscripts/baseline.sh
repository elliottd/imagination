#!/bin/bash

# Multi30K Task 1
# Baseline NMT model
# Example usage: ./baseline.sh 5
# This will run five random seeds of the model

# Tell Python where the NMT and the Imaginet code can be found
export PYTHONPATH=.:/home/delliott/src/nmt:/home/delliott/src/nmt/imaginet
THEANO_FLAGS="device=gpu0,floatX=float32"
MODEL_DIR=${HOME}/src/nmt/models/ # This is where we'll save the NMT model parameters.

COUNTER="$1" # the first argument to the bash script.
YAML="$2" # path to the YAML configuration file

until [ $COUNTER = 0 ]; do

  OUTPUT_DIR=${MODEL_DIR}/multi30k_baseline_${COUNTER}/
  mkdir -p ${OUTPUT_DIR}

  cd $HOME/src
  THEANO_FLAGS=${THEANO_FLAGS} python2 -m nmt train ${YAML} --exp_id ${COUNTER} &> ${OUTPUT_DIR}/training.log
  let COUNTER-=1
done

