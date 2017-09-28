#!/bin/bash

# Multi30K Task 1
# Baseline NMT model
# Example usage: ./baseline.sh 5
# This will run five random seeds of the model

# Tell Python where the NMT and the Imaginet code can be found
export PYTHONPATH=.:/home/delliott/src/nmt:/home/delliott/src/imaginet
THEANO_FLAGS="device=gpu1,floatX=float32"
MODEL_DIR=${HOME}/src/nmt/models/ # This is where we'll save the NMT model parameters.
YAML=${HOME}/src/nmt/configs/baseline.yaml

COUNTER="$1" # the first argument to the bash script.

until [ $COUNTER = 0 ]; do

  OUTPUT_DIR=${MODEL_DIR}/multi-nmt_${COUNTER}
  mkdir -p ${OUTPUT_DIR}

  cd $HOME/src/
  THEANO_FLAGS=${THEANO_FLAGS} python2 -m nmt train ${YAML}
  let COUNTER-=1
done

