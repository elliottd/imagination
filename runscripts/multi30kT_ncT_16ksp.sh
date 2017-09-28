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
YAML=${HOME}/src/nmt/configs/multi30kT_ncT_sp16k.yaml

COUNTER="$1" # the first argument to the bash script.

until [ $COUNTER = 0 ]; do

  OUTPUT_DIR=${MODEL_DIR}/multi30kT_ncT_sp16k_${COUNTER}/
  mkdir -p ${OUTPUT_DIR}

  cd $HOME/src
  THEANO_FLAGS=${THEANO_FLAGS} python2 -m nmt train ${YAML} --exp_id ${COUNTER} &> ${OUTPUT_DIR}/training.log
  let COUNTER-=1
done
