#!/bin/bash
# Multi30K Task 1
# Multitask NMT + Imaginet InceptionV3 model
# Example usage: ./multitask.sh 5
# This will run five random seeds of the model
# --mtl --mtl-config --mtl-ratios control the multasking model
# The MTL parameters are saved as defined in --mtl-config

# Tell Python where the NMT and the Imaginet code can be found
export PYTHONPATH=.:/afs/inf.ed.ac.uk/group/project/europeana/src/imagination
THEANO_FLAGS="device=gpu1,floatX=float32"
MODEL_DIR=/home/delliot/europeana/src/imagination/models/ # This is where we'll save the NMT model parameters.
YAML=/home/delliot/europeana/src/imagination/configs/multi30kT_multi30kI_InceptionV3.yaml
COUNTER="$1" # the first argument to the bash script.

until [[ $COUNTER = 0 ]]; do

  OUTPUT_DIR=${MODEL_DIR}/europeana-multi30kT_multi30KI_InceptionV3_${COUNTER}/
  mkdir -p ${OUTPUT_DIR}

  cd $HOME/europeana/src
  THEANO_FLAGS=${THEANO_FLAGS} python2 -m nmt train ${YAML} --exp_id ${COUNTER}
# &> ${OUTPUT_DIR}/training.log
  let COUNTER-=1
done
