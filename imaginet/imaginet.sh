#!/bin/bash

# Imaginet model
# Example usage: ./imaginet.sh 5
# This will run five random seeds of the model

# Tell Python where the NMT and the Imaginet code can be found
CODE_HOME=/home/delliott/
export PYTHONPATH=.:${CODE_HOME}/src/imaginet
MODEL_DIR=${CODE_HOME}/src/imaginet/models/ # This is where we'll save the NMT model parameters.

if [[ $# -eq 0 ]] ; then
  echo 'You need to provide an integer that tells the script how many times to execute'
  exit 0
fi

if [[ $# -eq 1 ]] ; then
  echo 'You also need to tell the script which YAML file it should use'
  exit 0
fi

if [[ $# -eq 2 ]]; then
  echo 'You need to pass an argument that defines the cuda device'
  exit 0
fi

THEANO_FLAGS="device=$3,dnn.enabled=False,floatX=float32"
COUNTER="$1" # the first argument to the bash script.
YAML=$2 # the second argument is the yaml config file.

until [ $COUNTER = 0 ]; do

  OUTPUT_DIR=${MODEL_DIR}/imaginet_${COUNTER}/
  mkdir -p ${OUTPUT_DIR}

  cd $EUROPEANA_HOME/src
  THEANO_FLAGS=${THEANO_FLAGS} python -m imaginet train \
    $YAML #\
    #&> ${OUTPUT_DIR}/training.log

    let COUNTER-=1
done
