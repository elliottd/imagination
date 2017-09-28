# Imagination
Attention-based neural machine translation for Multimodal Translation that learns to translate and to predict the visual representation of an image from the encoder.

### Dependencies

* Python 2.7
* Theano
* numpy
* lxml
* pexpect

## Usage

See an example runscript in `runscripts/baseline.sh` for how to train a text-only baseline.

See an example in `runscripts/multi30kT_multi30kI-InceptionV3.sh` for how to train a model with image prediction.

You need to replace the hard-coded PYTHONPATH variables to the absolute directory on your machine.
