"""
Calculates perplexity of a model on a dev/test set.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import argparse
import numpy
from six.moves import cPickle as pkl
from six import iteritems
import theano
import json
import logging
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from nmt.model import build_model, load_params, init_params, pred_probs, \
    prepare_batch
from nmt.train import init_theano_params
from nmt.data_iterator_mem import TextIterator
from nmt.utils import load_dictionary, invert_dictionary, load_json

logging.basicConfig()
logger = logging.getLogger(__name__)


def main(model, test_src, test_trg, dictionary_src, dictionary_trg):
    # load model model_options
    config = load_json('{}.json'.format(model))

    # load source dictionary and invert
    word_dict = load_dictionary(dictionary_src)
    word_idict = invert_dictionary(word_dict)
    word_idict[0] = config['eos_symbol']
    word_idict[1] = config['unk_symbol']

    # load target dictionary and invert
    word_dict_trg = load_dictionary(dictionary_trg)
    word_idict_trg = invert_dictionary(word_dict_trg)
    word_idict_trg[0] = config['eos_symbol']
    word_idict_trg[1] = config['unk_symbol']

    # load data
    data_iter = TextIterator(test_src, test_trg, [dictionary_src],
                             dictionary_trg,
                             n_words_source=config['n_words_src'],
                             n_words_target=config['n_words_trg'],
                             batch_size=config['valid_batch_size'],
                             maxlen=100000, shuffle_each_epoch=False)

    print('Loading model')
    params = init_params(config)
    params = load_params(model + '.npz', params)
    tparams = init_theano_params(params)

    # random generator and global dropout/noise switch for this model
    trng = RandomStreams(1234)

    x, x_mask, y, y_mask, opt_ret, cost = build_model(
        tparams, trng, config, use_mask=True, use_noise=False)
    inps = [x, x_mask, y, y_mask]

    print('Building f_log_probs...', end="")
    f_log_probs = theano.function(inps, cost, profile=False)
    print('Done')

    # calculate the probabilities
    loss, perplexity = pred_probs(f_log_probs, prepare_batch, data_iter)
    mean_loss = loss.mean()

    print('Loss: %f' % mean_loss)
    print('PPX: %f' % perplexity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary_src', type=str)
    parser.add_argument('dictionary_trg', type=str)
    parser.add_argument('test_src', type=str)
    parser.add_argument('test_trg', type=str)

    args = parser.parse_args()

    main(args.model, args.test_src, args.test_trg, args.dictionary_src,
         args.dictionary_trg)
