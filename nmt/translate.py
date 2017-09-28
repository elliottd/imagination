from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import logging
import numpy as np
import os
import re
import time
from subprocess import Popen, PIPE
from builtins import bytes
from io import open

import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from nmt.sampler import generate_sample, build_sampler
from nmt.model import prepare_batch, init_params, init_theano_params, \
    load_params
from nmt.data_iterator_mem import TextIterator
from nmt.utils import idx_to_word, load_json, invert_dictionary, load_dictionary

logger = logging.getLogger(__name__)


class Translator:
    """
    Translates a data set given a model.
    """

    def __init__(self, fs_init=None, fs_next=None, config=None, src_dicts=None,
                 trg_idict=None,
                 n_words_src=-1, n_words_trg=-1, trng=None, **kwargs):

        self.trng = trng if trng is not None else RandomStreams(1234)
        self.fs_init = fs_init
        self.fs_next = fs_next
        self.config = config

        self.src_dicts = src_dicts
        self.trg_idict = trg_idict

        self.n_words_src = n_words_src
        self.n_words_trg = n_words_trg

    def load_from_disk(self, models=(), configs=(), src_dicts=(),
                       trg_dict=None):
        """
        Load models  and config from disk
        :param models:
        :param configs:
        :param src_dicts:
        :param trg_dict:
        :return:
        """
        configs = configs[0].split()
        models = models[0].split()
        self.fs_init, self.fs_next, self.config = self.build_samplers(models,
                                                                      configs,
                                                                      self.trng)
        self.src_dicts = [load_dictionary(d, self.n_words_src) for d in
                          src_dicts]
        trg_dict = load_dictionary(trg_dict, self.n_words_trg)
        self.trg_idict = invert_dictionary(trg_dict)
        self.n_words_src = self.config['n_words_src']
        self.n_words_trg = self.config['n_words_trg']

    def translate(self, input=None, maxlen=50, nbest=False, normalize=True,
                  k=12, suppress_unk=False, factors=1, **kwargs):
        """
        Translate sentences in input_path.

        :param input:
        :param maxlen:
        :param nbest:
        :param normalize:
        :param k:
        :param suppress_unk:
        :param kwargs:
        :return:
        """
        for translation in Translator.translate_all(
                input=input, trng=self.trng, fs_init=self.fs_init,
                fs_next=self.fs_next,
                src_dicts=self.src_dicts, trg_idict=self.trg_idict,
                n_words_src=self.n_words_src, n_words_trg=self.n_words_trg,
                maxlen=maxlen, nbest=nbest, normalize=normalize, k=k,
                factors=factors,
                suppress_unk=suppress_unk):
            yield translation

    def translate_and_save(self, output='translations.txt', **kwargs):
        """
        Translate and save to file.
        """
        logger.info('Saving translations to {}'.format(output))
        with open(output, mode='w', encoding='utf-8') as f:
            for translation in self.translate(**kwargs):
                f.write(translation)
                f.write('\n')
                f.flush()

    @staticmethod
    def build_samplers(models, configs, trng):
        """
        Builds multiple samplers for use in an ensemble.

        :param models: list of model paths
        :param configs: list of model config paths
        :param trng: random number generator
        :return:
        """

        logger.info('Building samplers')

        fs_init = []
        fs_next = []
        first_config = None

        for model_path, config_path in zip(models, configs):
            config = load_json(config_path)

            if first_config is None:
                first_config = config

            # allocate model parameters
            params = init_params(config)

            # load model parameters and set theano shared variables
            params = load_params(model_path, params[0])
            tparams = init_theano_params(params)

            # word index
            f_init, f_next = build_sampler(
                tparams, config, trng, return_alignment=True)

            fs_init.append(f_init)
            fs_next.append(f_next)

        logger.info('Done')

        return fs_init, fs_next, first_config

    @staticmethod
    def translate_sentence(x, trng, fs_init, fs_next, nbest=False, maxlen=100,
                           normalize=True, k=12, suppress_unk=False,
                           return_alignment=False):
        """
        Translate an input sequence (1 sentence).

        :param x:
        :param trng:
        :param fs_init:
        :param fs_next:
        :param nbest:
        :param maxlen:
        :param normalize:
        :param k:
        :param suppress_unk:
        :param return_alignment:
        :return:
        """
        assert x.shape[2] == 1, 'can only translate a single sentence'

        #print("Entering generate_sample")
        sample, score, word_probs, alignment = generate_sample(
            fs_init, fs_next, x, trng=trng, k=k, maxlen=maxlen,
            stochastic=False, argmax=False,
            return_alignment=return_alignment, suppress_unk=suppress_unk)
        #print("Exiting generate_sample")

        # normalize scores according to sequence lengths
        if normalize:
            lengths = np.array([len(s) for s in sample])
            score = score / lengths
        if nbest:
            return sample, score, word_probs, alignment
        else:
            sidx = np.argmin(score)
            return sample[sidx], score[sidx], word_probs[sidx], alignment[sidx]

    @staticmethod
    def translate_all(input=None, trng=None, fs_init=(), fs_next=(),
                      src_dicts=(), trg_idict=None,
                      n_words_src=-1, n_words_trg=-1, nbest=False,
                      normalize=True, k=15, factors=1,
                      suppress_unk=False, print_word_probabilities=False,
                      c=False, **kwargs):
        """
        Translate all sentences in input_path
        :param input_path:
        :param trng:
        :param fs_init:
        :param fs_next:
        :param src_dicts:
        :param trg_dict:
        :param n_words_src:
        :param n_words_trg:
        :param nbest:
        :param normalize:
        :param k:
        :return:
        """
        print(input)
        data_iterator = TextIterator(input, input, src_dicts, [src_dicts[0]],
                                     batch_size=1, maxlen=9999,
                                     n_words_source=n_words_src,
                                     n_words_target=n_words_trg,
                                     shuffle_each_epoch=False,
                                     sort_by_length=False, maxibatch_size=1,
                                     factors=factors)

        start_time = time.time()

        for i, (x, y) in enumerate(data_iterator):

            x, _, _, _ = prepare_batch(x, y)  # we only use the src side of data

            assert x.ndim == 3, 'x should be a 3d tensor'
            assert x.shape[2] == 1, 'expecting only one sentence'

            maxlen_sample = min(3 * x.shape[1], 100)

            # translate current sentence
            sample, score, word_probs, alignment = Translator.translate_sentence(
                x, trng, fs_init, fs_next, nbest=nbest, maxlen=maxlen_sample,
                normalize=normalize, k=k,
                suppress_unk=suppress_unk)

            #print("Obtained translation for %d" % i)
            trans_out = idx_to_word(sample, trg_idict, remove_eos_token=True)
            #print("Converted to words for %d" % i)

            yield trans_out

            if i != 0 and i % 100 == 0:
                logger.info('Translated {} lines...'.format(i))

        logger.info('Translation Took: {} minutes'.format(
            float(time.time() - start_time) / 60.))
