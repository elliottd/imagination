from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import sys
import os
import re
import json
import gzip
import logging
import numpy as np
import time
import theano
import theano.tensor as tensor
from contextlib import contextmanager

from io import open
from collections import OrderedDict
from six.moves import cPickle as pickle
from six import iteritems


logger = logging.getLogger(__name__)


def fopen(filename, mode='r', encoding='utf-8'):
    print(filename)
    if filename.endswith('.gz'):
        return gzip.open(filename, mode=mode, encoding=encoding)
    return open(filename, mode=mode, encoding=encoding)


def warning(*objs):
    """
    Prints warning text/object to stderr

    :param objs:
    :return:
    """
    print(*objs, file=sys.stderr)


def zipp(params, theano_params):
    """
    Push parameters to Theano shared variables

    :param params:
    :param theano_params:
    :return:
    """
    for kk, vv in params.items():
        theano_params[kk].set_value(vv)


def unzip(zipped):
    """
    Pull parameters from Theano shared variables

    :param zipped:
    :return:
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def pp(pre, name):
    """
    Make prefix-appended name

    :param pre:
    :param name:
    :return the string prefix_name:
    """
    return '{}_{}'.format(pre, name)


def ortho_weight(n_dim):
    """
    Get orthogonal Gaussian weight matrix for ndim by applying
    SVG on a random matrix W of size ndim x ndim,
    resulting in matrices u, s and v.
    This returns u.

    :param n_dim:
    :return u:
    """
    logger.warn('Orthogonal weight with {} dims'.format(n_dim))
    W = np.random.randn(n_dim, n_dim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    """
    Get a (orthogonal) Gaussian weight matrix W, scaled
    if desired (and not orthogonal).
    :param nin:
    :param nout:
    :param scale:
    :param ortho:
    :return W:
    """
    logger.warn('Norm weight with range: +/-{}'.format(scale))
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype(theano.config.floatX)


def uniform_weight(nin, nout, scale=0.01, **kwargs):
    """
    Get a (orthogonal) Gaussian weight matrix W, scaled
    if desired (and not orthogonal).
    :param nin:
    :param nout:
    :param scale:
    :return W:
    """
    logger.warn('Uniform weight with range: +/-{}'.format(scale))
    return np.random.uniform(
        low=-scale, high=scale, size=(nin, nout)).astype(theano.config.floatX)


def uniform_glorot(nin, nout, mean=0., gain=1., **kwargs):
    """
    Uniform Glorot Initialization
    :param nin:
    :param nout:
    :param mean:
    :param gain: float or 'relu'
    :return:
    """
    if gain == 'relu':
        gain = np.sqrt(2)
    logger.warn('Uniform Glorot weight with {} fan-in, {} fan-out, {} mean, {} gain'.format(nin, nout, mean, gain))

    std = gain * np.sqrt(2.0 / float((nin + nout)))
    a = mean - np.sqrt(3) * std
    b = mean + np.sqrt(3) * std
    W = np.random.uniform(low=a, high=b, size=(nin, nout))
    return W.astype(theano.config.floatX)


@contextmanager
def ignored(*exceptions):
    """
    Ignore exceptions when executing code
    :param exceptions:
    :return:
    """
    try:
        yield
    except exceptions:
        pass


def tanh(x):
    """
    Just returns theano.tensor.tanh(x)
    :param x:
    :return tanh(x):
    """
    return tensor.tanh(x)


def linear(x):
    """
    Returns x.

    :param x:
    :return x:
    """
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        x, y = theano.tensor.matrices('x', 'y')
        c = concatenate([x, y], axis=1)
    :param tensor_list: list of Theano tensor expressions that should be concatenated
    :param axis: join along this axis
    :return out: concatenated tensor expression
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


def load_pickle_dictionary(dictionary_path):
    """
    Load a dictionary and optionally also return the inverted dictionary

    :param dictionary_path:
    :param invert:
    :return dictionary:
    :return inverted_dictionary:
    """
    with open(dictionary_path, mode='rb') as f:
        dictionary = pickle.load(f)
        return dictionary


def load_json(filename):
    """
    json loader to load Nematus vocabularies
    :param filename:
    :return:
    """
    with open(filename, mode='rb') as f:
        # return unicode_to_utf8(json.load(f))
        return json.load(f)


def dump_json(object, filename):
    """
    Dump an object as JSON
    :param filename:
    :param object:
    :return:
    """
    with open(filename, 'w', encoding='utf8') as f:
        data = json.dumps(object, ensure_ascii=False, indent=2)
        f.write(unicode(data))  # unicode(data) auto-decodes data to unicode if str (workaround for bug)


def load_dictionary(path, max_words=0):
    """
    loads json-formatted vocabularies from disk
    :param path:
    :param max_words:
    :return:
    """
    # assert max_words > 0, 'you probably want to set max_words'  # TODO remove

    dictionary = load_json(path)

    if max_words > 0:
        for word, idx in list(dictionary.items()):
            if idx >= max_words:
                del dictionary[word]

    return dictionary


def invert_dictionary(dictionary):
    """
    Invert a dictionary

    :param dictionary:
    :return inverted_dictionary:
    """
    inverted_dictionary = dict()
    for k, v in dictionary.items():
        inverted_dictionary[v] = k

    return inverted_dictionary


# return name of word embedding for factor i
# special handling of factor 0 for backward compatibility
def embedding_name(i):
    if i == 0:
        return 'Wemb'
    else:
        return 'Wemb' + str(i)


def generate_bleu_path(directory, model_name, bleu_score):
    return os.path.join(directory, '{}_best_bleu_model_{}_BLEU{}.npz'.format(
        model_name, int(time.time()), bleu_score))


def clip_grad_norm(grads, clip_c):
    """
    Clip gradient
    :param grads:
    :param clip_c:
    :return grads:
    """
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g ** 2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c ** 2), g / tensor.sqrt(g2) * clip_c, g))
        grads = new_grads

    return grads


def idx_to_word(seq, ivocab, remove_eos_token=True):
    """
    Get the words for a sequence of word IDs
    :param seq:
    :param ivocab:
    :param unk_symbol:
    :param remove_eos_token:
    :return:
    """

    # remove EOS token
    if seq[-1] == 0 and remove_eos_token:
        seq = seq[:-1]

    unk_symbol = ivocab[1]
    translation = ' '.join([ivocab.get(idx, unk_symbol) for idx in seq])
    return translation


if __name__ == '__main__':
    pass
