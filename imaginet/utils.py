from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from builtins import range
from collections import Counter, OrderedDict, deque
from theano import tensor
import theano
import numpy as np
import yaml
from io import open
import logging
import os
import json
import cPickle as pkl
import tables
import math


logger = logging.getLogger(__name__)


def dump_params(output_dir, filename, list_of_params_to_dump):
    """
    Dumps parameters to output_dir/filename
    :param output_dir:
    :param filename:
    :param list_of_params_to_dump:
    :return:
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path = os.path.join(output_dir, filename)
    logger.info('Dumping yaml to {}'.format(path))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(path, mode='wb') as f:
        yaml.dump(list_of_params_to_dump, f, encoding='utf-8')

    logger.info('Finished yaml dump')


def load_params(output_dir, filename):
    """
    Loads parameters from output_dir/filename
    :param output_dir:
    :param filename:
    :return params:
    """
    path = os.path.join(output_dir, filename)
    logger.info('Loading parameters from {}'.format(path))
    params = yaml.load(open(path, mode='rb'))
    logger.info('Finished loading parameters')
    return params


def get_vocabulary(conll_path, max_words=None, normalize_words=False,
                   lowercase_words=False, unk='<UNK>', eos='</s>'):
    """
    Reads in a CoNLL file and returns word count, a word-to-index mapping, POS count, and relation count
    :param conll_path:
    :param max_words: cut-off to this number of words, keeping highest frequency ones
    :param normalize_words:
    :param lowercase_words:
    :param unk: unk symbol
    :param eos: eos symbol
    :param pad: pad symbol
    :return word count, word-to-index, POS count, relation count:
    """
    logger.info('Creating vocabulary (lc={}, normalize={}'.format(lowercase_words, normalize_words))

    word_counter = Counter()
    pos_counter = Counter()
    deprel_counter = Counter()

    # open a CoNLL file and count all words, POS-tags, and relations
    with open(conll_path, mode='r', encoding='utf-8') as f:
        for sentence in read_conll(f, lowercase=lowercase_words):
            word_counter.update([w.norm if normalize_words else w.form for w
                                 in sentence[1:]])
            pos_counter.update([w.pos for w in sentence])
            deprel_counter.update([w.deprel for w in sentence])

    # mappings, sorted by frequency: {word, pos, deprel} --> index
    most_common = word_counter.most_common(max_words)

    # min_count = 2
    # print('filtering out words < {}'.format(min_count))
    # most_common = [[word, count] for [word, count] in most_common if count >=
    #                min_count]
    # print(most_common[:10])

    w2i = OrderedDict([[w, i+2] for i, (w, _) in enumerate(most_common)])
    w2i[eos] = 0
    w2i[unk] = 1
    # w2i[pad] = 2

    # print('words in voc after filtering: {}'.format(len(w2i)))

    p2i = OrderedDict([[p, i+2] for i, (p, _) in enumerate(
        pos_counter.most_common())])
    p2i[eos] = 0
    p2i[unk] = 1
    # p2i[pad] = 2

    r2i = {r: i for i, (r, _) in enumerate(deprel_counter.most_common())}

    logger.info('Finished creating vocabulary')

    return w2i, p2i, r2i


def _p(pre, name):
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
    logger.warn('Orthogonal weight with {} dimensions'.format(n_dim))
    W = np.random.randn(n_dim, n_dim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')


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
    return W.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True, **kwargs):
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


def load_json(filename):
    """
    json loader to load Nematus vocabularies
    Note that we use unicode representations (not str)
    :param filename:
    :return:
    """
    with open(filename, mode='rb') as f:
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
        # f.write(unicode(data))  # unicode(data) auto-decodes data to unicode if str (workaround for bug)
        f.write(data)


def load_vocabularies(config):
    """
    loads json-formatted vocabularies from disk
    :param config:
    :return:
    """
    w2i = load_json(config['word_vocabulary'])
    max_words = config['max_words']

    if max_words > 0:
        for word, idx in list(w2i.items()):
            if idx >= max_words:
                # print('deleting from dict: word={} idx={}'.format(word, idx))
                del w2i[word]

    w2i['</s>'] = 0
    w2i['UNK'] = 1

    i2w = {v:k for k,v in w2i.iteritems()}

    return w2i, i2w


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

# return name of word embedding for factor i
# special handling of factor 0 for backward compatibility
def embedding_name(i):
    if i == 0:
        return 'Wemb'
    else:
        return 'Wemb' + str(i)
