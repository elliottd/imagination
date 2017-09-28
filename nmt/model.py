from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import warnings
import logging
import theano
import theano.tensor as tensor
import numpy as np

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# from theano.tensor.shared_randomstreams import RandomStreams

from collections import OrderedDict
from six import iteritems

from nmt.utils import norm_weight, concatenate, embedding_name, tanh, linear
from nmt.layers import inv_dropout_mask, init_ff, ff, init_gru, gru, \
    init_lstm, lstm, init_gru_cond, gru_cond, init_lstm_cond, lstm_cond, \
    init_gru_cond_simple, gru_cond_simple, \
    init_lstm_cond_simple, lstm_cond_simple

logger = logging.getLogger(__name__)

profile = False


def init_theano_params(params):
    """
    Initialize Theano shared variables according to the initial parameters

    :param params:
    :return:
    """
    theano_params = OrderedDict()
    for kk, pp in iteritems(params):
        logger.info(
            'Parameter: {} Shape: {}'.format(kk, params[kk].shape))
        theano_params[kk] = theano.shared(params[kk], name=kk)

    return theano_params


def load_params(path, params):
    """
    Loads parameters from disk
    :param path:
    :param params:
    :return params:
    """
    pp = np.load(path)
    for kk, vv in iteritems(params):
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params


def init_encoder_params(params, factors=0, encoder='gru', n_words_src=0,
                        dim_per_factor=0, dim=0, dim_emb=0, encoder_layers=0,
                        **kwargs):
    """
    Initialize encoder parameters.

    :param params:
    :param factors:
    :param encoder:
    :param n_words_src:
    :param dim_per_factor:
    :param dim:
    :param dim_emb:
    :param encoder_layers:
    :return:
    """

    # word embeddings, for each factor (word, POS-tag, etc.)
    for factor in range(factors):
        emb_init = norm_weight
        logger.warning('Word embeddings init: {}'.format(emb_init))
        params[embedding_name(factor)] = emb_init(n_words_src,
                                                  dim_per_factor[factor])

    rnn_init = init_gru if encoder == 'gru' else init_lstm

    # parameters for bidirectional RNN first layer in 2 directions
    params = rnn_init(params, prefix='enc_fw_0', nin=dim_emb, dim=dim)
    params = rnn_init(params, prefix='enc_bw_0', nin=dim_emb, dim=dim)

    # remember param names for sharing
    keys = [[key for key in params]]

    # parameters for all additional (>1) layers in case of deep encoder
    for i in range(1, encoder_layers):
        n = 2 * dim
        params = rnn_init(params, prefix='enc_fw_{}'.format(i), nin=n, dim=dim)
        params = rnn_init(params, prefix='enc_bw_{}'.format(i), nin=n, dim=dim)

        keys.append([key for key in params])

    return params, keys


def init_decoder_params(params, dim=0, dim_emb=0, n_words_trg=0,
                        decoder='gru', disable_attention=False, **kwargs):
    """
    Initialize decoder params (and connection between encoder and decoder)
    :param params:
    :param dim:
    :param dim_emb:
    :param n_words_trg:
    :param decoder:
    :param disable_attention:
    :return:
    """

    # first state of decoder
    ctx_dim = 2 * dim
    params = init_ff(params, prefix='ff_state', nin=ctx_dim, nout=dim)

    # decoder
    emb_init = norm_weight
    logger.warning('Word embeddings init: {}'.format(emb_init))
    params['Wemb_dec'] = emb_init(n_words_trg, dim_emb)
    use_gru = (decoder == 'gru')

    if disable_attention:
        init = init_gru_cond_simple if use_gru else init_lstm_cond_simple
    else:
        init = init_gru_cond if use_gru else init_lstm_cond

    init(params, prefix='decoder', nin=dim_emb, dim=dim, dimctx=ctx_dim)
    return params


def init_readout_params(params, dim=0, dim_emb=0, n_words_trg=0, **kwargs):
    """
    Initialize readout layer parameters.

    :param params:
    :param dim:
    :param ctx_dim:
    :param dim_emb:
    :param n_words_trg:
    :return:
    """
    ctx_dim = 2 * dim
    params = init_ff(
        params, prefix='ff_logit_hid', nin=dim, nout=dim_emb, ortho=False)
    params = init_ff(
        params, prefix='ff_logit_prev', nin=dim_emb, nout=dim_emb, ortho=False)
    params = init_ff(
        params, prefix='ff_logit_ctx', nin=ctx_dim, nout=dim_emb, ortho=False)
    params = init_ff(
        params, prefix='ff_logit', nin=dim_emb, nout=n_words_trg)

    return params


def init_params(config):
    """
    Initialize all parameters for encoder-decoder
    :param config:
    :param return_encoder_param_names: also return the keys of the encoder so we can find them back
    :return params:
    """

    # init encoder
    params = OrderedDict()
    params, keys = init_encoder_params(params, **config)

    # init decoder
    params = init_decoder_params(params, **config)

    # init readout
    params = init_readout_params(params, **config)

    # in case of multi-tasking we return the encoder parameter names
    n_shared_layers = config['n_shared_layers']
    if n_shared_layers > 0:
        return params, keys[n_shared_layers - 1]

    return params


def build_encoder(tparams, dim_emb=0, factors=1, dim_per_factor=None,
                  encoder='gru', encoder_layers=1,
                  dropout=False, dropout_src=0., dropout_emb=0.,
                  dropout_rec=0., trng=None, use_noise=False, use_mask=True,
                  **kwargs):
    """
    Build the bi-directional encoder
    :param dim_emb:
    :param factors:
    :param dim_per_factor:
    :param encoder: gru or lstm
    :param encoder_layers:
    :param dropout:
    :param dropout_src:
    :param dropout_emb:
    :param dropout_rec:
    :param trng:
    :param use_noise: apply dropout (use_noise) or not (test)
    :param use_mask: use mask for batch input
    :param kwargs:
    :return:
    """
    logger.info('Building {} encoder - dropout: {}, use_noise: {}'.format(
        encoder, dropout, use_noise))
    assert sum(dim_per_factor) == dim_emb, 'sum dim_per_factor != dim_emb'

    dropout = dropout and use_noise  # to disable dropout during test time
    rnn = gru if encoder == 'gru' else lstm

    # input to forward rnn (#factors x #words x #batch_size)
    x = tensor.tensor3('x_word', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32') if use_mask else None

    # input to backward rnn (x and x_mask reversed)
    x_bw = x[:, ::-1]
    x_mask_bw = x_mask[::-1] if use_mask else None

    n_timesteps = x.shape[1]  # length of longest sentence in batch
    batch_size = x.shape[2]  # size of this batch (can vary!)

    # forward RNN
    emb_fw = []
    for factor in range(factors):
        emb_fw.append(
            tparams[embedding_name(factor)][x[factor].flatten()])
    emb_fw = concatenate(emb_fw, axis=1)
    emb_fw = emb_fw.reshape([n_timesteps, batch_size, dim_emb])

    # drop out whole words by zero-ing their embeddings
    if dropout and dropout_src > 0.:
        logger.warn('Using src word dropout (p={})'.format(dropout_src))
        p = 1 - dropout_src
        word_drop = inv_dropout_mask((n_timesteps, batch_size, 1), trng, p)
        word_drop = tensor.tile(word_drop, (1, 1, dim_emb))
        emb_fw *= word_drop

    fw_layers = [rnn(
        tparams, emb_fw, trng=trng, prefix='enc_fw_0', mask=x_mask,
        dropout=dropout, dropout_inp=dropout_emb, dropout_rec=dropout_rec)]

    # backward rnn
    emb_bw = []
    for factor in range(factors):
        emb_bw.append(tparams[embedding_name(factor)][x_bw[factor].flatten()])
    emb_bw = concatenate(emb_bw, axis=1)
    emb_bw = emb_bw.reshape([n_timesteps, batch_size, dim_emb])

    # drop out the same words as above in forward rnn
    if dropout and dropout_src > 0.:
        emb_bw *= word_drop[::-1]

    bw_layers = [rnn(
        tparams, emb_bw, trng=trng, prefix='enc_bw_0', mask=x_mask_bw,
        dropout=dropout, dropout_inp=dropout_emb, dropout_rec=dropout_rec)]

    # add additional layers if Deep Encoder is specified
    for i in range(1, encoder_layers):  # add additional layers if specified
        input_states = concatenate(
            (fw_layers[i - 1][0], bw_layers[i - 1][0][::-1]), axis=2)

        fw_layers.append(
            rnn(tparams, input_states, trng=trng, prefix='enc_fw_{}'.format(i),
                mask=x_mask, dropout=dropout,
                dropout_inp=dropout_emb, dropout_rec=dropout_rec))

        bw_layers.append(
            rnn(tparams, input_states[::-1], trng=trng,
                prefix='enc_bw_{}'.format(i), mask=x_mask_bw, dropout=dropout,
                dropout_inp=dropout_emb, dropout_rec=dropout_rec))

    return x, x_mask, fw_layers, bw_layers


def build_decoder(tparams, x, x_mask, fw_layers, bw_layers, trng=None,
                  use_mask=True, one_step=False, dim=0, dim_emb=0,
                  dropout=False, dropout_trg=0., dropout_emb=0.,
                  dropout_hid=0., dropout_rec=0., decoder='gru',
                  disable_attention=False,
                  use_noise=True, **kwargs):
    """
    Build a decoder.

    :param tparams:
    :param x:
    :param x_mask:
    :param fw_layers:
    :param bw_layers:
    :param trng:
    :param use_mask:
    :param one_step:
    :param dim:
    :param dim_emb:
    :param dropout:
    :param dropout_trg:
    :param dropout_emb:
    :param dropout_hid:
    :param dropout_rec:
    :param decoder:
    :param use_noise:
    :param kwargs:
    :return:
    """
    dropout = dropout and use_noise
    logger.warn('Building decoder - use_noise: {}'.format(use_noise))

    if dropout:
        logger.warn('... with dropout')

    y = tensor.matrix('y', dtype='int64') if use_mask else tensor.vector(
        'y_sampler', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32') if use_mask else None

    n_timesteps_trg = y.shape[0] if not one_step else 1  # #words target
    batch_size = x.shape[2]  # size of this batch

    if one_step:  # sampling, so use a single word embedding (vector)
        emb_trg = tensor.switch(
            y[:, None] < 0, tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
            tparams['Wemb_dec'][y])
    else:
        # word embedding (target), we will shift the target sequence one time
        # step to the right. This is done because of the bi-gram connections
        # in the readout and decoder RNN. The first target will be all zeros
        # and we will not condition on the last output.
        emb_trg = tparams['Wemb_dec'][y.flatten()]
        emb_trg = emb_trg.reshape([n_timesteps_trg, batch_size, dim_emb])
        emb_trg_shifted = tensor.zeros_like(emb_trg)
        emb_trg_shifted = tensor.set_subtensor(
            emb_trg_shifted[1:], emb_trg[:-1])
        emb_trg = emb_trg_shifted

    # define context for decoder
    context = concatenate([fw_layers[-1][0], bw_layers[-1][0][::-1]],
                          axis=fw_layers[-1][0].ndim - 1)

    # initialize decoder (mean of the context across time)
    if use_mask:
        context_mean = (context * x_mask[:, :, None]).sum(0) \
                   / x_mask.sum(0)[:, None]
    else:
        context_mean = context.mean(0)

    # initialize decoder alternative (last fw state + first bw state)
    # WARNING: if you change this, you have to change it in other places too!
    # For example in the tools.
    # context_mean = concatenate(
    #     [fw_layers[-1][0][-1], bw_layers[-1][0][-1]], axis=2)

    # prepare dropout masks
    if dropout:

        # apply dropout to initial hidden state
        context_init = context_mean * inv_dropout_mask(
            (batch_size, 2 * dim), trng, 1 - dropout_hid)

        # apply dropout to target embeddings
        trg_dropout = inv_dropout_mask(
            (n_timesteps_trg, batch_size, 1), trng, 1 - dropout_trg)
        trg_dropout = tensor.tile(trg_dropout, (1, 1, dim_emb))
        emb_trg *= trg_dropout
    else:
        context_init = context_mean

    # initial decoder state
    init_state = ff(tparams, context_init, prefix='ff_state', activ=tanh)

    # when sampling (one step) we provide the state as an input to the function
    init_state_in = None
    if one_step:
        init_state_in = tensor.matrix('init_state', dtype=theano.config.floatX)
        init_state_dec = init_state_in
    else:
        init_state_dec = init_state

    # decoder - pass through the decoder conditional gru with attention
    if disable_attention:
        context = context_mean  # 2D context
        rnn_cond = gru_cond_simple if decoder == 'gru' else lstm_cond_simple
    else:
        rnn_cond = gru_cond if decoder == 'gru' else lstm_cond

    proj = rnn_cond(
        tparams, emb_trg, prefix='decoder', mask=y_mask,
        context=context, context_mask=x_mask,
        one_step=one_step, init_state=init_state_dec, dropout_inp=dropout_emb,
        dropout_rec=dropout_rec, dropout_ctx=dropout_hid)

    return y, y_mask, emb_trg, context, init_state, init_state_in, proj


def build_readout(tparams, trng, emb_trg, proj, one_step=False, dim=0,
                  dim_emb=0, dropout=False, dropout_emb=0., dropout_hid=0.,
                  use_noise=False, disable_attention=False,
                  context=None, **kwargs):
    """
    Build a readout layer that produces a prob. distribution over the next word.

    :param tparams:
    :param trng:
    :param emb_trg:
    :param proj:
    :param one_step:
    :param dim:
    :param dim_emb:
    :param dropout:
    :param dropout_emb:
    :param dropout_hid:
    :param use_noise:
    :param disable_attention:
    :param context: use a static context from the encoder (no attention)
    :param kwargs:
    :return:
    """

    dec_h = proj[0]  # hidden states of the decoder

    if disable_attention:
        contexts = context if one_step else context[None, :, :]
    else:
        contexts = proj[1]

    batch_size = proj[0].shape[1]
    retain_emb = 1 - dropout_emb
    retain_hid = 1 - dropout_hid

    # dropout hidden states, context vectors
    if dropout and dropout_hid > 0. and use_noise:
        logger.warn('Dropping out decoder states and contexts in readout')
        dec_h *= inv_dropout_mask((batch_size, dim), trng, retain_hid)
        contexts *= inv_dropout_mask((batch_size, 2 * dim), trng, retain_hid)

    # dropout target embeddings
    if dropout and dropout_emb > 0. and use_noise:
        logger.warn('Dropping out embeddings in readout')
        emb_trg *= inv_dropout_mask((batch_size, dim_emb), trng, retain_emb)

    # compute word probabilities
    lin = linear
    logit_hid = ff(tparams, dec_h, prefix='ff_logit_hid', activ=lin)
    logit_emb_prev = ff(tparams, emb_trg, prefix='ff_logit_prev', activ=lin)
    logit_context_i = ff(tparams, contexts, prefix='ff_logit_ctx', activ=lin)
    logit = tensor.tanh(logit_hid + logit_emb_prev + logit_context_i)

    if dropout and dropout_hid > 0. and use_noise:
        logger.warn('Dropping out readout output')
        logit *= inv_dropout_mask((batch_size, dim_emb), trng, retain_hid)

    logit = ff(tparams, logit, prefix='ff_logit', activ=lin)

    if not one_step:
        logit = logit.reshape([logit.shape[0] * logit.shape[1], logit.shape[2]])

    probs = tensor.nnet.softmax(logit)

    return probs


def build_cost(y, y_mask, probs, n_words_trg):
    """
    Build cost function
    :param y:
    :param y_mask:
    :param probs:
    :param n_words_trg:
    :return:
    """

    y_flat = y.flatten()

    # indexes of targets in y_flat
    y_flat_idx = tensor.arange(y_flat.shape[0]) * n_words_trg + y_flat

    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    return cost


def build_model(tparams, trng, config, use_mask=True, use_noise=True):
    """
    Builds an encoder-decoder model.

    :param tparams:
    :param config:
    :param use_mask: if model is used with mini-batches this should be True,
    otherwise False (single sentence input)
    :param use_noise:
    :return:
    """

    # build encoder
    x, x_mask, fw_layers, bw_layers = build_encoder(
        tparams, trng=trng, use_mask=use_mask, use_noise=use_noise, **config)

    # build decoder
    y, y_mask, emb_trg, context, init_state, _, proj = build_decoder(
        tparams, x, x_mask, fw_layers, bw_layers, trng=trng, use_mask=use_mask,
        one_step=False, use_noise=use_noise, **config)

    # build readout
    probs = build_readout(tparams, trng, emb_trg, proj, use_noise=use_noise,
                          context=context, **config)

    # build cost
    n_words_trg = config['n_words_trg']
    cost = build_cost(y, y_mask, probs, n_words_trg)

    # optional returns
    opt_ret = dict()
    if not config['disable_attention']:
        opt_ret['dec_alphas'] = proj[2]  # weights (alignment matrix)

    inputs = [x, x_mask, y, y_mask]

    return inputs, opt_ret, cost


def pred_probs(f_log_probs, prepare_batch, iterator, verbose=True):
    """
    Calculate the log probabilities on a given corpus using translation model.

    :param f_log_probs:
    :param prepare_batch:
    :param iterator:
    :param verbose:
    :return log probabilities:
    """
    probs = []

    n_done = 0
    num_target_words = 0  # total number of words

    for x, y in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask = prepare_batch(x, y)
        y = y[0]  # we only use the first factor (words) for target

        # keep track of how many words we have predicted - for perplexity
        num_target_words += sum(sum(y_mask))

        # get the actual probabilities
        pprobs = f_log_probs(x, x_mask, y, y_mask)

        if np.isnan(np.mean(pprobs)):
            logger.warning(x)
            logger.warning(pprobs)
            return

        for pp in pprobs:
            probs.append(pp)

        if np.isnan(np.mean(probs)):
            logger.warning(probs)
            return

        if verbose:
            logger.info('%d samples computed' % n_done)

    loss = np.array(probs)
    ppx = np.exp(loss.sum() / num_target_words)

    return loss, ppx


def prepare_batch(seqs_x, seqs_y, maxlen=None):
    """
    Batch preparation
    Creates padded matrices and masks for the data
    :param seqs_x:
    :param seqs_y:
    :param maxlen:
    :return:
    """
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        # check if we removed everything
        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    batch_size = len(seqs_x)
    factors = len(seqs_x[0][0])
    factors_trg = len(seqs_y[0][0])
    maxlen_x = np.max(lengths_x) + 1
    maxlen_y = np.max(lengths_y) + 1

    x = np.zeros((factors, maxlen_x, batch_size)).astype('int64')
    y = np.zeros((factors_trg, maxlen_y, batch_size)).astype('int64')
    x_mask = np.zeros((maxlen_x, batch_size)).astype(theano.config.floatX)
    y_mask = np.zeros((maxlen_y, batch_size)).astype(theano.config.floatX)

    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        # print(s_x)
        # print(s_y) FIXME
        x[:, :lengths_x[idx], idx] = zip(*s_x)
        x_mask[:lengths_x[idx] + 1, idx] = 1.
        y[:, :lengths_y[idx], idx] = zip(*s_y)
        y_mask[:lengths_y[idx] + 1, idx] = 1.

    return x, x_mask, y, y_mask
