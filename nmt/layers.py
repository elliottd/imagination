from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import theano
import theano.tensor as tensor
import numpy as np
import logging
import pdb

from nmt.utils import pp, norm_weight, ortho_weight, uniform_glorot, \
    uniform_weight, tanh

logger = logging.getLogger(__name__)

profile = False
floatX = theano.config.floatX


def dropout(state_before, use_noise, trng):
    """
    Apply dropout (when use_noise==1), otherwise scale layer values
    :param state_before:
    :param use_noise:
    :param trng:
    :return:
    """
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(
            state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
        state_before * 0.5)
    return proj


def shared_dropout(shape, use_noise, trng, value):
    """
    Shared dropout mask (pervasive dropout)
    :param shape:
    :param use_noise:
    :param trng:
    :param value:
    :return:
    """
    return tensor.switch(use_noise,
                         trng.binomial(shape, p=value, n=1,
                                       dtype=floatX),
                         theano.shared(np.float32(value)))


def inv_dropout_mask(shape, trng, p):
    """
    Inverted dropout mask.

    Scaling is done during train time, so no scaling required during test time.

    :param shape:
    :param trng:
    :param p: probability of *retaining* a unit
    :return:
    """
    assert isinstance(p, float), 'retain probability p should be a float'
    assert p >= 0.5, 'are you sure you want to drop out more than 50% of units?'
    return trng.binomial(shape, p=p, n=1, dtype=floatX) / p


def init_ff(params, prefix='ff', nin=None, nout=None, ortho=True,
            init='glorot', gain=1.):
    """
    Initializes a feed-forward layer.

    :param params:
    :param prefix:
    :param nin:
    :param nout:
    :param ortho:
    :param init: initialization function, glorot or norm_weight
    :param gain: gain for use in glorot (use 'relu' when using relu)
    :return params:
    """
    initializer = uniform_glorot if init == 'glorot' else uniform_weight if \
        init == 'uniform' else norm_weight

    W = initializer(nin, nout, ortho=ortho, gain=gain)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = np.zeros((nout,)).astype(floatX)

    logger.info('FF init: {} gain: {} var:{}'.format(init, gain, np.var(W)))
    return params


def ff(tparams, x, prefix='ff', activ=tanh, **kwargs):
    """
    Feedforward layer: affine transformation + point-wise nonlinearity
    :param tparams:
    :param x:
    :param prefix:
    :param activ:
    :param kwargs:
    :return:
    """
    return activ(
        tensor.dot(x, tparams[pp(prefix, 'W')]) + tparams[pp(prefix, 'b')])


def init_gru(params, prefix='gru', nin=None, dim=None, init='glorot',
             learn_init=True):
    """
    Initializes a GRU layer.

    :param params:
    :param prefix:
    :param nin:
    :param dim:
    :param init: initializer to use for weights W, 'glorot' or 'normal'
    :param learn_init: learn initial state
    :return:
    """
    assert nin > 0, 'nin should be provided'
    assert dim > 0, 'dim should be provided'

    initializer = uniform_glorot if init == 'glorot' else norm_weight
    logger.info('GRU W initialization with: {}'.format(init))

    # embedding to gates transformation weights, biases
    # concatenated for speed
    W_reset = initializer(nin, dim)
    W_update = initializer(nin, dim)
    W = np.concatenate([W_reset, W_update], axis=1)
    b = np.zeros((2 * dim,)).astype(floatX)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = b

    # recurrent transformation weights for gates
    U = np.concatenate([ortho_weight(dim), ortho_weight(dim)], axis=1)
    params[pp(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = initializer(nin, dim)
    bx = np.zeros((dim,)).astype(floatX)
    params[pp(prefix, 'Wx')] = Wx
    params[pp(prefix, 'bx')] = bx

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[pp(prefix, 'Ux')] = Ux

    # calculate the number of parameters for this GRU
    n_params = np.prod(W.shape) + np.prod(U.shape) + np.prod(b.shape) + \
               np.prod(Wx.shape) + np.prod(Ux.shape) + np.prod(bx.shape)

    # learn initial state
    if learn_init:
        init_h = initializer(1, dim)
        params[pp(prefix, 'init')] = init_h
        n_params += np.prod(init_h.shape)

    logger.info('GRU parameters: {}'.format(n_params))

    return params


def gru(tparams, x, prefix='gru', mask=None, trng=None, learn_init=True,
        dropout=False, dropout_inp=0., dropout_rec=0.):
    """
    Gated Recurrent Unit layer with optional dropout
    :param tparams:
    :param x:
    :param prefix:
    :param mask:
    :param trng:
    :param learn_init: learn initial state (use zeros if false)
    :param dropout:
    :param dropout_inp: dropout on input connections in [0,1]
    :param dropout_rec: dropout on recurrent connections in [0,1]
    :return:
    """
    if dropout:
        logger.info('GRU w/ dropout emb: {} rec: {}'.format(dropout_inp,
                                                            dropout_rec))
    else:
        logger.info('GRU without dropout (for prediction?)')

    nsteps = x.shape[0]
    if x.ndim == 3:
        n_samples = x.shape[1]
    else:
        n_samples = 1

    dim = tparams[pp(prefix, 'Ux')].shape[0]
    dim_emb = x.shape[-1]

    if mask is None:
        mask = tensor.alloc(1., x.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # prepare dropout masks --- x is the input (e.g. word embeddings)
    if dropout and dropout_inp > 0.:
        p = 1 - dropout_inp
        drop_inp = inv_dropout_mask((2, nsteps, n_samples, dim_emb), trng, p)

    else:
        drop_inp = tensor.alloc(1., 2)

    # prepare dropout masks for recurrent connections (if dropout)
    if dropout and dropout_rec > 0.:
        p = 1 - dropout_rec
        drop_rec = inv_dropout_mask((2, n_samples, dim), trng, p)
    else:
        drop_rec = tensor.alloc(1., 2)

    # input to the gates, concatenated
    x_ = tensor.dot(x * drop_inp[0], tparams[pp(prefix, 'W')]) + \
         tparams[pp(prefix, 'b')]

    # input to compute the hidden state proposal
    xx_ = tensor.dot(x * drop_inp[1], tparams[pp(prefix, 'Wx')]) + \
          tparams[pp(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux, drop_rec):

        preact = tensor.dot(h_ * drop_rec[0], U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_ * drop_rec[1], Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, x_, xx_]

    init_state = tensor.tile(tparams[pp(prefix, 'init')], (n_samples, 1)) \
        if learn_init else tensor.alloc(0., n_samples, dim)

    o_info = [init_state]

    shared_vars = [tparams[pp(prefix, 'U')], tparams[pp(prefix, 'Ux')],
                   drop_rec]

    _step = _step_slice
    result, _ = theano.scan(
        _step, sequences=seqs, outputs_info=o_info, non_sequences=shared_vars,
        name=pp(prefix, '_layers'), n_steps=nsteps, profile=profile,
        strict=True)
    return [result]


def init_lstm(params, prefix='lstm', nin=None, dim=None, init='glorot',
              learn_init=True):
    """
    Initialize the LSTM parameters.

    """
    assert nin > 0, 'nin should be provided'
    assert dim > 0, 'dim should be provided'

    logger.info('nin: {} dim: {}'.format(nin, dim))

    initializer = uniform_glorot if init == 'glorot' else norm_weight
    logger.info('LSTM W initialization with: {}'.format(init))

    W = np.concatenate([initializer(nin=nin, nout=dim),
                        initializer(nin=nin, nout=dim),
                        initializer(nin=nin, nout=dim),
                        initializer(nin=nin, nout=dim)], axis=1)

    params[pp(prefix, 'W')] = W

    U = np.concatenate([ortho_weight(dim), ortho_weight(dim),
                        ortho_weight(dim), ortho_weight(dim)], axis=1)
    params[pp(prefix, 'U')] = U

    b = np.concatenate([
        np.zeros(dim).astype(floatX),
        np.ones(dim).astype(floatX),  # init forget gate with 1s
        np.zeros(dim).astype(floatX),
        np.zeros(dim).astype(floatX),
    ])

    params[pp(prefix, 'b')] = b

    n_params = np.prod(W.shape) + np.prod(U.shape) + np.prod(b.shape)

    # learn initial state
    if learn_init:
        init_h = initializer(1, dim)
        params[pp(prefix, 'init_h')] = init_h
        n_params += np.prod(init_h.shape)

        init_c = initializer(1, dim)
        params[pp(prefix, 'init_c')] = init_c
        n_params += np.prod(init_c.shape)

    logger.info('LSTM parameters: {}'.format(n_params))

    return params


def lstm(tparams, x, prefix='lstm', mask=None, trng=None, learn_init=True,
         dropout=False, dropout_inp=0., dropout_rec=0.):
    """
    LSTM layer with dropout support.

    :param tparams:
    :param x:
    :param prefix:
    :param mask:
    :param trng:
    :param learn_init:
    :param dropout:
    :param dropout_inp: dropout on input connections (e.g. word embeddings)
    :param dropout_rec: dropout on recurrent connections
    :return:
    """

    n_steps = x.shape[0]

    if x.ndim == 3:
        n_samples = x.shape[1]
    else:
        n_samples = 1

    if mask is None:
        mask = tensor.alloc(1., x.shape[0], 1)

    dim = tparams[pp(prefix, 'U')].shape[0]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # step function for use in scan loop
    def _step(m_, x_, h_, c_, drop_rec):

        preact = tensor.dot(h_ * drop_rec, tparams[pp(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    # prepare input dropout mask
    if dropout and dropout_inp > 0.:
        drop_inp = inv_dropout_mask(x.shape, trng, 1 - dropout_inp)
        logger.info('LSTM dropout W: {}'.format(dropout_inp))
    else:
        drop_inp = tensor.alloc(1.)

    # prepare recurrent dropout mask
    if dropout and dropout_rec > 0.:
        drop_rec = inv_dropout_mask((n_samples, dim), trng, 1 - dropout_rec)
        logger.info('LSTM dropout U: {}'.format(dropout_rec))
    else:
        drop_rec = tensor.alloc(1.)

    # pre-compute input transformation
    x = (tensor.dot(x * drop_inp, tparams[pp(prefix, 'W')]) +
         tparams[pp(prefix, 'b')])

    # get initial hidden state
    init_h = tensor.tile(tparams[pp(prefix, 'init_h')], (n_samples, 1)) \
        if learn_init else tensor.alloc(0., n_samples, dim)

    # get initial memory cell
    init_c = tensor.tile(tparams[pp(prefix, 'init_c')], (n_samples, 1)) \
        if learn_init else tensor.alloc(0., n_samples, dim)

    result, _ = theano.scan(
        _step, sequences=[mask, x], outputs_info=[init_h, init_c],
        non_sequences=[drop_rec], name=pp(prefix, 'layers'), n_steps=n_steps)

    # for now, only return the states h
    return [result[0]]


def init_gru_cond(params, prefix='gru_cond', nin=None, dim=None,
                  init='glorot',
                  dimctx=None, nin_nonlin=None, dim_nonlin=None):
    """
    Conditional Gated Recurrent Unit (GRU) layer (e.g. used for Decoder).

    :param params:
    :param prefix:
    :param nin:
    :param dim:
    :param dimctx:
    :param nin_nonlin:
    :param dim_nonlin:
    :return:
    """

    initializer = uniform_glorot if init == 'glorot' else norm_weight
    logger.info('GRU-COND W initialization with: {}'.format(init))

    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    W = np.concatenate([initializer(nin, dim),
                        initializer(nin, dim)], axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = np.zeros((2 * dim,)).astype(floatX)
    U = np.concatenate([ortho_weight(dim_nonlin), ortho_weight(dim_nonlin)],
                       axis=1)
    params[pp(prefix, 'U')] = U

    Wx = initializer(nin_nonlin, dim_nonlin)
    params[pp(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux')] = Ux
    params[pp(prefix, 'bx')] = np.zeros((dim_nonlin,)).astype(floatX)

    U_nl = np.concatenate([ortho_weight(dim_nonlin), ortho_weight(dim_nonlin)],
                          axis=1)
    params[pp(prefix, 'U_nl')] = U_nl
    params[pp(prefix, 'b_nl')] = np.zeros((2 * dim_nonlin,)).astype(
        floatX)

    Ux_nl = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux_nl')] = Ux_nl
    params[pp(prefix, 'bx_nl')] = np.zeros((dim_nonlin,)).astype(
        floatX)

    # context to LSTM
    Wc = initializer(dimctx, dim * 2)
    params[pp(prefix, 'Wc')] = Wc

    Wcx = initializer(dimctx, dim)
    params[pp(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = initializer(dim, dimctx)
    params[pp(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = initializer(dimctx, dimctx)
    params[pp(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = np.zeros((dimctx,)).astype(floatX)
    params[pp(prefix, 'b_att')] = b_att

    # attention:
    U_att = initializer(dimctx, 1)
    params[pp(prefix, 'U_att')] = U_att
    c_att = np.zeros((1,)).astype(floatX)
    params[pp(prefix, 'c_tt')] = c_att  # attention: combined -> hidden
    W_comb_att = initializer(dim, dimctx)
    params[pp(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = initializer(dimctx, dimctx)
    params[pp(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = np.zeros((dimctx,)).astype(floatX)
    params[pp(prefix, 'b_att')] = b_att

    # attention:
    U_att = initializer(dimctx, 1)
    params[pp(prefix, 'U_att')] = U_att
    c_att = np.zeros((1,)).astype(floatX)
    params[pp(prefix, 'c_tt')] = c_att

    return params


def gru_cond(tparams, x, prefix='gru', mask=None, trng=None,
             context=None, one_step=False,
             init_memory=None, init_state=None, context_mask=None,
             dropout=False, dropout_inp=None,
             dropout_rec=None, dropout_ctx=None):
    """
    Conditional Gated Recurrent Unit (GRU) layer.
    
    :param tparams:
    :param x:
    :param prefix:
    :param mask:
    :param trng:
    :param context:
    :param one_step:
    :param init_memory:
    :param init_state:
    :param context_mask:
    :param dropout:
    :param dropout_inp: dropout on target word embeddings
    :param dropout_rec: dropout on recurrent connections
    :param dropout_ctx: dropout on context states
    :return:
    """

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    n_steps = x.shape[0]
    if x.ndim == 3:
        n_samples = x.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., x.shape[0], 1)

    dim = tparams[pp(prefix, 'Ux')].shape[1]
    dim_emb = x.shape[x.ndim - 1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'

    # helper function to get slices of a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # pre-compute dropout masks
    if dropout and dropout_inp > 0.:
        logger.info('GRU dropout W on with p={}'.format(dropout_inp))
        retain_emb = 1 - dropout_inp
        drop_emb = inv_dropout_mask((2, n_samples, dim_emb), trng, retain_emb)
    else:
        drop_emb = tensor.alloc(1., 2)

    if dropout and dropout_rec > 0.:
        logger.info('GRU dropout U on with p={}'.format(dropout_rec))
        retain_rec = 1 - dropout_rec
        drop_rec = inv_dropout_mask((5, n_samples, dim), trng, retain_rec)
    else:
        drop_rec = tensor.alloc(1., 5)

    if dropout and dropout_ctx > 0.:
        logger.info('GRU dropout CTX on with p={}'.format(dropout_ctx))
        retain_ctx = 1 - dropout_ctx
        drop_ctx = inv_dropout_mask((4, n_samples, 2 * dim), trng, retain_ctx)
    else:
        drop_ctx = tensor.alloc(1., 4)

    # projected context
    pctx_ = tensor.dot(context * drop_ctx[0], tparams[pp(prefix, 'Wc_att')]) + \
            tparams[pp(prefix, 'b_att')]

    # projected input x (e.g. target embeddings) for gates and for hidden state
    px = tensor.dot(x * drop_emb[0], tparams[pp(prefix, 'W')]) + \
         tparams[pp(prefix, 'b')]
    pxx = tensor.dot(x * drop_emb[1], tparams[pp(prefix, 'Wx')]) + \
          tparams[pp(prefix, 'bx')]

    def _step_slice(m_, px_, pxx_, h_, ctx_, alpha_, pctx_, cc_, drop_rec,
                    drop_ctx, U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):

        preact1 = tensor.dot(h_ * drop_rec[0], U)
        preact1 += px_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_ * drop_rec[1], Ux)
        preactx1 *= r1
        preactx1 += pxx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = tensor.dot(h1 * drop_rec[2], W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        # pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__ * drop_ctx[1], U_att) + c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))

        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        preact2 = tensor.dot(h1 * drop_rec[3], U_nl) + b_nl
        preact2 += tensor.dot(ctx_ * drop_ctx[2], Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1 * drop_rec[4], Ux_nl) + bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_ * drop_ctx[3], Wcx)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    seqs = [mask, px, pxx]
    # seqs = [mask, px, pxx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[pp(prefix, 'U')],
                   tparams[pp(prefix, 'Wc')],
                   tparams[pp(prefix, 'W_comb_att')],
                   tparams[pp(prefix, 'U_att')],
                   tparams[pp(prefix, 'c_tt')],
                   tparams[pp(prefix, 'Ux')],
                   tparams[pp(prefix, 'Wcx')],
                   tparams[pp(prefix, 'U_nl')],
                   tparams[pp(prefix, 'Ux_nl')],
                   tparams[pp(prefix, 'b_nl')],
                   tparams[pp(prefix, 'bx_nl')]]

    non_seqs = [pctx_, context, drop_rec, drop_ctx] + shared_vars

    if one_step:  # sampling
        out_info = [init_state, None, None]
        result = _step(*(seqs + out_info + non_seqs))
    else:
        out_info = [init_state,
                    tensor.alloc(0., n_samples, context.shape[2]),
                    tensor.alloc(0., n_samples, context.shape[0])]
        result, _ = theano.scan(
            _step, sequences=seqs, outputs_info=out_info,
            non_sequences=non_seqs, name=pp(prefix, 'layers'), n_steps=n_steps,
            profile=profile, strict=True)

    return result


def init_lstm_cond(params, prefix='lstm', nin=None, dim=None,
                   init='glorot', dimctx=None):
    # FIXME not implemented
    return params


def lstm_cond(tparams, x, prefix='gru', mask=None, trng=None,
              context=None, one_step=False,
              init_memory=None, init_state=None, context_mask=None,
              dropout=False, dropout_inp=None,
              dropout_rec=None, dropout_ctx=None):
    # FIXME not implemented
    return None


def init_gru_cond_simple(params, prefix='gru_cond_simple',
                         nin=None, dim=None, dimctx=None, init='glorot'):
    """
    GRU decoder without attention.

    :param params:
    :param prefix:
    :param nin:
    :param dim:
    :param dimctx:
    :param init:
    :return:
    """

    assert nin > 0, 'nin must be set'
    assert dim > 0, 'dim must be set'
    assert dimctx > 0, 'dimctx must be set'

    # first initialize as if this is a normal GRU
    params = init_gru(params, prefix, nin=nin, dim=dim, init=init,
                      learn_init=False)

    initializer = uniform_glorot if init == 'glorot' else norm_weight

    # context to GRU gates
    Wc = initializer(dimctx, dim * 2)
    params[pp(prefix, 'Wc')] = Wc

    # context to hidden proposal
    Wcx = initializer(dimctx, dim)
    params[pp(prefix, 'Wcx')] = Wcx

    return params


def gru_cond_simple(tparams, x, prefix='gru_cond_simple',
                    mask=None, trng=None, context=None, one_step=False,
                    init_state=None, dropout=False, dropout_inp=None,
                    dropout_rec=None, dropout_ctx=None, **kwargs):
    """
    Conditional Gated Recurrent Unit (GRU) layer.

    :param tparams:
    :param x: input (e.g. target word embeddings)
    :param prefix:
    :param mask:
    :param trng:
    :param context:
    :param one_step:
    :param init_state:
    :param dropout:
    :param dropout_inp: dropout on target word embeddings
    :param dropout_rec: dropout on recurrent connections
    :param dropout_ctx: dropout on context states
    :return:
    """

    assert context, 'Context must be provided'
    assert context.ndim == 2, 'Context must be 2-d: #sample x dim'

    if one_step:
        assert init_state, 'previous state must be provided'

    n_steps = x.shape[0]

    if x.ndim == 3:
        n_samples = x.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., x.shape[0], 1)

    dim = tparams[pp(prefix, 'U')].shape[0]
    dim_emb = x.shape[-1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # helper function to get slices of a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # pre-compute dropout masks
    if dropout and dropout_inp > 0.:
        logger.info('GRU dropout inp on with p={}'.format(dropout_inp))
        retain_emb = 1 - dropout_inp
        drop_emb = inv_dropout_mask((2, n_samples, dim_emb), trng, retain_emb)
    else:
        drop_emb = tensor.alloc(1., 2)

    if dropout and dropout_rec > 0.:
        logger.info('GRU dropout rec on with p={}'.format(dropout_rec))
        retain_rec = 1 - dropout_rec
        drop_rec = inv_dropout_mask((2, n_samples, dim), trng, retain_rec)
    else:
        drop_rec = tensor.alloc(1., 2)

    if dropout and dropout_ctx > 0.:
        logger.info('GRU dropout ctx on with p={}'.format(dropout_ctx))
        retain_ctx = 1 - dropout_ctx
        drop_ctx = inv_dropout_mask((2, n_samples, 2 * dim), trng, retain_ctx)
    else:
        drop_ctx = tensor.alloc(1., 2)

    # projected input (to gates and hidden state proposal)
    px = tensor.dot(x * drop_emb[0], tparams[pp(prefix, 'W')]) + tparams[
        pp(prefix, 'b')]
    pxx = tensor.dot(x * drop_emb[1], tparams[pp(prefix, 'Wx')]) + tparams[
        pp(prefix, 'bx')]

    # projected context (to gates and hidden state proposal)
    pctx = tensor.dot(context * drop_ctx[0], tparams[pp(prefix, 'Wc')])
    pctxx = tensor.dot(context * drop_ctx[1], tparams[pp(prefix, 'Wcx')])

    def _step(m_, px_, pxx_, h_, pctx_, pctxx_, drop_rec_, U_, Ux_):

        # compute the gates (together because it is faster)
        preact = tensor.dot(h_ * drop_rec_[0], U_)
        preact += px_
        preact += pctx_
        preact = tensor.nnet.sigmoid(preact)

        # slice out reset gate and update gate from the result
        dim_gate = Ux_.shape[0]
        r = _slice(preact, 0, dim_gate)
        u = _slice(preact, 1, dim_gate)

        # compute hidden state proposal
        preactx = tensor.dot(h_ * drop_rec_[1], Ux_)
        preactx *= r
        preactx += pxx_
        preactx += pctxx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare inputs to scan
    seqs = [mask, px, pxx]
    out_info = [init_state]
    shared_vars = [tparams[pp(prefix, 'U')], tparams[pp(prefix, 'Ux')]]
    non_seqs = [pctx, pctxx, drop_rec] + shared_vars

    if one_step:  # sampling
        result = _step(*(seqs + out_info + non_seqs))

    else:  # training
        result, _ = theano.scan(
            _step, sequences=seqs, outputs_info=out_info,
            non_sequences=non_seqs, name=pp(prefix, 'layers'),
            n_steps=n_steps, profile=profile, strict=True)

    return [result]


def init_lstm_cond_simple(params, prefix='lstm_cond_simple',
                          nin=None, dim=None, init='glorot', dimctx=None):
    """
    Initialize the LSTM decoder parameters.

    :param params:
    :param prefix:
    :param nin:
    :param dim:
    :param init:
    :param dimctx:
    :return:
    """
    assert nin > 0, 'nin should be provided'
    assert dim > 0, 'dim should be provided'
    assert dimctx > 0, 'dimctx should be provided'

    params = init_lstm(params, prefix=prefix, nin=nin, dim=dim, init=init,
                       learn_init=False)

    initializer = uniform_glorot if init == 'glorot' else norm_weight

    # context to gates and hidden state proposal
    Wc = initializer(dimctx, dim * 4)
    params[pp(prefix, 'Wc')] = Wc

    return params


def lstm_cond_simple(tparams, x, prefix='lstm_cond_simple',
                     mask=None, trng=None, context=None, one_step=False,
                     init_state=None, init_memory=None,
                     dropout=False, dropout_inp=None,
                     dropout_rec=None, dropout_ctx=None, **kwargs):
    """
    LSTM layer with dropout support.

    :param tparams:
    :param x:
    :param prefix:
    :param mask:
    :param trng:
    :param dropout:
    :param dropout_inp: dropout on input connections
    :param dropout_rec: dropout on recurrent connections
    :return:
    """
    n_steps = x.shape[0]
    if x.ndim == 3:
        n_samples = x.shape[1]
    else:
        n_samples = 1

    if one_step:
        assert init_state, 'previous state must be provided'

    if mask is None:
        mask = tensor.alloc(1., x.shape[0], 1)

    U = tparams[pp(prefix, 'U')]
    dim = U.shape[0]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # step function for use in scan loop
    def _step(m_, x_, h_, c_, pctx_, U_, drop_rec):

        preact = tensor.dot(h_ * drop_rec, U_)
        preact += x_  # condition on input
        preact += pctx_  # condition on context too

        dim = U_.shape[0]
        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        # update memory cell
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        # update state
        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    # prepare input dropout mask
    if dropout and dropout_inp > 0.:
        drop_inp = inv_dropout_mask(x.shape, trng, 1 - dropout_inp)
        logger.info('LSTM dropout W: {}'.format(dropout_inp))
    else:
        drop_inp = tensor.alloc(1.)

    # prepare context dropout mask
    if dropout and dropout_ctx > 0.:
        drop_ctx = inv_dropout_mask(context.shape, trng, 1 - dropout_ctx)
        logger.info('LSTM dropout CTX: {}'.format(dropout_ctx))
    else:
        drop_ctx = tensor.alloc(1.)

    # prepare recurrent dropout mask
    if dropout and dropout_rec > 0.:
        drop_rec = inv_dropout_mask((n_samples, dim), trng, 1 - dropout_rec)
        logger.info('LSTM dropout U: {}'.format(dropout_rec))
    else:
        drop_rec = tensor.alloc(1.)

    # pre-compute input transformation
    x = tensor.dot(x * drop_inp, tparams[pp(prefix, 'W')]) + \
        tparams[pp(prefix, 'b')]

    # pre-compute context transformation
    pctx = tensor.dot(context * drop_ctx, tparams[pp(prefix, 'Wc')])

    # get initial state/cell
    if init_state is None:
        logger.warn('Init state initialized with zeros ???')
        init_state = tensor.alloc(0., n_samples, dim)

    if init_memory is None:
        logger.warn('Init memory initialized with zeros')
        init_memory = tensor.alloc(0., n_samples, dim)

    seqs = [mask, x]
    out_info = [init_state, init_memory]
    non_seqs = [pctx, U, drop_rec]

    if one_step:  # sampling
        result = _step(*(seqs + out_info + non_seqs))

    else:
        result, _ = theano.scan(
            _step, sequences=seqs, outputs_info=out_info,
            non_sequences=non_seqs, name=pp(prefix, 'layers'),
            n_steps=n_steps)

    # for now, only return state h
    return [result[0]]

