import theano
import theano.tensor as tensor
import numpy as np
import logging

from utils import norm_weight, uniform_weight, ortho_weight, uniform_glorot, \
    _p

logger = logging.getLogger(__name__)
profile = False

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {
    'ff': ('param_init_fflayer', 'fflayer'),
    'gru': ('param_init_gru', 'gru_layer'),
    'lstm': ('param_init_lstm', 'lstm_layer'),
    'attn': ('param_init_attn', 'attn_layer')
}


def get_layer(name):
    """
    Get initializer and layer function pair
    :param name:
    :return initializer, layer:
    """
    fns = layers[name]
    return eval(fns[0]), eval(fns[1])


def dropout_layer(state_before, use_noise, trng):
    """
    Dropout layer

    :param state_before:
    :param use_noise:
    :param trng:
    :return projections:
    """
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj


def inv_dropout_mask(shape, trng, p):
    """
    Inverted dropout mask.

    Scaling is done during train time, so do *not* apply this during test time.

    :param shape:
    :param trng:
    :param p: probability of *retaining* a unit
    :return:
    """
    assert isinstance(p, float), 'retain probability p should be a float'
    assert p >= 0.5, 'are you sure you want to drop out more than 50% of units?'
    return trng.binomial(shape, p=p, n=1, dtype=theano.config.floatX) / p


def shared_dropout_layer(shape, use_noise, trng, value):
    """
    Dropout that will be re-used at different time steps
    :param shape:
    :param use_noise:
    :param trng:
    :param value:
    :return:
    """
    proj = tensor.switch(
        use_noise,
        trng.binomial(shape, p=value, n=1, dtype=theano.config.floatX),
        theano.shared(np.float32(value)))
    return proj

def param_init_fflayer(params, prefix='ff', nin=None, nout=None, ortho=True,
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
    b = np.zeros((nout,)).astype(theano.config.floatX)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = b

    logger.warn('FF init: {} gain: {} var:{}'.format(init, gain, np.var(W)))

    logger.info("Feed-forward layer parameters: {}".format(
        np.prod(W.shape) + np.prod(b.shape)))
    return params


def fflayer(tparams, state_below, prefix='ff', activ='lambda x: tensor.tanh(x)',
            **kwargs):
    """
    Feed-forward layer.

    Affine transformation + point-wise non-linearity.

    :param tparams:
    :param state_below:
    :param prefix:
    :param activ:
    :param kwargs:
    :return function:
    """
    logger.info("Building a feed-forward layer with prefix {}".format(prefix))
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[
            _p(prefix, 'b')])


def param_init_gru(params, prefix='gru', nin=None, dim=None, init='glorot',
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

    initializer = uniform_glorot if init == 'glorot' else uniform_weight if \
        init == 'uniform' else norm_weight
    logger.info('GRU W initialization with: {}'.format(init))

    # embedding to gates transformation weights, biases
    # concatenated for speed
    W_reset = initializer(nin, dim)
    W_update = initializer(nin, dim)
    W = np.concatenate([W_reset, W_update], axis=1)
    b = np.zeros((2 * dim,)).astype(theano.config.floatX)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = b

    # recurrent transformation weights for gates
    U = np.concatenate([ortho_weight(dim), ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = initializer(nin, dim)
    bx = np.zeros((dim,)).astype(theano.config.floatX)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = bx

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    # learn initial state
    if learn_init:
        init_h = initializer(1, dim)
        params[_p(prefix, 'init')] = init_h
        logger.warn('Init state params sum: {}'.format(sum(sum(init_h))))

    logger.info('GRU variances: W: (gates) {} {} W: {}'.format(
        np.var(W_reset), np.var(W_update), np.var(Wx)))

    n_init = np.prod(init_h.shape) if learn_init else 0

    logger.info('GRU parameters: {}'.format(
        np.prod(W.shape) + np.prod(U.shape) + np.prod(b.shape) +
        np.prod(Wx.shape) + np.prod(Ux.shape) + np.prod(bx.shape) + n_init))

    return params


def gru_layer(tparams, x, prefix='gru', mask=None, trng=None, learn_init=True,
              dropout=False, dropout_W=0., dropout_U=0.):
    """
    Gated Recurrent Unit layer with optional dropout
    :param tparams:
    :param x:
    :param prefix:
    :param mask:
    :param trng:
    :param learn_init: learn initial state (use zeros if false)
    :param dropout:
    :param dropout_W: dropout on input connections in [0,1]
    :param dropout_U: dropout on recurrent connections in [0,1]
    :return:
    """
    if dropout:
        logger.info('GRU with dropout W: {} U: {}'.format(dropout_W, dropout_U))

    nsteps = x.shape[0]
    if x.ndim == 3:
        n_samples = x.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[0]
    dim_emb = x.shape[x.ndim - 1]

    if mask is None:
        mask = tensor.alloc(1., x.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # prepare dropout masks --- x is the input (e.g. word embeddings)
    if dropout and dropout_W > 0.:
        logger.warn('dropout_W')
        p = 1 - dropout_W
        drop_W = inv_dropout_mask((2, nsteps, n_samples, dim_emb), trng, p)

    else:
        drop_W = tensor.alloc(1., 2)

    # prepare dropout masks for recurrent connections (if dropout)
    if dropout and dropout_U > 0.:
        logger.warn('dropout_U')
        p = 1 - dropout_U
        drop_U = inv_dropout_mask((2, n_samples, dim), trng, p)
    else:
        drop_U = tensor.alloc(1., 2)

    # input to the gates, concatenated
    x_ = tensor.dot(x * drop_W[0], tparams[_p(prefix, 'W')]) + \
         tparams[_p(prefix, 'b')]

    # input to compute the hidden state proposal
    xx_ = tensor.dot(x * drop_W[1], tparams[_p(prefix, 'Wx')]) + \
          tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux, drop_U):

        preact = tensor.dot(h_ * drop_U[0], U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_ * drop_U[1], Ux)
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

    init_state = tensor.tile(tparams[_p(prefix, 'init')], (n_samples, 1)) \
        if learn_init else tensor.alloc(0., n_samples, dim)

    out_info = [init_state]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')], tparams[_p(prefix, 'Ux')], drop_U]

    rval, updates = theano.scan(_step, sequences=seqs, outputs_info=out_info,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps, profile=profile, strict=True)
    rval = [rval]
    return rval


def param_init_lstm(params, prefix='lstm', nin=None, dim=None, init='glorot',
                    learn_init=True):
    """
    Initialize the LSTM parameters.

    """
    assert nin > 0, 'nin should be provided'
    assert dim > 0, 'dim should be provided'

    logger.warn('nin: {} dim: {}'.format(nin, dim))

    initializer = uniform_glorot if init == 'glorot' else uniform_weight if \
        init == 'uniform' else norm_weight
    logger.info('LSTM W initialization with: {}'.format(init))

    W = np.concatenate([initializer(nin=nin, nout=dim),
                        initializer(nin=nin, nout=dim),
                        initializer(nin=nin, nout=dim),
                        initializer(nin=nin, nout=dim)], axis=1)

    params[_p(prefix, 'W')] = W

    U = np.concatenate([ortho_weight(dim), ortho_weight(dim),
                        ortho_weight(dim), ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # peephole connections for gates
    peephole = np.concatenate([initializer(nin=1, nout=dim),
                               initializer(nin=1, nout=dim),
                               initializer(nin=1, nout=dim)], axis=0)
    params[_p(prefix, 'peephole')] = peephole

    b = np.concatenate([
        np.zeros(dim).astype(theano.config.floatX),
        np.ones(dim).astype(theano.config.floatX),  # init forget gate with 1s
        np.zeros(dim).astype(theano.config.floatX),
        np.zeros(dim).astype(theano.config.floatX),
    ])

    params[_p(prefix, 'b')] = b

    logger.info('LSTM variances: W: {} U: {}'.format(np.var(W), np.var(U)))

    # learn initial state
    if learn_init:
        init_h = initializer(1, dim)
        init_c = initializer(1, dim)
        params[_p(prefix, 'init_h')] = init_h
        params[_p(prefix, 'init_c')] = init_c
        logger.warn('Init state/mem var - h: {} c: {}'.format(
            np.var(init_h), np.var(init_c)))
        n_init = np.prod(init_h.shape) * 2
    else:
        n_init = 0

    logger.info('LSTM parameters: {}'.format(
        np.prod(W.shape) + np.prod(U.shape) + np.prod(b.shape) + n_init))

    return params


def lstm_layer(tparams, x, prefix='lstm', mask=None, trng=None, learn_init=True,
               dropout=False, dropout_W=0., dropout_U=0.):
    """
    LSTM layer with dropout support.

    :param tparams:
    :param x:
    :param prefix:
    :param mask:
    :param trng:
    :param learn_init:
    :param dropout:
    :param dropout_W: dropout on input connections
    :param dropout_U: dropout on recurrent connections
    :return:
    """
    logger.info('Building LSTM layer. Dropout: {}'.format(dropout))

    nsteps = x.shape[0]
    if x.ndim == 3:
        n_samples = x.shape[1]
    else:
        n_samples = 1

    assert mask is not None, 'mask must be provided'

    dim = tparams[_p(prefix, 'U')].shape[0]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # step function for use in scan loop
    def _step(m_, x_, h_, c_, drop_U):

        preact = tensor.dot(h_ * drop_U, tparams[_p(prefix, 'U')])
        preact += x_

        peep_i = c_ * tparams[_p(prefix, 'peephole')][0]
        peep_f = c_ * tparams[_p(prefix, 'peephole')][1]
        peep_o = c_ * tparams[_p(prefix, 'peephole')][2]

        # peeph = theano.printing.Print('ph')(peeph)

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim) + peep_i)
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim) + peep_f)
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim) + peep_o)
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    # prepare input dropout mask
    if dropout and dropout_W > 0.:
        drop_W = inv_dropout_mask(x.shape, trng, 1. - dropout_W)
        logger.warn('LSTM dropout W: {}'.format(dropout_W))
    else:
        drop_W = tensor.alloc(1.)

    # prepare recurrent dropout mask
    if dropout and dropout_U > 0.:
        drop_U = inv_dropout_mask((n_samples, dim), trng, 1. - dropout_U)
        logger.warn('LSTM dropout U: {}'.format(dropout_U))
    else:
        drop_U = tensor.alloc(1.)

    # pre-compute input transformation
    x = (tensor.dot(x * drop_W, tparams[_p(prefix, 'W')]) +
         tparams[_p(prefix, 'b')])

    # get initial hidden state
    init_h = tensor.tile(tparams[_p(prefix, 'init_h')], (n_samples, 1)) \
        if learn_init else tensor.alloc(0., n_samples, dim)

    # get initial memory cell
    init_c = tensor.tile(tparams[_p(prefix, 'init_c')], (n_samples, 1)) \
        if learn_init else tensor.alloc(0., n_samples, dim)
    # init_c = tensor.alloc(0., n_samples, dim)

    rval, updates = theano.scan(
        _step, sequences=[mask, x], outputs_info=[init_h, init_c],
        non_sequences=[drop_U], name=_p(prefix, '_layers'), n_steps=nsteps)

    return rval

if __name__ == '__main__':
    pass
