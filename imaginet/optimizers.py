from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import logging
import numpy as np
import theano
from theano import tensor
from six import iteritems, itervalues


logger = logging.getLogger(__name__)
profile = False


def adam(lr, tparams, grads, inp, cost, opt_ret=None, beta1=0.9, beta2=0.999,
         e=1e-8):  # FIXME
    """
    The adam optimizer

    :param lr:
    :param tparams:
    :param grads:
    :param inp:
    :param cost:
    :param opt_ret: return optional output
    :param beta1:
    :param beta2:
    :param e:
    :return f_grad_shared, f_update:
    """
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k) for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    outs = [cost]

    logger.warn('Adam: beta1=%1.6f beta2=%1.6f epsilon=%1.0e' % (beta1, beta2, e))

    if opt_ret is not None:  # we expect a dictionary here for opt_ret
        outs += list(opt_ret.values())

    f_grad_shared = theano.function(inp, outs, updates=gsup, profile=profile)

    updates = []

    t_prev = theano.shared(np.float32(0.))
    t = t_prev + 1.
    lr_t = lr * tensor.sqrt(1. - beta2**t) / (1. - beta1**t)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0., p.name + '_mean')
        v = theano.shared(p.get_value() * 0., p.name + '_variance')
        m_t = beta1 * m + (1. - beta1) * g
        v_t = beta2 * v + (1. - beta2) * g ** 2
        step = lr_t * m_t / (tensor.sqrt(v_t) + e)
        p_t = p - step
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t_prev, t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, inp, cost, opt_ret=None, rho=0.99, eta=1e-7):
    """
    Adadelta optimizer
    :param lr:
    :param tparams:
    :param grads:
    :param inp:
    :param cost:
    :param opt_ret:
    :param rho: adadelta rho
    :param eta: adadelta eta
    :return f_grad_shared, f_update:
    """
    logger.info('Adadelta with rho={} and eta={}'.format(rho, eta))
    zipped_grads = [theano.shared(p.get_value() * np.float32(0.), name='%s_grad' % k) for k, p in iteritems(tparams)]
    running_up2 = [theano.shared(p.get_value() * np.float32(0.), name='%s_rup2' % k) for k, p in iteritems(tparams)]
    running_grads2 = [theano.shared(p.get_value() * np.float32(0.), name='%s_rgrad2' % k) for k, p in iteritems(tparams)]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, rho * rg2 + (1-rho) * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    outs = [cost]

    if opt_ret is not None:  # we expect a dictionary here for opt_ret
        outs += list(opt_ret.values())

    f_grad_shared = theano.function(inp, outs, updates=zgup+rg2up,
                                    profile=profile)

    updir = [-tensor.sqrt(ru2 + eta) / tensor.sqrt(rg2 + eta) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2,
                                     running_grads2)]
    ru2up = [(ru2, rho * ru2 + (1-rho) * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itervalues(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, inp, cost, opt_ret=None):
    """
    Stochastic gradient descent (SGD) optimizer

    :param lr:
    :param tparams:
    :param grads:
    :param inp:
    :param cost:
    :param opt_ret:
    :return f_grad_shared, f_update:
    """
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    outs = [cost]
    if opt_ret is not None:  # opt_ret should be a dict
        outs += list(opt_ret.values())

    f_grad_shared = theano.function(inp, outs, updates=gsup, profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itervalues(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update

def adagrad(lr, tparams, grads, inp, cost, opt_ret=None):
    '''
    Adagrad is a per-parameter adaptive optimisation method. It works by keeping
    a record of the squared(gradients) at each update, and uses this record
    to perform relative weight modifications.

    cached_p = cached_p + gradient**2
    weight^new = weight - learning_rate*gradient / sqrt(cached_p+epsilon)
    '''
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k) for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    epsilon = 1e-6

    outs = [cost]
    if opt_ret is not None:  # opt_ret should be a dict
        outs += list(opt_ret.values())

    f_grad_shared = theano.function(inp, outs, updates=gsup, profile=profile)

    acc_up = []
    pup = []

    for p,g in zip(tparams.values(), gshared):
      acc = theano.shared(p.get_value() * 0)
      acc_t = acc + g ** 2
      acc_up.append((acc, acc_t))

      p_t = p - (lr / tensor.sqrt(acc_t + epsilon)) * g
      pup.append((p, p_t))

    f_update = theano.function([lr], [], updates=pup+acc_up, profile=profile)
    return f_grad_shared, f_update
