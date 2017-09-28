from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import theano
import theano.tensor as tensor
import numpy as np
import copy
import logging

from nmt.model import build_encoder, build_decoder, build_readout

profile = False

logger = logging.getLogger(__name__)


def build_sampler(tparams, config, trng, return_alignment=False):
    """
    Build a tools
    :param tparams:
    :param config:
    :param trng:
    :param return_alignment:
    :return:
    """

    # build encoder
    x, _, fw_layers, bw_layers = build_encoder(
        tparams, trng=trng, use_mask=False, use_noise=False, **config)

    # build decoder
    y, _, emb_trg, context, init_state, init_state_input, proj = build_decoder(
        tparams, x, None, fw_layers, bw_layers, trng=trng, use_mask=False,
        one_step=True, use_noise=False, **config)

    # compile decoder initialization function f_init
    logger.info('Building f_init...')
    f_init = theano.function([x], [init_state, context], name='f_init',
                             profile=profile)
    logger.info('Done')

    # build readout
    probs = build_readout(tparams, trng, emb_trg, proj, one_step=True,
                          use_noise=False, context=context, **config)

    # get the next hidden state
    # TODO
    # proj[0] = theano.printing.Print('proj_rah')(proj[0])
    next_state = proj[0]

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    logger.info('Building f_next..')
    inputs = [y, context, init_state_input]
    outputs = [probs, next_sample, next_state]

    if return_alignment:
        dec_alphas = proj[2]  # alignment matrix (attention model)
        outputs.append(dec_alphas)

    f_next = theano.function(inputs, outputs, name='f_next', profile=profile)
    logger.info('Done')

    return f_init, f_next


def generate_sample(f_init, f_next, x, trng=None, k=1, maxlen=30,
                    stochastic=True, argmax=False,
                    return_alignment=False, suppress_unk=False):
    """
    Generate sample, either with stochastic sampling or beam search. Note that,
    this function iteratively calls f_init and f_next functions.
    :param f_init:
    :param f_next:
    :param x:
    :param trng:
    :param k:
    :param maxlen:
    :param stochastic:
    :param argmax:
    :param return_alignment:
    :param suppress_unk:
    :return:
    """
    assert type(f_init) == list, 'f_init must be a list (ensemble support)'
    assert type(f_next) == list, 'f_next must be a list (ensemble support)'

    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    sample_word_probs = []
    alignment = []

    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    # to store hypotheses
    hyp_samples = [[]] * live_k
    word_probs = [[]] * live_k
    hyp_scores = np.zeros(live_k).astype('float32')
    hyp_states = []
    if return_alignment:
        hyp_alignment = [[] for _ in range(live_k)]

    num_models = len(f_init)
    next_state = [None] * num_models
    context_init = [None] * num_models
    next_p = [None] * num_models
    dec_alphas = [None] * num_models

    # get initial state of decoder rnn and encoder context
    for i in range(num_models):
        ret = f_init[i](x)
        next_state[i] = ret[0]
        context_init[i] = ret[1]

    next_w = -1 * np.ones((1,)).astype('int64')  # BOS-indicator

    for ii in range(maxlen):
        for i in range(num_models):
            context = np.tile(context_init[i], [live_k, 1])
            inps = [next_w, context, next_state[i]]
            ret = f_next[i](*inps)  # predict next word

            next_p[i], next_w_tmp, next_state[i] = ret[0], ret[1], ret[2]

            if return_alignment:
                # dimension of dec_alpha: (k-beam-size, #input-hidden-units)
                dec_alphas[i] = ret[3]

            if suppress_unk:
                next_p[i][:, 1] = -np.inf

        if stochastic:

            if argmax:
                nw = sum(next_p)[0].argmax()  # word with highest probability
            else:
                nw = next_w_tmp[0]
            sample.append(nw)
            sample_score += np.log(next_p[0][0, nw])
            if nw == 0:
                break
        else:  # beam search

            # cand_scores shape: [beam, voc_size]
            cand_scores = hyp_scores[:, None] - sum(np.log(next_p))  # next_p holds next word distributions for a batch
            probs = sum(next_p) / num_models
            cand_flat = cand_scores.flatten()  # shape: [beam * voc_size] - so all the scores within the beam!
            probs_flat = probs.flatten()
            ranks_flat = cand_flat.argpartition(k - dead_k - 1)[
                         :(k - dead_k)]  # indexes of top scores e.g. [3,6,2,7,4] w/ beam 5

            # averaging the attention weights across models
            if return_alignment:
                mean_alignment = sum(dec_alphas) / num_models

            voc_size = next_p[0].shape[1]  # just the vocabulary size
            trans_indices = ranks_flat // voc_size  # recovers which sentence in the beam the rank belongs to
            word_indices = ranks_flat % voc_size  # modulo here because we index over the vocabulary k times
            costs = cand_flat[ranks_flat]  # scores for the selected hypotheses

            new_hyp_samples = []
            new_hyp_scores = np.zeros(k - dead_k).astype('float32')
            new_word_probs = []
            new_hyp_states = []

            if return_alignment:
                # holds the history of attention weights for each time step for each of the surviving hypothesis
                # dimensions (live_k * target_words * source_hidden_units]
                # at each time step we append the attention weights corresponding to the current target word
                new_hyp_alignment = [[] for _ in range(k - dead_k)]

            # ti -> index of k-best hypothesis
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                new_word_probs.append(word_probs[ti] + [probs_flat[ranks_flat[idx]].tolist()])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append([copy.copy(next_state[i][ti]) for i in xrange(num_models)])
                if return_alignment:
                    # get history of attention weights for the current hypothesis
                    new_hyp_alignment[idx] = copy.copy(hyp_alignment[ti])
                    # extend the history with current attention weights
                    new_hyp_alignment[idx].append(mean_alignment[ti])

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            word_probs = []
            if return_alignment:
                hyp_alignment

            # sample and sample_score hold the k-best translations and their scores
            for idx in range(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    sample_word_probs.append(new_word_probs[idx])
                    if return_alignment:
                        alignment.append(new_hyp_alignment[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    word_probs.append(new_word_probs[idx])
                    if return_alignment:
                        hyp_alignment.append(new_hyp_alignment[idx])
            hyp_scores = np.array(hyp_scores)

            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = np.array([w[-1] for w in hyp_samples])
            next_state = [np.array(state) for state in zip(*hyp_states)]

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in range(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
                sample_word_probs.append(word_probs[idx])
                if return_alignment:
                    alignment.append(hyp_alignment[idx])

    if not return_alignment:
        alignment = [None for _ in range(len(sample))]

    return sample, sample_score, sample_word_probs, alignment


def print_samples(x, y, trng, f_init, f_next, maxlen, factors,
                  worddicts_src_r, worddicts_trg_r, unk_symbol, num_samples=5,
                  stochastic=True):
    """
    Print a few samples from x.
    :param x:
    :param y: targets
    :param trng:
    :param f_init:
    :param f_next:
    :param maxlen:
    :param factors:
    :param worddicts_src_r:
    :param worddicts_trg_r:
    :param unk_symbol:
    :param num_samples: how many samples to take and print
    :param stochastic: stochastic sampling or return best
    :return:
    """
    for jj in range(np.minimum(num_samples, x.shape[1])):
        sample, sample_score, sample_word_probs, alignment = generate_sample(
            [f_init], [f_next], x[:, :, jj][:, :, None], trng=trng, k=1,
            maxlen=maxlen, stochastic=stochastic, argmax=False, return_alignment=False)

        # source
        sentence = []
        for pos in range(x.shape[1]):

            if x[0, pos, jj] == 0:
                break

            w = [worddicts_src_r[factor].get(x[factor, pos, jj], unk_symbol) for factor in range(factors)]
            sentence.append('|'.join(w))

        logger.info('Source {} : {}'.format(jj, ' '.join(sentence)))

        # truth
        sentence = []
        for vv in y[:, jj]:
            if vv == 0:
                break
            if vv in worddicts_trg_r[0]:
                sentence.append(worddicts_trg_r[-1].get(vv, unk_symbol))

        logger.info('Truth {} : {}'.format(jj, ' '.join(sentence)))

        # sample
        sentence = []
        if stochastic:
            ss = sample
        else:
            score = score / np.array([len(s) for s in sample])
            ss = sample[score.argmin()]

        for vv in ss:
            if vv == 0:
                break
            sentence.append(worddicts_trg_r[0].get(vv, unk_symbol))

        logger.info('Sample {} : {}'.format(jj, ' '.join(sentence)))


if __name__ == '__main__':
    pass
