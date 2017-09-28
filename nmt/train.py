from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import time
import numpy as np
import yaml

from pprint import pformat
from io import open
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from nmt.utils import invert_dictionary, clip_grad_norm, unzip, \
    load_dictionary, load_json, dump_json, ignored
from nmt.data_iterator_mem import TextIterator
from nmt.sampler import build_sampler, generate_sample, print_samples
from nmt.bleu import SimpleBleuValidator
from nmt.model import init_params, init_theano_params, load_params, build_model, \
    pred_probs, prepare_batch
from nmt.optimizers import *

from imaginet.train import Trainer as ImaginetTrainer

profile = False
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config, shared_theano_params=None):
        self.config = config
        self.update_idx = 0
        self.epoch_idx = 0
        self.history_errs = []
        self.history_bleu = []
        self.shared_params = shared_theano_params


    def apply_shared_theano_params(self, shared_theano_params, params, theano_params):
        """
        Override the parameters of the model with the provided 
        shared parameters. Used for Multi-task Learning.

        Auxiliary function.

        Note that we do not need to override all of them, just the ones
        in the provided dictionary.

        :param shared_theano_params:
        :param theano_params:
        :param params:
        :return: theano_params, params
        """
        for k in shared_theano_params:
            theano_params[k] = shared_theano_params[k]
            assert params[k].shape == theano_params[k].get_value().shape, 'shape mismatch'
            params[k] = shared_theano_params[k].get_value()
            logger.info(
                'Using external theano parameter: {} Shape: {}'.format(k, params[k].shape))
            return params, theano_params

    def train(self, model_name=None, output_dir=None, src_train=None,
              trg_train=None,
              src_valid=None, trg_valid=None, src_dicts=None,
              trg_dicts=None, factors=1, factors_trg=1,
              n_words_src=50000, n_words_trg=50000,
              dim_emb=100, dim_per_factor=(100,), dim=100, dim_att=200,
              encoder='gru', encoder_layers=1,
              decoder='gru_cond', optimizer='adadelta', learning_rate=1e-3,
              decay_c=0., clip_c=1., alpha_c=0.,
              dropout=False, dropout_src=0., dropout_trg=0., dropout_emb=0.,
              dropout_rec=0., dropout_hid=0.,
              batch_size=80, valid_batch_size=80, k=5, maxlen=50,
              max_epochs=20, bleu_script='nmt/multi-bleu.perl',
              bleu_val_burnin=0, val_set_out='validation.txt',
              validation_frequency=-1, display_frequency=100, save_frequency=-1,
              sample_frequency=200,
              beam_size=12, track_n_models=3, finish_after=-1,
              unk_symbol='<UNK>', eos_symbol='</s>', patience=10,
              early_stopping='cost',
              reload=False, verbose=1, disp_alignments=True, mtl=False,
              mtl_ratio=(), mtl_configs=(), mtl_decoder=False,
              n_shared_layers=1,
              **kwargs):
        """
        Train an NMT system
        :return:
        """

        # log options
        config = self.config
        logger.info(pformat(self.config))

        # Model options
        model_path = os.path.join(output_dir, model_name + '.npz')
        config_path = os.path.join(output_dir, model_name + '.json')

        # create output dir if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # load dictionaries and invert them
        worddicts_src = [load_dictionary(d) for d in src_dicts]
        worddicts_trg = [load_dictionary(d) for d in trg_dicts]
        worddicts_src_r = [invert_dictionary(d) for d in worddicts_src]
        worddicts_trg_r = [invert_dictionary(d) for d in worddicts_trg]

        # reload options
        if reload:
            if os.path.exists(config_path):
                logger.info('Reloading model options: %s' % config_path)
                config = load_json(config_path)
            else:
                logger.info('Did NOT reload model options (file did not exist)')

        logger.info('Loading data')
        train = TextIterator(src_train, trg_train, src_dicts, trg_dicts,
                             batch_size=batch_size,
                             maxlen=maxlen, n_words_source=n_words_src,
                             n_words_target=n_words_trg,
                             shuffle_each_epoch=True, sort_by_length=True,
                             maxibatch_size=20,
                             factors=factors, factors_trg=factors_trg)

        valid = TextIterator(src_valid, trg_valid, src_dicts, trg_dicts,
                             batch_size=batch_size,
                             maxlen=maxlen, n_words_source=n_words_src,
                             n_words_target=n_words_trg,
                             shuffle_each_epoch=False, sort_by_length=True,
                             maxibatch_size=20,
                             factors=factors, factors_trg=factors_trg)

        logger.info('Building model')
        params, encoder_param_names = init_params(config)

        # reload parameters
        if reload and os.path.exists(model_path):
            logger.info('Reloading model parameters')
            params = load_params(model_path, params)

        tparams = init_theano_params(params)

        if self.shared_params is not None:
            # multi-task support
            # we replace whatever parameters we already have at this point with
            # the ones that we received as optional input
            # this needs to be done BEFORE building the model
            params, tparams = self.apply_shared_theano_params(self.shared_params, params, tparams)

        # random generator and global dropout/noise switch for this model
        trng = RandomStreams(1234)

        inps, opt_ret, cost = build_model(tparams, trng, config)
        # x, x_mask, y, y_mask = inps

        cost = cost.mean()

        logger.info('Building tools')
        f_init, f_next = build_sampler(tparams, config, trng)

        # apply L2 regularization on weights
        if decay_c > 0.:
            decay_c = theano.shared(np.float32(decay_c), name='decay_c')
            weight_decay = 0.
            for kk, vv in iteritems(tparams):
                weight_decay += (vv ** 2).sum()
            weight_decay *= decay_c
            cost += weight_decay

        # regularize the alpha weights
        if alpha_c > 0. and not decoder.endswith('simple'):
            alpha_c = theano.shared(np.float32(alpha_c), name='alpha_c')
            alpha_reg = alpha_c * (
                (tensor.cast(y_mask.sum(0) // x_mask.sum(0), 'float32')[:,
                 None] -
                 opt_ret['dec_alphas'].sum(0)) ** 2).sum(1).mean()
            cost += alpha_reg

        # after all regularizers - compile the computational graph for cost
        logger.info('Building f_cost...')
        f_cost = theano.function(inps, cost, profile=profile)
        logger.info('Done')

        logger.info('Computing gradient...')
        grads = tensor.grad(cost, wrt=list(itervalues(tparams)))
        grads = clip_grad_norm(grads, clip_c)
        logger.info('Done')

        # compile the optimizer, the actual computational graph is compiled here
        lr = tensor.scalar(name='lr')
        logger.info('Building optimizers...')
        f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps,
                                                  cost, opt_ret)
        logger.info('Done')

        # log probability function (for validation, so use model without noise!)
        logger.info('Building f_log_probs...')
        test_inp, _, test_cost = build_model(
            tparams, trng, config, use_mask=True, use_noise=False)
        f_log_probs = theano.function(test_inp, test_cost, profile=profile)
        logger.info('Done')

        # bleu validation
        bleu_validator = SimpleBleuValidator(
            tparams, config, trng, f_init, f_next, k=beam_size,
            src_dicts=worddicts_src, trg_idict=worddicts_trg_r[0],
            normalize=True, main_loop=self) if bleu_script else None

        # multi-task learning
        mtl_tasks = []
        shared_params = OrderedDict()
        for k in encoder_param_names:
            shared_params[k] = tparams[k]

        if mtl:
            logger.info('Preparing MTL tasks')
            task_config = yaml.load(open(mtl_configs[0], mode='rb'))
            if task_config['model'] == 'imaginet':
                task_config['exp_id'] = self.config['exp_id']
                mtl_tasks.append(ImaginetTrainer(task_config, shared_params))
            elif task_config['model'] == 'nmt':
                mtl_tasks.append(Trainer(task_config, shared_params))
            assert sum(mtl_ratio) == 1., 'MTL ratio must sum to 1'

        # to check how many times a task was executed
        task_stats = np.zeros(len(mtl_tasks) + 1)

        # start of optimization main loop
        logger.info('Optimization started...')

        early_stop = False
        saved_model_paths = []  # history of saved models
        best_params = None
        bad_counter = 0

        # reload history
        if reload and os.path.exists(model_path):
            self.history_errs = list(np.load(model_path)['history_errs'])
            self.history_bleu = list(np.load(model_path)['history_bleu'])
            self.update_idx = np.load(model_path)['update_idx']

        # set frequencies - if -1 is specified then freq set to #iters in epoch
        validation_frequency = len(train) // batch_size \
            if validation_frequency == -1 else validation_frequency
        save_frequency = len(train) // batch_size \
            if save_frequency == -1 else save_frequency
        sample_frequency = len(train) // batch_size \
            if sample_frequency == -1 else sample_frequency

        # save initial model so we can re-use parameters (seed)
        logger.info('Saving initial model')
        params = unzip(tparams)
        with open(model_path + '.init', mode='wb') as f:
            np.savez(f, history_errs=self.history_errs,
                     history_bleu=self.history_bleu, update_idx=self.update_idx,
                     **params)
        dump_json(config, config_path)
        logger.info('Done saving model')

        for epoch_idx in range(max_epochs):

            self.epoch_idx = epoch_idx
            # self.update_idx // (len(train) // batch_size)

            n_samples = 0

            # iterate over data batches
            for x_, y_ in train:

                # multi-task learning -- we simply do other tasks until we are
                # allowed to perform the main task (this loop)
                if mtl:
                    n_tasks = len(mtl_ratio)
                    task = 1
                    while task > 0:
                        task = np.random.choice(n_tasks, 1, replace=False,
                                                p=mtl_ratio)[0]
                        task_stats[task] += 1

                        if task > 0:
                            mtl_tasks[task - 1].train_next_batch()
                            # print('Training on task {:d}'.format(task))

                # NMT training
                n_samples += len(x_)
                self.update_idx += 1

                x, x_mask, y, y_mask = prepare_batch(x_, y_, maxlen=None)
                y = y[0]  # only use first target factor for NMT

                inputs = [x, x_mask, y, y_mask]

                if x is None:
                    logger.warning('Empty mini-batch! maxlen={}'.format(maxlen))
                    self.update_idx -= 1
                    continue

                # get error on this batch
                update_start_time = time.time()
                ret_vals = f_grad_shared(*inputs)
                cost = ret_vals[0]

                # do the update on parameters
                f_update(learning_rate)

                update_time = time.time() - update_start_time

                # check for bad numbers
                if np.isnan(cost) or np.isinf(cost):
                    logger.warning('NaN detected')
                    return 1., 1., 1.

                # verbose
                if np.mod(self.update_idx, display_frequency) == 0:
                    if disp_alignments:  # display info with max alpha value
                        logger.info(
                            'Epoch %4d Update %8d Cost %4.8f UD %0.12f Max-alpha %0.4f' % (
                                self.epoch_idx, self.update_idx, cost,
                                update_time, ret_vals[1].max()))
                    else:  # display general info
                        logger.info(
                            'Epoch %4d Update %8d Cost %4.8f UD %0.12f' % (
                                self.epoch_idx, self.update_idx, cost,
                                update_time))

                # generate some samples
                if np.mod(self.update_idx, sample_frequency) == 0:
                    print_samples(x, y, trng, f_init, f_next, maxlen, factors,
                                  worddicts_src_r, worddicts_trg_r, unk_symbol)

                # validation
                if np.mod(self.update_idx, validation_frequency) == 0:

                    # intrinsic validation
                    valid_errs, perplexity = pred_probs(
                        f_log_probs, prepare_batch, valid)
                    valid_err = valid_errs.mean()

                    if np.isnan(valid_err):
                        logger.warning('valid_err NaN detected')
                        early_stop = True
                        break

                    # output validation info
                    logger.info('Validation error: {:1.12f} PPX: {:f}'.format(
                        valid_err, perplexity))

                    # BLEU validation
                    if bleu_validator and self.update_idx >= bleu_val_burnin:
                        bleu_score = bleu_validator.evaluate_model()
                        logger.info('BLEU = {}'.format(bleu_score))

                    # save the best 3 models according to early-stopping
                    if track_n_models > 0 and len(self.history_errs) > 0:

                        if early_stopping == 'cost':
                            if valid_err <= min(self.history_errs):
                                logger.info(
                                    'Saving model at epoch {} / iter {}...'.format(
                                        self.epoch_idx, self.update_idx))
                                path = os.path.join(
                                    output_dir,
                                    '{}.ep{}.iter{}.npz'.format(model_name,
                                                                self.epoch_idx,
                                                                self.update_idx))
                                with open(path, mode='wb') as f:
                                    np.savez(f, history_errs=self.history_errs,
                                             history_bleu=self.history_bleu,
                                             update_idx=self.update_idx,
                                             **unzip(tparams))

                                saved_model_paths.append(path)
                                logger.info('Done saving model')


                        # Save a model only if we've exceeding the point where
                        # we start measuring BLEU scores
                        elif early_stopping == 'bleu' and self.update_idx >= bleu_val_burnin:
                            if len(self.history_bleu) > 0 and bleu_score >= max(self.history_bleu):
                                bestbleuhandle = open('%s/bestBLEU' % output_dir, 'w')
                                bestbleuhandle.write("%f" % bleu_score)
                                bestbleuhandle.close()
                                logger.info(
                                    'Saving model at epoch {} / iter {}...'.format(
                                        self.epoch_idx, self.update_idx))
                                path = os.path.join(
                                    output_dir,
                                    '{}.ep{}.iter{}.bleu{}.npz'.format(model_name,
                                                                self.epoch_idx,
                                                                self.update_idx,
                                                                bleu_score))
                                with open(path, mode='wb') as f:
                                    np.savez(f, history_errs=self.history_errs,
                                             history_bleu=self.history_bleu,
                                             update_idx=self.update_idx,
                                             **unzip(tparams))

                                saved_model_paths.append(path)
                                logger.info('Done saving model')

                        # Remove un-needed saved models if necessary
                        if len(saved_model_paths) > track_n_models:
                            path = saved_model_paths[0]
                            logger.info('Deleting old model {}'.format(path))
                            with ignored(OSError):
                                os.remove(path)

                            saved_model_paths.pop(0)

                    # remember the validation result
                    self.history_errs.append(valid_err)
                    if early_stopping == 'bleu' and self.update_idx >= bleu_val_burnin:
                        # Remember the BLEU score at this point
                        self.history_bleu.append(bleu_score)

                    # reset bad counter (patience) if best validation so far
                    if early_stopping == 'cost':
                        if self.update_idx == 0 or valid_err <= \
                                np.array(self.history_errs).min():
                            best_params = unzip(tparams)
                            if mtl:
                                # Force the other tasks to save too
                                mtl_tasks[0].save(string=".cost{}".format(valid_err))
                            if bad_counter > 0:
                                bad_counter -= 1
                    elif early_stopping == 'bleu':
                        if self.update_idx >= bleu_val_burnin:
                            if bleu_score >= max(self.history_bleu):
                                best_params = unzip(tparams)
                                if mtl:
                                    # Force the other tasks to save too
                                    mtl_tasks[0].save(string=".bleu{}".format(bleu_score))
                                if bad_counter > 0:
                                    bad_counter -= 1

                    # save the best model so far (according to cost)
                    logger.info('Saving best model (according to {})'.format(early_stopping))
                    if best_params is not None:
                        params = best_params
                    else:
                        params = unzip(tparams)
                    np.savez(model_path,
                             history_errs=self.history_errs,
                             history_bleu=self.history_bleu,
                             update_idx=self.update_idx, **params)
                    logger.info('Done saving best model')

                    # check for early stop
                    if early_stopping == 'cost':
                        if len(self.history_errs) > patience and valid_err >= \
                                np.array(self.history_errs)[:-patience].min():
                            bad_counter += 1
                            logger.warn('Bad validation result. {}/{}'.format(
                                bad_counter, patience))

                            if bad_counter >= patience:
                                logger.info('Early stop activated.')
                                early_stop = True
                    elif early_stopping == 'bleu':
                        if len(self.history_bleu) > patience and bleu_score <= \
                                np.array(self.history_bleu)[:-patience].max():
                            bad_counter += 1
                            logger.warn('Bad validation result. {}/{}'.format(
                                bad_counter, patience))

                            if bad_counter >= patience:
                                logger.info('Early stop activated.')
                                early_stop = True

            # finish after this many updates
            if self.update_idx == finish_after:
                logger.info('Finishing after {:d} iterations!'.format(
                    self.update_idx))
                early_stop = True

            if early_stop:
                logger.info('Early Stop!')
                return 0

            if mtl:
                logger.info(task_stats / task_stats.sum())

        logger.info('Seen {:d} samples'.format(n_samples))
        logger.info('Finished with main loop')
        return 0


if __name__ == '__main__':
    pass
