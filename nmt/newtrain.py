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
    def __init__(self, config, shared_theano_params=None,
              model_name=None, output_dir=None, src_train=None,
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
              reload=False, verbose=1, disp_alignments=False, mtl=False,
              mtl_ratio=(), mtl_configs=(), mtl_decoder=False,
              n_shared_layers=1,
              **kwargs):

        self.config = config
        logger.info(pformat(self.config))

        self.model_name = config['model_name']
        self.output_dir = ''.join([config['output_dir'], str(config['exp_id'])])
        # create output dir if it does not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.config['output_dir']= self.output_dir
        self.src_train = config['src_train']
        self.trg_train = config['trg_train']
        self.src_valid = config['src_valid']
        self.trg_valid = config['trg_valid']
        self.src_dicts = config['src_dicts']
        self.trg_dicts = config['trg_dicts']
        self.factors = config['factors']
        self.factors_trg = config['factors_trg']
        self.n_words_src = config['n_words_src']
        self.n_words_trg = config['n_words_trg']
        self.unk_symbol = config['unk_symbol']
        self.eos_symbol = config['eos_symbol']
        self.early_stopping = config['early_stopping']
        self.finish_after = config['finish_after']
        self.patience = config['patience']
        self.mtl = config['mtl']
        self.mtl_ratio = config['mtl_ratio']
        self.learning_rate = float(config['learning_rate'])
        self.disp_alignments = config['disp_alignments']
        self.maxlen = config['maxlen']
        self.track_n_models = config['track_n_models']
        self.bleu_val_burnin = config['bleu_val_burnin']
        self.decay_c = config['decay_c']
        self.alpha_c = config['alpha_c']
        self.clip_c = config['clip_c']
        self.optimizer = config['optimizer']
        self.batch_size = config['batch_size']
        self.validation_frequency = config['validation_frequency']
        self.save_frequency = config['save_frequency']
        self.sample_frequency = config['sample_frequency']
        self.display_frequency = config['display_frequency']
        # Model options
        self.model_path = os.path.join(self.output_dir, self.model_name + '.npz')
        self.config_path = os.path.join(self.output_dir, self.model_name + '.json')

        self.update_idx = 0
        self.epoch_idx = 0
        self.history_errs = []
        self.history_bleu = []
        self.shared_params = shared_theano_params
        self.best_params = None

        # load dictionaries and invert them
        self.worddicts_src = [load_dictionary(d) for d in self.src_dicts]
        self.worddicts_trg = [load_dictionary(d) for d in self.trg_dicts]
        self.worddicts_src_r = [invert_dictionary(d) for d in self.worddicts_src]
        self.worddicts_trg_r = [invert_dictionary(d) for d in self.worddicts_trg]

        logger.info('Loading data')
        self.train_iterator = TextIterator(self.src_train, self.trg_train,
                self.src_dicts, self.trg_dicts,
                             batch_size=self.batch_size,
                             maxlen=self.maxlen, n_words_source=self.n_words_src,
                             n_words_target=self.n_words_trg,
                             shuffle_each_epoch=True, sort_by_length=True,
                             maxibatch_size=20,
                             factors=self.factors, factors_trg=self.factors_trg)

        self.valid_iterator = TextIterator(self.src_valid, self.trg_valid,
                self.src_dicts, self.trg_dicts,
                             batch_size=self.batch_size,
                             maxlen=self.maxlen, n_words_source=self.n_words_src,
                             n_words_target=self.n_words_trg,
                             shuffle_each_epoch=False, sort_by_length=True,
                             maxibatch_size=20,
                             factors=self.factors, factors_trg=self.factors_trg)

        self.build_models()
        self.initial_save()
        # set frequencies - if -1 is specified then freq set to #iters in epoch
        self.validation_frequency = len(self.train_iterator) // self.batch_size \
            if self.validation_frequency == -1 else self.validation_frequency
        self.save_frequency = len(self.train_iterator) // self.batch_size \
            if self.save_frequency == -1 else self.save_frequency
        self.sample_frequency = len(self.train_iterator) // self.batch_size \
            if self.sample_frequency == -1 else self.sample_frequency
        self.display_frequency = self.display_frequency

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

    def train_on_batch(self, x_, y_):
        """
        Train on a single batch online.

        :param batch:
        :return:
        """

        update_start_time = time.time()

        # prepare data
        x, x_mask, y, y_mask = prepare_batch(x_, y_, maxlen=None)
        y = y[0]  # only use first target factor for NMT
        inputs = [x, x_mask, y, y_mask]

        if x is None:
            logger.warning('Empty mini-batch! maxlen={}'.format(maxlen))
            self.update_idx -= 1
            return

        # get error on this batch
        ret_vals = self.f_grad_shared(*inputs)
        cost = ret_vals[0]

        # do the update on parameters
        self.f_update(self.learning_rate)

        update_time = time.time() - update_start_time

        # check for bad numbers
        if np.isnan(cost) or np.isinf(cost):
            logger.warning('NaN detected')
            return 1., 1., 1.

        return cost, update_time

    def train_next_batch(self):
        """
        Train on the next batch."

        TODO: does the iterator need to be explicitly rebuilt?
        """
        try:
            x_, y_ = next(self.train_iterator)
        except StopIteration:
            #self.train_iterator = self.get_data_iterator()
            #self.validate(self.epoch)
            #self.display_epoch_info()
            #self.reset_epoch_info()
            self.epoch_idx += 1
            x_, y_ = next(self.train_iterator)

        batch_loss, update_time = self.train_on_batch(x_, y_)
        self.update_idx += 1
        self.display_update_info(batch_loss, update_time)

    def display_update_info(self, loss, update_time):
        # verbose
        if np.mod(self.update_idx, self.display_frequency) == 0:
            if self.disp_alignments:  # display info with max alpha value
                logger.info(
                    'Epoch %4d Update %8d Cost %4.8f UD %0.12f Max-alpha %0.4f' % (
                        self.epoch_idx, self.update_idx, cost,
                        update_time, ret_vals[1].max()))
            else:  # display general info
                        logger.info(
                            'Epoch %4d Update %8d Cost %4.8f UD %0.12f' % (
                                self.epoch_idx, self.update_idx, loss,
                                update_time))

    def build_models(self, *kwargs):
        logger.info('Building model')
        self.params, self.encoder_param_names = init_params(self.config)

        # reload parameters
        if self.config['reload'] and os.path.exists(self.model_path):
            logger.info('Reloading model parameters')
            self.params = load_params(self.model_path, self.params)

        self.tparams = init_theano_params(self.params)

        if self.shared_params is not None:
            # multi-task support
            # we replace whatever parameters we already have at this point with
            # the ones that we received as optional input
            # this needs to be done BEFORE building the model
            self.params, self.tparams = self.apply_shared_theano_params(self.shared_params, self.params, self.tparams)

        # random generator and global dropout/noise switch for this model
        self.trng = RandomStreams(1234)

        inps, opt_ret, cost = build_model(self.tparams, self.trng, self.config)

        cost = cost.mean()

        logger.info('Building tools')
        self.f_init, self.f_next = build_sampler(self.tparams, self.config, self.trng)

        # apply L2 regularization on weights
        if self.decay_c > 0.:
            decay_c = theano.shared(np.float32(self.decay_c), name='decay_c')
            weight_decay = 0.
            for kk, vv in iteritems(self.tparams):
                weight_decay += (vv ** 2).sum()
            weight_decay *= self.decay_c
            cost += weight_decay

        # regularize the alpha weights
        if self.alpha_c > 0. and not decoder.endswith('simple'):
            alpha_c = theano.shared(np.float32(self.alpha_c), name='alpha_c')
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
        grads = tensor.grad(cost, wrt=list(itervalues(self.tparams)))
        grads = clip_grad_norm(grads, self.clip_c)
        logger.info('Done')

        # compile the optimizer, the actual computational graph is compiled here
        lr = tensor.scalar(name='lr')
        logger.info('Building optimizers...')
        self.f_grad_shared, self.f_update = eval(self.optimizer)(lr, self.tparams, grads, inps,
                                                  cost, opt_ret)
        logger.info('Done')

        # log probability function (for validation, so use model without noise!)
        logger.info('Building f_log_probs...')
        self.test_inp, _, self.test_cost = build_model(
            self.tparams, self.trng, self.config, use_mask=True, use_noise=False)
        self.f_log_probs = theano.function(self.test_inp, self.test_cost, profile=profile)
        logger.info('Done')

    def initial_save(self):
        # save initial model so we can re-use parameters (seed)
        logger.info('Saving initial model')
        self.params = unzip(self.tparams)
        with open(self.model_path + '.init', mode='wb') as f:
            np.savez(f, history_errs=self.history_errs,
                     history_bleu=self.history_bleu, update_idx=self.update_idx,
                     **self.params)
        dump_json(self.config, self.config_path)
        logger.info('Done saving model')

    def save(self):
        # save the best model so far (according to cost)
        logger.info('Saving best model (according to {})'.format(self.early_stopping))
        if self.best_params is not None:
            self.params = self.best_params
        else:
            self.params = unzip(self.tparams)
        np.savez(self.model_path,
                 history_errs=self.history_errs,
                 history_bleu=self.history_bleu,
                 update_idx=self.update_idx, **self.params)
        logger.info('Done saving best model')

    def train_loop(self):
        for epoch_idx in range(self.config['max_epochs']):

            self.epoch_idx = epoch_idx
            # self.update_idx // (len(train) // batch_size)

            self.n_samples = 0

            # iterate over data batches
            for x_, y_ in self.train_iterator:

                # multi-task learning -- we simply do other tasks until we are
                # allowed to perform the main task (this loop)
                if self.mtl:
                    n_tasks = len(self.mtl_ratio)
                    task = 1
                    while task > 0:
                        task = np.random.choice(n_tasks, 1, replace=False,
                                                p=self.mtl_ratio)[0]
                        self.task_stats[task] += 1

                        if task > 0:
                            self.mtl_tasks[task - 1].train_next_batch()
                            # print('Training on task {:d}'.format(task))

                # NMT training
                self.n_samples += len(x_)
                self.update_idx += 1

                cost, update_time = self.train_on_batch(x_, y_)

                # verbose
                if np.mod(self.update_idx, self.display_frequency) == 0:
                    if self.disp_alignments:  # display info with max alpha value
                        logger.info(
                            'Epoch %4d Update %8d Cost %4.8f UD %0.12f Max-alpha %0.4f' % (
                                self.epoch_idx, self.update_idx, cost,
                                update_time, ret_vals[1].max()))
                    else:  # display general info
                        self.display_update_info(cost, update_time)

                # generate some samples
                if np.mod(self.update_idx, self.sample_frequency) == 0:
                    x, x_mask, y, y_mask = prepare_batch(x_, y_, maxlen=None)
                    y = y[0]  # only use first target factor for NMT
                    print_samples(x, y, self.trng, self.f_init, self.f_next,
                                  self.maxlen, self.factors,
                                  self.worddicts_src_r, self.worddicts_trg_r,
                                  self.unk_symbol)

                # validation
                if np.mod(self.update_idx, self.validation_frequency) == 0:

                    # intrinsic validation
                    valid_errs, perplexity = pred_probs(
                        self.f_log_probs, prepare_batch, self.valid_iterator)
                    valid_err = valid_errs.mean()

                    if np.isnan(valid_err):
                        logger.warning('valid_err NaN detected')
                        early_stop = True
                        break

                    # output validation info
                    logger.info('Validation error: {:1.12f} PPX: {:f}'.format(
                        valid_err, perplexity))

                    # BLEU validation
                    if self.bleu_validator and self.update_idx >= self.bleu_val_burnin:
                        bleu_score = self.bleu_validator.evaluate_model()
                        logger.info('BLEU = {}'.format(bleu_score))

                    # save the best 3 models according to early-stopping
                    if self.track_n_models > 0 and len(self.history_errs) > 0:

                        if self.early_stopping == 'cost':
                            if valid_err <= min(self.history_errs):
                                logger.info(
                                    'Saving model at epoch {} / iter {}...'.format(
                                        self.epoch_idx, self.update_idx))
                                path = os.path.join(
                                    self.output_dir,
                                    '{}.ep{}.iter{}.npz'.format(self.model_name,
                                                                self.epoch_idx,
                                                                self.update_idx))
                                with open(path, mode='wb') as f:
                                    np.savez(f, history_errs=self.history_errs,
                                             history_bleu=self.history_bleu,
                                             update_idx=self.update_idx,
                                             **unzip(self.tparams))

                                self.saved_model_paths.append(path)
                                logger.info('Done saving model')


                        # Save a model only if we've exceeding the point where
                        # we start measuring BLEU scores
                        elif self.early_stopping == 'bleu' and self.update_idx >= self.bleu_val_burnin:
                            if len(self.history_bleu) > 0 and bleu_score >= max(self.history_bleu):
                                bestbleuhandle = open('%s/bestBLEU' % self.output_dir, 'w')
                                bestbleuhandle.write("%f" % bleu_score)
                                bestbleuhandle.close()
                                logger.info(
                                    'Saving model at epoch {} / iter {}...'.format(
                                        self.epoch_idx, self.update_idx))
                                path = os.path.join(
                                    self.output_dir,
                                    '{}.ep{}.iter{}.bleu{}.npz'.format(self.model_name,
                                                                self.epoch_idx,
                                                                self.update_idx,
                                                                bleu_score))
                                with open(path, mode='wb') as f:
                                    np.savez(f, history_errs=self.history_errs,
                                             history_bleu=self.history_bleu,
                                             update_idx=self.update_idx,
                                             **unzip(self.tparams))

                                self.saved_model_paths.append(path)
                                logger.info('Done saving model')

                        # Remove un-needed saved models if necessary
                        if len(self.saved_model_paths) > self.track_n_models:
                            path = self.saved_model_paths[0]
                            logger.info('Deleting old model {}'.format(path))
                            with ignored(OSError):
                                os.remove(path)

                            self.saved_model_paths.pop(0)

                    # remember the validation result
                    self.history_errs.append(valid_err)
                    if self.early_stopping == 'bleu' and self.update_idx >= self.bleu_val_burnin:
                        # Remember the BLEU score at this point
                        self.history_bleu.append(bleu_score)

                    # reset bad counter (patience) if best validation so far
                    if self.early_stopping == 'cost':
                        if self.update_idx == 0 or valid_err <= \
                                np.array(self.history_errs).min():
                            self.best_params = unzip(self.tparams)
                            if mtl:
                                # Force the other tasks to save too
                                mtl_tasks[0].save()#string=".cost{}".format(valid_err))
                            if self.bad_counter > 0:
                                self.bad_counter -= 1
                    elif self.early_stopping == 'bleu':
                        if self.update_idx >= self.bleu_val_burnin:
                            if bleu_score >= max(self.history_bleu):
                                self.best_params = unzip(self.tparams)
                                if self.mtl:
                                    # Force the other tasks to save too
                                    self.mtl_tasks[0].save()#string=".bleu{}".format(bleu_score))
                                if self.bad_counter > 0:
                                    self.bad_counter -= 1

                    # save the best model so far (according to cost)
                    logger.info('Saving best model (according to {})'.format(self.early_stopping))
                    if self.best_params is not None:
                        self.params = self.best_params
                    else:
                        self.params = unzip(self.tparams)
                    np.savez(self.model_path,
                             history_errs=self.history_errs,
                             history_bleu=self.history_bleu,
                             update_idx=self.update_idx, **self.params)
                    logger.info('Done saving best model')

                    # check for early stop
                    if self.early_stopping == 'cost':
                        if len(self.history_errs) > self.patience and valid_err >= \
                                np.array(self.history_errs)[:-self.patience].min():
                            self.bad_counter += 1
                            logger.warn('Bad validation result. {}/{}'.format(
                                self.bad_counter, self.patience))

                            if self.bad_counter >= self.patience:
                                logger.info('Early stop activated.')
                                self.early_stop = True
                    elif self.early_stopping == 'bleu':
                        if len(self.history_bleu) > self.patience and bleu_score <= \
                                np.array(self.history_bleu)[:-self.patience].max():
                            self.bad_counter += 1
                            logger.warn('Bad validation result. {}/{}'.format(
                                self.bad_counter, self.patience))

                            if self.bad_counter >= self.patience:
                                logger.info('Early stop activated.')
                                self.early_stop = True

            # finish after this many updates
            if self.update_idx == self.finish_after:
                logger.info('Finishing after {:d} iterations!'.format(
                    self.update_idx))
                early_stop = True

            if self.early_stop:
                logger.info('Early Stop!')
                return 0

            if self.mtl:
                logger.info(self.task_stats / self.task_stats.sum())

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
              reload=False, verbose=1, disp_alignments=False, mtl=False,
              mtl_ratio=(), mtl_configs=(), mtl_decoder=False,
              n_shared_layers=1,
              **kwargs):
        """
        Train an NMT system. This function now calls train_loop() to control
        the training process. This change was made to make it easier to
        multitask a NMT model with another NMT model.
        :return:
        """

        # log options
        config = self.config

        # reload options
        if reload:
            if os.path.exists(self.config_path):
                logger.info('Reloading model options: %s' % self.config_path)
                config = load_json(self.config_path)
        else:
            logger.info('Did NOT reload model options (file did not exist)')

        # setup bleu validation
        self.bleu_validator = SimpleBleuValidator(
            self.tparams, config, self.trng, self.f_init, self.f_next, k=beam_size,
            src_dicts=self.worddicts_src, trg_idict=self.worddicts_trg_r[0],
            normalize=True, main_loop=self) if bleu_script else None

        # setup multi-task learning
        self.mtl_tasks = []
        self.shared_params = OrderedDict()
        for k in self.encoder_param_names:
            self.shared_params[k] = self.tparams[k]

        if mtl:
            logger.info('Preparing MTL tasks')
            for t in mtl_configs:
                task_config = yaml.load(open(t, mode='rb'))
                if task_config['model'] == 'imaginet':
                    task_config['exp_id'] = self.config['exp_id']
                    self.mtl_tasks.append(ImaginetTrainer(task_config, self.shared_params))
                elif task_config['model'] == 'nmt':
                    task_config['exp_id'] = self.config['exp_id']
                    self.mtl_tasks.append(Trainer(task_config, self.shared_params))
            assert sum(mtl_ratio) == 1., 'MTL ratio must sum to 1'

        # to check how many times a task was executed
        self.task_stats = np.zeros(len(self.mtl_tasks) + 1)

        # start of optimization main loop
        logger.info('Optimization started...')

        self.early_stop = False
        self.saved_model_paths = []  # history of saved models
        self.best_params = None
        self.bad_counter = 0

        # reload history
        if reload and os.path.exists(self.model_path):
            self.history_errs = list(np.load(self.model_path)['history_errs'])
            self.history_bleu = list(np.load(self.model_path)['history_bleu'])
            self.update_idx = np.load(self.model_path)['update_idx']

        self.train_loop()

        logger.info('Seen {:d} samples'.format(self.n_samples))
        logger.info('Finished with main loop')
        return 0


if __name__ == '__main__':
    pass
