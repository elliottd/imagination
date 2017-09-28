from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import time
import logging
import cPickle as pkl
import math
from theano import tensor
import numpy

from io import open

from model import Model
from utils import dump_params, dump_json
from collections import OrderedDict
from metrics import simlex_correlation


rs = np.random.RandomState(1234)  # random state for shuffling data only

logger = logging.getLogger(__name__)

import tables
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

np.random.seed(1234)  # FIXME only for testing purposes!
logger.warning('Using random seed 1234!')

def iterate_pkl_minibatches(fc7_vectors, data, w2i=None, shuffle=True, batch_size=10,
                        lowercase=False, max_unk_ratio=1.0):
    """
    Yield mini-batches of sentences and fc7 feature vectors
    :param data:
    :param shuffle:
    :param batch_size:
    :param lowercase: lowercase_words words
    :return minibatch:
    """
    if shuffle:
        rs.shuffle(data)

    sentences = []
    fc7 = []

    for instance in data:
        if max_unk_ratio < 1.0:
            # calculate the unkyness of this sentence. If it's greater than 
            # the max_unk_ratio, bypass this example because it
            # will have too many unknown words
            words = instance[0].split()
            ids = [w2i[word] if word in w2i else 1 for word in words]
            unks = sum([x for x in ids if x == 1])
            if float(unks)/len(words) <= max_unk_ratio:
                sentences.append(instance[0])
                fc7.append(fc7_vectors[instance[1]])
            else:
                continue
        else:
            sentences.append(instance[0])
            fc7.append(fc7_vectors[instance[1]])

        if len(sentences) >= batch_size:
            yield [sentences, fc7]
            del sentences[:]
            del fc7[:]

    # last batch
    if len(sentences) > 0:
        yield [sentences, fc7]
        del sentences[:]
        del fc7[:]

def iterate_minibatches(fc7_vectors, data, w2i=None, shuffle=True, batch_size=10,
                        lowercase=False, max_unk_ratio=1.0,
                        sort_by_length=False):
    """
    Yield mini-batches of sentences and fc7 feature vectors
    :param data:
    :param shuffle:
    :param batch_size:
    :param lowercase: lowercase_words words
    :return minibatch:
    """
    if shuffle:
        rs.shuffle(data)

    sentences = []
    fc7 = []

    for instance in data:
        if max_unk_ratio < 1.0:
            # calculate the unkyness of this sentence. If it's greater than 
            # the max_unk_ratio, bypass this example because it
            # will have too many unknown words
            words = instance[0].split()
            ids = [w2i[word] if word in w2i else 1 for word in words]
            unks = sum([x for x in ids if x == 1])
            if float(unks)/len(words) <= max_unk_ratio:
                sentences.append(instance[0])
                fc7.append(fc7_vectors[instance[1]])
            else:
                continue
        else:
            sentences.append(instance[0])
            fc7.append(fc7_vectors[instance[1]])

        if len(sentences) >= batch_size:
            yield [sentences, fc7]
            del sentences[:]
            del fc7[:]

    # last batch
    if len(sentences) > 0:
        yield [sentences, fc7]
        del sentences[:]
        del fc7[:]


class Trainer:
    def __init__(self, config, shared_theano_params=None):
        self.config = config
        out_path = os.path.join(
			self.config['output_dir'], self.config['model_name'])
        try:
          os.mkdir(out_path)
        except OSError:
          # Directory already exists
          pass
        if config['exp_id'] != '':
            # helps us keep the parameters in different spaces
            # instead of only in the same model_name file
            config['model_name'] = '{}-{}'.format(config['exp_id'], config['model_name'])
        self.train_path = config['train']
        self.fc7_path = config['fc7_train']
        self.val_path = config['val']
        self.val_fc7_path = config['fc7_val']
        self.model = Model(config, shared_params=shared_theano_params)
        self.update_count = 0
        self.batch_size = config['batch_size']
        self.use_dropout = config['dropout']
        self.verbose = config['verbose']
	self.patience = config['patience']
        #self.sort_by_length = config['sort_by_length']
        #self.randomise = config['randomise']

	# Open the handles to the training data
	if self.train_path.endswith('pkl'):
	  self.fc7_file = tables.open_file(self.fc7_path, mode='r')
	  self.fc7_vectors = self.fc7_file.root.feats[:]
    	  self.data = pkl.load(open(self.train_path, 'rb'))
	else:
    	  self.fc7_file = tables.open_file(self.fc7_path, mode='r')
	  self.fc7_vectors = self.fc7_file.root.feats[:]
	  self.data = self.get_text_data(self.train_path)

	# Open the handles to the validation data
	if self.val_path.endswith('pkl'):
	  self.val_fc7_file = tables.open_file(self.val_fc7_path, mode='r')
	  self.val_fc7_vectors = self.val_fc7_file.root.feats[:]
    	  self.val_data = pkl.load(open(self.val_path, 'rb'))
	else:
    	  self.val_fc7_file = tables.open_file(self.val_fc7_path, mode='r')
	  self.val_fc7_vectors = self.val_fc7_file.root.feats[:]
	  self.val_data = self.get_text_data(self.val_path)

        self.iterator = self.get_data_iterator()

        self.epoch = 0
        self.epoch_time = 0.
        self.epoch_loss = 0.

        self.val_freq = config['validation_frequency']
        self.disp_freq = config['display_frequency']
        if self.val_freq == -1:
            n_batches = sum([1 for _ in self.get_data_iterator()])
            self.val_freq = n_batches
            logger.warn('There are {} training batches'.format(n_batches))

        self.correlations = []
        self.best_correlation = None

        self.retrieval_scores = []
        self.best_retrieval = None
        self.best_r1 = np.NINF
        self.best_r5 = np.NINF
        self.best_r10 = np.NINF
	self.best_medr = np.NINF
        self.best_val_loss = np.inf
        self.best_models = [] # store the paths to the npz files
        self.n_best_models = 3
        self.save(initial=True) # save the initial parameters and config

    def get_text_data(self, file_path):
	"""
	Extracts the sentences from a text file. 

	Not to be used with a pickle of sentences.

	Returns: a list of tuples ( strings
	"""
	with open(file_path, mode='r', encoding='utf-8') as f:
	    data = []
	    i = 0
	    for line in f:
	        data.append((line.replace('\n',''), i))
	        i += 1

            return data

    def get_data_iterator(self):
        """
        Returns an iterator over training data.

        Returns: iterator
        """
        if self.train_path.endswith('pkl'):
            # Pickled training data files contain multiple references per
            # image
            return iterate_pkl_minibatches(self.fc7_vectors, self.data, w2i=self.model.w2i, shuffle=True,
                                           batch_size=self.batch_size,
                                           max_unk_ratio=self.config['max_unk_ratio'])
        else:
            return iterate_minibatches(self.fc7_vectors, self.data, w2i=self.model.w2i, shuffle=True,
                                       batch_size=self.batch_size,
                                       max_unk_ratio=self.config['max_unk_ratio'],
                                       sort_by_length=True)

    def get_val_data_iterator(self):
        """
        Returns an iterator over validation data.

        Returns: iterator
        """
        if self.val_path.endswith('pkl'):
            return iterate_pkl_minibatches(self.val_fc7_vectors, self.val_data, shuffle=False, batch_size=self.batch_size)
        else:
            return iterate_minibatches(self.val_fc7_vectors, self.val_data, shuffle=False, batch_size=self.batch_size)


    def reset_epoch_info(self):
        """
        Reset all counters used during an epoch.
        """
        self.epoch_loss = 0.
        self.epoch_time = 0.

    def train_next_batch(self):
        """
        Train on the next batch.
        """
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = self.get_data_iterator()
            self.validate(self.epoch)
            self.display_epoch_info()
            self.reset_epoch_info()
            self.epoch += 1
            batch = next(self.iterator)

        batch_loss, update_time = self.train_on_batch(
            batch)
        self.epoch_loss += batch_loss
        self.epoch_time += update_time
        self.update_count += 1
        self.display_update_info(batch_loss, update_time)

    def val_next_batch(self):
        """
        Validate on the next batch.
        """
        try:
            batch = next(self.val_iterator)
        except StopIteration:
            self.iterator = self.get_val_data_iterator()

    def train_on_batch(self, batch):
        """
        Train on a single batch online.

        :param batch:
        :return:
        """

        update_start_time = time.time()

        # prepare data
        x, x_mask, y = \
            self.model.prepare_batch(
                batch[0], batch[1],
                verbose=self.verbose)

        # train on this batch
        lr = self.config['lr']
        batch_loss = self.model.train_on_batch(
            lr, x, x_mask, y)

        update_time = time.time() - update_start_time

        return batch_loss, update_time

    def display_update_info(self, batch_loss, update_time):
        """ Display information about training. """

        if self.update_count % self.disp_freq == 0:
            logger.info(
                'Epoch {:3d} Update {:6d} Loss {:6f} Time {:6f}'.format(
                    self.epoch, self.update_count, float(batch_loss),
                    update_time))

    def display_epoch_info(self):
        """ Displays info after en epoch. """

        epoch_mean_loss = self.epoch_loss / (
            self.update_count / (self.epoch + 1))
        simlex_str = "{} {:.6f})".format("SimLex rho", self.correlations[-1])
        logger.info(simlex_str)
        logger.info(
            'Finished epoch {:d} Loss {:.1f} Avg {:.6f} Time {:d}m{:d}s'.format(
                self.epoch, self.epoch_loss, epoch_mean_loss,
                int(self.epoch_time // 60), int(self.epoch_time) % 60,
                ))

    def display_final_info(self):
        """ Displays info after en epoch. """

        validation_str = "{} R@1: {} R@5: {} R@10: {} Medr: {}".format("Best retrieval peformance:", self.best_r1, self.best_r5, self.best_r10, self.best_medr)
        logger.info('Finished training after {:d} epochs.'.format(self.epoch))
        logger.info('Best validation loss {:.3f}.'.format(self.best_val_loss))
        logger.info('Best simlex {:.3f}.'.format(self.best_correlation))
        logger.info('{}'.format(validation_str))
        out_path = os.path.join(
			self.config['output_dir'], self.config['model_name'])
	with open("{}/result.txt".format(out_path), 'a') as f:
          f.write(validation_str)

    def save(self, initial=False, string=None):
        """ Save current model to disk. """
        out_path = os.path.join(
			self.config['output_dir'], self.config['model_name'])
        logger.info('Saving model to {}'.format(out_path))
        try:
          os.mkdir(out_path)
        except OSError:
          # Directory already exists
          pass
        if initial:
            dump_json(self.config,
                    '{}/{}.json'.format(out_path, self.config['model_name']))
            self.model.save('{}/{}.npz'.format(out_path, self.config['model_name']))
        else:
            if string is None:
                # We received a special string (most likely from a different
                # task in a multitask setup. This string will tell us how to
                # write the suffix for saving this model.
                model_name = '{}.{:02f}.{}'.format(self.config['model_name'], self.best_retrieval, self.epoch + 1)
        	self.model.save('{}/{}.npz'.format(out_path, model_name))
		self.prune_saved_models('{}/{}.npz'.format(out_path, model_name))

    def prune_saved_models(self, new_model):
        if len(self.best_models) >= self.n_best_models:
          logger.info("Removing {}".format(self.best_models[0]))
          os.remove(self.best_models[0]) # remove the 'worst' saved model
          self.best_models = self.best_models[1:]
          self.best_models.append(new_model)
        else:
          self.best_models.append(new_model)

    def train(self):

        # iterate over whole data set each epoch
        for epoch in range(self.config['num_epochs']):

            self.epoch = epoch
            logger.info('Starting epoch {:d}'.format(epoch))

            for sentences in self.get_data_iterator():
                batch_loss, update_time = \
                    self.train_on_batch(sentences)
                self.epoch_time += update_time
                self.epoch_loss += batch_loss
                self.update_count += 1
                self.display_update_info(batch_loss, update_time)

                # validate on dev set
                if self.update_count % self.val_freq == 0:
                    early_stop = self.validate(epoch)

	    if early_stop:
		break
            # print epoch information
            self.display_epoch_info()

            self.reset_epoch_info()
            self.epoch += 1
        self.display_final_info()

    def validation_cost(self, output, y, margin):
	'''
	Caluclate the validation cost for a mini-batch using the model predictions
        and the known true vectors.

	:param output
	:param y
	:param margin
	'''
        U_norm = output / np.linalg.norm(output, 2,  axis=1).reshape((output.shape[0], 1))
        V_norm = y / np.linalg.norm(y, 2, axis=1).reshape((y.shape[0], 1))
        errors = np.dot(U_norm, V_norm.T)
        diag = errors.diagonal()
        # compare every diagonal score to scores in its column (all contrastive images for each sentence)
        cost_s = np.maximum(0, margin - errors + diag)
        # all contrastive sentences for each image
        cost_i = np.maximum(0, margin - errors + diag.reshape((-1, 1)))
        cost_tot = cost_s + cost_i
        # clear diagonals
        np.fill_diagonal(cost_tot, 0)
	#if self.verbose:
        #  logger.warn("Full cost matrix {}".format(cost_tot))
	# np.mean(cost_tot, axis=1) # get a vector that represents the mean loss for each example
        loss = cost_tot.mean()
	return loss

    def validate(self, epoch):
        """ Validate on dev set. """

        logger.info('Imagineting the dev set')
        start_time = time.time()
        correlation = simlex_correlation(self.model.theano_params, self.model.w2i, self.config['sim_file'])[0]
        logger.info("SimLex correlation {}".format(correlation))

        # Collect the predicted image vectors on the validation dataset
        # We do this is batches so a dataset with a large validation split
        # won't cause GPU OutOfMemory errors.
        all_preds = None
        loss = 0.0
        for sentences in self.get_val_data_iterator():
            x, x_mask, y = self.model.prepare_batch(sentences[0], sentences[1])
	    predictions = self.model.predict_on_batch(x, x_mask)
	    #print([self.model.i2w[j] for i in x.T[0] for j in i if self.model.i2w[j] != '</s>'])
	    #print(alphas.T[0])
            if all_preds == None:
                all_preds = predictions
            else:
                all_preds = np.vstack((all_preds, predictions))
            loss += self.validation_cost(predictions, y, self.config['margin'])
	logger.info("Epoch {}: val cost {}".format(epoch, loss))

        # Measure the ranking performance on the validation dataset
        r1, r5, r10, medr = self.ranking(all_preds, self.val_fc7_vectors, k=self.config['ranking_k'])
        retrieval_str = "R@1: {} R@5: {} R@10: {} Medr: {}".format(r1, r5, r10, medr)
        out_path = os.path.join(self.config['output_dir'], self.config['model_name'])
	with open("{}/result.txt".format(out_path), 'a') as f:
          f.write("Epoch {}: {} \n".format(self.epoch, retrieval_str))
	logger.info(retrieval_str)

        # early stop the training process based on either the evaluation of
        # the retrieval performance or the word similarity correlation
        if self.config['early_stopping'] == 'medr':
            if epoch > self.config['patience']:
                if r1+r5+r10 < min(self.retrieval_scores[-self.config['patience']:]):
                    # we want to stop median rank from decreasing
                    return True
        elif self.config['early_stopping'] == 'corr':
            if epoch > self.config['patience']:
                if correlation < min(self.correlations[-self.config['patience']:]):
                    # we want to stop the word similarity correlation from increasing
                    return True

        self.correlations.append(correlation)
        self.retrieval_scores.append(r1+r5+r10)

        # update the best correlation and decide whether to save parameters
        if self.best_correlation == None:
            self.best_correlation = correlation
        elif np.abs(correlation) > np.absolute(self.best_correlation):
            self.best_correlation = correlation
            if self.config['early_stopping'] == 'corr':
                self.save()

        # update the best median rank and decide whether to save parameters
        if r1 + r5 + r10 > self.best_r1 + self.best_r5 + self.best_r10:
            self.best_r1 = r1
            self.best_r5 = r5
            self.best_r10 = r10
            self.best_medr = medr
            self.best_retrieval = r1+r5+r10
            if self.config['early_stopping'] == 'medr':
                self.save()

        if self.best_val_loss > loss:
            self.best_val_loss = loss

        validation_time = time.time() - start_time
        logger.info('Validation took {:6f}s'.format(validation_time))
        return False

    def mse(self, predictions, fc7_path):
        fc7_file = tables.open_file(fc7_path, mode='r')
        fc7_vectors = fc7_file.root.feats[:]
        images = fc7_vectors[:]
        logger.info(predictions.shape)
        logger.info(images.shape)
        mse = ((predictions.flatten() - images.flatten())**2).mean()
        logger.info("Validation MSE: {}".format(mse))

    def ranking(self, predictions, fc7_vectors, k=1, npts=None):
        """
        :param predictions: matrix of predicted image vectors
        :param fc7_path: path to the true image vectors
        :param k: number of predictions per image (usually based on the number
        of sentence encodings)

        TODO: vectorise the calculation
        """

        # Normalise the predicted vectors
        for i in range(len(predictions)):
            predictions[i] /= np.linalg.norm(predictions[i])

        images = fc7_vectors[:]

        # Normalise the true vectors
        for i in range(len(images)):
            images[i] /= np.linalg.norm(images[i])

        if npts == None:
            npts = predictions.shape[0]
            if npts > 25000:
                # The COCO validation pkl contains 25,010 instances???
                npts = 25000

        ranks = np.full(len(images), 1e20)
        for index in range(npts):
            # Get the predicted image vector
            p = predictions[index]

            # Compute cosine similarity between predicted vector and the
            # true vectors
            sim = np.dot(p, images.T)
            inds = np.argsort(sim) # should we reverse list?

            # Score
            # Iterate through the possible trues
            target = int(math.floor(index/k))
            tmp = np.where(inds == target)
            #print("Index {} target {} tmp {}".format(index, target, tmp[0]))
            tmp = tmp[0][0]
            if tmp < ranks[target]:
                ranks[target] = tmp

        # Compute metrics
        r1, r5, r10, medr = self.ranking_results(ranks)
        return (r1, r5, r10, medr)

    def ranking_results(self, ranks):
        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        return r1, r5, r10, medr


def train(config=None):
    # dump config for re-use
    #dump_params(config['output_dir'], config['config_name'], config)

    # print config
    logger.info('Training with the following config:')
    for arg, value in config.items():
        logger.info('{}: {}'.format(arg, value))

    # train
    trainer = Trainer(config)
    trainer.train()
