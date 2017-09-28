from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import logging
import os
import numpy as np
import time
from io import open
import math
import cPickle as pkl

from model import Model
from utils import get_vocabulary, dump_params, load_params
import tables
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)

#def iterate_pkl_minibatches(path, fc7_data, w2i=None, shuffle=True, batch_size=10,
#                        lowercase=False, max_unk_ratio=1.0):
#    """
#    Yield mini-batches of sentences and fc7 feature vectors
#    :param data:
#    :param shuffle:
#    :param batch_size:
#    :param lowercase: lowercase_words words
#    :return minibatch:
#    """
#    sentences = []
#    fc7_vectors = []
#
#    fc7_file = tables.open_file(fc7_data, mode='r')
#    fc7_vectors = fc7_file.root.feats[:]
#
#    data = pkl.load(open(path, 'rb'))
#    if shuffle:
#        rs.shuffle(data)
#
#    sentences = []
#    fc7 = []
#
#    for instance in data:
#        if max_unk_ratio < 1.0:
#            # calculate the unkyness of this sentence. If it's greater than 
#            # the max_unk_ratio, bypass this example because it
#            # will have too many unknown words
#            words = instance[0].split()
#            ids = [w2i[word] if word in w2i else 1 for word in words]
#            unks = sum([x for x in ids if x == 1])
#            if float(unks)/len(words) <= max_unk_ratio:
#                sentences.append(instance[0])
#                fc7.append(fc7_vectors[instance[1]])
#            else:
#                continue
#        else:
#            sentences.append(instance[0])
#            fc7.append(fc7_vectors[instance[1]])
#
#        if len(sentences) >= batch_size:
#            yield [sentences, fc7]
#            del sentences[:]
#            del fc7[:]
#
#    # last batch
#    if len(sentences) > 0:
#        yield [sentences, fc7]
#        del sentences[:]
#        del fc7[:]
#
#def iterate_minibatches(data, fc7_data, w2i=None, shuffle=True, batch_size=10,
#                        lowercase=False, max_unk_ratio=1.0):
#    """
#    Yield mini-batches of sentences and fc7 feature vectors
#    :param data:
#    :param shuffle:
#    :param batch_size:
#    :param lowercase: lowercase_words words
#    :return minibatch:
#    """
#    sentences = []
#    fc7_vectors = []
#
#    fc7_file = tables.open_file(fc7_data, mode='r')
#    fc7_vectors = fc7_file.root.feats[:]
#
#    with open(data, mode='r', encoding='utf-8') as f:
#        data = []
#        i = 0
#        for line in f:
#            data.append((line.replace('\n',''), i))
#            i += 1
#        if shuffle:
#            rs.shuffle(data)
#
#        sentences = []
#        fc7 = []
#
#        for instance in data:
#            if max_unk_ratio < 1.0:
#                # calculate the unkyness of this sentence. If it's greater than 
#                # the max_unk_ratio, bypass this example because it
#                # will have too many unknown words
#                words = instance[0].split()
#                ids = [w2i[word] if word in w2i else 1 for word in words]
#                unks = sum([x for x in ids if x == 1])
#                if float(unks)/len(words) <= max_unk_ratio:
#                    sentences.append(instance[0])
#                    fc7.append(fc7_vectors[instance[1]])
#                else:
#                    continue
#            else:
#                sentences.append(instance[0])
#                fc7.append(fc7_vectors[instance[1]])
#
#            if len(sentences) >= batch_size:
#                yield [sentences, fc7]
#                del sentences[:]
#                del fc7[:]
#
#        # last batch
#        if len(sentences) > 0:
#            yield [sentences, fc7]
#            del sentences[:]
#            del fc7[:]

class Tester:
    def __init__(self, config, shared_theano_params=None):
        self.config = config
        if config['exp_id'] != '':
            # helps us keep the parameters in different spaces
            # instead of only in the same model_name file
            config['model_name'] = '{}-{}'.format(config['exp_id'], config['model_name'])
        self.train_path = config['train']
        self.fc7_path = config['fc7_train']
        self.val_path = config['val']
        self.val_fc7_path = config['fc7_val']
        self.model = Model(config, load=True)
        self.update_count = 0
        self.batch_size = config['batch_size']
        self.use_dropout = config['dropout']
        self.verbose = config['verbose']

    def get_val_data_iterator(self):
        """
        Returns an iterator over validation data.

        Returns: iterator
        """
        if self.val_path.endswith('pkl'):
            return iterate_pkl_minibatches(self.val_path, self.val_fc7_path, shuffle=False,
                                       batch_size=self.batch_size)
        else:
            return iterate_minibatches(self.val_path, self.val_fc7_path, shuffle=False,
                                       batch_size=self.batch_size)

    def get_predictions(self):
        # Collect the predicted image vectors on the validation dataset
        # We do this is batches so a dataset with a large validation split
        # won't cause GPU OutOfMemory errors.
        all_preds = None
        for sentences in self.get_val_data_iterator():
            x, x_mask, y = self.model.prepare_batch(sentences[0], sentences[1])
            predictions = self.model.predict_on_batch(x, x_mask)
            if all_preds == None:
                all_preds = predictions
            else:
                all_preds = np.vstack((all_preds, predictions))
        return all_preds

    def calculate_ranking(self, predictions, fc7_path, k=1, npts=None):
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

        fc7_file = tables.open_file(fc7_path, mode='r')
        fc7_vectors = fc7_file.root.feats[:]
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
        logger.info('R@1 {} R@5 {} R@10 {} Median {}'.format(r1, r5, r10, medr))
        return r1, r5, r10, medr

    def predict_images(self):
        """ 
        Predict the most likely images, given sentences 
        """

        logger.info('Imagineting the dev set')
        start_time = time.time()

        # Collect the predicted image vectors on the validation dataset
        # We do this is batches so a dataset with a large validation split
        # won't cause GPU OutOfMemory errors.
        all_preds = self.get_predictions()

        # Measure the ranking performance on the validation dataset
        r1, r5, r10, medr = self.calculate_ranking(all_preds, self.config['fc7_val'], k=self.config['ranking_k'])

    def test(self):
        self.predict_images()

def test(config):
    """
    Make predictions
    :param config: yaml-file with configuration
    :return:
    """

    # print config
    logger.info('Testing with the following (loaded) config:')
    for arg, value in config.items():
        logger.info('{}: {}'.format(arg, value))
    
    # load model
    tester = Tester(config)
    tester.test()
