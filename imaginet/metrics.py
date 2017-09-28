from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import OrderedDict

def load_similarity_data(similarity_file, w2i):
    raw_data = open(similarity_file).readlines()
    raw_data = [x.replace('\n','') for x in raw_data] # strip newlines
    similarity_data = [x.split('\t') for x in raw_data]

    # word1, word2, similarity
    matching_data = prune_pairs(similarity_data, w2i)
    return matching_data

def unzip(params):
    new_params = OrderedDict()
    for kk, vv in params.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def prune_pairs(similarity_data, w2i):
    matches = []
    for instance in similarity_data:
        if instance[0] in w2i and instance[1] in w2i:
            matches.append(instance)
    return matches

def calculate_embedding_sims(matching_sim_data, params, w2i):
    similarities = []

    embeddings = params['Wemb']
    i = 0
    for word_pair in matching_sim_data:
        x_tok = word_pair[0]
        y_tok = word_pair[1]
        x_vec = embeddings[w2i[x_tok]]
        y_vec = embeddings[w2i[y_tok]]
        sim = cosine_similarity(x_vec, y_vec)
        similarities.append(sim[0][0])
    return similarities

def simlex_correlation(theano_params, w2i, similarity_file):
    # First we need to determine which pairs are relevant for our model
    matching_data = load_similarity_data(similarity_file, w2i)

    matching_sims = []
    matching_sims = [float(x[2]) for x in matching_data]
    params = unzip(theano_params)
    embedding_sims = calculate_embedding_sims(matching_data, params, w2i)
    corr = spearmanr(embedding_sims, matching_sims)
    return corr

#    def validate(self, epoch):
#        """ Validate on dev set. """
#
#        logger.info('Imagineting the dev set')
#        start_time = time.time()
#        correlation = self.simlex_correlation()[0]
#        logger.info("SimLex correlation {}".format(correlation))
#
#        # Collect the predicted image vectors on the validation dataset
#        # We do this is batches so a dataset with a large validation split
#        # won't cause GPU OutOfMemory errors.
#        all_preds = None
#        loss = 0.0
#        for sentences in self.get_val_data_iterator():
#            x, x_mask, y = self.model.prepare_batch(sentences[0], sentences[1])
#            predictions = self.model.predict_on_batch(x, x_mask)
#            if all_preds == None:
#                all_preds = predictions
#            else:
#                all_preds = np.vstack((all_preds, predictions))
#            loss += self.validation_cost(predictions, y, self.config['margin'])
#	logger.info("Epoch {}: val cost {}".format(epoch, loss))
#
#        # Measure the ranking performance on the validation dataset
#        r1, r5, r10, medr = self.ranking(all_preds, self.val_fc7_vectors, k=self.config['ranking_k'])
#
#        # early stop the training process based on either the evaluation of median 
#        # rank or the word similarity correlation
#        if self.config['early_stopping'] == 'medr':
#            if epoch > self.config['patience']:
#                if medr > max(self.retrieval_scores[-self.config['patience']:]):
#                    # we want to stop median rank from decreasing
#                    return True
#        elif self.config['early_stopping'] == 'corr':
#            if epoch > self.config['patience']:
#                if correlation < min(self.correlations[-self.config['patience']:]):
#                    # we want to stop the word similarity correlation from increasing
#                    return True
#
#        self.correlations.append(correlation)
#        self.retrieval_scores.append(medr)
#
#        # update the best correlation and decide whether to save parameters
#        if self.best_correlation == None:
#            self.best_correlation = correlation
#        elif np.abs(correlation) > np.absolute(self.best_correlation):
#            self.best_correlation = correlation
#            if self.config['early_stopping'] == 'corr':
#                self.save()
#
#        # update the best median rank and decide whether to save parameters
#        if self.best_retrieval == None:
#            self.best_retrieval = medr
#        elif medr < self.best_retrieval:
#            self.best_retrieval = medr
#            if self.config['early_stopping'] == 'medr':
#                self.save()
#
#        validation_time = time.time() - start_time
#        logger.info('Validation took {:6f}s'.format(validation_time))
#        return False
#
#    def mse(self, predictions, fc7_path):
#        fc7_file = tables.open_file(fc7_path, mode='r')
#        fc7_vectors = fc7_file.root.feats[:]
#        images = fc7_vectors[:]
#        logger.info(predictions.shape)
#        logger.info(images.shape)
#        mse = ((predictions.flatten() - images.flatten())**2).mean()
#        logger.info("Validation MSE: {}".format(mse))
#
#    def ranking(self, predictions, fc7_vectors, k=1, npts=None):
#        """
#        :param predictions: matrix of predicted image vectors
#        :param fc7_path: path to the true image vectors
#        :param k: number of predictions per image (usually based on the number
#        of sentence encodings)
#
#        TODO: vectorise the calculation
#        """
#
#        # Normalise the predicted vectors
#        for i in range(len(predictions)):
#            predictions[i] /= np.linalg.norm(predictions[i])
#
#        images = fc7_vectors[:]
#
#        # Normalise the true vectors
#        for i in range(len(images)):
#            images[i] /= np.linalg.norm(images[i])
#
#        if npts == None:
#            npts = predictions.shape[0]
#            if npts > 25000:
#                # The COCO validation pkl contains 25,010 instances???
#                npts = 25000
#
#        ranks = np.full(len(images), 1e20)
#        for index in range(npts):
#            # Get the predicted image vector
#            p = predictions[index]
#
#            # Compute cosine similarity between predicted vector and the
#            # true vectors
#            sim = np.dot(p, images.T)
#            inds = np.argsort(sim) # should we reverse list?
#
#            # Score
#            # Iterate through the possible trues
#            target = int(math.floor(index/k))
#            tmp = np.where(inds == target)
#            #print("Index {} target {} tmp {}".format(index, target, tmp[0]))
#            tmp = tmp[0][0]
#            if tmp < ranks[target]:
#                ranks[target] = tmp
#
#        # Compute metrics
#        r1, r5, r10, medr = self.ranking_results(ranks)
#        return (r1, r5, r10, medr)
#
#    def ranking_results(self, ranks):
#        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
#        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
#        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
#        medr = np.floor(np.median(ranks)) + 1
#        logger.info('R@1 {} R@5 {} R@10 {} Median {}'.format(r1, r5, r10, medr))
#        return r1, r5, r10, medr
