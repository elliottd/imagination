from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from six import itervalues, iteritems

import numpy as np
import theano
from theano import tensor
import logging
import os
import pdb
import json

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict
from io import open
from collections import deque

from layers import get_layer, shared_dropout_layer, inv_dropout_mask
from utils import norm_weight, get_vocabulary, concatenate, \
    embedding_name, dump_json, load_vocabularies, \
    uniform_weight, uniform_glorot
from optimizers import *
from theano.tensor.extra_ops import fill_diagonal


logger = logging.getLogger(__name__)
floatX = theano.config.floatX


class Model:
    """
    Predict the FC7 feature vector of an image from the RNN sentence
    representation, aka Imaginet.
    """

    def __init__(self, config, load=False, shared_params=None):
        self.config = config

        # vocabulary
        self.w2i, self.i2w = load_vocabularies(config)
        theano.config.compute_test_value = config['compute_test_values']  # warn to enable

        assert config['eos_symbol'] in self.w2i, \
            'word vocabulary needs to include eos'
        assert config['unk_symbol'] in self.w2i, \
            'word vocabulary needs to include UNK'

        # save vocabularies
        output_dir = config['output_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.warn('Vocabularies: {}'.format(len(self.w2i)))

        # hyper-parameters
        self.dim = config['dim']
        self.dim_emb = config['dim_emb']
        #self.dim_emb_image = config['dim_emb_image']
        self.dim_per_factor = config['dim_per_factor']
        self.dim_v = config['dim_v']
        self.dropout = config['dropout']
        self.dropout_word = config['dropout_word']
        self.dropout_emb = config['dropout_emb']
        self.dropout_rec = config['dropout_rec']
        self.verbose = config['verbose']
        gain = 'relu' if config['activation_mlp'] == 'relu' else 1.0

        if self.config['max_words'] == -1:
            self.voc_size = len(self.w2i)
        else:
            self.voc_size = self.config['max_words']

        logger.warn('Using actual vocsize : {}'.format(self.voc_size))

        # self.params is a dictionary that willi hold all the parameters in 
        # the strict order defined in this __init__()
        self.params = OrderedDict()
        self.theano_params = OrderedDict()

        # build the bi-rnn encoder
        # N.B. params are added inside this method
        self.init_encoder_params(**config)
        if config['mode'] == 'imaginet':
            # build the MLP for image prediction
            self.params = Model.init_mlp_params(self.params,
                                          gain=gain,
                                          **config)

        if load:
            self.load(os.path.join(config['output_dir'], config['model_name']))

        self.init_theano_params()

        # multi-task support
        # we replace whatever parameters we already have at this point with
        # the ones that we received as optional input
        # this needs to be done BEFORE building the model
        if shared_params is not None:
            self.apply_shared_theano_params(shared_params)

        # compile theano functions for training the model
        trng, f_loss, f_grad_shared, f_update, raw_grads = \
            self.compile_training_functions(config)
        self.trng = trng
        self.f_loss = f_loss
        self.f_grad_shared = f_grad_shared
        self.f_update = f_update
        self.raw_grads = raw_grads

        # compile theano functions for evaluating the model
        f_encode, f_predict = self.compile_mlp_predict(
            config, trng=trng)

        self.f_encode = f_encode
        self.f_predict = f_predict

    '''
    Compile the encoder and MLP, define the theano loss function, compute the
    gradient, and perform the parameter updates.
    '''

    def compile_training_functions(self, config):
        """
        Builds all theano functions for training.

        :param config:
        :return:
        """

        trng = RandomStreams(1234)

        # build encoder for training
        x, x_mask, enc_states, mlp_input = self.build_encoder(trng=trng,
                                                   use_noise=True,
                                                   **config)

        if config['mode'] == 'imaginet':
            # build MLP for training
            y, loss, outputs = Model.build_mlp(self.theano_params, trng,
                                               mlp_input, use_noise=True, **config)

        # compile loss function
        logger.info('Computing image vector (MLP) loss function...')
        # our inputs are the tokens, the mask, and the true image vector
        inputs = [x, x_mask, y]
        f_loss = theano.function(inputs, [loss])
        logger.info('Done')

        # apply L2 regularization on weights
        if config['decay_c'] > 0.:
            decay_c = theano.shared(np.float32(config['decay_c']), name='decay_c')
            weight_decay = 0.
            for kk, vv in self.theano_params.iteritems():
                weight_decay += (vv ** 2).sum()
            weight_decay *= decay_c
            loss += weight_decay

        # compute gradient
        logger.info('Computing gradient...')
        grads = tensor.grad(loss, wrt=list(itervalues(self.theano_params)))
        logger.info('Done')

        # apply gradient clipping
        clip_c = config['clip_c']
        if clip_c > 0.:
            g2 = 0.
            for g in grads:
                g2 += (g ** 2).sum()
            new_grads = []
            for g in grads:
                new_grads.append(tensor.switch(g2 > (clip_c ** 2),
                                               g / tensor.sqrt(g2) * clip_c, g))
            grads = new_grads

        # if you want to get the raw gradients for visual inspection
        raw_grads = theano.function([x, x_mask, y], grads)

        # compile the optimizer (with the above loss and gradient)
        logger.info('Building optimizers...')
        lr = tensor.scalar(name='lr', dtype=floatX)
        lr.tag.test_value = 0.001
        f_grad_shared, f_update = eval(config['optimizer'])(
            lr, self.theano_params, grads, inputs, loss)
        logger.info('Done')

        logger.info("Total model parameters {}".format(sum([np.prod(zz.shape) for zz in self.params.values()])))

        # random number generator, loss function, parameter updates
        return trng, f_loss, f_grad_shared, f_update, raw_grads

    '''
    Initialise, build, and compile the Bidirectional Encoder 
    '''
    def init_encoder_params(self, dim=0, dim_emb=0, factors=1,
                            dim_per_factor=None, encoder_layers=0,
                            encoder='gru', **kwargs):
        """
        Initialize all encoder parameter, adding them to self.params.
        MUST match nmt.model.init_encoder_params for multitasking

        :param dim:
        :param dim_emb:
        :param factors:
        :param dim_per_factor:
        :param encoder_layers:
        :param encoder: lstm or gru
        :param kwargs:
        :return:
        """

        # embedding
        for factor in range(factors):
            emb_init = uniform_weight
            logger.warning('Word embeddings init: {}'.format(emb_init))
            e = emb_init(self.voc_size, dim_per_factor[factor])
            self.params[embedding_name(factor)] = e
    	    logger.info("Embedding layer parameters: {}".format(
                np.prod(e.shape)))

        init = kwargs['init']

        # forward and backward RNNs, first layer
        self.params = get_layer(encoder)[0](
            self.params, prefix='enc_fw_0', nin=dim_emb, dim=dim, init=init)
        self.params = get_layer(encoder)[0](
            self.params, prefix='enc_bw_0', nin=dim_emb, dim=dim, init=init)

        # any additional layers
        for i in range(1, encoder_layers):
            self.params = get_layer(encoder)[0](
                self.params, prefix='enc_fw_{}'.format(i), nin=2 * dim,
                dim=dim, init=init)
            self.params = get_layer(encoder)[0](
                self.params, prefix='enc_bw_{}'.format(i), nin=2 * dim,
                dim=dim, init=init)

    def build_encoder(self, dim_emb=0, factors=1, dim_per_factor=(),
                      encoder_layers=1, encoder='gru',
                      dropout=False, dropout_word=0., dropout_emb=0.,
                      dropout_rec=0.,
                      trng=None, use_noise=True, **kwargs):
        """
        Build the bi-directional encoder (bi-gru / bi-lstm)
        MUST match nmt.model.build_encoder for multitasking

        :param dim_emb:
        :param factors:
        :param dim_per_factor:
        :param encoder_layers:
        :param encoder: gru or lstm
        :param dropout:
        :param dropout_word:
        :param dropout_emb:
        :param dropout_rec:
        :param trng:
        :param use_noise: apply dropout (use_noise) or not (test)
        :param kwargs:
        :return:
        """
        logger.warn('Building encoder - use_noise: {}'.format(use_noise))
        assert sum(dim_per_factor) == dim_emb, 'sum dim_per_factor != dim_emb'

        dropout = dropout and use_noise  # no dropout during test time

        # input to forward rnn (#factors x #words x #batch_size)
        x = tensor.tensor3('x_word', dtype='int64')
        x.tag.test_value = np.random.randint(1000, size=(1, 3, 2))

        # input to the masking function (#words x #batch_size)
        # mask is set to 0 when we are padding the input
        # see prepare_batch() for more details
        x_mask = tensor.matrix('x_mask', dtype=floatX)
        x_mask.tag.test_value = np.ones([3, 2]).astype('float32')
        x_mask.tag.test_value[2,0] = 0
        x_mask.tag.test_value[1,1] = 0
        x_mask.tag.test_value[2,1] = 0
        if kwargs['verbose'] and kwargs['compute_test_values'] == 'warn':
            logger.warn(x.tag.test_value)
            logger.warn(x_mask.tag.test_value)

        # input to backward rnn (x and x_mask reversed)
        x_bw = x[:, ::-1]
        x_mask_bw = x_mask[::-1]
        if kwargs['verbose']:
            logger.warn(x_bw.tag.test_value)
            logger.warn(x_mask_bw.tag.test_value)

        n_timesteps = x.shape[1]  # length of longest sentence in batch
        batch_size = x.shape[2]  # size of this batch (can vary!)

        # forward RNN
        # first build the forward embeddings
        # we do the concatenate() in the case that we have multiple factors
        # for the input data. Not currently used in this code but inherited
        # from nematus.
        emb_fw = []
        for factor in range(factors):
            emb_fw.append(
                self.theano_params[embedding_name(factor)][x[factor].flatten()])
        emb_fw = concatenate(emb_fw, axis=1)
        emb_fw = emb_fw.reshape([n_timesteps, batch_size, dim_emb])

        # drop out whole words by zero-ing their embeddings
        if dropout and dropout_word > 0.:
            logger.warn('Using word dropout (p={})'.format(dropout_word))
            p = 1 - dropout_word
            word_drop = inv_dropout_mask((n_timesteps, batch_size, 1), trng, p)
            word_drop = tensor.tile(word_drop, (1, 1, dim_emb))
            emb_fw *= word_drop

        # now build the forward rnn layer
        fw_layers = [get_layer(encoder)[1](
            self.theano_params, emb_fw, trng=trng, prefix='enc_fw_0',
            mask=x_mask, dropout=dropout, dropout_W=dropout_emb,
            dropout_U=dropout_rec)]

        # backward rnn
        # first build the backward embeddings
        # Same deal with the concatenate() of the factors as emb_fw.
        emb_bw = []
        for factor in range(factors):
            emb_bw.append(self.theano_params[embedding_name(factor)][
                              x_bw[factor].flatten()])
        emb_bw = concatenate(emb_bw, axis=1)
        emb_bw = emb_bw.reshape([n_timesteps, batch_size, dim_emb])

        # drop out the same words as above in forward rnn
        if dropout and dropout_word > 0.:
            logger.warn('Also dropping bw words (p={})'.format(dropout_word))
            emb_bw *= word_drop[::-1]

        # now build the backward rnn layer
        bw_layers = [get_layer(encoder)[1](
            self.theano_params, emb_bw, trng=trng, prefix='enc_bw_0',
            mask=x_mask_bw, dropout=dropout, dropout_W=dropout_emb,
            dropout_U=dropout_rec)]

        # add additional layers if Deep Encoder is specified
        for i in range(1, encoder_layers):  # add additional layers if specified
            input_states = concatenate(
                (fw_layers[i - 1][0], bw_layers[i - 1][0][::-1]), axis=2)

            fw_layers.append(
                get_layer(encoder)[1](self.theano_params, input_states,
                                      trng=trng, prefix='enc_fw_{}'.format(i),
                                      mask=x_mask, dropout=dropout,
                                      dropout_W=dropout_emb,
                                      dropout_U=dropout_rec))

            bw_layers.append(
                get_layer(encoder)[1](self.theano_params, input_states[::-1],
                                      trng=trng, prefix='enc_bw_{}'.format(i),
                                      mask=x_mask_bw, dropout=dropout,
                                      dropout_W=dropout_emb,
                                      dropout_U=dropout_rec))

        # combine final layers of forward and backward RNNs
        # this is a concatenated vector, which means it has dim=dim*2
        # [::-1] means reverse the bw_layers
        states = concatenate((fw_layers[-1][0], bw_layers[-1][0][::-1]), axis=2)
        if kwargs['verbose']:
            logger.info("encoder_states shape {}".format(states.tag.test_value.shape))

        if kwargs['mean_birnn']:
            # calculate a 2D vector over time but summing over axis 0
            mlp_input = (states * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]
        elif kwargs['final_birnn']:
            mlp_input = concatenate((fw_layers[-1][-1][-1,:,:],
                bw_layers[-1][-1][-1,:,:]), axis=fw_layers[-1][-1].ndim-2)
        if kwargs['verbose']:
            logger.info("FW LAYERS shape {}".format(fw_layers[-1][0][-1,:,:].tag.test_value.shape))
            logger.info(fw_layers[-1][0][-1,:,:].tag.test_value)
            logger.info("BW LAYERS shape {}".format(bw_layers[-1][0][-1,:,:].tag.test_value.shape))
            logger.info(bw_layers[-1][0][-1,:,:].tag.test_value)
            logger.info("CONCAT shape {}".format(mlp_input.tag.test_value.shape))
            logger.info(mlp_input.tag.test_value)

        # Why don't we return the reverse mask?
        return x, x_mask, states, mlp_input

    '''
    Initialise, build, and compile the embedding matching model
    '''

    @staticmethod
    def init_mlp_params(params, dim=1, dim_v=4096,
                        init='glorot', gain=1.,
                        **kwargs):
        """
        Initialize parameters for the scoring function, an MLP.

        :param params:
        :param dim:
        :param init: glorot initialization or uniform or normal
        :param gain: parameter for glorot initializer
        :param kwargs:
        :return:
        """
        logger.warn('MLP - init: {} gain: {}'.format(init, gain))

        dim_in = 2 * dim # we concatenate the forward and backward RNN
        dim_out = dim_v

        # The first layer goes from the bi-directional concatenation
        # to the visual feature vector dimensionality. e.g. 2 x 1000 -> 4,096
        params = get_layer('ff')[0](params, prefix='mlp', nin=dim_in,
                                    nout=dim_out, ortho=False,
                                    init=init, gain=gain)

        return params

    @staticmethod
    def build_mlp(theano_params, trng, enc_states,
                  activation_mlp='relu', dropout=False, dropout_hid=0.,
                  use_noise=False, **kwargs):
        """
        Builds an MLP scoring function for use during training.

        We are trying to predict a single FC7 vector of dimensionality 4096.

        The cost function is MSE between FC7_true and FC7_predicted.

        :param trng:
        :param enc_states:
        :param activation_mlp:
        :param dropout:
        :param dropout_hid:
        :param use_noise:
        :param kwargs:
        :return:

        """
        # apply dropout on encoder states
        if dropout and dropout_hid > 0. and use_noise:
            logger.warn('Applying dropout mask on bi-states')
            mask = inv_dropout_mask(enc_states.shape, trng, 1 - dropout_hid)
            enc_states *= mask
        else:
            logger.warn('No dropout on Encoder output')

        # set MLP activation function
        assert activation_mlp in ('relu', 'tanh'), \
            'MLP activation function must be tanh or relu'

        activation_mlp = 'lambda x: tensor.nnet.relu(x)' \
            if activation_mlp == 'relu' else 'lambda x: tensor.tanh(x)'
        logger.info('Using MLP activation function: {}'.format(activation_mlp))

        # The input to the MLP will be the mean value of the hidden states for
        # each instance in the minibatch.
        if kwargs['verbose']:
            logger.warn(enc_states.tag.test_value)

        # targets for the MLP -- dim: batch, fc_7 vector
        y = tensor.matrix('y', dtype='float64')
        # take the RELU over the visual features
        #y = eval('lambda x: tensor.nnet.relu(x)')(y)
        y.tag.test_value = np.ones((2,4096))
        if kwargs['verbose']:
            logger.warn(y.tag.test_value)

        # train a single layer MLP to do everything
        output = get_layer('ff')[1](
            theano_params, enc_states, prefix='mlp', activ=activation_mlp)

        if kwargs['verbose']:
            logger.warn("MLP output {}".format(output.tag.test_value))

        if kwargs['loss'] == 'mse':
            loss = ((output.flatten() - y.flatten())**2).mean()
        elif kwargs['loss'] == 'constrastive':
            margin = kwargs['margin']
            U_norm = output / output.norm(2,  axis=1).reshape((output.shape[0], 1))
            V_norm = y / y.norm(2, axis=1).reshape((y.shape[0], 1))
            errors = tensor.dot(U_norm, V_norm.T)
            diag = errors.diagonal()
            # compare every diagonal score to scores in its column (all contrastive images for each sentence)
            cost_s = tensor.maximum(0, margin - errors + diag)
            # all contrastive sentences for each image
            cost_i = tensor.maximum(0, margin - errors + diag.reshape((-1, 1)))
            cost_tot = cost_s + cost_i
            # clear diagonals
            cost_tot = fill_diagonal(cost_tot, 0)
	    if kwargs['verbose']:
              logger.warn("Full cost matrix {}".format(cost_tot.tag.test_value))
            loss = cost_tot.mean()
        elif kwargs['loss'] == 'dot':
            margin = kwargs['margin']
            errors = tensor.dot(output, y.T)
            diag = errors.diagonal()
            # compare every diagonal score to scores in its column (all contrastive images for each sentence)
            cost_s = tensor.maximum(0, margin - errors + diag)
            # all contrastive sentences for each image
            cost_i = tensor.maximum(0, margin - errors + diag.reshape((-1, 1)))
            cost_tot = cost_s + cost_i
            # clear diagonals
            cost_tot = fill_diagonal(cost_tot, 0)
	    if kwargs['verbose']:
              logger.warn("Full cost matrix {}".format(cost_tot.tag.test_value))
            loss = cost_tot.mean()

        if kwargs['verbose']:
            logger.warn("Batch loss {}".format(loss.tag.test_value))

        return y, loss, output

    def compile_mlp_predict(self, config, padding=None, trng=None):
        """
        Compile functions for doing predictions with an MLP

        :param config:
        :param padding:
        :param trng:
        :return f_encode, f_predict:

        """
        # build encoder for test / prediction
        x, x_mask, enc_states, mlp_input = self.build_encoder(
            trng=trng, use_noise=False, **config)

        # compile prediction function (given encoder states)
        logger.info('Compiling prediction function(s)...')
	if config['mode'] == 'imaginet':
	        outputs = self.build_mlp_predict(mlp_input, **config)  # MLP for test

        	f_predict = theano.function([mlp_input], outputs)
        logger.info('Done')

        # compile bi-directional encoder function
        logger.info('Compiling encoder function...')
        f_encoder = theano.function([x, x_mask], [enc_states, mlp_input])
        logger.info('Done')

        return f_encoder, f_predict

    """
    Functions for predicting the image vector of unseen data, e.g. the
    validation or test data. Only sees the sentence data as input.
    """

    def build_mlp_predict(self, enc_states,
                          dim_emb=0, dim=0,
                          activation_mlp='relu',
                          **kwargs):
        """
        Builds an MLP scoring function for use during prediction / test time.

        We want this to predict a single 4096d vector for each input.

        :param dim_emb:
        :param dim:
        :param activation_mlp:
        :param kwargs:
        :return:

        TODO: Redefine the predict function so it predicts the embedding of
        the sentence and the image embedding.
        """

        # set MLP activation function
        assert activation_mlp in ('relu', 'tanh'), \
            'MLP activation function must be tanh or relu'
        activation_mlp = 'lambda x: tensor.nnet.relu(x)' \
            if activation_mlp == 'relu' else 'lambda x: tensor.tanh(x)'
        logger.warn('Using MLP activation function: {}'.format(activation_mlp))

        theano_params = self.theano_params
        # The input to the MLP will be the mean value of the hidden states for
        # each instance in the minibatch.
        if kwargs['verbose']:
            logger.warn(states_mean.tag.test_value)

        # train a single layer MLP to do everything
        output = get_layer('ff')[1](
            theano_params, enc_states, prefix='mlp', activ=activation_mlp)

        return output

    def predict_on_batch(self, x, x_mask, verbose=False):
        """
        Predict image vectors given the sentence inputs

        :param sentences:
        :param fc7_vectors:
        :param verbose:
        :return:
        """

        # Get the birnn encoding states for the input sequence
        enc_states, mlp_states = self.f_encode(x, x_mask)
	if self.config['mode'] == 'imaginet':
	        prediction = self.f_predict(mlp_states)
        	return prediction

    '''
    Prepare a batch of data for training / prediction
    '''

    def prepare_batch(self, sentences, fc7_vecs,
                      verbose=False):
        """
        Turn sentences into batch input format.

        :param sentences: the batch of sentences
        :param fc7_vecs: the batch of image vectors
        :param verbose: display detailed info on data
        :return:
        """

        seqs_x = []
        seqs_y = []

        for i, sentence in enumerate(sentences):
            words = sentence.split()
            word_id_seq = [self.w2i[word] if word in self.w2i else 1 for word in words]
            #if verbose:
            #    print(sentence)
            #    print(words)
            #    print(word_id_seq)

            seqs_x.append(word_id_seq)
            seqs_y.append(fc7_vecs[i])

        # compute lengths to determine batch shape
        lengths_x = [len(s) for s in seqs_x]
        maxlen_x = np.max(lengths_x)
        batch_size = len(seqs_x)
        lengths_y = [len(d) for d in seqs_y]

        factors = 1
        x = np.zeros((factors, maxlen_x, batch_size)).astype('int64')
        x_mask = np.zeros_like(x[0]).astype(floatX)

        # create the sentence batch for encoder
        for i, words in enumerate(seqs_x):
            x[0, :lengths_x[i], i] = words
            x_mask[:lengths_x[i], i] = 1.

        # the image batch is a matrix of 4096-d image vectors
        y = np.zeros((batch_size, self.dim_v)).astype('float64')
        for i, targets in enumerate(seqs_y):
            y[i,:] = targets

        return x, x_mask, y

    def train_on_batch(self, lr, x, x_mask, y):
        """
        Train imaginet on a batch of sentences.

        :param lr: learning rate
        :param x:
        :param x_mask:
        :param y:
        :return:
        """

        # get loss
        ret_vals = self.f_grad_shared(x, x_mask, y)
        loss = ret_vals[0]

        #if self.verbose:
        #    # Show the mean values of the gradients
        #    for idx, x in enumerate(self.raw_grads(x, x_mask, y)):
        #        if idx == 0:
        #            print("Gradient {}, count {}, mean value: {}".format(idx,
        #                np.array(x).shape, self.grad_norm(x)))
         ##       else:
         #           print("Gradient {}, count {}, mean value: {}".format(idx,
         #               np.array(x).shape, self.grad_norm(x)))

        # do the update on parameters
        self.f_update(lr)

        return loss

    '''
    Auxiliary functions used throughout this class.
    '''

    def init_theano_params(self):
        """
        Initialize Theano shared variables according to the initial parameters

        Auxiliary function.

        :return:
        """
        for kk, pp in iteritems(self.params):
            logger.info(
                'Parameter: {} Shape: {}'.format(kk, self.params[kk].shape))
            self.theano_params[kk] = theano.shared(self.params[kk], name=kk)

    def apply_shared_theano_params(self, shared_theano_params):
        """
        Override the parameters of the model with the provided 
        shared parameters. Used for Multi-task Learning.

        Auxiliary function.

        Note that we do not need to override all of them, just the ones
        in the provided dictionary.

        :param shared_theano_params:
        :return:
        """
        for k in shared_theano_params:
            self.theano_params[k] = shared_theano_params[k]
            assert self.params[k].shape == self.theano_params[
                k].get_value().shape, 'shape mismatch'
            self.params[k] = shared_theano_params[k].get_value()
            logger.info(
                'Using external theano parameter: {} Shape: {}'.format(
                    k, self.params[k].shape))

    def save(self, path):
        """
        Save current model parameters to disk
        :param path:
        :return:
       """
        params = OrderedDict()

        for kk, vv in self.theano_params.items():
            params[kk] = vv.get_value()

        with open(path, mode='wb') as f:
            np.savez(f, **params)

    def load(self, path):
        """
        Load saved parameters
        :param path:
        :return:
        """
        pp = np.load(path)
        for kk, vv in self.params.items():
            if kk not in pp:
                logger.warn('%s is not in the archive' % kk)
                continue
            self.params[kk] = pp[kk]

    def grad_norm(self, array):
        """
        Calculate the norm of the gradients. Can be useful for debugging.
        :param array:
        """
	return tensor.norm(array, 2)

