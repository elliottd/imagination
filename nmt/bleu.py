from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import logging
import numpy as np
import os
import re
import time
import subprocess
from builtins import bytes
from io import open
import time

from nmt.sampler import generate_sample
from nmt.model import prepare_batch
from nmt.data_iterator_mem import TextIterator
from nmt.utils import idx_to_word
from nmt.translate import Translator

logger = logging.getLogger(__name__)


class SimpleBleuValidator:
    """
    Translates a data set and obtains the BLEU score.
    Loosely based on the Blocks Examples BLEU validator.
    """

    def __init__(self, tparams, config, trng, f_init, f_next, k=12,
                 src_dicts=None, trg_idict=None, normalize=True,
                 main_loop=None):
        self.config = config
        self.main_loop = main_loop
        # self.use_noise = use_noise

        n_words_src = config['n_words_src']
        n_words_trg = config['n_words_trg']

        self.translator = Translator(fs_init=[f_init], fs_next=[f_next],
                                     config=config,
                                     src_dicts=src_dicts, trg_idict=trg_idict,
                                     n_words_src=n_words_src,
                                     n_words_trg=n_words_trg,
                                     normalize=normalize, trng=trng, k=k)

        self.bleu_val_burnin = config['bleu_val_burnin']
        self.bleu_val_out = config[
            'bleu_val_out']  # prefix for translation output
        self.bleu_script = config['bleu_script']
        self.postprocess_script = config['postprocess_script']
        self.model_name = config['model_name']
        self.output_dir = config['output_dir']  # where we save the translations

        self.multibleu_cmd = [self.bleu_script, config['bleu_val_ref']]
        self.postprocess_cmd = [self.postprocess_script]

    def bleu_score(self):
        out_file = os.path.join(self.output_dir,
                '{}.ep{}.iter{}.txt'.format(self.bleu_val_out,
                    self.main_loop.epoch_idx, self.main_loop.update_idx))

        subprocess.check_call(
                    ['perl %s -lc %s < %s > %s/BLEU'
                                     % (self.bleu_script,
                                     self.config['bleu_val_ref'],
                                     out_file, self.output_dir)],
                                shell=True)
        # BLEU = %f, B1p/B2p/B3p/B4p (BP=%f, ratio=%f, hyp_len=%d, ref_len=%d)
        bleudata = open("%s/BLEU" % (self.output_dir)).readline()
        logger.info(bleudata)
        data = bleudata.split(",")[0]
        bleu4 = data.split("=")[1]
        bleu4 = float(bleu4.lstrip())
        return bleu4

    def evaluate_model(self):
        """
        Translate the data set and get a BLEU score
        :return:
        """

        logger.info('Started BLEU validation')
        start_time = time.time()

        epochs = self.main_loop.epoch_idx
        iterations = self.main_loop.update_idx
        f_path = os.path.join(self.output_dir,
                              '{}.ep{}.iter{}.txt'.format(self.bleu_val_out,
                                                          epochs, iterations))

        # self.use_noise.set_value(0.)  # do not apply dropout

        #p_post = subprocess.Popen(self.postprocess_cmd, stdin=subprocess.PIPE,
        #                          stdout=subprocess.PIPE)

        translations = []
        for translation in self.translator.translate(
                    input=self.config['src_valid'], output=f_path,
                    maxlen=self.config['maxlen'], k=self.config['beam_size'],
                    factors=self.config['factors']):
                translations.append(translation)

        h = open(f_path, mode='w', encoding='utf-8')
        for t in translations:
            h.write('{}\n'.format(t))
        h.close()

        #with open(f_path, mode='w', encoding='utf-8') as f:
        #    for translation in self.translator.translate(
        #            input=self.config['src_valid'], output=f_path,
        #            maxlen=self.config['maxlen'], k=self.config['beam_size'],
        #            factors=self.config['factors']):
        #        f.write(translation)
        #        f.write('\n')
        #        f.flush()

        #        p_post.stdin.write(bytes(translation, 'utf-8'))
        #        p_post.stdin.write('\n')
        #        p_post.stdin.flush()

        #p_post.stdin.close()

        if self.config['at_replace']:
            # The output contains @@ or @-@ symbols, which we want to strip
            subprocess.check_call(["sed -i -r 's/ @(.*?)@ //g' {}".format(f_path)], shell=True)

        if self.config['subword_at_replace']:
            # The output contains '@@ ' symbols, which we want to strip
            subprocess.check_call(["sed -i -r 's/@@ //g' {}".format(f_path)], shell=True)

        if self.config['sp_replace']:
            subprocess.check_call(['{} --model {} < {} > {}.decoded'.format(self.config['sp_path'], self.config['sp_model'], f_path, f_path)], shell=True)
            subprocess.check_call(['cp {}.decoded {}'.format(f_path, f_path)], shell=True)

        bleu_score = self.bleu_score()

        logger.info('Validation Time: {}m Epoch: {} Iter: {} BLEU: {}'.format(
            float(time.time() - start_time) / 60., epochs, iterations,
            bleu_score))

        #p_post.terminate()

        return bleu_score
