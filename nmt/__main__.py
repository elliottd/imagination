import argparse
import logging
import yaml

from nmt.newtrain import Trainer
from nmt.translate import Translator

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

"""
TODO: Remove the duplicate functionality of parse_yaml_args and
handle_cli_args
"""

def main():
    """
    Train a Neural Machine Translation system with Attention or
    sample from it.
    """
    config = handle_args()
    print(config)

    if config['mode'] == 'train':
        logger.info('Train mode')
        trainer = Trainer(config, shared_theano_params=None, **config)
        trainer.train(**config)
        logger.warn('Exiting train mode')
    else:
        logger.info('Test mode')
        translator = Translator(**config)
        translator.load_from_disk(config['models'], config['configs'],
                config['src_dicts'], config['trg_dict'])
        translator.translate_and_save(**config)

def handle_args():
    try:
        args = parse_yaml_args()
        print(args)
        config = yaml.load(open(args.config, mode='rb'))
        config['exp_id'] = args.exp_id
        config['mode'] = args.mode
        print(config)
        return config
    except:
        pass

    try:
        config = handle_cli_args()
        return config
    except:
        pass

def parse_yaml_args():
    """
    Arguments parser.
    :return: args object with parsed arguments
    """
    ap = argparse.ArgumentParser('Train or test NMT model.')
    ap.add_argument('mode', choices=['train', 'test'], default='train',
                    help='mode: train or test')
    ap.add_argument('config', metavar='CONFIG', type=str,
                    default='config.yaml', help='configuration to use')
    ap.add_argument('--exp_id', type=int, required=True)
    return ap.parse_args()

def handle_cli_args():
    parser = argparse.ArgumentParser(description='Train or test NMT model.')

    subparsers = parser.add_subparsers(dest='mode')
    trp = subparsers.add_parser('train')
    tsp = subparsers.add_parser('test')

    # mandatory data parameters
    trp.add_argument('model_name', metavar='model-name', type=str,
                     help='the name of the trained model')
    trp.add_argument("--exp_id", type=str, default='',
                     help="Do you want to assign a unique experiment ID that \
                     will be prepended to the model name?")
    trp.add_argument('output_dir', metavar='output-dir', type=str,
                     help='where to save output files')
    trp.add_argument('src_train', metavar='src-train', type=str,
                     help='source language train file')
    trp.add_argument('trg_train', metavar='trg-train', type=str,
                     help='target language train file')
    trp.add_argument('src_valid', metavar='src-valid', type=str,
                     help='source language validation file')
    trp.add_argument('trg_valid', metavar='trg-valid', type=str,
                     help='target language validation file')
    trp.add_argument('--src-dicts', nargs='+', type=str, required=True,
                     help='source vocabularies, 1 per factor')
    trp.add_argument('--trg-dicts', nargs='+', type=str, required=True,
                     help='target vocabularies, 1 per factor')
    trp.add_argument('--factors', metavar='N', type=int,
                     help='number of factors in input', default=1)
    trp.add_argument('--factors-trg', metavar='N', type=int,
                     help='number of factors in output', default=1)

    trp.add_argument('--n-words-src', type=int,
                     help='max source vocabulary size', default=50000)
    trp.add_argument('--n-words-trg', type=int,
                     help='max target vocabulary size', default=50000)

    # optional hidden layer dimension parameters
    trp.add_argument('--dim-emb', type=int, help='word embedding dimension',
                     default=620)
    trp.add_argument('--dim-per-factor', type=int, nargs='+',
                     help='dimension for each factor -- this should sum to dim-emb',
                     default=[])
    trp.add_argument('--dim', metavar='N', type=int,
                     help='hidden state dimension', default=1000)
    trp.add_argument('--dim-att', type=int, help='attention dimension',
                     default=2000)

    # hmm jumps
    trp.add_argument('--jump', action='store_true',
                     help='use HMM-like soft-alignment')
    trp.add_argument('--jump-init-path', type=str,
                     help='init jumps from external aligner', default=None)
    trp.add_argument('--max-jump', type=int,
                     help='the maximum jump distance for which a weight is defined. '
                          'larger jumps are bucketed in this max jump distance',
                     default=50)
    trp.add_argument('--save-jumps', action='store_true',
                     help='save jump parameters')

    # encoder and decoder
    trp.add_argument('--encoder', choices=['gru', 'lstm'], type=str,
                     default='gru', help='use gru or lstm for the encoder')
    trp.add_argument('--encoder-layers', type=int,
                     help='how many layers to use for encoder', default=1)
    trp.add_argument('--n-shared-layers', type=int,
                     help='share parameters for first n layers', default=1)

    trp.add_argument('--decoder', choices=['gru', 'lstm'], type=str,
                     default='gru', help='use gru or lstm for the decoder')
    trp.add_argument('--disable-attention', action='store_true',
                     help='use a decoder without attention')

    # optimization and regularization
    trp.add_argument('--optimizer',
                     choices=['adadelta', 'adam', 'rmsprop', 'sgd'], type=str,
                     help='what optimizer to use', default='adam')
    trp.add_argument('--learning-rate', metavar='LRATE', type=float,
                     help='learning rate', default=0.001)
    trp.add_argument('--decay-c', metavar='DECAY', type=float, help='decay',
                     default=0.)
    trp.add_argument('--clip-c', metavar='CLIP', type=float, help='clip',
                     default=1.)
    trp.add_argument('--alpha-c', metavar='ALPHAC', type=float, help='clip',
                     default=0.)
    trp.add_argument('--dropout', action='store_true',
                     help='enable/disable all dropout layers')
    trp.add_argument('--dropout-src', type=float,
                     help='dropout complete source words', default=0.1)
    trp.add_argument('--dropout-trg', type=float,
                     help='dropout complete target words', default=0.2)
    trp.add_argument('--dropout-emb', type=float,
                     help='apply dropout on embeddings', default=0.2)
    trp.add_argument('--dropout-rec', type=float,
                     help='apply dropout on recurrent', default=0.2)
    trp.add_argument('--dropout-hid', type=float,
                     help='apply dropout on hidden states', default=0.2)

    # batch size and max length of inputs
    trp.add_argument('--batch-size', metavar='BSIZE', type=int,
                     help='batch size', default=80)
    trp.add_argument('--valid-batch-size', metavar='VBSIZE', type=int,
                     help='validation batch size', default=80)
    trp.add_argument('--sort-k-batches', metavar='k', type=int,
                     help='load k batches, sort by source sentence length, and re-batch',
                     default=20)
    trp.add_argument('--maxlen', type=int,
                     help='maximum sentence length to train on', default=50)
    trp.add_argument('--max-epochs', type=int,
                     help='maximum amount of epochs to train on', default=30)

    # validation
    trp.add_argument('--bleu-script', metavar='BLEU-path', type=str,
                     help='path to BLEU script',
                     default='nmt/multi-bleu.perl')
    trp.add_argument('--postprocess-script', metavar='SCRIPT-path', type=str,
                     help='path to post processing script',
                     default='cat')
    trp.add_argument('--bleu-val-burnin', metavar='ITER', type=int,
                     help='start validation with BLEU after this many iterations',
                     default=10000)  # 25000
    trp.add_argument('--bleu-val-ref', type=str,
                     help='references for BLEU validation (post-processed dev set)',
                     default='validation.tok')
    trp.add_argument('--bleu-val-out', type=str,
                     help='where to save validation result',
                     default='validation')
    trp.add_argument('--at_replace', action='store_true', help="Remove the \
            @@ and @-@ symbols in the generated outputs?")
    trp.add_argument('--subword_at_replace', action='store_true', help="Remove the \
            @@ symbols in the generated outputs? (use this for removing the \
            subword delimiters")

    # multi-tasking
    trp.add_argument('--mtl', action='store_true', help='Multi-Task Learning')
    trp.add_argument('--mtl-ratio', type=float, default=[0.9, 0.1], nargs='+',
                     help='Ratio between the tasks (must sum to 1)')
    trp.add_argument('--mtl-configs', type=str, default=['mtl.yaml'], nargs='+',
                     help='List of configuration files for MTL tasks.')
    trp.add_argument('--mtl-decoder', action='store_true',
                     help='predict dependencies at target side - this requires '
                          'factored y input (factors are used for other tasks only')

    # some frequencies
    trp.add_argument('--validation-frequency', type=int,
                     help='validate the every this many iterations',
                     default=-1)
    trp.add_argument('--display-frequency', type=int,
                     help='display info every this many iterations', default=10)
    trp.add_argument('--save-frequency', type=int,
                     help='save the model every this many iterations',
                     default=-1)
    trp.add_argument('--sample-frequency', type=int,
                     help='sample from the model every this many iterations',
                     default=-1)
    trp.add_argument('--beam-size', metavar='BEAMSIZE', type=int,
                     help='beam size for decoding', default=12)
    trp.add_argument('--track-n-models', metavar='N', type=int,
                     help='track n models', default=3)

    # misc
    trp.add_argument('--finish-after', type=int,
                     help='finish after this many updates', default=-1)
    trp.add_argument('--unk-symbol', type=str,
                     help='symbol to use for unknown words', default='<UNK>')
    trp.add_argument('--eos-symbol', type=str,
                     help='symbol to use for end of sentence', default='</s>')
    trp.add_argument('--patience', type=int, help='early stopping patience',
                     default=5)
    trp.add_argument('--early_stopping', type=str, help='cost || bleu',
                    default='cost')
    trp.add_argument('--reload', action='store_true',
                     help='try to reload a previously saved model')
    trp.add_argument('--verbose', '-v', action='count',
                     help='set verbosity level')
    trp.add_argument('--disp-alignments', action='store_true',
                     help='display alignments')

    # test
    tsp.add_argument('-k', type=int, default=5,
                     help='Beam size (default: %(default)s))')
    tsp.add_argument('-p', type=int, default=5,
                     help='Number of processes (default: %(default)s))')
    tsp.add_argument('-n', action='store_true',
                     help='Normalize scores by sentence length')
    tsp.add_argument('-c', action='store_true', help='Character-level')
    tsp.add_argument('-v', action='store_true', help='verbose mode.')
    tsp.add_argument('--models', '-m', type=str, nargs='+', required=True)
    tsp.add_argument('--configs', '-cfg', type=str, nargs='+', required=True)
    tsp.add_argument('--input', '-i', type=str, metavar='PATH',
                     help='Input file')
    tsp.add_argument('--output', '-o', type=str, metavar='PATH',
                     help='Output file')
    tsp.add_argument('--output-dir', metavar='output-dir', type=str,
                     default='.', help='where to save output files')
    tsp.add_argument('--output-alignment', '-a', type=str, default=None,
                     metavar='PATH',
                     help='Output file for alignment weights (default: standard output)')
    tsp.add_argument('--src-dicts', type=str, nargs='+', required=True,
                     help='source dictionaries, 1 per factor')
    tsp.add_argument('--trg-dict', type=str, required=True,
                     help='target dictionary')
    tsp.add_argument('--json-alignment', action='store_true',
                     help='Output alignment in json format')
    tsp.add_argument('--n-best', action='store_true',
                     help='Write n-best list (of size k)')
    tsp.add_argument('--suppress-unk', action='store_true',
                     help='Suppress hypotheses containing UNK.')
    tsp.add_argument('--print-word-probabilities', '-wp', action='store_true',
                     help='Print probabilities of each world')
    print(parser.parse_args())

    return vars(parser.parse_args())


if __name__ == '__main__':
    main()
