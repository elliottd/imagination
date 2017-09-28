from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import argparse
import yaml
from train import train
from test import test
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)


def parse_args():
    """
    Arguments parser.
    :return: args object with parsed arguments
    """
    ap = argparse.ArgumentParser('Train an visual feature vector prediction model or test it (predict).')
    ap.add_argument('mode', choices=['train', 'test'], default='train',
                    help='mode: train or test')
    ap.add_argument('config', metavar='CONFIG', type=str,
                    default='config.yaml', help='configuration to use')
    ap.add_argument("--exp_id", type=str, default='',
                    help="Do you want to prepend the parameter files with a \
                    special identifying token?")
    return ap.parse_args()


def main():
    """
    Fires up a train or test session.
    :param args:
    :return:
    """
    args = parse_args()
    config = yaml.load(open(args.config, mode='rb'))
    config['exp_id'] = args.exp_id

    if args.mode == 'train':
        train(config=config)
    elif args.mode == 'test':
        test(config=config)


if __name__ == '__main__':
    main()
