import numpy as np
import logging
from nmt.utils import load_dictionary, fopen
from future.utils import implements_iterator

logger = logging.getLogger(__name__)


@implements_iterator
class TextIterator:
    """
    A parallel corpus iterator that can shuffle the data each epoch.
    Loads the data into memory for fast access and easy shuffling.
    """

    def __init__(self, source_path, target_path, source_dicts, target_dicts,
                 batch_size=80, maxlen=50, n_words_source=-1,
                 n_words_target=-1, maxibatch_size=20, sort_by_length=True,
                 shuffle_each_epoch=True, factors=1, factors_trg=1,
                 max_unk_ratio=1.0):
        """
        This loads the source and target sentences into memory and sets up a batch iterator
        :param source_path:
        :param target_path:
        :param source_dicts:
        :param target_dicts
        :param batch_size:
        :param maxlen:
        :param n_words_source:
        :param n_words_target:
        :param maxibatch_size:
        :param shuffle_each_epoch:
        :param factors: how many factors to extract
        """
        if type(source_dicts[0]) != dict:
            self.source_dicts = [load_dictionary(d, max_words=n_words_source)
                                 for d in source_dicts]
            self.target_dicts = [load_dictionary(d, max_words=n_words_target)
                                 for d in target_dicts]
        else:
            self.source_dicts = source_dicts
            self.target_dicts = target_dicts

        source = TextIterator.load_data(source_path, dicts=self.source_dicts,
                                        n_words=n_words_source,
                                        get_factors=True, factors=factors)
        target = TextIterator.load_data(target_path, dicts=self.target_dicts,
                                        n_words=n_words_target,
                                        get_factors=factors_trg > 1,
                                        factors=factors_trg)

        assert len(source) == len(target), \
            'unequal amount of source {} and target sentences {}'.format(len(source), len(target))

        self.data = list(zip(source, target))
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.buffer = []
        self.k = batch_size * maxibatch_size
        self.end_of_data = False
        self.rs = np.random.RandomState(1986)  # HACK random state for shuffling data only

        self.index = 0
        self.reset()
        self.shuffle_each_epoch = shuffle_each_epoch

        if shuffle_each_epoch:
            self.shuffle()

        self.max_unk_ratio = max_unk_ratio

    def __iter__(self):
        return self

    def __next__(self):
        """
        Fills the buffer of k batches if needed and returns a batch
        :return: batch
        """

        # fill buffer if empty
        if len(self.buffer) == 0:
            if self.end_of_data:
                self.end_of_data = False
                self.reset()
                if self.shuffle_each_epoch:
                    self.shuffle()
                raise StopIteration
            else:
                self.fill_buffer()

        # reset data if buffer still empty! we ended an epoch
        if len(self.buffer) == 0:
            self.end_of_data = False
            self.reset()
            if self.shuffle_each_epoch:
                self.shuffle()
            raise StopIteration

        # fill batch
        batch_x = []
        batch_y = []
        while len(batch_x) < self.batch_size:
            try:
                x, y = self.buffer.pop()
                #if self.max_unk_ratio < 1.0:
                #    # HACK: This assumes source_dicts is not a list
                #    if self.is_too_unky(x, source_dicts):
                #        continue
                #    elif self.is_too_unky(y,target_dict):
                #        continue
                #    else:
                batch_x.append(x)
                batch_y.append(y)
            except IndexError:
                break

        return batch_x, batch_y

    def is_too_unky(self, data, dictionary):
        '''
        Calculates if an instance would fall below the desired UNK ratio
        specified by the user. Returns False if the instance is fine, returns
        True if the instance is too unky => should be skipped.
        '''
        words = data[0].split()
        ids = [dictionary[word] if word in w2i else 1 for word in words]
        unks = sum([x for x in ids if x == 1])
        if float(unks)/len(words) <= max_unk_ratio:
            return False
        else:
            return True

    def fill_buffer(self):
        """
        Fills the buffer with k batches of data and sorts the batches according to target sentence length
        """
        try:
            while len(self.buffer) < self.k:
                line = self.data[self.index]
                self.index += 1
                if len(line[0]) > self.maxlen or len(line[1]) > self.maxlen:
                    continue
                self.buffer.append(line)
        except IndexError:
            self.end_of_data = True

        lengths = np.array([len(pair[1]) for pair in self.buffer])
        indexes = lengths.argsort()
        indexes = reversed(indexes)
        _buf = [self.buffer[i] for i in indexes]
        self.buffer = _buf

    def reset(self):
        """
        Resets the index that points to the list of sentences
        :return:
        """
        self.index = 0

    def shuffle(self):
        """
        Shuffles the data set in memory
        :return:
        """
        logger.info('Shuffling data')
        self.rs.shuffle(self.data)

    def __len__(self):
        """
        Get the number of sentences in the iterator
        :return:
        """
        return len(self.data)

    @staticmethod
    def load_data(path, dicts=(), n_words=-1, get_factors=False, factors=1,
                  encoding='utf-8'):

        assert type(dicts) == list, 'provide dictionaries as a list'
        data = []

        with fopen(path, mode='r', encoding=encoding) as f:
            for s in f:
                s = s.strip().split()  # tokens of form 'factor1|factor2|...' (num_dicts) or 'word' (no num_dicts)

                if get_factors:
                    s= [[dicts[i][f] if f in dicts[i] else 1 for (i, f) in
                          enumerate(w.split('|')[:factors])]
                         for w in s]
                else:
                    s = [[dicts[0][w]] if w in dicts[0] else [1] for w in s]

                if len(s) == 0:
                    continue

                # replace out-of-vocabulary words with the unknown word symbol (1)
                if n_words > 0:
                    s = [[f if f < n_words else 1 for f in w] for w in s]

                data.append(s)

        logger.info("Loaded {} sentences from {}".format(len(data), path))
        logger.info("UNK replaced words with an index > {}".format(n_words))

        return data


def test():
    source = '/Users/joost/test'
    target = '/Users/joost/test'
    source_dict = '/Users/joost/git/nmt/tools/seqcopy/train.json'
    target_dict = '/Users/joost/git/nmt/tools/seqcopy/train.json'
    maxlen = 20
    batch_size = 1
    n_words_source = 20
    n_words_target = 20
    k = 2
    shuffle_each_epoch = True

    a = TextIterator(source, target, source_dicts=[source_dict],
                     target_dict=target_dict, batch_size=batch_size,
                     maxlen=maxlen, n_words_source=n_words_source,
                     n_words_target=n_words_target, maxibatch_size=k,
                     sort_by_length=True, shuffle_each_epoch=shuffle_each_epoch)

    for epoch in range(3):
        print("Epoch %d" % epoch)
        for batch_x, batch_y in a:
            print(list(zip(batch_x, batch_y)))


if __name__ == '__main__':
    test()
