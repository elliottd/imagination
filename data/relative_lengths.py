import numpy
import sys
from scipy.stats import iqr

src_lens = []
trg_lens = []
with open(sys.argv[1], 'r') as f:
    for ll in f:
        src_lens.append(len(ll.strip().split(' ')))

with open(sys.argv[2], 'r') as f:
    for ll in f:
        trg_lens.append(len(ll.strip().split(' ')))

relative_lens = []
for x,y in zip(src_lens, trg_lens):
    relative_lens.append(x/y)

print('Relative lengths of {} compared to {}'.format(sys.argv[1], sys.argv[2]))
print('max ', numpy.max(relative_lens), ' min ', numpy.min(relative_lens), ' average ', numpy.mean(relative_lens), ' median ',
        numpy.median(relative_lens))
print('75 percentile', numpy.percentile(relative_lens, 75))
