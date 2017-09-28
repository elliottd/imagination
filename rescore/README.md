Rescoring Attention-based encoder-decoder model using RNN Language Model

# Note

This stuff is from Nematus and *untested*.

## Training
Change the hard-coded paths to data in `nmt.py` then run
```
THEANO_FLAGS=device=gpu,floatX=float32 python train_nmt.py 
```
