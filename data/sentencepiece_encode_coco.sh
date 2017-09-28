import pickle as pkl
import subprocess
import os

data =
pkl.load(open('/home/delliott/src/imaginet/data/coco/train.pkl',
'rb'))
cmd = ["/usr/local/bin/spm_encode", "--model",
"/home/delliott/src/nmt/data/sentencepiece16k/joint.model"]
spm_encode = subprocess.Popen(cmd, stdout=subprocess.PIPE,
stdin=subprocess.PIPE)

sentences = '\n'.join(x[0].strip() for x in data)

encoded_lines = spm_encode.communicate(input=sentences.encode())[0]
enc = encoded_lines.decode('utf-8').split('\n')

newdata = []
imgcounter = 0
itemcounter = 1
for x,y in zip(data,enc):
    newdata.append((y,x[1]))

    newdata = newdata[:-1] # the last entry is just a newline because
    of the join above

    spmpickle = pkl.dump(newdata,
    open('/home/delliott/src/imaginet/data/coco/train.spm16k.pkl',
    'wb'), protocol=2)


    data =
    pkl.load(open('/home/delliott/src/imaginet/data/coco/dev.pkl',
    'rb'))
    cmd = ["/usr/local/bin/spm_encode", "--model",
    "/home/delliott/src/nmt/data/sentencepiece16k/joint.model"]
    spm_encode = subprocess.Popen(cmd, stdout=subprocess.PIPE,
    stdin=subprocess.PIPE)

    sentences = '\n'.join(x[0].strip() for x in data)

    encoded_lines =
    spm_encode.communicate(input=sentences.encode())[0]
    enc = encoded_lines.decode('utf-8').split('\n')

    newdata = []
    for x,y in zip(data,enc):
        newdata.append((y,x[1]))

        newdata = newdata[:-1] # the last entry is just a newline
        because of the join above

        spmpickle = pkl.dump(newdata,
        open('/home/delliott/src/imaginet/data/coco/dev.spm16k.pkl',
        'wb'), protocol=2)
