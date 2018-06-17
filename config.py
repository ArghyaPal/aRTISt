import numpy as np
from easydict import EasyDict as edict

default = edict()
cfg = default

default.DATASET_NAME = 'birds'
default.EMBEDDING_TYPE = 'cnn-rnn'
default.DATA_DIR = ''
default.VOCAB_FILENAME = ''

default.GPU_ID = '0'
default.CUDA = True
default.WORKERS = 6

# Recurrence options
default.INITIAL_IMAGE_SIZE = 16
default.FINAL_IMAGE_SIZE = 512
default.HIDDEN_STATE_SIZE = 8  # The hidden state will be of dimension 1 x HIDDEN_STATE_SIZE x HIDDEN_STATE_SIZE


default.ENSURE_CAPTION_CONSISTENCY = False

# Training options
default.TRAIN = edict()
default.TRAIN.BATCH_SIZE = 64
default.TRAIN.VIS_COUNT = 64  # Number of images to be visualized
default.TRAIN.MAX_EPOCH = 600
default.TRAIN.SNAPSHOT_INTERVAL = 2000
default.TRAIN.DISCRIMINATOR_LR = 2e-4
default.TRAIN.GENERATOR_LR = 2e-4
default.TRAIN.FLAG = True
default.TRAIN.NET_G = ''
default.TRAIN.NET_D = ''
default.TRAIN.NET_CCCN = ''

default.TRAIN.COEFF = edict()
default.TRAIN.COEFF.KL = 2.0
default.TRAIN.COEFF.UNCOND_LOSS = 1.0
default.TRAIN.COEFF.COLOR_LOSS = 0.0

# GAN options
default.GAN = edict()
default.GAN.EMBEDDING_DIM = 128
default.GAN.DF_DIM = 64
default.GAN.GF_DIM = 64
default.GAN.TEXT_CONDITION = False

default.TEXT = edict()
default.TEXT.DIMENSION = 1024  # Dimension of the original text embedding from SJE

default.CCCN = edict()
default.CCCN.MAX_CAPTION_LEN = 70

def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """
    Load a config file and merge it into the default options.
    """
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, default)
