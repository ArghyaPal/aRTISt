from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from config import cfg
from helper.create_vocab import create_CUB_vocab, create_FLOWER_vocab
from PIL import Image
import numpy as np
import pandas as pd
import os, sys, nltk, torch, random

import torchvision.transforms as transforms
import torch.utils.data as data

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def get_imgs(img_path, imsize, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size

    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    for i in range(len(imsize)):
        re_img = transforms.Resize(imsize[i])(img)
        ret.append(normalize(re_img))

    return ret


class BirdsDataset(data.Dataset):
    """
    Custom dataset handler for Caltech-UCSD Birds dataset.
    """
    def __init__(self, data_dir, split='train', embedding_type='cnn-rnn',
                 base_size=64, transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform

        # Define the sizes of the Image Ground-truths
        self.imsize = []
        self.num_progressive_steps = np.log2(cfg.FINAL_IMAGE_SIZE / cfg.INITIAL_IMAGE_SIZE)
        base_size = cfg.INITIAL_IMAGE_SIZE
        for i in range(int(self.num_progressive_steps)+1):
            self.imsize.append(base_size)
            base_size *= 2

        self.data = []
        self.data_dir = data_dir

        # Load Bounding box
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None

        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.captions = self.load_all_captions()

        self.vocab = self.load_vocabulary()

        if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    def load_vocabulary(self):
        data_dir = self.data_dir
        filenames = self.filenames
        vocab_path = os.path.join(data_dir, cfg.VOCAB_FILENAME)
        if not os.path.exists(vocab_path):
            vocab = create_CUB_vocab(data_dir, filenames, vocab_path)
        else:
            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)
            print ('Loaded vocabulary.')
        return vocab

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)

        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()

        print('Total file names: ', len(filenames), filenames[0])

        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in xrange(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox

        return filename_bbox

    def load_all_captions(self):
        def load_captions(caption_name):
            cap_path = caption_name
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
            captions = [cap.replace("\ufffd\ufffd", " ")
                        for cap in captions if len(cap) > 0]
            return captions

        caption_dict = {}
        for key in self.filenames:
            caption_name = '%s/text/%s.txt' % (self.data_dir, key)
            captions = load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict

    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/skip-thought-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
            print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def build_caption_tensors(self, captions):
        vocab = self.vocab
        caption_tensor = []
        len_vector = []
        for caption in captions:
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            target = list()
            target.append(vocab('<start>'))
            target.extend([vocab(word) for word in tokens])
            target.append(vocab('<end>'))
            len_vector.append(len(target))
            target = torch.Tensor(target)

            target_padded = torch.zeros(cfg.CCCN.MAX_CAPTION_LEN).long()
            end = len(target)
            end = cfg.CCCN.MAX_CAPTION_LEN if end > cfg.CCCN.MAX_CAPTION_LEN else end
            # print ('end index: ', end)
            target_padded[:end] = target[:end]

            caption_tensor.append(target_padded)
        len_vector = torch.LongTensor(len_vector)
        return torch.stack(caption_tensor, 0), len_vector

    def prepair_training_pairs(self, index):
        key = self.filenames[index]
        bbox = self.bbox[key]
        data_dir = '%s/CUB_200_2011' % self.data_dir

        # Caption tensor contains a tensor of each word in the captions associated with the image.
        caption_tensors = None
        len_vector = None
        caption_tensors, len_vector = self.build_caption_tensors(self.captions[key])

        # captions = self.captions[key]

        # Retrieve the embedding from the caption
        embeddings = self.embeddings[index, :, :]

        # Load the image. `imgs` contains images of multiple sizes.
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        # Load a wrong image
        wrong_ix = random.randint(0, len(self.filenames) - 1)
        if self.class_id[index] == self.class_id[wrong_ix]:
            wrong_ix = random.randint(0, len(self.filenames) - 1)
        wrong_key = self.filenames[wrong_ix]
        if self.bbox is not None:
            wrong_bbox = self.bbox[wrong_key]
        else:
            wrong_bbox = None
        wrong_img_name = '%s/images/%s.jpg' % (data_dir, wrong_key)
        wrong_imgs = get_imgs(wrong_img_name, self.imsize,
                              wrong_bbox, self.transform, normalize=self.norm)

        # Embeddings for all the captions in the dataset, for an image.
        embedding = embeddings[:5]
        if self.target_transform is not None:
            transformed_embeddings = []
            for i in range(embeddings.shape[0]):
                transformed_embeddings[i] = self.target_transform(embeddings[i, :])
            embedding = transformed_embeddings

        return imgs, wrong_imgs, embedding, key, caption_tensors, len_vector

    def prepair_test_pairs(self, index):
        key = self.filenames[index]
        bbox = self.bbox[key]
        data_dir = '%s/CUB_200_2011' % self.data_dir

        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        embedding = embeddings[:5]
        if self.target_transform is not None:
            embedding = self.target_transform(embeddings[:5])

        return imgs, embedding, key

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)


class FlowersDataset(data.Dataset):
    """
    Custom dataset handler for Oxford 102 Flowers dataset.
    """
    def __init__(self, data_dir, split='train', embedding_type='cnn-rnn',
                 base_size=64, transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform

        # Define the sizes of the Image Ground-truths
        self.imsize = []
        self.num_progressive_steps = np.log2(cfg.FINAL_IMAGE_SIZE / cfg.INITIAL_IMAGE_SIZE)
        base_size = cfg.INITIAL_IMAGE_SIZE
        for i in range(int(self.num_progressive_steps) + 1):
            self.imsize.append(base_size)
            base_size *= 2

        self.data = []
        self.data_dir = data_dir
        self.bbox = None

        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.captions = self.load_all_captions()

        self.vocab = self.load_vocabulary()

        if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    def load_vocabulary(self):
        data_dir = self.data_dir
        filenames = self.filenames
        vocab_path = os.path.join(data_dir, cfg.VOCAB_FILENAME)
        if not os.path.exists(vocab_path):
            vocab = create_FLOWER_vocab(data_dir, filenames, vocab_path, self.class_id)
        else:
            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)
            print ('Loaded vocabulary.')
        return vocab

    def load_all_captions(self):
        def load_captions(caption_name):
            cap_path = caption_name
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
            captions = [cap.replace("\ufffd\ufffd", " ")
                        for cap in captions if len(cap) > 0]
            return captions

        caption_dict = {}

        for i, key in enumerate(self.filenames):
            caption_name = '%s/text/class_%05d/%s.txt' % (self.data_dir, self.class_id[i], key.split('/')[1])
            captions = load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict

    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/skip-thought-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
            print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def build_caption_tensors(self, captions):
        vocab = self.vocab
        caption_tensor = []
        len_vector = []
        for caption in captions:
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            target = list()
            target.append(vocab('<start>'))
            target.extend([vocab(word) for word in tokens])
            target.append(vocab('<end>'))
            len_vector.append(len(target))
            target = torch.Tensor(target)

            target_padded = torch.zeros(cfg.CCCN.MAX_CAPTION_LEN).long()
            end = len(target)
            end = cfg.CCCN.MAX_CAPTION_LEN if end > cfg.CCCN.MAX_CAPTION_LEN else end
            # print ('end index: ', end)
            target_padded[:end] = target[:end]

            caption_tensor.append(target_padded)
        len_vector = torch.LongTensor(len_vector)
        return torch.stack(caption_tensor, 0), len_vector

    def prepair_training_pairs(self, index):
        key = self.filenames[index]

        bbox = None
        data_dir = self.data_dir

        # Caption tensor contains a tensor of each word in the captions associated with the image.
        caption_tensors = None
        len_vector = None
        caption_tensors, len_vector = self.build_caption_tensors(self.captions[key])

        # captions = self.captions[key]

        embeddings = self.embeddings[index, :, :]
        img_name = '%s/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        wrong_ix = random.randint(0, len(self.filenames) - 1)
        if(self.class_id[index] == self.class_id[wrong_ix]):
            wrong_ix = random.randint(0, len(self.filenames) - 1)
        wrong_key = self.filenames[wrong_ix]
        if self.bbox is not None:
            wrong_bbox = self.bbox[wrong_key]
        else:
            wrong_bbox = None
        wrong_img_name = '%s/%s.jpg' % \
            (data_dir, wrong_key)
        wrong_imgs = get_imgs(wrong_img_name, self.imsize,
                              wrong_bbox, self.transform, normalize=self.norm)


        embedding = embeddings[:2]
        if self.target_transform is not None:
            for i in range(embeddings.shape[0]):
                transformed_embeddings[i] = self.target_transform(embeddings[i, :])
            embedding = transformed_embeddings

        return imgs, wrong_imgs, embedding, key, caption_tensors, len_vector

    def prepair_test_pairs(self, index):
        key = self.filenames[index]

        bbox = None
        data_dir = self.data_dir

        embeddings = self.embeddings[index, :, :]
        img_name = '%s/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize, bbox, self.transform, normalize=self.norm)

        embedding = embeddings[:1]
        if self.target_transform is not None:
            embedding = self.target_transform(embeddings[:1])

        return imgs, embedding, key

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)
