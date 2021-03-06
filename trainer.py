from __future__ import print_function
import os, numpy as np, time

from config import cfg
from helper.utils import mkdir_p
from copy import deepcopy
from PIL import Image

import torch, torch.nn as nn, torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from tensorboard import summary
from tensorboard import FileWriter

from model import Discriminator, Generator, INCEPTION_V3

# Helper Functions : Start
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def compute_inception_score(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * \
             (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def negative_log_posterior_probability(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)


def load_network(gpus):
    # Generator
    netG = Generator()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    print(netG)

    # Discriminator
    netD = Discriminator()
    netD.apply(weights_init)
    netD = torch.nn.DataParallel(netD, device_ids=gpus)
    print(netD)

    # Loading pretrained weights, if exists.
    training_iter = 0
    if cfg.TRAIN.NET_G != '':
        state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict)
        print('Loaded Generator from saved model.', cfg.TRAIN.NET_G)

        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        training_iter = cfg.TRAIN.NET_G[istart:iend]
        training_iter = int(training_iter) + 1

    if cfg.TRAIN.NET_D != '':
        print('Loading Discriminator from %s.pth' % (cfg.TRAIN.NET_D))
        state_dict = torch.load('%s.pth' % (cfg.TRAIN.NET_D))
        netD.load_state_dict(state_dict)

    inception_model = INCEPTION_V3()

    # Moving to GPU
    if cfg.CUDA:
        netG.cuda()
        netD.cuda()
        inception_model = inception_model.cuda()

    inception_model.eval()

    return netG, netD, inception_model, training_iter


def define_optimizers(netG, netD):
    optimizerG = optim.Adam(netG.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR,
                            betas=(0.5, 0.999))

    optimizerD = optim.Adam(netD.parameters(),
                            lr=cfg.TRAIN.DISCRIMINATOR_LR,
                            betas=(0.5, 0.999))

    return optimizerG, optimizerD


def save_model(netG, avg_param_G, netD, epoch, model_dir):
    load_params(netG, avg_param_G)
    torch.save(
        netG.state_dict(),
        '%s/netG_%d.pth' % (model_dir, epoch))

    torch.save(
        netD.state_dict(),
        '%s/netD.pth' % (model_dir))

    print('Saved Generator and Discriminator models.')


def save_img_results(imgs_tcpu, fake_imgs, count, image_dir, summary_writer):

    num = cfg.TRAIN.VIS_COUNT

    # The range of real_img (i.e., self.imgs_tcpu[i][0:num])
    # is changed to [0, 1] by function vutils.save_image
    real_img = imgs_tcpu[0][0:num]
    vutils.save_image(
        real_img, '%s/real_samples.png' % (image_dir),
        normalize=True)
    real_img_set = vutils.make_grid(real_img).numpy()
    real_img_set = np.transpose(real_img_set, (1, 2, 0))
    real_img_set = real_img_set * 255
    real_img_set = real_img_set.astype(np.uint8)
    sup_real_img = summary.image('real_img', real_img_set)
    summary_writer.add_summary(sup_real_img, count)

    # Saving the output of the last time-step
    fake_img = fake_imgs[-1][0:num]
    # The range of fake_img.data (i.e., self.fake_imgs[i][0:num])
    # is still [-1. 1]...
    vutils.save_image(
        fake_img.data, '%s/count_%09d_fake_samples.png' %
        (image_dir, count), normalize=True)

    fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()
    fake_img_set = np.transpose(fake_img_set, (1, 2, 0))
    fake_img_set = (fake_img_set + 1) * 255 / 2
    fake_img_set = fake_img_set.astype(np.uint8)

    sup_fake_img = summary.image('fake_img%d' % count, fake_img_set)
    summary_writer.add_summary(sup_fake_img, count)
    summary_writer.flush()

#  Helper Functions : End


class RecurrentGANTrainer:
    def __init__(self, output_dir, data_loader, imsize):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'model')
            self.image_dir = os.path.join(output_dir, 'image')
            self.log_dir = os.path.join(output_dir, 'log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = FileWriter(self.log_dir)

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def prepare_data(self, data):
        imgs, w_imgs, t_embedding, _, caption_tensors, len_vector = data

        v_caption_tensors = []
        v_len_vector = []
        real_vimgs, wrong_vimgs = [], []
        if cfg.CUDA:
            vembedding = Variable(t_embedding).cuda()
            if caption_tensors is not None:
                v_caption_tensors = Variable(caption_tensors).cuda()
                v_len_vector = len_vector.cuda()
        else:
            vembedding = Variable(t_embedding)
            if caption_tensors is not None:
                v_caption_tensors = Variable(caption_tensors)
                v_len_vector = len_vector

        for i in range(len(imgs)):
            if cfg.CUDA:
                real_vimgs.append(Variable(imgs[i]).cuda())
                wrong_vimgs.append(Variable(w_imgs[i]).cuda())
            else:
                real_vimgs.append(Variable(imgs[i]))
                wrong_vimgs.append(Variable(w_imgs[i]))

        return imgs, real_vimgs, wrong_vimgs, vembedding, v_caption_tensors, v_len_vector

    def train_Dnet(self, count):
        flag = count % 100
        batch_size = self.real_imgs[0].size(0)
        criterion = self.citerion

        netD = self.netD
        optD = self.optimizerD
        real_imgs = self.real_imgs[0]
        wrong_imgs = self.wrong_imgs[0]
        fake_imgs = self.fake_imgs[-1] # Take only the last image

        netD.zero_grad()

        real_labels = self.real_labels[:batch_size]
        fake_labels = self.fake_labels[:batch_size]

        # Calculating the logits
        mu = self.mus[-1]
        real_logits = netD(real_imgs, mu.detach())
        wrong_logits = netD(wrong_imgs, mu.detach())
        fake_logits = netD(fake_imgs.detach(), mu.detach())

        # Calculating the error
        errD_real = criterion(real_logits[0], real_labels)
        errD_wrong = criterion(wrong_logits[0], fake_labels)
        errD_fake = criterion(fake_logits[0], fake_labels)

        if len(real_logits) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
            errD_real_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * criterion(real_logits[1], real_labels)
            errD_wrong_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * criterion(wrong_logits[1], fake_labels)
            errD_fake_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * criterion(fake_logits[1], fake_labels)

            errD_real += errD_real_uncond
            errD_wrong += errD_wrong_uncond
            errD_fake += errD_fake_uncond

            errD = errD_real + errD_wrong + errD_fake
        else:
            errD = errD_real + 0.5 * (errD_wrong + errD_fake)

        # Calculating the gradients
        errD.backward()

        # Backproping
        optD.step()

        if flag == 0:
            summary_D = summary.scalar('D_loss%d', errD.data[0])
            self.summary_writer.add_summary(summary_D, count)

        return errD

    def train_Gnet(self, count):
        self.netG.zero_grad()
        errG_total = 0
        flag = count % 100
        batch_size = self.real_imgs[0].size(0)

        criterion = self.citerion

        mus, logvars = self.mus, self.logvars

        real_labels = self.real_labels[:batch_size]

        # Looping through each time-step.
        for i in range(len(self.fake_imgs)):
            logits = self.netD(self.fake_imgs[i], mus[i])
            errG = criterion(logits[0], real_labels)
            if len(logits) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
                errG_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * criterion(logits[1], real_labels)
                errG += errG_uncond
            errG_total += errG

            if flag == 0:
                summary_D = summary.scalar('G_loss%d' % i, errG.data[0])
                self.summary_writer.add_summary(summary_D, count)

        kl_loss = 0
        for i in range(len(self.fake_imgs)):
            kl_loss += KL_loss(mus[i], logvars[i]) * cfg.TRAIN.COEFF.KL

        errG_total += kl_loss

        # Compute the gradients
        errG_total.backward()

        # BPTT
        self.optimizerG.step()

        return kl_loss, errG_total

    def save_singleimages(self, images, filenames,
                          save_dir, split_dir, sentenceID, imsize, mean=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i]+'_'+str(mean))
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d_sentence%d.png' % (s_tmp, imsize, sentenceID)
            # range from [-1, 1] to [0, 255]
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def train(self):
        self.netG, self.netD, self.inception_model, start_count = load_network(self.gpus)

        avg_param_G = copy_G_params(self.netG)

        self.optimizerG, self.optimizerD = define_optimizers(self.netG, self.netD)

        self.citerion = nn.BCELoss()

        self.real_labels = Variable(torch.FloatTensor(self.batch_size).fill_(1))
        self.fake_labels = Variable(torch.FloatTensor(self.batch_size).fill_(0))
        # self.gradient_one = torch.FloatTensor([1.0])
        # self.gradient_half = torch.FloatTensor([0.5])

        # Initial Hidden State
        h0 = Variable(torch.FloatTensor(self.batch_size, 1, cfg.HIDDEN_STATE_SIZE, cfg.HIDDEN_STATE_SIZE))
        h0_initalized = Variable(torch.FloatTensor(self.batch_size, 1, cfg.HIDDEN_STATE_SIZE, cfg.HIDDEN_STATE_SIZE).normal_(0,1))

        if cfg.CUDA:
            self.citerion.cuda()
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()
            h0 = h0.cuda()
            h0_initalized = h0_initalized.cuda()

        predictions = []
        count = start_count
        start_epoch = start_count // self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            for step, data in enumerate(self.data_loader, 0):
                self.imgs_tcpu, self.real_imgs, self.wrong_imgs, self.txt_embeddings, self.caption_tensors, self.len_vector = self.prepare_data(data)

                # 1. Generate Fake Data from Generator
                h0.data.normal_(0,1)
                self.fake_imgs, self.mus, self.logvars = self.netG(h0, self.txt_embeddings)

                # 2. Update Discriminator
                errD_total = self.train_Dnet(count)

                # 3. Update Generator
                kl_loss, errG_total = self.train_Gnet(count)

                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                pred = self.inception_model(self.fake_imgs[-1].detach())
                predictions.append(pred.data.cpu().numpy())

                if count % 100 == 0:
                    summary_D = summary.scalar('D_loss', errD_total.data[0])
                    summary_G = summary.scalar('G_loss', errG_total.data[0])
                    summary_KL = summary.scalar('KL_loss', kl_loss.data[0])
                    self.summary_writer.add_summary(summary_D, count)
                    self.summary_writer.add_summary(summary_G, count)
                    self.summary_writer.add_summary(summary_KL, count)

                count += 1

                if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                    save_model(self.netG, avg_param_G, self.netD, count, self.model_dir)

                    # Save Images
                    backup_para = copy_G_params(self.netG)
                    load_params(self.netG, avg_param_G)
                    self.fake_imgs, _, _ = self.netG(h0_initalized, self.txt_embeddings)
                    save_img_results(self.imgs_tcpu, self.fake_imgs, count, self.image_dir, self.summary_writer)
                    load_params(self.netG, backup_para)

                    # Compute Inception Score
                    if len(predictions) > 500:
                        predictions = np.concatenate(predictions, 0)
                        mean, std = compute_inception_score(predictions, 10)
                        score_summary = summary.scalar('Inception_mean', mean)
                        self.summary_writer.add_summary(score_summary, count)

                        mean_nlpp, std_nlpp = negative_log_posterior_probability(predictions, 10)
                        mean_nlpp_summary = summary.scalar('NLPP_mean', mean_nlpp)
                        self.summary_writer.add_summary(mean_nlpp_summary, count)

                        predictions = []
            end_t = time.time()
            print('''[%d/%d][%d--%d] Loss_D: %.2f Loss_G: %.2f Loss_KL: %.2f Time: %.2fs
                      '''
                  % (epoch, self.max_epoch, self.num_batches, count,
                     errD_total.data[0], errG_total.data[0], kl_loss.data[0], end_t - start_t))

        save_model(self.netG, avg_param_G, self.netD, count, self.model_dir)
        self.summary_writer.close()

    def evaluate(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: Could not find the saved Generator Model.')
        else:
            # Build and load the generator
            if split_dir == 'test':
                split_dir = 'valid'

            netG = Generator()

            netG.apply(weights_init)
            netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
            print(netG)

            state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Loaded weights to Generator Network.', cfg.TRAIN.NET_G)

            # the path to save generated images
            s_tmp = cfg.TRAIN.NET_G
            istart = s_tmp.rfind('_') + 1
            iend = s_tmp.rfind('.')
            iteration = int(s_tmp[istart:iend])
            s_tmp = s_tmp[:s_tmp.rfind('/')]
            save_dir = '%s/iteration%d' % (s_tmp, iteration)

            h0 = Variable(torch.FloatTensor(self.batch_size, 1, cfg.INITIAL_IMAGE_SIZE, cfg.INITIAL_IMAGE_SIZE))

            if cfg.CUDA:
                netG.cuda()
                h0 = h0.cuda()

            # switch to evaluate mode
            netG.eval()
            for step, data in enumerate(self.data_loader, 0):
                imgs, t_embeddings, filenames = data
                if cfg.CUDA:
                    t_embeddings = Variable(t_embeddings).cuda()
                else:
                    t_embeddings = Variable(t_embeddings)

                embedding_dim = t_embeddings.size(1)
                batch_size = imgs[0].size(0)
                h0.data.normal_(0, 1)

                fake_imgs, _, _ = netG(h0, t_embeddings)
                self.save_singleimages(fake_imgs[-1], filenames, save_dir, split_dir, 1, 32)
