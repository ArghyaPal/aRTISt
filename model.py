import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable

from config import cfg


class INCEPTION_V3(nn.Module):
    """
    Using the ImageNet pretrained Inception Network to analyse the Inception Score of the
    generated images while training. The final evaluation is done by a fine-tined Inception model.
    """
    def __init__(self):
        super(INCEPTION_V3, self).__init__()
        self.model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        # print(next(model.parameters()).data)
        state_dict = \
            model_zoo.load_url(url, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(next(self.model.parameters()).data)
        # print(self.model)

    def forward(self, input):
        # [-1.0, 1.0] --> [0, 1.0]
        x = input * 0.5 + 0.5
        # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        # --> mean = 0, std = 1
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        #
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.model(x)
        x = nn.Softmax()(x)
        return x


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# -- Preparing Generator -- #

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.fc = nn.Linear(self.t_dim, self.ef_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.ef_dim]
        logvar = x[:, self.ef_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class NewHiddenLayer(nn.Module):
    def __init__(self, ngf=cfg.GAN.GF_DIM, num_residual=1):
        super(NewHiddenLayer, self).__init__()
        self.ngf = ngf
        self.num_residual = num_residual
        self.hstate_depth = cfg.HIDDEN_STATE_DEPTH
        self.initial_depth = 1024 + self.hstate_depth  # (Depth of the current_image_features + hstate_depth)

        self.jointConv = Block3x3_relu(self.initial_depth, self.ngf)
        self.residual = self._make_layer(ResBlock, self.ngf)
        self.single_out = nn.Conv2d(self.ngf, self.hstate_depth, kernel_size=1)

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def forward(self, prev_hidden_state, current_image_features):
        stack = torch.cat((prev_hidden_state, current_image_features),1)
        out_code = self.jointConv(stack)
        out_code = self.residual(out_code)
        out_code = self.single_out(out_code)
        return out_code


class FinalImageLayer(nn.Module):
    def __init__(self, ngf):
        super(FinalImageLayer, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class upStage(nn.Module):
    def __init__(self, ndf, num_residuals=1):
        super(upStage, self).__init__()
        self.df_dim = ndf
        self.num_residuals = num_residuals

        self.residual = self._pack_layers(ResBlock, self.df_dim)
        self.upsample = upBlock(self.df_dim, self.df_dim // 2)

    def _pack_layers(self, block, num_channels):
        layers = []
        for i in range(self.num_residuals):
            layers.append(block(num_channels))
        return nn.Sequential(*layers)

    def forward(self, activations):
        out_acts = self.residual(activations)
        out_acts = self.upsample(out_acts)
        return out_acts


class GenerateImage(nn.Module):
    def __init__(self, num_residual=2):
        super(GenerateImage, self).__init__()

        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.gf_dim = cfg.GAN.GF_DIM * 16
        self.num_residual = num_residual
        self.hstate_depth = cfg.HIDDEN_STATE_DEPTH

        self.joinConv = Block3x3_relu(self.hstate_depth + self.ef_dim, self.gf_dim)  # SD: 8 x 8 (if HIDDEN_STATE_SIZE = 8, SD: Spatial Dimension)
        self.upsample1 = upBlock(self.gf_dim, self.gf_dim // 2)                      # SD: 16 x 16
        self.upsample2 = upBlock(self.gf_dim // 2, self.gf_dim // 4)                 # SD: 32 x 32
        self.upsample3 = upBlock(self.gf_dim // 4, self.gf_dim // 16)                # SD: 64 x 64
        self.image_layer_1 = FinalImageLayer(self.gf_dim // 16)                      # D: 3 x 64 x 64

        self.gf_dimn = cfg.GAN.GF_DIM
        self.up_stage_1 = upStage(self.gf_dimn)                                      # SD: 128 x 128
        self.image_layer_2 = FinalImageLayer(self.gf_dimn // 2)
        self.up_stage_2 = upStage(self.gf_dimn // 2)                                 # SD: 256 x 256
        self.image_layer_3 = FinalImageLayer(self.gf_dimn // 4)

    def forward(self, hidden_state, caption_vector):
        imgs = []
        s_size = hidden_state.size(2)
        c_code = caption_vector.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)
        h_c_code = torch.cat((c_code, hidden_state), 1)

        out_code_8 = self.joinConv(h_c_code)
        out_code_16 = self.upsample1(out_code_8)
        out_code_32 = self.upsample2(out_code_16)
        out_code_64 = self.upsample3(out_code_32)
        img_64 = self.image_layer_1(out_code_64)
        imgs.append(img_64)

        out_us_128 = self.up_stage_1(out_code_64)
        img_128 = self.image_layer_2(out_us_128)
        imgs.append(img_128)

        out_us_256 = self.up_stage_2(out_us_128)
        img_256 = self.image_layer_3(out_us_256)
        imgs.append(img_256)

        return imgs, out_code_16


class GenerateSpatialHState(nn.Module):
    def __init__(self):
        super(GenerateSpatialHState, self).__init__()

        hvec_size = cfg.HIDDEN_VEC_SIZE  # Depth of the H-Vector
        self.hstate_size = cfg.HIDDEN_STATE_SIZE  # Spatial dimension of H-State
        self.hstate_depth = cfg.HIDDEN_STATE_DEPTH  # Depth of the H-State

        self.fc = nn.Sequential(
            nn.Linear(hvec_size, self.hstate_depth * self.hstate_size * self.hstate_size * 2, bias=False),
            nn.BatchNorm1d(self.hstate_depth * self.hstate_size * self.hstate_size * 2),
            GLU())

    def forward(self, h_vector):
        fc_out = self.fc(h_vector)
        h_state = fc_out.view(-1, self.hstate_depth, self.hstate_size, self.hstate_size)
        return h_state


def update_hidden_vector(ngf, hidden_vec_size):
    # for 64 x 64 image.
    h_vec = nn.Sequential(
        # --> state size. ngf x in_size/2 x in_size/2
        nn.Conv2d(3, ngf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ngf x x in_size/4 x in_size/4
        nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ngf x in_size/8 x in_size/8
        nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ngf x in_size/16 x in_size/16
        nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ngf * 8, hidden_vec_size, kernel_size=4, stride=4)
    )
    return h_vec


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.hidden_vec_size = cfg.HIDDEN_VEC_SIZE
        self.gen_image_stage = GenerateImage()
        self.vec_h_to_spatial_h_converter = GenerateSpatialHState()
        self.hidden_vector_update_stage = update_hidden_vector(cfg.GAN.GF_DIM, cfg.HIDDEN_VEC_SIZE)
        # self.hidden_state_update_stage = NewHiddenLayer()
        if cfg.GAN.TEXT_CONDITION:
            self.ca_net = CA_NET()

    def forward(self, hidden_vec, all_caption_vecs):
        c_codes = []
        mus = []
        logvars = []
        images = []
        feature_maps = []

        # Building Recurrence
        spatial_hidden_state = self.vec_h_to_spatial_h_converter(hidden_vec)
        for i in range(all_caption_vecs.size(1)):
            caption_vec = all_caption_vecs[:, i, :]
            c, m, l = self.ca_net(caption_vec)

            # Generate Image
            imgs, out_code_16 = self.gen_image_stage(spatial_hidden_state, c)

            # Update Hidden State
            hidden_vec = self.hidden_vector_update_stage(imgs[0])
            hidden_vec = hidden_vec.view(-1, self.hidden_vec_size)
            spatial_hidden_state = self.vec_h_to_spatial_h_converter(hidden_vec)
            # hidden_state = self.hidden_state_update_stage(hidden_state, out_code_16)

            c_codes.append(c)
            mus.append(m)
            logvars.append(l)
            images.append(imgs)
            if cfg.ENSURE_CAPTION_CONSISTENCY:
                feature_maps.append(out_code_16)

        return images, mus, logvars, feature_maps


# -- Preparing Discriminator -- #

# Downsale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


# Downscale the spatial size by 2
def down_block(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


class Discriminator64(nn.Module):
    def __init__(self):
        super(Discriminator64, self).__init__()
        self.ndf = cfg.GAN.DF_DIM               # Depth of Discriminator Feature maps
        self.embed_dim = cfg.GAN.EMBEDDING_DIM  # Depth of the text embedding

        self.img_s16 = encode_image_by_16times(self.ndf)  # img_s4 contains image of size 4x4 (64/16=4)
        self.logits = nn.Sequential(nn.Conv2d(self.ndf * 8, 1, kernel_size=4, stride=4),
                                    nn.Sigmoid())

        if cfg.GAN.TEXT_CONDITION:
            self.jointConv = Block3x3_leakRelu(self.ndf * 8 + self.embed_dim, self.ndf * 8)
            self.uncond_logits = nn.Sequential(nn.Conv2d(self.ndf * 8, 1, kernel_size=4, stride=4),
                                               nn.Sigmoid())

    def forward(self, image, caption_code=None):
        img_code = self.img_s16(image)
        if cfg.GAN.TEXT_CONDITION and caption_code is not None:
            caption_code = caption_code.view(-1, self.embed_dim, 1, 1)
            caption_code = caption_code.repeat(1, 1, 4, 4)
            h_code = torch.cat((caption_code, img_code), 1)
            h_code = self.jointConv(h_code)
        else:
            h_code = img_code

        output = self.logits(h_code)
        if cfg.GAN.TEXT_CONDITION:
            output_uncond = self.uncond_logits(img_code)
            return [output.view(-1), output_uncond.view(-1)]
        else:
            return [output.view(-1)]


class Discriminator128(nn.Module):
    def __init__(self):
        super(Discriminator128, self).__init__()
        self.ndf = cfg.GAN.DF_DIM               # Depth of Discriminator Feature maps
        self.embed_dim = cfg.GAN.EMBEDDING_DIM  # Depth of the text embedding

        self.img_s16 = encode_image_by_16times(self.ndf)  # img_s4 contains image of size 8x8 (128/16=8)
        self.img_s_32 = down_block(self.ndf * 8, self.ndf * 16)
        self.img_s_32_1 = Block3x3_leakRelu(self.ndf * 16, self.ndf * 8)

        self.logits = nn.Sequential(nn.Conv2d(self.ndf * 8, 1, kernel_size=4, stride=4),
                                    nn.Sigmoid())

        if cfg.GAN.TEXT_CONDITION:
            self.jointConv = Block3x3_leakRelu(self.ndf * 8 + self.embed_dim, self.ndf * 8)
            self.uncond_logits = nn.Sequential(nn.Conv2d(self.ndf * 8, 1, kernel_size=4, stride=4),
                                               nn.Sigmoid())

    def forward(self, image, caption_code=None):
        img_code = self.img_s16(image)
        img_code = self.img_s_32(img_code)
        img_code = self.img_s_32_1(img_code)

        if cfg.GAN.TEXT_CONDITION and caption_code is not None:
            caption_code = caption_code.view(-1, self.embed_dim, 1, 1)
            caption_code = caption_code.repeat(1, 1, 4, 4)
            h_code = torch.cat((caption_code, img_code), 1)
            h_code = self.jointConv(h_code)
        else:
            h_code = img_code

        output = self.logits(h_code)
        if cfg.GAN.TEXT_CONDITION:
            output_uncond = self.uncond_logits(img_code)
            return [output.view(-1), output_uncond.view(-1)]
        else:
            return [output.view(-1)]


class Discriminator256(nn.Module):
    def __init__(self):
        super(Discriminator256, self).__init__()
        self.ndf = cfg.GAN.DF_DIM  # Depth of Discriminator Feature maps
        self.embed_dim = cfg.GAN.EMBEDDING_DIM  # Depth of the text embedding

        self.img_s16 = encode_image_by_16times(self.ndf)  # img_s4 contains image of size 8x8 (128/16=8)
        self.img_s_32 = down_block(self.ndf * 8, self.ndf * 16)
        self.img_s_64 = down_block(self.ndf * 16, self.ndf * 32)
        self.img_s_64_1 = Block3x3_leakRelu(self.ndf * 32, self.ndf * 16)
        self.img_s_64_2 = Block3x3_leakRelu(self.ndf * 16, self.ndf * 8)

        self.logits = nn.Sequential(nn.Conv2d(self.ndf * 8, 1, kernel_size=4, stride=4),
                                    nn.Sigmoid())

        if cfg.GAN.TEXT_CONDITION:
            self.jointConv = Block3x3_leakRelu(self.ndf * 8 + self.embed_dim, self.ndf * 8)
            self.uncond_logits = nn.Sequential(nn.Conv2d(self.ndf * 8, 1, kernel_size=4, stride=4),
                                               nn.Sigmoid())

    def forward(self, image, caption_code=None):
        img_code = self.img_s16(image)
        img_code = self.img_s_32(img_code)
        img_code = self.img_s_64(img_code)
        img_code = self.img_s_64_1(img_code)
        img_code = self.img_s_64_2(img_code)

        if cfg.GAN.TEXT_CONDITION and caption_code is not None:
            caption_code = caption_code.view(-1, self.embed_dim, 1, 1)
            caption_code = caption_code.repeat(1, 1, 4, 4)
            h_code = torch.cat((caption_code, img_code), 1)
            h_code = self.jointConv(h_code)
        else:
            h_code = img_code

        output = self.logits(h_code)
        if cfg.GAN.TEXT_CONDITION:
            output_uncond = self.uncond_logits(img_code)
            return [output.view(-1), output_uncond.view(-1)]
        else:
            return [output.view(-1)]


# -- Preparing CCCN: Cross-Caption Consistency Network -- #

class CCCN_NET(nn.Module):

    def __init__(self, vis_dim=512, vis_num=256, embed_dim=512, hidden_dim=512,
                 vocab_size=10000, num_layers=1, dropout_ratio=0.5):
        """
        :param vis_dim: Depth of the input feature map of the Image.
        :param vis_num: W * H of the input feature map of the Image. (16*16)
        :param embed_dim: Embedding dimension of the word vectors
        :param hidden_dim: Dimension of the hidden state of the LSTM
        :param vocab_size: Vocabulary size of the words
        :param num_layers: Number of layers of LSTM
        :param dropout_ratio: Dropout Ratio :)
        """
        super(CCCN_NET, self).__init__()

        self.embed_dim = embed_dim
        self.vis_dim = vis_dim      # The depth of the image feature tensor.
        self.vis_num = vis_num
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.lstm_cell = nn.LSTMCell(embed_dim + vis_dim, hidden_dim, num_layers)
        self.fc_dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # attention
        self.att_vw = nn.Linear(self.vis_dim, self.vis_dim, bias=False)
        self.att_hw = nn.Linear(self.hidden_dim, self.vis_dim, bias=False)
        self.att_bias = nn.Parameter(torch.zeros(vis_num))
        self.att_w = nn.Linear(self.vis_dim, 1, bias=False)

    def _attention_layer(self, features, hiddens):
        """
        :param features:  batch_size  * 196 * 512
        :param hiddens:  batch_size * hidden_dim
        :return:
        """
        att_fea = self.att_vw(features)
        # N-L-D
        att_h = self.att_hw(hiddens).unsqueeze(1)
        # N-1-D
        att_full = nn.ReLU()(att_fea + att_h + self.att_bias.view(1, -1, 1))
        att_out = self.att_w(att_full).squeeze(2)
        alpha = nn.Softmax()(att_out)
        # N-L
        context = torch.sum(features * alpha.unsqueeze(2), 1)
        return context, alpha

    def forward(self, features, captions, lengths, isTestRun=False):
        """
        :param features: batch_size * 196 * 512
        :param captions: batch_size * time_steps
        :param lengths:
        :return:
        """
        if isTestRun:
            return self.sample(features)

        batch_size, time_step = captions.data.shape
        vocab_size = self.vocab_size
        embed = self.embed
        dropout = self.dropout
        attention_layer = self._attention_layer
        lstm_cell = self.lstm_cell
        fc_dropout = self.fc_dropout
        fc_out = self.fc_out

        features = features.view(features.size(0), self.vis_dim, self.vis_num).transpose(1, 2)

        word_embeddings = embed(captions)
        word_embeddings = dropout(word_embeddings) if dropout is not None else word_embeddings

        feas = torch.mean(features, 1)  # batch_size * 512
        h0, c0 = self.get_start_states(batch_size)

        predicts = self.to_var(torch.zeros(batch_size, time_step, vocab_size))

        for step in xrange(time_step):
            batch_size = sum(i >= step for i in lengths)
            if batch_size == 0:
                break

            if step != 0:
                feas, alpha = attention_layer(features[:batch_size, :], h0[:batch_size, :])

            words = (word_embeddings[:batch_size, step, :]).squeeze(1)
            inputs = torch.cat([feas, words], 1)
            h0, c0 = lstm_cell(inputs, (h0[:batch_size, :], c0[:batch_size, :]))
            outputs = fc_out(fc_dropout(h0)) if fc_dropout is not None else fc_out(h0)
            predicts[:batch_size, step, :] = outputs

        return predicts

    def sample(self, feature, max_len=50):
        # Greedy sample
        embed = self.embed
        lstm_cell = self.lstm_cell
        fc_out = self.fc_out
        attend = self._attention_layer
        batch_size = feature.size(0)

        sampled_ids = []
        alphas = [0]

        words = embed(self.to_var(torch.ones(batch_size, 1).long())).squeeze(1)
        h0, c0 = self.get_start_states(batch_size)
        feas = torch.mean(feature, 1) # convert to batch_size*512

        for step in xrange(max_len):
            if step != 0:
                feas, alpha = attend(feature, h0)
                alphas.append(alpha)
            inputs = torch.cat([feas, words], 1)
            h0, c0 = lstm_cell(inputs, (h0, c0))
            outputs = fc_out(h0)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted.unsqueeze(1))
            words = embed(predicted)

        sampled_ids = torch.cat(sampled_ids, 1)
        return sampled_ids.squeeze(), alphas

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def get_start_states(self, batch_size):
        hidden_dim = self.hidden_dim
        h0 = self.to_var(torch.zeros(batch_size, hidden_dim))
        c0 = self.to_var(torch.zeros(batch_size, hidden_dim))
        return h0, c0
