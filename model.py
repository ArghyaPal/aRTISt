import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models


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


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
