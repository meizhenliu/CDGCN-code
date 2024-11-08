"""VSE modules"""

import torch
import torch.nn as nn
import numpy as np
# from collections import OrderedDict

from transformers import BertModel

from lib.modules.resnet import ResnetFeatureExtractor
# from lib.modules.aggr.gpo import GPO
from lib.modules.mlp import MLP

import logging

logger = logging.getLogger(__name__)


def padding_mask(embs, lengths):

    mask = torch.ones(len(lengths), embs.shape[1], device=lengths.device)
    for i in range(mask.shape[0]):
        end = int(lengths[i])
        mask[i, :end] = 0.

    return mask.bool()
class weightpool(nn.Module):
    def __init__(self):
        super(weightpool, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.act = nn.ReLU()

    def forward(self, vec):
        out_features = self.fc1(vec)
        out_features = self.act(out_features)
        out_features = self.fc2(out_features)
        n_image = out_features.shape[0]
        # out_weights = nn.Softmax(dim=1)(out_features)
        out_weights = nn.Softmax(dim=1)(
            out_features - torch.max(out_features, dim=1)[0].unsqueeze(1))  # max o是每列最大值，1是每行最大值
        # print(out_features.shape)
        # print(torch.max(out_features, dim=1)[0].shape)

        #out_weights2 = nn.Softmax(dim=2)(out_features - torch.max(out_features, dim=0)[0].repeat(n_image, 1, 1))
        out_weights2 = nn.Softmax(dim=2)(out_features)
        out_emb2 = torch.mul(vec, out_weights2)
        out_emb2 = out_emb2.permute(0, 2, 1)
        pool_emb2 = torch.sum(out_emb2.view(out_emb2.size(0), out_emb2.size(1), -1), dim=2)
        out_emb = torch.mul(vec, out_weights)
        out_emb = out_emb.permute(0, 2, 1)
        pool_emb = torch.sum(out_emb.view(out_emb.size(0), out_emb.size(1), -1), dim=2)
        return pool_emb, pool_emb2

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


def get_text_encoder(embed_size, no_txtnorm=False):
    return EncoderText(embed_size, no_txtnorm=no_txtnorm)


def get_image_encoder( data_name,img_dim, embed_size, precomp_enc_type='basic',
                      backbone_source=None, backbone_path=None, no_imgnorm=False):#data_name,
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc= EncoderImageAggr(
            img_dim, embed_size, precomp_enc_type, no_imgnorm)
    elif precomp_enc_type == 'backbone':
        backbone_cnn = ResnetFeatureExtractor(backbone_source, backbone_path, fixed_blocks=2)
        img_enc = EncoderImageFull(backbone_cnn, img_dim, embed_size, precomp_enc_type, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.precomp_enc_type = precomp_enc_type
        if precomp_enc_type == 'basic':
            self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        # self.gpool = GPO(32, 32)
        self.init_weights()
        self.wpool = weightpool()
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(embed_size, embed_size)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=16,
        #                                            dim_feedforward=embed_size, dropout=0.1)
        # self.aggr = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=None)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths):
        """Extract image feature vectors."""
        features = self.fc(images)
        if self.precomp_enc_type == 'basic':
            # When using pre-extracted region features, add an extra MLP for the embedding transformation
            features = self.mlp(images) + features
            img_emb1, img_emb2 = self.wpool(features)

            # img_emb3 = img_emb1
            features_in = self.linear1(features)
            rand_list_1 = torch.rand(features.size(0), features.size(1)).to(features.device)
            # rand_list_2 = torch.rand(features.size(0), features.size(1)).to(features.device)
            mask1 = (rand_list_1 >= 0.2).unsqueeze(
                 -1)  # 将上述随机数矩阵与 0.2 进行比较，生成一个对应掩码矩阵，该掩码矩阵的大小为 (batch_size, max_len, 1)，对应矩阵中的每一个值为 True（1）表示该位置需要参与Dropout，否则为 False（0）
            # mask2 = (rand_list_2 >= 0.2).unsqueeze(-1)

            feature_1 = features_in.masked_fill(mask1 == 0, -10000)
            features_k_softmax1 = nn.Softmax(dim=1)(
                feature_1 - torch.max(feature_1, dim=1)[0].unsqueeze(1))  # 在进行 softmax 操作时，为了避免数值上溢或下溢，可以通过减去最大值来缩小值域
            attn1 = features_k_softmax1.masked_fill(mask1 == 0, 0)
            feature_img1 = torch.sum(attn1 *  features, dim=1)
            img_emb3 = img_emb1+feature_img1
            #img_emb3 =  img_emb1
            # src_key_padding_mask = padding_mask(features, image_lengths)
            # #
            # # #switch the dim
            # features1 = features.transpose(1, 0)
            # features1 = self.aggr(features1, src_key_padding_mask=src_key_padding_mask)
            # features1 = features1.transpose(1, 0)
            # img_emb3 = img_emb2
            # features, pool_weights = self.gpool(features, image_lengths)
            # features2, features3 = self.wpool(features1)
            # features4 = features2+ features3
            # img_emb = self.fusion1(img_emb1, img_emb2) + self.fusion1(img_emb2, img_emb1)
            # the final global embedding
            # img_emb =  (self.opt.residual_weight * img_emb_res
            #             + (1-self.opt.residual_weight) * img_emb)
            # img_emb_fusion1 = self.fusion1(img_emb, img_glo)
            # img_emb3 = (0.8 * img_emb3+ 0.2 * features4)
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features,img_emb3


class EncoderImageFull(nn.Module):
    def __init__(self, backbone_cnn, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageFull, self).__init__()
        self.backbone = backbone_cnn
        self.image_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type, no_imgnorm)
        self.backbone_freezed = False

    def forward(self, images):
        """Extract image feature vectors."""
        base_features = self.backbone(images)

        if self.training:
            # Size Augmentation during training, randomly drop grids
            base_length = base_features.size(1)
            features = []
            feat_lengths = []
            rand_list_1 = np.random.rand(base_features.size(0), base_features.size(1))
            rand_list_2 = np.random.rand(base_features.size(0))
            for i in range(base_features.size(0)):
                if rand_list_2[i] > 0.2:
                    feat_i = base_features[i][np.where(rand_list_1[i] > 0.20 * rand_list_2[i])]
                    len_i = len(feat_i)
                    pads_i = torch.zeros(base_length - len_i, base_features.size(-1)).to(base_features.device)
                    feat_i = torch.cat([feat_i, pads_i], dim=0)
                else:
                    feat_i = base_features[i]
                    len_i = base_length
                feat_lengths.append(len_i)
                features.append(feat_i)
            base_features = torch.stack(features, dim=0)
            base_features = base_features[:, :max(feat_lengths), :]
            feat_lengths = torch.tensor(feat_lengths).to(base_features.device)
        else:
            feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device)
            feat_lengths[:] = base_features.size(1)

        features = self.image_encoder(base_features, feat_lengths)

        return features

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info('Backbone freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.backbone.set_fixed_blocks(fixed_blocks)
        self.backbone.unfreeze_base()
        logger.info('Backbone unfreezed, fixed blocks {}'.format(self.backbone.get_fixed_blocks()))


# Language Model with BERT
class EncoderText(nn.Module):
    def __init__(self, embed_size, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.bert = BertModel.from_pretrained('/root/autodl-tmp/bert-base-uncased')
        self.linear = nn.Linear(768, embed_size)
        # self.gpool = GPO(32, 32)
        # self.dropout = nn.Dropout(0.4)
        self.wpool = weightpool()
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(embed_size, embed_size)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=16,
        #                                            dim_feedforward=embed_size, dropout=0.1)
        # self.aggr = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=None)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D

        cap_len = lengths
        # bert_emb = self.dropout(bert_emb)

        cap_emb = self.linear(bert_emb)
        cap_emb1, cap_emb2 = self.wpool(cap_emb)
        # cap_emb3 = cap_emb1

        cap_emb = self.dropout(cap_emb)
        lengths =torch.tensor(lengths).to(cap_emb.device)
        max_len = int(lengths.max())
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(cap_emb.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1).to(cap_emb.device)
        cap_emb = cap_emb[:, :int(lengths.max()), :]
        features_in = self.linear1(cap_emb)
        features_in = features_in.masked_fill(mask == 0,
                                              -10000)  # 对 features_in 中的无效特征进行掩码处理，将被掩码的位置的值设置为 -10000。这么做是为了排除掉那些被 padding 的无效特征，避免它们对 softmax 注意力分布的计算产生影响。
        features_k_softmax = nn.Softmax(dim=1)(features_in - torch.max(features_in, dim=1)[0].unsqueeze(1))
        attn = features_k_softmax.masked_fill(mask == 0, 0)
        feature_cap = torch.sum(attn * cap_emb, dim=1)
        cap_emb3 = cap_emb1 +feature_cap
        #cap_emb3 = cap_emb1

        # lengths = torch.tensor(lengths).cuda()
        #
        # src_key_padding_mask = padding_mask(cap_emb, lengths)
        # #
        # # #switch the dim
        # cap_emb4 = cap_emb.transpose(1, 0)
        # cap_emb4 = self.aggr(cap_emb4, src_key_padding_mask=src_key_padding_mask)
        # cap_emb4 = cap_emb4.transpose(1, 0)
        # img_emb3 = img_emb2
        # features, pool_weights = self.gpool(features, image_lengths)
        # cap_emb5, cap_emb6 = self.wpool(cap_emb4)
        # cap_emb7 = cap_emb5+ cap_emb6
        # img_emb = self.fusion1(img_emb1, img_emb2) + self.fusion1(img_emb2, img_emb1)
        # the final global embedding
        # img_emb =  (self.opt.residual_weight * img_emb_res
        #             + (1-self.opt.residual_weight) * img_emb)
        # img_emb_fusion1 = self.fusion1(img_emb, img_glo)
        # cap_emb3 = (0.8 * cap_emb3 + 0.2 * cap_emb7)
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb,cap_emb3
