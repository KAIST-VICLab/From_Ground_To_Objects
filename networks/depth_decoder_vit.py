import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvBlock, Conv3x3, upsample

from einops import rearrange
import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)

    def forward(self, x):
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, in_feature):
        x = in_feature
        b, c, _, _ = in_feature.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        out = avg_out
        return self.sigmoid(out).expand_as(in_feature) * in_feature

class Attention_Module(nn.Module):
    def __init__(self, high_feature_channel, output_channel=None):
        super(Attention_Module, self).__init__()
        in_channel = high_feature_channel
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        channel = in_channel
        self.ca = ChannelAttention(channel)
        # self.sa = SpatialAttention()
        # self.cs = CS_Block(channel)
        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_features):
        features = high_features

        features = self.ca(features)
        # features = self.sa(features)
        # features = self.cs(features)

        return self.relu(self.conv_se(features))


class fSEModule(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None):
        super(fSEModule, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        reduction = 16
        channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_features, low_features):
        features = [upsample(high_features)]
        features += low_features
        features = torch.cat(features, 1)

        b, c, _, _ = features.size()
        y = self.avg_pool(features).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        y = self.sigmoid(y)
        features = features * y.expand_as(features)

        return self.relu(self.conv_se(features))

class SoftAttnDepth(nn.Module):
    def __init__(self, alpha=0.01, beta=1.0, dim=1, discretization='SOD'):
        super(SoftAttnDepth, self).__init__()
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.discretization = discretization

    def get_depth_sid(self, depth_labels):
        alpha_ = torch.FloatTensor([self.alpha])
        beta_ = torch.FloatTensor([self.beta])
        t = []
        for K in range(depth_labels):
            K_ = torch.FloatTensor([K])
            t.append(torch.exp(torch.log(alpha_) + torch.log(beta_ / alpha_) * K_ / depth_labels))
        t = torch.FloatTensor(t)
        return t

    def forward(self, input_t, eps=1e-6):
        batch_size, depth, height, width = input_t.shape
        if self.discretization == 'SID':
            grid = self.get_depth_sid(depth).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            grid = torch.linspace(
                self.alpha, self.beta, depth,
                requires_grad=False).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        grid = grid.repeat(batch_size, 1, height, width).float()

        z = F.softmax(input_t, dim=self.dim)
        z = z * (grid.to(z.device))
        z = torch.sum(z, dim=1, keepdim=True)

        return z


class DepthDecoderViT(nn.Module):
    def __init__(self, opt, ch_enc=[64, 128, 216, 288, 288], scales=range(4), num_ch_enc=[64, 64, 128, 256, 512],
                 backproject_depth=None, min_depth=0.1, max_depth=100):
        super(DepthDecoderViT, self).__init__()
        self.opt = opt
        self.num_ch_enc = num_ch_enc
        self.ch_enc = ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()
        self.norms = nn.ModuleDict()

        self.backproject_depth = backproject_depth
        self.min_depth = min_depth
        self.max_depth = max_depth

        if self.opt['num_output_channels'] > 1:
            alpha = 1e-3,
            beta = 1.0,
            discretization = 'SID'
            self.depth_layer = SoftAttnDepth(alpha=alpha, beta=beta, discretization=discretization)
        # feature fusion
        self.convs["f4"] = Attention_Module(self.ch_enc[4], num_ch_enc[4])
        self.convs["f3"] = Attention_Module(self.ch_enc[3], num_ch_enc[3])
        self.convs["f2"] = Attention_Module(self.ch_enc[2], num_ch_enc[2])
        self.convs["f1"] = Attention_Module(self.ch_enc[1], num_ch_enc[1])

        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]

        self.relu = nn.ReLU()

        for j in range(5):
            for i in range(5 - j):
                # upconv 0
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:
                    num_ch_in //= 2
                num_ch_out = num_ch_in // 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

        # declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                + self.num_ch_dec[row + 1] * (col - 1))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if col == 1:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(num_ch_enc[row + 1] // 2 +
                                                                                 self.num_ch_enc[row],
                                                                                 self.num_ch_dec[row + 1])
            else:
                self.convs["X_" + index + "_downsample"] = Conv1x1(num_ch_enc[row + 1] // 2 + self.num_ch_enc[row]
                                                                   + self.num_ch_dec[row + 1] * (col - 1),
                                                                   self.num_ch_dec[row + 1] * 2)
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2,
                                                                                 self.num_ch_dec[row + 1])

        for i in range(4):
            self.convs["dispconv{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.opt['num_output_channels'])

        self.decoder = nn.ModuleList(list(self.convs.values()) + list(self.norms.values()))
        self.sigmoid = nn.Sigmoid()

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def transposed_attention(self, feature, surface_normal, index):

        surface_normal = self.norms["X_{}_query".format(index)](surface_normal)
        feature = self.norms["X_{}".format(index)](feature)
        if self.opt['use_cross_attention']:
            q = self.convs["X_{}_query_dwconv".format(index)](self.convs["X_{}_query".format(index)](surface_normal))
        else:
            feature = torch.cat([feature, surface_normal], dim=1)
            q = self.convs["X_{}_query_dwconv".format(index)](self.convs["X_{}_query".format(index)](feature))
        k = self.convs["X_{}_key_dwconv".format(index)](self.convs["X_{}_key".format(index)](feature))
        v = self.convs["X_{}_value_dwconv".format(index)](self.convs["X_{}_value".format(index)](feature))

        B, C, H, W = feature.shape
        q  = q.view(B, C, H*W)
        k  = k.view(B, C, H*W)
        v  = v.view(B, C, H*W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.view(B, C, H, W)
        out = self.convs["X_{}_proj".format(index)](out)

        out = feature + out
        return out


    def forward(self, input_features, inputs=None):
        outputs = {}
        feat = {}
        feat[4] = self.convs["f4"](input_features[4])
        feat[3] = self.convs["f3"](input_features[3])
        feat[2] = self.convs["f2"](input_features[2])
        feat[1] = self.convs["f1"](input_features[1])
        feat[0] = input_features[0]

        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = feat[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])

            # add fSE block to decoder
            if index in self.attention_position:
                features["X_" + index] = self.convs["X_" + index + "_attention"](
                    self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)](features["X_{}{}".format(row + 1, col - 1)]),
                    low_features)
                if row != 3:
                    if self.opt['num_output_channels'] > 1:
                        outputs[("disp", row+1)] = self.depth_layer(self.convs["dispconv{}".format(row+1)](features["X_" + index]))
                    else:
                        outputs[("disp", row+1)] = self.sigmoid(self.convs["dispconv{}".format(row+1)](features["X_" + index]))
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row + 1, col - 1)], low_features)

        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](upsample(x))
        if self.opt['num_output_channels'] > 1:
            outputs[("disp", 0)] = self.depth_layer(self.convs["dispconv0"](x))
        else:
            outputs[("disp", 0)] = self.sigmoid(self.convs["dispconv0"](x))
        return outputs

