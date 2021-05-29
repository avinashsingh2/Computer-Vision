import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' #mac specific issue

class Encoder(nn.Module):
    def __init__(self, input_channel, filter):
        super(Encoder, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channel, filter, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(filter),
            nn.ReLU(),
            nn.Conv2d(filter, filter, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, input):
        U_E = self.conv_block(input)
        P = self.maxpool(U_E)
        return U_E, P


class Decoder(nn.Module):
    def __init__(self, input_filter, out_filter):
        super(Decoder, self).__init__()
        self.inv_conv_block1 = nn.Sequential(
            nn.ConvTranspose2d(input_filter, out_filter, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_filter),
            nn.ReLU()
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(2 * out_filter, out_filter, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_filter),
            nn.ReLU()
        )

    def forward(self, input, skip_feature):
        inv_conv1 = self.inv_conv_block1(input)
        skip_cat = torch.cat((skip_feature, inv_conv1), 1)
        inv_conv2 = self.conv_block(skip_cat)
        return inv_conv2


class Unet(nn.Module):
    def __init__(self, in_channel, out_class, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(Encoder(in_channel, feature))
            in_channel = feature
        for feature in reversed(features):
            self.decoder.append(Decoder(feature * 2, feature))
        self.bottle_neck = Encoder(features[-1], features[-1] * 2)
        self.out = nn.Conv2d(features[0], out_class, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        skip_conn = []
        for en in self.encoder:
            res = en(input)
            input = res[1]
            # print(input.shape)
            skip_conn.append(res[0])
        input = self.bottle_neck(input)[0]
        # print(input.shape)
        for dec, skip in zip(self.decoder, reversed(skip_conn)):
            # print(skip.shape)
            input = dec(input, skip)
        res = self.out(input)
        return res


if __name__ == "__main__":
    input = torch.randn((1, 3, 256, 256))
    obj = Unet(3, 1)
    result = obj(input)
    # print(result)
    print(result.shape)