import torch
from torch import nn
from torchvision import transforms


class UNET(torch.nn.Module):

    def encoderLayer(self, in_channels, out_channels, isLast=False):

        enc_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='valid'),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding='valid'),
            nn.ReLU()
        )
        enc_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        if isLast:
            return enc_conv

        return enc_conv, enc_pool

    def decoderLayer(self, in_channels, out_channels, isLast=False):

        if isLast:
            return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding='valid')

        dec_up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 2), stride=2)
        dec_copy_crop = CropCopyConcat()
        dec_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='valid'),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding='valid'),
            nn.ReLU()
        )

        return dec_up, dec_copy_crop, dec_conv

    def __init__(self, noClasses=10):

        super().__init__()

        ##Encoder##

        self.enc_conv_a, self.enc_pool_a = self.encoderLayer(in_channels=3, out_channels=64)
        self.enc_conv_b, self.enc_pool_b = self.encoderLayer(in_channels=64, out_channels=128)
        self.enc_conv_c, self.enc_pool_c = self.encoderLayer(in_channels=128, out_channels=256)
        self.enc_conv_d, self.enc_pool_d = self.encoderLayer(in_channels=256, out_channels=512)
        self.enc_conv_e = self.encoderLayer(in_channels=512, out_channels=1024, isLast=True)

        ##Decoder##

        self.dec_up_d, self.copy_crop_d, self.dec_conv_d = self.decoderLayer(in_channels=1024, out_channels=512)
        self.dec_up_c, self.copy_crop_c, self.dec_conv_c = self.decoderLayer(in_channels=512, out_channels=256)
        self.dec_up_b, self.copy_crop_b, self.dec_conv_b = self.decoderLayer(in_channels=256, out_channels=128)
        self.dec_up_a, self.copy_crop_a, self.dec_conv_a = self.decoderLayer(in_channels=128, out_channels=64)
        self.dec_conv_e = self.decoderLayer(in_channels=64, out_channels=noClasses, isLast=True)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        ##Encoder forward##
        enc_conved_a = self.enc_conv_a(x)
        enc_out_a = self.enc_pool_a(enc_conved_a)
        #         print(enc_out_a.shape)

        enc_conved_b = self.enc_conv_b(enc_out_a)
        enc_out_b = self.enc_pool_b(enc_conved_b)
        #         print(enc_out_b.shape)

        enc_conved_c = self.enc_conv_c(enc_out_b)
        enc_out_c = self.enc_pool_c(enc_conved_c)
        #         print(enc_out_c.shape)

        enc_conved_d = self.enc_conv_d(enc_out_c)
        enc_out_d = self.enc_pool_d(enc_conved_d)
        #         print(enc_out_d.shape)

        enc_conved_e = self.enc_conv_e(enc_out_d)
        enc_out_e = enc_conved_e
        #         print(enc_out_e.shape)

        ##Decoder Forward##
        #         print(enc_out_e.shape)
        dec_out_d = self.dec_conv_d(self.copy_crop_d(enc_conved_d, self.dec_up_d(enc_out_e)))
        #         print(dec_out_d.shape)

        dec_out_c = self.dec_conv_c(self.copy_crop_c(enc_conved_c, self.dec_up_c(dec_out_d)))
        #         print(dec_out_c.shape)

        dec_out_b = self.dec_conv_b(self.copy_crop_b(enc_conved_b, self.dec_up_b(dec_out_c)))
        #         print(dec_out_b.shape)

        dec_out_a = self.dec_conv_a(self.copy_crop_a(enc_conved_a, self.dec_up_a(dec_out_b)))
        #         print(dec_out_a.shape)

        dec_out_e = self.dec_conv_e(dec_out_a)
        #         print(dec_out_e.shape)

        #         outputMask = self.softmax(dec_out_e)
        #         print(outputMask.shape)

        return dec_out_e


class CropCopyConcat(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, enc_inp, dec_inp):
        croped = transforms.CenterCrop(size=dec_inp.shape[2:])(enc_inp)
        out = torch.cat((croped, dec_inp), dim=1)

        return out
