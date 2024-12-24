import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    A U-Net–style model for singing voice separation (or similar tasks).

    Model Inputs:
        mix (B, 1, 512, 128)
    Model Outputs:
        mask (B, 1, 512, 128) in the range (0, 1).
    """

    def __init__(self):
        super(UNet, self).__init__()

        # ---------------------------------------------------------
        #   Downsampling layers (encoder)
        # ---------------------------------------------------------
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # ---------------------------------------------------------
        #   Upsampling layers (decoder)
        # ---------------------------------------------------------
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2)
        self.deconv1_post = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )

        self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size=5, stride=2, padding=2)
        self.deconv2_post = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )

        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=5, stride=2, padding=2)
        self.deconv3_post = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )

        self.deconv4 = nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2, padding=2)
        self.deconv4_post = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )

        self.deconv5 = nn.ConvTranspose2d(64, 16, kernel_size=5, stride=2, padding=2)
        self.deconv5_post = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )

        self.deconv6 = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2)

    # ==============================================================
    #   Forward
    # ==============================================================
    def forward(self, mix):
        """
        Forward pass:
            mix (B, 1, H, W) -> mask (B, 1, H, W)

        Returns a predicted mask in (0, 1) range.
        """
        # ----------------- Encoder -----------------
        conv1_out = self.conv1(mix)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)

        # ----------------- Decoder -----------------
        # Each deconv is followed by BN, ReLU, Dropout
        deconv1_out = self.deconv1(conv6_out, output_size=conv5_out.size())
        deconv1_out = self.deconv1_post(deconv1_out)

        deconv2_out = self.deconv2(torch.cat([deconv1_out, conv5_out], dim=1),
                                   output_size=conv4_out.size())
        deconv2_out = self.deconv2_post(deconv2_out)

        deconv3_out = self.deconv3(torch.cat([deconv2_out, conv4_out], dim=1),
                                   output_size=conv3_out.size())
        deconv3_out = self.deconv3_post(deconv3_out)

        deconv4_out = self.deconv4(torch.cat([deconv3_out, conv3_out], dim=1),
                                   output_size=conv2_out.size())
        deconv4_out = self.deconv4_post(deconv4_out)

        deconv5_out = self.deconv5(torch.cat([deconv4_out, conv2_out], dim=1),
                                   output_size=conv1_out.size())
        deconv5_out = self.deconv5_post(deconv5_out)

        deconv6_out = self.deconv6(torch.cat([deconv5_out, conv1_out], dim=1),
                                   output_size=mix.size())

        # Use sigmoid for mask in range (0, 1)
        mask = torch.sigmoid(deconv6_out)
        return mask
