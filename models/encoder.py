from torch import nn
import numpy as np
import lightning as L
import torch
"""
encoder codes adapted from https://github.com/alain-ryser/tvae  
"""

def assemble_encoder(batch_normalisation=True):
    layers = []
    layers.append(Conv(1, 8, 3, batch_normalisation=batch_normalisation))#128x128 
    layers.append(Residual(8, 32, 3, batch_normalisation=batch_normalisation)) #64x64
    layers.append(Residual(32, 64, 3, batch_normalisation=batch_normalisation)) #32x32
    layers.append(Residual(64, 64, 3, batch_normalisation=batch_normalisation)) #16x16
    layers.append(Residual(64, 128, 3, batch_normalisation=batch_normalisation)) #8x8
    layers.append(Residual(128, 256, 3, batch_normalisation=batch_normalisation)) #4x4
    layers.append(Residual(256, 512, 3, batch_normalisation=batch_normalisation)) #2x2
    layers.append(nn.AdaptiveAvgPool2d(1)) #1x1
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)

class Conv(L.LightningModule):

    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 padding='same',
                 stride=1,
                 batch_normalisation=True,
                 activation=True
                 ):
        super().__init__()

        layers = []
        layers.append(
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=(not batch_normalisation)
                      )
        )
        if batch_normalisation:
            layers.append(nn.BatchNorm2d(out_ch))
        if activation:
            layers.append(nn.ReLU())

        self.conv = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.conv(inputs)


class Residual(L.LightningModule):
    def __init__(self, in_ch, out_ch, kernel_size,batch_normalisation=True):
        super().__init__()
        self.pre_residual = Conv(in_ch, out_ch, kernel_size=kernel_size, stride=1, batch_normalisation=batch_normalisation)
        self.up_channel = Conv(in_ch, out_ch, kernel_size=1, stride=1, batch_normalisation=batch_normalisation)
        self.post_residual = Conv(out_ch, out_ch, kernel_size=kernel_size, stride=2, padding=1, batch_normalisation=batch_normalisation)

    def forward(self, x):
        conv_x = self.pre_residual(x)
        x = self.up_channel(x) + conv_x
        return self.post_residual(x)

# direction adapted from https://github.com/wyhsirius/LIA
class Direction(nn.Module):
    def __init__(self, z_dim, motion_dim):
        super(Direction, self).__init__()

        self.weight = nn.Parameter(torch.randn(z_dim, motion_dim))

    def forward(self, input):
        # input: (bs*t) x z_dim

        weight = self.weight + 1e-8
        Q, R = torch.linalg.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)
            return out
        

class AEEncoder(L.LightningModule):
    """
    Encoder base model
    """

    def __init__(self, zdim=128, motion_dim=24, batch_normalisation=True):
        super().__init__()
        # Encoder backbone
        self.encoder = assemble_encoder(batch_normalisation=batch_normalisation)
        
        # Extract mean latent embedding 
        self.b = nn.Sequential(
            nn.Linear(512, zdim),
            nn.ReLU(),
            nn.Linear(zdim, zdim)
        )

        # Set motion subspace basis 
        self.motion = Direction(zdim, motion_dim)

        # Extract motion subspace coordinates 
        self.motion_coordinates = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, motion_dim)
        )

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.reshape((B*T, C, H, W))
        latent = self.encoder(x)
        # video level mean latent 
        assemble_video_x = latent.reshape((B, T, -1)).mean(axis=1)
        mean_latent = self.b(assemble_video_x) # [B, zdim]

        # frame level motion direction 
        alpha = self.motion_coordinates(latent)
        move = self.motion(alpha) # [B*T, zdim]
        move = move.reshape((B,T,-1))
        # frame level new latent 
        bmean = mean_latent.unsqueeze(1).repeat((1,T,1))
        frame_latent = bmean + move
        return bmean, frame_latent, alpha #(B,T,ZDIM)
