from torch import nn
import lightning as L
from .encoder import Conv

"""
decoder codes adapted from https://github.com/alain-ryser/tvae 
"""
def assemble_decoder(zdim):
    # Assemble decoder 
    return nn.Sequential(
        DeConv(zdim,256,4), # 4x4
        DeConv(256,128,4), # 8x8
        DeConv(128,64,4), # 16x16
        DeConv(64,64,4), # 32x32
        DeConv(64,32,4), # 64x64
        DeConv(32,8,4), # 128x128
        Conv(8,1, 3,batch_normalisation=False, activation=False)
    )

class DeConv(L.LightningModule):

    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 batch_normalisation=True,
                 output_padding=0,
                 activation=True
                ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, padding=1, stride=2, output_padding=output_padding, bias=(not batch_normalisation)),
        ]

        if batch_normalisation:
            layers.append(nn.BatchNorm2d(out_ch))
        if activation:
            layers.append(nn.ReLU())
        self.trans_conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.trans_conv(x)
        return x
    
class AEDecoder(L.LightningModule):
    
    """
    Decoder base model
    """
    def __init__(self, zdim):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2)
        # Decoder backbone
        self.decoder = assemble_decoder(zdim)
        self.zdim = zdim

    def forward(self, x):
        B,T = x.shape[:2]
        x = x.reshape((B*T, self.zdim, 1, 1))
        din = self.upsample(x)
        out = self.decoder(din)
        return out

        
    