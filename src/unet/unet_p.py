# The U-Net network implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class UNetp(nn.Module):
    def __init__(self, n_channels, n_classes, device, alfa_type='free', rule='hebb', nbf=128):
        """
        Creates new U-Net network with plastic learning rule implemented
        Arguments:
            n_channels: The number of input n_channels
            n_classes: The number of ouput classes to be learned
            device: The torch device to execute tensor operations
            alfa_type: The plasticity coefficient ['free', 'yoked'] (if the latter, alpha is a single scalar learned parameter, shared across all connection)
            rule: The name of plasticity rule to apply ['hebb', 'oja'] (The Oja rule can maintain stable weight values indefinitely in the absence of stimulation, thus allowing stable long-term memories, while still preventing runaway divergences)
            nbf: The number of features in plasticity rule vector (width * height)
        """
        super(UNetp, self).__init__()

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.nbf = nbf # the number of features to be used for plastic rule learning
        self.torch_dev = device
        self.alfa_type = alfa_type
        self.rule = rule

        # The plastic rule paprameters to be learned
        self.w =  torch.nn.Parameter((.01 * torch.randn(self.nbf, self.nbf, device=self.torch_dev)), requires_grad=True) # Fixed weights
        self.alpha =  torch.nn.Parameter((.01 * torch.rand(self.nbf, self.nbf, device=self.torch_dev)), requires_grad=True) # Plasticity coeffs.
        self.eta = torch.nn.Parameter((.01 * torch.ones(1, device=self.torch_dev)), requires_grad=True)  # The “learning rate” of plasticity (the same for all connections)


        # The DOWN network structure
        self.inc = inconv(n_channels, 8)
        self.down1 = down(8, 16)
        self.down2 = down(16, 32)
        self.down3 = down(32, 64)
        self.down4 = down(64, 64)
        # The UP network structure
        self.up1 = up(128, 32)
        self.up2 = up(64, 16)
        self.up3 = up(32, 8)
        self.up4 = up(16, 8)
        self.outc = outconv(8, n_classes)

         # Move network parameters to the specified device
        self.to(device)

        # output info
        print("UNet plastic model with plastic rule [%s] initialized" % self.rule)

    def forward(self, x, hebb):
        if x.shape[0] != 1:
            raise ValueError("Only batch size: 1 is supported, but was: %d" % x.shape[0])

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        # The Plasticity rule implementation
        activin = x.view(self.nbf, self.nbf) # The batch size assumed to be 1

        if self.alfa_type == 'free':
            activ = activin.mm(self.w + torch.mul(self.alpha, hebb))
        elif self.alfa_type == 'yoked':
            activ = activin.mm(self.w + self.alpha * hebb)
        else:
            raise ValueError("Must select one plasticity coefficient type ('free' or 'yoked')")

        activout = torch.sigmoid(activ)

        if self.rule == 'hebb':
            hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(activin.unsqueeze(2), activout.unsqueeze(1))[0] # bmm used to implement outer product; remember activs have a leading singleton dimension
        elif self.rule == 'oja':
            hebb = hebb + self.eta * torch.mul((activin[0].unsqueeze(1) - torch.mul(hebb, activout[0].unsqueeze(0))), activout[0].unsqueeze(0)) # Oja's rule. Remember that yin, yout are row vectors (dim (1,N)). Also, broadcasting!
        else:
            raise ValueError("Must select one learning rule ('hebb' or 'oja')")

        return activout, hebb

    def initialZeroHebb(self):
        """
        Creates variable to store Hebbian plastisity coefficients
        """
        return torch.zeros(self.nbf, self.nbf, dtype=torch.float, device=self.torch_dev)

class double_conv(nn.Module):
    """
    Creates two subsequent unpadded convolution layers with 3x3 kernel size
    followed by batch normalization (optional) and ReLU
    """
    def __init__(self, in_ch, out_ch, batch_norm=True):
        super(double_conv, self).__init__()
        if batch_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
