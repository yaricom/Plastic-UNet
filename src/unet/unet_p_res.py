# The plastic U-Net implementation with infusion of ResNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class UNetpRes(nn.Module):
    def __init__(self, n_channels, n_classes, device, neurons=16, dropout_ratio=0.5, alfa_type='free', rule='hebb', nbf=128, batch_norm=False, bilinear_upsample=False):
        """
        Creates new U-Net network with plastic learning rule implemented
        Arguments:
            n_channels: The number of input n_channels
            n_classes:  The number of ouput classes to be learned
            device:     The torch device to execute tensor operations
            neurons:    The # of neurons for the first leayer
            alfa_type:  The plasticity coefficient ['free', 'yoked'] (if the latter, alpha is a single scalar learned parameter, shared across all connection)
            rule:       The name of plasticity rule to apply ['hebb', 'oja'] (The Oja rule can maintain stable weight values indefinitely in the absence of stimulation, thus allowing stable long-term memories, while still preventing runaway divergences)
            nbf:        The number of features in plasticity rule vector (width * height)
        """
        super(UNetpRes, self).__init__()

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
        # 101 -> 50
        self.conv1 = down(n_channels, neurons, batch_norm=batch_norm)
        self.pool1 = pool_drop(dropout_ratio=dropout_ratio/2)
        # 50 -> 25
        self.conv2 = down(neurons, neurons * 2, batch_norm=batch_norm)
        self.pool2 = pool_drop(dropout_ratio=dropout_ratio)
        # 25 -> 12
        self.conv3 = down(neurons * 2, neurons * 4, batch_norm=batch_norm)
        self.pool3 = pool_drop(dropout_ratio=dropout_ratio)
        # 12 -> 6
        self.conv4 = down(neurons * 4, neurons * 8, batch_norm=batch_norm)
        self.pool4 = pool_drop(dropout_ratio=dropout_ratio)

        # Middle
        self.mid = middle(neurons * 8, neurons * 16, batch_norm=batch_norm)

        # The UP network structure
        # 6 -> 12
        self.uconv4 = up(neurons * 16, neurons * 8, dropout_ratio=dropout_ratio, batch_norm=batch_norm)
        # 12 -> 25
        self.uconv3 = up(neurons * 8, neurons * 4, dropout_ratio=dropout_ratio, batch_norm=batch_norm)
        # 25 -> 50
        self.uconv2 = up(neurons * 4, neurons * 2, dropout_ratio=dropout_ratio, batch_norm=batch_norm)
        # 50 -> 101
        self.uconv1 = up(neurons * 2, neurons * 1, dropout_ratio=dropout_ratio, batch_norm=batch_norm)

        self.outc = outconv(neurons, n_classes)

         # Move network parameters to the specified device
        self.to(device)

        # output info
        print("UNet plastic model with plastic rule [%s] initialized" % self.rule)

    def forward(self, x, hebb):
        # 101 -> 50
        xc1 = self.conv1(x)
        x1 = self.pool1(xc1)
        #print("X1", x1.shape)

        # 50 -> 25
        xc2 = self.conv2(x1)
        x2 = self.pool2(xc2)
        #print("X2", x2.shape)

        # 25 -> 12
        xc3 = self.conv3(x2)
        x3 = self.pool3(xc3)
        #print("X3", x3.shape)

        # 12 -> 6
        xc4 = self.conv4(x3)
        x4 = self.pool4(xc4)
        #print("X4", x4.shape)

        # Middle
        x5 = self.mid(x4)
        #print("X5", x5.shape)

        # 6 -> 12
        x = self.uconv4(x5, xc4)
        #print("X6", x.shape)

        # 12 -> 25
        x = self.uconv3(x, xc3)
        #print("X7", x.shape)

        # 25 -> 50
        x = self.uconv2(x, xc2)
        #print("X8", x.shape)

        # 50 -> 101
        x = self.uconv1(x, xc1)
        #print("X9", x.shape)

        x = self.outc(x)
        #print("OUT", x.shape)

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

class conv_module(nn.Module):
    """
    The simple convolution module with optional batch normalization and activation
    """
    def __init__(self, out_ch, kernel_size, stride=1, padding=1, activation=True, batch_norm=False):
        super(conv_module, self).__init__()
        if batch_norm == True:
            self.conv = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

        self.activation = activation
        if activation == True:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.activation == True:
            x = self.activ(x)
        return x

class residual_block(nn.Module):
    """
    The residual block
    """
    def __init__(self, out_ch, batch_norm=False):
        super(residual_block, self).__init__()
        if batch_norm == True:
            self.conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch),
                conv_module(out_ch, kernel_size=3),
                conv_module(out_ch, kernel_size=3, activation=False)
            )
        else:
            self.conv = nn.Sequential(
                nn.ReLU(inplace=True),
                conv_module(out_ch, kernel_size=3),
                conv_module(out_ch, kernel_size=3, activation=False)
            )

    def forward(self, input):
        x = self.conv(input)
        x = x.add(input)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    """
    The ascending convolution module increasing features by 2 in each dimension and
    decreasing channels number by 2 at the same time
    """
    def __init__(self, in_ch, out_ch, dropout_ratio, batch_norm=False):
        super(up, self).__init__()
        self.dconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=0)
        self.uconv =  nn.Sequential(
            nn.Dropout2d(p=dropout_ratio, inplace=True),
            middle(in_ch, out_ch, batch_norm=False)
        )

    def forward(self, x1, x2):
        x = self.dconv(x1)
        diffX = x2.size()[2] - x.size()[2] #TODO Check the correct order
        diffY = x2.size()[3] - x.size()[3]
        x = F.pad(x, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x, x2], dim=1)
        x = self.uconv(x)
        return x


class middle(nn.Module):
    """
    The middle convolution
    """
    def __init__(self, in_ch, out_ch, batch_norm=False):
        super(middle, self).__init__()
        self.mconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            residual_block(out_ch=out_ch, batch_norm=batch_norm),
            residual_block(out_ch=out_ch, batch_norm=batch_norm),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.mconv(x)
        return x

class pool_drop(nn.Module):
    """
    The pooling with subsequential dropout
    """
    def __init__(self, dropout_ratio):
        super(pool_drop, self).__init__()
        self.dpool = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_ratio, inplace=True)
        )

    def forward(self, x):
        x = self.dpool(x)
        return x


class down(nn.Module):
    """
    The downscending convolution increasing
    channels number by 2
    """
    def __init__(self, in_ch, out_ch, batch_norm=False):
        super(down, self).__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            residual_block(out_ch=out_ch, batch_norm=batch_norm),
            residual_block(out_ch=out_ch, batch_norm=batch_norm),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.dconv(x)
        return x
