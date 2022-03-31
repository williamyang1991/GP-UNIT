import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
from torch.autograd import Function
from model.base_network import BaseNetwork, AdaptiveInstanceNorm

# GradReverse.apply(inp, lambd)
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.lambd), None

class Conv2dBlock(BaseNetwork):
    def __init__(self, fin, fout, kernel_size, padding, stride, param_free_norm_type='none', activation='relu'):
        super(Conv2dBlock, self).__init__()

        # create conv layers
        self.conv = spectral_norm(nn.Conv2d(fin, fout, kernel_size=kernel_size, 
                                            padding=padding, stride=stride, padding_mode='reflect'))

        # define normalization layers
        if param_free_norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(fout)
        elif param_free_norm_type == 'batch':
            self.norm = nn.BatchNorm2d(fout)
        elif param_free_norm_type == 'adain':
            self.norm = AdaptiveInstanceNorm(128, fout)
        elif param_free_norm_type == 'none':
            self.norm = None
        else:
            raise ValueError('Unsupported norm %s' % param_free_norm_type)
            
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()            
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError('Unsupported activation %s' % activation)     

    def forward(self, x, s=None):
        x = self.conv(x)
        # for domain-level based style moodulation
        if isinstance(self.norm, AdaptiveInstanceNorm):
            x = self.norm(x, s)
        elif self.norm:
            x = self.norm(x)
            # for image-level based style modulation
            if s:
                mean, std = s
                x = x * std + mean
        if self.activation:
            x = self.activation(x)
        return x


class ContentEncoder(BaseNetwork):    
    def __init__(self, nef=64):
        super(ContentEncoder, self).__init__()
        
        self.layer1 = nn.Sequential(
            Conv2dBlock(3, nef, 3, 1, 1, 'instance', 'lrelu'), 
            Conv2dBlock(nef, nef, 3, 1, 2, 'instance', 'lrelu')  # B*64*128*128
        )
        self.layer2 = nn.Sequential(
            Conv2dBlock(nef*1, nef*1, 3, 1, 1, 'instance', 'lrelu'), 
            Conv2dBlock(nef*1, nef*2, 3, 1, 2, 'instance', 'lrelu'), # B*128*64*64
        )
        self.layer3 = nn.Sequential(
            Conv2dBlock(nef*2, nef*2, 3, 1, 1, 'instance', 'lrelu'),
            Conv2dBlock(nef*2, nef*4, 3, 1, 2, 'instance', 'lrelu') # B*256*32*32
        )
        self.layer4 = nn.Sequential(
            Conv2dBlock(nef*4, nef*4, 3, 1, 1, 'instance', 'lrelu'), 
            Conv2dBlock(nef*4, nef*8, 3, 1, 2, 'instance', 'lrelu') # B*512*16*16
        )
        self.layer5 = nn.Sequential(
            Conv2dBlock(nef*8, nef*8, 3, 1, 1, 'instance', 'lrelu'), 
            Conv2dBlock(nef*8, nef*8, 3, 1, 2, 'instance', 'lrelu') # B*512*8*8
        )
        self.layer6 = nn.Sequential(
            Conv2dBlock(nef*8, nef*8, 3, 1, 1, 'instance', 'lrelu'), 
            Conv2dBlock(nef*8, 1, 3, 1, 1, 'instance', 'lrelu') # B*1*8*8
        )

    def forward(self, x, get_feature=False):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        c6 = self.layer6(c5)
        if get_feature:
            return [c6, c5, c4, c3, c2, c1]
        return c6
    
class StyleEncoder(BaseNetwork):
    def __init__(self, nef=64):
        super(StyleEncoder, self).__init__()
        
        self.layer1 = Conv2dBlock(3, nef, 3, 1, 2, 'none', 'lrelu')  # B*64*128*128
        self.layer2 = Conv2dBlock(nef*1, nef*2, 3, 1, 2, 'none', 'lrelu') # B*128*64*64
        self.layer3 = Conv2dBlock(nef*2, nef*4, 3, 1, 2, 'none', 'lrelu') # B*256*32*32
        self.layer4 = Conv2dBlock(nef*4, nef*8, 3, 1, 2, 'none', 'lrelu') # B*512*16*16
        self.layer5 = Conv2dBlock(nef*8, nef*8, 3, 1, 2, 'none', 'lrelu') # B*512*8*8
        self.layer6 = Conv2dBlock(nef*8, nef*8, 3, 1, 2, 'none', 'lrelu') # B*512*4*4
    
    # directly compute the mean and standrad deviation of the image features
    # no need to psss to the linear layers as in AdaIN
    def getstyle(self, f):
        f = f.view(f.shape[0],f.shape[1],-1)
        return [f.mean(2, keepdim=True).unsqueeze(dim=3), f.std(2, keepdim=True).unsqueeze(dim=3)]
        
    def forward(self, x):
        out = []
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        f5 = self.layer5(f4)
        f6 = self.layer6(f5)
        out.append(self.getstyle(f1)) # 128
        out.append(self.getstyle(f2)) # 64
        out.append(self.getstyle(f3)) # 32
        out.append(self.getstyle(f4)) # 16
        out.append(self.getstyle(f5)) # 8
        out.append(self.getstyle(f6)) # 4
        return out
    
class Decoder(BaseNetwork):
    def __init__(self, ndf=64):
        super(Decoder, self).__init__()
        
        self.layer1 = nn.Sequential(
            Conv2dBlock(ndf*8, ndf*8, 3, 1, 1, 'adain'), # B*512*4*4
            Conv2dBlock(ndf*8, ndf*8, 3, 1, 1, 'adain'), # B*512*4*4
            Conv2dBlock(ndf*8, ndf*8, 3, 1, 1, 'adain') # B*512*4*4
        )
        self.layer2 = Conv2dBlock(ndf*8, ndf*8, 3, 1, 1, 'adain') # B*512*8*8
        self.layer3 = Conv2dBlock(ndf*8, ndf*8, 3, 1, 1, 'adain') # B*512*16*16
        self.layer4 = Conv2dBlock(ndf*8, ndf*4, 3, 1, 1, 'instance') # B*256*32*32
        self.layer5 = Conv2dBlock(ndf*4, ndf*2, 3, 1, 1, 'instance') # B*128*64*64
        self.layer6 = Conv2dBlock(ndf*2, ndf*1, 3, 1, 1, 'instance') # B*64*128*128
        self.layer7 = Conv2dBlock(ndf*1, 3, 3, 1, 1, 'none', 'tanh') # B*3*256*256
        self.tomask = Conv2dBlock(ndf*8, 3, 3, 1, 1, 'none', 'tanh') # B*3*16*16
        self.up = nn.Upsample(scale_factor=2)
        
    def forward(self, x, s, y):
        f1 = x
        for conv in self.layer1:
            f1 = conv(f1, y)
        f2 = self.up(self.layer2(f1, y))
        tmp = self.layer3(f2, y)
        mask = self.tomask(tmp)
        f3 = self.up(tmp)
        f4 = self.up(self.layer4(f3, s[2]))
        f5 = self.up(self.layer5(f4, s[1]))
        f6 = self.up(self.layer6(f5, s[0]))
        out = self.layer7(f6)
        return out, mask

class Classifer(BaseNetwork):
    def __init__(self, ncls=10):
        super(Classifer, self).__init__()

        self.fc3 = nn.Linear(ncls, ncls)
               
        self.layer1 = Conv2dBlock(1, 32, 3, 1, 2, 'none', 'lrelu')  # B*32*8*4
        self.layer2 = Conv2dBlock(32, 64, 3, 1, 1, 'none', 'lrelu') # B*63*4*4
        self.layer3 = Conv2dBlock(64, ncls, 4, 0, 1, 'none', 'lrelu') # B*C*1*1

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc3(x.view(x.size(0), -1)) # B*C
        return x

class AutoEncoder(BaseNetwork):
    def __init__(self, nf=64, ncls=292, noise=0.2):
        super(AutoEncoder, self).__init__()
        
        self.contentE = ContentEncoder(nf)
        self.styleE = StyleEncoder(nf)
        self.G = Decoder(nf)
        self.C = Classifer(ncls)
        self.so = 8
        self.fc = Conv2dBlock(1, nf*8, 1, 0, 1, 'instance', 'lrelu')
        self.nf = nf
        self.embed = nn.Embedding(ncls, 128)
        self.noise = noise
        
    def forward(self, x, l, ratio=1.0):
        # z: content feature, scode: image-level style codes
        z, ccode, scode = self.encode(x, ratio)
        # domain label to domain-level style codes
        y = self.embed(l)
        xr, mask = self.decode(scode, y, z)
        return xr, ccode, mask
    
    # introduce a classifier C with a gradient reversal layer to make the content feature domain-agnostic
    def classify(self, ccode):
        if isinstance(ccode, list):
            x = torch.cat(ccode, dim=1)
        else:
            x = ccode
        x = GradReverse.apply(x, 1)
        out = self.C(x)
        return out
        
    def encode(self, x, ratio = 1.0):
        ccode = self.contentE(x)
        z = self.reparameterize(ccode)
        if np.random.rand(1) > ratio:
            scode = self.styleE(x[:,:,:,torch.arange(x.size(3) - 1, -1, -1).long()])
        else:
            scode = self.styleE(x)
        return z, ccode, scode
    
    def decode(self, scode, y, z=None):
        if z is None:
            z = torch.randn(scode[0][0].size(0), self.so*self.so, dtype=torch.float32, device=scode[0][0].get_device())
        x = self.fc(z.reshape(-1, 1, self.so, self.so))
        out, mask = self.G(x, scode, y)
        return out, mask
    
    # add Gaussian noise of a fixed variance for robustness
    def reparameterize(self, ccode):
        return ccode + torch.randn_like(ccode) * self.noise