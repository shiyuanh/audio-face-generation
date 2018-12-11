import torch.nn as nn
import functools
import torch
def conv_block(in_dim,out_dim):
    return nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0),
                         nn.AvgPool2d(kernel_size=2,stride=2))
def deconv_block(in_dim,out_dim):
    return nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.UpsamplingNearest2d(scale_factor=2))

class Discriminator(nn.Module):
    def __init__(self,nc,ndf,hidden_size,imageSize):
        super(Discriminator,self).__init__()
        # 64 x 64 
        self.conv1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1),
                                    nn.ELU(True),
                                    conv_block(ndf,ndf))
        # 32 x 32 
        self.conv2 = conv_block(ndf, ndf*2)
        # 16 x 16 
        self.conv3 = conv_block(ndf*2, ndf*3)
        if(imageSize == 64):
            # 8 x 8
            self.conv4 = nn.Sequential(nn.Conv2d(ndf*3,ndf*3,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ndf*3,ndf*3,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True)) 
            self.embed1 = nn.Linear(ndf*3*8*8, hidden_size)
        else:
            self.conv4 = conv_block(ndf*3, ndf*4)
            self.conv5 = nn.Sequential(nn.Conv2d(ndf*4,ndf*4,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ndf*4,ndf*4,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True)) 
            self.embed1 = nn.Linear(ndf*4*8*8, hidden_size)
        self.embed2 = nn.Linear(hidden_size, ndf*8*8)

        # 8 x 8
        self.deconv1 = deconv_block(ndf, ndf)
        # 16 x 16
        self.deconv2 = deconv_block(ndf, ndf)
        # 32 x 32
        self.deconv3 = deconv_block(ndf, ndf)
        if(imageSize == 64):
        # 64 x 64
            self.deconv4 = nn.Sequential(nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ndf, nc, kernel_size=3, stride=1, padding=1))
        else:
            self.deconv4 = deconv_block(ndf, ndf)
            self.deconv5 = nn.Sequential(nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ndf, nc, kernel_size=3, stride=1, padding=1),
                             nn.Tanh())

        self.ndf = ndf
        self.imageSize = imageSize

    def forward(self,x, feature = False):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        if(self.imageSize == 128):
            out = self.conv5(out)
            out = out.view(out.size(0), self.ndf*4 * 8 * 8)
        else:
            out = out.view(out.size(0), self.ndf*3 * 8 * 8)
        if feature:
            fea = out
        out = self.embed1(out)
        out = self.embed2(out)
        out = out.view(out.size(0), self.ndf, 8, 8)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        if(self.imageSize == 128):
            out = self.deconv5(out)
        if feature:
            return out, fea
        else:
            return out

class Generator(nn.Module):
    def __init__(self,nc,ngf,nz,imageSize):
        super(Generator,self).__init__()
        self.embed1 = nn.Linear(nz, ngf*8*8)

        # 8 x 8
        self.deconv1 = deconv_block(ngf, ngf)
        # 16 x 16
        self.deconv2 = deconv_block(ngf, ngf)
        # 32 x 32
        self.deconv3 = deconv_block(ngf, ngf)
        if(imageSize == 128):
            self.deconv4 = deconv_block(ngf, ngf)
            # 128 x 128 
            self.deconv5 = nn.Sequential(nn.Conv2d(ngf,ngf,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ngf,ngf,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1))
        else:
            self.deconv4 = nn.Sequential(nn.Conv2d(ngf,ngf,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ngf,ngf,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1),
                             nn.Tanh())
        self.ngf = ngf
        self.imageSize = imageSize

    def forward(self,x):
        out = self.embed1(x)
        out = out.view(out.size(0), self.ngf, 8, 8)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        if(self.imageSize == 128):
            out = self.deconv5(out)
        return out

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='instance', use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm_type == 'none':
            norm_layer = None
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
    
# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)