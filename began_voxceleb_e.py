from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from models import Discriminator
from models import Generator
from Data.dataset import VoxCeleb
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--loadSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--fineSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--flip', type=int, default=0, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--crop', action='store_true', help='1 for croppping faces, 0 for not')
parser.add_argument('--norm', action='store_true', help='1 for normalizing audios using softmax, 0 for not')
parser.add_argument('--nz', type=int, default=56, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--niter', type=int, default=300000, help='number of iterations to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='dcgan/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dataPath', default='/home/linxd/gan/data/voxceleb/unzippedFaces', help='which dataset to train on')
parser.add_argument('--lambda_k', type=float, default=0.001, help='learning rate of k')
parser.add_argument('--gamma', type=float, default=0.5, help='balance bewteen D and G')
parser.add_argument('--save_step', type=int, default=5000, help='save weights every 50000 iterations ')
parser.add_argument('--hidden_size', type=int, default=64, help='bottleneck dimension of Discriminator')
parser.add_argument('--lr_decay_every', type=int, default=3000, help='decay lr this many iterations')
parser.add_argument('--netG', type=str, default=None)
parser.add_argument('--netD', type=str, default=None)
parser.add_argument('--device', type=int, default = 0)
parser.add_argument('--resume',type = int, default = None)
parser.add_argument('--recon', type=float, default=0., help='reconstruction loss for G')
parser.add_argument('--res', type=float, default=0., help='matching residue')
parser.add_argument('--metric', type=float, default=0., help='metric loss')

opt = parser.parse_args()
print(opt)


os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.device)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

###############   DATASET   ##################
dataset = VoxCeleb(loadSize=opt.loadSize,fineSize=opt.fineSize,flip=opt.flip, crop = opt.crop)
loader_ = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=opt.batchSize,
                                           shuffle=True,
                                           num_workers=8)
loader = iter(loader_)

###############   MODEL   ####################
criterion = nn.L1Loss()
ndf = opt.ndf
ngf = opt.ngf
nc = 3

netD = Discriminator(nc, ndf, opt.hidden_size,opt.fineSize)
netG = Generator(nc, ngf, opt.nz +8,opt.fineSize)
if(opt.cuda):
    netD.cuda()
    netG.cuda()

    
if(opt.netG is not None):
    netG.load_state_dict(torch.load(opt.netG))
if(opt.netD is not None):
    netD.load_state_dict(torch.load(opt.netD))


###########   LOSS & OPTIMIZER   ##########
optimizerD = torch.optim.Adam(netD.parameters(),lr=opt.lrD, betas=(opt.beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))

##########   GLOBAL VARIABLES   ###########
noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
real = torch.FloatTensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
audio = torch.FloatTensor(opt.batchSize, 8, 1, 1)
label = torch.FloatTensor(1)

noise = Variable(noise)
real = Variable(real)
label = Variable(label)
audio = Variable(audio)
if(opt.cuda):
    noise = noise.cuda()
    audio = audio.cuda()
    real = real.cuda()
    label = label.cuda()
    criterion.cuda()

########### Training   ###########
def adjust_learning_rate(optimizer, niter, lrt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lrt * (0.95 ** (niter // opt.lr_decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(torch.nn.functional.normalize(inputs_), torch.nn.functional.normalize(inputs_).t())
    return sim

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# get logger
trainLogger = open('%s/train.log' % opt.outf, 'w')
M_global = AverageMeter()

start_epoch = 1
if not opt.resume == None:
    cpt_G = torch.load('%s/netG_%d.tar' % (opt.outf, opt.resume))
    cpt_D = torch.load('%s/netD_%d.tar' % (opt.outf, opt.resume))
    start_epoch = cpt_G['epoch']
    optimizerD.load_state_dict(cpt_D['optim'])
    optimizerG.load_state_dict(cpt_G['optim'])
    netG.load_state_dict('%s/netG_%d.pth' % (opt.outf, opt.resume))
    netD.load_state_dict('%s/netD_%d.pth' % (opt.outf, opt.resume))
    
    
k = 0 
for iteration in range(start_epoch,opt.niter+1):
    try:
        images, audios = loader.next()
    except StopIteration:
        loader = iter(loader_)
        images, audios = loader.next()
    
    #if(opt.cuda):
    #    images.cuda()
    #    audios.cuda()
    #print(audios.shape)
    netD.zero_grad()

    real.data.resize_(images.size()).copy_(images)
    audio.data.resize_(audios.size()).copy_(audios)
    
    if opt.norm:
        audio = torch.nn.functional.normalize(audio)
        #audio = torch.nn.functional.softmax(audio, 1)
    # generate fake data
    noise.data.resize_(images.size(0), opt.nz)
    noise.data.uniform_(-1,1)
    #print(noise.shape, audios.shape)
    condition = torch.cat((noise, audio), 1)
    #print(condition.shape)
    fake = netG(condition)
    #fake = netG(audio)

    fake_recons, fea_fake = netD(fake.detach(), True)
    real_recons, fea_real = netD(real, True)

    err_real = torch.mean(torch.abs(real_recons-real))
    err_fake = torch.mean(torch.abs(fake_recons-fake))

    errD = err_real + 0.5 * opt.metric * torch.mean(torch.abs((similarity(audio) - similarity(fea_real)))) \
    - k * (err_fake + opt.res*torch.mean(torch.abs((real_recons-real).detach() - (fake_recons-fake))))# \
           #. * opt.metric * torch.mean(torch.abs((similarity(audio) - similarity(fea_fake)))))
    errD.backward()
    optimizerD.step()

    netG.zero_grad()
    fake = netG(condition)
    fake_recons, fea = netD(fake, True)
    real_recons = netD(real)                           
    errG = torch.mean(torch.abs(fake_recons-fake)) 
    errRecon = opt.recon * torch.mean(torch.abs(real - fake))
    metric = opt.metric * torch.mean(torch.abs((similarity(audio) - similarity(fea))))
    errT = errG + errRecon + opt.res * torch.mean(torch.abs(real_recons-real - (fake_recons-fake)))+ metric
                                
    errT.backward()
    optimizerG.step()

    balance = (opt.gamma * err_real - err_fake).data[0]
    k = min(max(k + opt.lambda_k * balance,0),1)
    measure = err_real.data[0] + np.abs(balance)
    M_global.update(measure, real.size(0))
    ########### Logging #########
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_Recon: %.4f Loss_Res: %.4f Loss_Metric: %.4f Measure: %.4f K: %.4f LR: %.8f'
              % (iteration, opt.niter, 
                 errD.data[0], errG.data[0], errRecon.data[0], errT.data[0] - errG.data[0] - errRecon.data[0]-metric.data[0], metric.data[0], measure, k, optimizerD.param_groups[0]['lr']))
    sys.stdout.flush()
    trainLogger.write('%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % \
                        (iteration, errD.data[0], errG.data[0], errRecon.data[0], measure, M_global.avg, k, balance, errT.data[0] - errG.data[0] - errRecon.data[0]-metric.data[0], metric.data[0]))
    trainLogger.flush()
    ########### Learning Rate Decay #########
    optimizerD = adjust_learning_rate(optimizerD,iteration, opt.lrD)
    optimizerG = adjust_learning_rate(optimizerG,iteration, opt.lr)
    ########## Visualize #########
    if(iteration % 1000 == 0):
        vutils.save_image(fake.data,
                    '%s/fake_samples_iteration_%03d.png' % (opt.outf, iteration),
                    normalize=True)
        vutils.save_image(real.data,
                    '%s/real_samples_iteration_%03d.png' % (opt.outf, iteration),
                    normalize=True)
        vutils.save_image(real_recons.data,
                    '%s/real_recons_samples_iteration_%03d.png' % (opt.outf, iteration),
                    normalize=True)

    if(iteration % opt.save_step == 0):
        torch.save(netG.state_dict(), '%s/netG_%d.pth' % (opt.outf,iteration))
        torch.save(netD.state_dict(), '%s/netD_%d.pth' % (opt.outf,iteration))
        cpt_G = {'optim':optimizerG.state_dict(),'epoch':iteration}
        cpt_D = {'optim':optimizerD.state_dict(),'epoch':iteration}
        torch.save(cpt_G,'%s/netG_%d.tar' % (opt.outf,iteration))
        torch.save(cpt_D,'%s/netD_%d.tar' % (opt.outf,iteration))
        #torch.save(netG.state_dict(), '%s/netG_%d.pth' % (opt.outf,iteration))
        #torch.save(netD.state_dict(), '%s/netD_%d.pth' % (opt.outf,iteration))
