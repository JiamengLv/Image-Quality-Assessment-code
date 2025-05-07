from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from itertools import chain
from torch.autograd import Variable
from model.Generator import Generator
from utils.fitsFun import DATASET_fits
from utils.fitsFun import LoadSaveFits

parser = argparse.ArgumentParser(description='train pix2pix model')
parser.add_argument('--batchSize', type=int, default=1, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=150000, help='number of iterations to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay in network D, default=1e-4')
parser.add_argument('--cuda', default=False, action='store_true', help='enables cuda')
parser.add_argument('--outf', default='checkpoints/', help='folder to output model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dataPath', default='./dataset/train', help='image data')
parser.add_argument('--loadSize', type=int, default=500, help='scale image to this size')
parser.add_argument('--fineSize', type=int, default=500, help='random crop image to this size')
parser.add_argument('--flip', type=int, default=1, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--input_nc', type=int, default=1, help='channel number of input image')
parser.add_argument('--output_nc', type=int, default=1, help='channel number of output image')
parser.add_argument('--kernelSize', type=int, default=3, help='random crop kernel to this size')
parser.add_argument('--G_AB', default='', help='path to pre-trained G_AB')
parser.add_argument('--save_step', type=int, default=1000, help='save interval')
parser.add_argument('--log_step', type=int, default=100, help='log interval')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

opt.seed = random.randint(1, 10000)
print("Random Seed: ", opt.seed)
torch.manual_seed(opt.seed)
cudnn.benchmark = True

##########    dataset fits  ###########
datasetA = DATASET_fits(opt.dataPath, opt.fineSize)
loader_A = torch.utils.data.DataLoader(dataset=datasetA,
                                       batch_size=opt.batchSize,
                                       shuffle=True,
                                       num_workers=0)
loaderA = iter(loader_A)

############   MODEL   ###########
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

ndf = opt.ndf
ngf = opt.ngf

G_AB = Generator(opt.input_nc, opt.output_nc, opt.ngf)

if (opt.G_AB != ''):
    print('Warning! Loading pre-trained weights.')
    G_AB.load_state_dict(torch.load(opt.G_AB))
else:
    G_AB.apply(weights_init)

if (opt.cuda):
    G_AB.cuda()

###########   LOSS & OPTIMIZER   ##########
criterionMSE = nn.L1Loss()
optimizerG = torch.optim.Adam(chain(G_AB.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

############   GLOBAL VARIABLES   ###########
input_nc = opt.input_nc
output_nc = opt.output_nc
fineSize = opt.fineSize
batchSize = opt.batchSize
kernelSize = opt.kernelSize
real_A = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
AB = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
real_B = torch.FloatTensor(opt.batchSize, output_nc, fineSize, fineSize)

real_A = Variable(real_A)
real_B = Variable(real_B)
AB = Variable(AB)

if (opt.cuda):
    real_A = real_A.cuda()
    real_B = real_B.cuda()
    AB = AB.cuda()
    criterionMSE.cuda()

###########   Testing    ###########
def test(niter):
    loaderA = iter(loader_A)
    imgA = next(loaderA)
    AB = G_AB(imgA)
    LoadSaveFits.save_fit(AB.data, 'AB_%03d_' % niter, './out_picture/out_image_train/')
    LoadSaveFits.save_fit(imgA.data, 'realA_%03d_' % niter, './out_picture/out_image_train/')

###########   Training   ###########
def train(loaderA):
    G_AB.train()
    loss_least = float('inf')
    lossdata = []
    for iteration in range(1, opt.niter + 1):
        ###########   data  ###########
        try:
            imgA = next(loaderA)
        except StopIteration:
            loaderA = iter(loader_A)
            imgA = next(loaderA)

        ###########   model   ###########
        G_AB.zero_grad()
        AB = G_AB(imgA)

        ###########   loss   ###########
        errMSE = criterionMSE(AB, imgA)

        datanumber = torch.Tensor.cpu(errMSE.data)
        datanumber = datanumber.data.numpy()
        lossdata.append((datanumber.data).tolist())

        optimizerG.step()

        ###########   Logging   ############
        if (iteration % opt.log_step):
            print('[%d/%d] Loss_MSE: %.4f '
                  % (iteration, opt.niter, errMSE.data))


        ###########   Visualize  ###########
        if (iteration % opt.save_step == 0):
            test(iteration)
            filepath = './losstest.txt'
            plt.plot(range(len(lossdata)), lossdata)
            plt.xlabel('Iter')
            plt.ylabel('Loss')
            plt.title('Loss Trend')
            plt.savefig('loss_train.png')
            with open(filepath, 'w') as f:
                f.write(str(lossdata))

        if datanumber < loss_least:
            loss_least = datanumber
            torch.save(G_AB.state_dict(), '{}/model_latest.pth'.format(opt.outf))

if __name__=='__main__':
    train(loaderA)




