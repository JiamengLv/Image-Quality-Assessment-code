import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from utils.Tools import get_mask
from utils.Tools import linear_scale
from utils.Tools import zscale_scale
from torch.autograd import Variable
from model.Generator import Generator
from utils.fitsFun import DATASET_fits

parser = argparse.ArgumentParser(description='train pix2pix model')
parser.add_argument('--batchSize', type=int, default=1, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--patch_size', type=int, default=32)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='./output/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dataPath', default='./dataset/test/', help='path to training images')
parser.add_argument('--loadSize', type=int, default=512, help='scale image to this size')
parser.add_argument('--fineSize', type=int, default=512, help='random crop image to this size')
parser.add_argument('--flip', type=int, default=0, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--input_nc', type=int, default=1, help='channel number of input image')
parser.add_argument('--output_nc', type=int, default=1, help='channel number of output image')
parser.add_argument('--G_AB', default='./checkpoints/model_latest.pth', help='path to pre-trained G_AB')
parser.add_argument('--imgNum', type=int, default=2, help='image number')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 1)")


opt = parser.parse_args()
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
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

opt.seed = random.randint(1, 10000)
print("Random Seed: ", opt.seed)
torch.manual_seed(opt.seed)
cudnn.benchmark = True

##########   DATASET   ###########
listp = os.listdir(opt.dataPath)
listp.sort()
datasetA = DATASET_fits(opt.dataPath,opt.fineSize)
loader_A= torch.utils.data.DataLoader(dataset=datasetA,
                                      batch_size=opt.batchSize,
                                      shuffle=True,
                                      num_workers=0)
loaderA = iter(loader_A)

###########  MODEL   ###########
ndf = opt.ndf
ngf = opt.ngf
G_AB = Generator(opt.input_nc, opt.output_nc, opt.ngf)

if(opt.G_AB != ''):
    print('Warning! Loading pre-trained weights.')
    G_AB.load_state_dict(torch.load(opt.G_AB, map_location=torch.device('cpu')))
else:
    print('ERROR! G_AB must be provided!')

if(opt.cuda):
    G_AB.cuda()

###########   GLOBAL VARIABLES   ###########
input_nc = opt.input_nc
output_nc = opt.output_nc
fineSize = opt.fineSize

real_A = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
real_A = Variable(real_A)

if(opt.cuda):
    real_A = real_A.cuda()

###########   Testing    ###########
def test():
    for i in range(0,len(loaderA),1):
        loss_list = []
        imgA = next(loaderA)
        criterionMSE = nn.L1Loss()
        real_A.resize_(imgA[:,:,:,:].size()).copy_(imgA[:,: ,:,:])
        b, c, image_h, image_w = np.shape(real_A)
        white = np.zeros((image_h, image_w), np.uint8)
        num_y = image_h // opt.fineSize
        num_x = image_w // opt.fineSize
        for x in range(num_x):
            for y in range(num_y):
                xy = real_A[:, :, opt.fineSize * y:opt.fineSize * (y + 1),opt.fineSize * x:opt.fineSize * (x + 1)]
                AB = G_AB(xy)
                errMSE = criterionMSE(AB, xy)
                datanumber = torch.Tensor.cpu(errMSE.data)
                datanumber = datanumber.data.numpy()
                loss_list.append(datanumber)
                mask = get_mask(opt.fineSize, opt.fineSize, opt.patch_size, opt.patch_size, xy, AB)
                white[opt.fineSize * y:opt.fineSize * (y + 1), opt.fineSize * x:opt.fineSize * (x + 1)] = mask

        plt.rcParams['figure.figsize'] = [12, 12]

        plt.subplot(1, 2, 1)
        img = np.array(real_A, dtype=np.float32)
        img = linear_scale(img[0, 0, :, :], 0, 255)
        img = zscale_scale(img)
        plt.imshow(img)
        plt.title("original", fontsize=16)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(white)
        plt.title("mask", fontsize=16)
        plt.axis('off')
        plt.show()


if __name__=='__main__':
    test()
















