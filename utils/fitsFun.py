import torchvision.transforms as transforms
import numpy as np
import torch
import os
from astropy.io import fits

# preprocessing
class LoadSaveFits:
    def __init__(self, path, img, name):
        self.path = path
        self.img = img
        self.name = name

    def norm(img):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # normalization
        img -= np.mean(img)  # take the mean
        img /= np.std(img)  # standardization
        img = np.array(img, dtype='float32')
        return img

    def norm2(img, z):
        for i in range(z):
            img[i] = (img[i] - np.min(img[i])) / (np.max(img[i]) - np.min(img[i]))  # normalization
            img[i] -= np.mean(img[i])  # take the mean
            img[i] /= np.std(img[i])  # standardization
            img[i] = np.array(img[i], dtype='float32')
        return img

    def read_fits(path):
        hdu = fits.open(path)
        img = hdu[0].data
        img = np.array(img, dtype=np.float32)
        hdu.close()
        
        return img

    def save_fit_cpu(img, name, path):
        if os.path.exists(path + name + '.fits'):
            os.remove(path + name + '.fits')
        grey = fits.PrimaryHDU(img)
        greyHDU = fits.HDUList([grey])
        greyHDU.writeto(path + name + '.fits')

    def save_fit(img, name, path):
        if torch.cuda.is_available():
            img = torch.Tensor.cpu(img)
            img = img.data.numpy()
            IMG = img[ 0, :, :]
        else:
            img = np.array(img)
        if os.path.exists(path + name + '.fits'):
            os.remove(path + name + '.fits')
        grey = fits.PrimaryHDU(img)
        greyHDU = fits.HDUList([grey])
        greyHDU.writeto(path + name + '.fits')
        
    def save_jpg(img, name, path):
        img = (img-img.min())/(img.max()-img.min())
        img = transforms.ToPILImage()(img.float())
        save_path = path + name + ".jpg"
        img.save(save_path)

# load data
class DATASET_fits():
    def __init__(self, dataPath='', fineSize=512):
        super(DATASET_fits, self).__init__()
        self.list = os.listdir(dataPath)
        self.list.sort()
        self.dataPath = dataPath
        self.fineSize = fineSize

    def __getitem__(self, index):
        path = os.path.join(self.dataPath, self.list[index])
        img = LoadSaveFits.read_fits(path)
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, dim=0)
        img = img.numpy()
        z, h, w = img.shape
        img = LoadSaveFits.norm2(img, z)
        img = torch.from_numpy(img)
       
        return img

    def __len__(self):
        return len(self.list)











