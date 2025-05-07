import torch
import torch.nn as nn
import numpy as np
from astropy.visualization import ZScaleInterval
    
def get_mask(h, w, patch_h, patch_w, real_A, AB):
    loss_patch = []
    num_x = w//patch_w
    num_y = h//patch_h

    criterionMSE = nn.L1Loss()
    white_small = np.zeros((h, h), np.uint8)
    for x in range(num_x):
        for y in range(num_y):
            xy = real_A[0, 0, patch_h*y:patch_h*(y+1), patch_w*x:patch_w*(x+1)]
            xy_AB = AB[0, 0, patch_h*y:patch_h*(y+1), patch_w*x:patch_w*(x+1)]
            loss_AB = criterionMSE (xy, xy_AB)
            errMSE = loss_AB
            datanumber = torch.Tensor.cpu(errMSE.data)
            datanumber = datanumber.data.numpy()
            loss_patch.append(datanumber)
            if datanumber > 0.45:
                img_mask = np.full((patch_h,patch_w),1)
            else:
                img_mask = np.full((patch_h,patch_w),0)
            white_small[patch_h*x:patch_h*(x+1), patch_w*y:patch_w*(y+1)] = img_mask
    flipped_white = white_small[::-1, :]
    flipped_white = flipped_white*255
    return flipped_white, loss_patch

def linear_scale(image, min_out, max_out):
    min_in = np.min(image)
    max_in = np.max(image)
    scaled_image = (image - min_in) / (max_in - min_in) * (max_out - min_out) + min_out
    return scaled_image

def zscale_scale(image):
    interval = ZScaleInterval()
    scaled_image = interval(image)
    return scaled_image



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    