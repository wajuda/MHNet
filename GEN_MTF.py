import torch
import numpy as np
from scipy.io import loadmat
from Configs import train_cfg

filename = './mtf_wv3.mat'
MTF = loadmat(filename)
MTF = np.array(MTF['mtf_wv3'],dtype=np.float32)
MTF = torch.from_numpy(MTF).cuda()
MTF = MTF.reshape(train_cfg.channel, train_cfg.kernel_size, train_cfg.kernel_size)
MTF = MTF.unsqueeze(dim=0).repeat(train_cfg.batch_size,1,1,1)
print('模糊核大小：', MTF.shape)