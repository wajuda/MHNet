import torch.utils.data as data
import torch
import h5py
import numpy as np




class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3=8806x8x64x64

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047.
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:




        ms1 = data["ms"][...]  # NxCxHxW=0,1,2,3
        ms1 = np.array(ms1, dtype=np.float32) / 2047.  # NxHxWxC

        self.ms = torch.from_numpy(ms1)  # NxCxHxW:

        pan1 = data["pan"][...]  # NxCxHxW=0,1,2,3
        pan1 = np.array(pan1, dtype=np.float32) / 2047.  # NxHxWxC

        self.pan = torch.from_numpy(pan1)  # NxCxHxW:

    #####必要函数
    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.ms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]
