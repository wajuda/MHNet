# This is a pytorch version for the work of PanNet
# YW Jin, X Wu, LJ Deng(UESTC);
# 2020-09;

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init
import torch.nn.functional as F
import cv2

def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):  ## initialization for Conv2d
                variance_scaling_initializer(m.weight)  # method 1: initialization
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):  ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):  ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


############### torch.nn.functional.unfold ########################
def im2col(input, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2
    l = kernel_size
    s = stride
    # If a 4-tuple, uses(padding_left , padding_right , padding_top , padding_bottom)
    pad_data = torch.nn.ReflectionPad2d(padding=(pad, pad, pad, pad))
    input_padded = pad_data(input)
    col_input = torch.nn.functional.unfold(input_padded, kernel_size=(l, l), stride=(s, s))
    #print(col_input.shape)
    return col_input


def BC_im2col(input, channel, kernel_size, stride=1):
    x = input
    x_col = im2col(input=x, kernel_size=kernel_size, stride=stride)
    #print('hr展开：',x_col.shape)
    B, C_ks_ks, hw = x_col.size()
    x_col = x_col.reshape(B, channel, kernel_size * kernel_size, hw)
    #x_col = x_col.permute(0, 2, 1, 3)
    #x_col = x_col.reshape(B, kernel_size * kernel_size, channel * hw)
    #print('hr展开reshape:', x_col.shape)
    return x_col


################## Bacthblur ##################
#### 验证过是对的，但是输入的kernel是：B x feild_h x feild_w (或 feild_h x feild_w)
#### 打算通过stride来控制缩放尺度，试了一下好像行
class BatchBlur(object):
    def __init__(self, l=11):
        self.l = l
        if l % 2 == 1:
            self.pad = (l // 2, l // 2, l // 2, l // 2)
        else:
            self.pad = (l // 2, l // 2 - 1, l // 2, l // 2 - 1)
        # self.pad = nn.ZeroPad2d(l // 2)

    def __call__(self, input, kernel, stride=1):
        B, C, H, W = input.size()
        pad = F.pad(input, self.pad, mode='reflect')
        H_p, W_p = pad.size()[-2:]
        h = (H_p - self.l) // stride + 1
        w = (W_p - self.l) // stride + 1

        # if len(kernel.size()) == 2:
        #     input_CBHW = pad.view((C * B, 1, H_p, W_p))
        #     kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
        #     return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, h, w))
        # else:
        input_CBHW = pad.view((1, C * B, H_p, W_p))
        #kernel_var = (
        #    kernel.contiguous()
        #        .view((B, 1, self.l, self.l))
        #        .repeat(1, C, 1, 1)
       #         .view(())
       # )
        #kernel_var = kernel.view
        kernel_var  = kernel.view(B * C, 1, self.l, self.l)
        out = F.conv2d(input_CBHW, kernel_var, stride=stride, groups=B*C).view(
            (B, C, h, w))
        #print('卷积下采样：',out.shape)
        return out  # groups=B*C以后，相当于是逐channel


##############################################################

#################### Batch conv_transpose2d ##################
class BatchTranspose(object):
    def __init__(self, l=11):
        self.l = l
        self.pad = (self.l - 1) // 2

    def __call__(self, input, kernel, stride=1, output_padding=0):
        B, C, h, w = input.size()
        a = output_padding
        input_CBhw = input.view((1, B * C, h, w))
        H = (h - 1) * stride + self.l + a - 2 * self.pad
        W = (w - 1) * stride + self.l + a - 2 * self.pad
        #kernel = (kernel.contiguous()
        #          .view(B, 1, self.l, self.l)
        #          .repeat(1, C, 1, 1)
       #           .view(B * C, 1, self.l, self.l))
        kernel = kernel.view(B * C, 1, self.l, self.l)

        return F.conv_transpose2d(input_CBhw, kernel, stride=stride, padding=self.pad, output_padding=a,
                                  groups=B * C).view(B, C, H, W)


###################################################################

#################### Inverse of PixelShuffle #####################


#################### PixelShuffle #########################


# XNet Proximal operator
class MTF_ProNet(nn.Module):
    def __init__(self, channel):
        super(MTF_ProNet, self).__init__()
        self.channels = channel
        self.resx1 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            #    nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            #    nn.BatchNorm2d(self.channels),
            #    CALayer(self.channels),
        )
        self.resx2 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            #   nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            #   nn.BatchNorm2d(self.channels),
            #   CALayer(self.channels)
        )
        self.resx3 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            #   nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            #   nn.BatchNorm2d(self.channels),
            #   CALayer(self.channels),
        )
        self.resx4 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            #   nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            #   nn.BatchNorm2d(self.channels),
            #   CALayer(self.channels),
        )
        self.resx5 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1))

    def forward(self, input):
        x1 = F.relu(input + 0.1 * self.resx1(input))
        x2 = F.relu(x1 + 0.1 * self.resx2(x1))
        x3 = F.relu(x2 + 0.1 * self.resx3(x2))
        x4 = F.relu(x3 + 0.1 * self.resx4(x3))
        x5 = F.relu(input + 0.1 * self.resx5(x4))
        return x5


# KNet Proximal operator
class HR_ProNet(nn.Module):
    def __init__(self, channels):
        super(HR_ProNet, self).__init__()
        self.channels = channels

        self.resk1 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            #    nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            #    nn.BatchNorm2d(self.channels),
        )
        self.resk2 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            #   nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            #   nn.BatchNorm2d(self.channels),
        )
        self.resk3 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            #  nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            #   nn.BatchNorm2d(self.channels),
        )
        self.resk4 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            #   nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            #   nn.BatchNorm2d(self.channels),
        )
        self.resk5 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1))

    def forward(self, input):
        k1 = F.relu(input + 0.1 * self.resk1(input))
        k2 = F.relu(k1 + 0.1 * self.resk2(k1))
        k3 = F.relu(k2 + 0.1 * self.resk3(k2))
        k4 = F.relu(k3 + 0.1 * self.resk4(k3))
        k5 = F.relu(input + 0.1 * self.resk5(k4))
        return k5


###############################

class HRNet(nn.Module):
    def __init__(self, channel=8, kernel_size=11):
        super(HRNet, self).__init__()
        self.HR_ProNet = HR_ProNet(channels=channel)
        self.BatchBlur = BatchBlur(l=kernel_size)
        self.BatchTranspose = BatchTranspose(l=kernel_size)
        self.channels = channel

    def forward(self, hrms , ms, k, p, sf, eta):
        ##### Done 这里也考虑换成Bxhxw的kernel作为输入
        N, C, H, W = hrms.size()
        N_k, C_k, h_k, w_k = k.size()
        pad = (h_k - 1) // 2
        transpose_out_padding = (H + 2 * pad - h_k) % sf


        R = self.BatchBlur(input=hrms, kernel=k, stride=sf) - ms           #  N. C ,h,w
        G_h1 = self.BatchTranspose(input=R, kernel=k, stride=sf, output_padding=transpose_out_padding)   #  N. C ,H,W  hrms的误差第一项

        #####################   E_ones for uneven overlap ################
        R_ones = torch.ones_like(R)
        G_ones_h1 = self.BatchTranspose(input=R_ones, kernel=k, stride=sf, output_padding=transpose_out_padding)

        G_h1 = G_h1 / (G_ones_h1 + 1e-10)


        A = torch.ones(size=[N,1,C]) / self.channels      #   p = A* hrms   N.1.C
        A =A.cuda()
        G_h2 = torch.matmul(A.view(N,C,1), torch.matmul(A,hrms.view(N,C,H*W))-p.view(N,1,H*W) ).view(N,C,H,W)

        G_h = G_h1 + G_h2
        hrms_target = hrms - eta / 10 * G_h  # 近端输入  N, C, k ,k
        h_next = self.HR_ProNet(hrms_target)

        return h_next


class MTFNet(nn.Module):
    def __init__(self,channel, kernel_size):
        super(MTFNet, self).__init__()
        self.MTF_ProNet = MTF_ProNet(channel)
        self.BatchBlur = BatchBlur(l=kernel_size)

    def forward(self, hrms, ms, k, sf, gamma):
        # x: NxCxHxW
        # K^(t+1/2)
        N, C, H, W = hrms.size()
        N_ms, C_ms, h, w = ms.size()
        N_k, C_k, h_k, w_k, = k.size()

        hrms_fold = BC_im2col(input=hrms, channel=C, kernel_size=h_k, stride=sf)    #  N , C , k*2, hw

        R = (self.BatchBlur(input=hrms, kernel=k, stride=sf) - ms).view(N,  h * w, C)    #  N,  h * w, C  (误差)

        for i in range(C):
            if i == 0:
                G_K = (1 / ( h * w)) * torch.matmul(hrms_fold[:,i,:,:].squeeze(dim=1), R[:,:,i].unsqueeze(dim=2))
            else:
                G_k = (1 / ( h * w)) * torch.matmul(hrms_fold[:,i,:,:].squeeze(dim=1), R[:, :, i].unsqueeze(dim=2))
                G_K = torch.cat([G_K,G_k],dim=2)             #核的梯度  N, k*k, C


        k_target = k - gamma / 10 * G_K.view(N, C, h_k , w_k)    #近端输入  N, C, k ,k
        k_next = self.MTF_ProNet(k_target)

        ######### K_est normalize F.relu #############
        # K = F.relu(K + 1e-5)
        k_next = F.relu(k_next)
        k_next= k_next / torch.sum(k_next, dim=[2, 3], keepdim=True)

        return k_next



# main Network architecture

class MHNet(nn.Module):
    def __init__(self, MTF_init,channel = 8, s_iter = 19,
                 kernel_size = 5,stride =4):
        super(MHNet, self).__init__()
        self.iter = s_iter
        self.ksize = kernel_size
        # self.K_Net = KNet()
        # self.X_Net = XNet()
        self.channels = channel
        self.stride = stride
        self.HR_stage = self.Make_HRNet(self.iter)
        self.MTF_stage = self.Make_MTFNet(self.iter)
        self.mtf = MTF_init

        # Auxiliary Variable
        # self.AV_X_ker0 = AV_X_ker_def.expand(32, 3, -1, -1)
        # self.AV_X_ker = nn.Parameter(self.AV_X_ker0, requires_grad=True)
        # self.kernel_map0 = nn.Parameter(torch.load(ker_auxi_path), requires_grad=False)  # [1, 441, 10] or [24, 21, 21]
        # self.kernel_map = nn.Parameter(self.kernel_map0, requires_grad=True)
        #######################
        self.gamma_k = torch.Tensor([1.0])
        self.eta_x = torch.Tensor([1.0])
        self.gamma_stage = self.Make_Para(self.iter, self.gamma_k)
        self.eta_stage = self.Make_Para(self.iter, self.eta_x)


    def Make_HRNet(self, iters):
        layers = []
        for i in range(iters):
            layers.append(HRNet(self.channels, self.ksize))
        return nn.Sequential(*layers)

    def Make_MTFNet(self, iters):
        layers = []
        for i in range(iters):
            layers.append(MTFNet(self.channels, self.ksize))
        return nn.Sequential(*layers)

    def gaussian_kernel_2d(self, kernel_size, sigma):
        kx = cv2.getGaussianKernel(kernel_size, sigma)
        ky = cv2.getGaussianKernel(kernel_size, sigma)
        return np.multiply(kx, np.transpose(ky))

    def Make_Para(self, iters, para):
        para_dimunsq = para.unsqueeze(dim=0)
        para_expand = para_dimunsq.expand(iters, -1)
        para = nn.Parameter(data=para_expand, requires_grad=True)
        return para

    # def forward(self, x, sf):
    def forward(self, ms, p):
        HRMS = []       #output of HRMS in every stage
        MTF = []        #MTF in every stage

        # initialization and preparation calculation
        B, C, h, w = ms.shape
        hrms= nn.functional.interpolate(input=ms, scale_factor=4, mode='bicubic', align_corners=False)
        #mtf= torch.as_tensor(self.gaussian_kernel_2d(kernel_size=self.ksize, sigma=1.0), dtype=torch.float32)
        mtf = self.mtf
        #mtf = mtf.unsqueeze(dim=0).unsqueeze(dim=1).repeat(B, C, 1, 1).cuda()

        for i in range(self.iter):
            # print("################ stage:{} ##################".format(i))
            mtf = self.MTF_stage[i](hrms, ms, mtf, self.stride,self.gamma_stage[i])

            hrms = self.HR_stage[i](hrms, ms, mtf,  p, self.stride,  self.eta_stage[i])

            HRMS.append(hrms)
            MTF.append(mtf)

        # return [srs_init, srs, kernel]
        return [HRMS,MTF]



def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x / 10 * 1.28

    variance_scaling(tensor)

    return tensor
