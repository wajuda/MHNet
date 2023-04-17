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
                # try:
                #     import tensorflow as tf
                #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
                #     m.weight.data = tensor.eval()
                # except:
                #     print("try error, run variance_scaling_initializer")
                # variance_scaling_initializer(m.weight)
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




import sys
#import utils as util
import matplotlib.pyplot as plt


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

    def forward(self, x , y, k, p, sf, eta):
        ##### Done 这里也考虑换成Bxhxw的kernel作为输入
        N, C, H, W = x.size()
        k_B, k_C, k_h, k_w = k.size()
        pad = (k_h - 1) // 2
        transpose_out_padding = (H + 2 * pad - k_h) % sf
        E = self.BatchBlur(input=x, kernel=k, stride=sf) - y
        # print('E.min():{}, E.max():{}'.format(E.min(), E.max()))
        #print('HR残差：',E.shape)
        G_t = self.BatchTranspose(input=E, kernel=k, stride=sf, output_padding=transpose_out_padding)
        #print('HR梯度第一项：', G_t.shape)
        # print('G_t.max:{}, G_t.min:{}, G_t.shape():{}'.format(G_t.max(), G_t.min(), G_t.shape))
        #####################   E_ones for uneven overlap ################
        E_ones = torch.ones_like(E)
        G_ones_t = self.BatchTranspose(input=E_ones, kernel=k, stride=sf, output_padding=transpose_out_padding)
        # print('G_ones_t.max():{}, G_ones_t.min():{}'.format(G_ones_t.max(), G_ones_t.min()))
        # print("k.max():{}, k.min():{}".format(k.max(), k.min()))
        G_t = G_t / (G_ones_t + 1e-10)

        #R = torch.ones_like(x) / self.channels
        R = torch.ones(size=[N,1,C]) / self.channels
        R =R.cuda()
        #print(R)
        #print((p).shape)
        G_t = G_t + torch.matmul(R.view(N,C,1), torch.matmul(R,x.view(N,C,H*W))-p.view(N,1,H*W) ).view(N,C,H,W)
        #print('HR梯度第二项', .shape)
        HR = self.HR_ProNet(G_t)
        return HR


class MTFNet(nn.Module):
    def __init__(self,channel, kernel_size):
        super(MTFNet, self).__init__()
        self.MTF_ProNet = MTF_ProNet(channel)
        self.BatchBlur = BatchBlur(l=kernel_size)

    def forward(self, x, y, k, sf, gamma):
        # x: NxCxHxW
        # K^(t+1/2)
        N, C, H, W = x.size()
        N_y, C_y, h, w = y.size()
        k_B, k_C, k_h, k_w, = k.size()
        #pad = (k_h - 1) // 2
        x_col_trans = BC_im2col(input=x, channel=C, kernel_size=k_h, stride=sf)
        #print(x_col_trans.shape)# x_col_trans:(B, kxk, Cxhxw)
        # x_col = x_col_trans.permute(0,2,1)  # permute(0,2,1) x_col:(B, Cxhxw, kxk)
        # k = k.view(k_B, k_h*k_w, 1)
        # y_t = y.view(N_y, C_y*h*w, 1)
        # print('x_col.shape:', x_col.shape)
        # print('k.shape:', k.shape)
        # print('torch.matmul(x_col, k).shape:', torch.matmul(x_col, k).shape)
        # print('y_t.shape:', y_t.shape)
        # R = torch.matmul(x_col, k.view(k_B, k_h*k_w,1)) - y_t    # R:(B, Cxhxw, 1)
        #print(x.shape)
        #print(k.shape)
        R = (self.BatchBlur(input=x, kernel=k, stride=sf) - y).view(N_y,  h * w, C_y)
        #print('分channel更新的:',x_col_trans[:,0,:,:].squeeze(dim=1).shape)
        for i in range(C_y):
            if i == 0:
                G_K = (1 / ( h * w)) * torch.matmul(x_col_trans[:,i,:,:].squeeze(dim=1), R[:,:,i].unsqueeze(dim=2))
            else:
                G_k = (1 / ( h * w)) * torch.matmul(x_col_trans[:,i,:,:].squeeze(dim=1), R[:, :, i].unsqueeze(dim=2))
                G_K = torch.cat([G_K,G_k],dim=2)
        #G_K = (1 / (C_y * h * w)) * torch.matmul(x_col_trans, R)  # G_K:(B, kxk, 1)

        G_K = k - gamma / 10 * G_K.view(N, k_C, k_h , k_w)
        # G_K = torch.unsqueeze(G_K.reshape(k_B, k_h, k_w), dim=1).repeat(1, 3, 1, 1)
        # G_K = G_K.reshapre(k_B, 1 ,k_h, k_2)
        #print('核的梯度更新：',G_K.shape)
        # C_auxi = torch.cat((G_K, AV_k), dim=1)  # AV_k is auxiliary variable
        # C_auxi = torch.cat((torch.unsqueeze(k.reshape(k_B, k_h, k_w), dim=1), AV_k), dim=1)   # only Prox
        K = self.MTF_ProNet(G_K)  # K^(t+1/2)
        # K_temp1 = torch.mean(K_temp1_auxi[:, :3, :, :], dim=1, keepdim=True)
        # K_temp1 = K_temp1_auxi[:, :1, :, :]
        # AV_k = K_temp1_auxi[:, 3:, :, :]

        ################ K_est normalize projection #################
        # K_temp1 = torch.squeeze(K_temp1).reshape(k_B, k_h*k_w, 1)

        # # K^(t+1)
        # I = torch.ones((k_B, k_h*k_w, 1)).cuda()
        # I_trans = I.permute(0,2,1)
        # K_temp2 = (torch.matmul(I_trans, K_temp1) - 1) / (k_h * k_w)
        # # print('I.shape:', I.shape)
        # # print('K_temp2.shape:', K_temp2.shape)
        # K_est = K_temp1 - torch.mul(K_temp2, I)
        # K_est = K_est.reshape(k_B, k_h, k_w)

        ########### K_est normalize K_est - K_est.min() / K_est.max() - K_est.min() #################
        # K_est_temp = torch.zeros_like(K_temp1)
        # for i in range(len(K_temp1)):
        #     K_est_temp[i] = ((K_temp1[i] - K_temp1[i].min()) / (K_temp1[i].max() - K_temp1[i].min())) / ((K_temp1[i] - K_temp1[i].min()) / (K_temp1[i].max() - K_temp1[i].min())).sum()
        #     # K_est_temp[i] = K_est_temp[i] / K_est_temp[i].sum()
        # K_est = K_est_temp.squeeze(dim=1)
        ########## K_est normalize torch.clamp ##############
        # K_est_temp = torch.clamp(K_temp1, 0, 0.999)
        # K_est = (K_est_temp / torch.sum(K_est_temp, dim=[2, 3], keepdim=True)).squeeze(dim=1)

        ######### K_est normalize F.relu #############
        # K = F.relu(K + 1e-5)
        K = F.relu(K)
        # K_est = (K_est_temp / torch.sum(K_est_temp, dim=[2, 3], keepdim=True)).squeeze(dim=1)
        MTF = K / torch.sum(K, dim=[2, 3], keepdim=True)
        #print('核的更新：', MTF.shape)
        return MTF


######################################

##### Auxiliary Variable's kernel ######
#AV_X_ker_def = (torch.FloatTensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])).unsqueeze(dim=0).unsqueeze(
#    dim=0)


# AV_X_ker_def = (torch.FloatTensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]])).unsqueeze(dim=0).unsqueeze(dim=0)

# main Network architecture

class MHNet(nn.Module):
    def __init__(self, channel = 8, s_iter = 19,
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

        # self.upscale = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
        #     nn.PixelShuffle(2),
        #     nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        # )

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
    def forward(self, x, p):
        srs = []
        kernel = []
        # srs_init = []

        B, C, h, w = x.shape
        # initialization and preparation calculation
        HR = nn.functional.interpolate(input=x, scale_factor=4, mode='bicubic', align_corners=False)
        #print(HR.shape)
        # X_est = self.upscale(x)
        # srs_init.append(X_est)

        MTF = torch.as_tensor(self.gaussian_kernel_2d(kernel_size=self.ksize, sigma=1.0), dtype=torch.float32)

        # k_est = self.Make_kernel_ini(h_k=self.ksize, w_k=self.ksize, ratio=sf, mode='bicubic')
        # print('Init_k_est.min:{}, Init_k_est.max:{}, Init_sum_k:{}'.format(torch.min(k_est), torch.max(k_est), torch.sum(k_est)))
        MTF = MTF.unsqueeze(dim=0).unsqueeze(dim=1).repeat(B, C, 1, 1).cuda()
        #print(MTF.shape)
        # print('Init_k_est.min:{}, Init_k_est.max:{}, Init_sum_k:{}'.format(torch.min(k_est), torch.max(k_est), torch.sum(k_est)))
        ##### Done #######
        # AV_X = F.conv2d(X_est, self.AV_X_ker, stride=1, padding=1)  # Auxiliary Variable
        # AV_X = F.conv2d(x, self.AV_X_ker, stride=1, padding=1)  # Auxiliary Variable
        # AV_k = self.kernel_map.permute(1,0).reshape(10, self.ksize, self.ksize).unsqueeze(dim=0)
        # AV_k = self.kernel_map.unsqueeze(dim=0)
        # AV_k = AV_k.repeat(B, 1, 1, 1)

        for i in range(self.iter):
            # print("################ stage:{} ##################".format(i))
            MTF = self.MTF_stage[i](HR, x, MTF, self.stride,self.gamma_stage[i])
            # k_est = k
            # print("k_est.shape in loop:{}".format(k_est.shape))
            # k_est = k
            HR = self.HR_stage[i](HR, x, MTF, p, self.stride,  self.eta_stage[i])
            # print('stage:{}, gamma:{}'.format(i, self.gamma_stage[i]))
            # print('stage:{}, eta:{}'.format(i, self.eta_stage[i]))
            # print('k_est.min:{}, k_est.max:{}, k_est.sum:{}'.format(torch.min(k_est), torch.max(k_est), torch.sum(k_est)))
            # print('X_est.min:{}, X_est.max:{}'.format(torch.min(X_est), torch.max(X_est)))
            # print("kernel shape:{}".format(k_est.shape))
            srs.append(HR)
            kernel.append(MTF)

        # return [srs_init, srs, kernel]
        return [srs, kernel]


# -------------Initialization----------------------------------------


# -------------ResNet Block (One)----------------------------------------

# -----------------------------------------------------


# ----------------- End-Main-Part ------------------------------------
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
