# This is a pytorch version for the work of PanNet
# YW Jin, X Wu, LJ Deng(UESTC);
# 2020-09;

import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import Dataset_Pro
from model import MHNet #, summaries
import numpy as np
from scipy.io import loadmat
from torch.utils.tensorboard import SummaryWriter
from Configs import train_cfg
from GEN_MTF import MTF

###################################################################
# ------------------- Pre-Define Part----------------------
###################################################################
# ============= 1) Pre-Define =================== #
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True
cudnn.benchmark = False


# ============= 3) Load Model + Loss + Optimizer + Learn_rate_update ==========#
model = MHNet(MTF_init=MTF, channel = train_cfg.channel, s_iter = train_cfg.iter,
                 kernel_size = train_cfg.kernel_size,stride =4).cuda()
if os.path.isfile(train_cfg.model_path):
    model.load_state_dict(torch.load(train_cfg.model_path))   ## Load the pretrained Encoder
    print('Efficient is Successfully Loaded from %s' % (train_cfg.model_path))


criterion = nn.MSELoss(size_average=True).cuda()  ## Define the Loss function

optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)   ## optimizer 1: Adam
#lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)   # learning-rate update

#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-7)  ## optimizer 2: SGD
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=train_cfg.gamma)  # learning-rate update: lr = lr* 1/gamma for each step_size = 180



writer = SummaryWriter(train_cfg.train_logs)    ## Tensorboard_show: case 2

def save_checkpoint(model, epoch):  # save model function
    model_out_path = 'Weights' + '/' +train_cfg.model_version+ "_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)

###################################################################
# ------------------- Main Train (Run second)----------------------
###################################################################
def train(training_data_loader, validate_data_loader,start_epoch=0):
    print('Start training...')
    # epoch 450, 450*550 / 2 = 123750 / 8806 = 14/per imgae

    for epoch in range(start_epoch, train_cfg.epochs, 1):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):

            gt, ms, pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()

            optimizer.zero_grad()  # fixed

            [pre,kernel] = model(ms, pan)
            loss = 0# call model
            for i in range(train_cfg.iter):
                loss = loss + train_cfg.a[i]*criterion(pre[i], gt) +  train_cfg.b[i]*criterion(kernel[i], MTF)

            #loss = criterion(pre, gt)  # compute loss
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

            loss.backward()  # fixed
            optimizer.step()
            #lr_scheduler.step()# fixed
            #print('lr:{:.4E}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
            #for name, layer in model.named_parameters():
                # writer.add_histogram('torch/'+name + '_grad_weight_decay', layer.grad, epoch*iteration)
                #writer.add_histogram('net/'+name + '_data_weight_decay', layer, epoch*iteration)

        lr_scheduler.step()  # if update_lr, activate here!

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        writer.add_scalar('mse_loss/t_loss', t_loss, epoch)  # write to tensorboard to check
        print('Epoch: {}/{} training loss: {:.7f}'.format(train_cfg.epochs, epoch, t_loss))  # print loss for each epoch

        if epoch % train_cfg.ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)



        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():  # fixed
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, ms, pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()

                [pre, kernel] = model(ms, pan)
                loss = 0  # call model
                for i in range(train_cfg.iter):
                    loss = loss + train_cfg.a[i]*criterion(pre[i], gt) +  train_cfg.b[i]*criterion(kernel[i], MTF)
                epoch_val_loss.append(loss.item())

        if epoch % 10 == 0:
            v_loss = np.nanmean(np.array(epoch_val_loss))
            writer.add_scalar('val/v_loss', v_loss, epoch)
            print('             validate loss: {:.7f}'.format(v_loss))

    writer.close()  # close tensorboard


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    train_set = Dataset_Pro('../EfficientPan/training_data/train_small.h5')  # creat data for training
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=train_cfg.batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = Dataset_Pro('../EfficientPan/training_data/valid_small.h5')  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=train_cfg.batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    train(training_data_loader, validate_data_loader)  # call train function (call: Line 66)
