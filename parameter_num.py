from model import MHNet
from GEN_MTF import MTF
from Configs import train_cfg

model = MHNet(MTF_init=MTF,channel =  train_cfg.channel, s_iter = train_cfg.iter,
                 kernel_size = train_cfg.kernel_size,stride =4).cuda()
print(sum([param.nelement() for param in model.parameters()]))