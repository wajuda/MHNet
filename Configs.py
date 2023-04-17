from easydict import EasyDict as edict
train_cfg = edict({
'lr' : 0.001,
'epochs': 450 ,
'ckpt' : 50,
'batch_size' : 16,
'gamma':3/4,
'weight_decay': 1e-8,
'model_path' : "Weights/.pth",
'iter' :2,
'channel':8,
'kernel_size' : 41,
'train_logs': './train_logs/',
'model_version': 1
})
a=[]
b=[]
for i in range(train_cfg.iter):
    if i<(train_cfg.iter-1):
        a.append(0.1)
        b.append(0.1)
    else:
        a.append(1)
        b.append(1)
train_cfg['a'] =a
train_cfg['b'] =b   #loss  系数
