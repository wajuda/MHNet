# This is a pytorch version for the work of PanNet
# YW Jin, X Wu, LJ Deng(UESTC);
# 2020-09;

###################################################################
# ------------------- Sub-Functions (will be used) -------------------
###################################################################
'''def get_edge(data):  # get high-frequency
    rs = np.zeros_like(data)
    if len(rs.shape) == 3:
        for i in range(data.shape[2]):
            rs[:, :, i] = data[:, :, i] - cv2.boxFilter(data[:, :, i], -1, (5, 5))
    else:
        rs = data - cv2.boxFilter(data, -1, (5, 5))
    return rs'''
'''''
def load_set(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    # tensor type:

    ms= torch.from_numpy(data['ms'] / 2047.0).permute(2, 0, 1)  # CxHxW= 8x64x64
    pan = torch.from_numpy(data['pan'] / 2047.0)   # HxW = 256x256

    return ms, pan
'''
'''
def get_edge(data):  # for training: HxWxC
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs


def Dataset_Pro(filepath):

        data = sio.loadmat(file_path)  # HxWxC=256x256x8

        # tensor type:

        ms =(data['ms'] / 2047.0).transpose(2, 0, 1)  # CxHxW= 8x64x64
        ms=np.expand_dims(ms,axis=0)
        ms1 = np.array(ms.transpose(0, 2, 3, 1), dtype=np.float32)  # NxHxWxC
        ms_hp= get_edge(ms1)
        ms_hp = torch.from_numpy(ms_hp).permute(0, 3, 1, 2)


        lms = torch.from_numpy(data['lms'] / 2047.0).permute(2, 0, 1)  # CxHxW= 8x64x64
        lms = lms.unsqueeze(dim=0).float()



        pan =(data['pan'] / 2047.0)  # HxW = 256x256
        pan =np.expand_dims(pan, axis=0)
        pan = np.expand_dims(pan, axis=0)
        pan = np.array(pan.transpose(0, 2, 3, 1), dtype=np.float32)# / 2047.  # NxHxWx1
        pan1 = np.squeeze(pan, axis=3)  # NxHxW
        pan_hp = get_edge(pan1)   # NxHxW
        pan_hp = np.expand_dims(pan_hp, axis=3)   # NxHxWx1
        pan_hp = torch.from_numpy(pan_hp).permute(0, 3, 1, 2) # Nx1xHxW:
        return ms_hp, pan_hp, lms
###################################################################
# ------------------- Main Test (Run second) -------------------
###################################################################
ckpt = "Weights/pannet_small_450.pth"   # chose model

def test(file_path):
    ms, pan,lms = Dataset_Pro(file_path)

    model = EfficientPan().cuda().eval()   # fixed, important!

    weight = torch.load(ckpt)  # load Weights!
    model.load_state_dict(weight) # fixed

    with torch.no_grad():

        x1, x2, x3 = ms, pan, lms    # read data: CxHxW (numpy type)
        print(x1.shape)
        x1=x1.cuda()
        x2=x2.cuda()
        x3= x3.cuda()
        #x1 = x1.cuda().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))

        #x2 = x2.cuda().unsqueeze(dim=0).unsqueeze(dim=1).float()  # convert to tensor type: 1x1xHxW

        sr = model(x1, x2, x3)  # tensor type: CxHxW
             # tensor type: CxHxW

        # convert to numpy type with permute and squeeze: HxWxC (go to cpu for easy saving)
        sr = torch.squeeze(sr).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC

        print(sr.shape)
        save_name = os.path.join("test_results", "pannetsmall_new_data6_EfficientPan.mat")  # fixed! save as .mat format that will used in Matlab!
        sio.savemat(save_name, {'pannetsmall_new_data6_EfficientPan': sr})  # fixed!

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':
    file_path = "test_data/new_data6.mat"
    test(file_path)   # recall test function
'''
a= [1,2]
print(a)
print(list(set(a)-set([1])))

