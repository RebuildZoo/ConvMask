from ctypes import c_void_p
import os, glob
import numpy as np 
import cv2 
import pickle
from enum import Enum
import torch

class SingleHandDataName(Enum):
    Freihand = 1
    STB = 2
    HO3D_v3 = 3
    DexYCB = 4

SingleHandDataDirs = {
    SingleHandDataName.Freihand: r"\\105.1.1.1\Hand\Hand3D\FreiHand\exported_img2mask", 
    SingleHandDataName.STB: r"\\105.1.1.1\Hand\Hand3D\STB\exported_img2mask", 
    SingleHandDataName.HO3D_v3: r"\\105.1.1.1\Hand\Hand3D\HO3D_v3\exported_img2mask", 
    SingleHandDataName.DexYCB: r"\\105.1.1.1\Hand\HO&HH\DexYCB\exported_img2mask", 

}

class singleHandSegDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name_list: list(SingleHandDataName), 
        image_netin_size = 256, mask_netout_sizes = [16, 32, 64]):
        self.image_netin_size = image_netin_size
        self.mask_netout_sizes = mask_netout_sizes

        self.pkl_absfilename_Lst = []
        for ds in dataset_name_list:
            cur_namelist = glob.glob(os.path.join(SingleHandDataDirs[ds], "*.pkl"))
            self.pkl_absfilename_Lst += cur_namelist
            print('loading %s #file  = %d '%(ds.name, len(self.pkl_absfilename_Lst)))

    def __len__(self):
        return len(self.pkl_absfilename_Lst)


    def __getitem__(self, index):
        data_filename = self.pkl_absfilename_Lst[index]
        data_Dic = pickle.load(open(data_filename, 'rb'), encoding='latin1')

        # crop the full image as net in size (256, 256, 3), (256, 256)
        bbx_top = 0; bbx_bot = data_Dic['mask'].shape[0]
        bbx_left = 0; bbx_right = data_Dic['mask'].shape[1]
        mask_inds = np.where(data_Dic['mask']>0)
        if len(mask_inds[0]) > 15*15:
            # mini region
            # bbx_top, bbx_bot = mask_inds[0].min(), mask_inds[0].max()
            # bbx_left, bbx_right = mask_inds[1].min(), mask_inds[1].max()
            loose_term = 0.2
            bbx_top = int(max((1-loose_term) * mask_inds[0].min(), 0))
            bbx_bot = int(min((1+0.8*loose_term) * mask_inds[0].max(), bbx_bot))

            bbx_left = int(max((1-loose_term) * mask_inds[1].min(), 0))
            bbx_right = int(min((1+0.8*loose_term) * mask_inds[1].max(), bbx_right))
     
        img_Arr = data_Dic['image'].astype(np.float32) / 255.0 # range [0, 1]
        mask_Arr = data_Dic['mask'].astype(np.float32) / 255.0 # range [0, 1]
        
        imgRoI_Arr = img_Arr[bbx_top:bbx_bot, bbx_left:bbx_right]
        maskRoI_Arr = mask_Arr[bbx_top:bbx_bot, bbx_left:bbx_right]
        to_net_scale = self.image_netin_size/ max(imgRoI_Arr.shape[0], imgRoI_Arr.shape[1])
        imgRoI_Arr = cv2.resize(imgRoI_Arr, dsize=None, fx = to_net_scale, fy = to_net_scale)
        maskRoI_Arr = cv2.resize(maskRoI_Arr, dsize=None, fx = to_net_scale, fy = to_net_scale)
        
        imgRoI_Arr = cv2.copyMakeBorder(imgRoI_Arr, 0, self.image_netin_size - imgRoI_Arr.shape[0], 
        0, self.image_netin_size - imgRoI_Arr.shape[1], cv2.BORDER_REPLICATE) # BORDER_CONSTANT, value= [0,0,0]
        maskRoI_Arr = cv2.copyMakeBorder(maskRoI_Arr, 0, self.image_netin_size - maskRoI_Arr.shape[0], 
        0, self.image_netin_size - maskRoI_Arr.shape[1], cv2.BORDER_CONSTANT, value=0.0)
        
        image_Tsor = torch.from_numpy(imgRoI_Arr - 0.5) # data range: (-0.5, 0.5)
        
        mask_Tsor_list = []
        for out_size in self.mask_netout_sizes:
            resize_maskRoI_Arr = cv2.resize(maskRoI_Arr, dsize=(out_size, out_size))
            resize_mask_Tsor = torch.from_numpy(np.stack([resize_maskRoI_Arr, np.zeros_like(resize_maskRoI_Arr)], axis=-1)) # default to be left; 
            if data_Dic['right']:
                resize_mask_Tsor = torch.from_numpy(np.stack([np.zeros_like(resize_maskRoI_Arr), resize_maskRoI_Arr], axis=-1))
            mask_Tsor_list.append(resize_mask_Tsor.permute(2,0,1))
        # permute(2,0,1) : HWC (012) -> CHW(201)
        return [image_Tsor.permute(2,0,1)] + mask_Tsor_list

if __name__ == "__main__":
    # SingleHandDataName.Freihand,
    gm_dataset = singleHandSegDataset([SingleHandDataName.Freihand])

    gm_trainloader = torch.utils.data.DataLoader(
        dataset = gm_dataset, 
        batch_size = 64,
        shuffle = False,
        num_workers = 1, 
        pin_memory = True, 
    )


    for iter_idx, datapac_i in enumerate(gm_trainloader):
        print(datapac_i[0].shape,  "->", datapac_i[1].shape, datapac_i[2].shape, datapac_i[3].shape)