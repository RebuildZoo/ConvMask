import torch.nn as nn
import torch.nn.functional as F

import _ProcConfigger as ut_cfg
import _ProcLogger as ut_log
from SegNet import *
from SegDataLoader import * 

PROJ_ROOT = os.getcwd()
to_cpu = lambda tensor: tensor.detach().cpu().numpy()
list_avg = lambda num_Lst: sum(num_Lst) / len(num_Lst)

class train_config(ut_cfg.ProcConfig):
    def __init__(self):
        super(train_config, self).__init__(saving_id = "HandLRSegNet",
            pBs = 32, pWn = 8, p_force_cpu = False)
        
        self.total_epoch = 1000
        self.image_netin_size = 256
        self.mask_netout_sizes = [16, 32, 64]
        self.method_init = ut_cfg.InitMode.NORM # "xavier", "kaiming", "preTrain", "norm"

        self.log_root_path = self.check_path_valid(os.path.join(PROJ_ROOT, "outputs", "SegLR_220121log")) # save logs and checkpoints. 
        self.checkpoints_root_path = self.check_path_valid(os.path.join(self.log_root_path, self.saving_id + "_checkpoints"))

        self.opt_baseLr = 1e-5
        self.opt_weightdecay = 5e-4

    def create_dataset(self, istrain):
        dataset_names = None
        if istrain: 
            dataset_names = [SingleHandDataName.Freihand] # SingleHandDataName.STB
        load_full_dataset = singleHandSegDataset([SingleHandDataName.Freihand, SingleHandDataName.STB], 
            image_netin_size = self.image_netin_size, mask_netout_sizes=self.mask_netout_sizes)
        return load_full_dataset

if __name__ == "__main__":

    gm_cfg = train_config()
    gm_log = ut_log.ProcLog(gm_cfg.log_root_path, gm_cfg.saving_id)
    gm_dataset = gm_cfg.create_dataset(istrain = True)
    gm_trainloader = torch.utils.data.DataLoader(dataset = gm_dataset,  batch_size = gm_cfg.ld_batchsize, shuffle = True, num_workers = gm_cfg.ld_workers)
    total_iter = len(gm_trainloader)

    gm_net = HandLRSegNet()
    gm_cfg.init_model(gm_net, gm_cfg.method_init, isTrain=True)
    gm_optimizer = torch.optim.SGD(gm_net.parameters(), lr = gm_cfg.opt_baseLr, momentum=0.9, weight_decay = gm_cfg.opt_weightdecay)

    gm_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = gm_optimizer,
        mode='min',
        factor=0.8, patience=5, verbose=True, 
        threshold=0.0001, threshold_mode='rel', 
        cooldown=0, min_lr=0, eps=1e-08
    )

    gm_log.summarize_netarch(gm_net)
    gm_log.summarize_config(gm_cfg)

    loss_an_epoch_Lst = []
    print("Train_Begin".center(40, "*"))
    for epoch_i in range(gm_cfg.total_epoch):
        gm_net.train()
        for iter_idx, datapac_i in enumerate(gm_trainloader):
            for j, data_j in enumerate(datapac_i):
                datapac_i[j] = data_j.to(gm_cfg.device)

            img_Tsor_bacth_i, *gtMask_Tsor_bacth_i_Lst = datapac_i
            
            gm_optimizer.zero_grad()  ######################################

            predMask_Tsor_bacth_i_Lst = gm_net(img_Tsor_bacth_i)

            assert len(gtMask_Tsor_bacth_i_Lst) == len(predMask_Tsor_bacth_i_Lst), "unmatched gt and pred"

            mask_iter_loss = F.mse_loss(predMask_Tsor_bacth_i_Lst[0], gtMask_Tsor_bacth_i_Lst[0], reduction='sum') + \
            F.mse_loss(predMask_Tsor_bacth_i_Lst[1], gtMask_Tsor_bacth_i_Lst[1], reduction='sum') + \
            F.mse_loss(predMask_Tsor_bacth_i_Lst[2], gtMask_Tsor_bacth_i_Lst[2], reduction='sum')
            mask_iter_loss.backward() ######################################

            if iter_idx % 100 == 1:
                gm_log.log_scalars_singleline([
                        ["\t epoch", epoch_i], 
                        ["Iter%04d|"%(iter_idx), len(gm_trainloader),], 
                        ["time", gm_cfg.get_localtime()], 
                        ["train_mask", mask_iter_loss.item()], 
                    ])
            loss_an_epoch_Lst.append(mask_iter_loss.item())

            gm_optimizer.step()  ######################################

        # end an epoch
        train_loss = list_avg(loss_an_epoch_Lst)
        gm_scheduler.step(train_loss)
        gm_log.log_scalars_singleline([
        ["epoch", epoch_i], 
        ["time", gm_cfg.get_localtime()], 
        ["train_mask", train_loss], 
        ])

        gm_log.board_scalars_singlechart("SR training loss", 
        {
            "train_mask": train_loss, 
        },epoch_i
        )

        loss_an_epoch_Lst.clear() # prepare for next epoch
        gm_cfg.save_model(gm_net, ut_cfg.SaveMode.PROC, ut_cfg.SaveSuffix.pyPTH, epoch_i)

    # end the training process
    gm_cfg.save_model(gm_net, ut_cfg.SaveMode.TERMINATE, ut_cfg.SaveSuffix.pyPTH)



