import os
import time
import torch
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict
from enum import Enum

class InitMode(Enum):
    XAVIER = 0
    KAIMING = 1
    NORM = 2
    PRETRAIN = 3
    NONE = 4

class SaveMode(Enum):
    PROC = 0
    TERMINATE = 1

class SaveSuffix(Enum):
    pyPTH = 0
    CppPT = 1
    ONNX = 2

def init_xavier(pNet):
    for m in pNet.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)


def init_kaiming(pNet):
    for m in pNet.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)

def init_norm(pNet):
    for m in pNet.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.normal_(m.weight, mean = 0.0, std = 0.02)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight, mean = 1.0, std = 0.02)
            init.constant_(m.bias, 0)

class ProcConfig(object):
    def __init__(self, saving_id: str, pBs : int, pWn : int, p_force_cpu : bool):
        # device & its name
        self.saving_id = saving_id
        self.device_index = 0 # default
        self.device, self.device_info = self.select_device(p_force_cpu)
        # path 
        self.checkpoints_root_path = None
        # loader
        self.ld_batchsize = pBs #8
        self.ld_workers = pWn #2

        # input shape config info 
        self.img_C = None
        self.img_H = None
        self.img_W = self.img_H

    def get_localtime(self) -> str:
        localtime = time.localtime(time.time())

        return "%02d:%02d"%(localtime.tm_hour, localtime.tm_min)
    
    def get_optLR(self, optimizer)-> float: 
        return optimizer.param_groups[0]['lr']

    def select_device(self, force_cpu = False):
        cuda_flag = False if force_cpu else torch.cuda.is_available()
        device = torch.device('cuda:%d'%(self.device_index) if cuda_flag else 'cpu')

        device_str = None
        if not cuda_flag:
            import platform 
            device_str = "cpu" + "(name: %s)"%(platform.processor())
            
        else: 
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True 

            c = 1024 ** 3  # bytes to MB

            ng = torch.cuda.device_count()
            device_prop = torch.cuda.get_device_properties(self.device_index)
            device_str = "cuda" + f"(index: {self.device_index}/{ng}, name: {device_prop.name}, mem: {float(device_prop.total_memory/c)} G)" 

        return device, device_str

    def check_path_valid(self, path) -> str:
        os.makedirs(path, exist_ok = True)
        return(path)
    
    def save_model(self, pNet, save_mode, suffix = SaveSuffix.pyPTH, epochX = None):
        model_filename = self.saving_id + "-" + pNet._get_name()
        if save_mode == SaveMode.PROC:
            assert epochX is not None, "miss the epoch info" 
            model_filename += "_check%03d"%(epochX)
            pNet.train() # important: PROC for re-training. 
        elif save_mode == SaveMode.TERMINATE:
            assert self.total_epoch is not None, "total epoch un-defined. "
            model_filename += "_final%03d"%(self.total_epoch)
            pNet.eval() # important: TERMINATE for deploy, the data-flow may change. 

        weight_path = os.path.join(self.checkpoints_root_path, model_filename) # w/o suffix
        if suffix == SaveSuffix.pyPTH:
            weight_path += ".pth"
            torch.save(obj = pNet.state_dict(), f = weight_path)
        elif suffix == SaveSuffix.CppPT:
            weight_path += ".pt"
            traced_script_module = torch.jit.script(pNet)
            traced_script_module.save(weight_path)
        elif suffix == SaveSuffix.ONNX:
            # current only test for SRNet (2021. 03. 26)
            assert (self.img_C is not None) and (self.img_H is not None) and (self.img_W is not None)
            dummy_input = torch.randn(1, self.img_C, self.img_H, self.img_W, device = self.device)
            weight_path += ".onnx"
            torch.onnx.export(
                model = pNet, 
                args = dummy_input, 
                f = weight_path, 
                verbose=True, 
                opset_version=11,
                do_constant_folding = True,
                input_names = ['input'],
                output_names = ['output'],
                dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}}
                )



    def init_model(self, pNet, init_mode, isTrain, pretrain_path = None) -> None:
        if init_mode == InitMode.XAVIER:
            init_xavier(pNet)
        elif init_mode == InitMode.KAIMING:
            init_kaiming(pNet)
        elif init_mode == InitMode.NORM:
            init_norm(pNet)
        elif init_mode == InitMode.PRETRAIN:
            assert pretrain_path is not None, "weight path is not given"
            self.load_checkpoints(pNet, pretrain_path)
        pNet.to(self.device)
        if isTrain : pNet.train()
        else : pNet.eval()

    def load_checkpoints(self, pNet, pretrain_path):
            if ".pth" in pretrain_path:
                # torch dictionary without Arch Defination. 
                try : 
                    pNet.load_state_dict(torch.load(pretrain_path))
                except RuntimeError: 
                    pre_state_Dic= torch.load(pretrain_path)
                    temp_state_Dic = OrderedDict()
                    for key_self, key_out in zip(pNet.state_dict(), pre_state_Dic.keys()):
                        temp_state_Dic[key_self] = pre_state_Dic[key_out]
                    pNet.load_state_dict(temp_state_Dic)
            elif ".pt" in pretrain_path:
                # torch script for C++ deployment. 
                pNet_loading = torch.jit.load(pretrain_path)
                pNet.load_state_dict(pNet_loading.state_dict())
                print("[load pre-trained torch-script weight]")

            elif ".weight" in pretrain_path:
                # yolo darknet. 
                pNet.load_darknet_weights(pretrain_path)

    def to_cpu(self, device_Tsor:torch.Tensor):
        if device_Tsor.requires_grad:
            device_Tsor = device_Tsor.detach()
        if device_Tsor.is_cuda:
            device_Tsor = device_Tsor.cpu()
        return device_Tsor
    def should_evaluate_checkpoints(self, pEpoch:int):
        raise NotImplementedError
    def should_save_checkpoints(self, pEpoch:int):
        raise NotImplementedError

    def create_dataset(self, isTrain):
        raise NotImplementedError
