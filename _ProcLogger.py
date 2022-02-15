import os 
import time 
import sys
from functools import partial
try:
    from terminaltables import AsciiTable
except ImportError: 
    print("please pip install terminaltables...")
    from terminaltables import AsciiTable
import numpy as np
import torch 
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter
# print("for torch1.4, please pip install tensorboard==2.0.2")
'''
tensorboard --logdir board --port 8890
'''

'''
Copyright (C) 2019 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
 https://github.com/sovrasov/flops-counter.pytorch/blob/master/ptflops/flops_counter.py
'''
##########################  Begin flops_counter ###############################
def get_model_complexity_info(model, input_res,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None, ost=sys.stdout,
                              verbose=False, ignore_modules=[],
                              custom_modules_hooks={}):
    assert type(input_res) is tuple
    assert len(input_res) >= 1
    assert isinstance(model, nn.Module)
    global CUSTOM_MODULES_MAPPING
    CUSTOM_MODULES_MAPPING = custom_modules_hooks
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count(ost=ost, verbose=verbose,
                                  ignore_list=ignore_modules)
    if input_constructor:
        input = input_constructor(input_res)
        _ = flops_model(**input)
    else:
        try:
            batch = torch.ones(()).new_empty((1, *input_res),
                                             dtype=next(flops_model.parameters()).dtype,
                                             device=next(flops_model.parameters()).device)
        except StopIteration:
            batch = torch.ones(()).new_empty((1, *input_res))

        _ = flops_model(batch)

    flops_count, params_count = flops_model.compute_average_flops_cost()
    if print_per_layer_stat:
        print_model_with_flops(flops_model, flops_count, params_count, ost=ost)
    flops_model.stop_flops_count()
    CUSTOM_MODULES_MAPPING = {}

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count


def flops_to_string(flops, units='GMac', precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, 2)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, 2)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)


def accumulate_flops(self):
    if is_supported_instance(self):
        return self.__flops__
    else:
        sum = 0
        for m in self.children():
            sum += m.accumulate_flops()
        return sum


def print_model_with_flops(model, total_flops, total_params, units='GMac',
                           precision=3, ost=sys.stdout):
    if total_flops < 1:
        total_flops = 1

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()
            return sum

    def flops_repr(self):
        accumulated_params_num = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops() / model.__batch_counter__
        return ', '.join([params_to_string(accumulated_params_num,
                                           units='M', precision=precision),
                          '{:.3%} Params'.format(accumulated_params_num / total_params),
                          flops_to_string(accumulated_flops_cost,
                                          units=units, precision=precision),
                          '{:.3%} MACs'.format(accumulated_flops_cost / total_flops),
                          self.original_extra_repr()])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(repr(model), file=ost)
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(
                                                    net_main_module)

    net_main_module.reset_flops_count()

    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """

    for m in self.modules():
        m.accumulate_flops = accumulate_flops.__get__(m)

    flops_sum = self.accumulate_flops()

    for m in self.modules():
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    params_sum = get_model_parameters_number(self)
    return flops_sum / self.__batch_counter__, params_sum


def start_flops_count(self, **kwargs):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)

    seen_types = set()

    def add_flops_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if is_supported_instance(module):
                module.__params__ = 0
        elif is_supported_instance(module):
            if hasattr(module, '__flops_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(
                                        CUSTOM_MODULES_MAPPING[type(module)])
            else:
                handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
            module.__flops_handle__ = handle
            seen_types.add(type(module))
        else:
            if verbose and not type(module) in (nn.Sequential, nn.ModuleList) and \
               not type(module) in seen_types:
                print('Warning: module ' + type(module).__name__ +
                      ' is treated as a zero-op.', file=ost)
            seen_types.add(type(module))

    self.apply(partial(add_flops_counter_hook_function, **kwargs))


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


# ---- Internal functions
def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    # pytorch checks dimensions, so here we don't care much
    output_last_dim = output.shape[-1]
    bias_flops = output_last_dim if module.bias is not None else 0
    module.__flops__ += int(np.prod(input.shape) * output_last_dim + bias_flops)


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    module.__flops__ += int(np.prod(input.shape))


def bn_flops_counter_hook(module, input, output):
    input = input[0]

    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('Warning! No positional inputs found for a module,'
              ' assuming batch size is 1.')
    module.__batch_counter__ += batch_size


def rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    flops += w_ih.shape[0]*w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    flops += w_hh.shape[0]*w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size*3
        # last two hadamard product and add
        flops += rnn_module.hidden_size*3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size*4
        # two hadamard product and add for C state
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def rnn_flops_counter_hook(rnn_module, input, output):
    """
    Takes into account batch goes at first position, contrary
    to pytorch common rule (but actually it doesn't matter).
    IF sigmoid and tanh are made hard, only a comparison FLOPS should be accurate
    """
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def rnn_cell_flops_counter_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    flops = rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


def multihead_attention_counter_hook(multihead_attention_module, input, output):
    flops = 0
    q, k, v = input
    batch_size = q.shape[1]

    num_heads = multihead_attention_module.num_heads
    embed_dim = multihead_attention_module.embed_dim
    kdim = multihead_attention_module.kdim
    vdim = multihead_attention_module.vdim
    if kdim is None:
        kdim = embed_dim
    if vdim is None:
        vdim = embed_dim

    # initial projections
    flops = q.shape[0] * q.shape[2] * embed_dim + \
        k.shape[0] * k.shape[2] * kdim + \
        v.shape[0] * v.shape[2] * vdim
    if multihead_attention_module.in_proj_bias is not None:
        flops += (q.shape[0] + k.shape[0] + v.shape[0]) * embed_dim

    # attention heads: scale, matmul, softmax, matmul
    head_dim = embed_dim // num_heads
    head_flops = q.shape[0] * head_dim + \
        head_dim * q.shape[0] * k.shape[0] + \
        q.shape[0] * k.shape[0] + \
        q.shape[0] * k.shape[0] * head_dim

    flops += num_heads * head_flops

    # final projection, bias is always enabled
    flops += q.shape[0] * embed_dim * (embed_dim + 1)

    flops *= batch_size
    multihead_attention_module.__flops__ += int(flops)


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops__') or hasattr(module, '__params__'):
            print('Warning: variables __flops__ or __params__ are already '
                  'defined for the module' + type(module).__name__ +
                  ' ptflops can affect your code!')
        module.__flops__ = 0
        module.__params__ = get_model_parameters_number(module)


CUSTOM_MODULES_MAPPING = {}

MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_flops_counter_hook,
    nn.Conv2d: conv_flops_counter_hook,
    nn.Conv3d: conv_flops_counter_hook,
    # activations
    nn.ReLU: relu_flops_counter_hook,
    nn.PReLU: relu_flops_counter_hook,
    nn.ELU: relu_flops_counter_hook,
    nn.LeakyReLU: relu_flops_counter_hook,
    nn.ReLU6: relu_flops_counter_hook,
    # poolings
    nn.MaxPool1d: pool_flops_counter_hook,
    nn.AvgPool1d: pool_flops_counter_hook,
    nn.AvgPool2d: pool_flops_counter_hook,
    nn.MaxPool2d: pool_flops_counter_hook,
    nn.MaxPool3d: pool_flops_counter_hook,
    nn.AvgPool3d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_flops_counter_hook,
    # BNs
    nn.BatchNorm1d: bn_flops_counter_hook,
    nn.BatchNorm2d: bn_flops_counter_hook,
    nn.BatchNorm3d: bn_flops_counter_hook,
    # FC
    nn.Linear: linear_flops_counter_hook,
    # Upscale
    nn.Upsample: upsample_flops_counter_hook,
    # Deconvolution
    nn.ConvTranspose1d: conv_flops_counter_hook,
    nn.ConvTranspose2d: conv_flops_counter_hook,
    nn.ConvTranspose3d: conv_flops_counter_hook,
    # RNN
    nn.RNN: rnn_flops_counter_hook,
    nn.GRU: rnn_flops_counter_hook,
    nn.LSTM: rnn_flops_counter_hook,
    nn.RNNCell: rnn_cell_flops_counter_hook,
    nn.LSTMCell: rnn_cell_flops_counter_hook,
    nn.GRUCell: rnn_cell_flops_counter_hook,
    nn.MultiheadAttention: multihead_attention_counter_hook
}


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__

##########################  End flops_counter ###############################

class ProcLog(object):
    def __init__(self, log_dir : str, tag : str):
        os.makedirs(log_dir, exist_ok= True)
        localtime = time.localtime(time.time()); date = "%02d%02d"%(localtime.tm_mon, localtime.tm_mday)
        logfile_path = os.path.join(log_dir, tag + "_" + date + "_" +"log.txt")
        self.txt_writer = open(logfile_path, 'a+')
        self.board_writer = SummaryWriter(log_dir=os.path.join(log_dir, "board"))

    def __del__(self):
        self.txt_writer.close()
        self.board_writer.close()

    def log_the_str(self, *print_paras):
        for para_i in print_paras:
            print(para_i, end= "")
            print(para_i, end= "", file = self.txt_writer)
        print("")
        print("", file = self.txt_writer)
        self.txt_writer.flush()

    def log_the_table(self, title, info_table: list):
        table = AsciiTable(info_table, title).table
        self.log_the_str(table)


    def log_scalars_singleline(self, info_table: list):
        line_str = ""
        for idx_i, info_pair in enumerate(info_table):
            info_val = info_pair[1]
            if isinstance(info_val, float): 
                info_val = str(round(info_val, 5))
            elif isinstance(info_val, int):
                info_val = "%03d"%(info_val)
            info_str = "= ".join([str(info_pair[0]), str(info_val)])
            if idx_i != len(info_table): 
                info_str += ", "
            line_str += info_str
        self.log_the_str(line_str)
        

    def summarize_config(self, pConfig):
        info_table = [['item', 'detail']]
        info_table.append(["config id", pConfig.saving_id])
        info_table.append(["# epochs", pConfig.total_epoch])
        info_table.append(["batch size", pConfig.ld_batchsize])
        info_table.append(["#workers ", pConfig.ld_workers])
        info_table.append(["init mode", pConfig.method_init])
        info_table.append(["device", pConfig.device_info])
        
        self.log_the_table("config" , info_table)

    def summarize_netarch(self, pNet):
        # input_size : (C, H, W)
        # in terminal:
        # summary(pNet, input_size)
        # in file
        info_table = [['item', 'detail']]
        info_table.append(["is Cuda", next(pNet.parameters()).is_cuda])
        para_num = 0
        for para_i in pNet.parameters():
            # para_i is a torch.nn.parameter.Parameter, grad is True by default. 
            # print(para_i.shape, para_i.requires_grad)
            para_num += para_i.numel()
        info_table.append(["#parameters", format(para_num, ",")])

        conv_num = 0; fc_num = 0
        for m in pNet.modules():
            if isinstance(m, nn.modules.conv._ConvNd):
                # nn.convNd
                conv_num += 1
            elif isinstance(m, nn.Linear):
                fc_num += 1
        info_table.append(["#conv_layers", conv_num])
        info_table.append(["#fc_layers", fc_num])

        self.log_the_table("Net-" + pNet._get_name() , info_table)
    
    def fetch_netweights(self, pNet):
        para_Lst = []
        for para_i in pNet.parameters():
            para_Lst.append(para_i.reshape(-1))
        return torch.cat(para_Lst)

    def board_net_weightdist(self, chart_tag, pNet, step_i):
        '''
        view the distribution of the net weights in tfboard. 
        '''
        weights = self.fetch_netweights(pNet)
        self.board_writer.add_histogram(chart_tag, weights, step_i)

    def board_scalars_singlechart(self, chart_tag, data_dic, step_i):
        self.board_writer.add_scalars(chart_tag, data_dic, step_i)
    
    def board_scalars_multicharts(self, chart_tag, data_dic, step_i):
        # key : str; item : val
        for key, val in data_dic.item():
            self.board_writer.add_scalar(chart_tag + "_"+ key, val, step_i)
        
    def board_imgs_singlefig(self, fig_tag, img_Tsor, step_i):
        '''
        img_Tsor : (C, H, W)
        img_Tsor = torchvision.utils.make_grid(tensor = imgF_Tsor_bacth_i, nrow= w_layout)
        '''
        self.board_writer.add_image(fig_tag, img_Tsor, step_i)

    def board_geos_singlefig(self, fig_tag, vert_Tsor, color_Tsor, faces_Tsor, step_i):
        '''
        vert_Tsor, color_Tsor, faces_Tsor (BS, N, 3)
        '''
        self.board_writer.add_mesh(fig_tag, vert_Tsor, color_Tsor, faces_Tsor, global_step = step_i)