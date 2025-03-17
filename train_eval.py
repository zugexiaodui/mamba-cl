import argparse
import random
from collections import OrderedDict
import tqdm
from typing import Any, Literal
from copy import deepcopy
from time import time as ttime
import inspect
import scipy.ndimage
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch import nn, Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.utils.hooks
from torch.utils.data import DataLoader, TensorDataset
import torch.linalg
import torchvision
from einops import rearrange
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from utils.mod_adam import ModAdam
from utils.config import build_model, get_config
from utils.dataset_builder import ImagePathDatasetClassManager, ImagePathDataset, define_dataset
from utils.continual_manager import ClassIncrementalManager
from utils import misc

import utils.mamba_block
import utils.defocus_attention_network
from utils.defocus_attention_network import DefocusAttentionNetwork

torch.set_float32_matmul_precision("high")


class GlobalVarsManager:
    args: argparse.Namespace
    path_data_dict: dict[str, ImagePathDataset]
    cl_mngr: ClassIncrementalManager
    acc_mat_dict: OrderedDict[str, np.ndarray]
    cache_dict: dict
    param_dict: dict[Literal['base_params', 'task_params_'], OrderedDict[str, Tensor]]
    label_map_g2l: dict[int, tuple[int, int, int]]  # {original_label: (taskid, local_label, global_label)}

    def init_from_args(self, args, this_file=__file__):
        self.args = args
        _dataset_class_manager = ImagePathDatasetClassManager(**{args.dataset: args.data_root})
        self.path_data_dict = {'train': _dataset_class_manager[args.dataset](train=True),
                               'eval': _dataset_class_manager[args.dataset](train=False)}
        self.cl_mngr = ClassIncrementalManager(self.path_data_dict['eval'].class_list, args.num_tasks, args.seed, shuffle=args.shuffle_classes)
        self.acc_mat_dict = OrderedDict(AccClassIncMat=np.zeros([_nt := self.cl_mngr.num_tasks, _nt]), AccClassIncList=np.zeros([_nt]))
        self.cache_dict = {}
        self.param_dict = {}
        self.label_map_g2l = {}

    def update_label_maps(self, taskid: int, task_classes: list[int]) -> tuple[dict[int, int], dict[str, int]]:
        _g2l_map = misc.make_label_maps(taskid, task_classes)
        if not all([_k not in self.label_map_g2l.keys() for _k in _g2l_map.keys()]):
            print("The global_to_local label map has been fully loaded, which is not expected.")
        self.label_map_g2l.update(_g2l_map)
        return _g2l_map


def get_args():
    parser = argparse.ArgumentParser(description='Class-incremental Learning')
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=('cifar100', 'imagenet_r', 'sdomainet'), help='use lowercase')
    parser.add_argument('-dr', '--data_root', type=str, default="")
    parser.add_argument('-t', '--num_tasks', type=int, default=10, choices=(1, 2, 5, 10, 20, 25, 50, 100))
    parser.add_argument('--shuffle_classes', type=misc.str2bool, default=True)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('-m', '--model', type=str, default='defocus_mamba_large', help='')
    parser.add_argument('--pretrained_path', type=str, default="../datasets/defocus_mamba_large_cls_21k.pth", help='')
    parser.add_argument('--head_dim_type', type=str, choices=('task_classes', 'pretrained', 'text_dim'), default='task_classes')
    parser.add_argument('--logit_type', type=str, choices=('head_out', 'sim_imgtext'), default='head_out')
    parser.add_argument('--seperate_head', type=misc.str2bool, default=True)
    parser.add_argument('--pretrained_exp', type=str, default="")
    parser.add_argument('--pretrained_ignore_patterns', type=str, nargs="*", default=[])
    parser.add_argument('--use_null_space', action='store_true')
    parser.add_argument('--null_patterns', type=str, nargs='+', default=('x_proj', 'out_proj.weight', 'A_log'))
    parser.add_argument('--null_eta', type=float, default=1.)
    parser.add_argument('--null_interm_accum', type=str, choices=('sum', 'mean'), default='sum')
    parser.add_argument('--refine_head', type=misc.str2bool, default=False)
    parser.add_argument('--rand_conv', type=misc.str2bool, default=True)
    parser.add_argument('--refine_epochs', type=int, default=20)
    parser.add_argument('--transform_type', type=str, choices=('timm', 'autoaug',), default='autoaug')
    parser.add_argument('--interp_mode', type=str, choices=('auto', 'bilinear', 'bicubic'), default='auto')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=200)
    parser.add_argument('-jt', '--workers', type=int, default=16)
    parser.add_argument('-je', '--eval_workers', type=int, default=2)
    parser.add_argument('-et', '--expand_times', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--use_amp', type=misc.str2bool, default=False)
    parser.add_argument('--sample_type', type=str, choices=('path', 'image'), default='image')
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--persistent_workers', type=misc.str2bool, default=False)
    parser.add_argument('--training_string', type=str, nargs='+', default=('head', 'x_proj', 'out_proj.weight', 'A_log'))
    parser.add_argument('-eb', '--eval_batch_size', type=int, default=100)
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.0002)
    parser.add_argument('--lr_scale', type=float, default=50)
    parser.add_argument('--lr_scale_patterns', type=str, nargs='+', default=('head',))
    parser.add_argument('--optimizer', type=str, default='mod_adam')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--lr_sch', type=str, default='multistep', choices=('cosine', 'step', 'multistep'))
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('-dm', '--decay_milestones', type=int, nargs='+', default=[5, 8])
    parser.add_argument('--decay_epochs', type=int, default=1000)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    args = parser.parse_args()

    if args.interp_mode == 'auto':
        match args.dataset:
            case 'imagenet_r' | 'sdomainet':
                args.interp_mode = 'bilinear'
            case 'cifar100':
                args.interp_mode = 'bicubic'

    return args


def seed_etc_options(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.set_printoptions(precision=4, linewidth=256)
    torch.set_printoptions(linewidth=256)
    torchvision.set_image_backend('accimage')


def set_model_mode(GVM: GlobalVarsManager, model: DefocusAttentionNetwork, training: bool, to_gpu: bool = True, training_string: tuple[str] = ('',)) -> DefocusAttentionNetwork:
    for n, p in model.named_parameters():
        if training and any([_s in n for _s in training_string]):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    params_requires_grad = [n for n, p in model.named_parameters() if p.requires_grad]

    model.eval()
    for n, m in model.named_modules():
        if training and any([n.endswith(_s) and not isinstance(m, nn.Identity) for _s in training_string]):
            m.train()
        else:
            m.eval()
    modules_training = [n for n, m in model.named_modules() if m.training]

    if to_gpu:
        model.cuda()

    if training:
        pass
    else:
        assert len(params_requires_grad) == 0, f"{params_requires_grad}"
        assert len(modules_training) == 0, f"{modules_training}"

    return model


def set_learning_rates(GVM: GlobalVarsManager, model: DefocusAttentionNetwork, base_lr: float, lr_scale: float, lr_scale_patterns: str) -> list[dict[str: Tensor | float]]:
    param_lr_groups = [{'params': [], 'lr': base_lr},
                       {'params': [], 'lr': base_lr * lr_scale}]
    lr_param_dict = {_p['lr']: [] for _p in param_lr_groups}

    for n, p in model.named_parameters():
        if p.requires_grad:
            _group_idx = 1 if any(_s in n for _s in lr_scale_patterns) else 0
            param_lr_groups[_group_idx]['params'].append(p)
            lr_param_dict[param_lr_groups[_group_idx]['lr']].append(n)

    return param_lr_groups


def train_one_epoch(GVM: GlobalVarsManager, curr_epoch: int, dataloader: DataLoader, model: DefocusAttentionNetwork, criterion: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer) -> str:
    args = GVM.args
    temperature: float = args.temperature
    use_amp: bool = args.use_amp
    assert temperature > 0.

    amp_scalar = GradScaler(enabled=use_amp)
    scalar_meter = misc.ScalarMeter(loss="samp_avg:.4f", batch_time="step_sum:.3f", acc_top1="samp_avg:>6.2%")
    _btimer = ttime()

    for i_batch, (images, target) in tqdm.tqdm(enumerate(dataloader, 1), total=len(dataloader), dynamic_ncols=True, disable=True):
        images: Tensor = images.cuda(non_blocking=True)
        target: Tensor = target.cuda(non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            logits: Tensor = model(images)

        if i_batch == 1:
            if args.seperate_head:
                assert logits.shape[1] == len(GVM.cl_mngr.current_task_classes)
            else:
                assert logits.shape[1] == len(GVM.cl_mngr.sofar_task_classes)

        loss: Tensor = criterion(logits / temperature, target)

        optimizer.zero_grad()
        amp_scalar.scale(loss).backward()
        amp_scalar.step(optimizer)
        amp_scalar.update()

        acc_top1, = misc.calc_accuracy(logits, target, topk=(1,))
        batch_time = ttime() - _btimer

        scalar_meter.add_step_value(len(images), loss=loss.item(), batch_time=batch_time, acc_top1=acc_top1)
        _btimer = ttime()

    _epoch_scalar_str = scalar_meter.format_outout(scalar_meter.update_epoch_average_value())
    return _epoch_scalar_str


def cache_state(GVM: GlobalVarsManager, taskid: int, epoch: int, model: DefocusAttentionNetwork, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler):

    task_params = OrderedDict()

    for n, p in model.named_parameters():
        if p.requires_grad:
            task_params[n] = p.clone().cpu()

    assert not f'task_params_{taskid}' in GVM.param_dict
    GVM.param_dict[f'task_params_{taskid}'] = dict(filter(lambda n_p: 'head' in n_p[0], task_params.items()))


def train_one_task(GVM: GlobalVarsManager, taskid: int, task_classes: list[int], model: DefocusAttentionNetwork, **kwargs) -> DefocusAttentionNetwork:
    args = GVM.args
    if args.epochs == 0:
        return model

    _ttimer = ttime()
    _ntstr = str(GVM.cl_mngr.num_tasks)

    model: DefocusAttentionNetwork = set_model_mode(GVM, model, training=True, training_string=GVM.cache_dict['training_string'])
    model = modify_head(GVM, model, training=True, task_classes=task_classes)

    dataset = define_dataset(GVM, task_classes, training=True, transform_type=args.transform_type, target_map_to_local=args.seperate_head, expand_times=args.expand_times)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, timeout=args.timeout if args.workers > 0 else 0,
                            drop_last=False, persistent_workers=args.persistent_workers)

    criterion = nn.CrossEntropyLoss().cuda()

    if args.lr_scale == 1:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())
    else:
        param_groups = set_learning_rates(GVM, model, args.lr, args.lr_scale, args.lr_scale_patterns)

    if taskid == 0:
        GVM.cache_dict['update_proj_dict'] = {}
    if args.use_null_space:
        if taskid == 0:
            GVM.cache_dict['null_param_id_dict'] = get_param_id_dict(model, args.null_patterns)
            GVM.cache_dict['interm_tensor_dict'] = {}
        else:
            assert sorted(list(GVM.cache_dict['update_proj_dict'].keys())) == sorted(list(GVM.cache_dict['null_param_id_dict'].keys()))
    else:
        assert GVM.cache_dict['update_proj_dict'] == {}

    if args.optimizer == 'mod_adam':
        optimizer = ModAdam(param_groups, update_projection_dict=GVM.cache_dict['update_proj_dict'], arg_dict={}, lr=args.lr, weight_decay=args.weight_decay, foreach=True)
    else:
        optimizer = create_optimizer_v2(param_groups, opt=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, foreach=True)

    scheduler, num_epochs = create_scheduler_v2(optimizer, sched=args.lr_sch, num_epochs=args.epochs, decay_epochs=args.decay_epochs, decay_milestones=args.decay_milestones,
                                                decay_rate=args.decay_rate, min_lr=args.min_lr, warmup_epochs=args.warmup_epochs, warmup_lr=args.min_lr)
    assert num_epochs == args.epochs

    torch.cuda.empty_cache()
    for epoch in range(0, args.epochs + 1):
        if epoch > 0:
            _epoch_scalar_str = train_one_epoch(GVM, epoch, dataloader, model, criterion, optimizer)
            print(f"Task [{taskid + 1:>{len(_ntstr)}}/{_ntstr}] Epoch [{epoch:>{len(_nestr := str(args.epochs))}}/{_nestr}]:: {_epoch_scalar_str}")
        scheduler.step(epoch)

    cache_state(GVM, taskid, epoch, model, optimizer, scheduler)

    if args.use_null_space and taskid + 1 < GVM.cl_mngr.num_tasks:
        new_interm_tensor_dict = get_interm_tensor_dict(GVM, model, GVM.cache_dict['null_param_id_dict'])
        GVM.cache_dict['interm_tensor_dict'] = accumulate_interm_tensor_dict(GVM, GVM.cache_dict['interm_tensor_dict'], new_interm_tensor_dict)
        del new_interm_tensor_dict
        GVM.cache_dict['update_proj_dict'] = get_update_projection_dict(GVM, GVM.cache_dict['null_param_id_dict'], GVM.cache_dict['interm_tensor_dict'])

    if args.refine_head:
        extract_class_features(GVM, model)
        if args.refine_head:
            refine_head(GVM, model)

    print(f"Task [{taskid + 1:>{len(_ntstr)}}/{_ntstr}]:: Training time = {misc.format_duration(ttime() - _ttimer)}")

    return model


def evaluate_one_task(GVM: GlobalVarsManager, train_taskid: int, eval_taskid: int, eval_task_classes: list[int], model: DefocusAttentionNetwork) -> OrderedDict[str, float]:
    use_amp: bool = GVM.args.use_amp
    _ttimer = ttime()

    dataset = define_dataset(GVM, eval_task_classes, training=False, transform_type=GVM.args.transform_type, target_map_to_local=False)
    dataloader = DataLoader(dataset, batch_size=GVM.args.eval_batch_size, shuffle=False, num_workers=GVM.args.eval_workers, pin_memory=True, timeout=GVM.args.timeout if GVM.args.eval_workers > 0 else 0)

    set_model_mode(GVM, model, training=False)
    scalar_meter = misc.ScalarMeter(acc_class_inc="samp_avg:>6.2%")

    for images, target in tqdm.tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, disable=True):
        images: Tensor = images.cuda(non_blocking=True)
        target: Tensor = target.cuda(non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                logits: Tensor = model(images)

        assert logits.ndim == 2
        assert logits.shape[1] == len(GVM.cl_mngr.sofar_task_classes), f"{logits.shape}, {len(GVM.cl_mngr.sofar_task_classes)}"

        class_inc_preds = logits.argmax(dim=1)

        acc_class_inc, acc_topnn_class, num_nn_class = misc.calc_acc_topnn_dynamically(class_inc_preds, target)
        scalar_meter.add_step_value(target.shape[0], acc_class_inc=acc_class_inc)

    assert len(dataset) == len(scalar_meter)
    result_dict = scalar_meter.update_epoch_average_value()

    print(f"Task [{train_taskid + 1}/{GVM.cl_mngr.num_tasks}]:: Eval [{eval_taskid + 1:>{len(_tt := str(train_taskid + 1))}}/{_tt}]: eval_time={ttime() - _ttimer:.1f}s, {scalar_meter.format_outout(result_dict)}")

    result_dict['num_samples'] = len(dataset)

    return result_dict


def evaluate_tasks_sofar(GVM: GlobalVarsManager, train_taskid: int, model: DefocusAttentionNetwork):
    model = modify_head(GVM, model, training=False)
    average_acc_meter = misc.ScalarMeter(acc_class_inc="samp_avg:>6.2%")

    for eval_taskid in range(GVM.cl_mngr.current_taskid + 1):
        eval_task_classes = GVM.cl_mngr.get_classes(eval_taskid)
        one_result_dict = evaluate_one_task(GVM, train_taskid, eval_taskid, eval_task_classes, model)
        GVM.acc_mat_dict[f'AccClassIncMat'][train_taskid, eval_taskid] = one_result_dict['acc_class_inc']
        average_acc_meter.add_step_value(**one_result_dict)

    avg_result_dict = average_acc_meter.update_epoch_average_value()
    GVM.acc_mat_dict[f'AccClassIncList'][train_taskid] = avg_result_dict['acc_class_inc']


def task_ending_info(GVM: GlobalVarsManager):
    current_taskid = GVM.cl_mngr.current_taskid

    acc_info_dict = {
        'class_inc_last_acc': float(GVM.acc_mat_dict['AccClassIncList'][current_taskid]),
        'class_inc_last_forg': misc.calc_forgetting(GVM.acc_mat_dict['AccClassIncMat'], current_taskid),
    }
    _formatter = misc.ScalarFormatter(sep=' | ', class_inc_last_acc=">6.2%", class_inc_last_forg=">6.2%")

    print(f":: ** Results of task [{current_taskid + 1}]: [ {_formatter(**acc_info_dict)} ] **")
    print(f":: ** Time so far: {misc.format_duration(ttime() - GVM.cache_dict['exp_start_time'])} **")


def get_param_id_dict(model: DefocusAttentionNetwork, patterns: list[str]) -> dict[int, dict[Literal['name', 'shape'], str | list[int]]]:
    param_id_dict = {}
    for n, p in model.named_parameters():
        if p.requires_grad and any([_s in n for _s in patterns]):
            param_id_dict[id(p)] = {'name': n, 'shape': list(p.shape)}
    assert len(param_id_dict) > 0, f"{param_id_dict}"
    return param_id_dict


def get_head_dim_arg_dict(GVM: GlobalVarsManager, args: argparse.Namespace) -> dict[Literal['num_classes'], int]:
    head_dim_arg_dict = {}
    head_dim_type = args.head_dim_type

    match args.logit_type:
        case 'sim_imgtext':
            assert head_dim_type in ('pretrained', 'text_dim')
        case 'head_out':
            assert head_dim_type in ('task_classes')

    match head_dim_type:
        case 'task_classes':
            head_dim_arg_dict['num_classes'] = len(current_task_classes) if args.seperate_head else len(GVM.cl_mngr.sofar_task_classes)
        case 'pretrained':
            pass
        case 'text_dim':
            head_dim_arg_dict['num_classes'] = 512
        case _:
            raise ValueError(head_dim_type)
    return head_dim_arg_dict


def modify_head(GVM: GlobalVarsManager, model: DefocusAttentionNetwork, training: bool, **kwargs):
    args: argparse.Namespace = GVM.args

    if training:
        _target_classes = kwargs['task_classes'] if args.seperate_head else GVM.cl_mngr.sofar_task_classes
    else:
        _target_classes = GVM.cl_mngr.sofar_task_classes

    if args.logit_type == 'head_out':
        if model.head.out_features != len(_target_classes):
            _mh = deepcopy(model.head)
            _mdevice = _mh.weight.device
            _mdtype = _mh.weight.dtype
            model.head = _mh.__class__(_mh.in_features, len(_target_classes), _mh.bias is not None, _mdevice, _mdtype)
            model.head.requires_grad_(_mh.weight.requires_grad)

            if training:
                assert model.head.weight.requires_grad
            else:
                assert _mh.out_features == len(GVM.cl_mngr.current_task_classes), f"{_mh.out_features}, {len(GVM.cl_mngr.current_task_classes)}"
                _hw = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.weight'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
                assert model.head.weight.data.shape == _hw.shape
                model.head.weight.data = _hw

                if _mh.bias is not None:
                    _hb = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.bias'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
                    assert model.head.bias.data.shape == _hb.shape
                    model.head.bias.data = _hb
        else:
            pass
    else:
        raise ValueError(args.logit_type)

    return model


def get_interm_tensor_dict(GVM: GlobalVarsManager, model: DefocusAttentionNetwork, null_param_id_dict: dict) -> dict[int, Tensor]:
    interm_tensor_dict: dict[int, dict[str, Tensor]] = {}

    def _forward_hook(module: nn.Module, args: tuple[Tensor], output: Tensor):
        if isinstance(module, utils.mamba_block.IntermReader):
            _pid = module.dst_param_id
            if _pid in null_param_id_dict:
                _mname = module.module_name
                _interm_tensor: Tensor = args[0]  # [b*l, d]

                if _mname == 'interm_reader_step':
                    _interm_tensor = rearrange(_interm_tensor, 'b l d -> (b l) d')
                    _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]
                elif _mname == 'interm_reader_B':
                    _interm_tensor = rearrange(_interm_tensor, 'b l d -> (b l) d')
                    _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]
                elif _mname == 'interm_reader_C':
                    _interm_tensor = rearrange(_interm_tensor, 'b l d -> (b l) d')
                    _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]
                elif _mname == 'interm_reader_A':
                    _interm_tensor = rearrange(_interm_tensor, 'b l d -> (b d) l')
                    _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]
                elif _mname == 'interm_reader_out':
                    _interm_tensor = rearrange(_interm_tensor, 'b l d -> (b l) d')
                    _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]
                else:
                    raise ValueError()

                assert _mname.startswith('interm_reader_')

                if _pid not in interm_tensor_dict:
                    interm_tensor_dict[_pid] = {}
                if _mname not in interm_tensor_dict[_pid]:
                    interm_tensor_dict[_pid][_mname] = torch.zeros_like(_interm_tensor)
                interm_tensor_dict[_pid][_mname] += _interm_tensor
        else:
            raise NotImplementedError()

    _handle_list: list[torch.utils.hooks.RemovableHandle] = []
    for n, m in model.named_modules():
        if 'interm_reader' in n and isinstance(m, utils.mamba_block.IntermReader):
            _handle_list.append(m.register_forward_hook(_forward_hook))

    model = set_model_mode(GVM, model, training=False)

    args = GVM.args
    train_dataset = define_dataset(GVM, GVM.cl_mngr.current_task_classes, training=True, transform_type=args.transform_type, target_map_to_local=args.seperate_head, use_eval_transform=True, expand_times=1)

    dataloader = DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.eval_workers, pin_memory=True, timeout=args.timeout if args.eval_workers > 0 else 0)

    for nb, (img, _) in enumerate(dataloader, 1):
        with torch.no_grad():
            img: Tensor
            model(img.cuda(non_blocking=True))

    for _h in _handle_list:
        _h.remove()

    assert len(interm_tensor_dict) > 0
    assert sorted(list(interm_tensor_dict.keys())) == sorted(list(null_param_id_dict.keys())), f"({len(interm_tensor_dict.keys())}){interm_tensor_dict.keys()}; ({len(null_param_id_dict)}){null_param_id_dict}"

    if (_k := 'interm_sample_list') not in GVM.cache_dict:
        GVM.cache_dict[_k] = []
    GVM.cache_dict[_k].append(len(dataloader.dataset))

    return interm_tensor_dict


def accumulate_interm_tensor_dict(GVM: GlobalVarsManager, cached_interm_tensor_dict: dict[int, dict[str, Tensor]], new_interm_tensor_dict: dict[int, dict[str, Tensor]]) -> dict[int, dict[str, Tensor]]:
    assert len(new_interm_tensor_dict) > 0
    args = GVM.args

    if cached_interm_tensor_dict == {}:
        merged_interm_tensor_dict = new_interm_tensor_dict
    else:
        assert (_lc := list(cached_interm_tensor_dict.keys())) == (_ln := list(new_interm_tensor_dict.keys())), f"{_lc}, {_ln}"
        merged_interm_tensor_dict: dict[int, Tensor] = {}
        for _pid in cached_interm_tensor_dict.keys():
            merged_interm_tensor_dict[_pid] = {}
            for _mname in cached_interm_tensor_dict[_pid].keys():
                _cached_tensor = cached_interm_tensor_dict[_pid][_mname]
                _new_tensor = new_interm_tensor_dict[_pid][_mname]
                assert _cached_tensor.shape == _new_tensor.shape

                match args.null_interm_accum:
                    case 'sum':
                        merged_interm_tensor_dict[_pid][_mname] = _cached_tensor + _new_tensor
                    case 'mean':
                        _num_list: list[int] = GVM.cache_dict['interm_sample_list']
                        merged_interm_tensor_dict[_pid][_mname] = sum(_num_list[:-1]) / sum(_num_list) * _cached_tensor + _num_list[-1] / sum(_num_list) * _new_tensor
                    case _:
                        raise ValueError()
    return merged_interm_tensor_dict


def get_update_projection_dict(GVM: GlobalVarsManager, null_param_id_dict: dict, interm_tensor_dict: dict[int, dict[str, Tensor]]) -> dict[int, dict[str, Tensor]]:
    args = GVM.args
    update_proj_dict = {}

    def adaptive_threshold(svals: torch.Tensor, offset: float = 0):
        points: np.ndarray = svals.cpu().numpy()
        assert points.ndim == 1
        if len(points) >= 128:
            fil_points = scipy.ndimage.gaussian_filter1d(points, sigma=10)
            _delta = 1
            diff_o1 = fil_points[:-_delta] - fil_points[_delta:]
            diff_o2 = diff_o1[:-1] - diff_o1[1:]
            _drop_ratio = 0.03
            drop_num = int(len(points) * _drop_ratio / 2)
            assert len(points) - drop_num >= 10
            valid_o2 = diff_o2[drop_num:-drop_num]
            thres_val = points[np.argmax(valid_o2) + int((len(points) - len(valid_o2)) / 2)]
        else:
            diff_o1 = points[:-1] - points[1:]
            diff_o2 = diff_o1[:-1] - diff_o1[1:]
            thres_val = points[np.argmax(diff_o2) + int((len(points) - len(diff_o2)) / 2)]
        i_thres = np.arange(len(points))[points >= thres_val].max()
        if 0 <= offset < 1:
            i_thres = min(i_thres + int(offset * (len(points) - i_thres)), len(points) - 1)
        else:
            i_thres = max(min(i_thres + int(offset), len(points) - 1), 0)

        zero_idx = np.zeros(len(points), dtype=np.int64)
        zero_idx[i_thres:] = 1
        zero_idx = torch.as_tensor(torch.from_numpy(zero_idx), dtype=torch.bool, device=svals.device)
        return zero_idx

    svals_dict = OrderedDict()
    zero_idx_dict = OrderedDict()
    for _pid in interm_tensor_dict.keys():
        update_proj_dict[_pid] = {}
        for _mname in interm_tensor_dict[_pid].keys():
            U, S, Vt = torch.linalg.svd(interm_tensor_dict[_pid][_mname], full_matrices=True)  # A=U@diag(S)@Vt
            S: Tensor
            Vt: Tensor
            zero_idx = adaptive_threshold(S)
            zero_idx: torch.BoolTensor
            assert torch.count_nonzero(zero_idx) > 0, f"{zero_idx}, {type(zero_idx)}, {torch.count_nonzero(zero_idx)}"

            svals_dict[_dkey := f"{null_param_id_dict[_pid]['name']}-{_mname}"] = S.cpu().clone()
            zero_idx_dict[_dkey] = zero_idx.cpu().clone()

            basis = Vt[zero_idx]
            proj = basis.T @ basis
            proj = proj / torch.norm(proj)

            assert proj.shape[0] == proj.shape[1], proj.shape

            null_eta: float = args.null_eta
            update_proj_dict[_pid][_mname] = null_eta * proj.detach() + (1 - null_eta) * torch.eye(proj.shape[0], device=proj.device, dtype=proj.dtype)

    return update_proj_dict


def extract_class_features(GVM: GlobalVarsManager, model: DefocusAttentionNetwork) -> None:
    model = set_model_mode(GVM, model, training=False)

    dataset = define_dataset(GVM, GVM.cl_mngr.current_task_classes, training=True, transform_type=args.transform_type, target_map_to_local=False, use_eval_transform=True, expand_times=1)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.eval_workers, pin_memory=True, timeout=args.timeout if args.eval_workers > 0 else 0)

    feats = torch.empty([len(dataset), 1024], dtype=torch.float32)
    label = torch.empty([len(dataset)], dtype=torch.long)

    smp_idx = 0
    for img, lbl in dataloader:
        with torch.no_grad():
            img: Tensor
            lbl: Tensor
            _feat = model.encode_image(img.cuda(non_blocking=True), pre_logits=True).cpu()
            for _f, _l in zip(_feat, lbl):
                feats[smp_idx] = _f
                label[smp_idx] = _l
                smp_idx += 1
    assert smp_idx == len(dataset)

    if GVM.args.refine_head:
        _mean_list = []
        _cov_list = []
        _class_list = []
        for _l in label.unique():
            _cls_feats = feats[label == _l]
            _mean_list.append(torch.mean(_cls_feats, dim=0, keepdim=False))
            _cov_list.append(torch.cov(torch.tensor(_cls_feats, dtype=torch.float64).T) + torch.eye(_cls_feats.shape[-1]) * 1e-4)
            _class_list.append(_l)
        _mean_list = torch.stack(_mean_list)
        _cov_list = torch.stack(_cov_list)
        _class_list = torch.stack(_class_list)

        _key = 'class_features'
        if _key not in GVM.cache_dict:
            GVM.cache_dict[_key] = {'mean': _mean_list, 'cov': _cov_list, 'class': _class_list}
        else:
            GVM.cache_dict[_key]['mean'] = torch.cat([GVM.cache_dict[_key]['mean'], _mean_list])
            GVM.cache_dict[_key]['cov'] = torch.cat([GVM.cache_dict[_key]['cov'], _cov_list])
            GVM.cache_dict[_key]['class'] = torch.cat([GVM.cache_dict[_key]['class'], _class_list])
            assert len(GVM.cache_dict[_key]['mean']) == len(GVM.cache_dict[_key]['cov']) == len(GVM.cache_dict[_key]['class']) == len(GVM.cl_mngr.sofar_task_classes)

    return None


def refine_head(GVM: GlobalVarsManager, model: DefocusAttentionNetwork):
    feats_mean: Tensor = GVM.cache_dict['class_features']['mean']
    feats_cov: Tensor = GVM.cache_dict['class_features']['cov']
    feats_class: Tensor = GVM.cache_dict['class_features']['class']
    assert len(feats_class.unique()) == len(GVM.cl_mngr.sofar_task_classes)

    stat_dataset = TensorDataset(feats_mean, feats_cov, feats_class)

    model = modify_head(GVM, model, training=False)
    mhead = model.head

    mhead.train()
    mhead.cuda()
    mhead.requires_grad_()

    optimizer = create_optimizer_v2(mhead, opt='sgd', lr=0.001, weight_decay=1e-4, momentum=0.9)
    scheduler, num_epochs = create_scheduler_v2(optimizer, 'multistep', num_epochs=GVM.args.refine_epochs, decay_milestones=[999,], decay_rate=0.1)
    criterion = nn.CrossEntropyLoss().cuda()
    from torch.distributions.multivariate_normal import MultivariateNormal

    scalar_meter = misc.ScalarMeter(loss="samp_avg:.4f", acc_top1="samp_avg:>6.2%")
    for epoch in range(1, num_epochs + 1):
        scheduler.step(epoch)

        smp_inp = []
        smp_tgt = []
        assert len(stat_dataset) == len(GVM.cl_mngr.sofar_task_classes)
        _ns = 256
        for _cmean, _ccov, _cclass in stat_dataset:
            if not GVM.args.rand_conv:
                m = MultivariateNormal(_cmean.float(), _ccov.float())
            else:
                _tmp = torch.randn([1, _cmean.shape[0]], device=_cmean.device)
                _tmp = torch.matmul(_tmp.T, _tmp)
                _tmp = torch.eye(_cmean.shape[0]) + _tmp
                m = MultivariateNormal(_cmean.float(), _tmp)
            _smp = m.sample(sample_shape=(_ns,))
            smp_inp.append(_smp)
            smp_tgt.append(torch.as_tensor([_cclass,] * _ns, dtype=torch.long))
        smp_inp = torch.cat(smp_inp)
        smp_tgt = torch.cat(smp_tgt)

        train_data = TensorDataset(smp_inp, smp_tgt)
        assert len(train_data) == len(stat_dataset) * _ns
        dataloader = DataLoader(train_data, batch_size=256, shuffle=True)

        for inp, tgt in dataloader:
            out: Tensor = mhead(inp.cuda(non_blocking=True))
            logits = out
            loss: Tensor = criterion(logits / GVM.args.temperature, tgt.cuda(non_blocking=True))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_top1, = misc.calc_accuracy(logits.cpu(), tgt.cpu(), topk=(1, ))
            scalar_meter.add_step_value(len(inp), loss=loss.item(), acc_top1=acc_top1)
        if (epoch % 10 == 0 or epoch == num_epochs):
            print(f":: {inspect.stack()[0].function}: epoch [{epoch}/{num_epochs}]: {scalar_meter.format_outout(scalar_meter.update_epoch_average_value())}")


if __name__ == "__main__":
    args = get_args()
    seed_etc_options(args.seed)

    GVM = GlobalVarsManager()
    GVM.init_from_args(args)
    GVM.cache_dict['exp_start_time'] = ttime()

    for taskid, current_task_classes in GVM.cl_mngr:
        if taskid == 0:
            model: DefocusAttentionNetwork = build_model(get_config("./utils/defocus_mamba_large_22k.yaml"))
            ckpt = torch.load(args.pretrained_path, map_location='cpu')
            model.load_state_dict(ckpt['model'], strict=False)
            setattr(model, 'pretrained_cfg', {})
            model.pretrained_cfg['pretrained_path'] = args.pretrained_path
            GVM.cache_dict['pretrained_cfg'] = deepcopy(model.pretrained_cfg)
            model.reload_params()
            GVM.cache_dict['training_string'] = args.training_string

        GVM.update_label_maps(taskid, current_task_classes)
        model = train_one_task(GVM, taskid, current_task_classes, model)

        evaluate_tasks_sofar(GVM, taskid, model)
        task_ending_info(GVM)
