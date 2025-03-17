import os
from os.path import join, exists, realpath, dirname
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from typing import Union, Callable
from copy import deepcopy
from functools import partial
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class ImagePathDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.class_imgpath_dict: dict[int, str] = {}
        self.class_int_str_map: dict[int, str] = {}
        self.split_path = ""

    def __getitem__(self, index):
        return self.class_imgpath_dict[index]

    def __len__(self):
        return len(self.class_imgpath_dict)

    @property
    def class_list(self):
        return sorted(list(self.class_imgpath_dict.keys()))

    @property
    def num_classes(self):
        return len(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: image_dir=\"{self.split_path}\", full_dataset_classes={len(self)}"


class CIFAR100Path(ImagePathDataset):
    def __init__(self, root_dir: str, train: bool):
        super().__init__()
        cls_dir_list = sorted(os.listdir(split_path := join(root_dir, 'train' if train else 'val')))
        assert exists(split_path), split_path
        self.split_path = split_path

        name_map = {}
        if exists(_rf := join(f"{dirname(dirname(realpath(__file__)))}/tools/cifar100_classnames.txt")):
            with open(_rf, 'r') as f:
                for _i, line in enumerate(f.readlines()):
                    _dn = line.strip('\n')
                    name_map[_i] = _dn
        else:
            raise FileExistsError(f"{_rf}")
        assert len(name_map) == 100

        for cls_dir in cls_dir_list:
            cls_int = int(cls_dir)
            assert not cls_int in self.class_imgpath_dict
            cls_path = realpath(join(split_path, cls_dir))
            self.class_imgpath_dict[cls_int] = [join(cls_path, img_file) for img_file in os.listdir(cls_path)]
            self.class_int_str_map[cls_int] = name_map[cls_int]

        assert len(self.class_imgpath_dict) == 100


class ImageNetRPath(ImagePathDataset):
    def __init__(self, root_dir: str, train: bool):
        super().__init__()
        cls_dir_list = sorted(os.listdir(split_path := join(root_dir, 'train' if train else 'val')))
        assert exists(split_path), split_path
        self.split_path = split_path

        name_map = {}
        if exists(_rf := join(f"{dirname(dirname(realpath(__file__)))}/tools/imagenet_r_classnames.txt")):
            with open(_rf, 'r') as f:
                for line in f.readlines():
                    _sn = line.split(' ')[0]
                    _dn = line.strip('\n')[len(_sn) + 1:]
                    name_map[_sn] = _dn
        else:
            raise FileExistsError(f"{_rf}")
        assert len(name_map) == 200

        for cls_int, cls_dir in enumerate(cls_dir_list):
            assert not cls_int in self.class_imgpath_dict
            cls_path = realpath(join(split_path, cls_dir))
            self.class_imgpath_dict[cls_int] = [join(cls_path, img_file) for img_file in os.listdir(cls_path)]
            self.class_int_str_map[cls_int] = name_map[cls_dir]

        assert len(self.class_imgpath_dict) == 200


class SDomainNetPath(ImagePathDataset):
    def __init__(self, root_dir: str, train: bool):
        super().__init__()
        cls_dir_list = sorted(os.listdir(split_path := join(root_dir, 'train' if train else 'val')))
        assert exists(split_path), split_path
        self.split_path = split_path

        for cls_int, cls_dir in enumerate(cls_dir_list):
            assert not cls_int in self.class_imgpath_dict
            cls_path = realpath(join(split_path, cls_dir))
            self.class_imgpath_dict[cls_int] = [join(cls_path, img_file) for img_file in os.listdir(cls_path)]
            self.class_int_str_map[cls_int] = cls_int

        assert len(self.class_imgpath_dict) == 200


class ImagePathDatasetClassManager():
    def __init__(self, **kwargs):
        self.dataset_dict = {
            'cifar100': partial(CIFAR100Path, root_dir="../datasets/data.CIFAR100" if not (v := kwargs.get('cifar100')) else v),
            'imagenet_r': partial(ImageNetRPath, root_dir="../datasets/data.ImageNet-R" if not (v := kwargs.get('imagenet_r')) else v),
            'sdomainet': partial(SDomainNetPath, root_dir="../datasets/data.DomainNet" if not (v := kwargs.get('sdomainet')) else v),
        }

    def __getitem__(self, dataset: str) -> ImageNetRPath | CIFAR100Path | SDomainNetPath:
        dataset = dataset.lower()
        if dataset not in (_valid_names := self.dataset_dict.keys()):
            raise NameError(f"{dataset} is not in {_valid_names}")
        return self.dataset_dict[dataset]


class ClassIncremantalDataset(Dataset):
    def __init__(self, path_dataset: ImagePathDataset, task_class_list: list[int], transforms: T.Compose = None, target_transforms: Callable = None, expand_times: int = 1, return_index: bool = False, sample_type='path'):
        super().__init__()
        self.path_dataset = path_dataset
        self.task_class_list = tuple(deepcopy(task_class_list))
        assert isinstance(expand_times, int) and expand_times >= 1
        self.expand_times = expand_times
        self.transforms = transforms
        self.target_transforms = target_transforms
        assert sample_type in ('path', 'image')
        self.sample_type = sample_type

        self.samples, self.labels = self.get_all_samples(sample_type=self.sample_type)
        self.return_index = return_index
        self.num_samples = len(self.labels)

        self.cache_dict = {}

    def get_all_samples(self, sample_type: str = 'path') -> tuple[list[Image.Image | str], list[int]]:
        assert sample_type in ('path', 'image'), f"{sample_type}"
        smp_list = []
        lbl_list = []
        for cls_int in self.task_class_list:
            assert cls_int in self.path_dataset.class_list
            assert len(self.path_dataset[cls_int]) > 0
            for img_path in sorted(self.path_dataset[cls_int]):
                if sample_type == 'image':
                    sample: Image.Image = Image.open(img_path).convert('RGB')
                elif sample_type == 'path':
                    sample: str = img_path
                smp_list.append(sample)

                label = cls_int
                if self.target_transforms is not None:
                    label = self.target_transforms(label)
                lbl_list.append(label)

        return smp_list, lbl_list

    def read_one_image_label(self, index: int) -> tuple[Image.Image, int]:
        if self.sample_type == 'path':
            if self.expand_times == 1:
                img = Image.open(self.samples[index]).convert('RGB')
            elif self.expand_times > 1:
                img = Image.open(self.samples[index]).convert('RGB')
            else:
                raise ValueError(f"{self.expand_times}")

            lbl = self.labels[index]
            return img, lbl
        elif self.sample_type == 'image':
            return self.samples[index], self.labels[index]
        else:
            raise NameError(f"{self.sample_type}")

    def __getitem__(self, index: int) -> tuple[Union[Image.Image, Tensor], int]:
        index %= self.num_samples
        img, lbl = self.read_one_image_label(index)

        if self.transforms is not None:
            if self.return_index:
                return self.transforms(img), lbl, index
            else:
                return self.transforms(img), lbl
        if self.return_index:
            return img, lbl, index
        else:
            return img, lbl

    def __len__(self):
        return self.num_samples * self.expand_times

    def __repr__(self) -> str:
        _repr = f"{self.__class__.__name__} for {self.path_dataset.__repr__()}: task_class_list({len(self.task_class_list)})={self.task_class_list}, num_samples={len(self.samples)}, expand_times={self.expand_times}"
        return _repr


def define_dataset(GVM, task_classes: list[int], training: bool, transform_type: str = 'timm', target_map_to_local: bool = True,
                   use_eval_transform: bool = False, expand_times: int = 1, **kwargs) -> ClassIncremantalDataset:
    _current_dataset = GVM.args.dataset
    match GVM.args.interp_mode:
        case 'bilinear':
            interp_mode = T.InterpolationMode.BILINEAR
        case 'bicubic':
            interp_mode = T.InterpolationMode.BICUBIC
        case _:
            raise ValueError(GVM.args.interp_mode)
    bilinear = T.InterpolationMode.BILINEAR

    match transform_type:
        case 'timm':
            transforms = create_transform(**resolve_data_config(GVM.cache_dict['pretrained_cfg']), is_training=training if not use_eval_transform else False)
        case 'autoaug':
            if 'mean' in GVM.cache_dict['pretrained_cfg']:
                dmean: tuple[float] = GVM.cache_dict['pretrained_cfg']['mean']
                dstd: tuple[float] = GVM.cache_dict['pretrained_cfg']['std']
            else:
                from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
                dmean: tuple[float] = IMAGENET_DEFAULT_MEAN
                dstd: tuple[float] = IMAGENET_DEFAULT_STD
            if GVM.args.model == 'defocus_mamba_large':
                IMG_SIZE = 192
            else:
                IMG_SIZE = 224
            if training and not use_eval_transform:
                match _current_dataset:
                    case 'cifar100':
                        transforms = T.Compose([T.AutoAugment(T.AutoAugmentPolicy.CIFAR10, bilinear), T.RandomResizedCrop((IMG_SIZE, IMG_SIZE), interpolation=interp_mode, antialias=True), T.ToTensor(), T.Normalize(dmean, dstd)])
                    case 'imagenet_r' | 'sdomainet':
                        transforms = T.Compose([T.AutoAugment(T.AutoAugmentPolicy.IMAGENET, bilinear), T.RandomResizedCrop((IMG_SIZE, IMG_SIZE), interpolation=interp_mode, antialias=True), T.ToTensor(), T.Normalize(dmean, dstd)])
                    case _:
                        raise NotImplementedError(_current_dataset)
            else:
                match _current_dataset:
                    case 'cifar100':
                        transforms = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE), antialias=True, interpolation=interp_mode), T.ToTensor(), T.Normalize(dmean, dstd)])
                    case 'imagenet_r' | 'sdomainet':
                        transforms = T.Compose([T.Resize((IMG_SIZE + 32, IMG_SIZE + 32), antialias=True, interpolation=interp_mode), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(dmean, dstd)])
                    case _:
                        raise NotImplementedError(_current_dataset)
        case _:
            raise NotImplementedError(f"{transform_type}")

    class TargetTransform():
        def __init__(self, label_map_g2l: dict[int, tuple[int, int, int]], target_map_to_local: bool) -> None:
            self.label_map_g2l = deepcopy(label_map_g2l)  # {original_label: (taskid, local_label, global_label)}
            self.target_map_to_local = target_map_to_local

        def __call__(self, target: int):
            if self.target_map_to_local:
                return self.label_map_g2l[target][1]
            else:
                return self.label_map_g2l[target][2]

        def __repr__(self) -> str:
            label_map = {k: v[1] if self.target_map_to_local else v[2] for k, v in self.label_map_g2l.items()}
            _repr = str(label_map)
            return _repr

    target_transforms = TargetTransform(GVM.label_map_g2l, target_map_to_local)

    _mode = 'train' if training else 'eval'
    dataset = ClassIncremantalDataset(GVM.path_data_dict[_mode], task_classes, transforms, target_transforms, expand_times=expand_times, return_index=False, sample_type=GVM.args.sample_type)

    return dataset
