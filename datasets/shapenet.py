import os
import pickle
from typing import Any, Dict, List, Tuple
import lmdb
import cv2
import imageio
import numpy as np
from PIL import Image
import Imath
import OpenEXR
from pathlib import Path
import io
import gzip
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from functools import partial

from guided_diffusion import logger

def load_dataset(
    file_path: str = "",
    reso: int = 64,
    reso_encoder: int = 224, 
    batch_size: int = 1,
    num_workers: int = 6,
    load_depth: bool = False,
    preprocess = None,
    imgnet_normalize: bool = True,
    dataset_size: int = -1,
    trainer_name: str = 'input_rec',
    use_lmdb: bool = False,
    infi_sampler: bool = True
) -> DataLoader:
    """
    Load dataset for training/evaluation.
    
    Args:
        file_path: Path to dataset
        reso: Resolution of output images
        reso_encoder: Resolution for encoder input
        batch_size: Batch size
        num_workers: Number of data loading workers
        load_depth: Whether to load depth maps
        preprocess: Optional preprocessing function
        imgnet_normalize: Whether to normalize with ImageNet stats
        dataset_size: Number of samples to use (-1 for all)
        trainer_name: Name of trainer ('input_rec' or 'nv')
        use_lmdb: Whether to use LMDB dataset
        infi_sampler: Whether to use infinite sampler
        
    Returns:
        DataLoader for the dataset
    """
    if use_lmdb:
        logger.log('Using LMDB dataset')
        if 'nv' in trainer_name:
            dataset_cls = LMDBDataset_NV_Compressed
        else:
            dataset_cls = LMDBDataset_MV_Compressed
    else:
        if 'nv' in trainer_name:
            dataset_cls = NovelViewDataset
        else:
            dataset_cls = MultiViewDataset

    dataset = dataset_cls(
        file_path,
        reso,
        reso_encoder,
        test=False,
        preprocess=preprocess,
        load_depth=load_depth,
        imgnet_normalize=imgnet_normalize,
        dataset_size=dataset_size
    )

    logger.log(f'Dataset class: {trainer_name}, size: {len(dataset)}')
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        shuffle=False
    )
    return loader

def load_data(
    file_path: str = "",
    reso: int = 64,
    reso_encoder: int = 224,
    batch_size: int = 1,
    num_workers: int = 6,
    load_depth: bool = False,
    preprocess = None,
    imgnet_normalize: bool = True,
    dataset_size: int = -1,
    trainer_name: str = 'input_rec',
    use_lmdb: bool = False,
    infi_sampler: bool = True
):
    """
    Load dataset for training.
    """
    if use_lmdb:
        logger.log('Using LMDB dataset')
        if 'nv' in trainer_name:
            dataset_cls = LMDBDataset_NV_Compressed
        else:
            dataset_cls = LMDBDataset_MV_Compressed
    else:
        if 'nv' in trainer_name:
            dataset_cls = NovelViewDataset
        else:
            dataset_cls = MultiViewDataset

    dataset = dataset_cls(
        file_path,
        reso,
        reso_encoder,
        test=False,
        preprocess=preprocess,
        load_depth=load_depth,
        imgnet_normalize=imgnet_normalize,
        dataset_size=dataset_size
    )

    logger.log(f'Dataset class: {trainer_name}, size: {len(dataset)}')

    if infi_sampler:
        train_sampler = DistributedSampler(
            dataset=dataset,
            shuffle=True,
            drop_last=True
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            sampler=train_sampler
        )

        while True:
            yield from loader

    else:
        return dataset

def load_eval_rays(
    file_path: str = "",
    reso: int = 64,
    reso_encoder: int = 224,
    imgnet_normalize: bool = True
) -> List[torch.Tensor]:
    """
    Load camera rays for evaluation.
    """
    dataset = MultiViewDataset(
        file_path,
        reso,
        reso_encoder,
        imgnet_normalize=imgnet_normalize
    )
    pose_list = dataset.single_pose_list
    ray_list = []
    for pose_fname in pose_list:
        c2w = dataset.get_c2w(pose_fname).reshape(16)
        c = torch.cat([c2w, torch.from_numpy(dataset.intrinsics)], dim=0).reshape(25)
        ray_list.append(c)

    return ray_list

def load_eval_data(
    file_path: str = "",
    reso: int = 64,
    reso_encoder: int = 224,
    batch_size: int = 1,
    num_workers: int = 1,
    load_depth: bool = False,
    preprocess = None,
    imgnet_normalize: bool = True,
    interval: int = 1,
    **kwargs
) -> DataLoader:
    """
    Load dataset for evaluation.
    """
    dataset = MultiViewDataset(
        file_path,
        reso,
        reso_encoder,
        preprocess=preprocess,
        load_depth=load_depth,
        test=True,
        imgnet_normalize=imgnet_normalize,
        interval=interval
    )
    print(f'Eval dataset size: {len(dataset)}')
    
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )
    return loader

def load_memory_data(
    file_path: str = "",
    reso: int = 64,
    reso_encoder: int = 224,
    batch_size: int = 1,
    num_workers: int = 1,
    load_depth: bool = True,
    preprocess = None,
    imgnet_normalize: bool = True
):
    """
    Load single instance into memory for faster training.
    """
    dataset = MultiViewDataset(
        file_path,
        reso,
        reso_encoder,
        preprocess=preprocess,
        load_depth=True,
        test=False,
        overfitting=True,
        imgnet_normalize=imgnet_normalize,
        overfitting_bs=batch_size
    )
    logger.log(f'Memory dataset size: {len(dataset)}')
    
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        num_workers=num_workers,
        drop_last=False,
        shuffle=False
    )

    all_data = next(iter(loader))
    while True:
        start_idx = np.random.randint(0, len(dataset) - batch_size + 1)
        yield {
            k: v[start_idx:start_idx + batch_size]
            for k, v in all_data.items()
        }


class MultiViewDataset(Dataset):
    """
    Dataset for loading multi-view images and camera parameters.
    """

    def __init__(
        self,
        file_path: str,
        reso: int,
        reso_encoder: int,
        preprocess = None,
        classes: bool = False,
        load_depth: bool = False,
        test: bool = False,
        scene_scale: float = 1,
        overfitting: bool = False,
        imgnet_normalize: bool = True,
        dataset_size: int = -1,
        overfitting_bs: int = -1,
        interval: int = 1
    ):
        self.file_path = file_path
        self.overfitting = overfitting
        self.scene_scale = scene_scale
        self.reso = reso
        self.reso_encoder = reso_encoder
        self.classes = False
        self.load_depth = load_depth
        self.preprocess = preprocess
        assert not self.classes, "Class conditioning not supported"

        dataset_name = Path(self.file_path).stem.split('_')[0]
        self.dataset_name = dataset_name

        if test:
            if dataset_name == 'chair':
                self.ins_list = sorted(os.listdir(self.file_path))[1:2]
            else:
                self.ins_list = sorted(os.listdir(self.file_path))[0:1]
        else:
            ins_list_file = Path(self.file_path).parent / f'{dataset_name}_train_list.txt'
            assert ins_list_file.exists(), 'Training list required for ShapeNet'
            with open(ins_list_file, 'r') as f:
                self.ins_list = [name.strip() for name in f.readlines()][:dataset_size]

        if overfitting:
            self.ins_list = self.ins_list[:1]

        self.rgb_list = []
        self.pose_list = []
        self.depth_list = []
        self.data_ins_list = []
        self.instance_data_length = -1
        
        for ins in self.ins_list:
            cur_rgb_path = os.path.join(self.file_path, ins, 'rgb')
            cur_pose_path = os.path.join(self.file_path, ins, 'pose')

            cur_all_fname = sorted([
                t.split('.')[0] for t in os.listdir(cur_rgb_path)
                if 'depth' not in t
            ][::interval])
            
            if self.instance_data_length == -1:
                self.instance_data_length = len(cur_all_fname)
            else:
                assert len(cur_all_fname) == self.instance_data_length

            if test:
                mid_index = len(cur_all_fname) // 3 * 2
                cur_all_fname.insert(0, cur_all_fname[mid_index])

            self.pose_list += [
                os.path.join(cur_pose_path, fname + '.txt')
                for fname in cur_all_fname
            ]
            self.rgb_list += [
                os.path.join(cur_rgb_path, fname + '.png')
                for fname in cur_all_fname
            ]
            self.depth_list += [
                os.path.join(cur_rgb_path, fname + '_depth0001.exr')
                for fname in cur_all_fname
            ]
            self.data_ins_list += [ins] * len(cur_all_fname)

        if overfitting:
            assert overfitting_bs != -1
            self.pose_list = self.pose_list[::50//overfitting_bs+1]
            self.rgb_list = self.rgb_list[::50//overfitting_bs+1]
            self.depth_list = self.depth_list[::50//overfitting_bs+1]

        self.single_pose_list = [
            os.path.join(cur_pose_path, fname + '.txt')
            for fname in cur_all_fname
        ]

        transformations = [
            transforms.ToTensor(),
        ]
        if imgnet_normalize:
            transformations.append(
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)
                )
            )
        else:
            transformations.append(
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5)
                )
            )

        self.normalize = transforms.Compose(transformations)

        fx = fy = 525
        cx = cy = 256
        factor = self.reso / (cx * 2)
        self.fx = fx * factor
        self.fy = fy * factor
        self.cx = cx * factor
        self.cy = cy * factor

        self.cx /= self.reso
        self.cy /= self.reso
        self.fx /= self.reso
        self.fy /= self.reso

        intrinsics = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ]).reshape(9)
        self.intrinsics = intrinsics

    def __len__(self) -> int:
        return len(self.rgb_list)

    def get_c2w(self, pose_fname: str) -> torch.Tensor:
        """Load camera-to-world matrix from file."""
        with open(pose_fname, 'r') as f:
            cam2world = f.readline().strip()
            cam2world = [float(t) for t in cam2world.split(' ')]
        c2w = torch.tensor(cam2world, dtype=torch.float32).reshape(4, 4)
        return c2w

    def gen_rays(self, c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate camera rays from camera matrix."""
        self.h = self.reso
        self.w = self.reso
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
            indexing='ij'
        )
        xx = (xx - self.cx) / self.fx
        yy = (yy - self.cy) / self.fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(1, -1, 3, 1)
        dirs = (c2w[:, None, :3, :3] @ dirs)[..., 0]

        origins = c2w[:, None, :3, 3].expand(-1, self.h * self.w, -1).contiguous()
        origins = origins.view(-1, 3)
        dirs = dirs.view(-1, 3)

        return origins, dirs

    def read_depth(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load depth map from OpenEXR file."""
        depth_path = self.depth_list[idx]
        exr = OpenEXR.InputFile(depth_path)
        header = exr.header()
        size = (
            header['dataWindow'].max.x - header['dataWindow'].min.x + 1,
            header['dataWindow'].max.y - header['dataWindow'].min.y + 1
        )
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        depth_str = exr.channel('B', FLOAT)
        depth = np.frombuffer(depth_str, dtype=np.float32).reshape(size[1], size[0])
        depth = np.nan_to_num(depth, posinf=0, neginf=0)

        def resize_depth_mask(depth_to_resize: np.ndarray, resolution: int) -> np.ndarray:
            depth_resized = cv2.resize(
                depth_to_resize,
                (resolution, resolution),
                interpolation=cv2.INTER_LANCZOS4
            )
            return depth_resized > 0

        fg_mask_reso = resize_depth_mask(depth, self.reso)
        fg_mask_sr = resize_depth_mask(depth, 128)

        return torch.from_numpy(depth), torch.from_numpy(fg_mask_reso), torch.from_numpy(fg_mask_sr)

    def load_bbox(self, mask: torch.Tensor) -> torch.Tensor:
        """Get bounding box from mask."""
        nonzero_value = torch.nonzero(mask)
        height, width = nonzero_value.max(dim=0)[0]
        top, left = nonzero_value.min(dim=0)[0]
        bbox = torch.tensor([top, left, height, width], dtype=torch.float32)
        return bbox

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_fname = self.rgb_list[idx]
        pose_fname = self.pose_list[idx]

        raw_img = imageio.imread(rgb_fname)

        if self.preprocess is None:
            img_to_encoder = cv2.resize(
                raw_img,
                (self.reso_encoder, self.reso_encoder),
                interpolation=cv2.INTER_LANCZOS4
            )
            img_to_encoder = img_to_encoder[..., :3]
            img_to_encoder = self.normalize(img_to_encoder)
        else:
            img_to_encoder = self.preprocess(Image.open(rgb_fname))

        img = cv2.resize(
            raw_img,
            (self.reso, self.reso),
            interpolation=cv2.INTER_LANCZOS4
        )

        img_sr = cv2.resize(
            raw_img,
            (128, 128),
            interpolation=cv2.INTER_LANCZOS4
        )

        img = torch.from_numpy(img)[..., :3].permute(2, 0, 1) / 127.5 - 1
        img_sr = torch.from_numpy(img_sr)[..., :3].permute(2, 0, 1) / 127.5 - 1

        c2w = self.get_c2w(pose_fname).reshape(16)
        c = torch.cat([c2w, torch.from_numpy(self.intrinsics)], dim=0).reshape(25)

        ret_dict = {
            'img_to_encoder': img_to_encoder,
            'img': img,
            'c': c,
            'img_sr': img_sr,
        }

        if self.load_depth:
            depth, depth_mask, depth_mask_sr = self.read_depth(idx)
            bbox = self.load_bbox(depth_mask)
            ret_dict.update({
                'depth': depth,
                'depth_mask': depth_mask,
                'depth_mask_sr': depth_mask_sr,
                'bbox': bbox
            })
        return ret_dict



class MultiViewDatasetforLMDB(MultiViewDataset):

    def __init__(self,
                 file_path,
                 reso,
                 reso_encoder,
                 preprocess=None,
                 classes=False,
                 load_depth=False,
                 test=False,
                 scene_scale=1,
                 overfitting=False,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 overfitting_bs=-1):
        super().__init__(file_path, reso, reso_encoder, preprocess, classes,
                         load_depth, test, scene_scale, overfitting,
                         imgnet_normalize, dataset_size, overfitting_bs)

    def __len__(self):
        return super().__len__()
        # return 100 # for speed debug

    def __getitem__(self, idx):
        # ret_dict = super().__getitem__(idx)
        rgb_fname = self.rgb_list[idx]
        pose_fname = self.pose_list[idx]
        raw_img = imageio.imread(rgb_fname)[..., :3]

        c2w = self.get_c2w(pose_fname).reshape(16)  #[1, 4, 4] -> [1, 16]
        # c = np.concatenate([c2w, self.intrinsics], axis=0).reshape(25)  # 25, no '1' dim needed.
        c = torch.cat([c2w, torch.from_numpy(self.intrinsics)],
                      dim=0).reshape(25)  # 25, no '1' dim needed.

        depth, depth_mask, depth_mask_sr = self.read_depth(idx)
        bbox = self.load_bbox(depth_mask)
        ret_dict = {
            'raw_img': raw_img,
            'c': c,
            'depth': depth,
            # 'depth_mask': depth_mask, # 64x64 here?
            'bbox': bbox
        }
        return ret_dict


def load_data_dryrun(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        #   shuffle=True,
        num_workers=6,
        load_depth=False,
        preprocess=None,
        imgnet_normalize=True):

    dataset = MultiViewDataset(file_path,
                               reso,
                               reso_encoder,
                               test=False,
                               preprocess=preprocess,
                               load_depth=load_depth,
                               imgnet_normalize=imgnet_normalize)
    print('dataset size: {}'.format(len(dataset)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )


    return loader


class NovelViewDataset(MultiViewDataset):
    """Dataset for novel view synthesis training."""

    def __init__(self,
                 file_path,
                 reso,
                 reso_encoder,
                 preprocess=None,
                 classes=False,
                 load_depth=False,
                 test=False,
                 scene_scale=1,
                 overfitting=False,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 overfitting_bs=-1):
        super().__init__(file_path, reso, reso_encoder, preprocess, classes,
                         load_depth, test, scene_scale, overfitting,
                         imgnet_normalize, dataset_size, overfitting_bs)

    def __getitem__(self, idx):
        input_view = super().__getitem__(
            idx)  # get previous input view results

        # get novel view of the same instance
        novel_view = super().__getitem__(
            (idx // self.instance_data_length) * self.instance_data_length +
            random.randint(0, self.instance_data_length - 1)
        )

        input_view.update({f'nv_{k}': v for k, v in novel_view.items()})
        return input_view


def load_data_for_lmdb(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        #   shuffle=True,
        num_workers=6,
        load_depth=False,
        preprocess=None,
        imgnet_normalize=True,
        dataset_size=-1,
        trainer_name='input_rec'):

    dataset_cls = MultiViewDatasetforLMDB

    dataset = dataset_cls(file_path,
                          reso,
                          reso_encoder,
                          test=False,
                          preprocess=preprocess,
                          load_depth=load_depth,
                          imgnet_normalize=imgnet_normalize,
                          dataset_size=dataset_size)

    logger.log('dataset_cls: {}, dataset size: {}'.format(
        trainer_name, len(dataset)))

    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
    )

    return loader, dataset.dataset_name, len(dataset)


class LMDBDataset(Dataset):
    """Base LMDB dataset."""

    def __init__(self, lmdb_path: str):
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            max_readers=32,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.num_samples = self.env.stat()['entries']

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Any:
        with self.env.begin(write=False) as txn:
            key = str(idx).encode('utf-8')
            value = txn.get(key)

        sample = pickle.loads(value)
        return sample

def resize_depth_mask(depth_to_resize: np.ndarray, resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resize depth map and get foreground mask."""
    depth_resized = cv2.resize(
        depth_to_resize,
        (resolution, resolution),
        interpolation=cv2.INTER_LANCZOS4
    )
    return depth_resized, depth_resized > 0

class LMDBDataset_MV(LMDBDataset):
    """LMDB dataset for multi-view data."""

    def __init__(
        self,
        lmdb_path: str,
        reso: int,
        reso_encoder: int,
        imgnet_normalize: bool = True,
        **kwargs
    ):
        super().__init__(lmdb_path)

        self.reso_encoder = reso_encoder
        self.reso = reso

        transformations = [
            transforms.ToTensor(),
        ]
        if imgnet_normalize:
            transformations.append(
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)
                )
            )
        else:
            transformations.append(
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5)
                )
            )

        self.normalize = transforms.Compose(transformations)

    def _post_process_sample(
        self,
        raw_img: np.ndarray,
        depth: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Process raw image and depth data."""
        img_to_encoder = cv2.resize(
            raw_img,
            (self.reso_encoder, self.reso_encoder),
            interpolation=cv2.INTER_LANCZOS4
        )
        img_to_encoder = img_to_encoder[..., :3]
        img_to_encoder = self.normalize(img_to_encoder)

        img = cv2.resize(
            raw_img,
            (self.reso, self.reso),
            interpolation=cv2.INTER_LANCZOS4
        )

        if img.shape[-1] == 4:
            alpha_mask = img[..., -1:] > 0
            img = alpha_mask * img[..., :3] + (1-alpha_mask) * np.ones_like(img[..., :3]) * 255

        img = torch.from_numpy(img)[..., :3].permute(2, 0, 1) / 127.5 - 1
        img_sr = torch.from_numpy(raw_img)[..., :3].permute(2, 0, 1) / 127.5 - 1

        depth_reso, fg_mask_reso = resize_depth_mask(depth, self.reso)

        return {
            'img_to_encoder': img_to_encoder,
            'img': img,
            'depth_mask': fg_mask_reso,
            'img_sr': img_sr,
            'depth': depth_reso,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        return self._post_process_sample(sample['raw_img'], sample['depth'])

def load_bytes(inp_bytes: bytes, dtype: np.dtype, shape: Tuple[int, ...]) -> np.ndarray:
    """Load numpy array from bytes."""
    return np.frombuffer(inp_bytes, dtype=dtype).reshape(shape).copy()

def decompress_and_open_image_gzip(compressed_data: bytes, is_img: bool = False):
    """Decompress gzipped image data."""
    decompressed_data = gzip.decompress(compressed_data)
    if is_img:
        image = imageio.v3.imread(io.BytesIO(decompressed_data)).copy()
        return image
    return decompressed_data

def decompress_array(compressed_data: bytes, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Decompress gzipped array data."""
    decompressed_data = gzip.decompress(compressed_data)
    return load_bytes(decompressed_data, dtype, shape)

class LMDBDataset_MV_Compressed(LMDBDataset_MV):
    """LMDB dataset with compressed data."""

    def __init__(
        self,
        lmdb_path: str,
        reso: int,
        reso_encoder: int,
        imgnet_normalize: bool = True,
        **kwargs
    ):
        super().__init__(lmdb_path, reso, reso_encoder, imgnet_normalize, **kwargs)
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8')) - 40

        self.load_image_fn = partial(decompress_and_open_image_gzip, is_img=True)

    def __len__(self) -> int:
        return self.length

    def _load_lmdb_data(
        self,
        idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load compressed data from LMDB."""
        with self.env.begin(write=False) as txn:
            raw_img_key = f'{idx}-raw_img'.encode('utf-8')
            raw_img = self.load_image_fn(txn.get(raw_img_key))

            depth_key = f'{idx}-depth'.encode('utf-8')
            depth = decompress_array(txn.get(depth_key), (512,512), np.float32)

            c_key = f'{idx}-c'.encode('utf-8')
            c = decompress_array(txn.get(c_key), (25,), np.float32)

            bbox_key = f'{idx}-bbox'.encode('utf-8')
            bbox = decompress_array(txn.get(bbox_key), (4,), np.float32)

        return raw_img, depth, c, bbox

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_img, depth, c, bbox = self._load_lmdb_data(idx)

        return {
            **self._post_process_sample(raw_img, depth),
            'c': c,
            'bbox': bbox*(self.reso/64.0),
        }

class LMDBDataset_NV_Compressed(LMDBDataset_MV_Compressed):
    """LMDB dataset for novel view synthesis with compressed data."""

    def __init__(
        self,
        lmdb_path: str,
        reso: int,
        reso_encoder: int,
        imgnet_normalize: bool = True,
        **kwargs
    ):
        super().__init__(lmdb_path, reso, reso_encoder, imgnet_normalize, **kwargs)
        self.instance_data_length = 50

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_view = super().__getitem__(idx)

        try:
            novel_view = super().__getitem__(
                (idx // self.instance_data_length) * self.instance_data_length +
                random.randint(0, self.instance_data_length - 1)
            )
        except Exception as e:
            raise NotImplementedError(idx)

        assert input_view['ins_name'] == novel_view['ins_name'], 'Novel view must be from same instance'

        input_view.update({f'nv_{k}': v for k, v in novel_view.items()})
        return input_view