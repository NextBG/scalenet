'''
Dataset for ScaleNet
'''

import io
import os
import lmdb
import random
import pickle
from PIL import Image
from tqdm import tqdm
from typing import Tuple
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class ScaleNetDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str,
            dataset_type: str,
            max_scale_factor: float,
            offset_min: int,
            offset_max: int,
            num_samples: int,
            seed: int,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.trajs_path = os.path.join(self.dataset_dir, "trajectories")
        self.max_scale_factor = max_scale_factor
        self.offset_min = offset_min
        self.offset_max = offset_max
        self.num_samples = num_samples

        # Random seed
        torch.manual_seed(seed)
        random.seed(seed)

        # Partition
        partition_file = os.path.join(self.dataset_dir, "partitions", f"{dataset_type}.txt")
        with open(partition_file, "r") as f:
            self.traj_names = f.read().splitlines()

        # Build index
        self.idx_to_data = self._build_index()

        # Cache
        self.cache_file = os.path.join(self.dataset_dir, "cache_640x360.lmdb")
        self.lmdb_env = None

    def _build_index(self):
        samples_idx = []

        for traj_name in tqdm(
            self.traj_names, 
            desc="Building index",
            bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            ):

            frames_dir = os.path.join(self.trajs_path, traj_name, "frames")
            traj_len = len(os.listdir(frames_dir))

            for curr_t in range(traj_len - self.offset_max):
                for offset in range(self.offset_min, self.offset_max + 1, 2):
                    for _ in range(self.num_samples):
                        rand_scale = random.uniform(0.0, 1.0)
                        samples_idx.append((traj_name, curr_t, curr_t + offset, rand_scale))

        return samples_idx

    def _get_img(self, traj_name: str, idx_0: int, idx_1: int) -> Tuple[torch.tensor, torch.tensor]:
        # Open cache
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(self.cache_file, map_size=2**40, readonly=True) # 1TB cache

        # Get images
        with self.lmdb_env.begin() as txn:
            img_0_buf = txn.get(f"{traj_name}_{idx_0:06d}".encode())
            img_0_bytes = io.BytesIO(bytes(img_0_buf))
            img_1_buf = txn.get(f"{traj_name}_{idx_1:06d}".encode())
            img_1_bytes = io.BytesIO(bytes(img_1_buf))
        img_0 = Image.open(img_0_bytes)
        img_1 = Image.open(img_1_bytes)

        # Random crop and resize
        i_w, i_h = img_0.size
        o_w, o_h = 256, 144
        crop_w = random.randint(o_w, i_w)
        crop_h = int(crop_w * o_h / o_w)
        if crop_h > i_h:
            crop_h = i_h
            crop_w = int(crop_h * o_w / o_h)
        
        # Crop and resize
        top = (o_h - crop_h) // 2
        left = (o_w - crop_w) // 2
        img_0 = TF.resized_crop(img_0, top, left, crop_h, crop_w, (o_h, o_w))
        img_1 = TF.resized_crop(img_1, top, left, crop_h, crop_w, (o_h, o_w))

        # To tensor
        img_0 = TF.to_tensor(img_0)
        img_1 = TF.to_tensor(img_1)

        return img_0, img_1 
    
    def _quat_multiply(self, q0: torch.tensor, q1:torch.tensor) -> torch.tensor:
        """Multiply two quaternions."""
        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = q1
        w = w0*w1 - x0*x1 - y0*y1 - z0*z1
        x = w0*x1 + x0*w1 + y0*z1 - z0*y1
        y = w0*y1 - x0*z1 + y0*w1 + z0*x1
        z = w0*z1 + x0*y1 - y0*x1 + z0*w1
        return torch.tensor([w, x, y, z])
    
    def _vec_to_local_coord(self, v_glob: torch.tensor, q_norm: torch.tensor) -> torch.tensor:
        """Transform a vector from global to local coordinates."""
        w, x, y, z = q_norm
        q_conj = torch.tensor([w, -x, -y, -z])
        v_quat = torch.cat([torch.tensor([0]), v_glob])
        v_local = self._quat_multiply(self._quat_multiply(q_conj, v_quat), q_norm)
        return v_local[1:]

    def _get_transform(self, traj_name: str, idx_0: int, idx_1: int) -> torch.tensor:
        traj_path = os.path.join(self.trajs_path, traj_name, f'traj_est.pkl')
        with open(traj_path, 'rb') as f:
            traj_raw = pickle.load(f)
        traj = traj_raw["raw"]

        trans_0 = torch.tensor(traj[idx_0], dtype=torch.float32)
        trans_1 = torch.tensor(traj[idx_1], dtype=torch.float32)

        pos_0 = trans_0[:3] # [x, y, z]
        pos_1 = trans_1[:3] # [x, y, z]
        pos_delta = pos_1 - pos_0

        # Local position
        quat_0 = trans_0[3:] # [qw, qx, qy, qz]
        pos_delta = self._vec_to_local_coord(pos_delta, quat_0)

        return  pos_delta

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        traj_name, idx_0, idx_1, n_scale = self.idx_to_data[index]

        obs_0, obs_1 = self._get_img(traj_name, idx_0, idx_1)
        gt_trans = self._get_transform(traj_name, idx_0, idx_1)

        # Generate random scale factor float[0, max_scale_factor]
        scale_factor = n_scale * self.max_scale_factor
        scaled_trans = gt_trans * scale_factor

        return (
            obs_0,
            obs_1,
            scaled_trans,
            gt_trans,
            torch.tensor([n_scale])
        )
    
    def __len__(self) -> int:
        return len(self.idx_to_data)

if __name__ == "__main__":
    dataset = ScaleNetDataset(
        "/home/caoruixiang/datasets_mnt/scalenet", 
        "train",
        10.0,
        1,
        10,
        0,
    )

    o0, o1, trans, n_gt_scale = dataset[100]

    print(o0.shape)
    print(o1.shape)
    print(trans)
    print(n_gt_scale)

    plt.imshow(o0.permute(1, 2, 0))
    plt.savefig("o0.png")
    plt.imshow(o1.permute(1, 2, 0))
    plt.savefig("o1.png")