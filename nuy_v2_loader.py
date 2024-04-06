import os
import sys
import random
import tarfile
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import h5py
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision.transforms import v2

from utils import random_resized_crop, horizontal_flip
from constants import INPUT_SIZE


_URL = "http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz"
_WORKERS = 2
_DATA_FOLDER = "nyu_v2_data"

_RATIOS = (INPUT_SIZE[1] / INPUT_SIZE[0], INPUT_SIZE[1] / INPUT_SIZE[0])

DEFAULT_COMMON_TRANSFORMS = [
 (horizontal_flip, 0.5),
 (partial(random_resized_crop, resized_size=INPUT_SIZE, scale=(0.85, 0.95), ratio=_RATIOS), 0.3)
]

class NYUV2Dataset(Dataset):
  def __init__(
    self, 
    root_dir: str,
    rgb_transform, 
    depth_transform,
    split: str,
    download: bool = False,
    seed: int = 42,
    max_lenght: int = None,
    common_transforms = DEFAULT_COMMON_TRANSFORMS
  ):
    self._root_dir = root_dir
    self._common_transforms = common_transforms
    self._rgb_transform = rgb_transform
    self._depth_transform = depth_transform
    self._split = split
    self.seed = seed
    self.max_lenght = max_lenght
    self.common_transforms = common_transforms
    
    self._download = download

    if self._download:
      _download(self._root_dir)
    
    self._files = sorted(
      set(
        map(
          lambda p: os.path.splitext(p)[0],
          os.listdir(os.path.join(root_dir, _DATA_FOLDER, f"{self._split}"))
        )
      )
    )
  
  def __getitem__(self, index):
    random.seed(self.seed)
    
    file_path = os.path.join(self._root_dir, _DATA_FOLDER, self._split, self._files[index])
    rgb = Image.open(file_path + ".png")
    rgb = self._rgb_transform(rgb)
    
    depth = np.load(file_path + ".npy")
    depth = self._depth_transform(depth)

    return self._transform(rgb, depth)

  def __len__(self):
    override_lenght = self.max_lenght or sys.maxsize
    return min(override_lenght, len(self._files))
  
  def _transform(self, x, y):    
    for t, p in self.common_transforms:
      if np.random.rand() > p:
        x, y = t(x, y)
    
    return x, y
      

def _download(root_dir):
  print("Downloading from remote")
  
  tar = os.path.join(root_dir, _URL.split("/")[-1])
  if not os.path.exists(tar):
     download_url(_URL, root_dir)
      
  if os.path.exists(tar) and not os.path.exists(tar.split(".")[0]):
    _unpack(tar, root_dir)
  else:
    print("Failed to load tar file")
 
  destination_data_folder = os.path.join(root_dir, _DATA_FOLDER)
  if not os.path.exists(destination_data_folder):
    print("Started to creating raw data")
    create_files(
      tar.split(".")[0],
      destination_data_folder
    )
      
  print("Finished loading dataset")

def _unpack(file_path, dst):
  if file_path.endswith(".gz"):
    tar = tarfile.open(file_path, "r:gz")
    tar.extractall(dst)
    tar.close()
  else:
    print("Failed to unpack")
    
    
def create_files(data_path, dst):
  if not os.path.exists(data_path):
    print(f"Data Folder {data_path} doesn't exists")
    sys.exit(1)
  
  assert(not os.path.exists(dst))
  train_path = os.path.join(dst, "train")
  test_path = os.path.join(dst, "test")
  os.makedirs(train_path, exist_ok=True)
  os.makedirs(test_path, exist_ok=True)
  
  assert(os.path.exists(train_path) and os.path.exists(test_path))
  
  train_extracted_path = os.path.join(data_path, "train")
  with Pool(_WORKERS) as p:
    folders = os.listdir(train_extracted_path)
    folders_paths = [os.path.join(train_extracted_path, f) for f in folders]
    
    results = [p.apply_async(create_rgb_and_depth, (folders_paths, train_path, w, _WORKERS)) 
               for w in range(_WORKERS)]
    [r.get() for r in results]

  test_extracted_path = os.path.join(data_path, "val", "official")
  for file_name in os.listdir(test_extracted_path):
      file = h5py.File(os.path.join(test_extracted_path, file_name))
      file_name_without_ext = Path(file_name).stem
      
      rgb = file["rgb"]
      rgb = np.stack([(rgb[channel]) for channel in range(len(rgb))], axis=2)
      depth = file["depth"]
      
      destination_path = os.path.join(
        test_path, 
        f"{file_name_without_ext}"
      )
      Image.fromarray(rgb).save(destination_path + ".png")
      np.save(destination_path + ".npy", depth) 

      
def create_rgb_and_depth(folders, dst_path, start, each_nth):
  print(f"{os.getpid()} started data creation")
  
  folders_to_visit = folders[start::each_nth]
  for folder_path in folders_to_visit:
    folder_name = Path(folder_path).stem
    for file_name in os.listdir(folder_path):
      file = h5py.File(os.path.join(folder_path, file_name))
      file_name_without_ext = Path(file_name).stem
      
      rgb = file["rgb"]
      rgb = np.stack([(rgb[channel]) for channel in range(len(rgb))], axis=2)
      depth = file["depth"]
    
      destination_path = os.path.join(
        dst_path, 
        f"{folder_name}_{file_name_without_ext}"
      )
      Image.fromarray(rgb).save(destination_path + ".png")
      np.save(destination_path + ".npy", depth)