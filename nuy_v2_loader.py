import os
import sys
import h5py
import torch
import random
import tarfile
import numpy as np
from pathlib import Path
from multiprocessing import Pool

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

_URL = "http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz"
_WORKERS = 2

class NYUV2Dataset(Dataset):
  def __init__(
    self, 
    root_dir: str,
    rgb_transform, 
    depth_transform,
    split: str,
    download: bool = False,
    seed: int = 42
  ):
    self._root_dir = root_dir
    self._rgb_transform = rgb_transform
    self._depth_transform = depth_transform
    self._split = split
    self.seed = 42
    
    self._download = download

    if self._download:
      _download(self._root_dir)
      
    # self._files = sorted(os.listdir(os.path.join(root_dir, f"{self._split}_rgb")))
  
  def __getitem__(self, index):
    folder = lambda name :os.path.join(self._root_dir, f"{self._split}_{name}")
    random.seed(self.seed)
    
    rgb = Image.open(os.path.join(folder("rgb"), self._files[index]))
    rgb = self.rgb_transform(rgb)
    
    depth = Image.open(os.path.join(folder("depth"), self._files[index]))
    depth = self.depth_transform(depth)
    if isinstance(depth, torch.Tensor):
      # depth png is uint16
      depth = depth.float() / 1e4

    return rgb, depth

  def __len__(self):
    return len(self._files)


def _download(root_dir):
  print("Downloading from remote")
  
  tar = os.path.join(root_dir, _URL.split("/")[-1])
  if not os.path.exists(tar):
     download_url(_URL, root_dir)
      
  if os.path.exists(tar) and not os.path.exists(tar.split(".")[0]):
    _unpack(tar, root_dir)
  else:
    print("Failed to load tar file")
 
  destination_data_folder = os.path.join(root_dir, "nyu_v2_data")
  if not os.path.exists(destination_data_folder):
    print("Started to creating raw data")
    create_files(
      tar.split(".")[0],
      os.path.join(root_dir, "nyu_v2_data")
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