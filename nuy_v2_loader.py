import os
import sys
import h5py
import torch
import shutil
import random
import tarfile
import zipfile
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

_URL = "http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz"

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
      
  if not os.path.exists(tar.rstrip(".gz")) and os.path.exists(tar):
    _unpack(tar, root_dir)
  else:
    print("Failed to load tar file")
      
  print("Finished loading dataset")

def _unpack(file_path, dst):
  if file_path.endswith(".gz"):
    tar = tarfile.open(file_path, "r:gz")
    tar.extractall(dst)
    tar.close()
  else:
    print("Failed to unpack")
      
def _replace_folder(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)
    
def _rename_files(dst, name_mapper: callable):
    imgs_old = os.listdir(dst)
    imgs_new = [name_mapper(file) for file in imgs_old]
    for img_old, img_new in zip(imgs_old, imgs_new):
        shutil.move(os.path.join(dst, img_old), os.path.join(dst, img_new))
        
def _create_depth_files(mat_file: str, root: str, train_ids: list):
  os.mkdir(os.path.join(root, "train_depth"))
  os.mkdir(os.path.join(root, "test_depth"))
  train_ids = set(train_ids)

  depths = h5py.File(mat_file, "r")["depths"]
  for i in range(len(depths)):
      img = (depths[i] * 1e4).astype(np.uint16).T
      id_ = str(i + 1).zfill(4)
      folder = "train" if id_ in train_ids else "test"
      save_path = os.path.join(root, f"{folder}_depth", id_ + ".png")
      Image.fromarray(img).save(save_path)

