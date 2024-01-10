import os
import sys
import h5py
import torch
import shutil
import random
import tarfile
import zipfile
import requests
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

class NYUV2Dataset(Dataset):
  def __init__(
    self, 
    root_dir: str,
    rgb_transform, 
    depth_transform,
    download: bool = False,
  ):
    self._root_dir = root_dir
    self._rgb_transform = rgb_transform
    self._depth_transform = depth_transform
    
    self._download = download

    if self._download:
      self.downoad(self._root_dir)
  
  def download(self):
    if _check_exists(self._root_dir):
      return
    
    download_rgb(self._root_dir)
    download_depth(self._root_dir)
    
    if not _check_exists(self._root_dir):
      raise FileNotFoundError("Failed to load datasets")
    
    print("Finished loading dataset")

def _check_exists(data_path: str):
  for split in ("train", "test"):
    for data in ["rgb", "depth"]:
      path = os.path.join(data_path, f"{split}_{data}")
      if not os.path.exists(path):
        return False
  return True
  
def download_rgb(root_dir):
  train_url = "http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz"
  test_url = "http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz"
  
  def _proc(url: str, dst: str):
    if not os.path.exists(dst):
      tar = os.path.join(root_dir, url.split("/")[-1])
      if not os.path.exists(tar):
        download_url(url, root_dir)
      if os.path.exists(tar):
        _unpack(tar)
        _replace_folder(tar.rstrip(".tgz"), dst)
        _rename_files(dst, lambda x: x.split("_")[2])
  
  _proc(train_url, os.path.join(root_dir, "train_rgb"))
  _proc(test_url, os.path.join(root_dir, "test_rgb"))
    
def download_depth(root_dir):
  url = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
  train_dst = os.path.join(root_dir, "train_depth")
  test_dst = os.path.join(root_dir, "test_depth")

                
def _unpack(file_path):
  path = file_path.rsplit(".", 1)[0]

  if file_path.endswith(".tgz"):
      tar = tarfile.open(file_path, "r:gz")
      tar.extractall(path)
      tar.close()
  elif file_path.endswith(".zip"):
      zip = zipfile.ZipFile(file_path, "r")
      zip.extractall(path)
      zip.close()
      
def _replace_folder(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)
    
def _rename_files(dst, name_mapper: callable):
    imgs_old = os.listdir(dst)
    imgs_new = [name_mapper(file) for file in imgs_old]
    for img_old, img_new in zip(imgs_old, imgs_new):
        shutil.move(os.path.join(dst, img_old), os.path.join(dst, img_new))