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
      downoad()
  
  def download(self):
    pass    

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
    
def download_depth(self):
  url = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
  
                
def _unpack(file_path):
  ...
def _replace_folder(src, dst):
  ...
def _rename_files(dst, name_mapper: callable):
  pass