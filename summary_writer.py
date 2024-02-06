from abc import ABC, abstractmethod

import wandb

class SummaryWriter(ABC):
  @abstractmethod
  def register(self, **args):
    pass
  
  @abstractmethod
  def track_object(self, **args):
    pass
  
  def finalize(self):
    pass

class WandBSummaryWritter(SummaryWriter):
  def __init__(self, addon_stream = None):
    self.addon_stream = addon_stream
  
  def register(self, project, config):
    wandb.init(
      project=project,
      config=config
    )
    
  def track_object(self, obj):
    wandb.log(obj)
    if (stream := self.addon_stream):
      print(obj, file=stream)
    
  def finalize(self):
    wandb.finish()