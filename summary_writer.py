from abc import ABC, abstractmethod
import wandb

class SummaryWriter(ABC):
  @abstractmethod
  def register(self, **args):
    pass
  
  @abstractmethod
  def track_object(self, **args):
    pass
  
  def finalize():
    pass

class WandBSummaryWritter(SummaryWriter):
  def register(self, project, config):
    wandb.init(
      project=project,
      config=config
    )
    
  def track_object(self, obj):
    wandb.log(obj)
    
  def finalize():
    wandb.finish()