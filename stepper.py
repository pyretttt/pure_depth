import random

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

from summary_writer import SummaryWriter
from metrics import (
  absolute_relative_mean_error,
  root_mean_squared_error,
  average_absolute_log_error,
  threshold_acc
)

class Stepper:
  def __init__(
    self,
    model: nn.Module, 
    optim: Optimizer, 
    loss_fn: callable, 
    summary_writter: SummaryWriter = None
  ):
    self.model: nn.Module = model
    self.optim: Optimizer = optim
    self.loss_fn = loss_fn
    self.summary_writter: SummaryWriter = summary_writter
    
    self._train_loader = None
    self._val_loader = None
    
    self.total_epochs = 0
    
    self._train_step_fn = self._make_train_step_fn()
    self._val_step_fn = self._make_val_step_fn()
    
    self.train_losses = []
    self.val_losses = []
    
    self._lr_scheduler = None
    self._batch_lr_scheduler = None
    
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
  def to(self, device):
    self.model.to(device)
    self.device = device

  def set_loaders(self, train_loader, val_loader):
    self._train_loader = train_loader
    self._val_loader = val_loader

  def set_lr_schedulers(self, lr_scheduler, over_batch: bool):
    if lr_scheduler.optimizer != self.optim:
      raise RuntimeError("Scheduler initalized uncorrectly")
    
    if over_batch:
      self._batch_lr_scheduler = lr_scheduler
    else:
      self._lr_scheduler = lr_scheduler

    
  def _make_train_step_fn(self):
    def train_step(X_train, y_train):
      self.model.train()
      
      yhat = self.model(X_train)
      
      loss = self.loss_fn(yhat, y_train)
      loss.backward()
      
      self.optim.step()
      self.optim.zero_grad()
      
      return loss.item()
    
    return train_step
  
  def _make_val_step_fn(self):
    def val_step(X_val, y_val):
      self.model.eval()
      yhat = self.model(X_val)
      
      loss = self.loss_fn(yhat, y_val)
      
      return loss.item()

    return val_step
  
  def _mini_batch(self, val: bool):
    if val:
      step_fn = self._val_step_fn
      data_loader = self._val_loader
      desc = 'validation step'
      scheduler = None
      if isinstance(self.loss_fn, nn.Module):
        self.loss_fn.eval()
    else:
      step_fn = self._train_step_fn
      data_loader = self._train_loader
      desc = 'train step'
      scheduler = self._schedule_batch_lr
      if isinstance(self.loss_fn, nn.Module):
        self.loss_fn.train()
      
    running_loss = []
    with tqdm(data_loader, desc=desc, unit='batch') as t_epoch:
      for X_batch, y_batch in t_epoch:
        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
        loss = step_fn(X_batch, y_batch)
        running_loss.append(loss)

        if (scheduler := scheduler):
          scheduler()

        t_epoch.set_postfix(loss=loss)

    
    loss = np.mean(running_loss)
    return loss
  
  def set_seed(self, seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        self.train_loader.sampler.generator.manual_seed(seed)
    except AttributeError:
        pass
  
  def train(self, n_epochs, seed=43):
    self.set_seed(seed)
    
    for epoch in range(n_epochs):
      train_loss = self._mini_batch(val=False)
      self.train_losses.append(train_loss)
      
      with torch.no_grad():
        val_loss = self._mini_batch(val=True)
        self.val_losses.append(val_loss)
        
      self._schedule_epoch_lr(val_loss=val_loss)
      
      self.total_epochs += 1
      
      if (sw := self.summary_writter):
        sw.track_object({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch': epoch,
            'lr': self.optim.param_groups[0]['lr'],
            'alphas': {
              'beta': self.loss_fn.beta,
              'sil': self.loss_fn.z_scores[0].item(), # TODO: Fix later
              'grad_l1': self.loss_fn.z_scores[1].item(),
              'dssim': self.loss_fn.z_scores[2].item(),
            }
          } | self.gather_metrics()
        )
        
  def _schedule_epoch_lr(self, **kwargs):
   if (lr_scheduler := self._lr_scheduler):
    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
      lr_scheduler.step(kwargs['val_loss'])
    else:
      lr_scheduler.step()

    if (sw := self.summary_writter):
      sw.track_object({
        'epoch_lr': lr_scheduler.get_last_lr()
      })
      
  def _schedule_batch_lr(self, **kwargs):
    if (lr_scheduler := self._batch_lr_scheduler):
      lr_scheduler.step()

  def save_checkpoint(self, file_name: str):
    checkpoint = {
     'total_epochs': self.total_epochs,
     'train_losses': self.train_losses,
     'val_losses': self.val_losses,
     'model_state_dict': self.model.state_dict(),
     'optim_state_dict': self.optim.state_dict()
    }
    torch.save(checkpoint, file_name)
    
  def load_checkpoint(self, file_name):
    checkpoint = torch.load(file_name)
    self.total_epochs = checkpoint['total_epochs']
    self.train_losses = checkpoint['train_losses']
    self.val_losses = checkpoint['val_losses']
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optim.load_state_dict(checkpoint['optim_state_dict'])
  
  def predict(self, x_batch):
    '''
    Should be called inside torch.no_grad()
    '''
    x_batch = x_batch.to(self.device)
    self.model.eval()
    pred = self.model(x_batch)
    return pred

  def gather_metrics(self, validation: bool = True):
    running_metrics: np.array = None

    data_loader = (self._train_loader, self._val_loader)[validation]
    
    self.model.eval()
    with torch.no_grad():
      for x_batch, y_batch in data_loader:
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
        pred = self.model(x_batch)

        metrics = np.array((
          absolute_relative_mean_error(pred, y_batch).item(),
          root_mean_squared_error(pred, y_batch).item(),
          average_absolute_log_error(pred, y_batch).item(),
          *(
            metric.item() 
            for metric 
            in threshold_acc(pred, y_batch)
          )
        )).reshape(1, -1)
        
        if running_metrics is None:
          running_metrics = metrics
        else:
          running_metrics = np.vstack((running_metrics, metrics))
    
    mean_metrics = np.mean(running_metrics, axis=0)
    return {
      'evaluation_metrics': {
        'absolute_relative_mean_error': mean_metrics[0],
        'root_mean_squared_error': mean_metrics[1],
        'average_absolute_log_error': mean_metrics[2],
        'threshold_accuracy_delta_1': mean_metrics[3],
        'threshold_accuracy_delta_2': mean_metrics[4],
        'threshold_accuracy_delta_3': mean_metrics[5]
      }
    }

  def capture_gradients(self, layers_to_hook: list[str]):
    grads = dict()
    def capturer(name, param):
      def hook_fn(grad):
        grads[name][param].append(grad.tolist())
        return None
      return hook_fn
    
    hooks = []
    for name, layer in self.model.named_modules():
      if name in layers_to_hook:
        grads.update({name: {}})
        for param_id, p in layer.named_parameters():
          if p.requires_grad:
            grads[name].update({param_id: []})
            
            hooks.append(
              p.register_hook(capturer(name, param_id))
            )
    
    return grads, hooks        
    
class MultiTermLossStepper(Stepper):
  def _make_train_step_fn(self):
    def train_step(X_train, y_train):
      self.model.train()
      
      yhat = self.model(X_train)
      
      l0, l1 = self.loss_fn(yhat, y_train)
      l0.backward(retain_graph=True)
      self.optim.step()
      self.optim.zero_grad()
      l1.backward()
      self.optim.step()
      self.optim.zero_grad()
      
      return [l0, l1]
    
    return train_step
  
  def _make_val_step_fn(self):
    def val_step(X_val, y_val):
      self.model.eval()
      yhat = self.model(X_val)
      
      losses = self.loss_fn(yhat, y_val)
      
      return [l.item() for l in losses]

    return val_step
  
  def _mini_batch(self, val: bool):
    if val:
      step_fn = self._val_step_fn
      data_loader = self._val_loader
      desc = 'validation step'
    else:
      step_fn = self._train_step_fn
      data_loader = self._train_loader
      desc = 'train step'
      
    running_loss = []
    with tqdm(data_loader, desc=desc, unit='batch') as t_epoch:
      for X_batch, y_batch in t_epoch:
        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
        losses = step_fn(X_batch, y_batch)
        running_loss.append(losses)

        t_epoch.set_postfix(losses=losses)
    
    loss = np.mean(running_loss, axis=0)
    return loss