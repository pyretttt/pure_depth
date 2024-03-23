import random

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

from summary_writer import SummaryWriter

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

    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
  def to(self, device):
    self.model.to(device)
    self.device = device

  def set_loaders(self, train_loader, val_loader):
    self._train_loader = train_loader
    self._val_loader = val_loader
    
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
    else:
      step_fn = self._train_step_fn
      data_loader = self._train_loader
      desc = 'train step'
      
    running_loss = []
    with tqdm(data_loader, desc=desc, unit='batch') as t_epoch:
      for X_batch, y_batch in t_epoch:
        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
        loss = step_fn(X_batch, y_batch)
        running_loss.append(loss)

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
      
      self.total_epochs += 1
      
      if (sw := self.summary_writter):
        sw.track_object({
          'train_loss': train_loss,
          'val_loss': val_loss,
          'epoch': epoch,
        })

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