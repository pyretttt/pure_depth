import random

import numpy as np
import torch
from tqdm import tqdm

class Stepper:
  def __init__(self, model, optim, loss_fn):
    self.model = model
    self.optim = optim
    self.loss_fn = loss_fn
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    self._train_step_fn = self._make_train_step_fn()
    self._val_step_fn = self._make_val_step_fn()
    
    self.train_losses = []
    self.val_losses = []
    
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
      
    return np.mean(running_loss)
  
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