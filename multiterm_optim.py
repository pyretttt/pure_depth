from typing import Dict, Any, Union
import math

import torch
from torch.optim.optimizer import Optimizer


class MultiTermAdam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999, 0.9),
        eps=1e-8,
        weight_decay=0
    ):
      defaults = dict(
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
      )
      super(MultiTermAdam, self).__init__(params, defaults)
      self.total_grad = 0
      self.training_step = 0

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(
        self, multiterm_loss: Union[list, tuple], ranks, closure=None
    ):
      loss = None
      if closure:
        with torch.enabled_grad():
          loss = closure()

      self.update_weights(multiterm_loss, ranks)

      return loss

    def update_weights(self, loss_array, ranks):
        for loss_index, loss in enumerate(loss_array):
          should_retain = loss_index != len(loss_array) - 1
          loss.backward(retain_graph=should_retain)
          for group in self.param_groups:
            for p in group['params']:

              if not p.requires_grad or p.grad is None:
                continue

              if p.grad.is_sparse:
                raise RuntimeError('Sparse gradients aren\'t supported')

              state = self.state[p]

              # State initialization
              if len(state) == 0:
                state['step'] = 1
                
                ones = torch.ones(len(loss_array)).to(p.device)
                p.norms = ones
                for j, _ in enumerate(loss_array):
                  # Exponential moving average of gradient values
                  state['exp_avg'+str(j)] = torch.zeros_like(p.data)
                  # Exponential moving average of squared gradient values
                  state['exp_avg_sq'+str(j)] = torch.zeros_like(p.data)

              beta1, beta2, beta3 = group['betas']

              # normalize the norm of current loss gradients to be the same as the anchor
              if state['step'] == 1:
                p.norms[loss_index] = torch.norm(p.grad)
              else:
                p.norms[loss_index] = (p.norms[loss_index]*beta3) + ((1-beta3)*torch.norm(p.grad))
              
              if p.norms[loss_index] > 1e-10:
                for anchor_index in range(len(loss_array)):
                  if p.norms[anchor_index] > 1e-10:
                    p.grad = ranks[loss_index] * p.norms[anchor_index] * p.grad / p.norms[loss_index]
                    break

              exp_avg, exp_avg_sq = state['exp_avg'+str(loss_index)], state['exp_avg_sq'+str(loss_index)]

              bias_correction1 = 1 - beta1 ** state['step']
              bias_correction2 = 1 - beta2 ** state['step']
              if loss_index == len(loss_array) - 1:
                state['step'] += 1

              if group['weight_decay'] != 0:
                p.grad = p.grad.add(p, alpha=group['weight_decay'])

              exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
              exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

              denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

              step_size = group['lr'] / bias_correction1

              if loss_index == 0 or not hasattr(p, 'exp_avg'):
                p.exp_avg = [exp_avg]
                p.denom = [denom]
                p.step_size = [step_size]
              else:
                p.exp_avg.append(exp_avg)
                p.denom.append(denom)
                p.step_size.append(step_size)
              
              if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        for group in self.param_groups:
          for p in group['params']:
            if not hasattr(p, 'exp_avg'):
              continue
            
            temp = 0
            max_denom = p.denom[0]
            for index in range(1, len(p.exp_avg)):
                max_denom = torch.max(max_denom, p.denom[index])

            for index in range(len(p.exp_avg)):
              update_step = -p.step_size[index]*(p.exp_avg[index]/max_denom)
              temp += update_step
            p.add_(temp)

        self.training_step += 1

        
if __name__ == '__main__':
  # Debug
  from models.res_refinet import Resnet101RefineNet
  from loss import nyuv2_multiterm_loss_fn
  
  model = Resnet101RefineNet()
  model.train()
  optim = MultiTermAdam(model.parameters())
  
  x = torch.randn(1, 3, 224, 224)
  y_true = torch.nn.functional.relu(torch.randn(1, 1, 224, 224))
  losses = nyuv2_multiterm_loss_fn(model(x), y_true)
  
  optim.step(losses, [1] * len(losses))
  
  