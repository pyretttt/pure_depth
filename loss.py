from typing import Optional, Tuple
from math import exp

import torch
from torch import nn
import torch.nn.functional as F

from constants import SCALE_INVARIANCE_ALPHA, EPSILON

NUY_V2_DATA_RANGE = 10.0
_EPS = 1e-8
_SOBEL_X = torch.tensor(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float
).unsqueeze(0)
_SOBEL_Y = torch.tensor(
    [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float
).unsqueeze(0)

_SOBEL_XY = torch.stack((_SOBEL_X, _SOBEL_Y), dim=0)


def sil_loss(y_hat, y_true, alpha=0.5, mask: Optional[torch.Tensor] = None):
    """Scale invariant loss function, with tiny epsilon"""

    mask = mask if mask is not None else torch.ones_like(y_true).bool()
    diff = torch.log(y_hat[mask] + _EPS) - torch.log(y_true[mask] + _EPS)
    loss = (diff**2).mean() - (alpha * (diff.mean() ** 2))

    return loss


def grad_l1_loss(
    y_hat: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None
):
    """L1 Gradient loss"""

    mask = mask if mask is not None else torch.ones_like(y_true).bool()
    grad_pred = F.conv2d(y_hat, weight=_SOBEL_XY.to(y_hat.device), padding=1).sum(
        dim=1, keepdim=True
    )
    grad_true = F.conv2d(y_true, weight=_SOBEL_XY.to(y_true.device), padding=1).sum(
        dim=1, keepdim=True
    )

    return torch.abs(grad_pred - grad_true)[mask].mean()


def nyuv2_loss_fn(
    y_hat: torch.Tensor, y_true: torch.Tensor, alpha=SCALE_INVARIANCE_ALPHA, data_range=NUY_V2_DATA_RANGE
):
    """SiLogLoss with scale invariant alpha + L1 Gradient loss + DSSIM"""

    mask = y_true > 0
    l1 = sil_loss(y_hat, y_true, alpha=alpha, mask=mask)
    l2 = grad_l1_loss(y_hat, y_true, mask=mask)
    l3 = dssim(y_hat, y_true, data_range=data_range)

    return (l1 + l2 + l3) / 3


def nyuv2_multiterm_loss_fn(
    y_hat: torch.Tensor, 
    y_true: torch.Tensor,
    alpha=SCALE_INVARIANCE_ALPHA, 
    data_range=NUY_V2_DATA_RANGE
):
    mask = y_true > 0
    l1 = sil_loss(y_hat, y_true, alpha=alpha, mask=mask)
    l2 = grad_l1_loss(y_hat, y_true, mask=mask)
    l3 = dssim(y_hat, y_true, data_range=data_range)

    return (
        l1, 
        l2, 
        l3
    )

def nyuv2_multiterm_loss_fn_vit(
    y_hat: torch.Tensor, y_true: torch.Tensor, alpha=SCALE_INVARIANCE_ALPHA, data_range=NUY_V2_DATA_RANGE
):
    metric, struct = y_hat
    mask = y_true > 0
    
    # struct_error = grad_l1_loss(struct, y_true, mask) + dssim(struct, y_true, data_range=data_range)
    metric_error = 0.1 * sil_loss(metric, y_true, alpha=alpha, mask=mask) \
        + dssim(metric, y_true, data_range=data_range) \
        + grad_l1_loss(metric, y_true, mask)

    #return struct_error + metric_error
    return metric_error


def _gaussian(window_size: int, sigma: float):
    g = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return g / g.sum()


def _create_window(
    window_size: Tuple[int, int] = (11, 11), sigma: float = 1.5, channel: int = 1
):
    assert len(window_size) == 2
    assert window_size[0] == window_size[1]

    window_1D = _gaussian(window_size[0], sigma).unsqueeze(1)
    window_2D = window_1D.mm(window_1D.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2D.expand(channel, 1, *window_size).contiguous()
    return window


def dssim(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    data_range: float,
    size_average: bool = True,
    kernel_size: Tuple[int, int] = (11, 11),
    kernel_sigma: float = 1.5,
    win: Optional[torch.Tensor] = _create_window(),
    K: Tuple[float, float] = (0.01, 0.03),
):
    '''Disparity of Structuring Similarity Index Measure'''
    ssim_map, _ = _ssim(
        y_hat,
        y_true,
        data_range,
        size_average=size_average,
        kernel_size=kernel_size,
        kernel_sigma=kernel_sigma,
        win=win,
        K=K,
    )

    dssim = (1 - ssim_map) / 2

    return dssim


def _ssim(
    y0: torch.Tensor,
    y1: torch.Tensor,
    data_range: float,
    size_average: bool = True,
    kernel_size: Tuple[int, int] = (11, 11),
    kernel_sigma: float = 1.5,
    win: Optional[torch.Tensor] = None,
    K: Tuple[float, float] = (0.01, 0.03),
):
    K1, K2 = K
    L = data_range
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    if win is None:
        win = _create_window(kernel_size, kernel_sigma)
    win = win.to(y0.device)

    mu1 = F.conv2d(y0, weight=win, padding=0)
    mu2 = F.conv2d(y1, weight=win, padding=0)
    mu1mu2 = mu1 * mu2
    mu1_sq = mu1**2
    mu2_sq = mu2**2

    sigma1_sq = F.conv2d(y0 * y0, weight=win, padding=0) - mu1_sq
    sigma2_sq = F.conv2d(y1 * y1, weight=win, padding=0) - mu2_sq
    sigma12 = F.conv2d(y0 * y1, weight=win, padding=0) - mu1mu2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)

    ssim_map = (2 * mu1mu2 + C1) * v1 / (v2 * (mu1_sq + mu2_sq + C1))

    if size_average:
        ssim_map = ssim_map.mean()
    else:
        ssim_map = ssim_map.mean(dim=(1, 2, 3))

    return ssim_map, cs


def _ewma(prev, current, beta):
    return (1 - beta) * current + beta * prev

def _bias_corrected_ewma(prev, current, beta, step):
    return _ewma(prev, current, beta) / (1 - beta ** step)

def _calc_corrected_ewma(values, beta):
    results = []
    
    for i, value in enumerate(values):
        try: prev_value = results[-1]
        except: prev_value = 0
            
        value = _bias_corrected_ewma(prev_value, value, beta, step=i+1)
        results.append(value)
    
    return results


class ZScoreEWMALoss(nn.Module):
    def __init__(self, loss_fn, n_functions, device, beta=0.9):
        super(ZScoreEWMALoss, self).__init__()
        self.loss_fn = loss_fn
        self.n_functions = n_functions
        self.device = device
        self.beta = beta
        
        self.ewma_loss = torch.zeros(n_functions, device=device).float()
        self.squared_differences = torch.zeros(n_functions, device=device).float()
        self.ewma_std = torch.zeros(n_functions, device=device).float()
        self.z_scores = torch.zeros(n_functions, device=device).float()
        self.alphas = torch.ones(n_functions, device=device).float() / n_functions
        
        self.t = 0
    
    def forward(self, x, y):
        loss_raw = self.loss_fn(x, y)
        if not self.training:
            return sum(loss_raw) / self.n_functions
        
        losses = torch.tensor(loss_raw, device=self.device)
        
        # Update ewma_loss
        old_ewma_loss = self.ewma_loss
        self.ewma_loss = _ewma(prev=self.ewma_loss, current=losses, beta=self.beta)
        
        # Update loss_ratio and squared_diff
        self.squared_differences += (losses - old_ewma_loss) * (losses - self.ewma_loss)
        
        # Update ewma std
        if self.t:
            std = torch.sqrt((self.squared_differences + EPSILON) / (self.t)).to(self.device)
            self.ewma_std = _ewma(prev=self.ewma_std, current=std, beta=self.beta)
            
            z_score = (losses - self.ewma_loss) / self.ewma_std
            self.z_scores = _ewma(prev=self.z_scores, current=z_score, beta=self.beta)
            
            self.alphas = torch.abs(self.z_scores) / torch.sum(torch.abs(self.z_scores))

        self.t += 1
        
        return sum([self.alphas[i] * loss for i, loss in enumerate(loss_raw)])

class ComposeEWMALoss(nn.Module):
    def __init__(self, loss_fn, n_functions, device, beta=0.9):
        super(ComposeEWMALoss, self).__init__()
        self.loss_fn = loss_fn
        self.n_functions = n_functions
        self.device = device
        self.beta = beta
        
        self.ewma_loss = torch.zeros(n_functions, device=device).float()
        self.ewma_loss_ratios = torch.zeros(n_functions, device=device).float()
        self.squared_differences = torch.zeros(n_functions, device=device).float()
        self.ewma_std = torch.zeros(n_functions, device=device).float()
        self.alphas = torch.ones(n_functions, device=device).float() / n_functions
        
        self.t = 0
    
    def forward(self, x, y):
        loss_raw = self.loss_fn(x, y)
        if not self.training:
            return sum([self.alphas[i] * loss for i, loss in enumerate(loss_raw)])
        
        losses = torch.tensor(loss_raw, device=self.device)
        loss_ratios = losses / (losses, self.ewma_loss)[bool(self.t)]
        
        # Update ewma_loss
        self.ewma_loss = _ewma(prev=self.ewma_loss, current=losses, beta=self.beta)
        
        # Update loss_ratio and squared_diff
        old_ewma_loss_ratios = self.ewma_loss_ratios
        self.ewma_loss_ratios = _ewma(prev=self.ewma_loss_ratios, current=loss_ratios, beta=self.beta)
        self.squared_differences += (loss_ratios - old_ewma_loss_ratios) * (loss_ratios - self.ewma_loss_ratios)
        
        # Update ewma std
        current_std = torch.sqrt((self.squared_differences + EPSILON) / (self.t+1)).to(self.device)
        self.ewma_std = _ewma(prev=self.ewma_std, current=current_std, beta=self.beta)
        
        if self.t:
            covs = self.ewma_std / (self.ewma_loss_ratios + EPSILON)
            self.alphas = covs / torch.sum(covs)
        
        self.t += 1
        
        return sum([self.alphas[i] * loss for i, loss in enumerate(loss_raw)])
        
class CoVWeightingLoss(nn.Module):
    def __init__(self, loss_fn, num_losses, device):
        super(CoVWeightingLoss, self).__init__()
        self.num_losses = num_losses
        self.loss_fn = loss_fn
        self.device = device
        # How to compute the mean statistics: Full mean or decaying mean.
        self.current_iter = -1
        self.alphas = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)

        # Initialize all running statistics at 0.
        self.running_mean_L = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(
            self.device)
        self.running_mean_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(
            self.device)
        self.running_S_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(
            self.device)
        self.running_std_l = None

    def forward(self, pred, target):
        unweighted_losses = self.loss_fn(pred, target)
        L = torch.tensor(unweighted_losses, requires_grad=False).to(self.device)

        if not self.training:
            return torch.sum(L)

        # Increase the current iteration parameter.
        self.current_iter += 1
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        # If we are in the first iteration set alphas to all 1/32
        if self.current_iter <= 1:
            self.alphas = torch.ones((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(
                self.device) / self.num_losses
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        else:
            mean_param = 0.9

        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L

        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))]
        loss = sum(weighted_losses)
        return loss


if __name__ == "__main__":
    # Sanity Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_hat, y_true = torch.rand(2, 32, 1, 224, 320).to(device)
    try:
        print("nuy_loss is: ", nyuv2_loss_fn(y_hat, y_true))
        print("Sil Log Loss is: ", sil_loss(y_hat, y_true))
        print("Grad L1 Loss is: ", grad_l1_loss(y_hat, y_true))
        print("DSSIM Loss is: ", dssim(y_hat, y_true, data_range=NUY_V2_DATA_RANGE))
    except BaseException as e:
        print("Loss fn implementation is wrong: ", e)

    i = 0
    def loss_fn(x, y):
        global i
        i+=1
        return 1.0 * i, 2.0 * (i+1), 3.0 * (i+2)
    loss = ComposeEWMALoss(loss_fn, 3, device)
    loss.forward(y_hat, y_true)
    loss.forward(y_hat, y_true)
    loss.forward(y_hat, y_true)
    loss.forward(y_hat, y_true)