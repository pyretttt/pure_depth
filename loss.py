from typing import Optional, Tuple
from math import exp

import torch
import torch.nn.functional as F

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
    y_hat: torch.Tensor, y_true: torch.Tensor, alpha=0.7, data_range=NUY_V2_DATA_RANGE
):
    """SiLogLoss with scale invariant alpha + L1 Gradient loss + DSSIM"""

    mask = y_true > 0
    l1 = sil_loss(y_hat, y_true, alpha=alpha, mask=mask)
    l2 = grad_l1_loss(y_hat, y_true, mask=mask)
    l3 = dssim(y_hat, y_true, data_range=data_range)

    return 0.1 * l1 + l2 + l3


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
