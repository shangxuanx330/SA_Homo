import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .common import Conv2dBNAct, Bottleneck


class DepthwiseCrossCorrelation_org(nn.Module):
    def __init__(self, use_fft: bool) -> None:
        """Init a depth-wise cross-correlation layer.

        Args:
            use_fft (bool): Whether to use FFT to accelerate cross-correlation.
        """
        super(DepthwiseCrossCorrelation, self).__init__()
        self.use_fft = use_fft

    # @torch.jit.script_if_tracing
    def forward(self, x: Tensor, template: Tensor) -> Tensor:
        # Extract dimensions.
        bs, nc, hi, wi = x.shape
        ht, wt = template.shape[2:]
        # Calculate output size.
        ho = hi - ht + 1
        wo = wi - wt + 1

        # Use FFT to accelerate cross-correlation during training.
        if self.use_fft:
            # Calculate padded spectrum size.
            hs = hi + ht - 1
            ws = wi + wt - 1   
            x_spectrum = torch.fft.rfft2(x, s=(hs, ws))
            template_spectrum = torch.fft.rfft2(template, s=(hs, ws))
            xcorr_spectrum = x_spectrum * torch.conj(template_spectrum)
            xcorr: Tensor = torch.fft.irfft2(xcorr_spectrum)
            xcorr = xcorr[:, :, :ho, :wo]
        else:
            # Reshape templeate tensor to correlation kernel (flatten NC dimensions).
            groups = nc * bs
            input = x.reshape(1, groups, hi, wi)
            kernel = template.reshape(groups, 1, ht, wt)
            xcorr = F.conv2d(input, kernel, groups=groups)
            if bs > 1:
                xcorr = xcorr.reshape((bs, nc, ho, wo))
        return xcorr

class DepthwiseCrossCorrelation(nn.Module):
    def __init__(self, use_fft: bool) -> None:
        """Init a depth-wise cross-correlation layer.

        Args:
            use_fft (bool): Whether to use FFT to accelerate cross-correlation.
        """
        super(DepthwiseCrossCorrelation, self).__init__()
        self.use_fft = use_fft

    # @torch.jit.script_if_tracing
    def forward(self, x: Tensor, template: Tensor) -> Tensor:
        # Extract dimensions.
        bs, nc, hi, wi = x.shape
        ht, wt = template.shape[2:]

        # Use FFT to accelerate cross-correlation during training.
        if self.use_fft:
            # Calculate padded spectrum size.
            hs = hi + ht - 1
            ws = wi + wt - 1   
            x_spectrum = torch.fft.rfft2(x, s=(hs, ws))
            template_spectrum = torch.fft.rfft2(template, s=(hs, ws))
            xcorr_spectrum = x_spectrum * torch.conj(template_spectrum)
            xcorr: Tensor = torch.fft.irfft2(xcorr_spectrum)
            xcorr = xcorr[:, :, :hi, :wi]
        else:
            # Reshape templeate tensor to correlation kernel (flatten NC dimensions).
            groups = nc * bs
            input = x.reshape(1, groups, hi, wi)
            kernel = template.reshape(groups, 1, ht, wt)
            xcorr = F.conv2d(input, kernel, groups=groups,padding=ht//2)
            if bs > 1:
                xcorr = xcorr.reshape((bs, nc, hi, wi))
        return xcorr


class BatchDepthwiseCrossCorrelation(nn.Module):
    def __init__(self):
        """批量深度可分离互相关层"""
        super(BatchDepthwiseCrossCorrelation, self).__init__()
    
    def forward(self, x: Tensor, templates: Tensor) -> Tensor:
        """
        
        Args:
            x:  [bs, nc, hi, wi]
            templates:  [num_templates, bs, nc, ht, wt]
        
        Returns:
            xcorr:  [num_templates, bs, nc, hi, wi]
        """
        num_templates, bs, nc, ht, wt = templates.shape
        _, _, hi, wi = x.shape
        
        # [bs, nc, hi, wi] -> [bs*num_templates, nc, hi, wi]
        x_repeated = x.unsqueeze(1).repeat(1, num_templates, 1, 1, 1)  # [bs, num_templates, nc, hi, wi]
        x_repeated = x_repeated.reshape(bs * num_templates, nc, hi, wi)
        
        # [num_templates, bs, nc, ht, wt] -> [bs*num_templates*nc, 1, ht, wt]
        templates_permuted = templates.permute(1, 0, 2, 3, 4)  # [bs, num_templates, nc, ht, wt]
        kernels = templates_permuted.reshape(bs * num_templates * nc, 1, ht, wt)
        
        # [bs*num_templates, nc, hi, wi] -> [1, bs*num_templates*nc, hi, wi]
        x_input = x_repeated.reshape(1, bs * num_templates * nc, hi, wi)
        xcorr = F.conv2d(x_input, kernels, groups=bs * num_templates * nc, padding=ht//2)
        
        # [1, bs*num_templates*nc, hi, wi] -> [bs, num_templates, nc, hi, wi]
        xcorr = xcorr.reshape(bs, num_templates, nc, hi, wi)
        
        # [bs, num_templates, nc, hi, wi] -> [num_templates, bs, nc, hi, wi]
        xcorr = xcorr.permute(1, 0, 2, 3, 4)
        
        return xcorr
    

    
    

class CorrelationNet(nn.Module):
    def __init__(self, num_features: int, use_fft: bool = True) -> None:
        """Init a correlation network branch.

        Args:
            num_features (int): Number of feature map channels.
            use_fft (bool): Whether to use FFT to accelerate cross-correlation.
        """
        super(CorrelationNet, self).__init__()

        self.depthwise_xcorr = DepthwiseCrossCorrelation(use_fft)

    def forward(self, x_optical: Tensor, x_sar: Tensor) -> Tensor:
        xcorr = self.depthwise_xcorr(x_optical, x_sar)
        return xcorr
