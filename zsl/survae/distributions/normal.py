import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.distributions import Distribution
from survae.utils import sum_except_batch


class StandardNormal(Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape, sigma=1.0):
        super(StandardNormal, self).__init__()
        self.shape = torch.Size(shape)
        self.register_buffer('buffer', torch.zeros(1))
        self.sigma = sigma

    def log_prob(self, x):
        log_base = -0.5 * math.log(2 * math.pi) - math.log(self.sigma)
        log_inner = - 0.5 * (x / self.sigma)**2
        return sum_except_batch(log_base+log_inner)

    def log_prob_gradient(self, x):
        log_inner = - 0.5 * x**2
        return sum_except_batch(log_inner)

    def log_prob_with_mask(self, x, mask):
        log_base =  - 0.5 * math.log(2 * math.pi)
        log_inner = - 0.5 * (x * mask)**2
        return sum_except_batch(log_base+log_inner)

    def sample(self, num_samples):
        return torch.randn(num_samples, *self.shape, device=self.buffer.device, dtype=self.buffer.dtype) * self.sigma


class DiagonalNormal(Distribution):
    """A multivariate Normal with diagonal covariance."""

    def __init__(self, shape):
        super(DiagonalNormal, self).__init__()
        self.shape = torch.Size(shape)
        self.loc = nn.Parameter(torch.zeros(shape))
        self.log_scale = nn.Parameter(torch.zeros(shape))

    def log_prob(self, x):
        log_base =  - 0.5 * math.log(2 * math.pi) - self.log_scale
        log_inner = - 0.5 * torch.exp(-2 * self.log_scale) * ((x - self.loc) ** 2)
        return sum_except_batch(log_base+log_inner)

    def log_prob_with_mask(self, x, mask):
        log_base = - 0.5 * math.log(2 * math.pi) - self.log_scale.unsqueeze(0).expand(x.shape[0], -1) * mask
        log_inner = - 0.5 * torch.exp(-2 * self.log_scale.unsqueeze(0).expand(x.shape[0], -1) * mask) \
            * ((x * mask - self.loc.unsqueeze(0).expand(x.shape[0], -1) * mask) ** 2)
        return sum_except_batch(log_base + log_inner)

    def sample(self, num_samples):
        eps = torch.randn(num_samples, *self.shape, device=self.loc.device, dtype=self.loc.dtype)
        return self.loc + self.log_scale.exp() * eps


class ConvNormal2d(DiagonalNormal):
    def __init__(self, shape):
        super(DiagonalNormal, self).__init__()
        assert len(shape) == 3
        self.shape = torch.Size(shape)
        self.loc = torch.nn.Parameter(torch.zeros(1, shape[0], 1, 1))
        self.log_scale = torch.nn.Parameter(torch.zeros(1, shape[0], 1, 1))
