import torch
from torch import nn
from collections.abc import Iterable
from survae.distributions import Distribution
from survae.transforms import Transform
from torch.distributions import Normal
from survae.transforms import ConditionalSlice, ZSLMaxPool, ZSLMaxPoolV2, ConditionalSliceV2
from survae.transforms import InfReLU


def need_extra(transform):
    return isinstance(transform, ConditionalSlice) or\
           isinstance(transform, ZSLMaxPool) or\
           isinstance(transform, ZSLMaxPoolV2) or\
           isinstance(transform, ConditionalSliceV2) or\
           isinstance(transform, InfReLU)

class Flow(Distribution):
    """
    Base class for Flow.
    Flows use the forward transforms to transform data to noise.
    The inverse transforms can subsequently be used for sampling.
    These are typically useful as generative models of data.
    """

    def __init__(self, base_dist, transforms):
        super(Flow, self).__init__()
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.lower_bound = any(transform.lower_bound for transform in transforms)

    def log_prob(self, x, extra=None, feedback=None):
        log_prob = torch.zeros(x.shape[0], device=x.device)
        for transform in self.transforms:
            if isinstance(transform, ConditionalSlice) or isinstance(transform, ZSLMaxPool):
                x, ldj = transform(x, extra, feedback=feedback)
            else:
                x, ldj = transform(x)
            log_prob += ldj
        log_prob += self.base_dist.log_prob(x)
        self.latent = x
        return log_prob

    def sample(self, num_samples, extra=None, feedback=None):
        if extra is not None:
            num_samples = extra.shape[0]
        z = self.base_dist.sample(num_samples)
        for transform in reversed(self.transforms):
            if isinstance(transform, ConditionalSlice) or isinstance(transform, ZSLMaxPool):
                z = transform.inverse(z, extra, feedback=feedback)
            else:
                z = transform.inverse(z)
        return z

    def conditional_sample(self, z, extra=None, feedback=None):
        for transform in reversed(self.transforms):
            if isinstance(transform, ConditionalSlice) or isinstance(transform, ZSLMaxPool):
                z = transform.inverse(z, extra, feedback=feedback)
            else:
                z = transform.inverse(z)
        return z
    
    def get_latent(self, x, extra=None, feedback=None):
        _ = self.log_prob(x, extra=extra, feedback=feedback)
        return self.latent

    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")


class FlowComponent(Distribution):
    '''
    Flow that ends without base distribution.
    The output can be fed into other flows
    '''
    
    def __init__(self, transforms):
        super(FlowComponent, self).__init__()
        if isinstance(transforms, Transform):
            transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.transforms = nn.ModuleList(transforms)
        self.lower_bound = any(transform.lower_bound for transform in transforms)

    def log_prob(self, x, extra=None, feedback=None):
        log_prob = torch.zeros(x.shape[0], device=x.device)
        for transform in self.transforms:
            if need_extra(transform):
                x, ldj = transform(x, extra, feedback=feedback)
            else:
                x, ldj = transform(x)
            log_prob += ldj
        return log_prob, x

    def sample(self, z, extra=None, deterministic=False, feedback=None):
        for transform in reversed(self.transforms):
            if need_extra(transform):
                z = transform.inverse(z, extra, deterministic=deterministic, feedback=feedback)
            else:
                z = transform.inverse(z)
        return z

    def get_latent(self, x):
        for transform in self.transforms:
            x, ldj = transform(x)
        return x

    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("FlowComponent does not support sample_with_log_prob, see InverseFlow instead.")


class DataPriorFlow(Distribution):
    """
    Flow with a data dependent prior
    The prior has a normal distribution with mean equals to the given data_prior
    The contribution of prior to log prob is controlled by parameter sigma.
    Larger sigma means less confidence in the prior data
    """

    def __init__(self, transforms, sigma=0.1):
        super(DataPriorFlow, self).__init__()
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.transforms = nn.ModuleList(transforms)
        self.lower_bound = any(transform.lower_bound for transform in transforms)
        self.sigma = sigma

    def log_prob(self, x, data_prior, transform_f=None):
        log_prob = torch.zeros(x.shape[0], device=x.device)
        for transform in self.transforms:
            x, ldj = transform(x)
            log_prob += ldj
        base_dist = Normal(loc=data_prior, scale=self.sigma * torch.ones(data_prior.shape, device=x.device))
        # log_prob += (-self.beta * ((x - data_prior)**2).sum(dim=1))
        if transform_f is not None:
            x = transform_f(x)    # such as ReLU
        log_prob += (base_dist.log_prob(x)).sum(dim=1)
        self.latent = x
        return log_prob

    def sample(self, data_prior):
        z = data_prior
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z

    def get_latent(self, x):
        for transform in self.transforms:
            x, ldj = transform(x)
        return x

    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")