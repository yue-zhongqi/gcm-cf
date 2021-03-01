import math
import torch
import torch.nn as nn
from torch.distributions import Normal, HalfNormal
from survae.distributions import StandardHalfNormal
from survae.distributions.conditional import ConditionalDistribution
from survae.utils import sum_except_batch


class ConditionalMeanNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and fixed std."""

    def __init__(self, net, scale=1.0):
        super(ConditionalMeanNormal, self).__init__()
        self.net = net
        self.scale = scale

    def cond_dist(self, context):
        mean = self.net(context)
        return Normal(loc=mean, scale=self.scale)

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean


class ConditionalMeanStdNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and learned std."""

    def __init__(self, net, scale_shape):
        super(ConditionalMeanStdNormal, self).__init__()
        self.net = net
        self.log_scale = nn.Parameter(torch.zeros(scale_shape))

    def cond_dist(self, context):
        mean = self.net(context)
        return Normal(loc=mean, scale=self.log_scale.exp())

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean


class ConditionalNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and log_std."""

    def __init__(self, net, split_dim=-1, relu_mu=False):
        super(ConditionalNormal, self).__init__()
        self.net = net
        self.split_dim = split_dim
        self.relu_mu = relu_mu

    def cond_dist(self, context, feedback=None):
        if feedback is None:
            params = self.net(context)
        else:
            params = self.net(context, feedback=feedback)
        mean, log_std = torch.chunk(params, chunks=2, dim=self.split_dim)
        if self.relu_mu:
            mean = nn.ReLU()(mean)
        return Normal(loc=mean, scale=log_std.exp())

    def log_prob(self, x, context, feedback=None):
        dist = self.cond_dist(context, feedback=feedback)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context, feedback=None):
        dist = self.cond_dist(context, feedback=feedback)
        return dist.rsample()

    def sample_with_log_prob(self, context, feedback=None):
        dist = self.cond_dist(context, feedback=feedback)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def mean(self, context, feedback=None):
        return self.cond_dist(context, feedback=feedback).mean

    def mean_stddev(self, context):
        dist = self.cond_dist(context)
        return dist.mean, dist.stddev

class ConditionalHalfNormal(ConditionalDistribution):
    """A multivariate half Normal (with support > 0) with conditional mean and log_std."""

    def __init__(self, net, split_dim=-1):
        super(ConditionalHalfNormal, self).__init__()
        self.net = net
        self.split_dim = split_dim

    def cond_dist_params(self, context, feedback=None):
        params = self.net(context, feedback=feedback)
        mean, log_std = torch.chunk(params, chunks=2, dim=self.split_dim)
        mean = mean.abs()
        return mean, log_std.exp() + 1e-6

    def log_prob(self, x, context, should_sum=True, feedback=None):
        mean, std = self.cond_dist_params(context, feedback=feedback)
        dist = Normal(torch.zeros(mean.shape, device=mean.device), torch.ones(std.shape, device=std.device))
        adjusted_x = (x - mean) / std
        adjusted_a = (0 - mean) / std
        log_gx = dist.log_prob(adjusted_x)
        log_c = ((1 - dist.cdf(adjusted_a)) * std).log()
        log_prob = log_gx - log_c
        # return sum_except_batch(dist.log_prob((x - mean).abs()))
        '''
        # Folded normal distribution
        mean, std = self.cond_dist_params(context)
        dist1 = Normal(mean, std)
        dist2 = Normal(-mean, std)
        log_prob = (dist1.log_prob(x).exp() + dist2.log_prob(x).exp()).log()
        '''
        if should_sum:
            return sum_except_batch(log_prob)
        else:
            return log_prob

    def sample(self, context, feedback=None):
        mean, std = self.cond_dist_params(context, feedback=feedback)
        std_normal = Normal(torch.zeros(mean.shape, device=mean.device), torch.ones(std.shape, device=std.device))
        mu = self.mean(context)
        alpha = (0 - mean) / std
        z = 1 - std_normal.cdf(alpha)
        phi_alpha = std_normal.log_prob(alpha).exp()
        sigma = std * torch.sqrt(1 + alpha * phi_alpha / z - (phi_alpha / z) ** 2)
        dist = Normal(mu, sigma)
        return dist.rsample().abs()

    def sample_with_log_prob(self, context, feedback=None):
        mean, std = self.cond_dist_params(context, feedback=feedback)
        dist = HalfNormal(std)
        z = dist.rsample() + mean
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def mean(self, context, feedback=None):
        # get mean of truncated normal
        mean, std = self.cond_dist_params(context, feedback=feedback)
        std_normal = Normal(torch.zeros(mean.shape, device=mean.device), torch.ones(std.shape, device=std.device))
        adjusted_a = (0 - mean) / std
        additional = std * std_normal.log_prob(adjusted_a).exp() / (1 - std_normal.cdf(adjusted_a))
        return mean + additional

    def mean_stddev(self, context):
        assert False  # Not implement correctly
        mean, std = self.cond_dist_params(context)
        return mean, std