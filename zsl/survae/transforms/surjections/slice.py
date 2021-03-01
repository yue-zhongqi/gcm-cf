import torch
from survae.utils import sum_except_batch
from survae.distributions import ConditionalDistribution
from survae.transforms.surjections import Surjection


class Slice(Surjection):
    '''
    A simple slice layer which factors out some elements and returns
    the remaining elements for further transformation.
    This is useful for constructing multi-scale architectures [1].

    References:
        [1] Density estimation using Real NVP,
            Dinh et al., 2017, https://arxiv.org/abs/1605.08803
    '''

    stochastic_forward = False

    def __init__(self, decoder, num_keep, dim=1):
        super(Slice, self).__init__()
        assert dim >= 1
        self.decoder = decoder
        self.dim = dim
        self.num_keep = num_keep
        self.cond = isinstance(self.decoder, ConditionalDistribution)

    def split_input(self, input):
        split_proportions = (self.num_keep, input.shape[self.dim] - self.num_keep)
        return torch.split(input, split_proportions, dim=self.dim)

    def forward(self, x):
        z, x2 = self.split_input(x)
        if self.cond: ldj = self.decoder.log_prob(x2, context=z)
        else:         ldj = self.decoder.log_prob(x2)
        return z, ldj

    def inverse(self, z):
        if self.cond: x2 = self.decoder.sample(context=z)
        else:         x2 = self.decoder.sample(num_samples=z.shape[0])
        x = torch.cat([z, x2], dim=self.dim)
        return x


class ConditionalSlice(Surjection):
    '''
    A simple slice layer which factors out some elements and returns
    the remaining elements for further transformation.
    This is useful for constructing multi-scale architectures [1].

    References:
        [1] Density estimation using Real NVP,
            Dinh et al., 2017, https://arxiv.org/abs/1605.08803
    '''

    stochastic_forward = False

    def __init__(self, decoder, num_keep, dim=1):
        super(ConditionalSlice, self).__init__()
        assert dim >= 1
        self.decoder = decoder
        self.dim = dim
        self.num_keep = num_keep
        assert isinstance(self.decoder, ConditionalDistribution)

    def split_input(self, input):
        split_proportions = (self.num_keep, input.shape[self.dim] - self.num_keep)
        return torch.split(input, split_proportions, dim=self.dim)

    def forward(self, x, extra, feedback):
        z, x2 = self.split_input(x)
        ldj = self.decoder.log_prob(x2, context=torch.cat((z, extra), dim=-1), feedback=feedback)
        return z, ldj

    def inverse(self, z, extra, deterministic=False, feedback=None):
        if deterministic:
            x2 = self.decoder.mean(context=torch.cat((z, extra), dim=-1), feedback=feedback)
        else:
            x2 = self.decoder.sample(context=torch.cat((z, extra), dim=-1), feedback=feedback)
        x = torch.cat([z, x2], dim=self.dim)
        return x


class ConditionalSliceV2(Surjection):
    '''
    A simple slice layer which factors out some elements and returns
    the remaining elements for further transformation.
    This is useful for constructing multi-scale architectures [1].

    References:
        [1] Density estimation using Real NVP,
            Dinh et al., 2017, https://arxiv.org/abs/1605.08803
    '''

    stochastic_forward = False

    def __init__(self, decoder, num_keep, z_dim, dim=1):
        super(ConditionalSliceV2, self).__init__()
        assert dim >= 1
        self.decoder = decoder
        self.dim = dim
        self.num_keep = num_keep
        self.z_dim = z_dim
        assert isinstance(self.decoder, ConditionalDistribution)

    def split_input(self, input):
        split_proportions = (self.num_keep, input.shape[self.dim] - self.num_keep)
        return torch.split(input, split_proportions, dim=self.dim)

    def forward(self, x, extra):
        z, x2 = self.split_input(x)
        ldj = self.decoder.log_prob(x2, context=torch.cat((z[:, :self.z_dim], extra), dim=-1))
        return z, ldj

    def inverse(self, z, extra, deterministic=False):
        if deterministic:
            x2 = self.decoder.mean(context=torch.cat((z[:, :self.z_dim], extra), dim=-1))
        else:
            x2 = self.decoder.sample(context=torch.cat((z[:, :self.z_dim], extra), dim=-1))
        x = torch.cat([z, x2], dim=self.dim)
        return x