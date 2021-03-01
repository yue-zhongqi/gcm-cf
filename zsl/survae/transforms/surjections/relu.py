import torch
from survae.transforms.surjections import Surjection
from survae.distributions import ConditionalDistribution


class GenReLU(Surjection):
    '''
    Generative ReLU: x=ReLU(z)
    '''
    stochastic_forward = True

    def __init__(self, decoder):
        super(GenReLU, self).__init__()
        self.decoder = decoder
        assert (not isinstance(self.decoder, ConditionalDistribution))
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        z = self.decoder.sample(num_samples=x.shape[0])
        mask = (x == 0)
        logqz = self.decoder.log_prob_with_mask(z, mask)
        z = mask * z + (~mask) * x
        ldj = -logqz
        return z, ldj

    def inverse(self, z):
        return self.relu(z)

class InfReLU(Surjection):
    stochastic_forward = False

    def __init__(self, decoder):
        super(InfReLU, self).__init__()
        self.decoder = decoder
        # assert (not isinstance(self.decoder, ConditionalDistribution))
        self.relu = torch.nn.ReLU()

    def forward(self, x, extra):
        z = self.relu(x)
        mask = (z == 0)
        x_lt_zero = x * mask
        logpx = self.decoder.log_prob(-x_lt_zero, context=extra, should_sum=False)
        logpx = (mask * logpx).sum(dim=-1)
        ldj = logpx
        return z, ldj

    def inverse(self, z, extra, deterministic=False):
        if deterministic:
            x = self.decoder.mean(context=extra)
        else:
            x = self.decoder.sample(context=extra)
        mask = (z <= 0)
        x = mask * (-x) + (~mask) * z
        return x