import math
import torch
import torch.nn as nn
from survae.distributions import Distribution, ConditionalBernoulli
from survae.transforms.surjections import Surjection


class ZSLMaxPool(Surjection):
    '''
    forward: For every pair of input, output its max.
    inverse: The parameters of Bernoulli are generated from class attributes;
             The parameters of X distribution are generated from class attributes + features after max
    Inference surjection.
    '''
    stochastic_forward = False

    def __init__(self, x_dim, index_decoder, value_decoder):
        super(ZSLMaxPool, self).__init__()
        self.index_decoder = index_decoder
        self.value_decoder = value_decoder
        self.x_dim = x_dim
        assert self.x_dim % 2 == 0
        assert isinstance(index_decoder, ConditionalBernoulli)

    def forward(self, x, extra, feedback):
        '''
        idx = torch.zeros((x.shape[0], self.x_dim // 2), device=x.device).float()
        z = torch.zeros((x.shape[0], self.x_dim // 2), device=x.device).float()
        diff = torch.zeros((x.shape[0], self.x_dim // 2), device=x.device).float()
        discarded = torch.zeros((x.shape[0], self.x_dim // 2), device=x.device).float()
        for i in range(self.x_dim // 2):
            idx[:, i] = (x[:, 2 * i] <= x[:, 2 * i + 1])
            max_v, _ = x[:, 2 * i:2 * i + 2].max(dim=1)
            min_v, _ = x[:, 2 * i:2 * i + 2].min(dim=1)
            z[:, i] = max_v
            discarded[:, i] = min_v
            diff[:, i] = z[:, i] - discarded[:, i]
        assert (diff < 0).sum() == 0
        '''
        odd = x[:, 1::2]    # Odd indices elements
        even = x[:, 0::2]   # Even indices elements
        idx = (odd > even).int()
        z = torch.max(odd, even)
        discarded = torch.min(odd, even)
        diff = z - discarded
        assert (diff < 0).sum() == 0

        idx_ldj = self.index_decoder.log_prob(idx, context=torch.cat((z, extra), dim=-1), feedback=feedback)
        diff_ldj = self.value_decoder.log_prob(diff, context=torch.cat((z, extra), dim=-1), feedback=feedback)
        # Calculate entropy for debugging
        '''
        p = self.index_decoder.logits(context=torch.cat((z, extra), dim=-1))
        p = nn.Sigmoid()(p)
        q = 1 - p
        self.index_entropy = -q * q.log() - p * p.log()
        self.index_entropy = self.index_entropy.sum(dim=1)
        '''
        return z, idx_ldj + diff_ldj

    def splice(self, even, odd):
        x = torch.zeros((even.shape[0], self.x_dim), device=even.device).float()
        x[:, 0::2] = even
        x[:, 1::2] = odd
        return x

    def inverse(self, z, extra, deterministic=False, feedback=None):
        if deterministic:
            p = self.index_decoder.logits(context=torch.cat((z, extra), dim=-1), feedback=feedback)
            p = nn.Sigmoid()(p)
            idx = torch.round(p)
        else:
            idx = self.index_decoder.sample(context=torch.cat((z, extra), dim=-1), feedback=feedback)
        
        if deterministic:
            diff = self.value_decoder.mean(context=torch.cat((z, extra), dim=-1), feedback=feedback)
        else:
            diff = self.value_decoder.sample(context=torch.cat((z, extra), dim=-1), feedback=feedback)

        if (diff < 0).sum() != 0:
            print(diff[diff < 0])
            assert False
        even = (1 - idx) * z + idx * (z - diff)
        odd = idx * z + (1 - idx) * (z - diff)
        x = self.splice(even, odd)
        return x

class ZSLMaxPoolV2(Surjection):
    '''
    forward: For every pair of input, output its max.
    inverse: The parameters of Bernoulli are generated from class attributes;
             The parameters of X distribution are generated from class attributes + features after max
    Inference surjection.
    '''
    stochastic_forward = False

    def __init__(self, x_dim, index_decoder, odd_decoder, even_decoder):
        super(ZSLMaxPoolV2, self).__init__()
        self.index_decoder = index_decoder
        self.odd_decoder = odd_decoder      # Used when odd bin is larger
        self.even_decoder = even_decoder    # Used when even bin is larger
        self.x_dim = x_dim
        assert self.x_dim % 2 == 0
        assert isinstance(index_decoder, ConditionalBernoulli)

    def forward(self, x, extra):
        odd = x[:, 1::2]    # Odd indices elements
        even = x[:, 0::2]   # Even indices elements
        idx = (odd > even).int()
        z = torch.max(odd, even)
        discarded = torch.min(odd, even)
        diff = z - discarded
        assert (diff < 0).sum() == 0
        idx_ldj = self.index_decoder.log_prob(idx, context=torch.cat((z, extra), dim=-1))
        odd_diff_ldj = self.odd_decoder.log_prob(diff, context=torch.cat((z, extra), dim=-1), should_sum=False)
        even_diff_ldj = self.even_decoder.log_prob(diff, context=torch.cat((z, extra), dim=-1), should_sum=False)
        diff_ldj = (odd_diff_ldj * idx + even_diff_ldj * (1 - idx)).sum(dim=-1)
        # Calculate entropy for debugging
        # p = self.index_decoder.logits(context=torch.cat((z, extra), dim=-1))
        # p = nn.Sigmoid()(p)
        # q = 1 - p
        # self.index_entropy = -q * q.log() - p * p.log()
        # self.index_entropy = self.index_entropy.sum(dim=1)
        return z, idx_ldj + diff_ldj

    def splice(self, even, odd):
        x = torch.zeros((even.shape[0], self.x_dim), device=even.device).float()
        x[:, 0::2] = even
        x[:, 1::2] = odd
        return x

    def inverse(self, z, extra, deterministic=False):
        if deterministic and (not self.training):
            p = self.index_decoder.logits(context=torch.cat((z, extra), dim=-1))
            p = nn.Sigmoid()(p)
            idx = torch.round(p)
        else:
            idx = self.index_decoder.sample(context=torch.cat((z, extra), dim=-1))

        if deterministic:
            odd_diff = self.odd_decoder.mean(context=torch.cat((z, extra), dim=-1))
            even_diff = self.even_decoder.mean(context=torch.cat((z, extra), dim=-1))
        else:
            odd_diff = self.odd_decoder.sample(context=torch.cat((z, extra), dim=-1))
            even_diff = self.even_decoder.sample(context=torch.cat((z, extra), dim=-1))
        diff = odd_diff * idx + even_diff * (1 - idx)
        if (diff < 0).sum() != 0:
            print(diff[diff < 0])
            assert False
        # assert (diff < 0).sum() == 0
        even = (1 - idx) * z + idx * (z - diff)
        odd = idx * z + (1 - idx) * (z - diff)
        x = self.splice(even, odd)
        return x