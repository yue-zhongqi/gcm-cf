import torch
from torch.distributions import Bernoulli
from survae.distributions.conditional import ConditionalDistribution
from survae.utils import sum_except_batch


class ConditionalBernoulli(ConditionalDistribution):
    """A Bernoulli distribution with conditional logits."""

    def __init__(self, net):
        super(ConditionalBernoulli, self).__init__()
        self.net = net

    def cond_dist(self, context, feedback=None):
        logits = self.net(context, feedback=feedback)
        return Bernoulli(logits=logits)

    def log_prob(self, x, context, feedback=None):
        dist = self.cond_dist(context, feedback=feedback)
        return sum_except_batch(dist.log_prob(x.float()))

    def sample(self, context, feedback=None):
        dist = self.cond_dist(context, feedback=feedback)
        return dist.sample().long()

    def sample_with_log_prob(self, context, feedback=None):
        dist = self.cond_dist(context, feedback=feedback)
        z = dist.sample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z.long(), log_prob

    def logits(self, context, feedback=None):
        return self.cond_dist(context, feedback=feedback).logits

    def probs(self, context, feedback=None):
        return self.cond_dist(context, feedback=feedback).probs

    def mean(self, context, feedback=None):
        return self.cond_dist(context, feedback=feedback).mean

    def mode(self, context):
        return (self.cond_dist(context).logits>=0).long()
