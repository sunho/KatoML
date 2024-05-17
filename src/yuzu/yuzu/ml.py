"""
yuzu.ml is a core machine learning library based on yuzu node abstraction
"""

from abc import ABC, abstractmethod
import yuzu as yz

def mse_loss(x: yz.Node, y: yz.Node, reduction = 'mean') -> yz.Node:
    if x.shape() != y.shape():
        raise Exception("mse_loss: shapes mismatch")
    delta = (x-y)
    d2 = (delta*delta)
    if reduction == 'mean':
        return d2.mean(dim=-1)
    elif reduction == 'sum':
        return d2.sum(dim=-1)
    elif reduction == 'none':
        return d2
    else:
        raise Exception("mse_loss: invalid reduction mode")


class Distribution(ABC):
    @abstractmethod
    def log_prob(self, value: yz.Node):
        pass

    @abstractmethod
    def sample(self) -> yz.Node:
        pass

    @abstractmethod
    def entropy(self) -> yz.Node:
        pass

# gumbel inverse transform
# e^e^-(x-mu) = F
# solve for x we get:
# x = mu - beta ln(-ln(F))
def gumbel_inverse_transform(F, mu=0.0, b=1.0):
    return mu - b * yz.log(-yz.log(F))

class Categorical(Distribution):
    def __init__(self, logits: yz.Node):
        self.logits = logits
        self.n = logits.shape()[-1]

    def log_prob(self, value: yz.Node):
        return self.logits[value]

    def sample(self, shape) -> yz.Node:
        # gumbel distribution trick to draw from logits
        g = gumbel_inverse_transform(yz.rand(shape + (self.n, )))
        return (g + self.logits).argmax(dim=-1)

    def entropy(self) -> yz.Node:
        return -(self.logits.exp() * self.logits).sum()
