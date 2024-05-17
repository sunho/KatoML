"""
yuzu backend implementation based on pytorch
"""

from .base import BaseNode, BaseDevice, BaseBackend, ElementType
from . import base
from typing import Optional, Union, List, Tuple, Any

import torch

class PyTorchNode(BaseNode):
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.requires_grad = False

    def shape(self):
        return self.data.shape

    def dim(self) -> int:
        return self.data.ndim

    def unwrap(self, x):
        return x.data if isinstance(x, PyTorchNode) else x

    def is_close(self, other: 'PyTorchNode', rtol: float = 1e-5, atol: float = 1e-8):
        return torch.isclose(self.data, other.data, rtol, atol)

    def __add__(self, other: Union['PyTorchNode', float]):
        return PyTorchNode(self.data + self.unwrap(other))

    def __sub__(self, other: Union['PyTorchNode', float]):
        return PyTorchNode(self.data - self.unwrap(other))

    def __mul__(self, other: Union['PyTorchNode', float]):
        return PyTorchNode(self.data * self.unwrap(other))

    def __truediv__(self, other: Union['PyTorchNode', float]):
        return PyTorchNode(self.data / self.unwrap(other))

    def __radd__(self, other: Union['PyTorchNode', float]):
        return PyTorchNode(self.unwrap(other) + self.data)

    def __rsub__(self, other: Union['PyTorchNode', float]):
        return PyTorchNode(self.unwrap(other) - self.data)

    def __rmul__(self, other: Union['PyTorchNode', float]):
        return PyTorchNode(self.unwrap(other) * self.data)

    def __rtruediv__(self, other: Union['PyTorchNode', float]):
        return PyTorchNode(self.unwrap(other) / self.data)

    def __iadd__(self, other: Union['PyTorchNode', Any]):
        self.data += other.data if isinstance(other, PyTorchNode) else other
        return self

    def __isub__(self, other: Union['PyTorchNode', Any]):
        self.data -= other.data if isinstance(other, PyTorchNode) else other
        return self

    def __imul__(self, other: Union['PyTorchNode', Any]):
        self.data *= other.data if isinstance(other, PyTorchNode) else other
        return self

    def __itruediv__(self, other: Union['PyTorchNode', Any]):
        self.data /= other.data if isinstance(other, PyTorchNode) else other
        return self

    def __neg__(self):
        return PyTorchNode(-self.data)

    def __eq__(self, other: Union['PyTorchNode', float]):
        return self.data == other.data if isinstance(other, PyTorchNode) else self.data == other

    def __getitem__(self, index):
        return PyTorchNode(self.data[index])

    def __setitem__(self, index, value):
        if isinstance(value, PyTorchNode):
            self.data[index] = value.data
        else:
            self.data[index] = value

    def __repr__(self):
        return f"PyTorchNode({self.data})"

    def detach(self):
        return PyTorchNode(self.data.detach())

    def argmax(self, dim: Optional[int] = None, keepdim: bool = False):
        return PyTorchNode(torch.argmax(self.data, dim=dim, keepdim=keepdim))

    def reshape(self, *shape):
        return PyTorchNode(self.data.reshape(*shape))

    def float(self):
        return PyTorchNode(self.data.float())

    def long(self):
        return PyTorchNode(self.data.long())

    def double(self):
        return PyTorchNode(self.data.double())

    def exp(self):
        return PyTorchNode(torch.exp(self.data))

    def log(self):
        return PyTorchNode(torch.log(self.data))

    def mean(self, dim: Optional[int] = None, keepdim: bool = False):
        return PyTorchNode(torch.mean(self.data, dim=dim, keepdim=keepdim))

    def sum(self, dim: Optional[int] = None, keepdim: bool = False):
        return PyTorchNode(torch.sum(self.data, dim=dim, keepdim=keepdim))

    def gather(self, dim: int, index: 'PyTorchNode') -> 'PyTorchNode':
        return PyTorchNode(torch.gather(self.data, dim, index.data))

    def clamp(self, min: Optional['float'] = None, max: Optional['float'] = None) -> 'PyTorchNode':
        return PyTorchNode(self.data.clamp(min=min, max=max))

    def max(self, other: 'PyTorchNode') -> 'PyTorchNode':
        return PyTorchNode(torch.max(self.data, other.data))

    def min(self, other: 'PyTorchNode') -> 'PyTorchNode':
        return PyTorchNode(torch.min(self.data, other.data))

    def reduce_max(self, dim: Optional[int] = None, keepdim: bool = False) -> 'PyTorchNode':
        return PyTorchNode(torch.max(self.data, dim=dim, keepdim=keepdim)[0])

    def reduce_min(self, dim: Optional[int] = None, keepdim: bool = False) -> 'PyTorchNode':
        return PyTorchNode(torch.min(self.data, dim=dim, keepdim=keepdim)[0])

    def item(self):
        return self.data.item()

    def all(self):
        return self.data.all()

    def any(self):
        return self.data.any()

    def eval(self):
        return PyTorchNode(self.data.eval())

    def backward(self):
        self.data.backward()

    def inner(self):
        return self.data

    def to(self, device: 'PyTorchDevice'):
        if device is None:
            return self
        else:
            return PyTorchNode(self.data.to(device.device))


class PyTorchDevice(BaseDevice):
    def __init__(self, device):
        super().__init__()
        self.device = device

class PyTorchBackend(BaseBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def register():
        base.Node = PyTorchNode
        base.Backend = PyTorchBackend
        base.Device = PyTorchDevice

    def wrap(self, data):
        return PyTorchNode(data)

    def zeros(self, *size, requires_grad=False, device=None, dtype=None):
        return PyTorchNode(torch.zeros(*size, requires_grad=requires_grad, dtype=PyTorchBackend.translate_type(dtype), device=device.device if device else None))

    def rand(self, *size, requires_grad=False, device=None, dtype=None):
        return PyTorchNode(torch.rand(*size, requires_grad=requires_grad, dtype=PyTorchBackend.translate_type(dtype), device=device.device if device else None))

    def from_value(self, value, device=None):
        tensor = torch.tensor(value, device=device)
        return PyTorchNode(tensor)

    def from_numpy(self, arr, device=None):
        return PyTorchNode(torch.from_numpy(arr).to(device.device if device else None))

    def set_seed(self, seed):
        torch.manual_seed(seed)

    @staticmethod
    def translate_type(typ: Optional[ElementType]):
        if typ is None:
            return None
        if typ == ElementType.Float32:
            return torch.float32
        elif typ == ElementType.Float64:
            return torch.float64
        elif typ == ElementType.Int32:
            return torch.int32
        elif typ == ElementType.Int64:
            return torch.int64
        else:
            raise ValueError(f"Unsupported ElementType: {typ}")

def init():
    PyTorchBackend.register()
    base.backend = PyTorchBackend()
