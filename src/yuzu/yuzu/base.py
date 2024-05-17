"""
yuzu base abstractions: BaseNode abstracts tensor node, BaseBackend abstracts tensor
auto grad engine (katoml or pytorch)
"""

from abc import ABC, abstractmethod, abstractclassmethod
from typing import Optional, Union, List, Tuple, Any
from enum import Enum
import math

class BaseNode(ABC):
    @abstractmethod
    def shape(self) -> tuple:
        pass

    @abstractmethod
    def dim(self) -> int:
        pass

    @abstractmethod
    def is_close(self, other: 'BaseNode', rtol: float = 1e-5, atol: float = 1e-8) -> 'BaseNode':
        pass

    @abstractmethod
    def __add__(self, other: Union['BaseNode', float]) -> 'BaseNode':
        pass

    @abstractmethod
    def __sub__(self, other: Union['BaseNode', float]) -> 'BaseNode':
        pass

    @abstractmethod
    def __mul__(self, other: Union['BaseNode', float]) -> 'BaseNode':
        pass

    @abstractmethod
    def __truediv__(self, other: Union['BaseNode', float]) -> 'BaseNode':
        pass

    @abstractmethod
    def __radd__(self, other: Union['BaseNode', float]) -> 'BaseNode':
        pass

    @abstractmethod
    def __rsub__(self, other: Union['BaseNode', float]) -> 'BaseNode':
        pass

    @abstractmethod
    def __rmul__(self, other: Union['BaseNode', float]) -> 'BaseNode':
        pass

    @abstractmethod
    def __rtruediv__(self, other: Union['BaseNode', float]) -> 'BaseNode':
        pass

    @abstractmethod
    def __neg__(self) -> 'BaseNode':
        pass

    @abstractmethod
    def __eq__(self, other: Union['BaseNode', float]) -> 'BaseNode':
        pass

    @abstractmethod
    def __iadd__(self, other: Union['BaseNode', Any]) -> 'BaseNode':
        pass

    @abstractmethod
    def __isub__(self, other: Union['BaseNode', Any]) -> 'BaseNode':
        pass

    @abstractmethod
    def __imul__(self, other: Union['BaseNode', Any]) -> 'BaseNode':
        pass

    @abstractmethod
    def __itruediv__(self, other: Union['BaseNode', Any]) -> 'BaseNode':
        pass

    @abstractmethod
    def __getitem__(self, index) -> 'BaseNode':
        pass

    @abstractmethod
    def __setitem__(self, index, value) -> 'BaseNode':
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def detach(self) -> 'BaseNode':
        pass

    @abstractmethod
    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> 'BaseNode':
        pass

    @abstractmethod
    def reshape(self, *shape) -> 'BaseNode':
        pass

    @abstractmethod
    def float(self) -> 'BaseNode':
        pass

    @abstractmethod
    def long(self) -> 'BaseNode':
        pass

    @abstractmethod
    def double(self) -> 'BaseNode':
        pass

    @abstractmethod
    def exp(self) -> 'BaseNode':
        pass

    @abstractmethod
    def log(self) -> 'BaseNode':
        pass

    @abstractmethod
    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> 'BaseNode':
        pass

    @abstractmethod
    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> 'BaseNode':
        pass

    @abstractmethod
    def gather(self, dim: int, index: 'BaseNode') -> 'BaseNode':
        pass

    @abstractmethod
    def clamp(self, min: Optional['float'] = None, max: Optional['float'] = None) -> 'BaseNode':
        pass

    @abstractmethod
    def max(self, other: Union['BaseNode', 'float']) -> 'BaseNode':
        pass

    @abstractmethod
    def min(self, other: Union['BaseNode', 'float']) -> 'BaseNode':
        pass

    @abstractmethod
    def reduce_max(self, dim: Optional[int] = None, keepdim: bool = False) -> 'BaseNode':
        pass

    @abstractmethod
    def reduce_min(self, dim: Optional[int] = None, keepdim: bool = False) -> 'BaseNode':
        pass

    @abstractmethod
    def item(self) -> 'float':
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def inner(self) -> Any:
        pass

    @abstractmethod
    def all(self) -> bool:
        pass

    @abstractmethod
    def any(self) -> bool:
        pass

    @abstractmethod
    def to(self, device: 'BaseDevice') -> 'BaseNode':
        pass
        
class BaseDevice(ABC):
    def __init__(self):
        super().__init__()

class ElementType(Enum):
    Float32 = 1
    Float64 = 2
    Int32 = 3
    Int64 = 4

int32 = ElementType.Int32
int64 = ElementType.Int32
long = ElementType.Int64
float32 = ElementType.Float32
float64 = ElementType.Float64

class BaseBackend(ABC):
    @staticmethod
    @abstractmethod
    def register():
        pass

    @staticmethod
    @abstractmethod
    def translate_type(typ: ElementType) -> Any:
        pass

    @abstractmethod
    def zeros(self, *size, requires_grad=False, device=None, dtype=None) -> 'BaseNode':
        pass

    @abstractmethod
    def rand(self, *size, requires_grad=False, device=None, dtype=None) -> 'BaseNode':
        pass

    @abstractmethod
    def from_numpy(self, arr, device=None) -> 'BaseNode':
        pass 

    @abstractmethod
    def from_value(self, value, device=None) -> 'BaseNode':
        pass 

    @abstractmethod
    def wrap(self, data) -> 'BaseNode':
        pass

    @abstractmethod
    def set_seed(self, seed):
        pass

    def clamp(self, input: 'BaseNode', min=None, max=None) -> 'BaseNode':
        return input.clamp(min, max)

    def reduce_min(self, input: 'BaseNode', dim=None, keepdim=False) -> 'BaseNode':
        return input.reduce_min(dim, keepdim)

    def reduce_max(self, input: 'BaseNode', dim=None, keepdim=False) -> 'BaseNode':
        return input.reduce_max(dim, keepdim)

    def sum(self, input: 'BaseNode', dim=None, keepdim=False) -> 'BaseNode':
        return input.sum(dim, keepdim)

    def gather(self, input: 'BaseNode', dim, index: 'YuzuNode') -> 'BaseNode':
        return input.gather(dim, index)

    def mean(self, input: 'BaseNode', dim=None, keepdim=False) -> 'BaseNode':
        return input.mean(dim, keepdim)

    def max(self, lhs: Union['BaseNode', float], rhs: Union['BaseNode', float]) -> 'BaseNode':
        if isinstance(lhs, BaseNode) and isinstance(rhs, BaseNode):
            return lhs.max(rhs)
        elif isinstance(lhs, BaseNode):
            return lhs.max(rhs)
        elif isinstance(rhs, BaseNode):
            return rhs.max(lhs)
        else:
            return self.from_value(max(lhs, rhs))

    def min(self, lhs: Union['BaseNode', float], rhs: Union['BaseNode', float]) -> 'BaseNode':
        if isinstance(lhs, BaseNode) and isinstance(rhs, BaseNode):
            return lhs.min(rhs)
        elif isinstance(lhs, BaseNode):
            return lhs.min(rhs)
        elif isinstance(rhs, BaseNode):
            return rhs.min(lhs)
        else:
            return self.from_value(min(lhs, rhs))


Node = BaseNode
Device = BaseDevice
Backend = BaseBackend
backend: Optional[BaseBackend] = None

BackendUninitException = Exception("backend not initialized")

def zeros(*size, requires_grad=False, device=None, dtype=None) -> 'BaseNode':
    if backend is None:
        raise BackendUninitException
    return backend.zeros(*size, requires_grad=requires_grad, device=device, dtype=dtype)

def rand(*size, requires_grad=False, device=None, dtype=None) -> 'BaseNode':
    if backend is None:
        raise BackendUninitException
    return backend.rand(*size, requires_grad=requires_grad, device=device, dtype=dtype)

def from_numpy(arr, device=None):
    if backend is None:
        raise BackendUninitException
    return backend.from_numpy(arr, device)

def clamp(input: 'BaseNode', min=None, max=None) -> 'BaseNode':
    if backend is None:
        raise BackendUninitException
    return backend.clamp(input, min, max)

def reduce_min(input: 'BaseNode', dim=None, keepdim=False) -> 'BaseNode':
    if backend is None:
        raise BackendUninitException
    return backend.reduce_min(input, dim, keepdim)

def reduce_max(input: 'BaseNode', dim=None, keepdim=False) -> 'BaseNode':
    if backend is None:
        raise BackendUninitException
    return backend.reduce_max(input, dim, keepdim)

def sum(input: 'BaseNode', dim=None, keepdim=False) -> 'BaseNode':
    if backend is None:
        raise BackendUninitException
    return backend.sum(input, dim, keepdim)

def gather(input: 'BaseNode', dim, index: 'BaseNode') -> 'BaseNode':
    if backend is None:
        raise BackendUninitException
    return backend.gather(input, dim, index)

def mean(input: 'BaseNode', dim=None, keepdim=False) -> 'BaseNode':
    if backend is None:
        raise BackendUninitException
    return backend.mean(input, dim, keepdim)

def exp(input: Union['BaseNode','float']) -> 'BaseNode':
    if backend is None:
        raise BackendUninitException
    if isinstance(input, BaseNode):
        return input.exp()
    return math.exp(input)

def log(input: Union['BaseNode','float']) -> 'BaseNode':
    if backend is None:
        raise BackendUninitException
    if isinstance(input, BaseNode):
        return input.log()
    return math.log(input)

def max(lhs: Union['BaseNode', float], rhs: Union['BaseNode', float]) -> 'BaseNode':
    if backend is None:
        raise BackendUninitException
    return backend.max(lhs, rhs)

def min(lhs: Union['BaseNode', float], rhs: Union['BaseNode', float]) -> 'BaseNode':
    if backend is None:
        raise BackendUninitException
    return backend.min(lhs, rhs)

def wrap(val: Any) -> 'BaseNode':
    if backend is None:
        raise BackendUninitException
    return backend.wrap(val)

def set_seed(seed):
    if backend is None:
        raise BackendUninitException
    backend.set_seed(seed)

def from_value(value, device = None) -> 'BaseNode':
    if backend is None:
        raise BackendUninitException
    return backend.from_value(value, device)
