import torch
import numpy as np
torch.manual_seed(0)
torch.set_printoptions(precision=7)

def test(shape1, shape2):
  LHS = torch.randn(shape1, requires_grad=True)
  RHS = torch.randn(shape2, requires_grad=True)
  B = torch.randn(1)
  res = B*torch.matmul(LHS,RHS)
  print(f"{B}*matmul({LHS},{RHS})={res}")
  res.backward(gradient = torch.tensor(np.ones(res.shape)))
  print("LHS grad: ", LHS.grad)
  print("RHS grad: ", RHS.grad)
  print()

test([2,3],[3,2])