import torch
import numpy as np

def test(L, R):
  LHS = torch.tensor(L, requires_grad=True)
  RHS = torch.tensor(R, requires_grad=True)
  res = LHS / RHS
  print(f"{LHS}/{RHS}={res}")
  res.backward(gradient = torch.tensor(np.ones(res.shape)))
  print("LHS grad: ", LHS.grad)
  print("RHS grad: ", RHS.grad)
  print()

test([42.0,1.0],[2.0,4.0])
test([1.0],[2.0,4.0])
test([2.0,4.0],[2.0])
test([[42.0],[1.0]],[2.0,5.0])
test([[42.0,10.0],[3.0,0.5]],[2.0,5.0])