import torch
import numpy as np

def test(V):
  VAL = torch.tensor(V, requires_grad=True)
  res = torch.exp(VAL)
  print(f"exp({VAL})={res}")
  res.backward(gradient = torch.tensor(np.ones(res.shape)))
  print("VAL grad: ", VAL.grad)

test([0.6,1.0])