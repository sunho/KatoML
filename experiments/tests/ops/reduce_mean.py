import torch
import numpy as np
torch.manual_seed(0)
torch.set_printoptions(precision=7)

def test(VAL):
  VAL = torch.tensor(VAL, requires_grad=True)
  B = torch.randn(VAL.shape[0])
  res = B*torch.mean(VAL)
  print(f"{B}*{VAL}.sum()={res}")
  res.backward(gradient = torch.tensor(np.ones(res.shape)))
  print("VAL grad: ", VAL.grad)
  print()

test([42.0,1.0,53.0,20.0,10.0])