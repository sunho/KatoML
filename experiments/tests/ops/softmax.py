import torch
import numpy as np
torch.set_printoptions(precision=7)
def test(V):
  VAL = torch.tensor(V, requires_grad=True)
  res = torch.log(torch.softmax(VAL, dim=len(VAL.shape)-1))
  B = torch.randn(res.shape)
  res = B*res
  print(f"B: {B}")
  print(f"{B} * log(softmax({VAL}))={res}")
  res.backward(gradient = torch.tensor(np.ones(res.shape)))
  print("VAL grad: ", VAL.grad)

test([0.6,1.0])
test([[0.6,1.0],[0.1,0.4]])