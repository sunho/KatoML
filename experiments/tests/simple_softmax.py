import torch
torch.manual_seed(0)
torch.set_printoptions(precision=7)

class LogisticRegression(torch.nn.Module):
  def __init__(self, n_inputs, n_outputs):
    super(LogisticRegression, self).__init__()
    self.linear = torch.nn.Linear(n_inputs, n_outputs)
    self.linear.weight.data.fill_(0.01)
    self.linear.bias.data.fill_(0.01)
  def forward(self, x):
    y_pred = self.linear(x)
    return y_pred

x = torch.randn(10, 5)
y_ = []
for i in range(10):
  if x[i].sum() > 0.0:
    y_.append([0.0,0.0,1.0])
  else:
    y_.append([1.0,0.0,0.0])
y = torch.tensor(y_)
print(x)
print(y)

criterion = torch.nn.CrossEntropyLoss()
model = LogisticRegression(5, 3)
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
criterion = torch.nn.CrossEntropyLoss()
losses = []
for i in range(5):
  optimizer.zero_grad()
  outputs = model(x)
  loss = criterion(outputs, y)
  loss.backward()
  optimizer.step()
  losses.append(loss.item())
  for param in model.parameters():
    print(param.grad)

print(losses)