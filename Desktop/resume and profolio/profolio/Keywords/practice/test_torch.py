import numpy as np
import torch as torch

x = torch.rand((500,1))
y_true = x * 3 + 0.8

w = torch.rand((1,1),requires_grad = True)
b = torch.tensor(0, requires_grad = True,dtype = torch.float32)
y_predict = torch.matmul(x,w) + b
print(y_predict)