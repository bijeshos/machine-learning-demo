from __future__ import print_function
import torch

# construct a 5x3 matrix, uninitialized
x = torch.empty(5, 3)
print(x)

# construct a randomly initialized matrix
x = torch.rand(5, 3)
print(x)

# construct a matrix filled zeros and of dtype long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

