import torch
from torchvision import datasets
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.__version__)
print(dir(datasets))