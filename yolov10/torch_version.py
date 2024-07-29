import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
import torch
print(torch.cuda.is_available())  # True가 나와야 합니다
print(torch.version.cuda)  # 11.4가 나와야 합니다
print(torch.backends.cudnn.version())  # 8201이 나와야 합니다

