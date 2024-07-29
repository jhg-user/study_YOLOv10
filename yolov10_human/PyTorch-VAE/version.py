#import torch
#print(torch.cuda.is_available())
#print(torch.cuda.get_device_name(0))

import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

#import torch
#print(torch.__version__)
#print(torch.version.cuda)

