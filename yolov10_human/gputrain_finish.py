import torch

if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
torch.cuda.empty_cache()
