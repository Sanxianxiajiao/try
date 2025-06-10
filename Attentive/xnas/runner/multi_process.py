import torch.distributed as dist


def allow_save_checkpoint():
    return dist.get_rank()==0

def allow_validate():
    return dist.get_rank()==0

def allow_save_tb():
    return dist.get_rank()==0

def allow_logging():
    return dist.get_rank()==0
