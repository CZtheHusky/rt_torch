from .initialize import *


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def print_with_rank(message):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        print(f"rank: {rank}", message, flush=True)
    else:
        print(message, flush=True)

