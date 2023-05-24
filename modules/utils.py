import torch
import math
import numpy as np
from torch import nn
from torch.nn import functional
from typing import Optional, Tuple, Union


try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
    cross_entropy_fn = CrossEntropyLoss()
    print('>>>> Flash Attention CE Loss')
except ImportError:
    CrossEntropyLoss = torch.nn.CrossEntropyLoss
    cross_entropy_fn = CrossEntropyLoss()

# @torch.compile
def _shift(lm_logits, labels):
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return shift_logits, shift_labels
    
def gpt_loss_func(input, target):
    lm_logits, labels = input, target
    shift_logits, shift_labels = _shift(lm_logits, labels)
    loss = cross_entropy_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss