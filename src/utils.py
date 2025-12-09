import os
import random
import torch
import numpy as np

def set_seed(seed: int = 42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)