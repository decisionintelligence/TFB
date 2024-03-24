import random

import numpy as np
import torch
SEED = 2021

def fix_random_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)


    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    # np.random.seed(SEED)
    # random.seed(SEED)
    # torch.cuda.manual_seed(SEED)

    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    # np.random.seed(SEED)
    # random.seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    # torch.cuda.manual_seed(SEED)

    # os.environ['PYTHONHASHSEED'] = str(1)