import torch 
import random
import numpy as np
import os

def get_available_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    
CONFIG = {
    "seed": 42,
    "epochs": 12, # 42, ~MAX 20 hours of training
    "train_batch_size": 16,
    "valid_batch_size": 64,
    "learning_rate": 5e-5,
    "scheduler": 'CosineAnnealingLR',
    "min_lr": 5e-7,
    "T_max": 12,
    "weight_decay": 1e-6,
    "fold" : 0,
    "n_fold": 5,
    "n_accumulate": 1,
    "device": get_available_device(),
    'labels': ['unable_to_assess', 'not_close_match','close_match', 'near_exact_match', 'exact_match'], # 'unable_to_assess', 'not_close_match','close_match', 'near_exact_match', 'exact_match'
    'FP': 'molformer', # 'fp', 'molformer', 'ECFP', 'grover'
}
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(seed=CONFIG['seed'])

CONFIG_ECFP = {
    "seed": 42,
    'labels': ['unable_to_assess', 'not_close_match','close_match', 'near_exact_match', 'exact_match'], # 'unable_to_assess', 'not_close_match','close_match', 'near_exact_match', 'exact_match'
    "train_batch_size": 16,
    "valid_batch_size": 64,
    'FP': 'ECFP', # 'fp', 'molformer', 'ECFP', 'grover'
    'input_size': 2048
}

CONFIG_molformer = {
    "seed": 42,
    'labels': ['unable_to_assess', 'not_close_match','close_match', 'near_exact_match', 'exact_match'], # 'unable_to_assess', 'not_close_match','close_match', 'near_exact_match', 'exact_match'
    "train_batch_size": 16,
    "valid_batch_size": 64,
    'FP': 'molformer', # 'fp', 'molformer', 'ECFP', 'grover'
    'input_size': 768
}

CONFIG_fp = {
    "seed": 42,
    'labels': ['unable_to_assess', 'not_close_match','close_match', 'near_exact_match', 'exact_match'], # 'unable_to_assess', 'not_close_match','close_match', 'near_exact_match', 'exact_match'
    "train_batch_size": 16,
    "valid_batch_size": 64,
    'FP': 'fp',  # 'fp', 'molformer', 'ECFP', 'grover'
    'input_size': 2215
}

CONFIG_grover = {
    "seed": 42,
    'labels': ['unable_to_assess', 'not_close_match','close_match', 'near_exact_match', 'exact_match'], # 'unable_to_assess', 'not_close_match','close_match', 'near_exact_match', 'exact_match'
    "train_batch_size": 16,
    "valid_batch_size": 64,
    'FP': 'grover', # 'fp', 'molformer', 'ECFP', 'grover'
    'input_size': 5000
}