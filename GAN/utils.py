from PIL import Image
import random
import numpy as np
import torch
import torchvision
import os
import re


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # 原始为False
    torch.backends.cudnn.enabled = True # 新添加， 原始无
    
    
def load_dataset(input_path):
    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),        #  (3,H,W), 0-1
        torchvision.transforms.Normalize((0.5,), (0.5,)) # normalization
    ])
    img = torchvision.datasets.ImageFolder(input_path, transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(img, batch_size=64, shuffle=True)
    return test_loader