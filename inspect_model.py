import os

import torch

checkpoint_path = os.path.join(os.path.dirname(__file__), 'src', 'checkpoints', 'CNN_1024_30', '53.pth')

try:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    print(f"Checkpoint keys: {checkpoint.keys()}")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
