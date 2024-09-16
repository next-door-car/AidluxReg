import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 检测是否有gpu, device='cuda' if torch.cuda.is_available() else 'cpu'