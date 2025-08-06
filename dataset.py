import torch

from torch import nn

from torch.utils.data import DataLoader, Dataset

class DummyDataset(Dataset):
    def __init__(self, length=1000, max_iterations=None):
        self.length = length
        self.max_iterations = max_iterations

    def __len__(self):
        return self.max_iterations if self.max_iterations else self.length

    def __getitem__(self, idx):
        # Return dummy data
        idx = idx % self.length
        # 原本形状是 (3, 16, 768, 1280) 的视频张量
        # vae 压缩倍率为 4x8x8
        #rand_video_tensor = torch.randn(3, 16, 768, 1280)
        rand_video_tensor = torch.randn(16, 4, 96, 160)
        prompt = "This is a dummy prompt."
        rand_video_tensor = rand_video_tensor.to(torch.bfloat16)
        return {"video": rand_video_tensor, "prompt": prompt}
