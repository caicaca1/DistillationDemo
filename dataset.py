import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import glob
import pyarrow.parquet as pq
import numpy as np

def load_parquet_record(path, idx=0):
    """读取 parquet 文件中的第 idx 条记录，并还原所有张量"""
    table = pq.read_table(path)
    data = table.to_pydict()  # 转成 Python dict（列名 -> list）

    def restore_tensor(prefix):
        """根据字段前缀还原 numpy/tensor"""
        arr_bytes = data[f"{prefix}_bytes"][idx]
        arr_shape = data[f"{prefix}_shape"][idx]
        arr_dtype = data[f"{prefix}_dtype"][idx]
        np_array = np.frombuffer(arr_bytes, dtype=arr_dtype).reshape(arr_shape)
        return torch.from_numpy(np_array)  # 返回 torch.Tensor

    record = {
        "id": data["id"][idx],
        "prompt": data["caption"][idx],
        "file_name": data["file_name"][idx],
        "media_type": data["media_type"][idx],
        "width": data["width"][idx],
        "height": data["height"][idx],
        "num_frames": data["num_frames"][idx],
        "duration_sec": data["duration_sec"][idx],
        "fps": data["fps"][idx],
        "video_latents": restore_tensor("vae_latent"),
        "prompt_embeds": restore_tensor("text_embedding"),
        "pooled_prompt_embeds": restore_tensor("pooled_text_embedding"),
        "prompt_attention_mask": restore_tensor("text_attention_mask"),
    }
    return record

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
        rand_video_tensor = torch.randn(16, 16, 56, 104)
        prompt = "This is a dummy prompt."
        rand_video_tensor = rand_video_tensor.to(torch.bfloat16)
        return {"video": rand_video_tensor, "prompt": prompt}

class ParquetHunyuanMixkitDataset(Dataset):
    def __init__(self, data_folder, max_samples=None):
        self.data_path = data_folder
        self.data_files = glob.glob(f"{self.data_path}/*.parquet")
        self.load_fn = load_parquet_record
        self.max_samples = max_samples
        if max_samples:
            self.data_files = self.data_files[:max_samples]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        """获取指定索引的样本"""
        item = self.data_files[idx]
        record = self.load_fn(item, idx=0)
        return record
