from huggingface_hub import snapshot_download

dataset_id = "FastVideo/Mixkit-Src"

# 下载并保存到本地指定路径
local_dir = snapshot_download(
    repo_id=dataset_id,
    repo_type="dataset",
    local_dir="/work/hdd/bcjw/jcai2/dataset/mixkit-src",
    local_dir_use_symlinks=False, 
    resume_download=True,
    max_workers=4
)

print(f"Dataset saved at: {local_dir}")