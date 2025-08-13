#!/bin/bash

#================================================================
# SBATCH 指令部分: 告诉 Slurm 需要什么资源
#================================================================
# --account：指定作业的账户名称

#SBATCH --account=bcjw-delta-gpu

# --job-name：定义作业的名称，方便识别
#SBATCH --job-name=hunyuan_mixkit_preprocess

# --output：指定标准输出（STDOUT）和标准错误（STDERR）的输出文件
# %j 是一个占位符，会被替换为作业的ID (Job ID)
#SBATCH --output=/u/jcai2/slurm/hunyuan_mixkit_preprocess.out

# --ntasks：指定作业需要的总核心数（tasks）
#SBATCH --gres=gpu:1

#SBATCH --ntasks=1  # 开启一个进程

# --mem-per-cpu：为每个核心分配的内存大小
#SBATCH --mem=128G  # 申请2GB内存

#SBATCH --cpus-per-task=6

# --time：作业运行的最长时间限制 (格式: D-HH:MM:SS)
# 如果作业运行超过这个时间，Slurm 会自动终止它
#SBATCH --time=6:00:00  # 3小时

# --partition：指定作业提交到哪个分区（partition/queue）
#SBATCH --partition=gpuA100x4

# 要执行的命令部分: 你的程序和任务
#================================================================


# 打印一些有用的信息到输出文件
echo "作业开始了！"
echo "作业ID: $SLURM_JOB_ID"
echo "提交作业的主机: $SLURM_SUBMIT_HOST"
echo "运行作业的节点: $(hostname)"
echo "核心数: $SLURM_NTASKS"
echo "-----------------------------------"

# 加载你可能需要的软件模块 (这是HPC上常见的做法)
# module load python/3.9.1
# module load gcc/10.2.0

source /u/jcai2/miniconda3/etc/profile.d/conda.sh
conda activate fastvideo
cd /u/jcai2/video/MyDistillation

# hugging face cache dir
export HF_HOME=/tmp/jcai2/.cache/huggingface

huggingface-cli login --token $HuggingfaceToken

python /u/jcai2/video/MyDistillation/preprocess/HunyuanPreprocessT2V.py

echo "所有任务已完成！"
echo "作业结束！"