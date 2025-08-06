# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

import copy
import os
from dataclasses import dataclass, field
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from einops import rearrange
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator
from huggingface_hub import snapshot_download

# 假设我们有DiT模型定义，类似于diffusers中的UNet
# from my_models import DiTModel, DiTConfig # 这是一个假设的模型定义
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, AutoencoderKLHunyuanVideo

#TO DO： check FSDP; check scheduler; check memory usage

DEFAULT_PROMPT_TEMPLATE = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}

# --- 1. 配置参数 (使用dataclass替代argparse，更清晰) ---
@dataclass
class DistillHunyuanArgs:
    # 模型路径
    hunyuan_model_path: str = "hunyuanvideo-community/HunyuanVideo" # 完整混元模型路径
    output_dir: str = "/work/hdd/bcjw/jcai2/hunyuan_distilled_output"

    # 学生和批评家模型配置
    student_num_layers: int = 10  # 学生模型深度 (原版可能为24或48)
    student_num_single_layers: int = 20 # 学生模型隐藏层维度 (原版可能为1152)
    student_num_attention_heads: int = 18 # 学生模型注意力头数 10,20,18 刚好塞满一张卡 vae先处理后的setting下

    # 训练参数
    learning_rate: float = 1e-5
    critic_learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    max_train_iterations: int = 10000
    lr_warmup_steps: int = 100
    batch_size: int = 1
    num_workers: int = 4
    gradient_accumulation_steps: int = 4
    total_timesteps: int = 1000 # 总时间步数 (Flow Matching调度器的时间步数)
    train_log_loss_steps: int = 100 # 每10步记录一次训练损失
    guidance_scale_min : float = 1.0 # 最小guidance scale
    guidance_scale_max : float = 10.0 # 最大guidance scale
    FSDP: bool = False # 是否使用FSDP分布式训练
    enable_checkpointing: bool = False # 是否启用检查点保存
    # 蒸馏核心参数
    generator_update_interval: int = 5 # 每训练5次批评家，才训练1次学生
    real_score_guidance_scale: float = 7.5 # 教师模型的CFG scale
    flow_shift: float = 1.0 # 时间步偏移，用于Flow Matching
    dmd: bool = True # 是否启用DMD

    # 日志和保存
    log_with: str = "wandb"
    validation_steps: int = 500
    checkpointing_steps: int = 1000


class HunyuanDistillationPipeline:
    """
    用于蒸馏HunyuanVideo的对抗性蒸馏管线。
    借鉴了源文件的“学生-教师-批评家”三体博弈思想。
    """
    _callback_tensor_inputs = ["latents", "prompt_embeds"]
    def __init__(self, args: DistillHunyuanArgs, accelerator: Optional[Accelerator] = None):
        
        self.args = args
        self.accelerator = accelerator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = self.accelerator.device if self.accelerator else self.device
        
        # 初始化模型和优化器为空
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.noise_scheduler = None
        
        self.teacher_hunyuan_dit = None # 教师/真评委
        self.student_dit = None         # 学生/生成器
        self.fake_score_dit = None          # 批评家/假评委

        self.student_optimizer = "AdamW"
        self.fake_score_optimizer = None
        self.student_lr_scheduler = 1e-5
        self.fake_score_lr_scheduler = None
        
        os.makedirs(self.args.output_dir, exist_ok=True)
          
    def setup_models_and_optimizers(self):
        """
        初始化所有需要的模型、调度器和优化器。
        """
        print("--- 正在设置模型和优化器 ---")
        
        # --- 初始化三个核心DiT模型 ---
        # 1. teacher
        self.teacher_pipe = HunyuanVideoPipeline.from_pretrained(
            self.args.hunyuan_model_path, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.teacher_transformer = self.teacher_pipe.transformer
        # --- 初始化噪声调度器 (使用Flow Matching调度器) ---
        if self.args.dmd:
            if self.accelerator is None or self.accelerator.is_main_process:
                self.noise_scheduler = self.teacher_pipe.scheduler
            if self.accelerator is not None:
                self.noise_scheduler = self.accelerator.broadcast(self.noise_scheduler)
        else:
            if self.accelerator is None or self.accelerator.is_main_process:
                self.noise_scheduler = DDPMScheduler(
                    num_train_timesteps=self.args.total_timesteps,
                ).to(self.device)
            if self.accelerator is not None:
                self.noise_scheduler = self.accelerator.broadcast(self.noise_scheduler)
        print(f"使用FlowMatching噪声调度器: {self.noise_scheduler.__class__.__name__}")
        
        self.vae = self.teacher_pipe.vae
        
        # 冻结这些不需要训练的组件
        self.teacher_transformer.requires_grad_(False)
        self.teacher_transformer.eval()
        self.vae.requires_grad_(False)
        self.vae.eval()
        print(f"教师模型 (teacher_hunyuan_dit) 加载完成并冻结。")

        # 2. 学生/生成器 (创建轻量化版本)
        teacher_config = self.teacher_transformer.config
        student_config = dict(copy.deepcopy(teacher_config))
        student_config["num_layers"] = self.args.student_num_layers
        student_config["num_single_layers"] = self.args.student_num_single_layers
        student_config["num_attention_heads"] = self.args.student_num_attention_heads
        print(f"学生模型配置: {student_config}")
        self.student_dit = HunyuanVideoTransformer3DModel.from_config(student_config).to(self.device)     
        print(f"学生模型 (student_dit) 初始化完成。")
        self.student_dit.train()
            
        # ************************************************************************************
        #sys.exit(0)
        # ************************************************************************************

        # --- 初始化优化器和学习率调度器 ---
        self.student_optimizer = torch.optim.AdamW(
            self.student_dit.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        self.student_lr_scheduler = get_scheduler(
            "cosine", optimizer=self.student_optimizer,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.max_train_iterations
        )
        if self.args.dmd:
            print(f"Initializing DMD fake score model with same student config")
            self.fake_score_dit = HunyuanVideoTransformer3DModel.from_config(student_config).to(self.device)
            self.fake_score_dit.train()
            self.fake_score_optimizer = torch.optim.AdamW(
                self.fake_score_dit.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
            self.fake_score_lr_scheduler = get_scheduler(
                "cosine", optimizer=self.fake_score_optimizer,
                num_warmup_steps=self.args.lr_warmup_steps,
                num_training_steps=self.args.max_train_iterations
            )
        if self.args.FSDP:
            if not self.args.dmd:
                self.student_dit, self.student_optimizer, self.train_dataloader = self.accelerator.prepare(
                    self.student_dit, self.student_optimizer, self.train_dataloader
                )
            else:
                self.student_dit, self.student_optimizer, self.fake_score_dit, self.fake_score_optimizer, self.train_dataloader = self.accelerator.prepare(
                    self.student_dit, self.student_optimizer, self.fake_score_dit, self.fake_score_optimizer, self.train_dataloader
                )
            
        if self.args.enable_checkpointing:
            self.student_dit.enable_checkpointing()
            if self.args.dmd:
                self.fake_score_dit.enable_checkpointing()
            print("学生模型启用检查点保存。")
        print("--- 设置完成 ---")

    def setup_models_and_optimizers_FSDP(self):
        """
        初始化所有需要的模型、调度器和优化器，兼容多进程 + FSDP。
        """
        self.accelerator.print("--- 正在设置FSDP兼容的模型和优化器 ---")

        # ---------------------------
        # 1. 主进程负责预下载模型到本地缓存
        # ---------------------------
        # 使用 main_process_first() 确保下载只发生一次
        with accelerator.main_process_first():
            snapshot_download(repo_id="hunyuanvideo-community/HunyuanVideo")

        # --------------------------
        # 2. 所有进程都从本地缓存加载自己的 teacher_pipe 实例
        # ---------------------------
        # 使用 low_cpu_mem_usage=True 可以优化内存使用，特别是在多进程环境下
        self.teacher_pipe = HunyuanVideoPipeline.from_pretrained(
            self.args.hunyuan_model_path, 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True 
        ).to(self.device)

        # 从 pipe 中提取需要的组件
        self.teacher_transformer = self.teacher_pipe.transformer
        self.noise_scheduler = self.teacher_pipe.scheduler
        self.text_encoder = self.teacher_pipe.text_encoder
        self.text_encoder_2 = self.teacher_pipe.text_encoder_2
        self.vae = self.teacher_pipe.vae
        
        # 冻结并移动到设备。这些组件不参与训练，所以手动管理
        self.teacher_transformer.requires_grad_(False)
        self.teacher_transformer.eval()
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.text_encoder_2.requires_grad_(False)
        self.text_encoder_2.eval()
        self.accelerator.print("所有进程的 Teacher Pipeline 加载完成并冻结。")

        # ---------------------------
        # 4. 初始化学生模型
        # ---------------------------
        teacher_config = self.teacher_transformer.config
        student_config = dict(copy.deepcopy(teacher_config))
        student_config["num_layers"] = self.args.student_num_layers
        student_config["num_single_layers"] = self.args.student_num_single_layers
        student_config["num_attention_heads"] = self.args.student_num_attention_heads

        self.student_dit = HunyuanVideoTransformer3DModel.from_config(student_config)
        self.student_dit.train()

        # ---------------------------
        # 5. 初始化优化器 & 学习率调度器
        # ---------------------------
        self.student_optimizer = torch.optim.AdamW(
            self.student_dit.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        self.student_lr_scheduler = get_scheduler(
            "cosine", optimizer=self.student_optimizer,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.max_train_iterations
        )

        if self.args.dmd:
            self.accelerator.print("Initializing DMD fake score model...")
            self.fake_score_dit = HunyuanVideoTransformer3DModel.from_config(student_config)
            self.fake_score_dit.train()

            self.fake_score_optimizer = torch.optim.AdamW(
                self.fake_score_dit.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
            self.fake_score_lr_scheduler = get_scheduler(
                "cosine", optimizer=self.fake_score_optimizer,
                num_warmup_steps=self.args.lr_warmup_steps,
                num_training_steps=self.args.max_train_iterations
            )

        # ---------------------------
        # 6. FSDP 分发
        # ---------------------------

        if self.args.dmd:
            self.student_dit, self.fake_score_dit, self.student_optimizer, self.fake_score_optimizer, self.train_dataloader = self.accelerator.prepare(
                self.student_dit, self.fake_score_dit, self.student_optimizer, self.fake_score_optimizer, self.train_dataloader
            )
        else:
            self.student_dit, self.student_optimizer, self.train_dataloader = self.accelerator.prepare(
                self.student_dit, self.student_optimizer, self.train_dataloader
            )

        # ---------------------------
        # 7. FSDP checkpointing
        # ---------------------------
        if self.args.enable_checkpointing:
            self.student_dit.enable_checkpointing()
            if self.args.dmd:
                self.fake_score_dit.enable_checkpointing()
            self.accelerator.print("学生模型启用检查点保存。")

        self.accelerator.print("--- 设置完成 ---")

    def _get_llama_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        prompt_template: Dict[str, Any],
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
        num_hidden_layers_to_skip: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.teacher_pipe._get_llama_prompt_embeds(
                    prompt=prompt,
                    prompt_template=prompt_template,
                    num_videos_per_prompt=num_videos_per_prompt,
                    device=device,
                    dtype=dtype,
                    max_sequence_length=max_sequence_length,
                    num_hidden_layers_to_skip=num_hidden_layers_to_skip,
                )
        
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 77,
    ) -> torch.Tensor:
        return self.teacher_pipe._get_clip_prompt_embeds(
                    prompt=prompt,
                    num_videos_per_prompt=num_videos_per_prompt,
                    device=device,
                    dtype=dtype,
                    max_sequence_length=max_sequence_length,
                )
        
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]] = None,
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
    ):
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_llama_prompt_embeds(
                prompt,
                prompt_template,
                num_videos_per_prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=max_sequence_length,
            )

        if pooled_prompt_embeds is None:
            if prompt_2 is None:
                prompt_2 = prompt
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt,
                num_videos_per_prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=77,
            )

        return prompt_embeds, pooled_prompt_embeds, prompt_attention_mask
    
    def _set_dataset(self):
        """
        设置训练和验证数据集。先实现虚拟数据集
        """
        from dataset import DummyDataset  

        self.train_dataset = DummyDataset(length=1000, max_iterations=self.args.max_train_iterations)
        self.validation_dataset = DummyDataset(length=100)
        print("dataset 设置完成。")
        
    def _set_dataloaders(self):
        """
        设置数据加载器，使用 PyTorch 的 DataLoader。
        """
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        self.validation_dataloader = DataLoader(
            self.validation_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        print("dataloader 设置完成。")
    
    def _add_noise_to_latents(self, latents: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        根据给定的噪声和时间步，将噪声添加到潜变量中。
        """
        return self.noise_scheduler.add_noise(latents, noise, timestep)

    def _sample_timestep(self, batch_size: int) -> torch.Tensor:
        """
        随机选择一个时间步，用于生成噪声，返回形状为 [batch_size] 的 tensor。
        """
        idx = torch.randint(0, len(self.noise_scheduler.timesteps), (batch_size,), device=self.device)
        sampled_timesteps = self.noise_scheduler.timesteps[idx].to(self.device)
        return sampled_timesteps
    
    
    def _pred_noise_to_previous_step(self, pred_noise, noise_input_latent, timestep):
        """辅助函数：根据预测的噪声计算预测的视频潜变量xt-1"""
        return self.noise_scheduler.step(pred_noise, timestep, noise_input_latent, return_dict=False)[0]

    def _get_inputs(self, batch_data: Dict[str, Any]):
        
        video_tensors = batch_data["video"]
        
        prompt = batch_data["prompt"]  
        prompt_2 = batch_data.get("prompt_2", prompt)
        prompt_embeds = batch_data.get("prompt_embeds", None)
        pooled_prompt_embeds = batch_data.get("pooled_prompt_embeds", None)
        prompt_attention_mask = batch_data.get("prompt_attention_mask", None)
        
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_template=DEFAULT_PROMPT_TEMPLATE,
            num_videos_per_prompt=1,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            device=self.device,
            max_sequence_length=256,
        )
        
        batch_data["prompt_embeds"] = prompt_embeds.to(self.device)
        batch_data["pooled_prompt_embeds"] = pooled_prompt_embeds.to(self.device)
        batch_data["prompt_attention_mask"] = prompt_attention_mask.to(self.device)
        
        # 对视频进行VAE编码并采样 Hunyuan VAE的downsample rate是 4x8x8 之后patchify还有 1x2x2
        #with autocast(dtype=torch.bfloat16):
        #    video_latents = self.vae.encode(video_tensors.to(self.device)).latent_dist.sample()
        # 对潜变量进行缩放
        #video_latents = video_latents * self.vae.config.scaling_factor
        #########################################
        #Debugging: 先假装直接有处理后的latents
        video_latents = video_tensors
        #########################################
        batch_data['video_latents'] = video_latents.to(self.device)
        batch_data.pop("video", None)  # 移除原始视频数据，节省内存
        return batch_data
    
    def _train_step(self, batch_data: Dict[str, Any], attention_kwargs: Optional[Dict[str, Any]] = None):
        """
        一个标准的、非蒸馏的训练步骤。
        只训练学生模型，让其学会预测噪声。
        """
        # --- 1. 数据准备 ---
        # 获取潜变量和文本嵌入
        video_latents = batch_data["video_latents"]
        prompt_embeds = batch_data["prompt_embeds"]
        pooled_prompt_embeds = batch_data["pooled_prompt_embeds"]
        prompt_attention_mask = batch_data["prompt_attention_mask"]

        # 随机采样时间步 t
        t = self._sample_timestep(video_latents.shape[0])

        # 创建目标噪声 (ground truth)
        noise = torch.randn_like(video_latents)
        
        # 将噪声添加到干净的潜变量中，得到加噪潜变量 xt
        noisy_latents = self._add_noise_to_latents(video_latents, noise, t)
        w_min = self.args.guidance_scale_min
        w_max = self.args.guidance_scale_max
        # 生成一个形状为 (batch_size,) 的张量，每个元素都是一个随机的guidance_scale
        guidance_scale = (w_max - w_min) * torch.rand(self.args.batch_size, device=self.device) + w_min

        guidance_for_model = guidance_scale * 1000.0
        guidance_for_model = guidance_for_model.to(dtype=self.student_dit.dtype)
        # --- 2. 学生模型预测 ---
        # 模型的目标是根据 xt 和条件，预测出我们刚刚加入的 noise
        with autocast(dtype=torch.bfloat16):
            student_pred_noise = self.student_dit(
                        hidden_states=noisy_latents,
                        timestep=t,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        pooled_projections=pooled_prompt_embeds,
                        guidance=guidance_for_model,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
            # --- 3. 计算损失 ---
            # 计算学生预测的噪声和真实添加的噪声之间的均方误差 (MSE)
            # 这是扩散模型训练最核心的损失函数
            loss = F.mse_loss(student_pred_noise, noise)

        # 在实际的训练循环中，你会在这里执行:
        # self.student_optimizer.zero_grad()
        # loss.backward()
        # self.student_optimizer.step()
        
        # --- 4. 返回损失值用于日志记录 ---
        return {
            "loss": loss
        }
    
    def _student_one_step(self, batch_data: Dict[str, Any], attention_kwargs: Optional[Dict[str, Any]] = None):
        video_latents = batch_data["video_latents"]
        prompt_embeds = batch_data["prompt_embeds"]
        pooled_prompt_embeds = batch_data["pooled_prompt_embeds"]
        prompt_attention_mask = batch_data["prompt_attention_mask"]

        fixed_max_timestep = self.noise_scheduler.timesteps[0]

        # 创建输入噪声
        noise = torch.randn_like(video_latents)

        v_target_student = noise - video_latents

        w_min = self.args.guidance_scale_min
        w_max = self.args.guidance_scale_max
        # 生成一个形状为 (batch_size,) 的张量，每个元素都是一个随机的guidance_scale
        guidance_scale = (w_max - w_min) * torch.rand(self.args.batch_size, device=self.device) + w_min

        guidance_for_model = guidance_scale * 1000.0
        guidance_for_model = guidance_for_model.to(dtype=self.student_dit.dtype)
        # --- 2. 学生模型预测 ---
        with autocast(dtype=torch.bfloat16):
            student_pred_noise = self.student_dit(
                        hidden_states=noise,
                        timestep=fixed_max_timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        pooled_projections=pooled_prompt_embeds,
                        guidance=guidance_for_model,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
        student_pred_video = self._fm_from_pred_velocity_to_pred_video(student_pred_noise, noise, fixed_max_timestep)
        regression_loss = F.mse_loss(student_pred_noise, v_target_student)
        return regression_loss, student_pred_video, guidance_for_model
    
    def pred_noise_to_pred_video(self, pred_noise, noisy_latents, timestep): #DDPM
        """
        根据预测的噪声和加噪潜变量，计算预测的视频潜变量 xt-1。
        """
        t = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
        prev_t = self.noise_scheduler.previous_timestep(t)
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.noise_scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.noise_scheduler.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://huggingface.co/papers/2006.11239
        pred_original_sample = (noisy_latents - beta_prod_t ** (0.5) * pred_noise) / alpha_prod_t ** (0.5)
        
        return pred_original_sample
    
    def _dmd_step(self, batch_data: Dict[str, Any], student_pred_video, attention_kwargs: Optional[Dict[str, Any]] = None, guidance_for_model: Optional[float] = None):
        video_latents = batch_data["video_latents"]
        prompt_embeds = batch_data["prompt_embeds"]
        pooled_prompt_embeds = batch_data["pooled_prompt_embeds"]
        prompt_attention_mask = batch_data["prompt_attention_mask"]

        # 随机采样时间步 t
        t = self._sample_timestep(video_latents.shape[0])

        # 创建目标噪声 (ground truth)
        noise = torch.randn_like(student_pred_video)
        
        # 将噪声添加到干净的潜变量中，得到加噪潜变量 xt
        noisy_student_pred_video = self.noise_scheduler.scale_noise(student_pred_video, t, noise)

        v_traget_fake_score = noise - student_pred_video ######### 应该是 noise - x0 还是 x0 - noise ?

        # --- 2. teacher模型预测 ---
        with autocast(dtype=torch.bfloat16):
            teacher_pred_noise = self.teacher_transformer(
                        hidden_states=noisy_student_pred_video,
                        timestep=t,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        pooled_projections=pooled_prompt_embeds,
                        guidance=guidance_for_model,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
            fake_score_pred_noise = self.fake_score_dit(
                        hidden_states=noisy_student_pred_video,
                        timestep=t,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        pooled_projections=pooled_prompt_embeds,
                        guidance=guidance_for_model,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
        teacher_pred_video = self._fm_from_pred_velocity_to_pred_video(teacher_pred_noise, noisy_student_pred_video, t)
        fake_score_pred_video = self._fm_from_pred_velocity_to_pred_video(fake_score_pred_noise, noisy_student_pred_video, t)
        diffusion_loss = F.mse_loss(fake_score_pred_noise, v_traget_fake_score.detach())
        return teacher_pred_video, fake_score_pred_video, diffusion_loss
    
    def _fm_from_pred_velocity_to_pred_video(self, model_output, noisy_latents, timestep): #FlowMatching
        """
        from velocity to video
        """
        sigmas = self.noise_scheduler.sigmas.to(noisy_latents.device)
        schedule_timesteps = self.noise_scheduler.timesteps.to(noisy_latents.device)
        step_indices = [self.noise_scheduler.index_for_timestep(t, schedule_timesteps) for t in timestep]
        sigma = sigmas[step_indices].flatten()
        
        # 调整 sigma 的维度用于广播
        while len(sigma.shape) < len(noisy_latents.shape):
            sigma = sigma.unsqueeze(-1)
            
        # x_clean_pred = x_t - t * v_pred 其中 v_pred 为 noise - x0
        pred_video = noisy_latents - sigma * model_output
        
        return pred_video
    
    def dmd_distill(self, attention_kwargs: Optional[Dict[str, Any]] = None,):
        self.setup_pipeline()
        dmd_losses = 0
        regression_losses = 0
        diffusion_losses = 0
        for i, batch_data in enumerate(self.train_dataloader):
            with self.accelerator.autocast(), self.accelerator.accumulate(self.student_dit):
                # 1. 准备输入数据
                i = i + 1
                batch_data = self._get_inputs(batch_data)
                
                # 2. 执行训练步骤
                regression_loss, student_pred_video, guidance_for_model = self._student_step(batch_data, attention_kwargs)
                
                teacher_pred_video, fake_score_pred_video, diffusion_loss = self._dmd_step(batch_data, student_pred_video, attention_kwargs, guidance_for_model)

                with torch.no_grad():
                    grad = (fake_score_pred_video - teacher_pred_video) / torch.abs(
                        student_pred_video - teacher_pred_video).mean()
                    grad = torch.nan_to_num(grad)

                dmd_loss = 0.5 * F.mse_loss(
                    student_pred_video.float(),
                    (student_pred_video.float() - grad.float()).detach())
                
                distill_loss = regression_loss + dmd_loss
                
                self.student_optimizer.zero_grad(set_to_none=True)
                self.accelerator.backward(distill_loss)
                self.student_optimizer.step()
                self.student_lr_scheduler.step()
                
                self.fake_score_optimizer.zero_grad(set_to_none=True)
                # 注意：fake_score_loss 是在 autocast 上下文中计算的，所以它的反向传播也应该在这里进行
                self.accelerator.backward(diffusion_loss)
                self.fake_score_optimizer.step()
                self.fake_score_lr_scheduler.step()

                dmd_losses += dmd_loss.detach().item()
                regression_losses += regression_loss.detach().item()
                diffusion_losses += diffusion_loss.detach().item()
                print(dmd_losses)
                sys.exit(0)
            
    def setup_pipeline(self):
        # 加载模型和优化器
        # 加载数据集
        self.setup_models_and_optimizers()
        self._set_dataset()
        self._set_dataloaders() 
        
    def normal_train(self, attention_kwargs: Optional[Dict[str, Any]] = None,):
        
        self.setup_pipeline()
        #sys.exit(0)
        
        losses = 0
        
        for i, batch_data in enumerate(self.train_dataloader):
            # 1. 准备输入数据
            i = i + 1
            batch_data = self._get_inputs(batch_data)
            
            # 2. 执行训练步骤
            outputs = self._train_step(batch_data, attention_kwargs)

            loss = outputs["loss"]
            
            # 2.1 梯度清零
            self.student_optimizer.zero_grad()
            # 2.2 反向传播
            loss.backward()
            # 2.3 更新参数
            self.student_optimizer.step()
            self.student_lr_scheduler.step()
            
            losses += outputs["loss"].item()
            print(losses)
            sys.exit(0)
            if i % self.args.train_log_loss_steps == 0:
                # 3. 日志记录
                print(f"Step {i}, Loss: {losses / self.args.train_log_loss_steps:.4f}")
                losses = 0
                
            # 4. 保存模型检查点 (可选)
            if i % self.args.checkpointing_steps == 0:
                print(f"保存模型检查点到 {self.args.output_dir} ...")
                # torch.save(self.student_dit.state_dict(), os.path.join(self.args.output_dir, f"student_epoch_{epoch}.pth"))
      
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Union[str, List[str]] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        num_inference_steps: int = 50,
        sigmas: List[float] = None,
        true_cfg_scale: float = 1.0,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        max_sequence_length: int = 256,
    ):
        # 1. Check inputs. Raise error if not correct

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        transformer_dtype = self.transformer.dtype
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_template=prompt_template,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            device=self.device,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

        if do_true_cfg:
            negative_prompt_embeds, negative_pooled_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_template=prompt_template,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                prompt_attention_mask=negative_prompt_attention_mask,
                device=self.device,
                max_sequence_length=max_sequence_length,
            )
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(transformer_dtype)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            self.device,
            generator,
            latents,
        )

        # 6. Prepare guidance condition
        guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=self.device) * 1000.0

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    pooled_projections=pooled_prompt_embeds,
                    guidance=guidance,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if do_true_cfg:
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_attention_mask=negative_prompt_attention_mask,
                        pooled_projections=negative_pooled_prompt_embeds,
                        guidance=guidance,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HunyuanVideoPipelineOutput(frames=video)
    
    def train_critic_step(self, training_batch):
        """训练批评家/假评委 (`critic_dit`) 的一轮"""
        self.critic_optimizer.zero_grad()
        
        # 1. 学生模型生成视频 (在no_grad下，因为此步不训练学生)
        with torch.no_grad():
            # 随机选择一个初始的去噪时间步
            timestep_gen = torch.randint(0, len(self.noise_scheduler.timesteps), [1], device=self.device).long()
            noise = torch.randn_like(training_batch["latents"])
            noisy_latents = self.noise_scheduler.add_noise(training_batch["latents"], noise, timestep_gen)
            
            # 学生模型进行预测
            pred_noise_student = self.student_dit(
                hidden_states=noisy_latents,
                encoder_hidden_states=training_batch["prompt_embeds"],
                timestep=timestep_gen
            ).sample
            
            # 得到学生预测的视频x0
            student_pred_video = self._pred_noise_to_pred_video(pred_noise_student, noisy_latents, timestep_gen)

        # 2. 准备评判样本 (给学生生成的视频加噪)
        timestep_critic = torch.randint(0, len(self.noise_scheduler.timesteps), [1], device=self.device).long()
        critic_noise = torch.randn_like(student_pred_video)
        noisy_student_video = self.noise_scheduler.add_noise(student_pred_video, critic_noise, timestep_critic)

        # 3. 批评家进行预测
        critic_pred_noise = self.critic_dit(
            hidden_states=noisy_student_video,
            encoder_hidden_states=training_batch["prompt_embeds"],
            timestep=timestep_critic
        ).sample

        # 4. 计算损失 (目标是预测出我们刚加入的噪声`critic_noise`)
        # 注意: 源文件中的target是 `noise - video`，这与预测x0的公式有关，这里简化为直接预测噪声
        loss = F.mse_loss(critic_pred_noise.float(), critic_noise.float(), reduction="mean")
        
        # 5. 反向传播，更新批评家
        loss.backward()
        self.critic_optimizer.step()
        self.critic_lr_scheduler.step()
        
        return loss.detach().item()

    def train_generator_step(self, training_batch):
        """训练学生/生成器 (`student_dit`) 的一轮 (对抗性蒸馏)"""
        self.student_optimizer.zero_grad()
        
        # 1. 学生模型生成视频 (这次需要计算梯度)
        timestep_gen = torch.randint(0, len(self.noise_scheduler.timesteps), [1], device=self.device).long()
        noise = torch.randn_like(training_batch["latents"])
        noisy_latents = self.noise_scheduler.add_noise(training_batch["latents"], noise, timestep_gen)
        
        pred_noise_student = self.student_dit(
            hidden_states=noisy_latents,
            encoder_hidden_states=training_batch["prompt_embeds"],
            timestep=timestep_gen
        ).sample
        student_pred_video = self._pred_noise_to_pred_video(pred_noise_student, noisy_latents, timestep_gen)

        # 2. 获取教师和批评家的“意见” (在no_grad下)
        with torch.no_grad():
            # 准备相同的加噪样本
            timestep_adv = torch.randint(0, len(self.noise_scheduler.timesteps), [1], device=self.device).long()
            adv_noise = torch.randn_like(student_pred_video)
            noisy_student_video = self.noise_scheduler.add_noise(student_pred_video, adv_noise, timestep_adv)

            # 教师的预测
            teacher_pred_noise = self.teacher_hunyuan_dit(
                hidden_states=noisy_student_video,
                encoder_hidden_states=training_batch["prompt_embeds"],
                timestep=timestep_adv
            ).sample
            teacher_pred_video = self._pred_noise_to_pred_video(teacher_pred_noise, noisy_student_video, timestep_adv)

            # 批评家的预测
            critic_pred_noise = self.critic_dit(
                hidden_states=noisy_student_video,
                encoder_hidden_states=training_batch["prompt_embeds"],
                timestep=timestep_adv
            ).sample
            critic_pred_video = self._pred_noise_to_pred_video(critic_pred_noise, noisy_student_video, timestep_adv)
        
        # 3. 计算“改进梯度” (Adversarial Gradient)
        # 这个梯度指明了学生当前输出与教师输出之间的差距方向
        # 我们希望学生向教师靠近，所以要减去这个梯度
        grad = critic_pred_video - teacher_pred_video
        
        # 4. 计算DMD损失
        # 目标是让 student_pred_video 经过修正后 (减去grad)，与原始的 student_pred_video 尽可能接近
        loss = 0.5 * F.mse_loss(student_pred_video.float(), (student_pred_video.float() - grad.float()).detach())
        
        # 5. 反向传播，更新学生
        loss.backward()
        self.student_optimizer.step()
        self.student_lr_scheduler.step()
        
        return loss.detach().item()


    def train_loop(self):
        """
        主训练循环
        """
        # --- 此处省略数据加载逻辑 ---
        # 你需要一个可以产出 `{"video": tensor, "prompt": str}` 字典的数据加载器
        # dummy_dataset = [{"video": torch.randn(3, 16, 256, 256), "prompt": "an astronaut riding a horse"}]
        # train_dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=self.args.batch_size)
        # --- 数据加载逻辑结束 ---
        
        progress_bar = tqdm(range(self.args.max_train_steps))
        progress_bar.set_description("蒸馏训练中...")
        
        for step in range(self.args.max_train_steps):
            # for batch_data in train_dataloader: # 假设从数据加载器获取数据
            
            # --- 模拟获取一批数据 ---
            prompts = ["an astronaut riding a horse on mars"] * self.args.batch_size
            videos = torch.randn(self.args.batch_size, 3, 16, 256, 256).to(self.device)
            # --- 模拟结束 ---
            
            with torch.no_grad():
                # VAE编码和文本编码
                latents = self.vae.encode(rearrange(videos, "b c t h w -> (b t) c h w")).latent_dist.sample()
                latents = rearrange(latents, "(b t) c h w -> b c t h w", t=16)
                latents = latents * self.vae.config.scaling_factor
                
                prompt_embeds = self._get_text_embeddings(prompts)
                uncond_embeds = self._get_text_embeddings(prompts, is_uncond=True)
            
            training_batch = {
                "latents": latents.to(self.device),
                "prompt_embeds": prompt_embeds.to(self.device),
                "uncond_embeds": uncond_embeds.to(self.device),
            }

            # 1. 训练批评家
            critic_loss = self.train_critic_step(training_batch)

            # 2. 按指定频率训练学生
            generator_loss = 0.0
            if (step + 1) % self.args.generator_update_interval == 0:
                generator_loss = self.train_generator_step(training_batch)

            progress_bar.update(1)
            progress_bar.set_postfix({
                "批评家损失": f"{critic_loss:.4f}",
                "学生损失": f"{generator_loss:.4f}",
            })
            
            # --- 此处添加日志记录 (wandb) 和模型保存逻辑 ---
            if (step + 1) % self.args.checkpointing_steps == 0:
                print(f"\n步骤 {step+1}: 保存模型检查点...")
                # ... 保存 self.student_dit, self.critic_dit, 和优化器状态 ...
            
            if (step + 1) % self.args.validation_steps == 0:
                print(f"\n步骤 {step+1}: 运行验证...")
                # ... 使用 self.student_dit 生成视频并记录 ...

if __name__ == '__main__':
    # 1. 初始化配置
    args = DistillHunyuanArgs()
    accelerator = None
    if args.FSDP: 
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with=args.log_with,
            project_dir=args.output_dir,
        )
    # 2. 初始化蒸馏管线
    pipeline = HunyuanDistillationPipeline(args, accelerator)
    
    # 3. 设置模型和优化器
    #pipeline.normal_train()
    pipeline.dmd_distill()
