from dataclasses import dataclass, field
import torch
import PIL.Image
from transformers import AutoTokenizer
from typing import Any, List, Union

@dataclass
class ForwardBatch:
    """
    Complete state passed through the pipeline execution.
    
    This dataclass contains all information needed during the diffusion pipeline
    execution, allowing methods to update specific components without needing
    to manage numerous individual parameters.
    """
    # TODO(will): double check that args are separate from fastvideo_args
    # properly. Also maybe think about providing an abstraction for pipeline
    # specific arguments.
    data_type: str

    generator: torch.Generator | list[torch.Generator] | None = None

    # Image inputs
    image_path: str | None = None
    image_embeds: list[torch.Tensor] = field(default_factory=list)
    pil_image: PIL.Image.Image | None = None
    preprocessed_image: torch.Tensor | None = None

    # Text inputs
    prompt: str | list[str] | None = None
    negative_prompt: str | list[str] | None = None
    prompt_path: str | None = None
    output_path: str = "outputs/"
    output_video_name: str | None = None
    # Primary encoder embeddings
    prompt_embeds: list[torch.Tensor] = field(default_factory=list)
    negative_prompt_embeds: list[torch.Tensor] | None = None
    prompt_attention_mask: list[torch.Tensor] | None = None
    pooled_prompt_embeds: list[torch.Tensor] = field(default_factory=list)
    negative_attention_mask: list[torch.Tensor] | None = None
    clip_embedding_pos: list[torch.Tensor] | None = None
    clip_embedding_neg: list[torch.Tensor] | None = None

    # Latent tensors
    latents: torch.Tensor | None = None
    raw_latent_shape: torch.Tensor | None = None
    noise_pred: torch.Tensor | None = None
    image_latent: torch.Tensor | None = None

    # Latent dimensions
    height_latents: int | None = None
    width_latents: int | None = None
    num_frames: int = 1  # Default for image models
    num_frames_round_down: bool = False  # Whether to round down num_frames if it's not divisible by num_gpus

    # Original dimensions (before VAE scaling)
    height: int | None = None
    width: int | None = None
    fps: int | None = None

    # Final output (after pipeline completion)
    output: Any = None
    guidance_scale: float = 1.0
    do_classifier_free_guidance: bool = False

    def __post_init__(self):
        """Initialize dependent fields after dataclass initialization."""

        # Set do_classifier_free_guidance based on guidance scale and negative prompt
        if self.guidance_scale > 1.0:
            self.do_classifier_free_guidance = True
        if self.negative_prompt_embeds is None:
            self.negative_prompt_embeds = []


@dataclass
class PreprocessHunyuanArgs:
    # 模型与数据
    model_path: str = "hunyuanvideo-community/HunyuanVideo"
    data_merge_path: str = "/u/jcai2/video/MyDistillation/preprocess/folder_anno_pair.txt"
    output_dir: str = None
    model_type: str = "hunyuan"
    cache_dir: str = "./cache_dir"
    seed : int = 42
    samples_per_file: int = 1
    # 视频预处理参数
    preprocess_video_batch_size: int = 1
    max_height: int = 480
    max_width: int = 832
    num_frames: int = 81
    num_latent_t: int = 28
    video_length_tolerance_range: float = 5.0
    train_fps: int = 16
    use_image_num: int = 0
    group_frame: bool = False
    group_resolution: bool = False
    preprocess_task: str = "t2v"
    text_max_length: int = 256
    speed_factor: float = 1.0
    drop_short_ratio: float = 1.0
    do_temporal_sample: bool = False

    # 训练相关
    text_encoder_name: str = "google/t5-v1_1-xxl"
    training_cfg_rate: float = 0.0
    dataloader_num_workers: int = 0

    # 额外参数
    flush_frequency: int = 1