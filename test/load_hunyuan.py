import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, AutoencoderKLHunyuanVideo

teacher_pipe = HunyuanVideoPipeline.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", 
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            vae=None,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

