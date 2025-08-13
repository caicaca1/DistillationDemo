from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel
import torch
import numpy as np # 导入 NumPy 库

model_pipe = HunyuanVideoPipeline.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", 
            transformer=None,
            vae=None,
            text_encoder_2=None,
            tokenizer_2=None,
            torch_dtype=torch.bfloat16
        ).to("cuda")

text = "<PAD>"
num_hidden_layers_to_skip = 2
text_inputs = model_pipe.tokenizer(
    text, padding="max_length", max_length=10, return_tensors="pt"
)
text_input_ids = text_inputs.input_ids.to("cuda")
prompt_attention_mask = text_inputs.attention_mask.to("cuda")

prompt_embeds = model_pipe.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]
print("PAD embedding:", prompt_embeds[0,-3])
print("PAD embedding:", prompt_embeds[0,-2])