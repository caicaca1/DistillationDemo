# SPDX-License-Identifier: Apache-2.0
# schema.py
"""
Unified data schema and format for saving and loading image/video data after
preprocessing.

It uses apache arrow in-memory format that can be consumed by modern data
frameworks that can handle parquet or lance file.
"""

import pyarrow as pa

pyarrow_schema_t2v_hunyuan = pa.schema([
    pa.field("id", pa.string()),

    # --- Image/Video VAE latents ---
    pa.field("vae_latent_bytes", pa.binary()),
    pa.field("vae_latent_shape", pa.list_(pa.int64())),   # e.g., [C, T, H, W] or [C, H, W]
    pa.field("vae_latent_dtype", pa.string()),            # e.g., 'float32'

    # --- Text encoder output tensor ---
    pa.field("text_embedding_bytes", pa.binary()),
    pa.field("text_embedding_shape", pa.list_(pa.int64())),   # e.g., [SeqLen, Dim]
    pa.field("text_embedding_dtype", pa.string()),            # e.g., 'bfloat16' or 'float32'

    # --- Pooled text embedding (global) ---
    pa.field("pooled_text_embedding_bytes", pa.binary()),
    pa.field("pooled_text_embedding_shape", pa.list_(pa.int64())),
    pa.field("pooled_text_embedding_dtype", pa.string()),

    # --- Text attention mask ---
    pa.field("text_attention_mask_bytes", pa.binary()),
    pa.field("text_attention_mask_shape", pa.list_(pa.int64())),  # usually [SeqLen]
    pa.field("text_attention_mask_dtype", pa.string()),           # e.g., 'uint8' or 'int32'

    # --- Metadata ---
    pa.field("file_name", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("media_type", pa.string()),  # 'image' or 'video'
    pa.field("width", pa.int64()),
    pa.field("height", pa.int64()),

    # -- Video-specific (can be null/default for images) ---
    pa.field("num_frames", pa.int64()),     # number of frames processed
    pa.field("duration_sec", pa.float64()),
    pa.field("fps", pa.float64()),
])
