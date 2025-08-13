# SPDX-License-Identifier: Apache-2.0
"""
T2V Data Preprocessing pipeline implementation.

This module contains an implementation of the T2V Data Preprocessing pipeline
using the modular pipeline architecture.
"""
from PreprocessPipeline import BasePreprocessPipeline
from schema import pyarrow_schema_t2v_hunyuan
from parser import parse_args
from preprocess_args import PreprocessHunyuanArgs

class PreprocessPipelineHunyuan_T2V(BasePreprocessPipeline):
    """T2V preprocessing pipeline implementation."""

    _required_config_modules = ["text_encoder", "text_encoder_2","tokenizer", "tokenizer_2","vae"]

    def get_schema_fields(self):
        """Get the schema fields for T2V pipeline."""
        return [f.name for f in pyarrow_schema_t2v_hunyuan]


if __name__ == '__main__':
    # 1. 初始化配置
    cli_args = parse_args()
    args = PreprocessHunyuanArgs(**vars(cli_args))
    # 2. 初始化蒸馏管线
    pipeline = PreprocessPipelineHunyuan_T2V(args)
    pipeline.forward(args=args)
