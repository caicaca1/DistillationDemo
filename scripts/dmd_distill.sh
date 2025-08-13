#!/bin/bash
export HF_HOME=/tmp/jcai2/.cache/huggingface
huggingface-cli login --token $HuggingfaceToken

accelerate launch --config_file ./accelerate_config/accelerate_config.yaml DistillPipeline.py \
        --dmd  \
        --FSDP \
        --enable_checkpointing

