import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hunyuan_model_path", type=str, default="hunyuanvideo-community/HunyuanVideo")
    parser.add_argument("--output_dir", type=str, default="/work/hdd/bcjw/jcai2/hunyuan_distilled_output")

    parser.add_argument("--student_num_layers", type=int, default=20)
    parser.add_argument("--student_num_single_layers", type=int, default=40)
    parser.add_argument("--student_num_attention_heads", type=int, default=24)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--critic_learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_train_iterations", type=int, default=10000)
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--total_timesteps", type=int, default=1000)
    parser.add_argument("--train_log_loss_steps", type=int, default=100)
    parser.add_argument("--guidance_scale_min", type=float, default=1.0)
    parser.add_argument("--guidance_scale_max", type=float, default=10.0)
    parser.add_argument("--FSDP", action="store_true")
    parser.add_argument("--enable_checkpointing", action="store_true")

    parser.add_argument("--real_score_guidance_scale", type=float, default=7.5)
    parser.add_argument("--dmd", action="store_true")
    parser.add_argument("--log_with", type=str, default="wandb")
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)

    return parser.parse_args()
