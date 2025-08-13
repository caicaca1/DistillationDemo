import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="hunyuanvideo-community/HunyuanVideo")
    parser.add_argument("--data_merge_path", type=str, default="/u/jcai2/video/MyDistillation/preprocess/folder_anno_pair.txt")
    parser.add_argument("--output_dir", type=str, default="/work/hdd/bcjw/jcai2/dataset/mixkit-processed")
    parser.add_argument("--model_type", type=str, default="hunyuan")

    parser.add_argument("--preprocess_video_batch_size", type=int, default=1)
    parser.add_argument("--max_height", type=int, default=480)
    parser.add_argument("--max_width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_fps", type=int, default=16)
    parser.add_argument("--preprocess_task", type=str, default="t2v")

    return parser.parse_args()
