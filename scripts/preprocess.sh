export HF_HOME=/tmp/jcai2/.cache/huggingface
huggingface-cli login --token $HuggingfaceToken

python /u/jcai2/video/MyDistillation/preprocess/HunyuanPreprocessT2V.py
