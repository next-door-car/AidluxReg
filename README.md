# WJ-AI
# 分布式训练 launch
python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 main.py --launcher pytorch 