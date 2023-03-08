#on ip1 the master node
python3 -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=2 --node_rank=0 \
--master_addr=104.171.200.62 --master_port=1234 \
main.py \
--backend=nccl --use_syn --batch_size=8192 --arch=resnet152

# On ip2 the worker node
python3 -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=2 --node_rank=1 \
--master_addr=104.171.200.62 --master_port=1234 \
main.py \
--backend=nccl --use_syn --batch_size=8192 --arch=resnet152
# torch1.10+ 版本选择使用torchrun
#master node
torchrun \
--nproc_per_node=2 --nnodes=2 --node_rank=0 \
--master_addr=104.171.200.62 --master_port=1234 \
main.py \
--backend=nccl --use_syn --batch_size=8192 --arch=resnet152

# work node
torchrun \
--nproc_per_node=2 --nnodes=2 --node_rank=1 \
--master_addr=104.171.200.62 --master_port=1234 \
main.py \
--backend=nccl --use_syn --batch_size=8192 --arch=resnet152