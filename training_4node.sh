#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive

source /starmap/nas/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-transfer1-126
cd /starmap/nas/workspace/yzy/code/cosmos-transfer1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=warn
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_LAUNCH_MODE=GROUP

# ==== IB/RoCE 网络优化（按集群网卡命名修改 IFNAME）====
export NCCL_IB_HCA=mlx5
export NCCL_CROSS_NIC=1
export NCCL_IB_GID_INDEX=0
export NCCL_IB_TC=106
export NCCL_IB_TIMEOUT=22

# ==== GDR (GPU Direct RDMA) 建议开启（硬件/驱动需支持）====
export NCCL_NET_GDR_LEVEL=2

# ==== 网络接口绑定（替换为实际高性能网卡名）====
# 多节点网卡名不一致时，可用逗号列出多个，NCCL 会在本机选择存在的那个
 export NCCL_SOCKET_IFNAME=ens8f0np0,enp195s0f0np0

# ==== Ring 建议设置（NCCL 会自适应，但可给出下限）====
export NCCL_MIN_NRINGS=4

# Enforce block distribution and closest GPU binding to ensure all 8 ranks for TP=8 reside on a single node.
srun \
  --distribution=block:block \
  --gpu-bind=closest \
  --kill-on-bad-exit=1 \
  --export=ALL \
  bash -c 'export RANK=$SLURM_PROCID; export WORLD_SIZE=$SLURM_NTASKS; export LOCAL_RANK=$SLURM_LOCALID; \
  python -u -m cosmos_transfer1.diffusion.training.train \
    --config=cosmos_transfer1/diffusion/config/config_train.py \
    -- experiment=CTRL_7Bv1pt3_t2w_57frames_control_input_hdmap_block3_posttrain'