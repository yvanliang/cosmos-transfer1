#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive

source /starmap/nas/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-transfer1-126
cd /starmap/nas/workspace/yzy/code/cosmos-transfer1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export NCCL_DEBUG=warn
export NCCL_ASYNC_ERROR_HANDLING=1

# Enforce block distribution and closest GPU binding to ensure all 8 ranks for TP=8 reside on a single node.
srun \
  --distribution=block:block \
  --gpu-bind=closest \
  --kill-on-bad-exit=1 \
  --export=ALL \
  bash -c 'export RANK=$SLURM_PROCID; export WORLD_SIZE=$SLURM_NTASKS; export LOCAL_RANK=$SLURM_LOCALID; \
  python -u -m cosmos_transfer1.diffusion.training.train \
    --config=cosmos_transfer1/diffusion/config/config_train.py \
    -- experiment=CTRL_7Bv1pt3_t2w_sv2mv_57frames_control_input_hdmap_block3_posttrain'