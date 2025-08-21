#!/bin/bash
#BATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive

hostname
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "################################################"
echo "INFO: Master Address: $MASTER_ADDR"
echo "INFO: Master Port:    $MASTER_PORT"
echo "################################################"

source /starmap/nas/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-transfer1-126
cd /starmap/nas/workspace/yzy/code/cosmos-transfer1
srun python -u -m cosmos_transfer1.diffusion.training.train --config=cosmos_transfer1/diffusion/config/config_train.py -- experiment=CTRL_7Bv1pt3_t2w_sv2mv_57frames_control_input_hdmap_block3_posttrain
