#!/bin/bash
#SBATCH --job-name sam_nuclei_segmentation
#SBATCH --mail-type=BEGIN,END,FAIL            
#SBATCH --mail-user=elahe.ranjbari98@gmail.com
#SBATCH --cpus-per-task 2
#SBATCH --mem 32GB
#SBATCH --output /projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation/vis.out
#SBATCH --error /projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation/vis.out
#SBATCH -p gpuA6000,dgxV100,gpu3090,rtx5000
#SBATCH --gres=gpu:1
#SBATCH --exclude=dlhost02


cd /projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation
source /projects/ovcare/users/elahe_ranjbari/miniconda3/etc/profile.d/conda.sh
conda activate conda_env

data_dir=/projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation/NuInsSeg
prompt_type=bbx
num_pos_points=6
model_name=medsam
vis_dir=./plot/$model_name/$prompt_type

python zeroshot_sam.py \
    --data_dir $data_dir \
    --batch_size 1 \
    --prompt_type $prompt_type \
    --num_pos_points $num_pos_points \
    --vis_dir $vis_dir \
    --model_name $model_name \
    --model_weights Models/MedSAM/bbx/sam_best.pth