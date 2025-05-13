#!/bin/bash
#SBATCH --job-name=SA_baselineV1
#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ihabe.guenfoudi@student.uliege.be

# Load your conda environment
source /home/gihabe/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Make sure the logs directory exists
mkdir -p logs

# Display information about error/output files
echo "Job started at $(date)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Output log: $(pwd)/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
echo "Error log: $(pwd)/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

# Change to parent directory where main.py is located
cd ..

# Show current directory and confirm main.py exists
echo "Current working directory: $(pwd)"
if [ -f "main.py" ]; then
    echo "Found main.py in current directory"
else
    echo "ERROR: main.py not found in $(pwd)"
    exit 1
fi

# Run with WandB integration - Baseline model
python -u main.py \
  --data_path "${PWD}/Sentiment140.csv" \
  --output_dir model_output/V1_baseline \
  --vocab_size 30000 \
  --emb_dim 128 \
  --stack_depth 3 \
  --attn_heads 4 \
  --ff_expansion 2 \
  --max_len 128 \
  --dropout 0.1 \
  --batch_size 64 \
  --learning_rate 5e-4 \
  --epochs 10 \
  --random_seed 42 \
  --use_wandb \
  --wandb_project 7ess-Xin \
  --wandb_name V1_baseline \
  --wandb_entity gihabe290-university-of-li-ge
