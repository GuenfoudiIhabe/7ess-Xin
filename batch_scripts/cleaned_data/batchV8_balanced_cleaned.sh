#!/bin/bash
#SBATCH --job-name=SA_balanced_cleaned
#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
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

# Change to main project directory where main.py is located
# Need to go up two levels: from cleaned_data to batch_scripts, then to main directory
cd ../..

# Show current directory and confirm main.py exists
echo "Current working directory: $(pwd)"
if [ -f "main.py" ]; then
    echo "Found main.py in current directory"
else
    echo "ERROR: main.py not found in $(pwd)"
    exit 1
fi

# Check if cleaned data exists
if [ -f "cleaned_sentiment_data.csv" ]; then
    echo "Found cleaned sentiment data"
else
    echo "WARNING: cleaned_sentiment_data.csv not found, model will not train correctly!"
fi

# Run with WandB integration - Balanced width-depth architecture with cleaned data
python -u main.py \
  --data_path "${PWD}/cleaned_sentiment_data.csv" \
  --output_dir model_output/V8_balanced_cleaned \
  --vocab_size 35000 \
  --emb_dim 192 \
  --stack_depth 5 \
  --attn_heads 6 \
  --ff_expansion 3 \
  --max_len 148 \
  --dropout 0.1 \
  --batch_size 64 \
  --learning_rate 5e-4 \
  --epochs 15 \
  --random_seed 42 \
  --use_wandb \
  --wandb_project 7ess-Xin \
  --wandb_name V8_balanced_cleaned \
  --wandb_entity gihabe290-university-of-li-ge
