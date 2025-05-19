#!/bin/bash
#SBATCH --job-name=Clean_Sentiment140
#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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

# Change to directory where clean_sentiment_data.py is located
cd ..

# Show current directory and confirm script exists
echo "Current working directory: $(pwd)"
if [ -f "clean_sentiment_data.py" ]; then
    echo "Found clean_sentiment_data.py in current directory"
else
    echo "ERROR: clean_sentiment_data.py not found in $(pwd)"
    exit 1
fi

# Make sure the required packages are installed
pip install -q transformers torch pandas tqdm

# Run the sentiment data cleaning script
echo "Starting sentiment dataset cleaning with GPU acceleration"
python -u clean_sentiment_data.py

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo "Dataset cleaning completed successfully at $(date)"
else
    echo "ERROR: Dataset cleaning failed with exit code $? at $(date)"
    exit 1
fi

echo "Job completed at $(date)"
