# 7ess-Xin: Sentiment Analysis with Transformers

## 📊 Overview

7ess-Xin (pronounced "hess-shin," meaning "heart sentiments") is a transformer-based sentiment analysis system trained on the Sentiment140 dataset. It classifies Twitter text as positive or negative using a custom transformer architecture with rotary positional encoding.

## 🔍 Key Features

- Custom transformer architecture with rotary positional encoding
- Byte-pair encoding tokenization for Twitter text
- Attention visualization tools
- Query-key normalization for improved performance

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/7ess-Xin.git
cd 7ess-Xin
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Download the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) and place it in the root directory of the project.

## 🚀 Usage

### Training

```bash
python main.py --data_path "Sentiment140.csv" --output_dir "model_output/my_model" --emb_dim 128 --stack_depth 6 --attn_heads 8 --max_len 128 --batch_size 64 --lr 0.0001 --dropout 0.1 --ff_expansion 4 --epochs 10
```

#### Training Parameters

- `--data_path`: Path to the Sentiment140 dataset
- `--output_dir`: Directory to save model checkpoints
- `--emb_dim`: Embedding dimension size (default: 128)
- `--stack_depth`: Number of transformer layers (default: 6)
- `--attn_heads`: Number of attention heads (default: 8)
- `--max_len`: Maximum sequence length (default: 128)
- `--batch_size`: Training batch size (default: 64)
- `--lr`: Learning rate (default: 0.0001)
- `--dropout`: Dropout probability (default: 0.1)
- `--ff_expansion`: Feed-forward expansion factor (default: 4)
- `--epochs`: Number of training epochs (default: 10)

### Analyzing Tweets

```bash
python utils/tweet_analysis.py --sent "I love this product, it works great!"
```

## 📁 Repository Structure

```
7ess-Xin/
├── data/                           # Data directory
│   ├── data_loader.py              # Handles loading the Sentiment140 dataset
│   └── preprocessing.py            # Text preprocessing and dataset splitting
├── models/                         # Model implementations
│   ├── __init__.py                 # Package initialization
│   ├── transformer.py              # Current transformer implementation
│   └── transformerV0.py            # Original transformer implementation
├── model_output/                   # Trained model checkpoints
│   └── V6_optimized/               # Best model checkpoint 
├── utils/                          # Utility scripts
│   ├── metrics.py                  # Evaluation metrics
│   └── tweet_analysis.py           # Tweet analysis tool
├── qualitative_images/             # Generated visualizations
├── main.py                         # Main training script
├── requirements.txt                # Project dependencies
└── README.md                       # Documentation
```
