import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import os
import time
from tqdm import tqdm

from data.data_loader import load_sentiment140
from data.preprocessing import refine_sentiment_dataset, build_subword_tokenizer, create_dataset_splits
from models.transformer import EmotionAnalysisModel 
from utils.metrics import calculate_metrics


def configure_training():
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("--data_path", type=str, required=True)
    cmd_parser.add_argument("--output_dir", type=str, default="model_output")
    cmd_parser.add_argument("--vocab_size", type=int, default=30000)
    cmd_parser.add_argument("--emb_dim", type=int, default=256)
    cmd_parser.add_argument("--stack_depth", type=int, default=6)
    cmd_parser.add_argument("--attn_heads", type=int, default=8)
    cmd_parser.add_argument("--ff_expansion", type=int, default=2)
    cmd_parser.add_argument("--max_len", type=int, default=128)
    cmd_parser.add_argument("--dropout", type=float, default=0.1)
    cmd_parser.add_argument("--batch_size", type=int, default=64)
    cmd_parser.add_argument("--learning_rate", type=float, default=5e-4)
    cmd_parser.add_argument("--epochs", type=int, default=10)
    cmd_parser.add_argument("--subset_size", type=int, default=None)
    cmd_parser.add_argument("--random_seed", type=int, default=42)
    return cmd_parser.parse_args()


def train_sentiment_model(model, train_texts, train_labels, val_texts, val_labels, 
                          tokenizer, config, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    metrics_log = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    n_train = len(train_texts)
    n_train_batches = (n_train + config.batch_size - 1) // config.batch_size
    
    n_val = len(val_texts)
    n_val_batches = (n_val + config.batch_size - 1) // config.batch_size
    
    best_f1 = 0.0
    
    for epoch in range(config.epochs):
        print(f"\nTraining epoch {epoch+1}/{config.epochs}")
        epoch_start = time.time()
        
        model.train()
        train_epoch_loss = 0.0
        train_predictions = []
        train_ground_truth = []
        
        shuffled_indices = np.random.permutation(n_train)
        
        progress_bar = tqdm(range(n_train_batches))
        for batch_idx in progress_bar:
            batch_start = batch_idx * config.batch_size
            batch_end = min((batch_idx + 1) * config.batch_size, n_train)
            sample_indices = shuffled_indices[batch_start:batch_end]
            
            batch_texts = [train_texts[i] for i in sample_indices]
            batch_sentiments = [train_labels[i] for i in sample_indices]
            
            batch_encodings = []
            for text in batch_texts:
                encoding = tokenizer.encode(text)
                ids = encoding.ids
                
                if len(ids) > config.max_len:
                    ids = ids[:config.max_len]
                else:
                    ids = ids + [0] * (config.max_len - len(ids))
                
                batch_encodings.append(ids)
            
            input_tensor = torch.tensor(batch_encodings, dtype=torch.long).to(device)
            label_tensor = torch.tensor(batch_sentiments, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            predictions = model(input_tensor)
            batch_loss = loss_fn(predictions, label_tensor)
            
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_epoch_loss += batch_loss.item()
            _, predicted_classes = torch.max(predictions.data, 1)
            train_predictions.extend(predicted_classes.cpu().numpy())
            train_ground_truth.extend(label_tensor.cpu().numpy())
            
            progress_bar.set_postfix({'loss': batch_loss.item()})
        
        mean_train_loss = train_epoch_loss / n_train_batches
        train_scores = calculate_metrics(np.array(train_ground_truth), np.array(train_predictions))
        
        model.eval()
        val_epoch_loss = 0.0
        val_predictions = []
        val_ground_truth = []
        
        with torch.no_grad():
            val_progress = tqdm(range(n_val_batches))
            for batch_idx in val_progress:
                batch_start = batch_idx * config.batch_size
                batch_end = min((batch_idx + 1) * config.batch_size, n_val)
                
                batch_texts = val_texts[batch_start:batch_end]
                batch_sentiments = val_labels[batch_start:batch_end]
                
                batch_encodings = []
                for text in batch_texts:
                    encoding = tokenizer.encode(text)
                    ids = encoding.ids
                    
                    if len(ids) > config.max_len:
                        ids = ids[:config.max_len]
                    else:
                        ids = ids + [0] * (config.max_len - len(ids))
                    
                    batch_encodings.append(ids)
                
                input_tensor = torch.tensor(batch_encodings, dtype=torch.long).to(device)
                label_tensor = torch.tensor(batch_sentiments, dtype=torch.long).to(device)
                
                predictions = model(input_tensor)
                batch_loss = loss_fn(predictions, label_tensor)
                
                val_epoch_loss += batch_loss.item()
                _, predicted_classes = torch.max(predictions.data, 1)
                val_predictions.extend(predicted_classes.cpu().numpy())
                val_ground_truth.extend(label_tensor.cpu().numpy())
                
                val_progress.set_postfix({'loss': batch_loss.item()})
        
        mean_val_loss = val_epoch_loss / n_val_batches
        val_scores = calculate_metrics(np.array(val_ground_truth), np.array(val_predictions))
        
        metrics_log['train_loss'].append(mean_train_loss)
        metrics_log['train_acc'].append(train_scores['accuracy'])
        metrics_log['train_f1'].append(train_scores['f1'])
        metrics_log['val_loss'].append(mean_val_loss)
        metrics_log['val_acc'].append(val_scores['accuracy'])
        metrics_log['val_f1'].append(val_scores['f1'])
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{config.epochs} - {epoch_time:.2f}s - "
              f"Train: loss={mean_train_loss:.4f}, acc={train_scores['accuracy']:.4f} - "
              f"Val: loss={mean_val_loss:.4f}, acc={val_scores['accuracy']:.4f}, f1={val_scores['f1']:.4f}")
        
        if val_scores['f1'] > best_f1:
            best_f1 = val_scores['f1']
            
            checkpoint_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_f1': best_f1,
                'config': vars(config)
            }, checkpoint_path)
            
            print(f"New best model saved, F1={best_f1:.4f}")
    
    tokenizer_path = os.path.join(save_dir, "tokenizer.pkl")
    tokenizer.save(tokenizer_path)
    
    return metrics_log


def evaluate_test_set(model, test_texts, test_labels, tokenizer, config, device):
    print("\nRunning test set evaluation...")
    model.eval()
    all_predictions = []
    
    n_test = len(test_texts)
    n_test_batches = (n_test + config.batch_size - 1) // config.batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(n_test_batches)):
            batch_start = batch_idx * config.batch_size
            batch_end = min((batch_idx + 1) * config.batch_size, n_test)
            
            batch_texts = test_texts[batch_start:batch_end]
            
            batch_encodings = []
            for text in batch_texts:
                encoding = tokenizer.encode(text)
                ids = encoding.ids
                
                if len(ids) > config.max_len:
                    ids = ids[:config.max_len]
                else:
                    ids = ids + [0] * (config.max_len - len(ids))
                
                batch_encodings.append(ids)
            
            input_tensor = torch.tensor(batch_encodings, dtype=torch.long).to(device)
            
            predictions = model(input_tensor)
            _, predicted_classes = torch.max(predictions.data, 1)
            all_predictions.extend(predicted_classes.cpu().numpy())
    
    results = calculate_metrics(np.array(test_labels), np.array(all_predictions))
    
    print("\nTest Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    
    return results


def run_pipeline():
    config = configure_training()
    
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for computation")
    
    print("\nLoading dataset...")
    raw_data = load_sentiment140(config.data_path)
    processed_data = refine_sentiment_dataset(raw_data, config.subset_size)
    print(f"Dataset ready: {len(processed_data)} entries")
    
    print("\nBuilding tokenizer...")
    tokenizer = build_subword_tokenizer(processed_data['normalized_text'].tolist(), config.vocab_size)
    print(f"Tokenizer ready, vocabulary size: {len(tokenizer.token_map)}")
    
    print("\nSplitting dataset...")
    dataset = create_dataset_splits(processed_data, tokenizer, config.max_len)
    train_texts, train_labels = dataset['training']
    val_texts, val_labels = dataset['validation']
    test_texts, test_labels = dataset['testing']
    print(f"Training: {len(train_texts)} examples")
    print(f"Validation: {len(val_texts)} examples")
    print(f"Testing: {len(test_texts)} examples")
    
    print("\nInitializing model...")
    model = EmotionAnalysisModel(
        vocab_size=len(tokenizer.token_map),
        emb_dim=config.emb_dim,
        stack_depth=config.stack_depth,
        attn_heads=config.attn_heads,
        ff_expansion=config.ff_expansion // 2,  # Convert to expansion factor
        max_len=config.max_len,
        dropout=config.dropout
    ).to(device)
    
    print(f"Model architecture:")
    print(f"- Vocabulary size: {len(tokenizer.token_map)}")
    print(f"- Embedding dimension: {config.emb_dim}")
    print(f"- Encoder layers: {config.stack_depth}")
    print(f"- Attention heads: {config.attn_heads}")
    print(f"- Feedforward dimension: {config.emb_dim * config.ff_expansion//2}")
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {param_count:,}")
    
    print("\nStarting training...")
    train_sentiment_model(
        model=model,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        tokenizer=tokenizer,
        config=config,
        device=device,
        save_dir=config.output_dir
    )
    
    print("\nLoading best model...")
    checkpoint = torch.load(os.path.join(config.output_dir, "best_model.pt"))
    model.load_state_dict(checkpoint['model'])
    
    test_results = evaluate_test_set(
        model=model,
        test_texts=test_texts,
        test_labels=test_labels,
        tokenizer=tokenizer,
        config=config,
        device=device
    )
    
    print(f"\nTraining complete, model saved to: {config.output_dir}")


if __name__ == "__main__":
    run_pipeline()
