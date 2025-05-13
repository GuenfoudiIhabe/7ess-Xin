import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import random


def refine_sentiment_dataset(df, subset_limit=None):
    if subset_limit:
        positives = df[df['sentiment'] == 1]
        negatives = df[df['sentiment'] == 0]
        
        n_pos = min(subset_limit // 2, len(positives))
        n_neg = min(subset_limit // 2, len(negatives))
        
        pos_subset = positives.sample(n=n_pos, random_state=31415)
        neg_subset = negatives.sample(n=n_neg, random_state=31415)
        
        combined = pd.concat([pos_subset, neg_subset])
        result = combined.sample(frac=1).reset_index(drop=True)
    else:
        result = df
    
    print(f"Text normalization in progress...")
    result['normalized_text'] = result['content'].apply(normalize_tweet_text)
    
    return result


def normalize_tweet_text(raw_text):
    lowercased = raw_text.lower()
    
    url_pattern = r'http\S+|www\S+|https\S+'
    no_urls = re.sub(url_pattern, '', lowercased)
    
    handle_pattern = r'@[^\s]+'
    no_handles = re.sub(handle_pattern, '', no_urls)
    
    hashtag_pattern = r'#(\S+)'
    extracted_hashtags = re.sub(hashtag_pattern, r'\1', no_handles)
    
    symbols_pattern = r'[^\w\s]'
    no_symbols = re.sub(symbols_pattern, '', extracted_hashtags)
    
    digits_pattern = r'\d+'
    no_digits = re.sub(digits_pattern, '', no_symbols)
    
    whitespace_pattern = r'\s+'
    normalized = re.sub(whitespace_pattern, ' ', no_digits).strip()
    
    return normalized


def build_subword_tokenizer(corpus, vocabulary_size=30000, output_path=None):
    print(f"Building subword tokenizer (vocabulary: {vocabulary_size})...")
    
    byte_level_bpe = Tokenizer(models.BPE())
    byte_level_bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    byte_level_bpe.decoder = decoders.ByteLevel()
    
    bpe_config = trainers.BpeTrainer(
        vocab_size=vocabulary_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        show_progress=True
    )
    
    byte_level_bpe.train_from_iterator(corpus, bpe_config)
    
    if output_path:
        byte_level_bpe.save(output_path)
    
    token_map = {t: i for i, t in enumerate(byte_level_bpe.get_vocab())}
    id_to_token = {i: t for t, i in token_map.items()}
    
    byte_level_bpe.token_map = token_map
    byte_level_bpe.id_to_token = id_to_token
    
    return byte_level_bpe


def create_dataset_splits(dataframe, tokenizer, sequence_length=128, test_ratio=0.1, val_ratio=0.1, seed=42):
    print("Creating data partitions...")
    
    remaining, testing = train_test_split(
        dataframe, 
        test_size=test_ratio, 
        random_state=seed, 
        stratify=dataframe['sentiment']
    )
    
    validation_size = val_ratio / (1 - test_ratio)
    training, validation = train_test_split(
        remaining, 
        test_size=validation_size, 
        random_state=seed, 
        stratify=remaining['sentiment']
    )
    
    train_content = training['normalized_text'].tolist()
    train_sentiment = training['sentiment'].tolist()
    
    val_content = validation['normalized_text'].tolist()
    val_sentiment = validation['sentiment'].tolist()
    
    test_content = testing['normalized_text'].tolist()
    test_sentiment = testing['sentiment'].tolist()
    
    dataset_partitions = {
        'training': (train_content, train_sentiment),
        'validation': (val_content, val_sentiment),
        'testing': (test_content, test_sentiment),
    }
    
    return dataset_partitions
