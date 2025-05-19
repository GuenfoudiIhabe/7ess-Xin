import pandas as pd
import os
import logging

def import_twitter_sentiment_dataset(filepath):
    """
    Import and preprocess Twitter sentiment data from various source formats.
    
    Handles both original Sentiment140 data and RoBERTa-cleaned datasets, with
    appropriate processing for each. Ensures consistent label format and provides
    detailed diagnostics about the imported data.
    
    Parameters:
        filepath (str): Path to the sentiment dataset file
        
    Returns:
        pd.DataFrame: Processed dataframe ready for model training
    """
    # Verify file existence
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at location: {filepath}")
    
    # Determine dataset type and apply appropriate processing
    if 'cleaned_sentiment_data' in filepath:
        return _process_cleaned_sentiment_data(filepath)
    else:
        return _process_original_sentiment140(filepath)

def _process_cleaned_sentiment_data(filepath):
    """Helper function to process cleaned sentiment datasets"""
    print(f"Importing cleaned sentiment dataset: {filepath}")
    
    # Try different encoding options
    try:
        sentiment_data = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 encoding failed, attempting latin-1 encoding")
        sentiment_data = pd.read_csv(filepath, encoding='latin-1')
    
    # Log dataset structure
    print(f"Dataset columns: {sentiment_data.columns.tolist()}")
    
    # Verify and process sentiment values
    if 'sentiment' not in sentiment_data.columns:
        raise ValueError(f"Required 'sentiment' column missing from dataset")
    
    # Convert sentiment to numeric if needed
    if sentiment_data['sentiment'].dtype == 'object':
        print("Converting text sentiment values to numeric format")
        sentiment_data['sentiment'] = sentiment_data['sentiment'].apply(
            lambda x: _convert_sentiment_to_numeric(x)
        )
    
    # Ensure integer type for sentiment
    sentiment_data['sentiment'] = sentiment_data['sentiment'].astype(int)
    
    return _finalize_sentiment_dataset(sentiment_data)

def _process_original_sentiment140(filepath):
    """Helper function to process original Sentiment140 data format"""
    print(f"Importing original Sentiment140 dataset: {filepath}")
    
    # Define standard column structure
    column_names = [
        'sentiment', 'id', 'timestamp', 'query_flag', 'username', 'content'
    ]
    
    # Try different encoding options
    try:
        sentiment_data = pd.read_csv(
            filepath, encoding='utf-8', names=column_names
        )
    except UnicodeDecodeError:
        print("UTF-8 encoding failed, attempting latin-1 encoding")
        sentiment_data = pd.read_csv(
            filepath, encoding='latin-1', names=column_names
        )
    
    # Map original Sentiment140 values (0=negative, 4=positive) to binary format
    if 4 in sentiment_data['sentiment'].unique():
        print("Converting Sentiment140 labels from 0/4 to 0/1 format")
        sentiment_data['sentiment'] = sentiment_data['sentiment'].map({0: 0, 4: 1})
    
    return _finalize_sentiment_dataset(sentiment_data)

def _convert_sentiment_to_numeric(sentiment_value):
    """Convert various sentiment representations to numeric values"""
    if isinstance(sentiment_value, (int, float)):
        return int(sentiment_value)
    
    # Convert string representations
    sentiment_str = str(sentiment_value).lower()
    if sentiment_str in ['1', 'positive', 'pos', 'true', 'yes']:
        return 1
    else:
        return 0

def _finalize_sentiment_dataset(dataframe):
    """Final processing steps for all sentiment datasets"""
    # Dataset statistics
    sentiment_distribution = dataframe['sentiment'].value_counts()
    class_count = len(sentiment_distribution)
    
    print(f"Dataset loaded with {len(dataframe)} entries")
    print(f"Class distribution: {sentiment_distribution.to_dict()}")
    print(f"Number of sentiment classes: {class_count}")
    
    # Ensure label values are sequential integers starting from 0
    max_label = dataframe['sentiment'].max()
    if max_label >= class_count:
        print("Normalizing label range to sequential integers")
        label_mapping = {
            original: idx for idx, original in enumerate(
                sorted(dataframe['sentiment'].unique())
            )
        }
        dataframe['sentiment'] = dataframe['sentiment'].map(label_mapping)
        print(f"Labels remapped to range: 0-{class_count-1}")
    
    return dataframe

# For backwards compatibility
load_sentiment140 = import_twitter_sentiment_dataset
