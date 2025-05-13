import pandas as pd
import os

def load_sentiment140(filepath):
    """
    Load the Sentiment140 dataset from a CSV file.
    
    Parameters:
        file_path (str): Path to the Sentiment140 CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing the Sentiment140 data
    """
    # Check if file exists
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Cannot locate dataset at: {filepath}")
    
    # Define column names for Sentiment140 dataset
    cols = ['sentiment', 'id', 'timestamp', 'query_flag', 'username', 'content']
    
    # Load data using pandas
    print(f"Reading Twitter sentiment dataset from {filepath}")
    try:
        sentiment_df = pd.read_csv(filepath, encoding='utf-8', names=cols)
    except UnicodeDecodeError:
        # If UTF-8 fails, try Latin-1
        print("Falling back to latin-1 encoding")
        sentiment_df = pd.read_csv(filepath, encoding='latin-1', names=cols)
    
    # Convert sentiment values from 0/4 to 0/1
    sentiment_df['sentiment'] = sentiment_df['sentiment'].replace({0: 0, 4: 1})
    
    print(f"Successfully processed {len(sentiment_df)} tweets")
    return sentiment_df
