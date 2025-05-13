import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate common evaluation metrics for binary classification.
    
    Parameters:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        
    Returns:
        dict: Dictionary containing metrics (accuracy, precision, recall, f1)
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return metrics
