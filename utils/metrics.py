import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_classification_performance(true_labels, predicted_labels):
    """
    Comprehensive evaluation of classification performance metrics.
    
    Automatically adapts to binary or multi-class classification scenarios by
    examining the unique classes present in the labels. Applies appropriate
    averaging methods based on the classification type.
    
    Parameters:
        true_labels (array-like): Ground truth class assignments
        predicted_labels (array-like): Model-predicted class assignments
        
    Returns:
        dict: Performance metrics including accuracy, precision, recall, and F1 score
    """
    # Determine classification type by counting unique classes across both arrays
    all_labels = np.concatenate([true_labels, predicted_labels])
    distinct_classes = np.unique(all_labels)
    
    # Select appropriate averaging strategy
    is_binary_classification = len(distinct_classes) <= 2
    averaging_method = 'binary' if is_binary_classification else 'weighted'
    
    # Log information for transparency
    print(f"Performance evaluation: detected {len(distinct_classes)} classes")
    print(f"Using '{averaging_method}' averaging for metric calculations")
    
    # Calculate comprehensive metrics
    performance_metrics = {
        'accuracy': accuracy_score(true_labels, predicted_labels),
        'precision': precision_score(
            true_labels, predicted_labels, 
            average=averaging_method, zero_division=0
        ),
        'recall': recall_score(
            true_labels, predicted_labels, 
            average=averaging_method, zero_division=0
        ),
        'f1': f1_score(
            true_labels, predicted_labels, 
            average=averaging_method, zero_division=0
        )
    }
    
    return performance_metrics

# Alias for backwards compatibility with existing code
calculate_metrics = evaluate_classification_performance
