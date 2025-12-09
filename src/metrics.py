import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "mae": mean_absolute_error(labels, preds)
    }

def calculate_l_score(mae):
    """
    Custom metric for project.
    L(y_hat, y) = 0.5 * (2 - MAE)
    """
    return 0.5 * (2 - mae)