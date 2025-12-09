import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y_true, y_pred, labels, output_path=None):
    """
    Generates and saves a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="magma",
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Confusion matrix saved to {output_path}")
    
    return fig