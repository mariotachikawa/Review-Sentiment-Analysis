# Sentiment Analysis on Restaurant Reviews

A robust, production-ready sentiment analysis pipeline comparing classical machine learning baselines against state-of-the-art Transformer models. The final system utilizes **DeBERTa-v3**, **Easy Data Augmentation (EDA)**, and **Monte-Carlo Dropout** for uncertainty estimation, achieving a Mean Absolute Error (MAE) of **0.2114**.

## Key Features

  * **SOTA Architectures:** Comparison of DistilBERT, Twitter-RoBERTa, and DeBERTa-v3.
  * **Data Engineering:** Implementation of Easy Data Augmentation (Synonym Replacement, Random Insertion/Swap/Deletion) to handle class imbalance.
  * **Uncertainty Estimation:** Inference pipeline uses Monte-Carlo Dropout (training mode during inference) to stabilize predictions.
  * **Modular Design:** Refactored from experimental notebooks into a clean `src` package structure.
  * **Reproducibility:** seeded random number generators for consistent results across PyTorch, NumPy, and Python.

## Performance

We evaluated models on a custom "L-score" (penalizing severe misclassifications) and MAE.

| Model | L-Score (↑) | MAE (↓) |
| :--- | :--- | :--- |
| **Logistic Regression (BoW)** | 0.7800 | 0.4402 |
| **DistilBERT** | 0.8437 | 0.3127 |
| **Twitter-RoBERTa** | 0.8699 | 0.2593 |
| **DeBERTa-v3** | 0.8899 | 0.2203 |
| **DeBERTa-v3 + EDA (Best)** | **0.8943** | **0.2114** |

## Installation

1.  **Clone the repository:**

    ```bash
    git clone git@github.com:mariotachikawa/ReviewSentimentAnalysis.git
    cd ReviewSentimentAnalysis
    ```

2.  **Create a virtual environment (Recommended):**

    ```bash
    conda create -n sentiment-analysis python=3.10
    conda activate sentiment-analysis
    ```

3.  **Install dependencies:**

    ```bash
    # Install project requirements
    pip install -r requirements.txt
    ```

## Usage

### 1\. Prepare Data

Split the raw training data into stratified train/validation sets:

```bash
python scripts/split_training_set.py
```

### 2\. Train a Model

You can train different models by changing the `--model_name`. The script supports automatic data augmentation via the `--use_eda` flag.

```bash
# Fast test run (DistilBERT)
python scripts/train.py --model_name distilbert-base-uncased --epochs 3 --batch_size 16

# Full training (DeBERTa-v3 + Augmentation)
python scripts/train.py --model_name microsoft/deberta-v3-base --epochs 4 --batch_size 8 --use_eda --lr 2e-5
```

### 3\. Evaluation (MC Dropout)

Run the evaluation script to calculate L-Score and MAE. This script enables dropout during inference to average predictions over multiple forward passes.

```bash
python scripts/evaluate.py --model_path models/deberta_v3/best_model --mc_samples 5
```

### 4\. Generate Predictions

Generate a submission CSV for unlabelled test data:

```bash
python scripts/predict.py --model_path models/deberta_v3/best_model --input_file data/raw/test.csv
```

## Project Report

For a comprehensive technical analysis, please refer to the **[Project Report](https://github.com/mariotachikawa/ReviewSentimentAnalysis/blob/main/report/Project_Report.pdf)** included in this repository. This document provides an in-depth discussion of the theoretical background, experimental methodology, hyperparameter optimization strategies, and a detailed error analysis of the proposed architectures.
## Authors

  * **Mario Tachikawa**
  * **Edi Zeqiri**
  * **Besche Awdir**
  * **Hamza Zarah**

*Developed as part of the Computational Intelligence Lab (CIL) 2025 at ETH Zurich.*
