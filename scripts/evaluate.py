import argparse
import torch
import pandas as pd
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.metrics import calculate_l_score
from src.data import preprocess_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--data_path", type=str, default="data/processed/validation_split.csv")
    parser.add_argument("--mc_samples", type=int, default=3, help="Number of MC Dropout passes")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    df = pd.read_csv(args.data_path)
    df["sentence"] = df["sentence"].apply(preprocess_text)
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["label"].map(label2id)

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    
    # CRITICAL: Set to train mode for MC Dropout
    model.train() 

    # Prepare DataLoader
    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=128)
    
    ds = Dataset.from_pandas(df).map(tokenize, batched=True)
    ds = ds.remove_columns(["sentence"])
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    dl = DataLoader(ds, batch_size=8, collate_fn=DataCollatorWithPadding(tokenizer))

    print(f"Running Inference with {args.mc_samples} MC samples...")
    
    mc_probs = []
    for i in range(args.mc_samples):
        print(f" Pass {i+1}/{args.mc_samples}")
        probs = []
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                probs.append(torch.softmax(logits, dim=-1).cpu())
        mc_probs.append(torch.cat(probs))

    # Average predictions
    avg_probs = torch.stack(mc_probs).mean(dim=0).numpy()
    
    # Calculate metrics
    y_true = df["label"].values
    p_raw = avg_probs.dot(np.arange(3))
    mae = np.abs(p_raw - y_true).mean()
    l_score = calculate_l_score(mae)

    print("\nResults:")
    print(f"MAE: {mae:.4f}")
    print(f"L-Score: {l_score:.4f}")

    # Save results
    results = {"mae": float(mae), "l_score": float(l_score)}
    with open(f"{args.model_path}/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()