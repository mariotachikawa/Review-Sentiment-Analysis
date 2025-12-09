import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import sys
import os

# Allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import preprocess_text

def main():
    parser = argparse.ArgumentParser(description="Generate predictions for Kaggle submission")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model folder")
    parser.add_argument("--input_file", type=str, default="data/raw/test.csv", help="Path to input CSV")
    parser.add_argument("--output_file", type=str, default="submission.csv", help="Path to save output CSV")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and Preprocess Data
    print(f"Loading data from {args.input_file}...")
    df = pd.read_csv(args.input_file)
    
    # Ensure text column exists (your test.csv uses 'sentence' or 'text')
    # This handles both cases just to be safe
    text_col = "sentence" if "sentence" in df.columns else "text"
    df["clean_text"] = df[text_col].apply(preprocess_text)

    # 2. Load Model & Tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    # 3. Prepare Dataset
    def tokenize(batch):
        return tokenizer(batch["clean_text"], truncation=True, max_length=128)

    ds = Dataset.from_pandas(df)
    ds = ds.map(tokenize, batched=True)
    # We don't need the text columns for inference, just IDs
    ds = ds.remove_columns([text_col, "clean_text"])
    # Keep 'id' if it exists for the submission file, but don't pass it to the model
    if "id" in ds.column_names:
        ids = ds["id"]
        ds = ds.remove_columns(["id"])
    else:
        ids = range(len(df))

    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=DataCollatorWithPadding(tokenizer))

    # 4. Inference Loop
    print("Running prediction...")
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 5. Map to Labels and Save
    # Define mapping (Ensure this matches your training!)
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label_names = [id2label[p] for p in all_preds]

    submission = pd.DataFrame({
        "id": ids,
        "label": label_names,
        # Optional: Save confidence score if you want to analyze it later
        # "confidence": [max(p) for p in all_probs] 
    })

    submission.to_csv(args.output_file, index=False)
    print(f"Submission saved: {args.output_file}")

if __name__ == "__main__":
    main()