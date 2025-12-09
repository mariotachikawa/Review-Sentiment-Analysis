import argparse
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    Trainer, TrainingArguments, DataCollatorWithPadding, 
    EarlyStoppingCallback
)
import sys
import os

# Allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import set_seed
from src.metrics import compute_metrics
from src.data import preprocess_text
from src.augmentation import augment_sentence

def main():
    parser = argparse.ArgumentParser(description="Train Sentiment Analysis Model")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--data_path", type=str, default="data/processed/training_split.csv")
    parser.add_argument("--val_path", type=str, default="data/processed/validation_split.csv")
    parser.add_argument("--output_dir", type=str, default="models/deberta_v3")
    parser.add_argument("--use_eda", action="store_true", help="Apply EDA augmentation")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    
    # 1. Load Data
    print(f"Loading data from {args.data_path}...")
    train_df = pd.read_csv(args.data_path)
    val_df = pd.read_csv(args.val_path)
    
    # Preprocessing
    train_df["sentence"] = train_df["sentence"].apply(preprocess_text)
    val_df["sentence"] = val_df["sentence"].apply(preprocess_text)

    # Label Mapping
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    id2label = {v: k for k, v in label2id.items()}
    train_df["label"] = train_df["label"].map(label2id)
    val_df["label"] = val_df["label"].map(label2id)

    # 2. Augmentation (Optional)
    if args.use_eda:
        print("Applying EDA Augmentation...")
        aug_sentences, aug_labels = [], []
        for sent, lbl in zip(train_df["sentence"], train_df["label"]):
            # Generate 4 augmented versions per sentence
            augmented = augment_sentence(sent, alpha=0.1, n_aug=4)
            aug_sentences.extend(augmented)
            aug_labels.extend([lbl] * len(augmented))
        
        aug_df = pd.DataFrame({"sentence": aug_sentences, "label": aug_labels})
        train_df = pd.concat([train_df, aug_df], ignore_index=True)
        print(f"Training set size after augmentation: {len(train_df)}")

    # 3. Tokenization
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(tokenize, batched=True)
    
    train_ds = train_ds.remove_columns(["sentence"])
    val_ds = val_ds.remove_columns(["sentence"])

    # 4. Model Setup
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=3, id2label=id2label, label2id=label2id
    )

    # 5. Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.02,
        warmup_ratio=0.1,
        fp16=False, 
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(f"{args.output_dir}/best_model")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()