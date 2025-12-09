import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/raw/training.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    stratify=df["label"],
    random_state=42
)

train_df.to_csv("../data/processed/training_split.csv", index=False)
val_df.to_csv("../data/processed/validation_split.csv", index=False)
print("Train:", len(train_df), "  Val:", len(val_df))