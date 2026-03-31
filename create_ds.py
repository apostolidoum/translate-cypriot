import pandas as pd
import json

df = pd.read_excel("oral-sentences-200.xlsx", sheet_name=0)
print(df.columns)
print(f"Total rows: {len(df)}")

# Randomly shuffle the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define the split size
train_size = int(0.8 * len(df))

# Split the DataFrame
train_df = df[:train_size]
test_df = df[train_size:]

def create_ds(df):
    ds = []
    for _, row in df.iterrows():
        entry = {}
        entry["instruction"] = "Translate the sentence into cypriot greek."
        entry["input"] = row["SMG_sentence"]
        entry["output"] = row["CG_sentence_normalized"]
        ds.append(entry)
    return ds

train_ds = create_ds(train_df)
test_ds = create_ds(test_df)

# save to json
with open("train_ds.json", "w") as f:
    json.dump(train_ds, f, indent=4)
with open("test_ds.json", "w") as f:
    json.dump(test_ds, f, indent=4)