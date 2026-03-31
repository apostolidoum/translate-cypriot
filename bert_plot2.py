import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer
import json
import unicodedata

# 1. Configuration & Normalization
def normalize(text):
    text = text.lower()
    return ''.join(c for c in unicodedata.normalize('NFD', text) 
                  if unicodedata.category(c) != 'Mn')

base_id = "nlpaueb/bert-base-greek-uncased-v1"
tuned_path = "./greekbert-cypriot-final" 
tokenizer = AutoTokenizer.from_pretrained(base_id)

# 2. Load Data - Increase to 30 pairs to see a real distribution
with open('train_ds.json', 'r', encoding='utf-8') as f:
    full_data = json.load(f)
    data = full_data

# get mean pooling
def mean_pooling(model_output, attention_mask):
    # Extract token embeddings
    token_embeddings = model_output[0] 
    # Expand the mask to match embedding dimensions
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Sum embeddings while ignoring padding, then divide by the number of non-pad tokens
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# get embeddings for the cypriot model
model = AutoModel.from_pretrained(tuned_path)
model.eval()
cyp_list = []

for item in data:
    input_cyp = tokenizer(normalize(item["output"]), return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        cup_out = model(**input_cyp)
    cup_out = mean_pooling(cup_out, input_cyp["attention_mask"]).squeeze()    
    cyp_list.append(cup_out)
cyp_embeddings = np.array(cyp_list)

pca_results = PCA(n_components=2).fit_transform(cyp_embeddings)

# Plotting the results
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.7, edgecolors='k')
ax.set_title('PCA of Cypriot Greek Embeddings', fontweight='bold')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.grid(True, linestyle=':', alpha=0.6)
plt.show()
