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

# 3. Enhanced Embedding Function
def get_embeddings(model_path, data_pairs):
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    std_list, cyp_list = [], []
    
    for item in data_pairs:
        # Standard Greek Embedding
        inputs_std = tokenizer(normalize(item['input']), return_tensors="pt", padding=True, truncation=True)
        # Cypriot Greek Embedding
        inputs_cyp = tokenizer(normalize(item['output']), return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            std_out = model(**inputs_std).last_hidden_state.mean(dim=1).squeeze().numpy()
            cyp_out = model(**inputs_cyp).last_hidden_state.mean(dim=1).squeeze().numpy()
            
        std_list.append(std_out)
        cyp_list.append(cyp_out)
    
    return np.array(std_list), np.array(cyp_list)

# 4. Process
print("Calculating embeddings for Base and Tuned models...")
std_base, cyp_base = get_embeddings(base_id, data)
std_tuned, cyp_tuned = get_embeddings(tuned_path, data)

# 5. Plotting Function
def plot_comparison(std_arr, cyp_arr, title, ax):
    # Combine to fit PCA on the whole local space
    all_points = np.vstack([std_arr, cyp_arr])
    pca_results = PCA(n_components=2).fit_transform(all_points)
    
    n = len(std_arr)
    std_pca = pca_results[:n]
    cyp_pca = pca_results[n:]
    
    # Add a tiny bit of jitter so overlapping dots are visible
    # jitter = 0.02
    # std_pca += np.random.uniform(-jitter, jitter, std_pca.shape)
    # cyp_pca += np.random.uniform(-jitter, jitter, cyp_pca.shape)

    # Plot lines first (so they stay behind dots)
    for i in range(n):
        ax.plot([std_pca[i, 0], cyp_pca[i, 0]], [std_pca[i, 1], cyp_pca[i, 1]], 
                color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Plot Dots
    ax.scatter(std_pca[:, 0], std_pca[:, 1], c='#1f77b4', label='Standard', s=50, edgecolors='k', alpha=0.8)
    ax.scatter(cyp_pca[:, 0], cyp_pca[:, 1], c='#ff7f0e', label='Cypriot', s=50, edgecolors='k', alpha=0.8)
    
    ax.set_title(title, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

# Execution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
plot_comparison(std_base, cyp_base, "Before Finetuning (Base GreekBERT)", ax1)
plot_comparison(std_tuned, cyp_tuned, "After Finetuning (Cypriot MLM Tuned)", ax2)

plt.suptitle("Semantic Alignment: Standard vs. Cypriot Greek Embeddings", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
