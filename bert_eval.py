import torch
from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import cosine_similarity
import unicodedata

# 1. Normalization (Required for BERT Uncased)
def normalize(text):
    text = text.lower()
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                  if unicodedata.category(c) != 'Mn')

# 2. Paths
base_id = "nlpaueb/bert-base-greek-uncased-v1"
tuned_path = "./greekbert-cypriot-final"

# 3. Helper to get sentence embedding
def get_sentence_embed(text, model, tokenizer):
    inputs = tokenizer(normalize(text), return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling of the last hidden states
    return outputs.last_hidden_state.mean(dim=1)

# Load Models
tokenizer = AutoTokenizer.from_pretrained(base_id)
model_base = AutoModel.from_pretrained(base_id)
model_tuned = AutoModel.from_pretrained(tuned_path)

# 4. The Test Pair
# Tί κάνεις / ίντα κάμεις
# vocabulary shift
# Στην Κύπρο βρίσκομαι με φίλους
# Στην Κύπρο βρέθουμαι με φίλους
# lexical shift
# Το παιδί παίζει στον δρόμο
# Το κοπελλούδι παίζει στον δρόμο
# morphological shift 
# θέλουμε να πάμε για φαγητό
# θέλομεν να πάμεν για φαητόν
# syntactic shift
# του είπα την αλήθεια
# είπα του την αλήθειαν
standard = "Στην Κύπρο βρίσκομαι με φίλους"
cypriot  = "Στην Κύπρο βρέθουμαι με φίλους" # Note: Often dialects share words but change context

# Calculate Embeddings
emb_base_std = get_sentence_embed(standard, model_base, tokenizer)
emb_base_cyp = get_sentence_embed(cypriot, model_base, tokenizer)

emb_tuned_std = get_sentence_embed(standard, model_tuned, tokenizer)
emb_tuned_cyp = get_sentence_embed(cypriot, model_tuned, tokenizer)

# 5. Compare Similarity
sim_base = cosine_similarity(emb_base_std, emb_base_cyp).item()
sim_tuned = cosine_similarity(emb_tuned_std, emb_tuned_cyp).item()

print(f"Original Similarity: {sim_base:.4f}")
print(f"Fine-tuned Similarity: {sim_tuned:.4f}")

