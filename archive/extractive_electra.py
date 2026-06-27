import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Using a Malay-specific ELECTRA model
MODEL_NAME = "mesolitica/electra-base-bahasa-standard"

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def get_embeddings(sentences, tokenizer, model):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the mean of the last hidden states as sentence representation
    return outputs.last_hidden_state.mean(dim=1).numpy()

def run_extractive(file_path="Cleaned_text/news.txt", top_n=3):
    print("--- Memulakan Extractive Summarization (ELECTRA) ---")
    
    sentences = read_file(file_path)
    if not sentences:
        return "Fail kosong."

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    # Generate embeddings for each sentence
    embeddings = get_embeddings(sentences, tokenizer, model)
    
    # Calculate similarity matrix (Centrality)
    sim_matrix = cosine_similarity(embeddings)
    scores = sim_matrix.sum(axis=1)
    
    # Rank sentences
    ranked_indices = np.argsort(scores)[::-1]
    
    # Select top N sentences in their original order
    top_indices = sorted(ranked_indices[:top_n])
    extracted_sentences = [sentences[i] for i in top_indices]
    
    return extracted_sentences

if __name__ == "__main__":
    result = run_extractive()
    print("\nSentences Extracted:")
    for sent in result:
        print(f"- {sent}")