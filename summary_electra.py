import inspect
import scipy
import numpy as np

# Python 3.11+ compatibility patch for Malaya
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

# SciPy 1.13+ compatibility patch for Malaya (Restoring deleted NumPy aliases)
if not hasattr(scipy, 'asarray'):
    scipy.asarray = np.asarray
if not hasattr(scipy, 'ones'):
    scipy.ones = np.ones
if not hasattr(scipy, 'zeros'):
    scipy.zeros = np.zeros
if not hasattr(scipy, 'array'):
    scipy.array = np.array

import malaya
import os
import warnings

# Hide TensorFlow and HuggingFace warnings for cleaner terminal output
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

def run_hybrid_summarization_electra(file_path):
    # 1. Load the Cleaned Text from your preprocess.py output
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            cleaned_text = f.read()
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found. Please check your path.")
        return

    print("\n" + "="*50)
    print("STEP 1: EXTRACTIVE SUMMARIZATION (Malaya ELECTRA Encoder)")
    print("="*50)
    print("Loading Semantic Extractive Model (Understands context and meaning)...")
    
    # Setup Malaya Deep Encoder (ELECTRA) with SDPA fix
    transformer_model = malaya.transformer.huggingface(
        model='mesolitica/electra-small-generator-bahasa-cased',
        attn_implementation="eager"
    )
    extractive_model = malaya.summarization.extractive.encoder(transformer_model)
    
    # --- DYNAMIC EXTRACTION ---
    # Estimate total sentences by splitting by periods
    total_sentences = len([s for s in cleaned_text.split('.') if len(s.strip()) > 5])
    
    # Extract about 10% of the text, but cap it at 8 sentences so we don't overwhelm T5 later
    sentences_to_extract = min(8, max(3, int(total_sentences * 0.10)))
    print(f"Total valid sentences detected: ~{total_sentences}.")
    print(f"Extracting top {sentences_to_extract} contextually heavy sentences...\n")
    
    # Extract the summary dictionary
    summary_data = extractive_model.sentence_level(cleaned_text, top_k=sentences_to_extract)
    
    # Pull out just the string of combined sentences
    extractive_combined = summary_data['summary']

    # Print the extractive result to terminal
    print(extractive_combined)

    print("\n" + "="*50)
    print("STEP 2: ABSTRACTIVE SUMMARIZATION (Malaya T5-Base)")
    print("="*50)
    print("Rewriting for natural flow... (Testing repetition_penalty=2.0)")

    # Load Malaya's Abstractive Model 
    abstractive_model = malaya.summarization.abstractive.huggingface(
        model='mesolitica/finetune-summarization-t5-base-standard-bahasa-cased'
    )
    
    # --- EXPERIMENTAL PARAMS ---
    # We added repetition_penalty=2.0 to force the AI to paraphrase
    final_summary_list = abstractive_model.generate(
        [extractive_combined],
        max_length=256, 
        num_beams=4, 
        no_repeat_ngram_size=3,
        repetition_penalty=2.0, 
        early_stopping=True
    )
    final_summary = final_summary_list[0]

    print(f"\nFINAL RESULT:\n{final_summary}")

    # 3. SAVE THE FINAL RESULT
    if not os.path.exists("Summaries"):
        os.makedirs("Summaries")
        
    output_path = "Summaries/final_report_ELECTRA.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("--- EXTRACTIVE KEY POINTS (ELECTRA ENCODER) ---\n")
        f.write(extractive_combined)
        f.write("\n\n--- FINAL ABSTRACTIVE SUMMARY (T5 BASE - HIGH PENALTY) ---\n")
        f.write(final_summary)
    
    print(f"\n[System] Full report saved to: {output_path}")

# --- EXECUTION ---
cleaned_file_path = "Cleaned_Text/culture_shock_cleaned.txt"
run_hybrid_summarization_electra(cleaned_file_path)