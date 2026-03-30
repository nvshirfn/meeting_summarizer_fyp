import inspect
# Python 3.11+ compatibility patch for Malaya
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

import malaya
import os
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Hide TensorFlow warnings for cleaner terminal output
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def run_hybrid_summarization_lsa(file_path):
    # 1. Load the Cleaned Text from your preprocess.py output
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            cleaned_text = f.read()
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found. Please check your path.")
        return

    print("\n" + "="*50)
    print("STEP 1: EXTRACTIVE SUMMARIZATION (Sumy LSA)")
    print("="*50)
    
    # Setup Sumy LSA
    # We use the "english" tokenizer as a base because it handles spacing/punctuation well for Malay
    parser = PlaintextParser.from_string(cleaned_text, Tokenizer("english"))
    extractive_model = LsaSummarizer()
    
    # --- DYNAMIC EXTRACTION ---
    # Count total sentences in the document
    total_sentences = len(parser.document.sentences)
    
    # Extract 20% of the text, but ensure we grab at least 3 sentences
    sentences_to_extract = max(3, int(total_sentences * 0.20))
    print(f"Total sentences detected: {total_sentences}.")
    print(f"Extracting top {sentences_to_extract} key sentences across different topics...\n")
    
    # Pull the dynamic amount of important sentences using LSA
    top_sentences_raw = extractive_model(parser.document, sentences_to_extract)
    
    # Convert Sumy output into normal Python strings
    top_sentences = [str(sentence) for sentence in top_sentences_raw]
    
    # Display the extractive result
    for i, sentence in enumerate(top_sentences, 1):
        print(f"{i}. {sentence}")

    # Combine these sentences to feed into the Malaya Abstractive model
    extractive_combined = " ".join(top_sentences)

    print("\n" + "="*50)
    print("STEP 2: ABSTRACTIVE SUMMARIZATION (Malaya T5-Base)")
    print("="*50)
    print("Rewriting for natural flow and accuracy (This may take a moment)...")

    # --- UPGRADED BASE MODEL ---
    # Load Malaya's Abstractive Model (Upgraded to BASE for smarter context understanding)
    abstractive_model = malaya.summarization.abstractive.huggingface(
        model='mesolitica/finetune-summarization-t5-base-standard-bahasa-cased'
    )
    
    # Apply the abstractive model specifically to the LSA text
    # Added repetition_penalty to FORCE the AI to use different words
    final_summary_list = abstractive_model.generate(
        [extractive_combined],
        max_length=256, 
        num_beams=4, 
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,  # <-- This heavily penalizes copying the exact input
        early_stopping=True
    )
    final_summary = final_summary_list[0]

    print(f"\nFINAL RESULT:\n{final_summary}")

    # 3. SAVE THE FINAL RESULT
    if not os.path.exists("Summaries"):
        os.makedirs("Summaries")
        
    output_path = "Summaries/final_report_LSA.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("--- EXTRACTIVE KEY POINTS (LSA) ---\n")
        f.write("\n".join(top_sentences))
        f.write("\n\n--- FINAL ABSTRACTIVE SUMMARY (T5 BASE) ---\n")
        f.write(final_summary)
    
    print(f"\n[System] Full report saved to: {output_path}")

# --- EXECUTION ---
# Make sure to point this to the file you want to test
cleaned_file_path = "Cleaned_Text/culture_shock_cleaned.txt"
run_hybrid_summarization_lsa(cleaned_file_path)