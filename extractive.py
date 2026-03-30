"""
Extractive Summarization Module for Malay Text
Supports 3 methods: TextRank, LSA, and ELECTRA
"""

import numpy as np


def extractive_textrank(text, ratio=0.20, min_sentences=3):
    """
    Extractive summarization using Sumy TextRank.
    Graph-based method — fast, language-agnostic.
    
    Args:
        text: Input text to summarize
        ratio: Fraction of sentences to extract (default 20%)
        min_sentences: Minimum number of sentences to extract
    
    Returns:
        dict with keys: method, sentences, combined, total_sentences, extracted_count
    """
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()

    total_sentences = len(parser.document.sentences)
    sentences_to_extract = max(min_sentences, int(total_sentences * ratio))

    top_sentences_raw = summarizer(parser.document, sentences_to_extract)
    top_sentences = [str(s) for s in top_sentences_raw]

    return {
        "method": "textrank",
        "sentences": top_sentences,
        "combined": " ".join(top_sentences),
        "total_sentences": total_sentences,
        "extracted_count": len(top_sentences)
    }


def extractive_lsa(text, ratio=0.20, min_sentences=3):
    """
    Extractive summarization using Sumy LSA (Latent Semantic Analysis).
    Topic-aware method — good diversity of extracted sentences.
    
    Args:
        text: Input text to summarize
        ratio: Fraction of sentences to extract (default 20%)
        min_sentences: Minimum number of sentences to extract
    
    Returns:
        dict with keys: method, sentences, combined, total_sentences, extracted_count
    """
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()

    total_sentences = len(parser.document.sentences)
    sentences_to_extract = max(min_sentences, int(total_sentences * ratio))

    top_sentences_raw = summarizer(parser.document, sentences_to_extract)
    top_sentences = [str(s) for s in top_sentences_raw]

    return {
        "method": "lsa",
        "sentences": top_sentences,
        "combined": " ".join(top_sentences),
        "total_sentences": total_sentences,
        "extracted_count": len(top_sentences)
    }


def extractive_electra(text, ratio=0.10, min_sentences=3, max_sentences=8):
    """
    Extractive summarization using Malaya ELECTRA Encoder.
    Deep semantic method — Malay-specific embeddings, best context understanding.
    
    Args:
        text: Input text to summarize
        ratio: Fraction of sentences to extract (default 10%)
        min_sentences: Minimum number of sentences to extract
        max_sentences: Maximum number of sentences to extract (to avoid overwhelming T5)
    
    Returns:
        dict with keys: method, sentences, combined, total_sentences, extracted_count
    """
    import inspect
    import scipy

    # Python 3.11+ compatibility patch for Malaya
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec

    # SciPy 1.13+ compatibility patch for Malaya
    if not hasattr(scipy, 'asarray'):
        scipy.asarray = np.asarray
    if not hasattr(scipy, 'ones'):
        scipy.ones = np.ones
    if not hasattr(scipy, 'zeros'):
        scipy.zeros = np.zeros
    if not hasattr(scipy, 'array'):
        scipy.array = np.array

    import malaya

    # Estimate total sentences
    total_sentences = len([s for s in text.split('.') if len(s.strip()) > 5])
    sentences_to_extract = min(max_sentences, max(min_sentences, int(total_sentences * ratio)))

    # Load ELECTRA model
    transformer_model = malaya.transformer.huggingface(
        model='mesolitica/electra-small-generator-bahasa-cased',
        attn_implementation="eager"
    )
    extractive_model = malaya.summarization.extractive.encoder(transformer_model)

    # Extract summary
    summary_data = extractive_model.sentence_level(text, top_k=sentences_to_extract)

    # Parse individual sentences from the combined summary
    combined = summary_data['summary']
    sentences = [s.strip() for s in combined.split('.') if len(s.strip()) > 5]
    # Add periods back
    sentences = [s + '.' if not s.endswith('.') else s for s in sentences]

    return {
        "method": "electra",
        "sentences": sentences,
        "combined": combined,
        "total_sentences": total_sentences,
        "extracted_count": len(sentences)
    }


# Lookup for convenient access by method name
METHODS = {
    "textrank": extractive_textrank,
    "lsa": extractive_lsa,
    "electra": extractive_electra,
}


def run_extractive(text, method="textrank", **kwargs):
    """
    Run extractive summarization using the specified method.
    
    Args:
        text: Input text to summarize
        method: One of "textrank", "lsa", "electra"
        **kwargs: Additional arguments passed to the method function
    
    Returns:
        dict with extractive summary results
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(METHODS.keys())}")
    
    return METHODS[method](text, **kwargs)


# --- STANDALONE EXECUTION ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extractive summarization for Malay text")
    parser.add_argument("--input", required=True, help="Path to input text file")
    parser.add_argument("--method", choices=["textrank", "lsa", "electra"], default="textrank",
                        help="Extractive method to use")
    
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"\n{'='*60}")
    print(f"  EXTRACTIVE SUMMARIZATION — {args.method.upper()}")
    print(f"{'='*60}\n")

    result = run_extractive(text, method=args.method)

    print(f"Total sentences: {result['total_sentences']}")
    print(f"Extracted: {result['extracted_count']} sentences\n")
    
    for i, sent in enumerate(result['sentences'], 1):
        print(f"  {i}. {sent}")
    
    print(f"\n{'='*60}")
