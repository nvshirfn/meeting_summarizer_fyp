"""
Extractive Summarization Module for Malay Text
Supports 3 methods: TextRank, LSA, and ELECTRA
"""

import numpy as np


def _apply_patches():
    import inspect
    import scipy
    if getattr(_apply_patches, "called", False): return
    _apply_patches.called = True
    
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec

    if not hasattr(scipy, 'asarray'):
        scipy.asarray = np.asarray
    if not hasattr(scipy, 'ones'):
        scipy.ones = np.ones
    if not hasattr(scipy, 'zeros'):
        scipy.zeros = np.zeros
    if not hasattr(scipy, 'array'):
        scipy.array = np.array


def tokenize_sentences(text):
    """
    Robust sentence tokenization that handles Malay abbreviations and multiple punctuation.
    Uses Malaya's sentence splitter with a smart Regex fallback.
    """
    import re
    try:
        from malaya.text.function import split_into_sentences
        raw_sentences = split_into_sentences(text)
    except Exception:
        # Fallback if Malaya function is unavailable
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        
    sentences = [s.strip(' \t\n\r"”“\'‘’') for s in raw_sentences if len(s.strip(' \t\n\r"”“\'‘’')) > 5]
    # Ensure they end with punctuation for clean generation later
    sentences = [s if re.search(r'[.!?]$', s) else s + '.' for s in sentences]
    return sentences


def extractive_textrank(text, ratio=0.20, min_sentences=3):
    """
    Extractive summarization using TFIDF + NetworkX (TextRank) tuned for Malay.
    Replaces sumy with a custom TextRank utilizing Malay stopwords.
    """
    _apply_patches()
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    sentences = tokenize_sentences(text)
    
    total_sentences = len(sentences)
    sentences_to_extract = max(min_sentences, int(total_sentences * ratio))
    
    if total_sentences <= sentences_to_extract:
        top_sentences = sentences
    else:
        try:
            from malaya.text.function import get_stopwords
            stopwords = get_stopwords()
        except:
            stopwords = ["yang", "dan", "untuk", "di", "ke", "dari", "ini", "itu", "dengan", "kepada", "adalah", "pada", "bahawa", "mereka", "kita", "saya", "dia", "dalam", "akan"]
            
        vectorizer = TfidfVectorizer(stop_words=stopwords)
        try:
            X = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(X)
            
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
            
            ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
            top_sentences = [s for score, s in ranked_sentences[:sentences_to_extract]]
            
            top_sentences = sorted(top_sentences, key=lambda x: sentences.index(x))
        except Exception:
            top_sentences = sentences[:sentences_to_extract]

    return {
        "method": "textrank",
        "sentences": top_sentences,
        "combined": " ".join(top_sentences),
        "total_sentences": total_sentences,
        "extracted_count": len(top_sentences)
    }


def extractive_lsa(text, ratio=0.20, min_sentences=3):
    """
    Extractive summarization using Malaya's unsupervised SKLearn interface (LSA).
    Native Malay processing, replaces sumy LSA.
    """
    _apply_patches()
    import malaya
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    sentences = tokenize_sentences(text)
    total_sentences = len(sentences)
    sentences_to_extract = max(min_sentences, int(total_sentences * ratio))

    if total_sentences <= sentences_to_extract:
        top_sentences = sentences
    else:
        try:
            from malaya.text.function import get_stopwords
            stopwords = get_stopwords()
        except:
            stopwords = ["yang", "dan", "untuk", "di", "ke", "dari", "ini", "itu", "dengan", "kepada", "adalah", "pada", "bahawa", "mereka", "kita", "saya", "dia", "dalam", "akan"]

        svd = TruncatedSVD(n_components=max(1, sentences_to_extract // 2))
        vectorizer = TfidfVectorizer(stop_words=stopwords)
        try:
            extractive_model = malaya.summarization.extractive.sklearn(svd, vectorizer)
            clean_text = " ".join(sentences)
            summary_data = extractive_model.sentence_level(clean_text, top_k=sentences_to_extract)
            
            if isinstance(summary_data, dict) and 'summary' in summary_data:
                combined = summary_data['summary']
                top_sentences = tokenize_sentences(combined)
            else:
                top_sentences = sentences[:sentences_to_extract]
        except Exception:
            top_sentences = sentences[:sentences_to_extract]

    return {
        "method": "lsa",
        "sentences": top_sentences,
        "combined": " ".join(top_sentences),
        "total_sentences": total_sentences,
        "extracted_count": len(top_sentences)
    }


def extractive_electra(text, ratio=0.20, min_sentences=3, min_words=6):
    """
    Extractive summarization using Malaya ELECTRA Encoder.
    Deep semantic method — Malay-specific embeddings, best context understanding.

    Args:
        text: Input text to summarize
        ratio: Fraction of sentences to extract (default 20%)
        min_sentences: Minimum number of sentences to extract
        min_words: Minimum word count for a sentence to be eligible for extraction

    Returns:
        dict with keys: method, sentences, combined, total_sentences, extracted_count
    """
    _apply_patches()

    import malaya

    # Tokenize and filter out short low-content sentences
    input_sentences = tokenize_sentences(text)
    total_sentences = len(input_sentences)
    eligible = [s for s in input_sentences if len(s.split()) >= min_words]

    # Deduplicate near-identical sentences (keeps first occurrence)
    seen = []
    for s in eligible:
        normalised = re.sub(r'\s+', ' ', s.strip().lower())
        is_duplicate = any(
            sum(w in normalised.split() for w in ref.split()) / max(len(ref.split()), 1) > 0.85
            for ref in seen
        )
        if not is_duplicate:
            seen.append(normalised)
    deduped_text = " ".join(
        s for s in eligible
        if re.sub(r'\s+', ' ', s.strip().lower()) in seen
    )

    eligible_count = len(seen)
    sentences_to_extract = max(min_sentences, int(eligible_count * ratio))

    # Load ELECTRA model
    transformer_model = malaya.transformer.huggingface(
        model='mesolitica/electra-base-generator-bahasa-cased',
        attn_implementation="eager"
    )
    extractive_model = malaya.summarization.extractive.encoder(transformer_model)

    # Extract summary from deduplicated, filtered text
    summary_data = extractive_model.sentence_level(deduped_text, top_k=sentences_to_extract)

    # Parse individual sentences from the combined summary
    combined = summary_data['summary']
    sentences = tokenize_sentences(combined)

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
