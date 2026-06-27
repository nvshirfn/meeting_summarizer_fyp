"""
Extractive Summarization Module for Malay Text
Supports 3 methods: TextRank, LSA, and ELECTRA
"""

import re
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


def extractive_textrank(text, ratio=0.12, min_sentences=3, max_sentences=15, min_words=9,
                        max_words=45, content_density_threshold=0.30, n_sections=3):
    """
    Extractive summarization using TFIDF + NetworkX (TextRank) + MMR re-ranking, tuned for Malay.
    """
    _apply_patches()
    import networkx as nx
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    all_sentences = tokenize_sentences(text)
    total_sentences = len(all_sentences)

    # Load stopwords early — needed for content density filter and TF-IDF
    try:
        from malaya.text.function import get_stopwords
        stopwords = set(get_stopwords())
    except Exception:
        stopwords = {"yang", "dan", "untuk", "di", "ke", "dari", "ini", "itu", "dengan",
                     "kepada", "adalah", "pada", "bahawa", "mereka", "kita", "saya", "dia",
                     "dalam", "akan", "tidak", "tak", "juga", "sudah", "atau", "oleh"}

    # 1. Minimum word filter — preserve original sentence positions for section tracking
    indexed = [(i, s) for i, s in enumerate(all_sentences) if len(s.split()) >= min_words]

    # 2. Content density filter — drop sentences with too few content words (catches filler/lyrics)
    def _content_density(s):
        words = re.sub(r'[^\w\s]', '', s.lower()).split()
        if not words:
            return 0.0
        return sum(1 for w in words if w not in stopwords) / len(words)

    indexed = [(i, s) for i, s in indexed if _content_density(s) >= content_density_threshold]

    # 3. Question filter — exclude short interview prompts and filler questions (< 20 words)
    #    Long ?-sentences (≥ 20 words) are kept: they tend to be informative debate challenges
    indexed = [(i, s) for i, s in indexed if not (s.rstrip().endswith('?') and len(s.split()) < 20)]

    # 4. Deduplication — bidirectional overlap check (runs after density filter to reduce cost)
    seen_norms = []
    deduped_indexed = []
    for i, s in indexed:
        normalised = re.sub(r'\s+', ' ', s.strip().lower())
        norm_words = normalised.split()
        is_duplicate = any(
            max(
                sum(w in norm_words for w in ref.split()) / max(len(ref.split()), 1),
                sum(w in ref.split() for w in norm_words) / max(len(norm_words), 1)
            ) > 0.75
            for ref in seen_norms
        )
        if not is_duplicate:
            seen_norms.append(normalised)
            deduped_indexed.append((i, s))

    original_indices = [i for i, _ in deduped_indexed]
    sentences = [s for _, s in deduped_indexed]

    # Base target on original document length so pre-filters don't shrink output count
    sentences_to_extract = min(max_sentences, max(min_sentences, int(total_sentences * ratio)))

    if len(sentences) <= sentences_to_extract:
        top_sentences = sentences
    else:
        vectorizer = TfidfVectorizer(stop_words=list(stopwords))
        try:
            X = vectorizer.fit_transform(sentences)

            # 4. PageRank scores
            similarity_matrix = cosine_similarity(X)
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
            pagerank_scores = np.array([scores[i] for i in range(len(sentences))])

            # 5. Graph-coherence filter — drop sentences poorly connected to the document body.
            #    Average inter-sentence similarity captures true semantic isolation better than
            #    centroid distance: off-topic segments share common words with the centroid but
            #    have low pairwise similarity to the main content sentences.
            n = len(sentences)
            if n > min_sentences:
                diag = np.diag(similarity_matrix)
                avg_inter = (similarity_matrix.sum(axis=1) - diag) / max(n - 1, 1)
                cutoff = avg_inter.mean() - avg_inter.std()
                coherent = avg_inter >= max(cutoff, 1e-6)
                if coherent.sum() >= min_sentences:
                    keep = np.where(coherent)[0]
                    sentences = [sentences[i] for i in keep]
                    original_indices = [original_indices[i] for i in keep]
                    similarity_matrix = similarity_matrix[np.ix_(keep, keep)]
                    pagerank_scores = pagerank_scores[keep]

            # 6. Length penalty — demote run-on sentences (> max_words words)
            word_counts = np.array([len(s.split()) for s in sentences])
            long_mask = word_counts > max_words
            if long_mask.any():
                pagerank_scores = pagerank_scores * np.where(long_mask, 0.25, 1.0)

            # 7. Section-based MMR — guarantee coverage across document thirds
            lambda_mmr = 0.85
            section_size = total_sentences / n_sections
            sentence_sections = [
                min(int(idx / section_size), n_sections - 1) for idx in original_indices
            ]
            min_per_section = max(1, sentences_to_extract // (n_sections + 1))

            selected_idx = []
            remaining_idx = list(range(len(sentences)))

            # First pass: enforce minimum from each section using MMR
            for section in range(n_sections):
                section_cands = [i for i in remaining_idx if sentence_sections[i] == section]
                picks = 0
                while picks < min_per_section and section_cands:
                    if not selected_idx:
                        best = max(section_cands, key=lambda i: pagerank_scores[i])
                    else:
                        best = max(
                            section_cands,
                            key=lambda i: (
                                lambda_mmr * pagerank_scores[i]
                                - (1 - lambda_mmr) * max(similarity_matrix[i][j] for j in selected_idx)
                            )
                        )
                    selected_idx.append(best)
                    remaining_idx.remove(best)
                    section_cands.remove(best)
                    picks += 1

            # Second pass: fill remaining slots with open MMR
            while len(selected_idx) < sentences_to_extract and remaining_idx:
                if not selected_idx:
                    best = max(remaining_idx, key=lambda i: pagerank_scores[i])
                else:
                    best = max(
                        remaining_idx,
                        key=lambda i: (
                            lambda_mmr * pagerank_scores[i]
                            - (1 - lambda_mmr) * max(similarity_matrix[i][j] for j in selected_idx)
                        )
                    )
                selected_idx.append(best)
                remaining_idx.remove(best)

            # Restore original document order
            top_sentences = [sentences[i] for i in sorted(selected_idx)]

        except Exception:
            top_sentences = sentences[:sentences_to_extract]

    return {
        "method": "textrank",
        "sentences": top_sentences,
        "combined": " ".join(top_sentences),
        "total_sentences": total_sentences,
        "extracted_count": len(top_sentences)
    }


def extractive_lsa(text, ratio=0.20, min_sentences=3, min_words=8, max_words=65, max_sentences=20):
    """
    Extractive summarization using Malaya's unsupervised SKLearn interface (LSA).
    Native Malay processing, replaces sumy LSA.
    """
    _apply_patches()
    import malaya
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    all_sentences = tokenize_sentences(text)
    total_sentences = len(all_sentences)

    # Word-count gate — keep only sentences within [min_words, max_words]
    # Applied before LSA so the model scores a clean candidate pool
    sentences = [s for s in all_sentences if min_words <= len(s.split()) <= max_words]

    # Deduplicate near-identical sentences — bidirectional word overlap > 0.75
    seen_norm = []
    deduped = []
    for s in sentences:
        norm = re.sub(r'\s+', ' ', s.strip().lower())
        nw = norm.split()
        if not any(
            max(
                sum(w in nw for w in ref.split()) / max(len(ref.split()), 1),
                sum(w in ref.split() for w in nw) / max(len(nw), 1)
            ) > 0.75
            for ref in seen_norm
        ):
            seen_norm.append(norm)
            deduped.append(s)
    sentences = deduped

    sentences_to_extract = max(min_sentences, int(total_sentences * ratio))
    sentences_to_extract = min(sentences_to_extract, max_sentences)

    if len(sentences) <= sentences_to_extract:
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


def extractive_electra(text, ratio=0.20, min_sentences=3, min_words=10, max_words=65, max_sentences=10,
                       model='mesolitica/electra-base-generator-bahasa-cased'):
    """
    Extractive summarization using Malaya ELECTRA Encoder.
    Deep semantic method — Malay-specific embeddings, best context understanding.

    Args:
        text: Input text to summarize
        ratio: Fraction of sentences to extract (default 20%)
        min_sentences: Minimum number of sentences to extract
        min_words: Minimum word count for a sentence to be eligible for extraction
        max_words: Maximum word count — excludes run-on sentences from candidate pool
        max_sentences: Hard ceiling on output sentence count

    Returns:
        dict with keys: method, sentences, combined, total_sentences, extracted_count
    """
    _apply_patches()

    import malaya

    # Tokenize and apply word-count gate [min_words, max_words]
    input_sentences = tokenize_sentences(text)
    total_sentences = len(input_sentences)
    eligible = [s for s in input_sentences if min_words <= len(s.split()) <= max_words]
    after_length = len(eligible)

    # Deduplicate near-identical sentences — bidirectional check (keeps first occurrence)
    seen = []
    for s in eligible:
        normalised = re.sub(r'\s+', ' ', s.strip().lower())
        norm_words = normalised.split()
        is_duplicate = any(
            max(
                sum(w in norm_words for w in ref.split()) / max(len(ref.split()), 1),
                sum(w in ref.split() for w in norm_words) / max(len(norm_words), 1)
            ) > 0.75
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
    sentences_to_extract = min(sentences_to_extract, max_sentences)

    print(f"  [ELECTRA] Total sentences:     {total_sentences}")
    print(f"  [ELECTRA] After length filter: {after_length} (min_words={min_words}, max_words={max_words})")
    print(f"  [ELECTRA] After dedup:         {eligible_count}")
    print(f"  [ELECTRA] Extracting:          {sentences_to_extract} sentences (ratio={ratio})")

    MALAYA_MODELS = {'mesolitica/electra-base-generator-bahasa-cased',
                     'mesolitica/electra-small-generator-bahasa-cased'}

    print(f"  [ELECTRA] Loading model: {model}")

    if model in MALAYA_MODELS:
        # Use Malaya's wrapper for known-supported generator models
        transformer_model = malaya.transformer.huggingface(
            model=model,
            attn_implementation="eager"
        )
        extractive_model = malaya.summarization.extractive.encoder(transformer_model)
        summary_data = extractive_model.sentence_level(deduped_text, top_k=sentences_to_extract)
        combined = summary_data['summary']
        sentences = tokenize_sentences(combined)
    else:
        # Direct HuggingFace path for models not supported by Malaya's wrapper
        # (e.g. discriminator variant). No silent fallback — errors surface fully.
        import torch
        from transformers import AutoTokenizer, AutoModel
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        tokenizer = AutoTokenizer.from_pretrained(model)
        hf_model = AutoModel.from_pretrained(model)
        hf_model.eval()

        input_sentences = tokenize_sentences(deduped_text)
        if not input_sentences:
            input_sentences = [deduped_text]

        # Get mean-pooled embeddings for each sentence
        def embed(sents):
            enc = tokenizer(sents, padding=True, truncation=True,
                            max_length=512, return_tensors="pt")
            with torch.no_grad():
                out = hf_model(**enc)
            return out.last_hidden_state.mean(dim=1).numpy()

        embeddings = embed(input_sentences)
        sim_matrix = cos_sim(embeddings)
        scores = sim_matrix.sum(axis=1)

        ranked_idx = scores.argsort()[::-1]
        top_idx = sorted(ranked_idx[:sentences_to_extract])
        sentences = [input_sentences[i] for i in top_idx]
        combined = " ".join(sentences)

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
    parser.add_argument("--model", default=None,
                        help="Override ELECTRA model (e.g. mesolitica/electra-base-discriminator-bahasa-cased)")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"\n{'='*60}")
    print(f"  EXTRACTIVE SUMMARIZATION — {args.method.upper()}")
    print(f"{'='*60}\n")

    kwargs = {}
    if args.model and args.method == "electra":
        kwargs["model"] = args.model

    result = run_extractive(text, method=args.method, **kwargs)

    print(f"Total sentences: {result['total_sentences']}")
    print(f"Extracted: {result['extracted_count']} sentences\n")
    
    for i, sent in enumerate(result['sentences'], 1):
        print(f"  {i}. {sent}")
    
    print(f"\n{'='*60}")
