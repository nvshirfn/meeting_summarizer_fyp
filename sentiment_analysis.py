"""
Sentiment Analysis for Malay Text
Supports three methods:
  - 'bert'            : Malaya's HuggingFace transformer model (Mesolitica NanoT5)
  - 'multinomial_nb'  : Malaya's pre-trained Multinomial Naive Bayes (ML-based)
  - 'lexicon'         : Simple polarity-based word-counting (legacy fallback)
"""

import re
import warnings

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
# METHOD 1 — TRANSFORMER (MALAYA BERT / NanoT5)
# ═══════════════════════════════════════════════════════════════════

_model = None


def _get_model():
    """Lazy-load the Malaya sentiment transformer model."""
    global _model
    if _model is not None:
        return _model

    import malaya

    print("[Sentiment] Loading Malaya transformer sentiment model …")
    _model = malaya.sentiment.huggingface()
    print("[Sentiment] Model loaded successfully.")
    return _model


def _split_sentences(text):
    """
    Split text into sentences for per-sentence sentiment analysis.
    Handles Malay punctuation patterns common in meeting transcripts.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 5]
    return sentences if sentences else [text.strip()]


def _analyze_bert(text):
    """Transformer-based sentiment analysis using Malaya."""
    model = _get_model()
    sentences = _split_sentences(text)

    proba_results = model.predict_proba(sentences)

    sentence_results = []
    total_pos = 0.0
    total_neg = 0.0
    total_neu = 0.0

    for sentence, proba in zip(sentences, proba_results):
        pos = proba.get("positive", 0.0)
        neg = proba.get("negative", 0.0)
        neu = proba.get("neutral", 0.0)

        sent_label = max(proba, key=proba.get)

        sentence_results.append({
            "text": sentence,
            "sentiment": sent_label,
            "positive": round(pos, 4),
            "negative": round(neg, 4),
            "neutral": round(neu, 4),
        })

        total_pos += pos
        total_neg += neg
        total_neu += neu

    n = len(sentences)
    avg_pos = total_pos / n
    avg_neg = total_neg / n
    avg_neu = total_neu / n

    scores = {"positive": avg_pos, "negative": avg_neg, "neutral": avg_neu}
    overall_sentiment = max(scores, key=scores.get)
    confidence = scores[overall_sentiment]

    return {
        "sentiment": overall_sentiment,
        "positive_score": round(avg_pos, 4),
        "negative_score": round(avg_neg, 4),
        "neutral_score": round(avg_neu, 4),
        "confidence": round(confidence, 4),
        "sentence_results": sentence_results,
    }


# ═══════════════════════════════════════════════════════════════════
# METHOD 2 — MULTINOMIAL NAIVE BAYES (MALAYA ML)
# ═══════════════════════════════════════════════════════════════════

_nb_model = None


def _repair_old_sklearn_nb_model(model):
    """
    Patch Malaya's old sklearn 0.22 pickled NB model so it can run on the
    newer sklearn version installed in this environment.
    """
    vectorizer = getattr(model, "_vectorize", None)
    classifier = getattr(model, "_multinomial", None)

    if vectorizer is not None:
        vocabulary = getattr(vectorizer, "vocabulary_", None)
        feature_count = len(vocabulary) if vocabulary is not None else None

        tfidf = getattr(vectorizer, "_tfidf", None)
        if tfidf is not None and feature_count is not None:
            tfidf.n_features_in_ = feature_count

    if classifier is not None:
        if not hasattr(classifier, "force_alpha"):
            classifier.force_alpha = True
        if not hasattr(classifier, "n_features_in_"):
            classifier.n_features_in_ = getattr(
                classifier,
                "n_features_",
                len(getattr(vectorizer, "vocabulary_", {}) or {}),
            )


def _get_nb_model():
    """Lazy-load the Malaya Multinomial Naive Bayes sentiment model."""
    global _nb_model
    if _nb_model is not None:
        return _nb_model

    import malaya

    print("[Sentiment] Loading Malaya Multinomial Naive Bayes sentiment model …")
    _nb_model = malaya.sentiment.multinomial()
    _repair_old_sklearn_nb_model(_nb_model)
    print("[Sentiment] Multinomial NB model loaded successfully.")
    return _nb_model


def _analyze_multinomial_nb(text):
    """ML-based sentiment analysis using Malaya's Multinomial Naive Bayes."""
    model = _get_nb_model()
    sentences = _split_sentences(text)

    proba_results = model.predict_proba(sentences)

    sentence_results = []
    total_pos = 0.0
    total_neg = 0.0
    total_neu = 0.0

    for sentence, proba in zip(sentences, proba_results):
        pos = proba.get("positive", 0.0)
        neg = proba.get("negative", 0.0)
        neu = proba.get("neutral", 0.0)

        sent_label = max(proba, key=proba.get)

        sentence_results.append({
            "text": sentence,
            "sentiment": sent_label,
            "positive": round(pos, 4),
            "negative": round(neg, 4),
            "neutral": round(neu, 4),
        })

        total_pos += pos
        total_neg += neg
        total_neu += neu

    n = len(sentences)
    avg_pos = total_pos / n
    avg_neg = total_neg / n
    avg_neu = total_neu / n

    scores = {"positive": avg_pos, "negative": avg_neg, "neutral": avg_neu}
    overall_sentiment = max(scores, key=scores.get)
    confidence = scores[overall_sentiment]

    return {
        "sentiment": overall_sentiment,
        "positive_score": round(avg_pos, 4),
        "negative_score": round(avg_neg, 4),
        "neutral_score": round(avg_neu, 4),
        "confidence": round(confidence, 4),
        "sentence_results": sentence_results,
    }


# ═══════════════════════════════════════════════════════════════════
# METHOD 3 — LEXICON-BASED (LEGACY FALLBACK)
# ═══════════════════════════════════════════════════════════════════

POSITIVE_WORDS = {
    'gembira', 'bahagia', 'senang', 'seronok', 'suka', 'cinta', 'sayang',
    'syukur', 'bersyukur', 'bangga', 'kagum', 'teruja', 'semangat',
    'tenang', 'damai', 'aman', 'selesa', 'puas', 'lega', 'harap', 'yakin',
    'baik', 'bagus', 'cantik', 'indah', 'hebat', 'cemerlang', 'gemilang',
    'terbaik', 'unggul', 'bijak', 'pandai', 'rajin', 'tekun', 'gigih',
    'bersih', 'kemas', 'teratur', 'maju', 'berjaya', 'jayanya',
    'positif', 'optimis', 'berkesan', 'efektif', 'produktif',
    'berjaya', 'menang', 'capai', 'mencapai', 'tercapai', 'meningkat',
    'berkembang', 'membangun', 'menyokong', 'membantu', 'menyumbang',
    'menghargai', 'menghormati', 'memperbaiki', 'memuji', 'mengagumi',
    'memperkasa', 'memperkukuh', 'memajukan', 'melindungi',
    'setuju', 'bersetuju', 'terima kasih', 'tahniah', 'syabas',
    'sempurna', 'lengkap', 'mantap', 'kukuh', 'stabil',
    'harmoni', 'sejahtera', 'makmur', 'adil', 'saksama',
    'sangat', 'amat', 'sungguh', 'betul',
}

NEGATIVE_WORDS = {
    'sedih', 'kecewa', 'marah', 'benci', 'takut', 'bimbang', 'risau',
    'gelisah', 'murung', 'putus asa', 'tertekan', 'stress', 'malu',
    'cemburu', 'dengki', 'iri', 'sakit hati', 'duka', 'sengsara',
    'buruk', 'teruk', 'jahat', 'kejam', 'zalim', 'kotor', 'busuk',
    'hodoh', 'lemah', 'bodoh', 'malas', 'cuai', 'gagal', 'rugi',
    'negatif', 'pesimis', 'mundur', 'merosot', 'rosak', 'musnah',
    'bahaya', 'berbahaya', 'beracun', 'merbahaya',
    'gagal', 'kalah', 'jatuh', 'runtuh', 'hancur', 'rosak',
    'menurun', 'merosot', 'merosakkan', 'menghancurkan', 'membunuh',
    'mencuri', 'menipu', 'mengkhianati', 'mengancam', 'menindas',
    'mengeksploitasi', 'menyalahgunakan', 'merugikan', 'menzalimi',
    'masalah', 'krisis', 'bencana', 'konflik', 'pertikaian',
    'rasuah', 'penipuan', 'jenayah', 'keganasan', 'kemiskinan',
    'pengangguran', 'pencemaran', 'kemusnahan',
    'tidak', 'bukan', 'tiada', 'tanpa', 'jangan',
}


def _analyze_lexicon(text):
    """Legacy lexicon-based sentiment analysis."""
    text_lower = text.lower()

    pos_found = []
    for word in POSITIVE_WORDS:
        matches = re.findall(rf'\b{re.escape(word)}\b', text_lower)
        pos_found.extend([word] * len(matches))

    neg_found = []
    for word in NEGATIVE_WORDS:
        matches = re.findall(rf'\b{re.escape(word)}\b', text_lower)
        neg_found.extend([word] * len(matches))

    pos_score = len(pos_found)
    neg_score = len(neg_found)

    if pos_score > neg_score:
        sentiment = "positive"
    elif neg_score > pos_score:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "sentiment": sentiment,
        "positive_score": pos_score,
        "negative_score": neg_score,
        "positive_words_found": sorted(set(pos_found)),
        "negative_words_found": sorted(set(neg_found)),
    }


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def analyze_sentiment(text, method="bert"):
    """
    Analyze sentiment of Malay text.

    Args:
        text (str): Preprocessed Malay text.
        method (str): 'bert' for transformer-based (default),
                      'multinomial_nb' for Naive Bayes ML model,
                      'lexicon' for word-counting fallback.

    Returns:
        dict with sentiment label and scores.
    """
    if not text or not text.strip():
        return {
            "sentiment": "neutral",
            "positive_score": 0.0,
            "negative_score": 0.0,
            "neutral_score": 0.0,
            "confidence": 0.0,
            "sentence_results": [],
        }

    if method == "lexicon":
        return _analyze_lexicon(text)
    elif method == "multinomial_nb":
        return _analyze_multinomial_nb(text)
    else:
        return _analyze_bert(text)


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sentiment Analysis for Malay Text"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to a text file to analyze",
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["bert", "multinomial_nb", "lexicon"],
        default="bert",
        help="Analysis method: 'bert' (transformer), 'multinomial_nb' (Naive Bayes), or 'lexicon' (word-counting). Default: bert",
    )
    args = parser.parse_args()

    # Load text from file or use built-in samples
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read().strip()
        print(f"File:       {args.input}")
        print(f"Method:     {args.method}")
        print(f"Length:     {len(text)} chars")
        print("=" * 60)

        result = analyze_sentiment(text, method=args.method)
        print(f"Sentiment:  {result['sentiment'].upper()}")

        if args.method in ("bert", "multinomial_nb"):
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Positive:   {result['positive_score']:.1%}")
            print(f"Negative:   {result['negative_score']:.1%}")
            print(f"Neutral:    {result['neutral_score']:.1%}")

            if result.get("sentence_results"):
                print(f"\nPer-sentence breakdown ({len(result['sentence_results'])} sentences):")
                print("-" * 60)
                for sr in result["sentence_results"]:
                    print(f"  [{sr['sentiment'].upper():>8}] {sr['text']}")
        else:
            print(f"Positive:   {result['positive_score']} words ({', '.join(result.get('positive_words_found', []))})")
            print(f"Negative:   {result['negative_score']} words ({', '.join(result.get('negative_words_found', []))})")
    else:
        # No file provided — run built-in demo samples
        samples = [
            "Terima kasih atas penerangan yang sangat jelas dan baik. Kami sangat gembira.",
            "Keadaan ekonomi semakin teruk dan rakyat kecewa dengan masalah rasuah.",
            "Mesyuarat hari ini membincangkan perkara biasa mengenai jadual kerja.",
        ]
        print(f"No --input file given. Running demo samples with method={args.method}.\n")

        for text in samples:
            result = analyze_sentiment(text, method=args.method)
            print(f"Text:       {text}")
            print(f"Sentiment:  {result['sentiment'].upper()}")
            if args.method in ("bert", "multinomial_nb"):
                print(f"Confidence: {result['confidence']:.1%}")
                print(f"Positive:   {result['positive_score']:.1%}")
                print(f"Negative:   {result['negative_score']:.1%}")
                print(f"Neutral:    {result['neutral_score']:.1%}")
            else:
                print(f"Positive:   {result['positive_score']} words")
                print(f"Negative:   {result['negative_score']} words")
            print("-" * 60)
