import re
import gensim
from gensim import corpora
from collections import Counter

# ── Stemmer Setup ───────────────────────────────────────────────
_stemmer = None

def get_stemmer():
    """Load and cache the Sastrawi stemmer."""
    global _stemmer
    if _stemmer is not None:
        return _stemmer
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        _stemmer = StemmerFactory().create_stemmer()
        return _stemmer
    except ImportError:
        print("[Warning] PySastrawi not installed. Stemming disabled.")
        return None

def stem_words(words):
    """
    Stem a list of words and build a reverse mapping.
    
    Returns:
        stemmed_words: list of stemmed tokens
        stem_map: dict mapping each stem -> Counter of original word forms
    """
    stemmer = get_stemmer()
    if stemmer is None:
        return words, {w: Counter({w: 1}) for w in words}
    
    stemmed_words = []
    stem_map = {}  # stem -> Counter({original_word: count})
    
    for word in words:
        stem = stemmer.stem(word)
        stemmed_words.append(stem)
        
        if stem not in stem_map:
            stem_map[stem] = Counter()
        stem_map[stem][word] += 1
    
    return stemmed_words, stem_map

def resolve_stems(stem_keywords, stem_map):
    """
    Map stem keywords back to their most common original word form.
    
    Example: stem 'bincang' -> returns 'perbincangan' if that was
             the most frequent original form in the text.
    """
    resolved = []
    for stem in stem_keywords:
        if stem in stem_map and stem_map[stem]:
            # Pick the most common original form
            most_common = stem_map[stem].most_common(1)[0][0]
            resolved.append(most_common)
        else:
            resolved.append(stem)
    return resolved

def is_noun_or_verb(word):
    """
    Rule-based check if a Malay word is likely a Noun or Verb
    based on common morphological patterns.
    """
    w = word.lower()
    
    # Common Malay NOUN patterns (prefix + suffix combinations)
    noun_patterns = [
        r'^pe\w+an$',       # pe-...-an  (e.g., perbelanjaan, pengurusan, pendidikan)
        r'^per\w+an$',      # per-...-an (e.g., perbincangan, pertumbuhan)
        r'^ke\w+an$',       # ke-...-an  (e.g., keuntungan, kewangan, kesihatan)
        r'^pem\w+an$',      # pem-...-an (e.g., pembangunan, pembelian)
        r'^pen\w+an$',      # pen-...-an (e.g., pendapatan, pencapaian)
        r'^peng\w+an$',     # peng-...-an (e.g., pengurangan, penggunaan)
    ]
    
    # Common Malay VERB patterns
    verb_patterns = [
        r'^ber\w+',         # ber- (e.g., berbincang, bermesyuarat)
        r'^me[mny]\w+',     # mem-, men-, meny- (e.g., membincangkan, menyediakan)
        r'^di\w+kan$',      # di-...-kan (e.g., dikurangkan, dimaksimumkan)
        r'^di\w+i$',        # di-...-i (e.g., diperbaiki, diambili)
        r'^ter\w+',         # ter- (e.g., terbesar, terpengaruh)
    ]
    
    for pattern in noun_patterns + verb_patterns:
        if re.match(pattern, w):
            return True
    
    # Also keep words that are at least 4 chars and don't match
    # common adjective/adverb suffixes
    adj_adv_patterns = [
        r'^se\w+nya$',      # se-...-nya (e.g., secepat, sepenuhnya)
        r'^ter\w+sekali$',  # superlative
    ]
    for pattern in adj_adv_patterns:
        if re.match(pattern, w):
            return False
    
    # Default: keep the word (we don't want to over-filter)
    return True

def filter_content_words(words):
    """
    Filter a list of words to keep only likely nouns and verbs.
    Short words (<=3 chars) and words matching adjective/adverb patterns are removed.
    """
    return [w for w in words if len(w) > 3 and is_noun_or_verb(w)]

def get_malay_stopwords():
    """
    Load Malay stopwords from Malaya if available, else use a hardcoded list.
    """
    try:
        from malaya.text.function import get_stopwords
        return set(get_stopwords())
    except Exception as e:
        # Fallback common Malay stopwords
        return {
            "yang", "dan", "untuk", "di", "ini", "itu", "pada", "dengan", "ke",
            "saya", "dia", "mereka", "kita", "kami", "kamu", "awak", "akan", 
            "boleh", "ada", "tidak", "tak", "dari", "daripada", "kepada", "seperti",
            "oleh", "atau", "juga", "sudah", "dah", "jadi", "jika", "kalau", 
            "dalam", "bagi", "ia", "adalah", "lagi", "lebih", "nya"
        }

def process_text_for_lda(text, stopwords):
    """
    Split text into 'documents' (sentences), tokenize, and stem for LDA.
    Returns:
        texts: list of list of stemmed tokens
        global_stem_map: dict mapping stem -> Counter of original forms
    """
    sentences = re.split(r'[.!?]+', text)
    
    texts = []
    global_stem_map = {}
    
    for sent in sentences:
        words = re.findall(r'\b[a-zA-Z]+\b', sent.lower())
        tokens = [w for w in words if w not in stopwords and len(w) > 2]
        tokens = filter_content_words(tokens)
        
        if tokens:
            stemmed, local_map = stem_words(tokens)
            texts.append(stemmed)
            # Merge local map into global map
            for stem, counter in local_map.items():
                if stem not in global_stem_map:
                    global_stem_map[stem] = Counter()
                global_stem_map[stem] += counter
            
    return texts, global_stem_map

def perform_lda(text, num_topics=3, num_words=5):
    """
    Perform Topic Modeling using Gensim LDA on the provided text.
    """
    if not text or not text.strip():
        return []
        
    stopwords = get_malay_stopwords()
    texts, stem_map = process_text_for_lda(text, stopwords)
    
    if not texts:
        return []
        
    # Create Dictionary
    dictionary = corpora.Dictionary(texts)
    if len(dictionary) == 0:
        return []
        
    # Create Corpus
    corpus = [dictionary.doc2bow(t) for t in texts]
    
    # Train LDA model
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        iterations=50
    )
    
    # Extract Topics and resolve stems back to original forms
    extracted_topics = []
    for topic_id, topic_term_prob in lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False):
        stem_keywords = [word for word, prob in topic_term_prob]
        resolved_words = resolve_stems(stem_keywords, stem_map)
        extracted_topics.append({
            "topic_id": topic_id + 1,
            "words": resolved_words
        })
        
    return extracted_topics

def perform_bertopic(text, num_words=5):
    """
    Perform Topic Modeling using BERTopic with a multilingual sentence transformer.
    """
    if not text or not text.strip():
        return []

    # Local imports for heavy libraries to avoid slowing down overall startup
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[Error] bertopic or sentence_transformers not installed. Fallback to []")
        return []

    # Split text into sentences
    sentences = [sent.strip() for sent in re.split(r'[.!?]+', text) if len(sent.strip()) > 5]
    if len(sentences) < 5:
        print("[BERTopic] Text is too short for clustering (requires at least 5 sentences).")
        return []

    # Build stemmed versions of sentences for the vectorizer + reverse map
    stopwords = get_malay_stopwords()
    global_stem_map = {}
    stemmed_sentences = []
    
    for sent in sentences:
        words = re.findall(r'\b[a-zA-Z]+\b', sent.lower())
        tokens = [w for w in words if w not in stopwords and len(w) > 2]
        stemmed, local_map = stem_words(tokens)
        stemmed_sentences.append(' '.join(stemmed))
        for stem, counter in local_map.items():
            if stem not in global_stem_map:
                global_stem_map[stem] = Counter()
            global_stem_map[stem] += counter

    # If transcript is extremely short (< 15 sentences), BERTopic might struggle to cluster.
    # Adjusting minimum cluster size for short documents:
    min_cluster = 2 if len(sentences) < 20 else 3

    # Dynamically adjust UMAP parameters for very small datasets to avoid eigh/scipy crashes
    try:
        from umap import UMAP
        n_neighbors = min(15, len(sentences) - 1)
        n_components = min(5, len(sentences) - 2)
        umap_model = UMAP(
            n_neighbors=max(2, n_neighbors), 
            n_components=max(1, n_components), 
            min_dist=0.0, 
            metric='cosine', 
            random_state=42
        )
    except ImportError:
        umap_model = None

    # Load multilingual model
    sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    # Pre-compute embeddings from ORIGINAL (unstemmed) sentences for quality
    embeddings = sentence_model.encode(sentences, show_progress_bar=False)
    
    # Vectorizer uses STEMMED sentences for better keyword grouping
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer_model = CountVectorizer(ngram_range=(1, 1))

    # Representation model to tune the topic words
    from bertopic.representation import KeyBERTInspired
    representation_model = KeyBERTInspired()
    
    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=sentence_model, 
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        min_topic_size=min_cluster,
        language="multilingual",
        calculate_probabilities=False
    )

    try:
        # Use stemmed_sentences for document representation but pre-computed embeddings for clustering
        topics, probs = topic_model.fit_transform(stemmed_sentences, embeddings=embeddings)
    except Exception as e:
        print(f"[BERTopic Error] {e}")
        return []

    # Extract topics and resolve stems back to original forms
    extracted_topics = []
    topic_info = topic_model.get_topic_info()
    
    # Ignore the outlier topic (-1)
    important_topics = topic_info[topic_info['Topic'] != -1]
    
    for idx, row in important_topics.iterrows():
        topic_id = row['Topic']
        top_words_probs = topic_model.get_topic(topic_id)
        if top_words_probs:
            stem_keywords = [word for word, prob in top_words_probs][:num_words]
            resolved_words = resolve_stems(stem_keywords, global_stem_map)
            extracted_topics.append({
                "topic_id": topic_id + 1,
                "words": resolved_words
            })

    return extracted_topics

def perform_nmf(text, num_topics=3, num_words=5):
    """
    Perform Topic Modeling using Non-Negative Matrix Factorization (NMF).
    """
    if not text or not text.strip():
        return []
        
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    
    stopwords = list(get_malay_stopwords())
    
    sentences = [sent.strip() for sent in re.split(r'[.!?]+', text) if len(sent.strip()) > 5]
    if not sentences:
        return []
    
    # Stem each sentence and build global reverse map
    global_stem_map = {}
    stemmed_sentences = []
    
    for sent in sentences:
        words = re.findall(r'\b[a-zA-Z]+\b', sent.lower())
        tokens = [w for w in words if w not in stopwords and len(w) > 2]
        stemmed, local_map = stem_words(tokens)
        stemmed_sentences.append(' '.join(stemmed))
        for stem, counter in local_map.items():
            if stem not in global_stem_map:
                global_stem_map[stem] = Counter()
            global_stem_map[stem] += counter
    
    if not any(s.strip() for s in stemmed_sentences):
        return []
        
    # NMF cannot have more components than documents
    num_topics = min(num_topics, len(stemmed_sentences))
        
    vectorizer = TfidfVectorizer(lowercase=True)
    try:
        tfidf = vectorizer.fit_transform(stemmed_sentences)
    except ValueError:
        return []
        
    nmf_model = NMF(n_components=num_topics, random_state=42, init='nndsvd')
    nmf_model.fit(tfidf)
    
    feature_names = vectorizer.get_feature_names_out()
    
    extracted_topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_features_ind = topic.argsort()[:-num_words - 1:-1]
        stem_keywords = [feature_names[i] for i in top_features_ind]
        resolved_words = resolve_stems(stem_keywords, global_stem_map)
        extracted_topics.append({
            "topic_id": topic_idx + 1,
            "words": resolved_words
        })
        
    return extracted_topics

def perform_topic_modeling(text, method="lda", num_topics=3, num_words=5):
    """
    Perform Topic Modeling on the provided text, dispatching based on method.
    """
    method = method.lower()
    if method == "bertopic":
        return perform_bertopic(text, num_words=num_words)
    elif method == "nmf":
        return perform_nmf(text, num_topics=num_topics, num_words=num_words)
    else:
        return perform_lda(text, num_topics=num_topics, num_words=num_words)

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run Topic Modeling on a text file.")
    parser.add_argument("--input", type=str, help="Path to the input text file (e.g., cleaned_text/news.txt)")
    parser.add_argument("--method", type=str, choices=["lda", "bertopic", "nmf", "both"], default="both", help="Topic modeling method")
    parser.add_argument("--num_topics", type=int, default=3, help="Number of topics (LDA only)")
    parser.add_argument("--num_words", type=int, default=5, help="Number of words per topic")
    
    args = parser.parse_args()
    
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Could not find file {args.input}")
            exit(1)
            
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
            
        print(f"\nComparing Topic Models for: {os.path.basename(args.input)}")
        
        if args.method in ["lda", "both"]:
            print(f"\n--- LDA Topics ---")
            topics = perform_topic_modeling(text, method="lda", num_topics=args.num_topics, num_words=args.num_words)
            for t in topics:
                print(f"Topic {t['topic_id']}: {', '.join(t['words'])}")
                
        if args.method in ["bertopic", "both"]:
            print(f"\n--- BERTopic Topics ---")
            topics = perform_topic_modeling(text, method="bertopic", num_topics=args.num_topics, num_words=args.num_words)
            for t in topics:
                print(f"Topic {t['topic_id']}: {', '.join(t['words'])}")
                
        if args.method in ["nmf", "both"]:
            print(f"\n--- NMF Topics ---")
            topics = perform_topic_modeling(text, method="nmf", num_topics=args.num_topics, num_words=args.num_words)
            for t in topics:
                print(f"Topic {t['topic_id']}: {', '.join(t['words'])}")
            
    else:
        # Fallback test if no input file is provided
        sample = (
            "Hari ini kita akan berbincang mengenai bajet kewangan syarikat untuk tahun depan. "
            "Kita perlu pastikan perbelanjaan dikurangkan dan keuntungan dimaksimumkan. "
            "Selain itu, pengurusan sumber manusia juga harus diberi perhatian khusus dalam memastikan "
            "produktiviti pekerja berada di tahap optimum. Proses perancangan strategik ini sangat penting "
            "untuk memastikan kelestarian syarikat pada masa akan datang. Saya harap semua ketua jabatan "
            "dapat memberikan kerjasama sepenuhnya. Mesyuarat pada hari ini juga akan membincangkan "
            "masalah kekurangan kakitangan di bahagian operasi. Kita perlu mencari jalan penyelesaian secepat mungkin."
        )
        print("\n--- Testing LDA ---")
        lda_topics = perform_topic_modeling(sample, method="lda")
        for t in lda_topics:
            print(f"Topic {t['topic_id']}: {', '.join(t['words'])}")
            
        print("\n--- Testing BERTopic ---")
        bertopic_topics = perform_topic_modeling(sample, method="bertopic")
        for t in bertopic_topics:
            print(f"Topic {t['topic_id']}: {', '.join(t['words'])}")
            
        print("\n--- Testing NMF ---")
        nmf_topics = perform_topic_modeling(sample, method="nmf")
        for t in nmf_topics:
            print(f"Topic {t['topic_id']}: {', '.join(t['words'])}")
