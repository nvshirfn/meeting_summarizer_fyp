import re
import gensim
from gensim import corpora

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
    Split text into 'documents' (sentences) and tokenize for LDA.
    """
    # Split text into sentences by common sentence delimiters
    sentences = re.split(r'[.!?]+', text)
    
    texts = []
    for sent in sentences:
        # keep alphanumeric
        words = re.findall(r'\b[a-zA-Z]+\b', sent.lower())
        # remove stopwords and short words
        tokens = [w for w in words if w not in stopwords and len(w) > 2]
        if tokens:
            texts.append(tokens)
            
    return texts

def perform_lda(text, num_topics=3, num_words=5):
    """
    Perform Topic Modeling using Gensim LDA on the provided text.
    """
    if not text or not text.strip():
        return []
        
    stopwords = get_malay_stopwords()
    texts = process_text_for_lda(text, stopwords)
    
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
        random_state=42, # For reproducibility
        passes=10,
        iterations=50
    )
    
    # Extract Topics
    extracted_topics = []
    for topic_id, topic_term_prob in lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False):
        topic_words = [word for word, prob in topic_term_prob]
        extracted_topics.append({
            "topic_id": topic_id + 1,
            "words": topic_words
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
    
    # 1. Stopwords removal configuration for extraction
    from sklearn.feature_extraction.text import CountVectorizer
    malay_stopwords = list(get_malay_stopwords())
    # Restrict to single words to prevent weird collapsed bi-grams
    vectorizer_model = CountVectorizer(stop_words=malay_stopwords, ngram_range=(1, 1))

    # 2. Representation model to tune the topic words
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
        topics, probs = topic_model.fit_transform(sentences)
    except Exception as e:
        print(f"[BERTopic Error] {e}")
        return []

    # Extract topics
    extracted_topics = []
    topic_info = topic_model.get_topic_info()
    
    # Ignore the outlier topic (-1)
    important_topics = topic_info[topic_info['Topic'] != -1]
    
    for idx, row in important_topics.iterrows():
        topic_id = row['Topic']
        # get_topic(id) returns list of (word, probability)
        top_words_probs = topic_model.get_topic(topic_id)
        if top_words_probs:
            words = [word for word, prob in top_words_probs][:num_words]
            extracted_topics.append({
                "topic_id": topic_id + 1,
                "words": words
            })

    return extracted_topics

def perform_topic_modeling(text, method="lda", num_topics=3, num_words=5):
    """
    Perform Topic Modeling on the provided text, dispatching based on method.
    """
    method = method.lower()
    if method == "bertopic":
        return perform_bertopic(text, num_words=num_words)
    else:
        return perform_lda(text, num_topics=num_topics, num_words=num_words)

if __name__ == "__main__":
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
