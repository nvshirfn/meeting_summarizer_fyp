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

def perform_topic_modeling(text, num_topics=3, num_words=5):
    """
    Perform Topic Modeling using Gensim LDA on the provided text.
    
    Args:
        text (str): Preprocessed Malay text.
        num_topics (int): Number of topics to discover.
        num_words (int): Number of descriptive words per topic.
        
    Returns:
        List of dictionaries containing topic details:
        [{'topic_id': 0, 'words': ['word1', 'word2', ...]}, ...]
    """
    if not text or not text.strip():
        return []
        
    stopwords = get_malay_stopwords()
    texts = process_text_for_lda(text, stopwords)
    
    if not texts:
        return []
        
    # Create Dictionary
    dictionary = corpora.Dictionary(texts)
    
    # Optional: Filter out words that occur in less than 2 documents, or more than 50% of the documents
    # However, for short meeting transcripts, it's safer to not filter too aggressively.
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

if __name__ == "__main__":
    sample = (
        "Hari ini kita akan berbincang mengenai bajet kewangan syarikat untuk tahun depan. "
        "Kita perlu pastikan perbelanjaan dikurangkan dan keuntungan dimaksimumkan. "
        "Selain itu, pengurusan sumber manusia juga harus diberi perhatian khusus dalam memastikan "
        "produktiviti pekerja berada di tahap optimum."
    )
    topics = perform_topic_modeling(sample)
    print("Detected Topics:")
    for t in topics:
        print(f"Topic {t['topic_id']}: {', '.join(t['words'])}")
