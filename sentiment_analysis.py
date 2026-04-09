"""
Lexicon-Based Sentiment Analysis for Malay Text
Uses a curated polarity lexicon of Malay words to determine
overall sentiment (positive / negative / neutral).
"""

import re


# ── MALAY SENTIMENT LEXICON ──────────────────────────────────────
# Curated list of common Malay positive and negative sentiment words.

POSITIVE_WORDS = {
    # Emotions & Feelings
    'gembira', 'bahagia', 'senang', 'seronok', 'suka', 'cinta', 'sayang',
    'syukur', 'bersyukur', 'bangga', 'kagum', 'teruja', 'semangat',
    'tenang', 'damai', 'aman', 'selesa', 'puas', 'lega', 'harap', 'yakin',
    # Qualities & Attributes
    'baik', 'bagus', 'cantik', 'indah', 'hebat', 'cemerlang', 'gemilang',
    'terbaik', 'unggul', 'bijak', 'pandai', 'rajin', 'tekun', 'gigih',
    'bersih', 'kemas', 'teratur', 'maju', 'berjaya', 'jayanya',
    'positif', 'optimis', 'berkesan', 'efektif', 'produktif',
    # Actions & Outcomes
    'berjaya', 'menang', 'capai', 'mencapai', 'tercapai', 'meningkat',
    'berkembang', 'membangun', 'menyokong', 'membantu', 'menyumbang',
    'menghargai', 'menghormati', 'memperbaiki', 'memuji', 'mengagumi',
    'memperkasa', 'memperkukuh', 'memajukan', 'melindungi',
    # Agreement & Appreciation
    'setuju', 'bersetuju', 'terima kasih', 'tahniah', 'syabas',
    'sempurna', 'lengkap', 'mantap', 'kukuh', 'stabil',
    'harmoni', 'sejahtera', 'makmur', 'adil', 'saksama',
    # Adverbs / Intensifiers (positive context)
    'sangat', 'amat', 'sungguh', 'betul',
}

NEGATIVE_WORDS = {
    # Emotions & Feelings
    'sedih', 'kecewa', 'marah', 'benci', 'takut', 'bimbang', 'risau',
    'gelisah', 'murung', 'putus asa', 'tertekan', 'stress', 'malu',
    'cemburu', 'dengki', 'iri', 'sakit hati', 'duka', 'sengsara',
    # Qualities & Attributes
    'buruk', 'teruk', 'jahat', 'kejam', 'zalim', 'kotor', 'busuk',
    'hodoh', 'lemah', 'bodoh', 'malas', 'cuai', 'gagal', 'rugi',
    'negatif', 'pesimis', 'mundur', 'merosot', 'rosak', 'musnah',
    'bahaya', 'berbahaya', 'beracun', 'merbahaya',
    # Actions & Outcomes
    'gagal', 'kalah', 'jatuh', 'runtuh', 'hancur', 'rosak',
    'menurun', 'merosot', 'merosakkan', 'menghancurkan', 'membunuh',
    'mencuri', 'menipu', 'mengkhianati', 'mengancam', 'menindas',
    'mengeksploitasi', 'menyalahgunakan', 'merugikan', 'menzalimi',
    # Problems & Issues
    'masalah', 'krisis', 'bencana', 'konflik', 'pertikaian',
    'rasuah', 'penipuan', 'jenayah', 'keganasan', 'kemiskinan',
    'pengangguran', 'pencemaran', 'kemusnahan',
    # Negation markers (contribute to negative tone)
    'tidak', 'bukan', 'tiada', 'tanpa', 'jangan',
}


class MalayLexiconSentiment:
    """
    Polarity-based, lexicon-based sentiment analyzer for Malay text.
    Scores text by counting matches against positive and negative word lists.
    """

    def __init__(self, positive_lexicon=None, negative_lexicon=None):
        self.positive_words = positive_lexicon or POSITIVE_WORDS
        self.negative_words = negative_lexicon or NEGATIVE_WORDS

    def analyze(self, text):
        """
        Analyze sentiment of Malay text.

        Args:
            text (str): Preprocessed Malay text.

        Returns:
            dict with keys:
                - sentiment: 'positive', 'negative', or 'neutral'
                - positive_score: count of positive word matches
                - negative_score: count of negative word matches
                - positive_words_found: list of matched positive words
                - negative_words_found: list of matched negative words
        """
        if not text or not text.strip():
            return {
                "sentiment": "neutral",
                "positive_score": 0,
                "negative_score": 0,
                "positive_words_found": [],
                "negative_words_found": [],
            }

        text_lower = text.lower()

        # Find matching positive words
        pos_found = []
        for word in self.positive_words:
            matches = re.findall(rf'\b{re.escape(word)}\b', text_lower)
            pos_found.extend([word] * len(matches))

        # Find matching negative words
        neg_found = []
        for word in self.negative_words:
            matches = re.findall(rf'\b{re.escape(word)}\b', text_lower)
            neg_found.extend([word] * len(matches))

        pos_score = len(pos_found)
        neg_score = len(neg_found)

        # Determine overall sentiment
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


# ── MODULE-LEVEL CONVENIENCE FUNCTION ────────────────────────────

_analyzer = None


def analyze_sentiment(text):
    """
    Quick utility to analyze sentiment of a Malay text string.

    Args:
        text (str): Malay text to analyze.

    Returns:
        dict with sentiment label and scores.
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = MalayLexiconSentiment()
    return _analyzer.analyze(text)


# ── STANDALONE TEST ──────────────────────────────────────────────

if __name__ == "__main__":
    samples = [
        "Terima kasih atas penerangan yang sangat jelas dan baik. Kami sangat gembira.",
        "Keadaan ekonomi semakin teruk dan rakyat kecewa dengan masalah rasuah.",
        "Mesyuarat hari ini membincangkan perkara biasa mengenai jadual kerja.",
    ]

    for text in samples:
        result = analyze_sentiment(text)
        print(f"Text:      {text}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Positive:  {result['positive_score']} ({', '.join(result['positive_words_found'])})")
        print(f"Negative:  {result['negative_score']} ({', '.join(result['negative_words_found'])})")
        print("-" * 60)
