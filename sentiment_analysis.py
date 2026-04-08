import malaya

class MalayaSentimentLexicon:
    def __init__(self):
        """
        Initialize the Lexicon-based Sentiment Analyzer from Malaya.
        """
        try:
            # Use multinomial which is standard in current Malaya versions
            self.model = malaya.sentiment.multinomial()
        except AttributeError:
            # Fallback to older lexicon 
            self.model = malaya.sentiment.lexicon()
    def analyze(self, text):
        """
        Predict sentiment of the provided text.
        
        Args:
            text: A preprocessed Malay string.
        
        Returns:
            A string containing the dominant sentiment or a dictionary with polarity scores.
            Malaya's lexicon predict typically returns a string ('positive', 'negative', 'neutral') 
            or a dictionary in predict_proba. We will return the predict result.
        """
        if not text or not text.strip():
            return "neutral"
            
        try:
            # predict accepts a list of texts
            results = self.model.predict([text])
            if results and len(results) > 0:
                return results[0]
            
            return "neutral"
        except Exception as e:
            print(f"[Sentiment Error] {e}")
            return "unknown"


# Provide a quick way to just run analyze if module is used as a utility
_analyzer = None

def analyze_sentiment(text):
    global _analyzer
    if _analyzer is None:
        _analyzer = MalayaSentimentLexicon()
    return _analyzer.analyze(text)

if __name__ == "__main__":
    sample = "Terima kasih atas penerangan yang sangat jelas dan baik."
    result = analyze_sentiment(sample)
    print(f"Sample Text: {sample}\nSentiment: {result}")
