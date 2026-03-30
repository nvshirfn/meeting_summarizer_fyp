import re


def preprocess_malay_transcript(text, mode="meeting"):
    """
    Preprocess Malay text for summarization.
    
    Args:
        text: Raw input text
        mode: "meeting" for spoken transcripts (removes fillers, slang, etc.)
              "written" for news/articles (lighter cleanup only)
    
    Returns:
        Cleaned text string
    """
    processed_text = text

    if mode == "meeting":
        # === MEETING/SPOKEN TEXT PROCESSING ===

        # 1. NON-LEXICAL FILLERS
        # "Thinking sounds" with no meaning: "err", "emm", "ahh"
        non_lexical = [r'\berr+\b', r'\bemm+\b', r'\bahh+\b', r'\behh+\b', r'\buhm\b']
        
        # 2. PARTICLES (Emphasis/Social markers)
        # Add flavor to speech but are "noise" for NLP: "lah", "kan", "eh"
        particles = [r'\bkan[.,!?]?\b', r'\blah[.,!?]?\b', r'\beh[.,!?]?\b']

        # 3. TRANSITIONS & "FLUFF" WORDS
        # Words used as fillers or structural bridges with low info value
        fluff_words = [r'\bapa itu\b', r'\bapa tu\b', r'\bright\b']

        # 4. SLANG / INFORMAL WORD REPLACEMENT
        # Normalize informal Malay to standard Malay
        replacements = {
            r'\bkat\b': 'dekat',
            r'\btu\b': 'itu',
            r'\bje\b': 'sahaja',
            r'\btak\b': 'tidak',
            r'\btapi\b': 'tetapi',
            r'\bsikit\b': 'sedikit',
            r'\bni\b': 'ini',
            r'\bso\b': 'jadi',
            r'\bmostly\b': 'kebanyakan',
            r'\bdiorang\b': 'mereka',
            r'\bstill\b': 'masih',
            r'\bbest\b': 'seronok',
            r'\bmacam contoh\b': 'contohnya'
        }

        for pattern, replacement in replacements.items():
            processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)

        # Apply Non-lexical, Particle, and Fluff removal
        all_patterns = non_lexical + particles + fluff_words

        for pattern in all_patterns:
            processed_text = re.sub(pattern, '', processed_text, flags=re.IGNORECASE)

        # 5. REPETITIONS (The "Double Word" Pattern)
        # Remove repeated words: "kita kita" → "kita"
        processed_text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', processed_text)

        # Remove repeated phrases (40+ chars)
        processed_text = re.sub(r'(.{40,}?)\s+\1', r'\1', processed_text)

    # === COMMON CLEANUP (both modes) ===

    # Remove space before punctuation
    processed_text = re.sub(r'\s+([.,!?])', r'\1', processed_text)

    # Remove extra spaces
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()

    return processed_text


def preprocess_file(input_path, output_path=None, mode="meeting"):
    """
    Preprocess a text file and optionally save the result.
    
    Args:
        input_path: Path to the raw input text file
        output_path: Path to save cleaned text (None = don't save)
        mode: "meeting" or "written"
    
    Returns:
        Cleaned text string
    """
    with open(input_path, "r", encoding="utf-8") as f:
        original = f.read()

    cleaned = preprocess_malay_transcript(original, mode=mode)

    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"[Preprocess] Cleaned text saved to: {output_path}")

    return cleaned


# --- STANDALONE EXECUTION ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Malay text for summarization")
    parser.add_argument("--input", required=True, help="Path to raw input text file")
    parser.add_argument("--output", default=None, help="Path to save cleaned text")
    parser.add_argument("--mode", choices=["meeting", "written"], default="meeting",
                        help="Processing mode: 'meeting' for spoken transcripts, 'written' for news/articles")
    
    args = parser.parse_args()

    # Default output path if not specified
    if args.output is None:
        import os
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"cleaned_text/{base}_cleaned.txt"

    cleaned = preprocess_file(args.input, args.output, mode=args.mode)

    # Show stats
    with open(args.input, "r", encoding="utf-8") as f:
        original = f.read()

    original_words = set(original.split())
    cleaned_words = set(cleaned.split())
    removed_words = original_words - cleaned_words

    print(f"\n[Stats] Original words: {len(original.split())} | Cleaned words: {len(cleaned.split())}")
    print(f"[Stats] Unique words removed: {len(removed_words)}")
    if removed_words:
        print(f"[Stats] Removed: {removed_words}")