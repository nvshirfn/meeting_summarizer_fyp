import re


NORMALIZATION_OPTIONS = {
    "dictionary": "Dictionary-Based Mapping",
    "model": "Model-Based Normalization",
    "hybrid": "Hybrid (Dictionary + Model)",
}

# ── Model-based normalization state ────────────────────────────
_malaya_normalizer = None
_malaya_normalizer_error = None
_malaya_rules_regex = None   # compiled regex of malaya's 3,374-entry slang dict
_malaya_rules_map = None     # lowercased lookup: informal → formal


def _get_malaya_normalizer():
    """Lazy-load malaya.normalizer.rules with a probabilistic spelling model."""
    global _malaya_normalizer, _malaya_normalizer_error
    if _malaya_normalizer is not None:
        return _malaya_normalizer
    if _malaya_normalizer_error is not None:
        raise RuntimeError(_malaya_normalizer_error)
    try:
        import malaya
        speller = malaya.spelling_correction.probability.load()
        _malaya_normalizer = malaya.normalizer.rules.load(speller=speller)
        return _malaya_normalizer
    except Exception as exc:
        _malaya_normalizer_error = str(exc)
        raise


def _get_malaya_rules_regex():
    """
    Build (once) a single compiled regex from malaya.preprocessing.rules_normalizer.
    That dict has 3,374 Malay slang→formal entries — far more comprehensive than
    the hand-written replacements dict. Longer phrases are matched first so
    multi-word entries (e.g. 'tak payah') beat their sub-words ('tak').
    """
    global _malaya_rules_regex, _malaya_rules_map
    if _malaya_rules_regex is not None:
        return _malaya_rules_regex, _malaya_rules_map
    import malaya
    raw = malaya.preprocessing.rules_normalizer          # dict {informal: formal}
    sorted_items = sorted(raw.items(), key=lambda x: len(x[0]), reverse=True)
    _malaya_rules_map = {k.lower(): v for k, v in sorted_items}
    pattern = '|'.join(rf'\b{re.escape(k)}\b' for k in _malaya_rules_map)
    _malaya_rules_regex = re.compile(pattern, re.IGNORECASE)
    return _malaya_rules_regex, _malaya_rules_map


def _apply_model_normalization(text):
    """
    Two-pass normalization using Malaya's built-in resources:

    Pass 1 — malaya.preprocessing.rules_normalizer (3,374 entries):
        Replaces informal/slang Malay words with their formal equivalents using
        a single compiled regex (efficient, whole-word matching, longest-match).

    Pass 2 — malaya.normalizer.rules + probability speller:
        Corrects remaining misspellings using a probabilistic Malay language
        model, then applies Malaya's morphological rules on top.

    Falls back to the unchanged text if Malaya cannot be loaded.
    """
    if not text:
        return text
    try:
        # Pass 1: comprehensive slang replacement
        regex, rules_map = _get_malaya_rules_regex()
        text = regex.sub(lambda m: rules_map[m.group(0).lower()], text)

        # Pass 2: spelling correction + morphological rules
        normalizer = _get_malaya_normalizer()
        result = normalizer.normalize(text)
        if isinstance(result, dict):
            return result.get("normalize", text)
        return str(result)
    except Exception as exc:
        print(f"[Preprocess] Model normalization unavailable; using current text. Reason: {exc}")
        return text


def preprocess_malay_transcript(text, mode="meeting", lowercase=False, normalization="dictionary"):
    """
    Preprocess Malay text for summarization.
    
    Args:
        text: Raw input text
        mode: "meeting" for spoken transcripts (removes fillers, slang, etc.)
              "written" for news/articles (lighter cleanup only)
        lowercase: Whether to convert the text to lowercase (be careful using this with Abstractive models)
        normalization: "dictionary", "model", or "hybrid"
    
    Returns:
        Cleaned text string
    """
    if normalization not in NORMALIZATION_OPTIONS:
        raise ValueError(
            f"Unsupported normalization '{normalization}'. "
            f"Choose one of: {', '.join(NORMALIZATION_OPTIONS)}"
        )

    processed_text = text

    if mode == "meeting":
        # === MEETING/SPOKEN TEXT PROCESSING ===

        # 1. REMOVE WORDS (Fillers, Particles, Interjections, Vocatives)
        remove_list = [
            'ah', 'aa', 'ahh', 'alah', 'aiyoh', 'aiyaya', 'bruh', 'beb', 'babe', 
            'ceh', 'ehh', 'err', 'emm', 'fuh', 'ha', 'haa', 'halah', 'hekeleh', 
            'hm', 'hmm', 'kan', 'lah', 'mat', 'ouh', 'peh', 'uhm', 'wuih'
        ]
        remove_patterns = [rf'\b{word}[.,!?]?\b' for word in remove_list]

        # 2. SLANG / INFORMAL / ENGLISH WORD REPLACEMENT
        replacements = {
            r'\bambik\b': 'ambil',
            r'\badoi\b': 'aduh',
            r'\bakak\b': 'kak',
            r'\basal\b': 'kenapa',
            r'\bapasal\b': 'kenapa',
            r'\baje\b': 'sahaja',
            r'\bbantai\b': 'hentam',
            r'\bbagitahu\b': 'beritahu',
            r'\bbagitau\b': 'beritahu',
            r'\bbagitahulah\b': 'beritahulah',
            r'\bbagitaulah\b': 'beritahulah',
            r'\bcamne\b': 'macam mana',
            r'\bcamni\b': 'macam ini',
            r'\bcamtu\b': 'macam itu',
            r'\bcenggitu\b': 'macam itu',
            r'\bcenggini\b': 'macam ini',
            r'\bcite\b': 'cerita',
            r'\bciter\b': 'cerita',
            r'\bcokia\b': 'biasa-biasa',
            r'\bdiorang\b': 'mereka',
            r'\bdia orang\b': 'mereka',
            r'\bdah\b': 'sudah',
            r'\bduk\b': 'duduk',
            r'\bdulu\b': 'dahulu',
            r'\bgi\b': 'pergi',
            r'\bgak\b': 'juga',
            r'\bgostan\b': 'undur',
            r'\bje\b': 'sahaja',
            r'\bjom\b': 'mari',
            r'\bjomlah\b': 'marilah',
            r'\bjap\b': 'sekejap',
            r'\bkejap\b': 'sekejap',
            r'\bkat\b': 'dekat',
            r'\bkarok\b': 'karaoke',
            r'\bkeuangan\b': 'kewangan',
            r'\bkorang\b': 'kamu semua',
            r'\bkitorang\b': 'kami',
            r'\bkitaorang\b': 'kami',
            r'\bkita orang\b': 'kami',
            r'\bkecek\b': 'cakap',
            r'\bkot\b': 'agaknya',
            r'\bko\b': 'kau',
            r'\bkasi\b': 'beri',
            r'\bkesehatan\b': 'kesihatan',
            r'\blast-last\b': 'akhirnya',
            r'\blast last\b': 'akhirnya',
            r'\blu\b': 'dahulu',
            r'\blaki\b': 'lelaki',
            r'\bleklok\b': 'elok-elok',
            r'\bmacam contoh\b': 'contohnya',
            r'\bnego\b': 'runding',
            r'\bni\b': 'ini',
            r'\bnak\b': 'mahu',
            r'\bnaklah\b': 'mahu',
            r'\bno hal\b': 'tiada masalah',
            r'\bngam\b': 'sesuai',
            r'\bngam-ngam\b': 'tepat-tepat',
            r'\bnak-nak\b': 'lebih-lebih lagi',
            r'\bok\b': 'okey',
            r'\bokay\b': 'okey',
            r'\botomatik\b': 'automatik',
            r'\bpas\b': 'selepas',
            r'\bpastu\b': 'selepas itu',
            r'\bpi\b': 'pergi',
            r'\bpayah\b': 'susah',
            r'\bpape\b': 'apa-apa',
            r'\brilek\b': 'bertenang',
            r'\bsikit\b': 'sedikit',
            r'\bskang\b': 'sekarang',
            r'\bsat\b': 'sekejap',
            r'\bsaja\b': 'sahaja',
            r'\bsaje\b': 'sahaja',
            r'\bstylo\b': 'bergaya',
            r'\bsehat\b': 'sihat',
            r'\bsetel\b': 'selesai',
            r'\btu\b': 'itu',
            r'\btak\b': 'tidak',
            r'\btakkan\b': 'tidak mungkin',
            r'\btakkanlah\b': 'tidak mungkin',
            r'\btakde\b': 'tidak ada',
            r'\btakpe\b': 'tidak mengapa',
            r'\btakyah\b': 'tidak perlu',
            r'\btak payah\b': 'tidak perlu',
            r'\btak payahlah\b': 'tidak perlulah',
            r'\btapi\b': 'tetapi',
            r'\btau\b': 'tahu',
            r'\bterkejut beruk\b': 'sangat terkejut',
            r'\bterer\b': 'hebat',
            r'\busya\b': 'tengok',
            r'\busya-usya\b': 'tengok-tengok',
            r'\buang\b': 'wang',
            r'\banyways\b': 'walau bagaimanapun',
            r'\band\b': 'dan',
            r'\balright\b': 'baiklah',
            r'\bbest\b': 'seronok',
            r'\bblur\b': 'keliru',
            r'\bboring\b': 'bosan',
            r'\bbimbo\b': 'wanita kurang cerdik',
            r'\bchill\b': 'santai',
            r'\bfrust\b': 'kecewa',
            r'\bfor example\b': 'sebagai contoh',
            r'\bmostly\b': 'kebanyakan',
            r'\bmember\b': 'kawan',
            r'\boutstation\b': 'kerja di luar kawasan',
            r'\bport\b': 'tempat',
            r'\bpower\b': 'hebat',
            r'\brelax\b': 'bertenang',
            r'\bso\b': 'jadi',
            r'\bstill\b': 'masih',
            r'\bsound\b': 'marah',
            r'\bserious\b': 'serius',
            r'\bsettle\b': 'selesai',
            r'\bterror\b': 'hebat',
            r'\bvibe\b': 'suasana'
        }

        if normalization in ("dictionary", "hybrid"):
            for pattern, replacement in replacements.items():
                processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)

        if normalization in ("model", "hybrid"):
            processed_text = _apply_model_normalization(processed_text)

        for pattern in remove_patterns:
            processed_text = re.sub(pattern, '', processed_text, flags=re.IGNORECASE)

        # 5. REPETITIONS (The "Double Word" Pattern)
        # Remove repeated words: "kita kita" → "kita"
        processed_text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', processed_text)

        # Remove repeated phrases (40+ chars)
        processed_text = re.sub(r'(.{40,}?)\s+\1', r'\1', processed_text)

    # === COMMON CLEANUP (both modes) ===

    # 1. MULTIPLE PUNCTUATION CLEANUP
    # Collapse repeating punctuation (e.g., ... -> ., !!! -> !, ??? -> ?)
    processed_text = re.sub(r'\.{2,}', '.', processed_text)
    processed_text = re.sub(r'!{2,}', '!', processed_text)
    processed_text = re.sub(r'\?{2,}', '?', processed_text)

    # 2. SPECIAL CHARACTER REMOVAL
    # Match any character that is NOT alphanumeric, whitespace, or standard punctuation
    processed_text = re.sub(r'[^\w\s.,!?\'"-]', '', processed_text)

    # 3. Remove space before punctuation
    processed_text = re.sub(r'\s+([.,!?])', r'\1', processed_text)

    # 4. Remove extra spaces
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()

    # 5. LOWERCASING
    if lowercase:
        processed_text = processed_text.lower()

    if mode != "meeting" and normalization in ("model", "hybrid"):
        processed_text = _apply_model_normalization(processed_text)
        processed_text = re.sub(r'\s+([.,!?])', r'\1', processed_text)
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        if lowercase:
            processed_text = processed_text.lower()

    return processed_text


def preprocess_file(input_path, output_path=None, mode="meeting", lowercase=False, normalization="dictionary"):
    """
    Preprocess a text file and optionally save the result.
    
    Args:
        input_path: Path to the raw input text file
        output_path: Path to save cleaned text (None = don't save)
        mode: "meeting" or "written"
        lowercase: Whether to convert text to lowercase
        normalization: "dictionary", "model", or "hybrid"
    
    Returns:
        Cleaned text string
    """
    with open(input_path, "r", encoding="utf-8") as f:
        original = f.read()

    cleaned = preprocess_malay_transcript(
        original,
        mode=mode,
        lowercase=lowercase,
        normalization=normalization,
    )

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
    parser.add_argument("--normalization", choices=list(NORMALIZATION_OPTIONS.keys()), default="dictionary",
                        help="Normalization strategy: dictionary, model, or hybrid")
    parser.add_argument("--lowercase", action="store_true", help="Enable lowercasing of text")
    
    args = parser.parse_args()

    # Default output path if not specified
    if args.output is None:
        import os
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"cleaned_text/{base}_cleaned.txt"

    cleaned = preprocess_file(
        args.input,
        args.output,
        mode=args.mode,
        lowercase=args.lowercase,
        normalization=args.normalization,
    )

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
