"""
Web Interface for Malay Meeting Summarization
Flask app providing upload + results pages.
"""

import os
import time
import torch
import warnings
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session

# Suppress noisy warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

from preprocess import preprocess_malay_transcript
from extractive import run_extractive
from abstractive import abstractive_summarize, AVAILABLE_MODELS, DEFAULT_MODEL_KEY
from topic_modeling import perform_topic_modeling
from sentiment_analysis import analyze_sentiment

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ── STT (Speech-to-Text) Setup ─────────────────────────────────
_stt_pipe = None


def get_stt_pipeline():
    """Lazy-load the Whisper STT pipeline (GPU-accelerated if available)."""
    global _stt_pipe
    if _stt_pipe is not None:
        return _stt_pipe

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "mesolitica/Malaysian-whisper-large-v3-turbo-v3"

    print(f"[STT] Loading {model_id} on {device} ...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    _stt_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )
    return _stt_pipe


# ── Dropdown options ────────────────────────────────────────────
SUMMARIZATION_MODELS = {
    "textrank":  "TextRank (TF-IDF + PageRank)",
    "lsa":       "LSA (Latent Semantic Analysis)",
    "electra":   "ELECTRA (Deep Semantic Encoder)",
}

SENTIMENT_MODELS = {
    "lexicon": "Lexicon-Based Polarity Analysis",
}

TOPIC_MODELS = {
    "lda": "LDA (Latent Dirichlet Allocation)",
}

ABSTRACTIVE_MODELS = {
    key: f"{key}  —  {info['description']}"
    for key, info in AVAILABLE_MODELS.items()
}


# ── Routes ──────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def upload_page():
    return render_template(
        "upload.html",
        summarization_models=SUMMARIZATION_MODELS,
        sentiment_models=SENTIMENT_MODELS,
        topic_models=TOPIC_MODELS,
        abstractive_models=ABSTRACTIVE_MODELS,
        default_abs=DEFAULT_MODEL_KEY,
    )


@app.route("/process", methods=["POST"])
def process():
    # ── Collect form data ───────────────────────────────────────
    audio = request.files.get("audio_file")
    if not audio or audio.filename == "":
        return redirect(url_for("upload_page"))

    ext_method     = request.form.get("summarization_model", "textrank")
    sentiment_key  = request.form.get("sentiment_model", "lexicon")
    topic_key      = request.form.get("topic_model", "lda")
    abs_model_key  = request.form.get("abstractive_model", DEFAULT_MODEL_KEY)

    # Save audio file
    filename = audio.filename
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    audio.save(save_path)

    start_time = time.time()

    # ── STEP 1  Speech-to-Text ──────────────────────────────────
    print("[Web] Step 1/5 — Transcribing audio …")
    stt = get_stt_pipeline()
    stt_result = stt(
        save_path,
        generate_kwargs={"language": "ms", "task": "transcribe"},
        return_timestamps=True,
    )
    raw_transcript = stt_result["text"]

    # ── STEP 2  Preprocessing ───────────────────────────────────
    print("[Web] Step 2/5 — Preprocessing …")
    cleaned_transcript = preprocess_malay_transcript(raw_transcript, mode="meeting")

    # ── STEP 3  Topic Modeling ──────────────────────────────────
    print("[Web] Step 3/5 — Topic modeling …")
    topics = perform_topic_modeling(cleaned_transcript)

    # ── STEP 4  Sentiment ───────────────────────────────────────
    print("[Web] Step 4/5 — Sentiment analysis …")
    sentiment = analyze_sentiment(cleaned_transcript)

    # ── STEP 5  Extractive → Abstractive ────────────────────────
    print(f"[Web] Step 5/5 — Summarization ({ext_method} → {abs_model_key}) …")
    extractive_result = run_extractive(cleaned_transcript, method=ext_method)
    summary = abstractive_summarize(
        extractive_result["combined"],
        model=abs_model_key,
        mode="beam",
        postprocess=True,
    )

    elapsed = round(time.time() - start_time, 2)

    # ── Build results context ───────────────────────────────────
    results = {
        "filename": filename,
        "raw_transcript": raw_transcript,
        "cleaned_transcript": cleaned_transcript,
        "summary": summary,
        "sentiment": sentiment,
        "topics": topics,
        "extractive_method": SUMMARIZATION_MODELS.get(ext_method, ext_method),
        "abstractive_model": ABSTRACTIVE_MODELS.get(abs_model_key, abs_model_key),
        "sentiment_model": SENTIMENT_MODELS.get(sentiment_key, sentiment_key),
        "topic_model": TOPIC_MODELS.get(topic_key, topic_key),
        "processing_time": elapsed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return render_template("results.html", r=results)


# ── Entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
