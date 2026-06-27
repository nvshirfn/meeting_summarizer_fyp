import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# 1. Setup Device & Dtype for CUDA
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Using float16 makes it ~2x faster on GPU with almost zero accuracy loss
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 2. Load Mesolitica's specialized Malay model
model_id = "mesolitica/Malaysian-whisper-large-v3-turbo-v3"

print(f"Loading {model_id} onto {device}...")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# 3. Initialize the Pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,      # Required for long audio files
    batch_size=16,          # Uses GPU memory to process chunks in parallel
    torch_dtype=torch_dtype,
    device=device,
)

# 4. Transcribe
# Replace 'culture_shock.mp3' with your actual file path
audio_path = "Audio Trimmed/pengalaman_camping.mp3" 

print("Starting transcription...")
result = pipe(
    audio_path, 
    generate_kwargs={"language": "ms", "task": "transcribe"},
    return_timestamps=True
)

# 5. Output Results
print("\n--- TRANSCRIPT ---\n")
print(result["text"])

# Save to a file in the same folder
with open("stt_transcription_trimmed/pengalaman_camping.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print("\nDone! Transcript saved to 'pengalaman_camping.txt'.")