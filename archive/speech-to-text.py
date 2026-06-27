import whisper

model = whisper.load_model("medium", device="cuda")

result = model.transcribe(
    "MeetingSample.mp3",
    language="ms",
    task="transcribe",
    temperature=0,
    condition_on_previous_text=False
)

print(result["text"])