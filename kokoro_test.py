from kokoro import KPipeline
import soundfile as sf
import torch

pipeline = KPipeline(lang_code="a", device="cuda" if torch.cuda.is_available() else "cpu")  # 'a' = American English

text = "Hello! This is Aly from Curve Dental. How can I help you today?"

# Kokoro returns a generator: (greedy_text, phonemes, audio_array)
generator = pipeline(text, voice="af_heart")

# Take the first output from the generator
_, _, audio = next(generator)

# Save to WAV
sf.write("kokoro_test.wav", audio, 24000)

print("✅ Saved to kokoro_test.wav")
