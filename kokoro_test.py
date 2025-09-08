from kokoro import Synthesizer

synth = Synthesizer(language="en")  # Default English model
text = "Hello, thank you for calling Curve Dental. How can I help you today?"

# Generate speech to a WAV file
synth.save_to_file(text, "kokoro_test.wav")

print("✅ Voice generated and saved to kokoro_test.wav")
