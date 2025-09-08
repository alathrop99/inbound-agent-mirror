from TTS.api import TTS

# Step 1: Load a pre-trained model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# Step 2: Generate speech and save to file
tts.tts_to_file(
    text="This is a test of the Coqui TTS system. It should sound much more natural.",
    file_path="test_output.wav"
)
