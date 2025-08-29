from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese"

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

audio_file = "testedeaudio2.wav"
speech, sample_rate = sf.read(audio_file)

inputs = processor(speech, sampling_rate=sample_rate, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)

transcription = processor.decode(predicted_ids[0])
print("Transcrição:", transcription.lower())
