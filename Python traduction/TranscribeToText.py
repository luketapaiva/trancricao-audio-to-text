import whisper

arquivo_audio = "C:\\Users\\lucas\\OneDrive\\Área de Trabalho\\Python traduction\\testedeaudio2.wav"

model = whisper.load_model("turbo")

result = model.transcribe(arquivo_audio, language="pt")
print(result)

texto = result["text"]

print("transcrição:")
print(texto)

with open("transcricao.txt", "w", encoding="utf-8") as f:
    f.write(texto)

