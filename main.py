from transcription_manager import TranscriptionManager

if __name__ == "__main__":
    model_path = "vosk-model-en-us-0.22"
    output_file = "transcription.txt"
    manager = TranscriptionManager(model_path, output_file)
    if manager.setup():
        manager.start()
