import pyaudio
import json
import os
import numpy as np
from vosk import Model, KaldiRecognizer

def real_time_transcription(output_file="transcription.txt"):
    # Path to the Vosk model
    model_path = "vosk-model-en-us-0.22"
    if not os.path.exists(model_path):
        print("Please download a Vosk model and place it in the 'model' directory.")
        return

    # Load the Vosk model
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # List audio input devices to identify the virtual audio cable
    print("Available audio input devices:")
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        print(f"Index {i}: {device_info['name']} - Channels: {device_info['maxInputChannels']}")

    # Set the input device index (use your virtual audio cable's index)
    input_device_index = int(input("Enter the index of the virtual audio device: "))

    # Open the audio stream
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,  # Force mono audio to avoid compatibility issues
        rate=16000,
        input=True,
        input_device_index=input_device_index,
        frames_per_buffer=4000
    )
    stream.start_stream()

    accumulated_text = ""
    max_length = 500  # Max length before writing to file

    print("Listening to system audio... Press Ctrl+C to stop.")
    try:
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                print(f"Recognized Text: {text}")
                accumulated_text += text + " "

                # Write to file if accumulated text is long enough
                if len(accumulated_text) >= max_length:
                    with open(output_file, "a") as f:
                        f.write(accumulated_text.strip() + "\n")
                        f.flush()
                    accumulated_text = ""  # Reset the buffer
    except KeyboardInterrupt:
        print("\nStopping transcription...")
        # Write any remaining text to the file
        if accumulated_text.strip():
            with open(output_file, "a") as f:
                f.write(accumulated_text.strip() + "\n")
                f.flush()
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    real_time_transcription()

