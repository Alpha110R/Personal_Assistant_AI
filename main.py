import queue

import pyaudio
import json
import os
import threading
from vosk import Model, KaldiRecognizer

def write_to_file(buffer, output_file):
    """Write buffer content to the file."""
    with open(output_file, "a") as f:
        f.write(" ".join(buffer).strip() + "\n")
        f.flush()
    buffer.clear()

def listen_audio(stream, audio_queue):
    """Continuously listen to audio and enqueue data."""
    try:
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            audio_queue.put(data)  # Enqueue audio data
    except Exception as e:
        print(f"Error in audio listening thread: {e}")
        audio_queue.put(None)  # Signal end of listening

def process_audio(recognizer, audio_queue, active_buffer, lock):
    """Process audio data from the queue."""
    try:
        while True:
            data = audio_queue.get()
            if data is None:  # Stop processing if sentinel is received
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                print(f"Recognized Text: {text}")
                with lock:
                    active_buffer.append(text)
    except Exception as e:
        print(f"Error in audio processing thread: {e}")

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
    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,  # Force mono audio to avoid compatibility issues
            rate=16000,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=4000
        )
        stream.start_stream()
    except Exception as e:
        print(f"Error opening audio stream: {e}")
        audio.terminate()
        return

    buffer_1 = []
    buffer_2 = []
    active_buffer = buffer_1
    write_buffer = buffer_2
    max_length = 500  # Max length before writing to file
    lock = threading.Lock()  # Ensure thread safety for buffer operations
    audio_queue = queue.Queue()  # Queue for audio data

    # Start audio listening thread
    listener_thread = threading.Thread(
        target=listen_audio, args=(stream, audio_queue), daemon=True
    )
    listener_thread.start()

    # Start audio processing thread
    processor_thread = threading.Thread(
        target=process_audio, args=(recognizer, audio_queue, active_buffer, lock), daemon=True
    )
    processor_thread.start()

    print("Listening to system audio... Press Ctrl+C to stop.")
    try:
        while True:
            # Switch buffers and write to file in a separate thread
            with lock:
                if sum(len(t) for t in active_buffer) >= max_length:
                    active_buffer, write_buffer = write_buffer, active_buffer
                    threading.Thread(target=write_to_file, args=(write_buffer, output_file)).start()
    except KeyboardInterrupt:
        print("\nStopping transcription...")
        audio_queue.put(None)  # Signal processing thread to stop
        listener_thread.join()  # Ensure listener thread finishes
        processor_thread.join()  # Ensure processor thread finishes
        # Write any remaining text in both buffers to the file
        with lock:
            if active_buffer:
                write_to_file(active_buffer, output_file)
            if write_buffer:
                write_to_file(write_buffer, output_file)
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    real_time_transcription()

