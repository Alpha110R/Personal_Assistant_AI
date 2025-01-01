import pyaudio
import json
import os
import threading
import queue
from vosk import Model, KaldiRecognizer

def write_to_file_thread(write_queue, lock, output_file):
    """Dedicated thread to write buffers to the file."""
    while True:
        buffer = write_queue.get()
        if buffer is None:  # Sentinel to stop the thread
            break
        with lock:  # Ensure thread safety during write
            print("\nWriting to file\n")
            with open(output_file, "a") as f:
                f.write(" ".join(buffer).strip() + "\n")
                f.flush()
        write_queue.task_done()  # Signal task completion

def listen_audio(stream, audio_queue, stop_event):
    """Continuously listen to audio and enqueue data."""
    try:
        while not stop_event.is_set():
            data = stream.read(512, exception_on_overflow=False)
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
            frames_per_buffer=512
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
    write_queue = queue.Queue()  # Queue for file-writing tasks
    stop_event = threading.Event()  # Stop event for threads

    # Start audio listening thread
    listener_thread = threading.Thread(
        target=listen_audio, args=(stream, audio_queue, stop_event), daemon=True
    )
    listener_thread.start()

    # Start audio processing thread
    processor_thread = threading.Thread(
        target=process_audio, args=(recognizer, audio_queue, active_buffer, lock), daemon=True
    )
    processor_thread.start()

    # Start file-writing thread
    writer_thread = threading.Thread(
        target=write_to_file_thread, args=(write_queue, lock, output_file), daemon=True
    )
    writer_thread.start()

    print("Listening to system audio... Press Ctrl+C to stop.")
    try:
        while True:
            with lock:
                if sum(len(t) for t in active_buffer) >= max_length:
                    # Switch buffers
                    active_buffer, write_buffer = write_buffer, active_buffer
                    # Enqueue write_buffer for writing
                    write_queue.put(list(write_buffer))  # Copy buffer for writing
                    write_buffer.clear()
    except KeyboardInterrupt:
        print("\nStopping transcription...")
        # Write remaining buffers to file before exiting
        with lock:
            if active_buffer:
                write_queue.put(list(active_buffer))
                active_buffer.clear()
            if write_buffer:
                write_queue.put(list(write_buffer))
                write_buffer.clear()
        # Signal threads to stop
        stop_event.set()  # Signal listener thread to stop
        audio_queue.put(None)  # Signal processing thread to stop
        write_queue.put(None)  # Signal writing thread to stop
        listener_thread.join(timeout=2)
        processor_thread.join(timeout=2)
        writer_thread.join(timeout=2)
    finally:
        if listener_thread.is_alive():
            print("Listener thread did not terminate properly.")
        if processor_thread.is_alive():
            print("Processor thread did not terminate properly.")
        if writer_thread.is_alive():
            print("Writer thread did not terminate properly.")
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    real_time_transcription()
