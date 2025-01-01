import pyaudio
from vosk import Model, KaldiRecognizer
from audio_listener import AudioListener
from audio_processor import AudioProcessor
from file_writer import FileWriter
import threading
import queue
import os

class TranscriptionManager:
    def __init__(self, model_path, output_file="transcription.txt"):
        self.model_path = model_path
        self.output_file = output_file
        self.audio_queue = queue.Queue()
        self.write_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.buffer_1 = []
        self.buffer_2 = []
        self.active_buffer = self.buffer_1
        self.write_buffer = self.buffer_2
        self.max_length = 500

    def setup_model_and_audio(self):
        if not os.path.exists(self.model_path):
            print("Please download a Vosk model and place it in the 'model' directory.")
            return False

        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)

        self.audio = pyaudio.PyAudio()
        print("Available audio input devices:")
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            print(f"Index {i}: {device_info['name']} - Channels: {device_info['maxInputChannels']}")

        self.input_device_index = int(input("Enter the index of the virtual audio device: "))

        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=512
            )
            self.stream.start_stream()
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.audio.terminate()
            return False

        return True

    def start_threads(self):
        self.listener = AudioListener(self.stream, self.audio_queue, self.stop_event)
        self.processor = AudioProcessor(self.recognizer, self.audio_queue, self.active_buffer, self.lock)
        self.writer = FileWriter(self.write_queue, self.lock, self.output_file)

        self.listener_thread = threading.Thread(target=self.listener.listen, daemon=True)
        self.processor_thread = threading.Thread(target=self.processor.process, daemon=True)
        self.writer_thread = threading.Thread(target=self.writer.write, daemon=True)

        self.listener_thread.start()
        self.processor_thread.start()
        self.writer_thread.start()

    def handle_buffers(self):
        with self.lock:
            if sum(len(t) for t in self.active_buffer) >= self.max_length:
                self.active_buffer, self.write_buffer = self.write_buffer, self.active_buffer
                self.write_queue.put(list(self.write_buffer))
                self.write_buffer.clear()

    def stop_threads(self):
        print("\nStopping transcription...")
        with self.lock:
            if self.active_buffer:
                self.write_queue.put(list(self.active_buffer))
                self.active_buffer.clear()
            if self.write_buffer:
                self.write_queue.put(list(self.write_buffer))
                self.write_buffer.clear()
        self.stop_event.set()
        self.audio_queue.put(None)
        self.write_queue.put(None)

        self.listener_thread.join(timeout=2)
        self.processor_thread.join(timeout=2)
        self.writer_thread.join(timeout=2)

    def cleanup(self):
        if self.listener_thread.is_alive():
            print("Listener thread did not terminate properly.")
        if self.processor_thread.is_alive():
            print("Processor thread did not terminate properly.")
        if self.writer_thread.is_alive():
            print("Writer thread did not terminate properly.")
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def start(self):
        self.start_threads()
        print("Listening to system audio... Press Ctrl+C to stop.")
        try:
            while True:
                self.handle_buffers()
        except KeyboardInterrupt:
            self.stop_threads()
        finally:
            self.cleanup()
