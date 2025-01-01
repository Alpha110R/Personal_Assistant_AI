import json
import threading
from vosk import Model, KaldiRecognizer

class TranscriptionManager:
    def __init__(self, model_path, output_file, max_length=500):
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.output_file = output_file
        self.max_length = max_length
        self.buffer_1 = []
        self.buffer_2 = []
        self.active_buffer = self.buffer_1
        self.write_buffer = self.buffer_2
        self.lock = threading.Lock()

    def process_audio(self, data):
        if self.recognizer.AcceptWaveform(data):
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "")
            print(f"Recognized Text: {text}")
            with self.lock:
                self.active_buffer.append(text)

    def manage_buffers(self):
        with self.lock:
            if sum(len(t) for t in self.active_buffer) >= self.max_length:
                self.active_buffer, self.write_buffer = self.write_buffer, self.active_buffer
                threading.Thread(target=self.write_to_file, args=(self.write_buffer,)).start()

    def write_to_file(self, buffer):
        with open(self.output_file, "a") as f:
            f.write(" ".join(buffer).strip() + "\n")
            f.flush()
        buffer.clear()
