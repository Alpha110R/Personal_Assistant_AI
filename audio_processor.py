import json

class AudioProcessor:
    def __init__(self, recognizer, audio_queue, active_buffer, lock):
        self.recognizer = recognizer
        self.audio_queue = audio_queue
        self.active_buffer = active_buffer
        self.lock = lock

    def process(self):
        try:
            while True:
                data = self.audio_queue.get()
                if data is None:
                    break
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    print(f"Recognized Text: {text}")
                    with self.lock:
                        self.active_buffer.append(text)
        except Exception as e:
            print(f"Error in audio processing thread: {e}")
