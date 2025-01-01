import threading

class AudioListener:
    def __init__(self, stream, audio_queue, stop_event):
        self.stream = stream
        self.audio_queue = audio_queue
        self.stop_event = stop_event

    def listen(self):
        try:
            while not self.stop_event.is_set():
                data = self.stream.read(512, exception_on_overflow=False)
                self.audio_queue.put(data)
        except Exception as e:
            print(f"Error in audio listening thread: {e}")
            self.audio_queue.put(None)
