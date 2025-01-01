import pyaudio

class AudioStream:
    def __init__(self, device_index, rate=16000, channels=1, frames_per_buffer=4000):
        self.device_index = device_index
        self.rate = rate
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def start(self):
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.frames_per_buffer
            )
            self.stream.start_stream()
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.audio.terminate()
            raise

    def read(self):
        return self.stream.read(self.frames_per_buffer, exception_on_overflow=False)

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
