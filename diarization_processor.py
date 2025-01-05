from pyannote.audio import Pipeline

class DiarizationProcessor:
    def __init__(self, diarization_pipeline, audio_queue, diarization_queue):
        self.diarization_pipeline = diarization_pipeline
        self.audio_queue = audio_queue
        self.diarization_queue = diarization_queue

    def process(self):
        try:
            while True:
                data = self.audio_queue.get()
                if data is None:
                    break
                # Save audio data to a temporary file for diarization
                with open("temp.wav", "wb") as temp_audio:
                    temp_audio.write(data)
                diarization = self.diarization_pipeline("temp.wav")
                results = []
                for segment, _, speaker in diarization.itertracks(yield_label=True):
                    results.append((segment.start, segment.end, speaker))
                self.diarization_queue.put(results)
        except Exception as e:
            print(f"Error in diarization thread: {e}")
