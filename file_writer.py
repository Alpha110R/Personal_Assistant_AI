class FileWriter:
    def __init__(self, write_queue, diarization_queue, lock, output_file):
        self.write_queue = write_queue
        self.diarization_queue = diarization_queue
        self.lock = lock
        self.output_file = output_file

    def write(self):
        while True:
            transcription_buffer = self.write_queue.get()
            diarization_buffer = self.diarization_queue.get()
            if transcription_buffer is None or diarization_buffer is None:
                break

            with self.lock:
                with open(self.output_file, "a") as f:
                    # Combine transcription with diarization results
                    for text, (start, end, speaker) in zip(transcription_buffer, diarization_buffer):
                        f.write(f"[{start:.2f}-{end:.2f}] Speaker {speaker}: {text}\n")
                    f.flush()
            self.write_queue.task_done()
            self.diarization_queue.task_done()
