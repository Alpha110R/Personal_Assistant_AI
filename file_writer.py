class FileWriter:
    def __init__(self, write_queue, lock, output_file):
        self.write_queue = write_queue
        self.lock = lock
        self.output_file = output_file

    def write(self):
        while True:
            buffer = self.write_queue.get()
            if buffer is None:
                break
            with self.lock:
                print("\nWriting to file\n")
                with open(self.output_file, "a") as f:
                    f.write(" ".join(buffer).strip() + "\n")
                    f.flush()
            self.write_queue.task_done()
