import time

class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = time.time()

    def stop(self):
        end_time = time.time()
        elapsed = end_time - self.start_time
        ms = int(elapsed * 1000)
        h = ms // (3600 * 1000)
        m = (ms // (60 * 1000)) % 60
        s = (ms // 1000) % 60
        ms_remain = ms % 1000
        print(f"TIMERTAG {self.name}; {h}h {m}min {s}sec {ms_remain}ms; {ms}ms")

if __name__ == "__main__":
 timer1 = Timer("part1")
 time.sleep(2.345)
 timer1.stop()