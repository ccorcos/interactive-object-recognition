import time

class Stopwatch:
    @classmethod
    def start(cls, name=None):
      s = cls()
      s.time = time.time()
      s.name = name
      return s
    def elapsed(self):
        if self.time is not None:
          return time.time() - self.time()
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        if name is not None:
            return name + ": " + str(self.elapsed())
        else:
            return str(self.elapsed())
    def stop(self):
        self.time = None
