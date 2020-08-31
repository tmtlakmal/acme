class Schedule(object):
    def value(self):
        raise  NotADirectoryError

    def increment(self):
        raise NotImplementedError

class LinearSchedule(Schedule):

    def __init__(self, total_steps, eps_fraction=0.6, eps_start=1.0, eps_end=0.02):
        self.eps_steps = total_steps * eps_fraction
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.current_step = 0

    def value(self):
        current_eps = min(float(self.current_step)/self.eps_steps, 1)
        self.increment()
        return self.eps_start + current_eps * (self.eps_end - self.eps_start)

    def increment(self):
        self.current_step += 1