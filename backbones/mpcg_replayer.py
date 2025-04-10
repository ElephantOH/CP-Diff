import random
import bisect

class ReplayBuffer:
    def __init__(self, sample_capacity=30, max_capacity=12000):
        self.sample_capacity = sample_capacity
        self.max_capacity = max_capacity
        self.elite_buffer = []
        self.normal_buffer = []

    def push(self, state, v):
        keys = [-x[0] for x in self.elite_buffer]
        pos = bisect.bisect_left(keys, -v)
        self.elite_buffer.insert(pos, (v, state))
        self.normal_buffer.insert(pos, (v, state))
        if len(self.elite_buffer) > self.sample_capacity:
            self.elite_buffer.pop()

    def sample_all(self, batch_size):
        if len(self.normal_buffer) < batch_size:
            raise ValueError(f"Not enough samples. Buffer size: {self.size()}, requested: {batch_size}")
        batch = random.sample(self.normal_buffer, batch_size)
        vs, states = zip(*batch)
        return states, vs

    def sample_elite(self, batch_size):
        if len(self.elite_buffer) < batch_size:
            raise ValueError(f"Not enough samples. Buffer size: {self.size()}, requested: {batch_size}")
        batch = random.sample(self.elite_buffer, batch_size)
        vs, states = zip(*batch)
        return states, vs

    def size(self):
        return len(self.normal_buffer)

    def __len__(self):
        return min(len(self.elite_buffer), len(self.normal_buffer))
