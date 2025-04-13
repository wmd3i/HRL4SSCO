from collections import namedtuple, deque
import random

# transition
Transition = namedtuple(
    'Transition', ['state', 'option', 'gain', 'next_state', 'done'])
Transition_sub = namedtuple(
    'Transition_sub', ['state', 'option', 'action', 'reward', 'next_state', 'next_option', 'done'])


class ReplayBuffer:
    def __init__(self, CAPACITY):
        self.memory = deque(maxlen=CAPACITY)

    def push(self, state, option, gain, next_state, done):
        self.memory.append(Transition(
            state, option, gain, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReplayBuffer_sub:
    def __init__(self, CAPACITY):
        self.memory = deque(maxlen=CAPACITY)

    def push(self, state, option, action, reward, next_state, next_option, done):
        self.memory.append(Transition_sub(
            state, option, action, reward, next_state, next_option, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
