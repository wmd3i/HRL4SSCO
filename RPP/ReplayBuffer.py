from collections import namedtuple, deque
import random

# transition
Transition = namedtuple(
    'Transition', ['state', 'context', 'option', 'gain'])
Transition_sub = namedtuple(
    'Transition_sub', ['state', 'context', 'option', 'action', 'reward', 'next_state', 'next_context', 'next_option', 'done'])


class ReplayBuffer:
    def __init__(self, CAPACITY):
        self.memory = deque(maxlen=CAPACITY)

    def push(self, state, context, option, gain):
        self.memory.append(Transition(
            state, context, option, gain))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReplayBuffer_sub:
    def __init__(self, CAPACITY):
        self.memory = deque(maxlen=CAPACITY)

    def push(self, state, context, option, action, reward, next_state, next_context, next_option, done):
        self.memory.append(Transition_sub(
            state, context, option, action, reward, next_state, next_context, next_option, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
