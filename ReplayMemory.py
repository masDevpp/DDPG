import numpy as np
from PIL import Image
import threading
from collections import deque

class ReplayMemory:
    def __init__(self, batch_size, minimum_memory, memory_limit, gray_scale=False, normalize=True):
        self.batch_size = batch_size
        self.minimum_memory = minimum_memory
        self.memory_limit = memory_limit
        self.gray_scale = gray_scale
        self.normalize = normalize

        self.episodes = []
        self.current_holding = 0

        self.batch_queue = deque([])

        self.event = threading.Event()
        self.event.clear()

        self.thread = threading.Thread(target=self.batch_worker, daemon=True, name="batch_worker")
        self.thread_started = False
        self.event.set()

    def have_enough_memory(self):
        if self.current_holding < self.minimum_memory:
            if self.current_holding / self.minimum_memory * 100 % 10 == 0:
                print(f"Memory {self.current_holding / self.minimum_memory * 100}%")

        return self.current_holding > self.minimum_memory
    
    def add_state(self, state, new_episode):
        if new_episode:
            episode = Episode()
            self.episodes.append(episode)

        if self.gray_scale: 
            state = np.array(Image.fromarray(state).convert("L"))
        if self.normalize:
            state = (state / 255).astype(np.float16)
        
        self.episodes[-1].add_state(state)
        
    def add_return(self, action, reward, terminal):
        self.episodes[-1].add_return(action, reward, terminal)

        self.current_holding += 1

        if self.current_holding > self.memory_limit:
            # Remove old episode
            self.current_holding -= len(self.episodes[0].states)
            self.episodes = self.episodes[1:]

    def add_episode(self, episode):
        self.episodes.insert(-1, episode)

    def get_last_states(self):
        return self.episodes[-1].get_last_states()

    def get_batch(self):
        if not self.thread_started:
            self.thread.start()
            self.thread_started = True
        elif len(self.batch_queue) == 0:
            print("Queue empty")

        while len(self.batch_queue) == 0: continue

        batch_data = self.batch_queue.popleft()

        if len(self.batch_queue) == 0: self.event.set()

        return batch_data

    def batch_worker(self):
        if self.current_holding < self.minimum_memory: raise IndexError

        state_batch = []
        action_batch = []
        reward_batch = []
        terminal_batch = []
        next_state_batch = []

        while len(state_batch) < self.batch_size:
            episode = np.random.choice(self.episodes)
            
            if not episode.have_enough_entry: continue

            state, action, reward, terminal, next_state = episode.get_random()
            
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            terminal_batch.append(terminal)
            next_state_batch.append(next_state)
        
        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        terminal_batch = np.array(terminal_batch)
        next_state_batch = np.array(next_state_batch)

        self.batch_queue.append([state_batch, action_batch, reward_batch, terminal_batch, next_state_batch])
        self.event.clear()
        self.event.wait()

class Episode:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.terminals = []

        self.have_enough_entry = False
        self.next = "state"

    def add_state(self, state):
        if self.next is not "state": raise ValueError

        self.states.append(state)
        self.next = "return"

    def add_return(self, action, reward, terminal):
        if self.next is not "return": raise ValueError

        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)

        if len(self.states) >= 2: self.have_enough_entry = True
        self.next = "state"

    def get_last_states(self, num_frames=3):
        states = []
        for i in range(len(self.states) - num_frames, len(self.states)):
            if i < 0:
                states.append(np.zeros_like(self.states[0]))
            else:
                states.append(self.states[i])
        
        states = np.concatenate(states, axis=-1)
        return states

    def get_random(self, num_frames=3):
        index = np.random.randint(0, len(self.states) - 1)

        states = []
        next_states = []

        for i in range(index - num_frames + 1, index + 1):
            if i < 0:
                states.append(np.zeros_like(self.states[0]))
            else:
                states.append(self.states[i])
            
            if i + 1 < 0:
                next_states.append(np.zeros_like(self.states[0]))
            else:
                next_states.append(self.states[i + 1])
        
        states = np.concatenate(states, axis=-1)
        next_states = np.concatenate(next_states, axis=-1)

        return states, self.actions[index], self.rewards[index], self.terminals[index], next_states
        
