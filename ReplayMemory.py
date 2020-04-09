import numpy as np
from PIL import Image
import threading
from collections import deque
import time

class ReplayMemory:
    def __init__(self, batch_size, minimum_memory, memory_limit, num_frames=3, gray_scale=False, normalize=True, dtype=np.float32, num_thread=8, queue_size=8):
        self.batch_size = batch_size
        self.minimum_memory = minimum_memory
        self.memory_limit = memory_limit
        self.num_frames = num_frames
        self.gray_scale = gray_scale
        self.normalize = normalize

        self.episodes = []
        self.current_holding = 0
        self.dtype = dtype

        self.batch_queue = deque([])
        self.queue_size = queue_size

        self.event = threading.Event()
        self.event.clear()

        self.threads = []
        for i in range(num_thread):
            self.threads.append(threading.Thread(target=self.batch_worker, daemon=True, name=f"batch_worker{i}"))
        self.thread_started = False
        self.event.set()

    def have_enough_memory(self):
        if self.current_holding < self.minimum_memory:
            if self.current_holding / self.minimum_memory * 100 % 10 == 0:
                print(f"Memory {self.current_holding / self.minimum_memory * 100}%")

        return self.current_holding > self.minimum_memory
    
    def add_state(self, state, new_episode):
        if new_episode:
            episode = Episode(num_frames=self.num_frames, gray_scale=self.gray_scale, normalize=self.normalize)
            self.episodes.append(episode)
        
        self.episodes[-1].add_state(state)
        
    def add_return(self, action, reward, terminal):
        self.episodes[-1].add_return(action, reward, terminal)

        self.current_holding += 1

        self.remove_old_episode()

    def add_episode(self, episode):
        self.current_holding += len(episode.states)
        self.episodes.insert(-1, episode)

        self.remove_old_episode()
    
    def remove_old_episode(self):
        if self.current_holding > self.memory_limit:
            self.current_holding -= len(self.episodes[0].states)
            self.episodes = self.episodes[1:]

    def get_last_states(self):
        return self.episodes[-1].get_last_states()

    def get_batch(self):
        if not self.thread_started:
            for thread in self.threads: thread.start()
            self.thread_started = True
        elif len(self.batch_queue) == 0:
            print("Queue empty")

        while len(self.batch_queue) == 0: time.sleep(0.1)

        batch_data = self.batch_queue.popleft()

        if len(self.batch_queue) < self.queue_size: self.event.set()

        return batch_data

    def batch_worker(self):
        while True:
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
            
            state_batch = np.array(state_batch).astype(self.dtype)
            action_batch = np.array(action_batch).astype(self.dtype)
            reward_batch = np.array(reward_batch).astype(self.dtype)
            terminal_batch = np.array(terminal_batch).astype(self.dtype)
            next_state_batch = np.array(next_state_batch).astype(self.dtype)

            self.batch_queue.append([state_batch, action_batch, reward_batch, terminal_batch, next_state_batch])
            if len(self.batch_queue) >= self.queue_size: self.event.clear()
            self.event.wait()

class Episode:
    def __init__(self, num_frames=3, gray_scale=False, normalize=True):
        self.states = []
        self.actions = []
        self.rewards = []
        self.terminals = []

        self.num_frames = num_frames
        self.gray_scale = gray_scale
        self.normalize = normalize

        self.have_enough_entry = False
        self.next = "state"

    def add_state(self, state):
        if self.next is not "state": raise ValueError

        if self.gray_scale: 
            state = np.array(Image.fromarray(state).convert("L"))
            state = state.reshape(state.shape + (1,))
        if self.normalize:
            state = (state / 255).astype(np.float16)

        self.states.append(state)
        self.next = "return"

    def add_return(self, action, reward, terminal):
        if self.next is not "return": raise ValueError

        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)

        if len(self.states) >= 2: self.have_enough_entry = True
        self.next = "state"

    def get_last_states(self):
        states = []
        for i in range(len(self.states) - self.num_frames, len(self.states)):
            if i < 0:
                states.append(np.zeros_like(self.states[0]))
            else:
                states.append(self.states[i])
        
        states = np.concatenate(states, axis=-1)
        return states

    def get_random(self):
        index = np.random.randint(0, len(self.states) - 1)

        states = []
        next_states = []

        for i in range(index - self.num_frames + 1, index + 1):
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
        
