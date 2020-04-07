import os
import numpy as np
import tensorflow as tf
from DDPG_Agent import DDPG_Agent
from ReplayMemory import ReplayMemory, Episode
import gym
from PIL import Image
import time
from multiprocessing import Process, Queue
from threading import Thread

LOG_DIR = os.path.join(os.getcwd(), "log")
BATCH_SIZE = 32
regularizer_coeff = 1e-2

class EpisodeGenerater():
    def __init__(self, step, max_iteration, epsilon_max_step, actor_variables, critic_variables, parameter_queue, episode_queue):
        with tf.device('/CPU:0'):
            self.agent = DDPG_Agent([96, 96, 9], 3, regularizer_coeff=regularizer_coeff)
            self.agent.copy_variables(actor_variables, critic_variables)

            self.step = step
            self.max_iteration = max_iteration
            self.epsilon_max_step = epsilon_max_step
            self.parameter_queue = parameter_queue
            self.episode_queue = episode_queue

            self.env = gym.make("CarRacing-v0", verbose=0)
        
    def run(self):
        with tf.device('/CPU:0'):
            while True:
                episode = Episode()
                state = self.env.reset()
                episode.add_state(state)

                for i in range(self.max_iteration):
                    progress = self.step / self.epsilon_max_step
                    action = self.agent.predict_with_noise(episode.get_last_states(), progress)

                    # Action 0: steer, 1: gas, 2, breake
                    action = [action[0], action[1] / 2 + 0.5, action[2] / 2 + 0.5]
                    
                    state, reward, terminal, _ = self.env.step(action)
                    accum_reward += reward
                    
                    episode.add_return(action, [reward], [terminal])
                    episode.add_state(state)

                    if terminal: break
                
                self.send_episode(episode)
    
    def check_new_parameter(self):
        try:
            step, actor_variables, critic_variables = self.parameter_queue.get()
            self.step = step
            self.agent.copy_variables(actor_variables, critic_variables)
        finally:
            pass

    def send_episode(self, episode):
        self.episode_queue.put(episode)


def EpisodeGenerater2(max_iteration, epsilon_max_step, get_parameter_func, set_parameter_func):
    with tf.device('/CPU:0'):
        agent = DDPG_Agent([96, 96, 9], 3, regularizer_coeff=regularizer_coeff)

        max_iteration = max_iteration
        epsilon_max_step = epsilon_max_step
        get_parameter_func = get_parameter_func
        set_parameter_func = set_parameter_func

        env = gym.make("CarRacing-v0", verbose=0)

        while True:
            step, actor_variables, critic_variables = get_parameter_func()
            agent.copy_variables(actor_variables, critic_variables)

            episode = Episode()
            state = env.reset()
            episode.add_state(state)

            for i in range(max_iteration):
                progress = step / epsilon_max_step
                action = agent.predict_with_noise(episode.get_last_states(), progress)

                # Action 0: steer, 1: gas, 2, breake
                action = [action[0], action[1] / 2 + 0.5, action[2] / 2 + 0.5]
                
                state, reward, terminal, _ = env.step(action)
                accum_reward += reward
                
                episode.add_return(action, [reward], [terminal])
                episode.add_state(state)

                if terminal: break
            


class Main():
    def __init__(self):
        self.agent = DDPG_Agent([96, 96, 9], 3, regularizer_coeff=regularizer_coeff)

        self.cp_managers = []
        for opt, model, name in [[self.agent.actor_optimizer, self.agent.actor, "actor"], [self.agent.critic_optimizer, self.agent.critic, "critic"]]:
            checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
            cp_manager = tf.train.CheckpointManager(checkpoint, os.path.join(LOG_DIR, name), 3, keep_checkpoint_every_n_hours=4)
            checkpoint.restore(cp_manager.latest_checkpoint)
            self.cp_managers.append(cp_manager)

        self.memory = ReplayMemory(BATCH_SIZE, 20000, 200000, gray_scale=False, normalize=True)

        self.env = gym.make("CarRacing-v0", verbose=0)
        
        self.train_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, "train"))
        self.test_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, "test"))

        self.do_render = True
        self.render_freq = 10
        self.train_freq = 4
        self.max_iteration = 1000
        self.epsilon_max_step = 100000
    
    def get_parameter_func(self):
        def func():
            return int(self.agent.actor_optimizer.iterations), self.agent.actor.trainable_variables, self.agent.critic.trainable_variables
        
        return func

    def set_parameter_func(self):
        def func(episode):
            self.memory.add_episode(episode)
        
        return func

    def run(self):
        state = self.env.reset()
        new_episode = True
        self.memory.add_state(state, new_episode)
        new_episode = False

        data_duration = 0
        train_duration = 0
        accum_reward = 0

        while True:
            with self.train_writer.as_default():
                for i in range(self.max_iteration):
                    progress = int(self.agent.actor_optimizer.iterations) / self.epsilon_max_step
                    action = self.agent.predict_with_noise(self.memory.get_last_states(), progress)

                    # Action 0: steer, 1: gas, 2, breake
                    action = [action[0], action[1] / 2 + 0.5, action[2] / 2 + 0.5]
                    
                    state, reward, terminal, _ = self.env.step(action)
                    accum_reward += reward
                    
                    self.memory.add_return(action, [reward], [terminal])
                    self.memory.add_state(state, new_episode)

                    if self.memory.have_enough_memory() and i % self.train_freq == 0:
                        start_time = time.time()
                        state_batch, action_batch, reward_batch, _, state_next_batch  = self.memory.get_batch()
                        data_duration += time.time() - start_time
                        
                        start_time = time.time()
                        actor_loss, critic_loss = self.agent.train0(state_batch, action_batch, reward_batch, state_next_batch)
                        train_duration += time.time() - start_time
                        
                        self.agent.update_target_network()

                    if self.do_render and i % self.render_freq == 0: 
                        self.env.render()

                    if terminal: break

            if self.memory.have_enough_memory():
                print(f"reward {accum_reward:.2f}, loss actor {actor_loss:.4f}, critic {critic_loss:.4f}, dur {train_duration:.2f} {data_duration:.2f}")
                data_duration = 0
                train_duration = 0
                for mgr in self.cp_managers: mgr.save()

            state = self.env.reset()
            new_episode = True
            self.memory.add_state(state, new_episode)
            new_episode = False


if __name__=="__main__":
    main = Main()
    main.run()

    print("asdf")
