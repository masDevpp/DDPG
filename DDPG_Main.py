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
BATCH_SIZE = 256#128#32
regularizer_coeff = 1e-2

class EpisodeGenerater():
    def __init__(self, step, max_iteration, epsilon_max_step, actor_variables, critic_variables, parameter_queue, episode_queue, do_render):
        with tf.device('/CPU:0'):
            self.agent = DDPG_Agent([96, 96, 9], 3, regularizer_coeff=regularizer_coeff)
            self.agent.copy_variables(actor_variables, critic_variables)
            #self.agent = agent

            self.step = step
            self.max_iteration = max_iteration
            self.epsilon_max_step = epsilon_max_step
            self.parameter_queue = parameter_queue
            self.episode_queue = episode_queue

            self.env = gym.make("CarRacing-v0", verbose=0)

            self.do_render = do_render
            self.render_freq = 10
            self.sync_count_down_initial = 400
            self.sync_count_down = self.sync_count_down_initial
        
    def run(self):
        with tf.device('/CPU:0'):
            while True:
                episode = Episode(num_frames=9, gray_scale=True, normalize=True)
                state = self.env.reset()
                episode.add_state(state)
                accum_reward = 0

                for i in range(self.max_iteration):
                    progress = self.step / self.epsilon_max_step
                    action = self.agent.predict_with_noise(episode.get_last_states(), progress)
                    action = np.array(action)

                    # Action 0: steer, 1: gas, 2, breake
                    action = [action[0], action[1] / 2 + 0.5, action[2] / 2 + 0.5]
                    
                    state, reward, terminal, _ = self.env.step(action)
                    accum_reward += reward
                    
                    episode.add_return(action, [reward], [terminal])
                    episode.add_state(state)
                    
                    if self.do_render and i % self.render_freq == 0:
                        self.env.render()

                    if terminal: break
                
                self.send_episode(episode, accum_reward)
                self.check_new_parameter()
    
    def check_new_parameter(self):
        if self.parameter_queue.empty():
            self.sync_count_down -= 1
            return
        try:
            step, actor_variables, critic_variables = self.parameter_queue.get()
            self.step = step
            self.agent.copy_variables(actor_variables, critic_variables)
            self.sync_count_down = self.sync_count_down_initial
        finally:
            pass
    
    def send_episode(self, episode, reward):
        self.episode_queue.put([episode, reward])


class Main():
    def __init__(self):
        #with tf.device('/CPU:0'):
        self.agent = DDPG_Agent([96, 96, 9], 3, regularizer_coeff=regularizer_coeff)

        self.cp_managers = []
        for opt, model, name in [[self.agent.actor_optimizer, self.agent.actor, "actor"], [self.agent.critic_optimizer, self.agent.critic, "critic"]]:
            checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
            cp_manager = tf.train.CheckpointManager(checkpoint, os.path.join(LOG_DIR, name), 3, keep_checkpoint_every_n_hours=4)
            checkpoint.restore(cp_manager.latest_checkpoint)
            self.cp_managers.append(cp_manager)

        self.memory = ReplayMemory(BATCH_SIZE, 30000, 300000, num_frames=9, gray_scale=True, normalize=True)
        #self.memory = ReplayMemory(BATCH_SIZE, 1000, 300000, gray_scale=False, normalize=True)

        self.env = gym.make("CarRacing-v0", verbose=0)
        
        self.train_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, "train"))
        self.test_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, "test"))

        self.episode_queue = Queue()
        self.parameter_queues = []
        
        self.do_render = False
        self.render_freq = 10
        self.train_freq = 2
        self.max_iteration = 800
        self.epsilon_max_step = 100000
        self.parameter_send_freq = 1000
    
    def add_parameter_queue(self, queue):
        self.parameter_queues.append(queue)
    
    def send_parameter(self):
        step = self.agent.actor_optimizer.iterations.numpy()
        actor_variables = [v.numpy() for v in self.agent.actor.trainable_variables]
        critic_variables = [v.numpy() for v in self.agent.critic.trainable_variables]

        for queue in self.parameter_queues:
            if queue.empty():
                param = [step, actor_variables, critic_variables]
                queue.put(param)
    
    def check_new_episode(self):
        reward = None
        while not self.episode_queue.empty():
            episode, reward = self.episode_queue.get()
            self.memory.add_episode(episode)
        
        return reward
    
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

    def run2(self):
        #with tf.device('/CPU:0'):
        accum_reward = 0

        while not self.memory.have_enough_memory():
            state = self.env.reset()
            self.memory.add_state(state, True)

            for i in range(self.max_iteration):
                progress = int(self.agent.actor_optimizer.iterations) / self.epsilon_max_step
                action = self.agent.predict_with_noise(self.memory.get_last_states(), progress)
                action = np.array(action)

                # Action 0: steer, 1: gas, 2, breake
                action = [action[0], action[1] / 2 + 0.5, action[2] / 2 + 0.5]
                
                state, reward, terminal, _ = self.env.step(action)
                accum_reward += reward
                
                self.memory.add_return(action, [reward], [terminal])
                self.memory.add_state(state, False)

                if self.do_render and i % self.render_freq == 0:
                    self.env.render()
            
            self.check_new_episode()
            print(f"Memory {self.memory.current_holding / self.memory.minimum_memory * 100:.1f}%")
        
        # Start train if have enough episode in memory
        data_duration = 0
        train_duration = 0
        total_duration_start = time.time()
        with self.train_writer.as_default():
            while True:
                start_time = time.time()
                state_batch, action_batch, reward_batch, terminal_batch, state_next_batch  = self.memory.get_batch()
                data_duration += time.time() - start_time
                
                start_time = time.time()
                actor_loss, critic_loss = self.agent.train(state_batch, action_batch, reward_batch, terminal_batch,state_next_batch)
                train_duration += time.time() - start_time
                
                self.agent.update_target_network()

                step = int(self.agent.actor_optimizer.iterations)

                if step % self.parameter_send_freq == 0:
                    self.send_parameter()
                    r = self.check_new_episode()
                    if r is not None: accum_reward = r
                    print(f"step {step}, reward {accum_reward:.2f}, loss actor {actor_loss:.4f}, critic {critic_loss:.4f}, dur {time.time() - total_duration_start:.2f} ({train_duration:.2f} {data_duration:.2f}), {time.asctime()}")
                    data_duration = 0
                    train_duration = 0
                    total_duration_start = time.time()
                    for mgr in self.cp_managers: mgr.save()


def invoke(step, max_iteration, epsilon_max_step, actor_variables, critic_variables, parameter_queue, episode_queue, do_render):
    ep_gen = EpisodeGenerater(
        step,
        max_iteration, 
        epsilon_max_step, 
        actor_variables, 
        critic_variables, 
        parameter_queue, 
        episode_queue,
        do_render
        )
    ep_gen.run()

if __name__=="__main__":
    main = Main()

    for i in range(2):
        parameter_queue = Queue()
        
        # Convert variables to numpy to not send variables directry
        actor_variables = [v.numpy() for v in main.agent.actor.trainable_variables]
        critic_variables = [v.numpy() for v in main.agent.critic.trainable_variables]

        """
        invoke(
            int(main.agent.actor_optimizer.iterations),
            main.max_iteration, 
            main.epsilon_max_step, 
            agent, 
            parameter_queue, 
            main.episode_queue
        )
        """
        proc = Process(target=invoke, daemon=True, args=(
            main.agent.actor_optimizer.iterations.numpy(),
            1001,
            main.epsilon_max_step, 
            actor_variables, 
            critic_variables,
            parameter_queue, 
            main.episode_queue,
            i == 0
        ))
        
        main.add_parameter_queue(parameter_queue)
        proc.start()
        
    #time.sleep(30)
    main.run2()

    print("asdf")
