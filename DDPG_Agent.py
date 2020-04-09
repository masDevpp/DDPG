import numpy as np
import tensorflow as tf

class DDPG_Agent:
    def __init__(self, input_shape, num_actions, regularizer_coeff=1e-2, gamma=0.99, tau=1e-3):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau

        self.actor = self.build_actor(regularizer_coeff)
        self.critic = self.build_critic(regularizer_coeff)
        self.actor_target = self.build_actor(regularizer_coeff)
        self.critic_target = self.build_critic(regularizer_coeff)

        #self.actor, self.critic = self.build_model()
        #self.actor_target, self.critic_target = self.build_model()

        # Copy variables
        for model, target in [[self.actor, self.actor_target], [self.critic, self.critic_target]]:
            for source, distination in zip(model.trainable_variables, target.trainable_variables):
                distination.assign(source)

        self.actor_optimizer = tf.optimizers.Adam(1e-4)
        self.critic_optimizer = tf.optimizers.Adam(1e-3)

        self.noise_mean = 0.0
        self.noise_sigma = 0.2
        self.noise_remain_step = 0
    
    def build_conv_net(self, regularizer_coeff=1e-2):
        conv_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(self.input_shape),
            tf.keras.layers.Conv2D(32, 4, strides=2, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, 4, strides=2, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, 4, strides=2, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(128, use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU()
        ], name="conv_net")

        return conv_net
    
    def build_actor(self, regularizer_coeff=1e-2):
        conv_net = self.build_conv_net(regularizer_coeff)
        
        actor = tf.keras.Sequential([
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.num_actions, use_bias=False, kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3)),
            tf.keras.layers.BatchNormalization(scale=False)
        ], name="actor_net")(conv_net.output)
        
        actor = tf.keras.activations.tanh(actor)

        return tf.keras.Model(inputs=conv_net.input, outputs=actor)

    def build_critic(self, regularizer_coeff=1e-2):
        actor_value = tf.keras.layers.InputLayer([self.num_actions])

        conv_net = self.build_conv_net(regularizer_coeff)
        
        critic_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(regularizer_coeff)),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(regularizer_coeff), kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
        ], name="critic_net")
        
        critic = critic_net(tf.concat([conv_net.output, actor_value.output], axis=-1))
        
        return tf.keras.Model(inputs=[conv_net.input, actor_value.input], outputs=critic)

    def build_model(self, regularizer_coeff=1e-2):
        # Conv net
        conv_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(self.input_shape),
            tf.keras.layers.Conv2D(32, 4, strides=2, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, 4, strides=2, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, 4, strides=2, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(128, use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU()
        ], name="conv_net")

        # Actor
        actor = tf.keras.Sequential([
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.num_actions, use_bias=False, kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3)),
            tf.keras.layers.BatchNormalization(scale=False)
        ], name="actor_net")(conv_net.output)

        actor = tf.keras.activations.tanh(actor)

        # Critic
        critic = tf.keras.Sequential([
            tf.keras.layers.Dense(64, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(regularizer_coeff)),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(regularizer_coeff), kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
        ], name="critic_net")(tf.concat([conv_net.output, actor], axis=-1))

        actor_model = tf.keras.Model(inputs=conv_net.input, outputs=actor)
        critic_model = tf.keras.Model(inputs=conv_net.input, outputs=critic)

        return actor_model, critic_model

    def copy_variables(self, actor_variables, critic_variables):
        for source, target in zip(actor_variables, self.actor.trainable_variables):
            target.assign(source)

        for source, target in zip(critic_variables, self.critic.trainable_variables):
            target.assign(source)

    def predict(self, state):
        state = tf.expand_dims(state, 0)
        return np.array(self.actor(state)[0])
    
    #tf.function
    def predict_with_noise(self, state, progress):
        state = tf.expand_dims(state, 0)
        action = self.actor(state, training=True)[0]

        if self.noise_remain_step == 0:
            self.noise_remain_step = np.random.randint(20, 200)
            
            if progress > 1.0: progress = 1.0
            mean_sigma = 0.1 + 0.4 * (1.0 - progress)

            self.noise_mean = np.random.normal(0, mean_sigma, 3)

            self.noise_sigma = np.random.normal(0.2, 0.15)
            if self.noise_sigma < 0.0: self.noise_sigma = 0.0
        
        self.noise_remain_step -= 1

        random = np.random.normal(self.noise_mean, self.noise_sigma, 3)
        action = action + random
        action = tf.clip_by_value(action, clip_value_min=-1.0, clip_value_max=1.0)
        
        return action

    def update_target_network(self):
        for model, target in [[self.actor, self.actor_target], [self.critic, self.critic_target]]:
            for source, distination in zip(model.trainable_variables, target.trainable_variables):
                distination.assign(self.tau * source + (1 - self.tau) * distination)

    def train0(self, states, actions, rewards, next_states):
        # Critic
        y = rewards + self.gamma * self.critic_target(next_states)
        with tf.GradientTape() as tape:
            critic_loss = tf.reduce_mean((y - self.critic(states)) ** 2)
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        # Actor
        with tf.GradientTape() as tape:
            actor_loss = tf.reduce_mean(self.critic(states))
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        tf.summary.scalar("critic_loss", critic_loss, self.critic_optimizer.iterations)
        tf.summary.scalar("actor_loss", actor_loss, self.actor_optimizer.iterations)
        #tf.summary.histogram("action", actions_pred, self.actor_optimizer.iterations)

        return actor_loss, critic_loss

    @tf.function
    def train(self, states, actions, rewards, terminals, next_states):
        # Critic
        y = rewards + self.gamma * self.critic_target([next_states, self.actor_target(next_states, training=True)], training=True)
        y = y * (1.0 - terminals) + rewards * terminals
        with tf.GradientTape() as tape:
            q = self.critic([states, actions], training=True)
            critic_loss = tf.reduce_mean(tf.square(y - q))
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        # Actor
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states, training=True)
            actor_loss = -tf.reduce_mean(self.critic([states, actions_pred], training=True))
            #actor_loss = tf.reduce_mean(self.critic([states, actions_pred]) * actions_pred)
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)

        #self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        tf.summary.scalar("critic_loss", critic_loss, self.critic_optimizer.iterations)
        tf.summary.scalar("actor_loss", actor_loss, self.actor_optimizer.iterations)
        tf.summary.histogram("q", q, self.actor_optimizer.iterations)
        tf.summary.histogram("action", actions_pred, self.actor_optimizer.iterations)

        return actor_loss, critic_loss
        
        
if __name__ == "__main__":
    shape = [96, 96, 3]
    model = DDPG_Agent(shape, 3)

    rand = np.random.random([1] + shape)
    a = model.actor(rand)
    c = model.critic([rand, a])

    model.train(rand, np.random.random([1, 3]), np.random.random([1]), rand)

    print("asdf")