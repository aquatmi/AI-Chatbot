# CartPole Deep Q Learning
# this was based off the following tutorial
# https://pylessons.com/CartPole-reinforcement-learning/
# i have slightly modified the code to help with my understanding
# all comments are my own

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop

ENV_NAME = 'CartPole-v1'
FILE_NAME = 'cartpole-deepq.h5'


def compile_model(input_shape, action_space):
    inputs = Input(input_shape)

    outputs = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(inputs)
    outputs = Dense(256, activation="relu", kernel_initializer='he_uniform')(outputs)
    outputs = Dense(64, activation="relu", kernel_initializer='he_uniform')(outputs)
    outputs = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(outputs)

    model = Model(inputs=inputs, outputs=outputs, name='CartPole DQN model')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    return model


class CartPoleAgent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        # CartPole has state shape of Box(4,),and a action shape of Discrete(2).
        # This will never change, and could be hard coded, but these lines of code can be
        # transferred to any environment, however, for example, the pendulum environment's action shape is Box(1,)
        # so the action size formula would need to be in the form of the state size one for this environment.
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.EPISODES = 1000
        # a deque is a list which can be added to from either end with .append and .appendleft
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # chance of random action
        self.epsilon_min = 0.001  # minimum chance of random action
        self.epsilon_decay = 0.999  # epsilon multiplier for every timescale
        self.batch_size = 64
        self.train_start = 1000  # how much memory data the network should have before non-random learning

        # create main model
        self.model = compile_model(input_shape=(self.state_size,), action_space=self.action_size)

    def remember(self, state, action, reward, next_state, done):
        # save to memory previous timeframes to help ai learn which actions returned a good reward
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:  # if enough memory gathered
            if self.epsilon > self.epsilon_min:     # and epsilon isn't at minimum
                self.epsilon *= self.epsilon_decay      # reduce epsilon

    def act(self, state):
        if np.random.random() <= self.epsilon:
            # random action
            return random.randrange(self.action_size)
        else:
            # calculated action
            return np.argmax(self.model.predict(state))

    def replay(self):
        # after every episode, train the ai with what was learnt in previous episode
        if len(self.memory) < self.train_start:
            return
        # gather a random sample of memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # train model with minibatch
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):  # load model
        self.model = load_model(name)

    def save(self):  # save model
        self.model.save(FILE_NAME)

    def prep(self):  # open and close window to make open in foreground
        self.env.reset()
        self.env.render()
        self.env.close()
    
    def train(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps - 1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    if i == 500:
                        print("Saving trained model as cartpole-deepq.h5")
                        self.save()
                        return
                self.replay()

    def test(self, show=True):
        self.load()
        wins = 0
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                if show:
                    self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}/500".format(e, self.EPISODES, i))
                    if i == 500:
                        wins += 1
                    break
        print("in {} episodes, model won {} times".format(self.EPISODES, wins))

    def play(self, show=True, episodes=1, n=FILE_NAME):
        self.load(n)
        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                if show:
                    self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("score: {}/500".format(i))
                    break
            self.env.close()

if __name__ == "__main__":
    agent = CartPoleAgent()
    # agent.train()
    agent.test(show=False)

