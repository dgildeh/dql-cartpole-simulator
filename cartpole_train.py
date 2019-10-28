# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import argparse

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):

        # Setup Model Training Hyper Parameters

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        '''
        Creates the Neural Net for Deep-Q Learning Model. Simple 3 layer DNN
        with input size the environment state array size and output the number
        of actions possible in the environment size.

        :return:    The Keras DQN Model for training
        '''
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        '''
        One of the challenges for DQN is that neural network used in the algorithm tends to forget the
        previous experiences as it overwrites them with new experiences. So we need a list of previous
        experiences and observations to re-train the model with the previous experiences. We will call
        this array of experiences memory and use remember() function to append state, action, reward,
        and next state to the memory.

        In our example, the memory list will have a form of:

        memory = [(state, action, reward, next_state, done)...]

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        '''

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        '''
        Load the model weights from disk

        :param name:    The name of the file to load
        '''
        print(f"Loading model weights from file: {name}")
        self.model.load_weights(name)

    def save(self, name):
        '''
        Save the model weights to disk

        :param name:    The name of the file to save
        '''
        print(f"Saving model weights to file: {name}")
        self.model.save_weights(name)


if __name__ == "__main__":

    # Add command line arguments
    parser = argparse.ArgumentParser(description='Runs training simulation for CartPole-v1 using QDL Model.')

    parser.add_argument('-r', '--render', action='store_true',
                        help='Render the CartPole-V1 environment on desktop popup')
    parser.add_argument("-e", "--episodes", type=int, default=EPISODES,
                        help="The number of episodes to train with")
    parser.add_argument("-l", "--load", type=str, default=None,
                        help="Load previously saved model weights from file")
    parser.add_argument("-s", "--save", type=str, default=None,
                        help="Saved model weights to file after training completed")
    args = parser.parse_args()

    # Setup training environment and agent
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Load model wieghts from file if provided in command line
    if args.load != None:
        agent.load(args.load)

    done = False
    batch_size = 32

    # Start training for number of episodes
    for e in range(args.episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):

            if args.render:
                env.render()

            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{args.episodes}, score: {time}, e: {round(agent.epsilon, 2)}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    # Save model wieghts from file if provided in command line
    if args.save != None:
        agent.save(args.save)