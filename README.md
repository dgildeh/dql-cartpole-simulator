# Deep Q-Network (DQN) Reinforcement Learning Agent

This project creates and uses a DQN Agent with reinforcement learning to train the agent to balance
a stick using the [OpenAI Gym](https://gym.openai.com/docs/) Python module.

![Balancing act](https://user-images.githubusercontent.com/25379378/67250610-eb943400-f420-11e9-90aa-299f6b10dc25.gif)

## How It Works

Deep Q-Network (DQN) is the first deep reinforcement learning method proposed by DeepMind.

Essentially the agent tries random actions in an environment (in this case the OpenAI Gym) and based
on the state and reaction to that action, will receive either a positive or negative reward, which
is used to train the model so it can learn which actions receive the highest rewards based on the state
of the current environment:

![Reinforcement Learning](https://www.kdnuggets.com/images/reinforcement-learning-fig1-700.jpg) 

Through multiple training iterations, the model will eventually learn to balance the stick indefinitely.

## Running The Simulation

First install the Python modules required for this project:

```bash
pip install -r requirements.txt
```

Once installed, you can then run the `cartpole_train.py` script in this repository to start the
simulation and training:

```bash
python cartpole_train.py
```

The following command line options are available:

* -h, --help: Display help for command line usage
* -r, --render: Render the CartPole-V1 environment on desktop popup
* -e EPISODES, --episodes EPISODES: Number of training iterations (default=1,000)
* -l LOAD, --load LOAD: Load previously saved model weights from file
* -s SAVE, --save SAVE: Saved model weights to file after training completed

## References

A list of useful resources to help you understand how this model works:

* https://keon.io/deep-q-learning/
* https://towardsdatascience.com/welcome-to-deep-reinforcement-learning-part-1-dqn-c3cab4d41b6b