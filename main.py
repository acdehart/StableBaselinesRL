# This is a sample Python script.
import math
import os
import time
from typing import Tuple

import gym
from sklearn.metrics import log_loss
import nnfs
from matplotlib import pyplot as plt
from nnfs.datasets import spiral_data
from sklearn.preprocessing import KBinsDiscretizer
from stable_baselines import ACER
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv

from Activation import *
from Loss import *


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self):
        pass


def get_soh_class(SoH, all_soh, classes):
    max_soh = max(all_soh)
    for i, percent in enumerate(np.linspace(1.0, 0.0, classes)):
        if SoH/classes>=percent:
            return max_soh
    return 0


def reinforcement():
    np.random.seed(0)
    nnfs.init()

    class_count = 2
    # X, y = spiral_data(10, class_count)

    # s, ds, is, ft
    X = np.array([[0.01, 0, 0.04, 500],
                  [1, 0, 4, 0]])
    y = [0, 1]

    print(X)
    print(y)
    map=['r', 'g', 'b', 'm', 'y', 'c']
    for i, SoH in enumerate(y):
        plt.scatter([X[i][0]], [X[i][1]], color=map[y[i]])
    # plt.plot(X, y)
    plt.grid()
    # plt.show()
    # input("Graph done?")

    # DEFINE NETWORK
    dense1 = Layer_Dense(4, 3)

    activation1 = Activation_ReLU()

    dense2 = Layer_Dense(3, 2)
    activation2 = Activation_Softmax()

    # STEP FORWARD
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # EXAMINE RESULTS
    # loss_function = Loss_Log()
    # loss = loss_function.calculate(activation2.output, y)

    loss = log_loss(y, activation2.output)

    print(f"Loss: ", loss)

    input()

def attempt2_q_learning(env):
    n_bins = (6, 12)
    lower_bounds = [env.observation_space.low[2], -math.radians(50)]
    upper_bounds = [env.observation_space.high[2], math.radians(50)]
    Q_table = np.zeros(n_bins + (env.action_space.n,))

    def discretizer(_, __, angle, pole_velocity) -> Tuple[int, ...]:
        """Convert continues state intro a discrete state"""
        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        est.fit([lower_bounds, upper_bounds])
        return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))

    def policy(state: tuple):
        """Choosing action based on epsilon-greedy policy"""
        return np.argmax(Q_table[state])

    def new_Q_value(reward: float, new_state: tuple, discount_factor=1) -> float:
        """Temperal diffrence for updating Q-value of state-action pair"""
        future_optimal_value = np.max(Q_table[new_state])
        learned_value = reward + discount_factor * future_optimal_value
        return learned_value

    # Adaptive learning of Learning Rate
    def learning_rate(n: int, min_rate=0.01) -> float:
        """Decaying learning rate"""
        return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))

    def exploration_rate(n: int, min_rate=0.1) -> float:
        """Decaying exploration rate"""
        return max(min_rate, min(1, 1.0 - math.log10((n + 1.0) / 25.0)))

    n_episodes = 10000
    count = 0
    for e in range(n_episodes):

        # Siscretize state into buckets
        current_state, done = discretizer(*env.reset()), False

        while done == False:
            count += 1
            print(f"{e}.{count}")

            # policy action
            action = policy(current_state)  # exploit

            # insert random action
            if np.random.random() < exploration_rate(e):
                action = env.action_space.sample()  # explore

            # increment enviroment
            obs, reward, done, _ = env.step(action)
            new_state = discretizer(*obs)

            # Update Q-Table
            lr = learning_rate(e)
            learnt_value = new_Q_value(reward, new_state)
            old_value = Q_table[current_state][action]
            Q_table[current_state][action] = (1 - lr) * old_value + lr * learnt_value

            current_state = new_state

            # Render the cartpole environment
            env.render()


def cartpole_evaluate_save(env, model, model_file):
    evaluate_policy(model, env, n_eval_episodes=10, render=False)
    env.close()
    model.save(model_file)
    return model

    # possible actions!
    # attempt1(env)

    # attempt2_q_learning(env)


def attempt1(env):
    policy = lambda obs: 1
    policy = lambda _, __, ___, tip_velocity: int(tip_velocity > 0)
    for _ in range(5):
        obs = env.reset()
        for _ in range(80):
            actions = policy(*obs)
            obs, reward, done, info = env.step(actions)
            env.render()
            time.sleep(0.05)
    env.close()


def run_once(env, model,  loops=200):
    obs = env.reset()
    for l in range(loops):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()


def cartpole_test():
    global env, day
    env = gym.make('CartPole-v1')
    env = DummyVecEnv([lambda: env])

    model_file = "ACER_model_cartpole"
    if os.path.isfile(model_file + ".zip"):
        print("Loading existing model")
        model = ACER.load(model_file, env=env)
    else:
        print("Creating model from scratch")
        model = ACER('MlpPolicy', env, verbose=1)

    day = 0
    total_time = 0
    while True:
        day += 1
        added_time = 2000
        total_time += added_time
        model.learn(total_timesteps=added_time)
        model = cartpole_evaluate_save(env, model, model_file=model_file)
        print(f"Day {day} with total time {total_time/1000}")
        run_once(env, model, loops=200)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # reinforcement()
    cartpole_test()

    # Need to develop an environment for XGBoost
    # https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai

