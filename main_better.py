# This is a sample Python script.
import math

import gym
import json
import datetime as dt
import nnfs
from nnfs.datasets import spiral_data

import numpy as np
# from stable_baselines import ACER
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.common.evaluation import evaluate_policy
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2
# from env.StockTradingEnv import StockTradingEnv
import pandas as pd


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from stable_baselines.common.vec_env import DummyVecEnv


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def lander():
    print_hi('LEZGOOOOOO!')
    environment_name = 'LunarLander-v2'
    # env = test_lunar(environment_name)
    # env = train_and_save_lunar(environment_name)
    load_and_run_lunar(environment_name)


def load_and_run_lunar(environment_name):
    env = gym.make(environment_name)
    env = DummyVecEnv([lambda: env])
    model = ACER.load("ACER_model", env=env)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()


def train_and_save_lunar(environment_name):
    env = gym.make(environment_name)
    env = DummyVecEnv([lambda: env])
    model = ACER('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    evaluate_policy(model, env, n_eval_episodes=10, render=True)
    env.close()
    model.save("ACER_model")
    del model
    return env


def test_lunar(environment_name):
    env = gym.make(environment_name)
    episodes = 10
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
    env.close()
    return env


def trader_no_gui():
    global df, env, i
    df = pd.read_csv('./data/AAPL.csv')
    df = df.sort_values('Date')
    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=20000)
    obs = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()


def trader_with_gui():
    global df, env, i
    df = pd.read_csv('./data/MSFT.csv')
    df = df.sort_values('Date')
    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=50)
    obs = env.reset()
    for i in range(len(df['Date'])):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render(mode="live")

def hp_tuner():
    ft = dict()
    ft['learning_rate'] = hyperparam(1, 0.05)
    for hp in ft.items():
        aj = agent(ft).mutate()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self):
        pass


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods



def reinforcement():
    np.random.seed(0)
    nnfs.init()

    X, y = spiral_data(100, 3)

    dense1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()

    dense2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()

    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss_function = Loss_CategoricalCrossentropy()
    loss = loss_function.calculate(activation2.output, y)

    print(f"Loss: ", loss)

    input()
    #
    weights1 = [[0.2, 0.8, -0.5, 1.0],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    biases1 = [2, 3, 0.5]

    weights2 = [[0.1, -0.14, 0.5],
                [-0.5, 0.12, -0.33],
                [-0.44, 0.73, -0.13]]
    biases2 = [-1, 2, -0.5]

    weights1 = np.array(weights1).T
    weights2 = np.array(weights2).T

    HL1_out = np.dot(X, weights1) + biases1
    HL2_out = np.dot(HL1_out, weights2) + biases2

    print(HL2_out)


def softmax_activation():
    layer_outputs = [[4.8, 1.21, 2.85],
                    [8.9, -1.81, 0.2],
                     [1.41, 1.051, 0.026]]

    exp_values = np.exp(layer_outputs)
    norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    print(norm_values)

    print(np.sum(layer_outputs, axis=1))


def reinforcement2():
    X, y = spiral_data(samples=100, classes=3)
    dense1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()

    dense2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()

    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    print(activation2.output)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # lander()
    # trader_no_gui()
    # trader_with_gui()
    # hp_tuner()
    reinforcement()
    # softmax_activation()

class ml_model():
    def __init__(self, feature_dict):
        self = get_regressor()
        self = self.set_params(**feature_dict)
        self.score = 0
        self.eval_stats = dict()

class hyperparam():
    def __init__(self, value, step_size):
        self.lr = value
        self.lr_01 = 0
        self.lr_02 = 0
        self.lr_03 = 0
        self.lr_vel = 0
        self.lr_acc = 0
        self.lr_acc_step = step_size

class agent():
    def __init__(self, feature_dict):
        self.starting_features()
        self.ml_model = ml_model(self.feature_dict)
        self.ml_model_1 = self.ml_model
        self.ml_model_2 = self.ml_model
        self.ml_model_3 = self.ml_model
        self.score = self.evaluate()
        self.score_1 = self.score
        self.score_2 = self.score
        self.score_3 = self.score

    def starting_features(self):
        self.feature_dict = dict()
        self.feature_dict['learning_rate'] = hyperparam(1, 0.05)

    def mutate(self):
        pass
        # shift velocity of random feature 1 unit step

    def evaluate(self):
        _, self.ml_model.eval_stats = train_eval(self.ml_model)
        self.score_3, self.score_2, self.score_1 = self.score_2, self.score_1, self.score
        self.score = self.score_eval_stats(eval_stats)




    def score_eval_stats(self, eval_stats):
        sigs = 1*eval_stats['sig_train'] + 2*eval_stats['sig_test'] + 3*eval_stats['sig_val']
        recalls = 1*eval_stats['err_poor'] + 3*eval_stats['err_fair'] + 3*eval_stats['err_good']
        return 1/sigs**2 + 1/recalls

        #Plug hyperparameters into XGB and get the evaluation metrics


