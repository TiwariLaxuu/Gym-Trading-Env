import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Categorical


# Available in the github repo : examples/data/BTC_USD-Hourly.csv
url = "https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/main/examples/data/BTC_USD-Hourly.csv"
df = pd.read_csv(url, parse_dates=["date"], index_col= "date")
df.sort_index(inplace= True)
df.dropna(inplace= True)
df.drop_duplicates(inplace=True)

# df is a DataFrame with columns : "open", "high", "low", "close", "Volume USD"

# Create the feature : ( close[t] - close[t-1] )/ close[t-1]
df["feature_close"] = df["close"].pct_change()

# Create the feature : open[t] / close[t]
df["feature_open"] = df["open"]/df["close"]

# Create the feature : high[t] / close[t]
df["feature_high"] = df["high"]/df["close"]

# Create the feature : low[t] / close[t]
df["feature_low"] = df["low"]/df["close"]

 # Create the feature : volume[t] / max(*volume[t-7*24:t+1])
df["feature_volume"] = df["Volume USD"] / df["Volume USD"].rolling(7*24).max()

df.dropna(inplace= True) # Clean again !
# Eatch step, the environment will return 5 inputs  : "feature_close", "feature_open", "feature_high", "feature_low", "feature_volume"

df = df.iloc[:60]

def reward_function(history):
        df = pd.DataFrame(history.history_storage, columns=history.columns)
        df.to_csv('history_storage.csv')
        return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])


class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x): 
        x = x.view(-1, 5*7)
        return self.fc(x)
    

# Env 
import gymnasium as gym
import gym_trading_env
env = gym.make(
        "TradingEnv",
        name= "BTCUSD",
        df = df,
        windows= 5,
        # positions = [ -1, -0.5, 0, 0.5, 1, 1.5, 2], # From -1 (=SHORT), to +1 (=LONG)
        positions = [-1, 0, 1],
        initial_position = 'random', #Initial position
        trading_fees = 0.01/100, # 0.01% per stock buy / sell
        borrow_interest_rate= 0.0003/100, #per timestep (= 1h here)
        reward_function = reward_function,
        portfolio_initial_value = 10000, # in FIAT (here, USD)
        max_episode_duration = 50,
    )

env.unwrapped.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )
env.unwrapped.add_metric('Episode Lenght', lambda history : len(history['position']) )
policy = Policy(5*7, 3)
# Run an episode until it ends :
done, truncated = False, False
observation, info = env.reset()
i = 0

while not done and not truncated:
    # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
    state = torch.tensor(observation, dtype=torch.float32)
    action_probs = policy(state)   # random bacause model is not trained
    dist = Categorical(action_probs)
    action = dist.sample()
    observation, reward, done, truncated, info = env.step(action)
    i = i + 1
    print('Count ', i )

# Render
env.save_for_render()