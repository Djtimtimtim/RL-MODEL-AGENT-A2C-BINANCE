import datetime  # Import the datetime module for working with dates and times
import pandas as pd  # Import the pandas library for working with data in tabular form
import numpy as np  # Import the numpy library for working with arrays and mathematical functions
from stable_baselines3 import A2C  # Import the A2C algorithm from the stable_baselines3 library
from stable_baselines3.common.vec_env import DummyVecEnv  # Import the DummyVecEnv class for creating a vectorized environment
import torch  # Import the torch library for working with neural networks
import time  # Import the time module for working with time
import os  # Import the os module for interacting with the operating system
from gym.spaces import MultiDiscrete  # Import the MultiDiscrete class for action spaces
import gymnasium as gym  # Import the gymnasium library for working with environments

# Check the available device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose the device: GPU if available, otherwise CPU
print("Device in use:", device)  # Output information about the selected device

from gym_trading_env.downloader import download  # Import the download function from the custom module

# Download historical data
download(exchange_names=["binance"],
         symbols=["ETH/USDT"],
         dir="data",
         timeframe="5m",
         since=datetime.datetime(year=2024, month=2, day=1))

# Read data from the pickle file
df = pd.read_pickle("./data/binance-ETHUSDT-5m.pkl")

# Calculate new features based on existing ones
df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()
df['volume_change'] = df['volume'].pct_change() / df["close"].pct_change()
df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"]/df["close"]
df["feature_high"] = df["high"]/df["close"]
df["feature_low"] = df["low"]/df["close"]

# Remove rows with missing values
df.dropna(inplace=True)

print(df.columns)

# Indexes in df for passing to the training function in the ENV
INDEX_OPEN = 0  # Index for opening price
INDEX_HIGH = 1  # Index for the highest price
INDEX_LOW = 2   # Index for the lowest price
INDEX_CLOSE = 3 # Index for closing price
INDEX_VOLUME = 4 # Index for trading volume
INDEX_FEATURE_VOLUME = 5 # Index for volume feature
INDEX_VOLUME_CHANGE = 6 # Index for volume change

# Define functions for processing history in the environment
def reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

def dynamic_feature_last_position_taken(history):
    return history['position', -1]

def dynamic_feature_real_position(history):
    return history['real_position', -1]

# Create a reinforcement learning environment
env = gym.make("TradingEnv",
               name="ETHUSDT",
               df=df,
               positions=[-1, 0, 1],
               trading_fees=0,
               borrow_interest_rate=0,
               reward_function=reward_function,
               dynamic_feature_functions=[dynamic_feature_last_position_taken,
                                          dynamic_feature_real_position])

# Add metrics to monitor environment performance
env.unwrapped.add_metric('Position Changes', lambda history: np.sum(np.diff(history['position']) != 0))
env.unwrapped.add_metric('Episode Length', lambda history: len(history['position']))

# Use MultiDiscrete to define action space
action_space = MultiDiscrete([3, 3, 3])

# Wrap the environment in DummyVecEnv for vectorization
env = DummyVecEnv([lambda: env])

# File name for saving the model
model_path = "trading_model_a2c.zip"

# Load existing model or create a new one
if os.path.exists(model_path):
    model = A2C.load(model_path, env=env)
    print("Loaded model from:", model_path)
else:
    model = A2C("MlpPolicy", env, verbose=0, learning_rate=0.0001)
    print("Created a new model with additional layers.")

# Model training parameters
max_episodes = 10000
save_interval = 300
start_time = time.time()

# Train the model
for current_episode in range(max_episodes):
    model.learn(total_timesteps=10000)

    # Save the model at a certain interval
    if time.time() - start_time >= save_interval:
        model.save(model_path)
        print("Model saved at:", time.strftime('%Y-%m-%d %H:%M:%S'))
        start_time = time.time()

    if current_episode + 1 >= max_episodes:
        break

# Reset the environment to start a new episode
initial_observation = env.reset()
print("Initial observation shape:", initial_observation.shape)

done, truncated = False, False
observation = initial_observation
action = None

# Trading process using the trained model
while not done and not truncated:
    observation, reward, done, info = env.step(action)
    env.render()

    if not done and not truncated:
        action, _states = model.predict(observation)

# Save the final version of the model
model.save(model_path)
print("Final model saved at:", time.strftime('%Y-%m-%d %H:%M:%S'))

