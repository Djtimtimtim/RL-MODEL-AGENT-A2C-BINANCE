import datetime  # Импортируем модуль datetime для работы с датами и временем
import pandas as pd  # Импортируем библиотеку pandas для работы с данными в виде таблиц
import numpy as np  # Импортируем библиотеку numpy для работы с массивами и математическими функциями
from stable_baselines3 import A2C  # Импортируем алгоритм A2C из библиотеки stable_baselines3
from stable_baselines3.common.vec_env import DummyVecEnv  # Импортируем класс DummyVecEnv для создания векторизованной среды
import torch  # Импортируем библиотеку torch для работы с нейронными сетями
import time  # Импортируем модуль time для работы со временем
import os  # Импортируем модуль os для взаимодействия с операционной системой
from gym.spaces import MultiDiscrete  # Импортируем класс MultiDiscrete для пространства действий
import gymnasium as gym  # Импортируем библиотеку gymnasium для работы с средами

# Проверка доступного устройства для обучения
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Выбираем устройство: GPU если доступен, иначе CPU
print("Device in use:", device)  # Выводим информацию о выбранном устройстве

from gym_trading_env.downloader import download  # Импортируем функцию download из пользовательского модуля

# Загрузка исторических данных
download(exchange_names=["binance"],
         symbols=["ETH/USDT"],
         dir="data",
         timeframe="5m",
         since=datetime.datetime(year=2024, month=2, day=1))

# Чтение данных из файла pickle
df = pd.read_pickle("./data/binance-ETHUSDT-5m.pkl")

# Вычисление новых признаков на основе существующих
df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()
df['volume_change'] = df['volume'].pct_change() / df["close"].pct_change()
df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"]/df["close"]
df["feature_high"] = df["high"]/df["close"]
df["feature_low"] = df["low"]/df["close"]

# Удаление строк с отсутствующими значениями
df.dropna(inplace=True)

print(df.columns)

# Индексы в df для передачи в функцию обучения в среде ENV
INDEX_OPEN = 0  # Индекс для открытия цены
INDEX_HIGH = 1  # Индекс для наивысшей цены
INDEX_LOW = 2   # Индекс для самой низкой цены
INDEX_CLOSE = 3 # Индекс для закрытия цены
INDEX_VOLUME = 4 # Индекс для объема торгов
INDEX_FEATURE_VOLUME = 5 # Индекс для признака объема
INDEX_VOLUME_CHANGE = 6 # Индекс для изменения объема

# Определение функций для обработки истории в среде
def reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

def dynamic_feature_last_position_taken(history):
    return history['position', -1]

def dynamic_feature_real_position(history):
    return history['real_position', -1]

# Создание среды для обучения с подкреплением
env = gym.make("TradingEnv",
               name="ETHUSDT",
               df=df,
               positions=[-1, 0, 1],
               trading_fees=0,
               borrow_interest_rate=0,
               reward_function=reward_function,
               dynamic_feature_functions=[dynamic_feature_last_position_taken,
                                          dynamic_feature_real_position])

# Добавление метрик для отслеживания производительности среды
env.unwrapped.add_metric('Position Changes', lambda history: np.sum(np.diff(history['position']) != 0))
env.unwrapped.add_metric('Episode Length', lambda history: len(history['position']))

# Используем MultiDiscrete для определения пространства действий
action_space = MultiDiscrete([3, 3, 3])

# Обертываем среду в DummyVecEnv для векторизации
env = DummyVecEnv([lambda: env])

# Имя файла для сохранения модели
model_path = "trading_model_a2c.zip"

# Загрузка существующей модели или создание новой
if os.path.exists(model_path):
    model = A2C.load(model_path, env=env)
    print("Loaded model from:", model_path)
else:
    model = A2C("MlpPolicy", env, verbose=0, learning_rate=0.0001)
    print("Created a new model with additional layers.")

# Параметры обучения модели
max_episodes = 10000
save_interval = 300
start_time = time.time()

# Обучение модели
for current_episode in range(max_episodes):
    model.learn(total_timesteps=10000)

    # Сохранение модели с определенной периодичностью
    if time.time() - start_time >= save_interval:
        model.save(model_path)
        print("Model saved at:", time.strftime('%Y-%m-%d %H:%M:%S'))
        start_time = time.time()

    if current_episode + 1 >= max_episodes:
        break

# Сброс среды для начала нового эпизода
initial_observation = env.reset()
print("Initial observation shape:", initial_observation.shape)

done, truncated = False, False
observation = initial_observation
action = None

# Процесс торговли с использованием обученной модели
while not done and not truncated:
    observation, reward, done, info = env.step(action)
    env.render()

    if not done and not truncated:
        action, _states = model.predict(observation)

# Сохранение окончательной версии модели
model.save(model_path)
print("Final model saved at:", time.strftime('%Y-%m-%d %H:%M:%S'))
