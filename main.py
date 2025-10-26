# Data Preprocessing
import pandas as pd
from pandas_datareader.data import DataReader
from ta.volume import VolumeWeightedAveragePrice

# Environment
import gym
from gym import spaces
import numpy as np
import random
import torch

# PyTorch
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import yfinance as yf

# Outputs
import matplotlib.pyplot as plt

# ===========================
# Data download & features
# ===========================
df = yf.download("AAPL", start="2017-01-01", end="2025-01-01", auto_adjust=False)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

# Add VWAP to DataFrame
vwap = VolumeWeightedAveragePrice(high=df["High"], low=df["Low"], close=df["Close"], 
                                  volume=df["Volume"], window=14, fillna=False)
df["VWAP"] = vwap.volume_weighted_average_price()
df.dropna(inplace=True)
df.head()

df_mod = df.copy()

# проценты / нормализация
rets = df_mod.pct_change() * 100
rets = rets.replace([np.inf, -np.inf], np.nan).dropna()

scaled = rets / rets.max()   # или min-max
# цена для графиков/анализа
scaled["Close_Price"] = df.loc[scaled.index, "Close"].astype(float)

df_mod = scaled   # тут сохраняется DatetimeIndex!


df_mod = df.copy()

# проценты / нормализация
rets = df_mod.pct_change() * 100
rets = rets.replace([np.inf, -np.inf], np.nan).dropna()

scaled = rets / rets.max()   # или min-max
# цена для графиков/анализа
scaled["Close_Price"] = df.loc[scaled.index, "Close"].astype(float)

df_mod = scaled   # тут сохраняется DatetimeIndex!

df_train = df_mod.loc["2017-01-01":"2021-12-31"].copy()
df_test  = df_mod.loc["2022-01-01":"2024-12-31"].copy()


df_train_env = df_train.reset_index(drop=True).copy()
df_test_env  = df_test.reset_index(drop=True).copy()

# Initialise variables
MAX_INT = 2147483647
MAX_TRADES = 10000
MAX_OPEN_POSITIONS = 1
INITIAL_ACCOUNT_BALANCE = 1000
PERCENT_CAPITAL = 0.3
TRADING_COSTS_RATE = 0.001
KILL_THRESH = 0.4 # Threshold for balance preservation


# Structure environment
class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        
        # Generic variables
        self.df = df
        
        # Account variables
        self.available_balance = INITIAL_ACCOUNT_BALANCE
        self.net_profit = 0
        
        # Position variables
        self.num_trades_long = 0
        self.num_trades_short = 0
        self.long_short_ratio = 0
        
        # Current Step
        self.current_step = 0
        self.lag = 20
        self.volatility = 1
        self.max_steps = len(df)

        # Actions of the format Long, Hold, Close
        self.action_space = spaces.Discrete(2)

        # Prices contains the Close and Close Returns etc
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7, ), dtype=np.float32)

    # Calculate Reward
    def _calculate_reward(self):
        reward = 0
        reward += self.net_profit / self.volatility
        reward += 0.01 if self.long_short_ratio >= 0.3 and self.long_short_ratio <= 0.6 else -0.01
        return reward
        
    # Structure sign observation data
    def _next_observation(self):
        
        item_0_T0 = self.df.loc[self.current_step - 0, "Open"].item()
        item_1_T0 = self.df.loc[self.current_step - 0, "High"].item()       
        item_2_T0 = self.df.loc[self.current_step - 0, "Low"].item()
        item_3_T0 = self.df.loc[self.current_step - 0, "Close"].item()
        item_4_T0 = self.df.loc[self.current_step - 0, "Volume"].item()
        item_5_T0 = self.df.loc[self.current_step - 0, "VWAP"].item()
        
        env_4 = 1 if self.long_short_ratio else 0
        
        obs = np.array([item_0_T0, item_1_T0, item_2_T0, item_3_T0, item_4_T0, item_5_T0, env_4])
        
        return obs

    # Set the current price to a random price within the time step
    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, "Close_Price"].item()
        next_price = self.df.loc[self.current_step + 1, "Close_Price"].item()
        next_return = next_price / current_price - 1
        
        # Go Long
        if action == 0:
            self.net_profit += self.available_balance * PERCENT_CAPITAL * next_return
            self.available_balance += self.net_profit
            self.num_trades_long += 1
                
        # Go Short
        if action == 1:
            self.net_profit += self.available_balance * PERCENT_CAPITAL * -next_return
            self.available_balance += self.net_profit
            self.num_trades_short += 1
        
        # Update metrics
        self.long_short_ratio = self.num_trades_long / (self.num_trades_long + self.num_trades_short)
        self.volatility = self.df.loc[self.current_step - self.lag, "Close_Price"].sum()

    # Execute one time step within the environment
    def step(self, action):
        self._take_action(action)

        reward = self._calculate_reward()
    
        self.current_step += 1
        
        is_max_steps_taken = self.current_step >= self.max_steps - self.lag - 1
        done = True if is_max_steps_taken else False
        
        obs = self._next_observation()

        return obs, reward, done, {}

    # Reset the state of the environment to an initial state
    def reset(self):
        self.available_balance = INITIAL_ACCOUNT_BALANCE
        self.net_profit = 0
        self.current_step = self.lag
        self.num_trades_long = 0
        self.num_trades_short = 0
        self.num_trades_ratio = 0

        return self._next_observation()

    # Render the environment to the screen
    def render(self, mode='human', close=False):
        pass


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo') #Original V1 has some awesome weights
        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.AdamW(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp/'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.AdamW(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        state = state.flatten(0)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            
            # Calculate the advantage
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-50):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 50 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':
    env = StockTradingEnv(df_train_env)
    N = 20
    batch_size = 5
    n_epochs = 3
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)

    n_games = 500

    figure_file = 'stock_training.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    
    print("... starting ...")
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
            observation = observation_
            
        # Save history
        score_history.append(score)
        avg_score = np.mean(score_history[-50:])
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        
        print(f"episide: {i}, score: {score}, avg score: {avg_score}, best_score: {best_score}")
            
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

# agent.save_models()
n_actions = env.action_space.n
input_dims = env.observation_space.shape
alpha = 0.0003
model = ActorNetwork(n_actions, input_dims, alpha)
model.load_state_dict(T.load("tmp/actor_torch_ppo"))
model.eval()

reporting_df = df_test.copy()  # тестовое окно
long_probs, short_probs = [], []

is_long = 1
is_short = 1
long_ratio = 0.5

import torch as T
import numpy as np

for step in range(5, len(reporting_df)):
    # Берём ИМЕННО reporting_df, а не df_mod
    row = reporting_df.iloc[step]

    item_0_T0 = float(row["Open"])
    item_1_T0 = float(row["High"])
    item_2_T0 = float(row["Low"])
    item_3_T0 = float(row["Close"])
    item_4_T0 = float(row["Volume"])
    item_5_T0 = float(row["VWAP"])

    obs = np.array([item_0_T0, item_1_T0, item_2_T0, item_3_T0, item_4_T0, item_5_T0, long_ratio],
                   dtype=np.float32)

    # если модель ждёт батч → добавим размерность
    state = T.tensor(obs).float().unsqueeze(0)
    if hasattr(model, "device"):
        state = state.to(model.device)

    out = model(state)  # logits или распределение
    # Если у тебя Actor возвращает Categorical(logits=...), то:
    if hasattr(out, "probs"):
        probs = out.probs.squeeze(0).detach().cpu().numpy()
    else:
        # иначе считаем softmax вручную (logits -> probs)
        probs_t = T.softmax(out, dim=-1)
        probs = probs_t.squeeze(0).detach().cpu().numpy()

    action = int(np.argmax(probs))
    print(action, probs)

    # корректное обновление счётчиков
    if action == 0:
        is_long += 1
    elif action == 1:
        is_short += 1

    den = is_long + is_short
    long_ratio = float(is_long) / den if den > 0 else 0.5

    long_probs.append(float(probs[0]))
    short_probs.append(float(probs[1]))

# --- helper: единый расчёт BH-кривой и аккуратный тестовый срез ---
def bh_curve_from_prices(price_series, alloc):
    ret = price_series.astype(float).pct_change().fillna(0.0).to_numpy()
    return (1.0 + alloc * ret).cumprod()

def bh_test_from_full(price_full, df_test, alloc):
    # если индексы дат — берём .loc по датам; иначе — позициями .iloc
    if isinstance(df_test.index, pd.DatetimeIndex) and isinstance(price_full.index, pd.DatetimeIndex):
        price_test = price_full.loc[df_test.index]
        price_test = price_test.reset_index(drop=True)
    else:
        start = len(price_full) - len(df_test)
        price_test = price_full.iloc[start : start + len(df_test)].reset_index(drop=True)
    return bh_curve_from_prices(price_test, alloc)

# --- ваш блок с исправлениями ---
capital = 1.0
perc_invest = 0.3

# используем adjusted цены, если доступны
price_full = (df["Adj Close"] if "Adj Close" in df.columns else df["Close"]).astype(float)

# Buy&Hold на всём датасете
bh_full = bh_curve_from_prices(price_full, perc_invest)
ROI_BH_FULL = (bh_full[-1] - 1.0) * 100

# Buy&Hold на тестовом окне df_test (тот же источник цен)
bh_test = bh_test_from_full(price_full, df_test, perc_invest)
ROI_BH_TEST = (bh_test[-1] - 1.0) * 100

# === Подготовка данных под стратегию из вероятностей (как у вас) ===
reporting_df = df_test.reset_index(drop=True).copy()
df_res = reporting_df[["Open", "Close_Price"]].copy()
df_res["Returns"] = df_res["Close_Price"].pct_change().fillna(0.0)

probs = pd.DataFrame({"Longs": long_probs, "Shorts": short_probs})
N = len(df_res); M = len(probs)
if M < N:
    pad = pd.DataFrame({"Longs": np.zeros(N - M, dtype=float),
                        "Shorts": np.zeros(N - M, dtype=float)})
    probs = pd.concat([pad, probs], ignore_index=True)
elif M > N:
    probs = probs.iloc[-N:].reset_index(drop=True)
else:
    probs = probs.reset_index(drop=True)

df_res = df_res.reset_index(drop=True)
df_res = pd.concat([df_res, probs], axis=1)

# направление: лонг = +Longs, шорт = -Shorts
df_res["DIR"] = np.where(df_res["Longs"] >= 0.5, df_res["Longs"], -df_res["Shorts"]).astype(float)

# equity стратегии (позиция со вчера * сегодняшняя доходность)
gross_step = 1.0 + perc_invest * df_res["DIR"].shift(1).fillna(0.0) * df_res["Returns"]
df_res["Equity"] = capital * gross_step.cumprod()

# Benchmark в таблице = тестовый срез BH, полный ROI печатаем отдельно
df_res["Benchmark"] = bh_test

ROI = (df_res["Equity"].iloc[-1] - 1.0) * 100
print(f"Strategy ROI:         {ROI:.2f}%")
print(f"Buy&Hold ROI (TEST):  {ROI_BH_TEST:.2f}%")
print(f"Buy&Hold ROI (FULL):  {ROI_BH_FULL:.2f}%")
