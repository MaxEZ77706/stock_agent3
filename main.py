# ============================================
# PPO Stock Trader (AAPL example)
# - Transformer backbone (sequence model)
# - Fixed actor std bug (state-dependent sigma)
# - Single-source costs (in reward only)
# - Reward aligned to PnL (no extra shaping)
# - Unified train/test mechanics (VOL_MODE="cap", same hysteresis)
# ============================================

# ===========================
# Imports
# ===========================
import os, math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym
from gym import spaces
from torch.utils.tensorboard import SummaryWriter  # optional, not required to run

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import yfinance as yf
from ta.volume import VolumeWeightedAveragePrice
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler
from collections import deque

# ===========================
# Hyperparams
# ===========================
INITIAL_ACCOUNT_BALANCE = 1_000.0

PERCENT_CAPITAL  = 0.30
TARGET_DAILY_VOL = 0.015         # slightly conservative target exposure
VOL_MODE         = "cap"         # unified train/test: "cap" => min(1.0, TARGET/vol)
TURNOVER_COST    = 1e-5          # per unit position change
SLIPPAGE_BPS     = 0.0           # per unit position change
LEVER_CAP        = 1.50
KILL_THRESH      = 0.50          # hard stop at 50% drawdown from peak
DD_COEF          = 0.50          # drawdown penalty weight

SMOOTH_COST = 1e-3               # slightly higher to tame churn
HOLD_COST   = 2e-4

REWARD_CLIP = 3.0

# Hysteresis (train & test unified)
TAU_ENTER = 0.05
TAU_EXIT  = 0.02

# ===========================
# Data
# ===========================
df = yf.download("AAPL", start="2017-01-01", end="2025-01-01", auto_adjust=False)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
df = df.sort_index()

train_df = df.loc["2017-01-01":"2021-12-31"].copy()
test_df  = df.loc["2022-01-01":"2024-12-31"].copy()

def add_features(ddf: pd.DataFrame) -> pd.DataFrame:
    vwap = VolumeWeightedAveragePrice(
        high=ddf["High"], low=ddf["Low"], close=ddf["Close"],
        volume=ddf["Volume"], window=14, fillna=False
    )
    ddf["VWAP"] = vwap.volume_weighted_average_price()

    ddf["RSI"]   = RSIIndicator(ddf["Close"], window=14).rsi()
    ddf["EMA20"] = EMAIndicator(ddf["Close"], window=20).ema_indicator()
    ddf["EMA50"] = EMAIndicator(ddf["Close"], window=50).ema_indicator()

    macd = MACD(ddf["Close"])
    ddf["MACD"]        = macd.macd()
    ddf["MACD_signal"] = macd.macd_signal()

    bb = BollingerBands(ddf["Close"])
    ddf["BB_high"]  = bb.bollinger_hband()
    ddf["BB_low"]   = bb.bollinger_lband()
    ddf["BB_width"] = ddf["BB_high"] - ddf["BB_low"]

    ret1    = ddf["Close"].pct_change()
    vol20   = ret1.rolling(20).std()
    vol100  = ret1.rolling(100).std()
    ratio   = (vol20 / vol100).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0, 5.0)

    ddf["RET1"]        = ret1.fillna(0.0)
    ddf["ATR"]         = (ddf["High"] - ddf["Low"]).rolling(14).mean().fillna(0.0)
    ddf["VOL_REGIME"]  = ratio
    ddf["Close_Price"] = ddf["Close"].astype(float)

    ddf["EMA20_SLOPE"] = ddf["EMA20"].diff().fillna(0.0)
    ddf["EMA50_SLOPE"] = ddf["EMA50"].diff().fillna(0.0)
    ddf["Trend_Dir"]   = np.where(ddf["EMA20"] > ddf["EMA50"], 1.0, -1.0).astype(float)

    adx = ADXIndicator(ddf["High"], ddf["Low"], ddf["Close"], window=14)
    ddf["ADX"] = adx.adx().fillna(0.0)

    return ddf

train_df = add_features(train_df)
test_df  = add_features(test_df)

feat_cols = ["Open","High","Low","Close","Volume","VWAP",
             "RSI","EMA20","EMA50","MACD","MACD_signal","BB_width",
             "RET1","ATR","VOL_REGIME","ADX", "Trend_Dir", "EMA20_SLOPE", "EMA50_SLOPE"]

scaler = StandardScaler().fit(train_df[feat_cols])
train_df[feat_cols] = scaler.transform(train_df[feat_cols])
test_df[feat_cols]  = scaler.transform(test_df[feat_cols])

df_train = train_df.reset_index(drop=True)
df_test  = test_df.reset_index(drop=True)
SEQ_LEN = 32

# ===========================
# Environment
# ===========================
class StockTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, episode_len=256, randomize=True, lag=20):
        super().__init__()
        self.df = df
        self.lag = int(lag)
        self.max_steps = len(df)
        self.episode_len = int(min(episode_len or self.max_steps, self.max_steps - self.lag))
        self.randomize = bool(randomize)

        # simple baseline only for feature (long/short ratio), not for reward
        sma20 = self.df["Close_Price"].rolling(20).mean()
        sma50 = self.df["Close_Price"].rolling(50).mean()
        self.baseline_pos_series = (sma20 > sma50).astype(float).fillna(0.0)

        self.available_balance = INITIAL_ACCOUNT_BALANCE
        self.net_profit = 0.0
        self.equity_peak = self.available_balance / INITIAL_ACCOUNT_BALANCE

        self.num_trades_long = 0
        self.num_trades_short = 0
        self.long_short_ratio = 0.0

        self.window_start = None
        self.window_end = None
        self.current_step = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(feat_cols)+1,), dtype=np.float32)

        self.prev_position = 0.0
        self.ret_series = self.df["Close_Price"].pct_change().fillna(0.0).astype(float)
        self.vol_series = self.ret_series.rolling(self.lag).std().fillna(self.ret_series.std())
        v0 = float(self.vol_series.iloc[self.lag]) if len(self.vol_series) > self.lag else 1e-6
        self.volatility = max(v0, 1e-6)

    def _next_observation(self):
        r = self.df.loc[self.current_step]
        core = np.array([float(r[c]) for c in feat_cols], dtype=np.float32)
        obs = np.concatenate([core, np.array([float(self.long_short_ratio)], dtype=np.float32)])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _vol_scale(self, vol):
        vol = max(vol, 1e-6)
        if VOL_MODE == "cap":
            return float(min(1.0, TARGET_DAILY_VOL / vol))
        else:
            return float(TARGET_DAILY_VOL / vol)

    def _take_action(self, action):
        position = float(action[0])

        # un-clipped asset return to match backtest
        asset_ret = float(self.ret_series.iloc[self.current_step + 1]) if (self.current_step + 1) < self.max_steps else 0.0

        vol = float(self.vol_series.iloc[self.current_step]) if self.current_step < len(self.vol_series) else 1e-6
        self.volatility = vol if (np.isfinite(vol) and vol > 0) else 1e-6

        scale = self._vol_scale(self.volatility)
        eff_position = float(np.clip(position * scale, -1.0, 1.0))

        # realized one-step return from the position
        raw_ret = eff_position * asset_ret
        realized_ret = float(raw_ret)

        traded_cap = max(self.available_balance, 0.0) * PERCENT_CAPITAL
        step_pnl   = traded_cap * realized_ret
        self.net_profit        += step_pnl
        self.available_balance += step_pnl

        delta_pos = eff_position - self.prev_position

        if eff_position > 0: self.num_trades_long  += 1
        if eff_position < 0: self.num_trades_short += 1
        den = self.num_trades_long + self.num_trades_short
        self.long_short_ratio = (self.num_trades_long / den) if den > 0 else 0.0

        return realized_ret, eff_position, delta_pos, raw_ret, asset_ret

    def step(self, action):
        realized_ret, eff_position, delta_pos, raw_ret, asset_ret = self._take_action(action)

        # regularizers
        smooth_pen = float(SMOOTH_COST * (delta_pos ** 2))
        hold_pen   = float(HOLD_COST   * (eff_position ** 2))

        equity = self.available_balance / INITIAL_ACCOUNT_BALANCE
        self.equity_peak = max(self.equity_peak, equity)
        dd = min(0.0, equity / self.equity_peak - 1.0)  # negative or 0

        # === ONE source of truth for costs (in reward only) ===
        txn_cost_ret = (TURNOVER_COST + SLIPPAGE_BPS) * abs(delta_pos)

        reward = realized_ret - smooth_pen - hold_pen - DD_COEF * (-dd) - txn_cost_ret
        reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))

        self.prev_position = eff_position
        hard_stop = (dd < -KILL_THRESH)

        self.current_step += 1
        done = hard_stop or (self.current_step >= self.window_end - 1)

        info = {
            "drawdown": float(dd),
            "raw_ret": float(raw_ret),
            "realized_ret": float(realized_ret),
            "vol": float(self.volatility),
            "eff_position": float(eff_position),
            "smooth_pen": float(smooth_pen),
            "hold_pen": float(hold_pen),
        }
        return self._next_observation(), reward, done, info

    def reset(self, start_balance=None):
        self.available_balance = float(start_balance) if start_balance is not None else INITIAL_ACCOUNT_BALANCE
        self.net_profit = 0.0
        self.num_trades_long = 0; self.num_trades_short = 0
        self.long_short_ratio = 0.0
        self.prev_position = 0.0
        self.equity_peak = self.available_balance / INITIAL_ACCOUNT_BALANCE

        if self.randomize and self.max_steps > self.episode_len:
            start_idx = int(np.random.randint(self.lag, self.max_steps - self.episode_len + 1))
        else:
            start_idx = self.lag
        self.window_start = start_idx
        self.window_end   = min(self.max_steps, self.window_start + self.episode_len)
        self.current_step = self.window_start

        vol = float(self.vol_series.iloc[self.current_step]) if self.current_step < len(self.vol_series) else 1e-6
        self.volatility = vol if (np.isfinite(vol) and vol > 0) else 1e-6

        return self._next_observation()

    def render(self, mode='human', close=False):
        pass

# ===========================
# PPO memory
# ===========================
class PPOMemory:
    def __init__(self, batch_size):
        self.states, self.probs, self.vals = [], [], []
        self.actions, self.rewards, self.dones = [], [], []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        states_np = np.array(self.states, dtype=np.float32)
        return (states_np,
                np.array(self.actions, dtype=np.float32),
                np.array(self.probs,   dtype=np.float32),
                np.array(self.vals,    dtype=np.float32),
                np.array(self.rewards, dtype=np.float32),
                np.array(self.dones,   dtype=np.float32),
                batches)

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states.clear(); self.probs.clear(); self.actions.clear()
        self.rewards.clear(); self.dones.clear(); self.vals.clear()

# ===========================
# Networks
# ===========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = T.zeros(max_len, d_model)
        pos = T.arange(0, max_len, dtype=T.float32).unsqueeze(1)
        div = T.exp(T.arange(0, d_model, 2, dtype=T.float32) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = T.sin(pos * div)
        pe[:, 1::2] = T.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):  # [B,L,d]
        return x + self.pe[:, :x.size(1), :]

class TransformerBackbone(nn.Module):
    def __init__(self, feat_dim, d_model=64, nhead=4, nlayers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.posenc = PositionalEncoding(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):           # [B,L,F]
        x = self.proj(x)
        x = self.posenc(x)
        x = self.encoder(x)         # [B,L,d]
        return self.norm(x[:, -1, :])

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, lr, d_model=32, nhead=4, nlayers=2, dropout=0.1, chkpt_dir='tmp/'):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_cont_cont_trx')
        feat_dim = int(input_dims[-1]) if isinstance(input_dims, (tuple, list, np.ndarray)) else int(input_dims)
        self.backbone = TransformerBackbone(feat_dim, d_model, nhead, nlayers, dropout)

        self.fc_mu = nn.Linear(d_model, 1)
        self.fc_logstd  = nn.Linear(d_model, 1)      # state-dependent log_std
        self.log_std_bias = nn.Parameter(T.zeros(1, 1))  # optional global bias
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state_seq):
        h = self.backbone(state_seq)
        mu = T.tanh(self.fc_mu(h))
        # FIX: use head for state-dependent sigma
        log_std = self.fc_logstd(h) + self.log_std_bias
        log_std = log_std.clamp(-5.0, 2.0)
        sigma = log_std.exp()
        return T.distributions.Normal(mu, sigma)

    def save_checkpoint(self): T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self): self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, lr, d_model=32, nhead=4, nlayers=2, dropout=0.1, chkpt_dir='tmp/'):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_cont_trx')
        feat_dim = int(input_dims[-1]) if isinstance(input_dims, (tuple, list, np.ndarray)) else int(input_dims)
        self.backbone = TransformerBackbone(feat_dim, d_model, nhead, nlayers, dropout)
        self.fc_value = nn.Linear(d_model, 1)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state_seq):
        h = self.backbone(state_seq)
        return self.fc_value(h)

    def save_checkpoint(self): T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self): self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

# ===========================
# PPO Agent
# ===========================
class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, lr=3e-4,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=512,
                 n_epochs=8, entropy_coef=0.02, vf_clip=0.2,
                 target_kl=0.02, kl_coef=0.0, max_grad_norm=1.0):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.entropy_coef_base = entropy_coef
        self.entropy_floor = 0.005
        self.learn_calls = 0

        self.vf_clip = vf_clip
        self.target_kl = target_kl
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm

        self.actor  = ActorNetwork(input_dims, lr)
        self.critic = CriticNetwork(input_dims, lr)
        self.memory = PPOMemory(batch_size)

        self.seq_len = SEQ_LEN
        self.state_window = deque(maxlen=self.seq_len)

        self.actor_sched  = optim.lr_scheduler.CosineAnnealingLR(self.actor.optimizer,  T_max=200)
        self.critic_sched = optim.lr_scheduler.CosineAnnealingLR(self.critic.optimizer, T_max=200)

        # Unified hysteresis for exploration/inference
        self.TAU_ENTER, self.TAU_EXIT = TAU_ENTER, TAU_EXIT
        self.policy_pos_cache = 0.0

    def remember(self, state_seq, action, log_prob, value, reward, done):
        self.memory.store_memory(state_seq, action, log_prob, value, reward, done)

    def save_models(self):
        self.actor.save_checkpoint(); self.critic.save_checkpoint()

    def choose_action(self, observation):
        # sequence build
        self.state_window.append(np.asarray(observation, dtype=np.float32))
        if len(self.state_window) < self.seq_len:
            return np.array([0.0], dtype=np.float32), 0.0, 0.0, None

        state_seq = np.stack(self.state_window, axis=0).astype(np.float32)
        state_tensor = T.tensor(state_seq, dtype=T.float32,
                                device=self.actor.device).unsqueeze(0)
        with T.no_grad():
            dist  = self.actor(state_tensor)
            value = self.critic(state_tensor)
            action = T.clamp(dist.sample(), -1.0, 1.0)

        # hysteresis / deadzone
        raw = float(action.squeeze().cpu().numpy())
        if self.policy_pos_cache == 0.0:
            if abs(raw) < self.TAU_ENTER: raw = 0.0
        else:
            if abs(raw) < self.TAU_EXIT:  raw = 0.0
        self.policy_pos_cache = raw

        a_tensor = T.tensor([[raw]], dtype=T.float32, device=self.actor.device)
        log_prob = dist.log_prob(a_tensor).sum(dim=-1).item()

        return np.array([raw], dtype=np.float32), float(log_prob), float(value.item()), state_seq

    def learn(self):
        if len(self.memory.states) < self.memory.batch_size:
            return

        device = self.actor.device
        curr_entropy_coef = max(self.entropy_floor,
                                self.entropy_coef_base * (0.5 ** (self.learn_calls / 10)))

        (state_arr, action_arr, old_logp_arr,
         vals_arr, reward_arr, dones_arr, batches) = self.memory.generate_batches()

        rewards = np.asarray(reward_arr, dtype=np.float32)
        values  = np.asarray(vals_arr,    dtype=np.float32)
        dones   = np.asarray(dones_arr,   dtype=np.float32)

        # GAE(λ)
        Tlen = len(rewards)
        advantage = np.zeros(Tlen, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(Tlen)):
            if t == Tlen - 1:
                next_nonterminal = 0.0; next_value = 0.0
            else:
                next_nonterminal = 1.0 - dones[t+1]; next_value = values[t+1]
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
            advantage[t] = gae

        advantage_t = T.tensor(advantage, dtype=T.float32, device=device)
        advantage_t = (advantage_t - advantage_t.mean()) / (advantage_t.std() + 1e-8)
        values_t    = T.tensor(values,    dtype=T.float32, device=device)

        states_all  = T.tensor(state_arr,  dtype=T.float32, device=device)
        actions_all = T.tensor(action_arr, dtype=T.float32, device=device).unsqueeze(-1)
        old_logp_all= T.tensor(old_logp_arr,dtype=T.float32, device=device)

        early_stop = False
        for _ in range(self.n_epochs):
            for batch in batches:
                states   = states_all[batch]
                actions  = actions_all[batch]
                old_logp = old_logp_all[batch]
                adv      = advantage_t[batch]
                old_vals = values_t[batch]
                returns  = adv + old_vals

                dist = self.actor(states)
                new_logp = dist.log_prob(actions).sum(dim=-1)
                ratio = (new_logp - old_logp).exp()
                surr1 = ratio * adv
                surr2 = T.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * adv
                actor_loss = -T.min(surr1, surr2).mean()

                approx_kl = (old_logp - new_logp).mean()
                actor_loss = actor_loss + self.kl_coef * approx_kl

                value_pred = self.critic(states).squeeze(-1)
                value_pred_clipped = old_vals + (value_pred - old_vals).clamp(-self.vf_clip, self.vf_clip)

                vf_loss_unclipped = (value_pred - returns).pow(2)
                vf_loss_clipped   = (value_pred_clipped - returns).pow(2)
                critic_loss = T.max(vf_loss_unclipped, vf_loss_clipped).mean()

                entropy = dist.entropy().mean()
                total_loss = actor_loss + 0.5 * critic_loss - curr_entropy_coef * entropy

                self.actor.optimizer.zero_grad(set_to_none=True)
                self.critic.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),  self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                if approx_kl.item() > 1.5 * self.target_kl:
                    early_stop = True
                    break

            self.actor_sched.step()
            self.critic_sched.step()
            if early_stop: break

        self.learn_calls += 1
        self.memory.clear_memory()

# ===========================
# Helpers to make env slices
# ===========================
def make_env_slice(df_full, start_idx, end_idx, episode_len=None, randomize=True, lag=20):
    df_slice = df_full.iloc[start_idx:end_idx].reset_index(drop=True).copy()
    return StockTradingEnv(df_slice, episode_len=episode_len, randomize=randomize, lag=lag)

# ===========================
# Curriculum (train)
# ===========================
train_windows = [
    ("2017-01-01", "2019-12-31"),
    ("2018-01-01", "2020-12-31"),
    ("2019-01-01", "2021-12-31"),
]
env_full = StockTradingEnv(df_train)

agent = Agent(
    n_actions=1,
    input_dims=env_full.observation_space.shape,
    lr=1e-3,
    batch_size=512,
    n_epochs=10,
    entropy_coef=0.06,
    policy_clip=0.30,
    target_kl=0.08,
    max_grad_norm=0.9,
    gae_lambda=0.97
)

stages = [
    ("stage1", 0,   900,              600),
    ("stage2", 300, 1200,             800),
    ("stage3", 0,   len(df_train),   2000),
]

best_score = -1e9
score_history, n_steps = [], 0

print("... starting aligned curriculum ...")
for name, s, e, n_games in stages:
    env = make_env_slice(df_train, s, e)

    window_len = (e - s)
    N = max(192, int(0.5 * window_len)) if name != "stage3" else max(256, int(0.6 * window_len))

    patience, min_delta = 25, 0.0015
    stale = 0
    best_local = -1e9

    for i in range(n_games):
        obs = env.reset(); agent.state_window.clear()
        done, score = False, 0.0
        zero_act = 0; total_act = 0

        while not done:
            action, logp, val, state_seq = agent.choose_action(obs)
            total_act += 1
            if abs(float(action[0])) < 1e-6:
                zero_act += 1

            obs_, reward, done, info = env.step(action)

            if state_seq is not None:
                agent.remember(state_seq, float(action[0]), logp, val, reward, done)

            n_steps += 1
            score   += reward
            if n_steps % N == 0:
                agent.learn()

            obs = obs_

        steps_in_ep = max(1, env.episode_len)
        mean_score  = score / steps_in_ep
        score_history.append(mean_score)
        avg50 = float(np.mean(score_history[-50:]))

        if avg50 > best_score:
            best_score = avg50
            agent.save_models()

        if avg50 > best_local + min_delta:
            best_local = avg50
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            print(f"{name}: early stop (no improve {patience} eps), best_local={best_local:.4f}")
            break

        zero_ratio = 100.0 * zero_act / max(1, total_act)
        pnl    = env.net_profit
        equity = env.available_balance

        print(f"{name} | ep {i:4d} "
              f"| mean {mean_score: .4f} "
              f"| avg50 {avg50: .4f} "
              f"| best {best_score: .4f} "
              f"| PnL ${pnl: .2f} | Equity ${equity: .2f} "
              f"| zero% {zero_ratio: .1f}")

# ===========================
# Inference (test) + Backtest
# ===========================
infer_model = agent.actor; infer_model.eval()
reporting_df = df_test.reset_index(drop=True).copy()

lag = 20
test_rets = reporting_df["Close_Price"].pct_change().fillna(0.0)
test_vol  = test_rets.rolling(lag).std().fillna(test_rets.std())

positions = []
state_window = deque(maxlen=SEQ_LEN)
pol_cache = 0.0

# for long_short_ratio feature on test
longs = shorts = 0

def vol_scale_infer(vol):
    vol = max(float(vol), 1e-6)
    if VOL_MODE == "cap":
        return float(min(1.0, TARGET_DAILY_VOL / vol))
    else:
        return float(TARGET_DAILY_VOL / vol)

with T.no_grad():
    for step in range(len(reporting_df)):
        r = reporting_df.iloc[step]
        den = max(1, longs + shorts)
        ls_ratio = float(longs) / den

        core = reporting_df.reindex(columns=feat_cols).iloc[step].astype(float).to_numpy(np.float32)
        obs  = np.concatenate([core, np.array([ls_ratio], dtype=np.float32)])
        obs  = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        state_window.append(obs)

        if len(state_window) < SEQ_LEN:
            positions.append(0.0); continue

        state_seq = np.stack(list(state_window), axis=0)[None, ...].astype(np.float32)
        state_t   = T.tensor(state_seq, dtype=T.float32, device=infer_model.device)
        dist = infer_model(state_t)
        mu = float(T.clamp(dist.mean, -1.0, 1.0).squeeze().cpu().numpy())

        # unified hysteresis
        raw = mu
        if pol_cache == 0.0:
            if abs(raw) < TAU_ENTER: raw = 0.0
        else:
            if abs(raw) < TAU_EXIT:  raw = 0.0
        pol_cache = raw

        curr_vol = max(float(test_vol.iloc[step]), 1e-6)
        scale = vol_scale_infer(curr_vol)
        eff_pos = float(np.clip(raw * scale, -1.0, 1.0))
        positions.append(eff_pos)

        if eff_pos > 0: longs += 1
        elif eff_pos < 0: shorts += 1

# --- Backtest table ---
df_bt = reporting_df.copy()
df_bt["Return"] = df_bt["Close_Price"].pct_change().fillna(0.0)

pos = np.asarray(positions, dtype=float).ravel()
N = len(df_bt)
if len(pos) < N: pos = np.concatenate([np.zeros(N-len(pos)), pos])
elif len(pos) > N: pos = pos[-N:]
df_bt["Position"] = pos
delta_pos = df_bt["Position"].diff().fillna(0.0)

active_ret = df_bt["Position"].shift(1).fillna(0.0) * df_bt["Return"] \
             - (TURNOVER_COST + SLIPPAGE_BPS) * delta_pos.abs()
df_bt["PortRet"]   = PERCENT_CAPITAL * active_ret
df_bt["Equity"]    = (1.0 + df_bt["PortRet"]).cumprod()

# --- Buy&Hold: consistent helpers ---
def bh_curve_from_prices(price_series, pct_capital=1.0):
    px = np.asarray(price_series, dtype=float)
    ret = np.zeros_like(px, dtype=float)
    ret[1:] = (px[1:] - px[:-1]) / np.where(px[:-1] == 0, 1e-12, px[:-1])
    return (1.0 + pct_capital * ret).cumprod()

def bh_test_from_full(price_full, test_like, pct_capital=1.0):
    idx = test_like.index if isinstance(test_like, (pd.DataFrame, pd.Series)) else test_like
    if isinstance(idx, pd.DatetimeIndex) and isinstance(price_full.index, pd.DatetimeIndex):
        px = price_full.loc[idx].reset_index(drop=True)
    else:
        px = price_full.iloc[-len(idx):].reset_index(drop=True)
    return bh_curve_from_prices(px, pct_capital)

price_full = (df["Adj Close"] if "Adj Close" in df.columns else df["Close"]).astype(float)
bh_full = bh_curve_from_prices(price_full, PERCENT_CAPITAL)
ROI_BH_FULL = (bh_full[-1] - 1.0) * 100.0

bh_test = bh_test_from_full(price_full, df_test, PERCENT_CAPITAL)
bh_series = pd.Series(bh_test)
if len(bh_series) != len(df_bt):
    if len(bh_series) > len(df_bt):
        bh_series = bh_series.iloc[-len(df_bt):].reset_index(drop=True)
    else:
        pad = pd.Series([bh_series.iloc[0]] * (len(df_bt) - len(bh_series)))
        bh_series = pd.concat([pad, bh_series], ignore_index=True)
df_bt["Benchmark"] = bh_series.values
ROI_BH_TEST = (df_bt["Benchmark"].iloc[-1] - 1.0) * 100.0

# --- Simple SMA baseline (costed) ---
sma20 = df_bt["Close_Price"].rolling(20).mean()
sma50 = df_bt["Close_Price"].rolling(50).mean()
df_bt["Baseline_Pos"] = (sma20 > sma50).astype(float).fillna(0.0)
b_delta = df_bt["Baseline_Pos"].diff().fillna(0.0).abs()
df_bt["BaselineRet"] = df_bt["Baseline_Pos"].shift(1).fillna(0.0)*df_bt["Return"] \
                       - (TURNOVER_COST + SLIPPAGE_BPS)*b_delta
df_bt["BaselineEq"]  = (1.0 + PERCENT_CAPITAL * df_bt["BaselineRet"]).cumprod()

# --- Metrics ---
ROI     = (df_bt["Equity"].iloc[-1]    - 1.0) * 100
ROI_SB  = (df_bt["BaselineEq"].iloc[-1]- 1.0) * 100
daily   = df_bt["PortRet"].to_numpy()
sharpe  = (daily.mean() / (daily.std() + 1e-12)) * np.sqrt(252.0)

def max_dd(x):
    x = np.asarray(x, float); peak = np.maximum.accumulate(x)
    return float((x/peak - 1.0).min())
mdd = max_dd(df_bt["Equity"])

print(f"\nTEST 2022–2024")
print(f"Strategy ROI:         {ROI:.2f}% | Sharpe {sharpe:.2f} | MDD {mdd:.2%}")
print(f"Buy&Hold ROI (TEST):  {ROI_BH_TEST:.2f}%")
print(f"Buy&Hold ROI (FULL):  {ROI_BH_FULL:.2f}%")
print(f"SMA20>50 ROI (TEST):  {ROI_SB:.2f}%")

# Optional: quick plot (comment out if running headless)
try:
    plt.figure(figsize=(10,5))
    plt.plot(df_bt["Equity"], label="Strategy")
    plt.plot(df_bt["Benchmark"], label="Buy&Hold (TEST)")
    plt.plot(df_bt["BaselineEq"], label="SMA20>50 Baseline")
    plt.title("Equity Curves (2022–2024)")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    # plt.show()
except Exception as e:
    pass
