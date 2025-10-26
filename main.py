# ===========================
# Imports
# ===========================
import os, math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym
from gym import spaces

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import yfinance as yf
from ta.volume import VolumeWeightedAveragePrice
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler
from collections import deque

# ---- capital & costs ----
PERCENT_CAPITAL  = 0.3# было 0.30 → снизить риск ×2
TARGET_DAILY_VOL = 0.015       # было 0.020 → меньше плечо от волы
TURNOVER_COST    = 0
SLIPPAGE_BPS     = 0

# ---- shape of reward ----
WIN_REWARD    = 0.7            # было 1.0
LOSS_PENALTY  = -0.7           # было -1.0
WINLOSS_EPS   = 1e-4           # было 5e-5
WINLOSS_Z     = 0.03           # было 0.02 → «мертвая зона» шире (меньше шумовых входов)
INITIAL_ACCOUNT_BALANCE = 1000
ALPHA_PNL     = 0.5           # было 0.9 → меньше веса «псевдо-PnL», меньше переобучения
REWARD_CLIP   = 2.0            # было 3.0 → стабильнее обучение/оценка

# ---- penalties ----
SMOOTH_COST   = 5e-4           # было 2e-4 → дороже дерганье
HOLD_COST     = 2e-4           # было 1e-5 → не стоим «на газу» без причины

# ---- risk guard ----
KILL_THRESH   = 0.40
VOL_MODE      = "cap"

# ===========================
# Data download & features
# ===========================
df = yf.download("AAPL", start="2017-01-01", end="2025-01-01", auto_adjust=False)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

vwap = VolumeWeightedAveragePrice(
    high=df["High"], low=df["Low"], close=df["Close"],
    volume=df["Volume"], window=14, fillna=False
)
df["VWAP"] = vwap.volume_weighted_average_price()
df["RSI"]   = RSIIndicator(df["Close"], window=14).rsi()
df["EMA20"] = EMAIndicator(df["Close"], window=20).ema_indicator()
df["EMA50"] = EMAIndicator(df["Close"], window=50).ema_indicator()

macd = MACD(df["Close"])
df["MACD"]        = macd.macd()
df["MACD_signal"] = macd.macd_signal()

bb = BollingerBands(df["Close"])
df["BB_high"]  = bb.bollinger_hband()
df["BB_low"]   = bb.bollinger_lband()
df["BB_width"] = df["BB_high"] - df["BB_low"]


ret1    = df["Close"].pct_change()
vol20   = ret1.rolling(20).std()
vol100  = ret1.rolling(100).std()
ratio   = (vol20 / vol100).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0, 5.0)

df["RET1"]       = ret1.fillna(0.0)
df["ATR"]        = (df["High"] - df["Low"]).rolling(14).mean().fillna(0.0)
df["VOL_REGIME"] = ratio
df["Close_Price"] = df["Close"].astype(float)

feat_cols = [
    "Open","High","Low","Close","Volume","VWAP",
    "RSI","EMA20","EMA50","MACD","MACD_signal","BB_width",
    "RET1","ATR","VOL_REGIME"
]

train_df = df.loc["2017-01-01":"2021-12-31"].copy()  # тренировка
test_df  = df.loc["2022-01-01":"2024-12-31"].copy()  # тест (ДОЛЖЕН остаться с DatetimeIndex!)


scaler = StandardScaler().fit(train_df[feat_cols])
train_std = train_df.copy()
test_std  = test_df.copy()
train_std[feat_cols] = scaler.transform(train_df[feat_cols])
test_std[feat_cols]  = scaler.transform(test_df[feat_cols])

# для модели и бэктеста стратегий удобно иметь RangeIndex:
df_train = train_std.reset_index(drop=True)
df_test  = test_std.reset_index(drop=True)

# но ДЛЯ BH используем ровно тест с DatetimeIndex:
test_df_bh = test_df 
SEQ_LEN = 32

# ===========================
# Buy&Hold helpers (те же, что во втором коде)
# ===========================
def bh_curve_from_prices(price_series, alloc):
    ret = price_series.astype(float).pct_change().fillna(0.0).to_numpy()
    return (1.0 + alloc * ret).cumprod()

def bh_test_from_full(price_full, df_test_like, alloc):
    if isinstance(df_test_like.index, pd.DatetimeIndex) and isinstance(price_full.index, pd.DatetimeIndex):
        price_test = price_full.loc[df_test_like.index].reset_index(drop=True)
    else:
        L = len(df_test_like)
        price_test = price_full.iloc[-L:].reset_index(drop=True)
    return bh_curve_from_prices(price_test, alloc)

# единый источник цен для BH
price_full = (df["Adj Close"] if "Adj Close" in df.columns else df["Close"]).astype(float)

# единый alloc
PERCENT_CAPITAL = globals().get("PERCENT_CAPITAL", 0.30)

# считаем BH так же, как во втором коде
bh_full = bh_curve_from_prices(price_full, PERCENT_CAPITAL)
ROI_BH_FULL = (bh_full[-1] - 1.0) * 100.0

bh_test = bh_test_from_full(price_full, test_df_bh, PERCENT_CAPITAL)  # ВАЖНО: по ДАТАМ!
ROI_BH_TEST = (bh_test[-1] - 1.0) * 100.0


import numpy as np
import gym
from gym import spaces

class StockTradingEnv(gym.Env):
    """Continuous-position trading env with adaptive win/loss reward."""
    metadata = {"render.modes": ["human"]}

    def __init__(self, df):
        super().__init__()
        self.df = df  # DataFrame: feat_cols + 'Close_Price'

        # ---- счёт --\--
        self.available_balance = INITIAL_ACCOUNT_BALANCE
        self.net_profit = 0.0

        # ---- счётчики ----
        self.num_trades_long = 0
        self.num_trades_short = 0
        self.long_short_ratio = 0.0

        # ---- время ----
        self.current_step = 0
        self.lag = 20
        self.max_steps = len(df)

        # ---- action: позиция ∈ [-1, 1] ----
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # ---- observation: 16 признаков (15 фич + long_short_ratio) ----
        feat_dim = 16
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(feat_dim,), dtype=np.float32)

        # ---- волатильность для таргетинга и адаптивного порога ----
        self.prev_position = 0.0
        rets = self.df["Close_Price"].pct_change().fillna(0.0)
        self.vol_series = rets.rolling(self.lag).std().fillna(rets.std())
        self.volatility = float(self.vol_series.iloc[0]) if np.isfinite(self.vol_series.iloc[0]) else 1e-6

        # ---- стоп по просадке ----
        self.equity_peak = 1.0

    # ---------- helpers ----------
    def _next_observation(self):
        row = self.df.loc[self.current_step]
        obs = np.array([
            float(row["Open"]),  float(row["High"]),  float(row["Low"]),
            float(row["Close"]), float(row["Volume"]), float(row["VWAP"]),
            float(row["RSI"]),   float(row["EMA20"]), float(row["EMA50"]),
            float(row["MACD"]),  float(row["MACD_signal"]),
            float(row["BB_width"]),
            float(row["RET1"]),  float(row["ATR"]),   float(row["VOL_REGIME"]),
            float(self.long_short_ratio),
        ], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _take_action(self, action):
        position = float(action[0])
    
        # доходность актива
        if self.current_step + 1 >= self.max_steps:
            asset_ret = 0.0
        else:
            p0 = float(self.df.loc[self.current_step, "Close_Price"])
            p1 = float(self.df.loc[self.current_step + 1, "Close_Price"])
            asset_ret = (p1 / max(p0, 1e-12)) - 1.0
    
        # волатильность
        vol = float(self.vol_series.iloc[self.current_step]) if self.current_step < len(self.vol_series) else 0.0
        self.volatility = vol if (np.isfinite(vol) and vol > 0) else 1e-6
    
        # вол-таргет
        if VOL_MODE == "full":
            scale = TARGET_DAILY_VOL / self.volatility
        else:
            scale = min(1.0, TARGET_DAILY_VOL / self.volatility)
    
        eff_position = float(np.clip(position * scale, -1.0, 1.0))
    
        # доходность сделки + сглаживание хвостов
        raw_ret = eff_position * asset_ret
        realized_ret = float(np.tanh(raw_ret / 0.01) * 0.01)
            
                # PnL на долю капитала
        traded_cap = self.available_balance * PERCENT_CAPITAL
        step_pnl   = traded_cap * realized_ret
        self.net_profit        += step_pnl
        self.available_balance += step_pnl
    
        # издержки за оборот
        delta_pos    = eff_position - self.prev_position
        turnover_fee = traded_cap * TURNOVER_COST * abs(delta_pos)
        if turnover_fee > 0:
            self.available_balance -= turnover_fee
            self.net_profit        -= turnover_fee
    
        # счётчики long/short
        if eff_position > 0: self.num_trades_long  += 1
        if eff_position < 0: self.num_trades_short += 1
        den = self.num_trades_long + self.num_trades_short
        self.long_short_ratio = (self.num_trades_long / den) if den > 0 else 0.0
    
        return realized_ret, eff_position, delta_pos, raw_ret

    def step(self, action):
        realized_ret, eff_position, delta_pos, raw_ret = self._take_action(action)

        # штрафы за дёрганье/удержание
        smooth_pen = float(SMOOTH_COST * (delta_pos ** 2))
        hold_pen   = float(HOLD_COST   * (eff_position ** 2))

        # адаптивный порог win/loss: максимум из абсолютного минимума и «доли волатильности»
        thr_abs = max(
            WINLOSS_EPS,
            WINLOSS_Z * self.volatility * (abs(eff_position) + 0.1)
        )

        # win/loss по «сырому» результату сделки
        if   raw_ret >  thr_abs: base = WIN_REWARD
        elif raw_ret < -thr_abs: base = LOSS_PENALTY
        else:                    base = 0.0
        
        # смешиваем дискретную (base) и непрерывную (realized_ret) части
        reward = base + ALPHA_PNL * realized_ret - smooth_pen - hold_pen


        
        # стабилизация дисперсии награды
        
        reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))



        # обновляем позицию
        self.prev_position = eff_position

        # стоп по просадке
        equity = self.available_balance / INITIAL_ACCOUNT_BALANCE
        self.equity_peak = max(self.equity_peak, equity)
        dd = equity / self.equity_peak - 1.0
        hard_stop = (dd < -KILL_THRESH)

        # время
        self.current_step += 1
        done = hard_stop or (self.current_step >= self.max_steps - 1)

        info = {
            "drawdown": float(dd),
            "base_reward": float(base),
            "thr_abs": float(thr_abs),
            "raw_ret": float(raw_ret),
            "realized_ret": float(realized_ret),
            "vol": float(self.volatility),
            "eff_position": float(eff_position),
            "smooth_pen": float(smooth_pen),
            "hold_pen": float(hold_pen),
        }
        return self._next_observation(), reward, done, info

    def reset(self, start_balance=None):
        # если передан баланс → стартуем с него
        self.available_balance = float(start_balance) if start_balance is not None else INITIAL_ACCOUNT_BALANCE
        self.net_profit = 0.0
        self.current_step = self.lag
        self.num_trades_long = 0
        self.num_trades_short = 0
        self.long_short_ratio = 0.0
        self.prev_position = 0.0
        self.equity_peak = self.available_balance / INITIAL_ACCOUNT_BALANCE
    
        vol = float(self.vol_series.iloc[self.current_step]) if len(self.vol_series) > self.current_step else 1e-6
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
        if isinstance(input_dims, (tuple, list, np.ndarray)):
            feat_dim = int(input_dims[-1])
        else:
            feat_dim = int(input_dims)
        self.backbone = TransformerBackbone(feat_dim, d_model, nhead, nlayers, dropout)


        self.fc_mu = nn.Linear(d_model, 1)
        self.log_std = nn.Parameter(T.zeros(1, 1))
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state_seq):
        h = self.backbone(state_seq)
        mu = T.tanh(self.fc_mu(h))
        sigma = F.softplus(self.log_std) + 1e-4
        return T.distributions.Normal(mu, sigma)

    def save_checkpoint(self): T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self): self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, lr, d_model=32, nhead=4, nlayers=2, dropout=0.1, chkpt_dir='tmp/'):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_cont_trx')
        if isinstance(input_dims, (tuple, list, np.ndarray)):
            feat_dim = int(input_dims[-1])
        else:
            feat_dim = int(input_dims)
        self.backbone = TransformerBackbone(feat_dim, d_model, nhead, nlayers, dropout)
        self.fc_value = nn.Linear(d_model, 1)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state_seq):
        h = self.backbone(state_seq)
        return self.fc_value(h)

    def save_checkpoint(self): T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self): self.load_state_dict(T.load(self.checkpoint_file))

REWARD_SCALE = 100.0 
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

    def remember(self, state_seq, action, log_prob, value, reward, done):
        self.memory.store_memory(state_seq, action, log_prob, value, reward, done)

    def save_models(self):
        self.actor.save_checkpoint(); self.critic.save_checkpoint()

    def choose_action(self, observation):
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
        TAU_ENTER, TAU_EXIT = 0.12, 0.04
        if not hasattr(self, "policy_pos_cache"):
            self.policy_pos_cache = 0.0
        raw = float(action.squeeze().cpu().numpy())
        if self.policy_pos_cache == 0.0:
            if abs(raw) < TAU_ENTER: raw = 0.0
        else:
            if abs(raw) < TAU_EXIT:  raw = 0.0
        self.policy_pos_cache = raw
        
        # ВАЖНО: лог-вероятность считаем для ИТОГОВОГО действия raw
        a_tensor = T.tensor([[raw]], dtype=T.float32, device=self.actor.device)
        log_prob = dist.log_prob(a_tensor).sum(dim=-1).item()
        
        return np.array([raw], dtype=np.float32), float(log_prob), float(value.item()), state_seq


    def learn(self):
        if len(self.memory.states) < self.memory.batch_size:
            return

        device = self.actor.device
        curr_entropy_coef = max(self.entropy_floor,
                                self.entropy_coef_base * (0.5 ** (self.learn_calls / 10)))

        for _ in range(self.n_epochs):
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

                ret_t = returns  # без нормализации
                vf_loss_unclipped = (value_pred - ret_t).pow(2)
                vf_loss_clipped   = (value_pred_clipped - ret_t).pow(2)

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

def make_env_slice(df_full, start_idx, end_idx):
    df_slice = df_full.iloc[start_idx:end_idx].reset_index(drop=True).copy()
    return StockTradingEnv(df_slice)

# ===========================
# Train loop with curriculum + per-stage early stop + sane 'best'
# ===========================
def make_env_slice(df_full, start_idx, end_idx):
    df_slice = df_full.iloc[start_idx:end_idx].reset_index(drop=True).copy()
    return StockTradingEnv(df_slice)

env = StockTradingEnv(df_train)
agent = Agent(
    n_actions=1,
    input_dims=env.observation_space.shape,
    lr=7e-4,
    batch_size=256,
    n_epochs=8,
    entropy_coef=0.04,
    policy_clip=0.25,
    target_kl=0.06,
    max_grad_norm=0.7,
    gae_lambda=0.98
)

# name, slice_start, slice_end, learn_every_N, episodes
stages = [
    ("stage1", 0,   900,               256,  400),
    ("stage2", 300, 1200,              384,  400),
    ("stage3", 0,   len(df_train),     768, 1200),
]

global_best = float("-inf")     # лучший avg50 за всё обучение (для сохранения модели)
print("... starting curriculum ...")

for name, s, e, N, n_games in stages:
    env = make_env_slice(df_train, s, e)

    # --- Early Stop: внутри стадии ---
    patience  = 15        # сколько эпизодов подряд без улучшения терпим
    min_delta = 2e-3      # минимальный прирост avg50, чтобы считать улучшением
    stale = 0
    local_best = float("-inf")

    stage_scores = []     # история mean_score в этой стадии
    n_steps = 0

    for i in range(n_games):
        obs = env.reset(); agent.state_window.clear()
        done = False
        score = 0.0
        zero_act = 0
        total_act = 0
        wins = losses = 0
        ep_steps = 0

        while not done:
            action, logp, val, state_seq = agent.choose_action(obs)
            total_act += 1
            if abs(float(action[0])) < 1e-6:
                zero_act += 1

            obs_, reward, done, info = env.step(action)
            ep_steps += 1
            if info["base_reward"] > 0: wins += 1
            elif info["base_reward"] < 0: losses += 1

            if state_seq is not None:
                agent.remember(state_seq, float(action[0]), logp, val, reward, done)

            n_steps += 1
            score += reward
            if n_steps % N == 0:
                agent.learn()

            obs = obs_

        # честная нормализация на фактическую длину эпизода
        mean_score = score / max(1, ep_steps)
        stage_scores.append(mean_score)
        avg50 = float(np.mean(stage_scores[-50:]))

        # --- сохраняем модель по глобальному лучшему avg50 ---
        if avg50 > global_best + 1e-12:
            global_best = avg50
            agent.save_models()

        # --- Early Stop внутри стадии ---
        if avg50 > local_best + min_delta:
            local_best = avg50
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            print(f"{name}: early stop at ep {i} | local_best={local_best:.4f} | global_best={global_best:.4f}")
            break

        zero_ratio = 100.0 * zero_act / max(1, total_act)
        wr = wins / max(1, wins + losses)

        pnl = env.net_profit
        equity = env.available_balance

        print(
            f"{name} | ep {i:4d} "
            f"| mean {mean_score: .4f} "
            f"| avg50 {avg50: .4f} "
            f"| local_best {local_best: .4f} "
            f"| global_best {global_best: .4f} "
            f"| PnL ${pnl: .2f} | Equity ${equity: .2f} "
            f"| winrate {wr:.1%} | zero% {zero_ratio: .1f}"
        )

# ===========================
# Inference (test) + Backtest (только против Buy&Hold)
# ===========================
infer_model = agent.actor; infer_model.eval()
MAX_LEVER = 1.0
reporting_df = df_test.reset_index(drop=True).copy()

lag = 20
test_rets = reporting_df["Close_Price"].pct_change().fillna(0.0)
test_vol  = test_rets.rolling(lag).std().fillna(test_rets.std())

positions = []
state_window = deque(maxlen=SEQ_LEN)
with T.no_grad():
    for step in range(len(reporting_df)):
        r = reporting_df.iloc[step]
        obs = np.array([
            float(r["Open"]),  float(r["High"]),  float(r["Low"]),
            float(r["Close"]), float(r["Volume"]), float(r["VWAP"]),
            float(r["RSI"]),   float(r["EMA20"]), float(r["EMA50"]),
            float(r["MACD"]),  float(r["MACD_signal"]),
            float(r["BB_width"]),
            float(r["RET1"]),  float(r["ATR"]),   float(r["VOL_REGIME"]),
            0.0
        ], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        state_window.append(obs)
        if len(state_window) < SEQ_LEN:
            positions.append(0.0); continue
        state_seq = np.stack(list(state_window), axis=0).astype(np.float32)[None, ...]
        state_t   = T.tensor(state_seq, dtype=T.float32, device=infer_model.device)
        dist = infer_model(state_t); mu = dist.mean
        pos = float(T.clamp(mu, -1.0, 1.0).squeeze().cpu().numpy())
        curr_vol = max(float(test_vol.iloc[step]), 1e-6)
        scale = (TARGET_DAILY_VOL / curr_vol) if VOL_MODE == "full" else min(1.0, TARGET_DAILY_VOL / curr_vol)
        eff_pos = float(np.clip(pos * scale, -MAX_LEVER, MAX_LEVER))
        positions.append(eff_pos)


# ===========================
# FIRST BLOCK (patched to match the second)
# ===========================
import numpy as np
import pandas as pd

# ---- constants (must match the second block) ----
# PERCENT_CAPITAL = globals().get("PERCENT_CAPITAL", 0.2)
# TURNOVER_COST   = globals().get("TURNOVER_COST",   1e-5)
SLIPPAGE_BPS    = globals().get("SLIPPAGE_BPS",    1e-4)

# ---- helpers for BH (same logic as in the second block) ----
def bh_curve_from_prices(price_series, pct_capital=1.0):
    px = np.asarray(price_series, dtype=float)
    ret = np.zeros_like(px, dtype=float)
    ret[1:] = (px[1:] - px[:-1]) / np.where(px[:-1]==0, 1e-12, px[:-1])
    port = (1.0 + pct_capital * ret)
    return np.maximum.accumulate(np.ones_like(port)) * port.cumprod()

def bh_test_from_full(price_full, test_like, pct_capital=1.0):
    """
    price_full: Series с DatetimeIndex (вся история)
    test_like : DataFrame/Series/Index тестового окна
    """
    if isinstance(test_like, (pd.DataFrame, pd.Series)):
        idx = test_like.index
    else:
        idx = test_like

    if isinstance(idx, pd.DatetimeIndex):
        px = price_full.loc[idx]
    else:
        # индекс не по датам (RangeIndex) -> берём хвост нужной длины
        n = len(idx)
        px = price_full.iloc[-n:]

    return bh_curve_from_prices(px, pct_capital)



# ---- source prices for BH (identical to the second block) ----
price_full = (df["Adj Close"] if "Adj Close" in df.columns else df["Close"]).astype(float)

# ---------------- Backtest table (exactly same formulas as in the second) ----------------
df_bt = reporting_df.copy()
# Return must come from the actual Close_Price (same as second block)
df_bt["Return"] = df_bt["Close_Price"].pct_change().fillna(0.0)

# align positions length to N exactly like you do in the second block
pos = np.asarray(positions, dtype=float).ravel()
N = len(df_bt)
if len(pos) < N:
    pos = np.concatenate([np.zeros(N - len(pos)), pos])
elif len(pos) > N:
    pos = pos[-N:]
df_bt["Position"] = pos

delta_pos = df_bt["Position"].diff().fillna(0.0)

active_ret = (
    df_bt["Position"].shift(1).fillna(0.0) * df_bt["Return"]
    - (TURNOVER_COST + SLIPPAGE_BPS) * delta_pos.abs()
)
df_bt["PortRet"] = PERCENT_CAPITAL * active_ret
df_bt["Equity"]  = (1.0 + df_bt["PortRet"]).cumprod()

# ------------- Buy&Hold (FULL + TEST) exactly as in the second block -------------
bh_full = bh_curve_from_prices(price_full, PERCENT_CAPITAL)
ROI_BH_FULL = (bh_full[-1] - 1.0) * 100.0

# ВАЖНО: используем ИМЕННО df_test с DatetimeIndex (а НЕ reporting_df)
bh_test = bh_test_from_full(price_full, df_test, PERCENT_CAPITAL)

# align BH curve length to df_bt length (safe guard; normally lengths match)
bh_series = pd.Series(bh_test, index=range(len(bh_test)))
if len(bh_series) != len(df_bt):
    if len(bh_series) > len(df_bt):
        bh_series = bh_series.iloc[-len(df_bt):].reset_index(drop=True)
    else:
        pad = pd.Series([bh_series.iloc[0]] * (len(df_bt) - len(bh_series)))
        bh_series = pd.concat([pad, bh_series], ignore_index=True)

df_bt["Benchmark"] = bh_series.values
ROI_BH_TEST = (df_bt["Benchmark"].iloc[-1] - 1.0) * 100.0

# ---------------- metrics (same outputs as the second block) ----------------
ROI     = (df_bt["Equity"].iloc[-1]    - 1.0) * 100.0

def max_drawdown(arr_like):
    x = np.asarray(arr_like, dtype=float)
    peak = np.maximum.accumulate(x)
    return float((x / peak - 1.0).min())

daily  = df_bt["PortRet"].to_numpy()
eq     = df_bt["Equity"].to_numpy()
sharpe = (daily.mean() / (daily.std() + 1e-12)) * np.sqrt(252.0)
mdd    = max_drawdown(eq)

print(f"\nBenchmark (BH @ {int(PERCENT_CAPITAL*100)}%) ROI: {ROI_BH_TEST:.2f}%")
print(f"Strategy ROI:                         {ROI:.2f}%")
print(f"Sharpe:                               {sharpe:.2f}")
print(f"Max Drawdown:                         {mdd:.2%}")
