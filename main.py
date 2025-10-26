# ===========================
# Universal PPO (Transformer) for Single-Ticker (AAPL)
# Train -> Save best -> Inference (with robust gating)
# ===========================
# pip install torch numpy pandas yfinance ta gym matplotlib

import os, math, random
from collections import deque
import numpy as np
import pandas as pd

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
from gym import spaces
import yfinance as yf

from ta.volume import VolumeWeightedAveragePrice
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands

# ===========================
# Global Hyperparams (shared)
# ===========================
SEED = 1337
np.random.seed(SEED); random.seed(SEED); T.manual_seed(SEED)

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")

# Account / costs
INITIAL_ACCOUNT_BALANCE = 1_000.0
PERCENT_CAPITAL  = 0.30
TARGET_DAILY_VOL = 0.06
TURNOVER_COST    = 1e-5
SLIPPAGE_BPS     = 0.0

# Trend gates (must conceptually match env & inference)
TREND_GATE = True
TREND_BONUS_COEF   = 0.05
TREND_STRENGTH_CLIP= (0.0, 1.0)
ADX_ENTER, ADX_FULL = 18.0, 28.0
TRADE_ADX_MIN = 10.0      # <= смягчили
EMA_GAP_MIN   = 0.0007    # <= смягчили

# Internal penalties (env)
SMOOTH_COST = 5e-4
HOLD_COST   = 5e-4
WIN_REWARD, LOSS_PENALTY = 0.3, -0.3
WINLOSS_EPS, WINLOSS_Z   = 2e-5, 0.015
ALPHA_PNL, BETA_EXCESS   = 0.9, 0.5
REWARD_CLIP = 3.0
KILL_THRESH = 0.50
DD_COEF     = 0.50

# Sequence length (Transformer window)
SEQ_LEN = 32
APPLY_DEADZONE_TRAIN = True

# Inference tuning (to avoid zero positions)
LONG_ONLY_INFER = False
TAU_ENTER_INFER, TAU_EXIT_INFER = 0.03, 0.01      # tighter hysteresis for test
GATE_FLOOR = 0.65                                  # allow more flow through gates
EMA_POS_ALPHA = 0.30                                # smoother -> reactive position
CE_MULT_WEAK, CE_MULT_STRONG = 3.2, 3.6            # wider CE to avoid premature exits

# ===========================
# Data
# ===========================
TICKER = "AAPL"
RAW = yf.download(TICKER, start="2017-01-01", end="2025-01-01", auto_adjust=False, progress=False)
RAW.columns = [c[0] if isinstance(c, tuple) else c for c in RAW.columns]
RAW = RAW.sort_index()
train_df = RAW.loc["2017-01-01":"2021-12-31"].copy()
test_df  = RAW.loc["2022-01-01":"2024-12-31"].copy()

# ===========================
# Features
# ===========================
def add_features(ddf: pd.DataFrame) -> pd.DataFrame:
    ddf = ddf.copy()

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
    ddf["Trend_Dir"] = np.where(ddf["EMA20"] > ddf["EMA50"], 1.0, -1.0).astype(float)

    adx = ADXIndicator(ddf["High"], ddf["Low"], ddf["Close"], window=14)
    ddf["ADX"] = adx.adx().fillna(0.0)

    # Raw copies for consistent gating in env & inference
    ddf["EMA20_RAW"] = ddf["EMA20"].astype(float)
    ddf["EMA50_RAW"] = ddf["EMA50"].astype(float)
    ddf["ADX_RAW"]   = ddf["ADX"].astype(float)
    ddf["CLOSE_RAW"] = ddf["Close_Price"].astype(float)
    return ddf

train_df = add_features(train_df)
test_df  = add_features(test_df)

# Observed features fed to policy (standardized)
feat_cols_obs = [
    "Open","High","Low","Close","Volume","VWAP",
    "RSI","EMA20","EMA50","MACD","MACD_signal","BB_width",
    "RET1","ATR","VOL_REGIME","ADX","Trend_Dir","EMA20_SLOPE","EMA50_SLOPE"
]
# Raw features (NOT standardized) for gates/safety
filter_cols_raw = ["EMA20_RAW","EMA50_RAW","ADX_RAW","CLOSE_RAW"]

# Rolling standardization (train and then test with rolling anchor)
def rolling_standardize(df_in, cols, win=252, min_periods=20):
    df_std = df_in.copy()
    for c in cols:
        roll_mean = df_std[c].rolling(win, min_periods=min_periods).mean()
        roll_std  = df_std[c].rolling(win, min_periods=min_periods).std()
        df_std[c] = (df_std[c] - roll_mean) / (roll_std.replace(0, 1e-12))
    return df_std

train_raw = train_df.copy()
test_raw  = test_df.copy()

train_std = rolling_standardize(train_df, feat_cols_obs, win=252)
tmp = pd.concat([train_df.tail(252), test_df], axis=0)
tmp_std = rolling_standardize(tmp, feat_cols_obs, win=252)
test_std = tmp_std.iloc[252:].copy()

CLIP_FEAT = 6.0
for ddf in (train_std, test_std):
    ddf[feat_cols_obs] = (
        ddf[feat_cols_obs]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-CLIP_FEAT, CLIP_FEAT)
    )

# Drop date index for env convenience
df_train = train_std.reset_index(drop=True)
df_test  = test_std.reset_index(drop=True)
train_raw = train_raw.reset_index(drop=True)
test_raw  = test_raw.reset_index(drop=True)

# Helpers for BH
def bh_curve_from_prices(price_series, pct_capital=1.0):
    px = np.asarray(price_series, dtype=float)
    ret = np.zeros_like(px, dtype=float)
    if px.size > 1:
        ret[1:] = (px[1:] - px[:-1]) / np.where(px[:-1]==0, 1e-12, px[:-1])
    return (1.0 + pct_capital * ret).cumprod()

def bh_test_from_full(price_full, test_like, pct_capital=1.0):
    idx = getattr(test_like, "index", test_like)
    if isinstance(idx, pd.DatetimeIndex) and isinstance(price_full.index, pd.DatetimeIndex):
        px = price_full.loc[idx].reset_index(drop=True)
    else:
        px = price_full.iloc[-len(idx):].reset_index(drop=True)
    return bh_curve_from_prices(px, pct_capital)

def soft_scale(vol, target=TARGET_DAILY_VOL, alpha=0.6):
    s_full = target / max(vol, 1e-6)
    s_cap  = min(1.0, s_full)
    return alpha * s_full + (1 - alpha) * s_cap

# ===========================
# Environment (std + raw)
# ===========================
class StockTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df_std, df_raw, episode_len=256, randomize=True, lag=20):
        super().__init__()
        self.df_std = df_std.reset_index(drop=True).copy()
        self.df_raw = df_raw.reset_index(drop=True).copy()
        assert len(self.df_std) == len(self.df_raw)
        self.lag = int(lag)
        self.max_steps = len(self.df_std)
        self.episode_len = int(min(episode_len or self.max_steps, self.max_steps - self.lag))
        self.randomize = bool(randomize)

        self.hold_clock = 0
        self.last_sign  = 0
        self.min_hold   = 3

        sma20 = self.df_raw["CLOSE_RAW"].rolling(20).mean()
        sma50 = self.df_raw["CLOSE_RAW"].rolling(50).mean()
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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(feat_cols_obs)+1,), dtype=np.float32)

        self.prev_position = 0.0
        self.ret_series = self.df_raw["CLOSE_RAW"].pct_change().fillna(0.0)

        self.vol_series = self.ret_series.rolling(self.lag).std().fillna(self.ret_series.std())
        v0 = float(self.vol_series.iloc[self.lag]) if len(self.vol_series) > self.lag else 1e-6
        self.volatility = max(v0, 1e-6)

    def _next_observation(self):
        r = self.df_std.loc[self.current_step]
        core = np.array([float(r[c]) for c in feat_cols_obs], dtype=np.float32)
        obs = np.concatenate([core, np.array([float(self.long_short_ratio)], dtype=np.float32)])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _safe(self, x): return float(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))

    def _take_action(self, action):
        position = float(action[0])
        asset_ret = float(self.ret_series.iloc[self.current_step + 1]) if (self.current_step + 1) < self.max_steps else 0.0
        asset_ret = float(np.clip(asset_ret, -0.12, 0.12))

        vol = float(self.vol_series.iloc[self.current_step]) if self.current_step < len(self.vol_series) else 0.0
        self.volatility = vol if (np.isfinite(vol) and vol > 0) else 1e-6

        scale = soft_scale(self.volatility)
        eff_position = float(np.clip(position * scale, -1.0, 1.0))

        # RAW gates (soft)
        adx      = float(self.df_raw.loc[self.current_step, "ADX_RAW"])
        ema20    = float(self.df_raw.loc[self.current_step, "EMA20_RAW"])
        ema50    = float(self.df_raw.loc[self.current_step, "EMA50_RAW"])
        close_px = float(self.df_raw.loc[self.current_step, "CLOSE_RAW"])

        ema_gap  = abs(ema20 - ema50) / max(abs(close_px), 1e-6)
        gate_adx = float(np.clip((adx - TRADE_ADX_MIN) / max(ADX_FULL - TRADE_ADX_MIN, 1e-6), 0.0, 1.0))
        gate_gap = float(np.clip(ema_gap / max(EMA_GAP_MIN, 1e-12), 0.0, 1.0))
        eff_position *= gate_adx * gate_gap

        trend_dir = 1.0 if ema20 >= ema50 else -1.0
        gate = float(np.clip((adx - ADX_ENTER) / max(ADX_FULL - ADX_ENTER, 1e-6), 0.0, 1.0))

        eff_position *= (0.6 + 0.4 * gate)
        if eff_position * trend_dir < 0:
            eff_position *= (1.0 - 0.5 * gate)

        if TREND_GATE:
            ts = float(np.clip(adx / 50.0, *TREND_STRENGTH_CLIP))
            eff_position = float(np.clip(eff_position * (0.5 + 0.5 * ts), -1.0, 1.0))

        # min-hold
        sign = 0 if abs(eff_position) < 1e-6 else (1 if eff_position > 0 else -1)
        if sign == 0:
            self.hold_clock += 1
        else:
            if self.last_sign == 0:
                self.last_sign = sign; self.hold_clock = 0
            elif sign != self.last_sign and self.hold_clock < self.min_hold:
                eff_position = abs(eff_position) * self.last_sign
                sign = self.last_sign; self.hold_clock += 1
            elif sign != self.last_sign and self.hold_clock >= self.min_hold:
                self.last_sign = sign; self.hold_clock = 0
            else:
                self.hold_clock += 1

        raw_ret = eff_position * asset_ret
        realized_ret = self._safe(raw_ret)

        # balance/pnl update
        if not np.isfinite(self.available_balance):
            self.available_balance = INITIAL_ACCOUNT_BALANCE; self.net_profit = 0.0

        traded_cap = max(self._safe(self.available_balance), 0.0) * PERCENT_CAPITAL
        step_pnl   = traded_cap * realized_ret
        if not np.isfinite(step_pnl): step_pnl = 0.0
        self.net_profit        = self._safe(self.net_profit + step_pnl)
        self.available_balance = self._safe(self.available_balance + step_pnl)

        delta_pos    = eff_position - self.prev_position
        turnover_fee = abs(traded_cap) * TURNOVER_COST * abs(delta_pos)
        if not np.isfinite(turnover_fee): turnover_fee = 0.0
        if turnover_fee != 0.0:
            self.available_balance = self._safe(self.available_balance - turnover_fee)
            self.net_profit        = self._safe(self.net_profit - turnover_fee)

        if eff_position > 0: self.num_trades_long  += 1
        if eff_position < 0: self.num_trades_short += 1
        den = self.num_trades_long + self.num_trades_short
        self.long_short_ratio = (self.num_trades_long / den) if den > 0 else 0.0

        return realized_ret, eff_position, delta_pos, raw_ret, asset_ret, turnover_fee

    def step(self, action):
        realized_ret, eff_position, delta_pos, raw_ret, asset_ret, turnover_fee = self._take_action(action)

        smooth_pen = float(SMOOTH_COST * (delta_pos ** 2))
        hold_pen   = float(HOLD_COST   * (eff_position ** 2))

        ema20 = float(self.df_raw.loc[self.current_step, "EMA20_RAW"])
        ema50 = float(self.df_raw.loc[self.current_step, "EMA50_RAW"])
        trend_dir = 1.0 if ema20 >= ema50 else -1.0

        adx = float(self.df_raw.loc[self.current_step, "ADX_RAW"])
        trend_strength = float(np.clip(adx / 50.0, *TREND_STRENGTH_CLIP))
        trend_bonus = TREND_BONUS_COEF * trend_strength * np.sign(eff_position) * trend_dir

        thr_abs = max(WINLOSS_EPS, WINLOSS_Z * self.volatility * (abs(eff_position) + 0.1))
        base = WIN_REWARD if raw_ret >  thr_abs else (LOSS_PENALTY if raw_ret < -thr_abs else 0.0)

        b_t = float(self.baseline_pos_series.iloc[self.current_step])
        excess_step = (eff_position - b_t) * asset_ret

        equity = self.available_balance / INITIAL_ACCOUNT_BALANCE
        self.equity_peak = max(self.equity_peak, equity)
        dd = min(0.0, equity / self.equity_peak - 1.0)

        reward = base + ALPHA_PNL * realized_ret + BETA_EXCESS * excess_step \
               + trend_bonus - smooth_pen - hold_pen - DD_COEF * (-dd)

        reward = float(np.clip(float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)), -REWARD_CLIP, REWARD_CLIP))

        self.prev_position = eff_position
        hard_stop = (dd < -KILL_THRESH)

        self.current_step += 1
        done = hard_stop or (self.current_step >= self.window_end - 1)

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
        self.available_balance = float(start_balance) if start_balance is not None else INITIAL_ACCOUNT_BALANCE
        self.net_profit = 0.0
        self.num_trades_long = 0; self.num_trades_short = 0
        self.long_short_ratio = 0.0
        self.prev_position = 0.0
        self.hold_clock = 0
        self.last_sign  = 0
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
        self.available_balance = float(np.nan_to_num(self.available_balance, nan=INITIAL_ACCOUNT_BALANCE, posinf=1e12, neginf=0.0))
        self.net_profit        = float(np.nan_to_num(self.net_profit,        nan=0.0,                   posinf=1e12, neginf=-1e12))

        return self._next_observation()

# ===========================
# Transformer Policy & Value
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
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_cont_trx.pt')
        feat_dim = int(input_dims[-1]) if isinstance(input_dims, (tuple, list, np.ndarray)) else int(input_dims)
        self.backbone = TransformerBackbone(feat_dim, d_model, nhead, nlayers, dropout)
        self.fc_mu = nn.Linear(d_model, 1)
        self.fc_logstd  = nn.Linear(d_model, 1)
        self.log_std = nn.Parameter(T.zeros(1, 1))
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        self.device = DEVICE
        self.to(self.device)
    def forward(self, state_seq):
        h = self.backbone(state_seq)
        mu = T.tanh(self.fc_mu(h))
        _ = self.fc_logstd(h)    # not used; could be mixed
        sigma = F.softplus(self.log_std) + 1e-4
        return T.distributions.Normal(mu, sigma)
    def save_checkpoint(self): T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self): self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, lr, d_model=32, nhead=4, nlayers=2, dropout=0.1, chkpt_dir='tmp/'):
        super().__init__()
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_cont_trx.pt')
        feat_dim = int(input_dims[-1]) if isinstance(input_dims, (tuple, list, np.ndarray)) else int(input_dims)
        self.backbone = TransformerBackbone(feat_dim, d_model, nhead, nlayers, dropout)
        self.fc_value = nn.Linear(d_model, 1)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        self.device = DEVICE
        self.to(self.device)
    def forward(self, state_seq):
        h = self.backbone(state_seq)
        return self.fc_value(h)
    def save_checkpoint(self): T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self): self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

# ===========================
# PPO Memory / Agent
# ===========================
class PPOMemory:
    def __init__(self, batch_size):
        self.states, self.probs, self.vals = [], [], []
        self.actions, self.rewards, self.dones = [], [], []
        self.batch_size = int(batch_size)
    def generate_batches(self):
        n = len(self.states)
        if n == 0:
            raise RuntimeError("PPOMemory is empty")
        batch_start = np.arange(0, n, self.batch_size)
        indices = np.arange(n, dtype=np.int64); np.random.shuffle(indices)
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
        self.actions.append(float(action))
        self.probs.append(float(probs))
        self.vals.append(float(vals))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
    def clear_memory(self):
        self.states.clear(); self.probs.clear(); self.actions.clear()
        self.rewards.clear(); self.dones.clear(); self.vals.clear()

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, lr=1e-3,
                 gae_lambda=0.97, policy_clip=0.30, batch_size=512,
                 n_epochs=10, entropy_coef=0.06, vf_clip=0.2,
                 target_kl=0.08, kl_coef=0.0, max_grad_norm=0.9):
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

        self.TAU_ENTER, self.TAU_EXIT = 0.12, 0.04
        self.policy_pos_cache = 0.0

    def remember(self, state_seq, action, log_prob, value, reward, done):
        self.memory.store_memory(state_seq, action, log_prob, value, reward, done)

    def save_models(self):
        self.actor.save_checkpoint(); self.critic.save_checkpoint()

    def choose_action(self, observation):
        self.state_window.append(np.asarray(observation, dtype=np.float32))
        if len(self.state_window) < self.seq_len:
            return np.array([0.0], dtype=np.float32), 0.0, 0.0, None, 0.0
        state_seq = np.stack(self.state_window, axis=0).astype(np.float32)
        state_tensor = T.tensor(state_seq, dtype=T.float32, device=self.actor.device).unsqueeze(0)
        with T.no_grad():
            dist  = self.actor(state_tensor)
            value = self.critic(state_tensor)
            a_unc = dist.sample()
            logp  = dist.log_prob(a_unc).sum(dim=-1)
            a_env = T.clamp(a_unc, -1.0, 1.0)
        raw_env = float(a_env.squeeze().cpu().numpy())
        a_unc_s = float(a_unc.squeeze().cpu().numpy())
        if APPLY_DEADZONE_TRAIN:
            if self.policy_pos_cache == 0.0:
                if abs(raw_env) < self.TAU_ENTER: raw_env = 0.0
            else:
                if abs(raw_env) < self.TAU_EXIT:  raw_env = 0.0
            self.policy_pos_cache = raw_env
        return np.array([raw_env], dtype=np.float32), float(logp.item()), float(value.item()), state_seq, a_unc_s

    def learn(self):
        if len(self.memory.states) < self.memory.batch_size:
            return
        device = self.actor.device
        curr_entropy_coef = max(self.entropy_floor, self.entropy_coef_base * (0.5 ** (self.learn_calls / 10)))
        (state_arr, action_arr, old_logp_arr,
         vals_arr, reward_arr, dones_arr, batches) = self.memory.generate_batches()

        rewards = np.asarray(reward_arr, dtype=np.float32)
        values  = np.asarray(vals_arr,    dtype=np.float32)
        dones   = np.asarray(dones_arr,   dtype=np.float32)

        # GAE(λ)
        Tlen = len(rewards)
        advantage = np.zeros(Tlen, dtype=np.float32); gae = 0.0
        for t in reversed(range(Tlen)):
            if t == Tlen - 1:
                next_nonterminal = 0.0; next_value = 0.0
            else:
                next_nonterminal = 1.0 - dones[t+1]; next_value = values[t+1]
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
            advantage[t] = gae

        advantage_t = T.tensor(advantage, dtype=T.float32, device=device)
        std_adv = advantage_t.std()
        if T.isnan(std_adv) or std_adv < 1e-8:
            advantage_t = advantage_t * 0.0
        else:
            advantage_t = (advantage_t - advantage_t.mean()) / (std_adv + 1e-8)

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
                    early_stop = True; break
            self.actor_sched.step(); self.critic_sched.step()
            if early_stop: break

        self.learn_calls += 1
        self.memory.clear_memory()

def make_env_slice(df_std_full, df_raw_full, start_idx, end_idx, episode_len=None, randomize=True, lag=20):
    df_std_slice = df_std_full.iloc[start_idx:end_idx].reset_index(drop=True).copy()
    df_raw_slice = df_raw_full.iloc[start_idx:end_idx].reset_index(drop=True).copy()
    return StockTradingEnv(df_std_slice, df_raw_slice, episode_len=episode_len, randomize=randomize, lag=lag)

# ===========================
# Train (curriculum)
# ===========================
env = StockTradingEnv(df_train, train_raw)
agent = Agent(
    n_actions=1,
    input_dims=env.observation_space.shape,
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
    ("stage3", 0,   len(df_train),   1200),   # можно повысить для лучшего результата
]

best_score = -1e9
score_history, n_steps = [], 0
print("... starting curriculum ...")

for name, s, e, n_games in stages:
    env = make_env_slice(df_train, train_raw, s, e)
    window_len = (e - s)
    N = max(192, int(0.5 * window_len)) if name != "stage3" else max(256, int(0.6 * window_len))
    patience, min_delta = 25, 0.0015
    stale, best_local = 0, -1e9

    for i in range(n_games):
        obs = env.reset(); agent.state_window.clear()
        done, score = False, 0.0
        while not done:
            action, logp, val, state_seq, _ = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            if state_seq is not None:
                agent.remember(state_seq, float(action[0]), logp, val, reward, done)
            n_steps += 1; score += reward
            if n_steps % N == 0: agent.learn()
            obs = obs_

        mean_score  = score / max(1, env.episode_len)
        score_history.append(mean_score)
        avg50 = float(np.mean(score_history[-50:]))

        if avg50 > best_score:
            best_score = avg50
            agent.save_models()

        if avg50 > best_local + min_delta:
            best_local = avg50; stale = 0
        else:
            stale += 1
        if stale >= patience:
            print(f"{name}: early stop (no improve {patience} eps), best_local={best_local:.4f}")
            break

        if (i+1) % 50 == 0:
            print(f"{name} | ep {i+1:4d} | mean {mean_score: .4f} | avg50 {avg50: .4f} | best {best_score: .4f}")

# ===========================
# Load best checkpoint (if any)
# ===========================
try:
    agent.actor.load_checkpoint()
    agent.critic.load_checkpoint()
    print("Loaded best checkpoint.")
except Exception as e:
    print("WARN: couldn't load checkpoint ->", e)

# ===========================
# Inference (test) + Backtest
# ===========================
TEST_START = "2022-01-01"
TEST_END   = "2024-12-31"

df_full = RAW.copy()
price_full_series = (df_full["Adj Close"] if "Adj Close" in df_full.columns else df_full["Close"]).astype(float)

if isinstance(test_std.index, pd.DatetimeIndex):
    reporting_std = test_std.loc[TEST_START:TEST_END].reset_index(drop=True).copy()
else:
    reporting_std = test_std.reset_index(drop=True).copy()

if isinstance(test_raw.index, pd.DatetimeIndex):
    reporting_raw = test_raw.loc[TEST_START:TEST_END].reset_index(drop=True).copy()
else:
    reporting_raw = test_raw.reset_index(drop=True).copy()

L = min(len(reporting_std), len(reporting_raw))
reporting_std = reporting_std.iloc[:L].reset_index(drop=True)
reporting_raw = reporting_raw.iloc[:L].reset_index(drop=True)

infer_model = agent.actor
infer_model.eval()

lag = 20
test_rets = reporting_raw["CLOSE_RAW"].pct_change().fillna(0.0)
test_vol  = test_rets.rolling(lag).std().fillna(test_rets.std())

positions = []
state_window = deque(maxlen=SEQ_LEN)

# deadzone for test (overrides)
pol_cache = 0.0
TAU_ENTER, TAU_EXIT = TAU_ENTER_INFER, TAU_EXIT_INFER

# sanity-check actor mean on test tail
with T.no_grad():
    if len(reporting_std) > SEQ_LEN + 256:
        buf = []
        sw = deque(maxlen=SEQ_LEN)
        for i in range(len(reporting_std)-256, len(reporting_std)):
            core = reporting_std.reindex(columns=feat_cols_obs).iloc[i].astype(float).to_numpy(np.float32)
            obs  = np.concatenate([core, np.array([0.5], dtype=np.float32)])  # fake ls_ratio
            sw.append(obs)
            if len(sw) < SEQ_LEN: continue
            st = T.tensor(np.stack(sw, axis=0)[None, ...], dtype=T.float32, device=infer_model.device)
            dist = infer_model(st)
            buf.append(float(dist.mean.squeeze().cpu().numpy()))
        print(f"infer mu: mean={np.mean(buf):.4f}, std={np.std(buf):.4f}, nonzero%={(np.mean(np.abs(buf)>1e-3)*100):.1f}%")

with T.no_grad():
    for step in range(len(reporting_std)):
        # simple ls_ratio proxy from realized history
        ls_ratio = 0.0 if len(positions)==0 else float(np.mean(np.asarray(positions)>0))
        core = reporting_std.reindex(columns=feat_cols_obs).iloc[step].astype(float).to_numpy(np.float32)
        obs  = np.concatenate([core, np.array([ls_ratio], dtype=np.float32)])
        obs  = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        state_window.append(obs)
        if len(state_window) < SEQ_LEN:
            positions.append(0.0); continue

        state_seq = np.stack(state_window, axis=0)[None, ...].astype(np.float32)
        state_t   = T.tensor(state_seq, dtype=T.float32, device=infer_model.device)

        # policy output
        dist = infer_model(state_t)
        mu = float(dist.mean.squeeze().cpu().numpy())
        raw = mu

        # hysteresis BEFORE vol target and gates
        if pol_cache == 0.0:
            if abs(raw) < TAU_ENTER: raw = 0.0
        else:
            if abs(raw) < TAU_EXIT: raw = 0.0
        pol_cache = raw

        # vol targeting (same soft_scale)
        curr_vol = max(float(test_vol.iloc[step]), 1e-6)
        eff_pos = float(np.clip(raw * soft_scale(curr_vol, alpha=EMA_POS_ALPHA), -1.0, 1.0))

        # soft gates (one pass)
        adx     = float(reporting_raw.loc[step, "ADX_RAW"])
        ema20   = float(reporting_raw.loc[step, "EMA20_RAW"])
        ema50   = float(reporting_raw.loc[step, "EMA50_RAW"])
        closepx = float(reporting_raw.loc[step, "CLOSE_RAW"])
        ema_gap = abs(ema20 - ema50) / max(abs(closepx), 1e-6)

        gate_adx = float(np.clip((adx - TRADE_ADX_MIN)/max(ADX_FULL-TRADE_ADX_MIN,1e-6), 0.0, 1.0))
        gate_gap = float(np.clip(ema_gap / max(EMA_GAP_MIN,1e-12),                   0.0, 1.0))
        gate_mult = max(GATE_FLOOR, gate_adx * gate_gap)  # allow a floor
        eff_pos *= gate_mult

        trend_dir = 1.0 if ema20 >= ema50 else -1.0
        gate = float(np.clip((adx - ADX_ENTER)/max(ADX_FULL-ADX_ENTER,1e-6), 0.0, 1.0))
        eff_pos *= (0.6 + 0.4*gate)
        if eff_pos * trend_dir < 0:
            eff_pos *= (1.0 - 0.5*gate)

        if TREND_GATE:
            ts = float(np.clip(adx/50.0, *TREND_STRENGTH_CLIP))
            eff_pos = float(np.clip(eff_pos * (0.5 + 0.5*ts), -1.0, 1.0))

        # optional long-only
        if LONG_ONLY_INFER and eff_pos < 0:
            eff_pos = 0.0

        positions.append(eff_pos)

# Backtest table
df_bt = reporting_raw.copy()
df_bt["Return"] = df_bt["CLOSE_RAW"].pct_change().fillna(0.0).clip(-0.12, 0.12)

pos = np.asarray(positions, dtype=float).ravel()
N = len(df_bt)
if len(pos) < N: pos = np.concatenate([np.zeros(N-len(pos)), pos])
elif len(pos) > N: pos = pos[-N:]
df_bt["Position"] = pos
delta_pos = df_bt["Position"].diff().fillna(0.0)

active_ret = df_bt["Position"].shift(1).fillna(0.0) * df_bt["Return"] \
             - (TURNOVER_COST + SLIPPAGE_BPS) * delta_pos.abs()
df_bt["PortRet"] = PERCENT_CAPITAL * active_ret
df_bt["Equity"]  = (1.0 + df_bt["PortRet"]).cumprod()

# Buy&Hold FULL & TEST (aligned)
bh_full = bh_curve_from_prices(price_full_series, PERCENT_CAPITAL)
ROI_BH_FULL = (bh_full[-1] - 1.0) * 100.0

full_df = RAW
test_dates = full_df.loc[TEST_START:TEST_END].index if isinstance(full_df.index, pd.DatetimeIndex) else pd.RangeIndex(N)
bh_test = bh_test_from_full(price_full_series, test_dates, PERCENT_CAPITAL)
bh_series = pd.Series(bh_test)
if len(bh_series) != len(df_bt):
    if len(bh_series) > len(df_bt):
        bh_series = bh_series.iloc[-len(df_bt):].reset_index(drop=True)
    else:
        pad = pd.Series([bh_series.iloc[0]] * (len(df_bt) - len(bh_series)))
        bh_series = pd.concat([pad, bh_series], ignore_index=True)
df_bt["Benchmark"] = bh_series.values
ROI_BH_TEST = (df_bt["Benchmark"].iloc[-1] - 1.0) * 100.0

# Metrics
def max_drawdown(arr_like):
    x = np.asarray(arr_like, dtype=float)
    if x.size == 0: return 0.0
    peak = np.maximum.accumulate(x)
    return float((x / peak - 1.0).min())

ROI    = (df_bt["Equity"].iloc[-1] - 1.0) * 100.0
daily  = df_bt["PortRet"].to_numpy()
eq     = df_bt["Equity"].to_numpy()
sharpe = (daily.mean() / (daily.std() + 1e-12)) * np.sqrt(252.0) if daily.size>1 else 0.0
mdd    = max_drawdown(eq)

pos_arr = np.asarray(positions, float)
zero_ratio = float(np.mean(np.isclose(pos_arr, 0.0)))
mean_abs   = float(np.mean(np.abs(pos_arr)))

blocked = 0
for step in range(len(reporting_raw)):
    adx  = float(reporting_raw.loc[step, "ADX_RAW"])
    ema20= float(reporting_raw.loc[step, "EMA20_RAW"])
    ema50= float(reporting_raw.loc[step, "EMA50_RAW"])
    closepx = float(reporting_raw.loc[step, "CLOSE_RAW"])
    ema_gap = abs(ema20 - ema50) / max(abs(closepx), 1e-6)
    if (adx < TRADE_ADX_MIN) or (ema_gap < EMA_GAP_MIN):
        blocked += 1
blocked_ratio = blocked / max(1, len(reporting_raw))

print(f"\nTEST {TEST_START}–{TEST_END}")
print(f"Buy&Hold ROI (TEST):  {ROI_BH_TEST:.2f}%")
print(f"Buy&Hold ROI (FULL):  {ROI_BH_FULL:.2f}%")
print(f"Strategy ROI:         {ROI:.2f}% | Sharpe {sharpe:.2f} | MDD {mdd:.2%}")
print(f"Diagnostics: zero_ratio={zero_ratio:.1%} | mean|pos|={mean_abs:.3f} | blocked={blocked_ratio:.1%}")
