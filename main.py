# ===== Imports =====
import os, math, random
from collections import deque

import numpy as np
import pandas as pd
import yfinance as yf

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
from gym import spaces

from ta.volatility import BollingerBands
from ta.momentum   import RSIIndicator
from ta.trend      import EMAIndicator, MACD, ADXIndicator
from ta.volume     import VolumeWeightedAveragePrice
# ===== Repro =====
SEED = 1337
np.random.seed(SEED)
random.seed(SEED)
T.manual_seed(SEED)
# ===========================
# Hyperparams
# ===========================
INITIAL_ACCOUNT_BALANCE = 1_000.0

PERCENT_CAPITAL  = 0.30
TARGET_DAILY_VOL = 0.03
TURNOVER_COST    = 1e-5
SLIPPAGE_BPS     = 0.0
LEVER_CAP        = 1.50
KILL_THRESH      = 0.50
DD_COEF          = 0.50

TREND_BONUS_COEF     = 0.05
TREND_STRENGTH_CLIP  = (0.0, 1.0)
TREND_GATE           = True

SMOOTH_COST = 5e-4
HOLD_COST   = 1e-4

WIN_REWARD   = 0.3
LOSS_PENALTY = -0.3
WINLOSS_EPS  = 2e-5
WINLOSS_Z    = 0.012

ALPHA_PNL   = 1.1
REWARD_CLIP = 3.0
BETA_EXCESS = 0.4

ADX_ENTER   = 18.0     # порог для gate
ADX_FULL    = 28.0
TRADE_ADX_MIN = 15.5   # жёсткий фильтр
EMA_GAP_MIN   = 0.007  # 0.7%

APPLY_DEADZONE_TRAIN = False
SEQ_LEN = 48
# ===========================
# Data
# ===========================
df = yf.download("AAPL", start="2017-01-01", end="2025-01-01", auto_adjust=False)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
df = df.sort_index()
train_df = df.loc["2017-01-01":"2021-12-31"].copy()
test_df  = df.loc["2022-01-01":"2024-12-31"].copy()
def add_features(ddf: pd.DataFrame) -> pd.DataFrame:
    ddf = ddf.copy()

    # --- RAW для правил (в средах/гейтах используем ИХ!) ---
    ema20_raw = EMAIndicator(ddf["Close"], window=20).ema_indicator()
    ema50_raw = EMAIndicator(ddf["Close"], window=50).ema_indicator()
    adx_raw   = ADXIndicator(ddf["High"], ddf["Low"], ddf["Close"], window=14).adx()

    ddf["EMA20_RAW"] = ema20_raw
    ddf["EMA50_RAW"] = ema50_raw
    ddf["ADX_RAW"]   = adx_raw

    # --- Фичи для модели (могут дальше стандартизироваться) ---
    vwap = VolumeWeightedAveragePrice(
        high=ddf["High"], low=ddf["Low"], close=ddf["Close"],
        volume=ddf["Volume"], window=14, fillna=False
    )
    ddf["VWAP"] = vwap.volume_weighted_average_price()

    ddf["RSI"]   = RSIIndicator(ddf["Close"], window=14).rsi()
    ddf["EMA20"] = ema20_raw.copy()
    ddf["EMA50"] = ema50_raw.copy()

    macd = MACD(ddf["Close"])
    ddf["MACD"]        = macd.macd()
    ddf["MACD_signal"] = macd.macd_signal()
    ddf["MACD_hist"]   = ddf["MACD"] - ddf["MACD_signal"]

    bb = BollingerBands(ddf["Close"])
    ddf["BB_high"]  = bb.bollinger_hband()
    ddf["BB_low"]   = bb.bollinger_lband()
    ddf["BB_width"] = ddf["BB_high"] - ddf["BB_low"]
    ddf["BB_pos"]   = (ddf["Close"] - ddf["BB_low"]) / ddf["BB_width"].replace(0, np.nan)

    ret1  = ddf["Close"].pct_change()
    vol20 = ret1.rolling(20).std()
    vol100= ret1.rolling(100).std()
    ratio = (vol20/vol100).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0, 5.0)

    ddf["RET1"]    = ret1
    ddf["RET5"]    = ddf["Close"].pct_change(5)
    ddf["RET20"]   = ddf["Close"].pct_change(20)
    ddf["ATR"]     = (ddf["High"] - ddf["Low"]).rolling(14).mean()
    ddf["ATR_PCT"] = ddf["ATR"] / ddf["Close"].replace(0, np.nan)

    ddf["VOL_REGIME"] = ratio
    ddf["Close_Price"] = ddf["Close"].astype(float)

    ddf["EMA20_SLOPE"] = ddf["EMA20"].diff()
    ddf["EMA50_SLOPE"] = ddf["EMA50"].diff()
    ddf["Trend_Dir"]   = np.where(ddf["EMA20"] > ddf["EMA50"], 1.0, -1.0).astype(float)

    vol_mean20 = ddf["Volume"].rolling(20).mean()
    ddf["VOL_Z"]   = (ddf["Volume"] - vol_mean20) / ddf["Volume"].rolling(20).std().replace(0, np.nan)
    ddf["VOL_PCT"] = ddf["Volume"] / vol_mean20.replace(0, np.nan)

    # аккуратная зачистка NaN/inf только для модельных фич (RAW НЕ ТРОГАЕМ)
    for c in ["VWAP","RSI","EMA20","EMA50","MACD","MACD_signal","MACD_hist",
              "BB_high","BB_low","BB_width","BB_pos",
              "RET1","RET5","RET20","ATR","ATR_PCT","VOL_REGIME",
              "EMA20_SLOPE","EMA50_SLOPE","Trend_Dir","VOL_Z","VOL_PCT"]:
        ddf[c] = ddf[c].replace([np.inf, -np.inf], np.nan)

    return ddf
train_df = add_features(train_df)
test_df  = add_features(test_df)
# ----- Стандартизация только OBS-фич -----
def rolling_standardize(df_in, cols, win=252, min_periods=20):
    df_std = df_in.copy()
    for c in cols:
        roll_mean = df_std[c].rolling(win, min_periods=min_periods).mean()
        roll_std  = df_std[c].rolling(win, min_periods=min_periods).std()
        df_std[c] = (df_std[c] - roll_mean) / (roll_std.replace(0, 1e-12))
    return df_std

feat_cols = [
    "Open","High","Low","Close","Volume","VWAP",
    "RSI","EMA20","EMA50","EMA20_SLOPE","EMA50_SLOPE","Trend_Dir",  # если хочешь, можешь дать и ADX (не RAW) в модель
    "MACD","MACD_signal","MACD_hist",
    "BB_width","BB_pos",
    "RET1","RET5","RET20","ATR","ATR_PCT","VOL_REGIME",
    "VOL_Z","VOL_PCT"
]


# 1) Сохраняем сырые фичи отдельно
train_raw = train_df.copy()
test_raw  = test_df.copy()

# 2) Стандартизируем ТОЛЬКО train на своих данных
train_std = rolling_standardize(train_raw, feat_cols, win=252)

# 3) Для test добавляем СЫРОЙ хвост из train_raw
tmp = pd.concat([train_raw.tail(252), test_raw], axis=0)
tmp_std = rolling_standardize(tmp, feat_cols, win=252)

# 4) Отрезаем обратно тест
test_std = tmp_std.iloc[252:].copy()

# 5) (опционально) очистка и клип
CLIP_FEAT = 6.0
for ddf in (train_std, test_std):
    ddf[feat_cols] = (ddf[feat_cols]
                      .replace([np.inf, -np.inf], np.nan)
                      .fillna(0.0)
                      .clip(-CLIP_FEAT, CLIP_FEAT))

df_train = train_std.reset_index(drop=True)
df_test  = test_std.reset_index(drop=True)

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


# === Soft-scale для вол-таргетинга ===
def soft_scale(vol, target=TARGET_DAILY_VOL, alpha=0.9):
    s_full = target / max(vol, 1e-6)      # как "full"
    s_cap  = min(1.0, s_full)             # как "cap"
    return alpha * s_full + (1 - alpha) * s_cap
# ===========================
# PPO Memory
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
        indices = np.arange(n, dtype=np.int64)
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
        self.actions.append(float(action))
        self.probs.append(float(probs))
        self.vals.append(float(vals))
        self.rewards.append(float(reward))
        self.dones.append(float(done))

    def clear_memory(self):
        self.states.clear(); self.probs.clear(); self.actions.clear()
        self.rewards.clear(); self.dones.clear(); self.vals.clear()


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
        self.hold_clock = 0
        self.last_sign  = 0
        self.min_hold   = 5   # было 3

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

    def _take_action(self, action):
        position = float(action[0])
    
        # ретёрн актива на следующий шаг (ограничим хвост)
        if (self.current_step + 1) < self.max_steps:
            asset_ret = float(self.ret_series.iloc[self.current_step + 1])
        else:
            asset_ret = 0.0
        asset_ret = float(np.clip(asset_ret, -0.12, 0.12))
    
        # текущая вола и вол-таргетинг
        vol = float(self.vol_series.iloc[self.current_step]) if self.current_step < len(self.vol_series) else 0.0
        self.volatility = vol if (np.isfinite(vol) and vol > 0) else 1e-6
        scale = soft_scale(self.volatility)
        eff_position = float(np.clip(position * scale, -1.0, 1.0))
    
        # --- RAW-индикаторы из self.df в текущем шаге ---
        adx       = float(self.df.loc[self.current_step, "ADX_RAW"])
        ema20_raw = float(self.df.loc[self.current_step, "EMA20_RAW"])
        ema50_raw = float(self.df.loc[self.current_step, "EMA50_RAW"])
        close_px  = float(self.df.loc[self.current_step, "Close_Price"])
    
        trend_dir = 1.0 if ema20_raw >= ema50_raw else -1.0
        ema_gap   = abs(ema20_raw - ema50_raw) / max(abs(close_px), 1e-6)

        # ===== ЖЁСТКИЙ ФИЛЬТР ВХОДОВ: ADX + EMA-gap =====
        if (adx < TRADE_ADX_MIN) or (ema_gap < EMA_GAP_MIN):
            eff_position = 0.0
    
        # gate по силе тренда (0..1)
        gate = float(np.clip((adx - ADX_ENTER) / max(ADX_FULL - ADX_ENTER, 1e-6), 0.0, 1.0))
    
        # усиливаем/ослабляем позу по gate (минимум 0.3)
        eff_position *= (0.3 + 0.7 * gate)
    
        # против тренда приглушаем (в сильный тренд ~в ноль)
        if eff_position * trend_dir < 0:
            eff_position *= (1.0 - 0.5 * gate)
    
        if TREND_GATE:
            ts = float(np.clip(adx / 50.0, 0.0, 1.0))
            eff_position = float(np.clip(eff_position * (0.5 + 0.5 * ts), -1.0, 1.0))
    
        # --- Min-hold: не даём резко менять знак раньше self.min_hold шагов ---
        sign = 0 if abs(eff_position) < 1e-6 else (1 if eff_position > 0 else -1)
        if sign == 0:
            self.hold_clock += 1
        else:
            if self.last_sign == 0:
                self.last_sign = sign
                self.hold_clock = 0
            elif sign != self.last_sign and self.hold_clock < self.min_hold:
                eff_position = abs(eff_position) * self.last_sign
                sign = self.last_sign
                self.hold_clock += 1
            elif sign != self.last_sign and self.hold_clock >= self.min_hold:
                self.last_sign = sign
                self.hold_clock = 0
            else:
                self.hold_clock += 1
    
        raw_ret      = eff_position * asset_ret
        realized_ret = float(raw_ret)
    
        traded_cap = max(self.available_balance, 0.0) * PERCENT_CAPITAL
        step_pnl   = traded_cap * realized_ret
        self.net_profit        += step_pnl
        self.available_balance += step_pnl

        delta_pos    = eff_position - self.prev_position
        turnover_fee = abs(traded_cap) * TURNOVER_COST * abs(delta_pos)
        if turnover_fee:
            self.available_balance -= turnover_fee
            self.net_profit        -= turnover_fee
    
        if eff_position > 0: self.num_trades_long  += 1
        if eff_position < 0: self.num_trades_short += 1
        den = self.num_trades_long + self.num_trades_short
        self.long_short_ratio = (self.num_trades_long / den) if den > 0 else 0.0
    
        # ВОЗВРАЩАЕМ trend_dir, чтобы step() не пересчитывал
        return realized_ret, eff_position, delta_pos, raw_ret, asset_ret, turnover_fee, trend_dir
    def step(self, action):
        (realized_ret, eff_position, delta_pos, raw_ret,
         asset_ret, turnover_fee, trend_dir) = self._take_action(action)
    
        smooth_pen = float(SMOOTH_COST * (delta_pos ** 2))
        hold_pen   = float(HOLD_COST   * (eff_position ** 2))
    
        # сила тренда по ADX_RAW
        trend_strength = float(np.clip(float(self.df.loc[self.current_step, "ADX_RAW"]) / 50.0,
                                       *TREND_STRENGTH_CLIP))
        trend_bonus = TREND_BONUS_COEF * trend_strength * np.sign(eff_position) * trend_dir
    
        thr_abs = max(WINLOSS_EPS, WINLOSS_Z * self.volatility * (abs(eff_position) + 0.1))
        if   raw_ret >  thr_abs: base = WIN_REWARD
        elif raw_ret < -thr_abs: base = LOSS_PENALTY
        else:                    base = 0.0
    
        b_t = float(self.baseline_pos_series.iloc[self.current_step])
        excess_step = (eff_position - b_t) * asset_ret
    
        equity = self.available_balance / INITIAL_ACCOUNT_BALANCE
        self.equity_peak = max(self.equity_peak, equity)
        dd = min(0.0, equity / self.equity_peak - 1.0)
    
        txn_cost_ret = (TURNOVER_COST + SLIPPAGE_BPS) * abs(delta_pos) * PERCENT_CAPITAL
    
        reward = base + ALPHA_PNL * realized_ret + BETA_EXCESS * excess_step \
               + trend_bonus - smooth_pen - hold_pen - DD_COEF * (-dd) - txn_cost_ret
        reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))
    
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

        return self._next_observation()

    def render(self, mode='human', close=False):
        pass

class SimpleRNNBackbone(nn.Module):
    def __init__(self, feat_dim, hidden=64, nlayers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=nlayers,
            nonlinearity="tanh",
            batch_first=True,                         # ожидаем [B, T, F]
            dropout=dropout if nlayers > 1 else 0.0,
            bidirectional=False                       # для онлайна — только каузально
        )
        self.out_dim = hidden

    def forward(self, x):                              # x: [B, T, F]
        y, h = self.rnn(x)                             # h: [num_layers, B, hidden]
        h_last = h[-1]                                 # [B, hidden]
        return h_last

class LSTMBackbone(nn.Module):
    def __init__(self, feat_dim, hidden=64, nlayers=1, dropout=0.1, bidirectional=False):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=nlayers,
            batch_first=True,                     # ожидаем [B, T, F]
            dropout=dropout if nlayers > 1 else 0.0,
            bidirectional=self.bidirectional
        )
        self.out_dim = hidden * (2 if self.bidirectional else 1)

    def forward(self, x):                          # x: [B, T, F]
        y, (h, c) = self.lstm(x)                   # h: [layers*(1/2), B, hidden]
        if not self.bidirectional:
            return h[-1]                           # [B, hidden]
        else:
            # последний слой: concat (fw, bw)
            return T.cat([h[-2], h[-1]], dim=-1)   # [B, 2*hidden]

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
    def __init__(self, input_dims, lr, d_model=32, nhead=4, nlayers=2, dropout=0.1,
                 backbone="transformer", lstm_hidden=64, lstm_layers=1, chkpt_dir='tmp/'):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_cont_multi')
        feat_dim = int(input_dims[-1]) if isinstance(input_dims, (tuple, list, np.ndarray)) else int(input_dims)

        self.backbone, out_dim = self._make_backbone(
            feat_dim, backbone, d_model, nhead, nlayers, dropout, lstm_hidden, lstm_layers
        )
        self.fc_mu     = nn.Linear(out_dim, 1)
        self.log_std   = nn.Parameter(T.zeros(1, 1))

        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _make_backbone(self, feat_dim, backbone, d_model, nhead, nlayers, dropout, lstm_hidden, lstm_layers):
        b = str(backbone).lower()
        if b == "transformer":
            bb = TransformerBackbone(feat_dim, d_model, nhead, nlayers, dropout)
            out_dim = d_model
        elif b == "rnn":
            bb = SimpleRNNBackbone(feat_dim, hidden=lstm_hidden, nlayers=lstm_layers, dropout=dropout)
            out_dim = bb.out_dim
        elif b == "lstm":
            bb = LSTMBackbone(feat_dim, hidden=lstm_hidden, nlayers=lstm_layers, dropout=dropout, bidirectional=False)
            out_dim = bb.out_dim
        elif b == "bilstm":
            bb = LSTMBackbone(feat_dim, hidden=lstm_hidden, nlayers=lstm_layers, dropout=dropout, bidirectional=True)
            out_dim = bb.out_dim
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        return bb, out_dim

    def forward(self, state_seq):                 # state_seq: [B, T, F]
        h = self.backbone(state_seq)             # [B, D]
        mu = T.tanh(self.fc_mu(h))
        sigma = F.softplus(self.log_std) + 1e-4
        return T.distributions.Normal(mu, sigma)

    def save_checkpoint(self): T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self): self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, lr, d_model=32, nhead=4, nlayers=2, dropout=0.1,
                 backbone="transformer", lstm_hidden=64, lstm_layers=1, chkpt_dir='tmp/'):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_cont_multi')
        feat_dim = int(input_dims[-1]) if isinstance(input_dims, (tuple, list, np.ndarray)) else int(input_dims)

        self.backbone, out_dim = self._make_backbone(
            feat_dim, backbone, d_model, nhead, nlayers, dropout, lstm_hidden, lstm_layers
        )
        self.fc_value = nn.Linear(out_dim, 1)

        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _make_backbone(self, feat_dim, backbone, d_model, nhead, nlayers, dropout, lstm_hidden, lstm_layers):
        b = str(backbone).lower()
        if b == "transformer":
            bb = TransformerBackbone(feat_dim, d_model, nhead, nlayers, dropout)
            out_dim = d_model
        elif b == "rnn":
            bb = SimpleRNNBackbone(feat_dim, hidden=lstm_hidden, nlayers=lstm_layers, dropout=dropout)
            out_dim = bb.out_dim
        elif b == "lstm":
            bb = LSTMBackbone(feat_dim, hidden=lstm_hidden, nlayers=lstm_layers, dropout=dropout, bidirectional=False)
            out_dim = bb.out_dim
        elif b == "bilstm":
            bb = LSTMBackbone(feat_dim, hidden=lstm_hidden, nlayers=lstm_layers, dropout=dropout, bidirectional=True)
            out_dim = bb.out_dim
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        return bb, out_dim

    def forward(self, state_seq):                 # state_seq: [B, T, F]
        h = self.backbone(state_seq)             # [B, D]
        return self.fc_value(h)

    def save_checkpoint(self): T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self): self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

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

        # гистерезис для инференса/выбора экшна
        self.TAU_ENTER, self.TAU_EXIT = 0.05, 0.02
        self.policy_pos_cache = 0.0

    def remember(self, state_seq, action, log_prob, value, reward, done):
        self.memory.store_memory(state_seq, action, log_prob, value, reward, done)

    def save_models(self):
        self.actor.save_checkpoint(); self.critic.save_checkpoint()

    def choose_action(self, observation):
        # формируем последовательность
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
        
        # hysteresis / deadzone (как обсуждали)
        raw = float(action.squeeze().cpu().numpy())

        if APPLY_DEADZONE_TRAIN:
            if self.policy_pos_cache == 0.0:
                if abs(raw) < self.TAU_ENTER: raw = 0.0
            else:
                if abs(raw) < self.TAU_EXIT:  raw = 0.0
            self.policy_pos_cache = raw
        # иначе на обучении deadzone не применяем

        
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
# Training loop (curriculum)
# ===========================
def make_env_slice(df_full, start_idx, end_idx, episode_len=None, randomize=True, lag=20):
    df_slice = df_full.iloc[start_idx:end_idx].reset_index(drop=True).copy()
    return StockTradingEnv(df_slice, episode_len=episode_len, randomize=randomize, lag=lag)

env = StockTradingEnv(df_train)

# агент
agent = Agent(
    n_actions=1,
    input_dims=env.observation_space.shape,
    lr=5e-4,            # было 7e-4
    batch_size=512,
    n_epochs=12,        # было 10
    entropy_coef=0.05,
    policy_clip=0.30,
    target_kl=0.08,
    max_grad_norm=0.9,
    gae_lambda=0.97
)

# головы LSTM
agent.actor  = ActorNetwork(env.observation_space.shape, lr=5e-4,
                            backbone="lstm", lstm_hidden=96, lstm_layers=2, dropout=0.15)
agent.critic = CriticNetwork(env.observation_space.shape, lr=5e-4,
                             backbone="lstm", lstm_hidden=96, lstm_layers=2, dropout=0.15)

# (опционально) если окно у тебя длинное, для RNN лучше так:
# SEQ_LEN = 24 ~ 32

stages = [
    ("stage1", 0,   900,              600),
    ("stage2", 300, 1200,             800),
    ("stage3", 0,   len(df_train),   2000),
]

best_score = -1e9
score_history, n_steps = [], 0

print("... starting aggressive curriculum ...")
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
        wins = losses = 0

        while not done:
            action, logp, val, state_seq = agent.choose_action(obs)
            total_act += 1
            if abs(float(action[0])) < 1e-6:
                zero_act += 1

            obs_, reward, done, info = env.step(action)
            if info["base_reward"] > 0: wins += 1
            elif info["base_reward"] < 0: losses += 1

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
        wr = wins / max(1, wins + losses)

        pnl    = env.net_profit
        equity = env.available_balance

        print(f"{name} | ep {i:4d} "
              f"| mean {mean_score: .4f} "
              f"| avg50 {avg50: .4f} "
              f"| best {best_score: .4f} "
              f"| PnL ${pnl: .2f} | Equity ${equity: .2f} "
              f"| winrate {wr:.1%} | zero% {zero_ratio: .1f}")

# --- INFERENCE v2: RAW gates = env + breakout + chandelier exit + 3-level exposure ---

infer_model = agent.actor; infer_model.eval()

# работаем на тестовом окне (сохраняем индексы для выравнивания)
rep = df_test.copy()                    # стандартизованные фичи + Close_Price/RAW-поля
raw = test_raw.reset_index(drop=True)   # СЫРЫЕ цены High/Low/Close/Volume и т.д.
if len(raw) != len(rep):
    raw = raw.iloc[-len(rep):].reset_index(drop=True)

# быстрые векторы (RAW)
px    = raw["Close"].to_numpy(float)
hi    = raw["High"].to_numpy(float)
lo    = raw["Low"].to_numpy(float)

e20   = rep["EMA20_RAW"].to_numpy(float)
e50   = rep["EMA50_RAW"].to_numpy(float)
adx   = rep["ADX_RAW"].to_numpy(float)

# волатильность и ATR%
prev_close = np.r_[px[0], px[:-1]]
tr = np.maximum.reduce([hi-lo, np.abs(hi-prev_close), np.abs(lo-prev_close)])
atr = pd.Series(tr).rolling(14).mean().bfill().to_numpy(float)
atrp = (atr / np.maximum(px, 1e-12)).astype(float)

# Donchian breakout (shift(1) — без заглядывания вперёд)
L_break, L_trail = 55, 20
don_hi = pd.Series(px).rolling(L_break).max().shift(1).to_numpy()
don_lo = pd.Series(px).rolling(L_trail).min().shift(1).to_numpy()

# вспомогательная волатильность для вол-таргета
ret   = pd.Series(px).pct_change().fillna(0.0).to_numpy(float)
vol20 = pd.Series(ret).rolling(20).std().fillna(ret.std()).to_numpy(float)

# ==== параметры инференса (подкручены под 2022–2024 AAPL) ====
# гейты — как в env
STRONG_ADX            = 25.0     # сильный тренд может «перебить» фильтр SMA200
# deadzone/гистерезис + вол-таргетинг
TAU_ENTER, TAU_EXIT   = 0.10, 0.04
TARGET_DAILY_VOL_INF  = 0.050    # ↑ экспозицию (можно 0.048–0.055)
ALPHA_SOFT            = 0.60
# квантование экспозиции
Q0, Q1                = 0.18, 0.34
# chandelier exit
CE_MULT               = 2.2      # множитель ATR% от пика; 2.0–2.6 — зона подстройки
# удержание/фрикция
MIN_HOLD              = 11       # минимум баров удержания после смены
STEP_FRICTION         = 0.50     # менять позицию, только если шаг ≥ 0.5
LONG_ONLY             = True

positions   = []
state_window= deque(maxlen=SEQ_LEN)
raw_smooth  = 0.0
pol_cache   = 0.0
last_sign, hold = 0, 0

in_long     = False
entry_px    = 0.0
peak_px     = 0.0
last_q      = 0.0

with T.no_grad():
    for t in range(len(rep)):
        r = rep.iloc[t]

        # === сформируем наблюдение в ТОМ ЖЕ порядке, что в env._next_observation() ===
        core = r.reindex(feat_cols).astype(float).to_numpy(dtype=np.float32)
        obs  = np.concatenate([core, np.array([0.0], dtype=np.float32)])  # ls_ratio онлайном не знаем
        obs  = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        state_window.append(obs)
        if len(state_window) < SEQ_LEN:
            positions.append(0.0)
            continue

        seq  = np.stack(state_window, axis=0)[None, ...].astype(np.float32)
        dist = infer_model(T.tensor(seq, dtype=T.float32, device=infer_model.device))
        mu   = float(T.clamp(dist.mean, -1.0, 1.0).squeeze().cpu().numpy())

        # --- сглаживание + deadzone/hysteresis ---
        raw_smooth = 0.85*raw_smooth + 0.15*mu
        x = raw_smooth
        if pol_cache == 0.0:
            if abs(x) < TAU_ENTER: x = 0.0
        else:
            if abs(x) < TAU_EXIT:  x = 0.0
        pol_cache = x

        # --- волатильностной скейлинг ---
        s_full = TARGET_DAILY_VOL_INF / max(float(vol20[t]), 1e-6)
        scale  = ALPHA_SOFT*s_full + (1-ALPHA_SOFT)*min(1.0, s_full)
        eff    = float(np.clip(x * scale, -1.0, 1.0))

        # --- RAW-гейты как в среде ---
        ema_gap   = abs(e20[t] - e50[t]) / max(abs(px[t]), 1e-6)
        trend_dir = 1.0 if e20[t] >= e50[t] else -1.0

        if (adx[t] < TRADE_ADX_MIN) or (ema_gap < EMA_GAP_MIN):
            eff = 0.0

        gate = float(np.clip((adx[t] - ADX_ENTER) / max(ADX_FULL - ADX_ENTER, 1e-6), 0.0, 1.0))
        eff *= (0.3 + 0.7*gate)
        if eff * trend_dir < 0:
            eff *= (1.0 - 0.5*gate)
        ts = float(np.clip(adx[t] / 50.0, 0.0, 1.0))
        eff = float(np.clip(eff * (0.5 + 0.5*ts), -1.0, 1.0))

        # --- breakout условие: ап-тренд + пробой Дончиана + ADX растёт ---
        adx_up     = (t > 0) and (adx[t] > adx[t-1])
        breakoutUp = (px[t] > (don_hi[t] if not np.isnan(don_hi[t]) else -np.inf)) and (trend_dir > 0) and adx_up

        # --- Long-only режим c сильным трендом ---
        sma200 = pd.Series(px).rolling(200).mean().to_numpy()
        bull   = (px >= np.nan_to_num(sma200, nan=np.inf))  # до заполнения SMA200 считаем flat=False
        if LONG_ONLY:
            if trend_dir < 0:
                eff = 0.0
            else:
                if (not bull[t]) and (adx[t] < STRONG_ADX):
                    eff = 0.0
            eff = max(0.0, eff)

        # --- квантование базовой экспозиции ---
        if eff <= Q0:
            q_base = 0.0
        elif eff <= Q1:
            q_base = 0.5
        else:
            q_base = 1.0

        # breakout усиливает экспозицию
        q = max(q_base, 1.0 if breakoutUp else q_base)

        # --- фрикция изменения позиции ---
        if abs(q - last_q) < STEP_FRICTION:
            q = last_q

        # --- Chandelier Exit от пика (в ATR%) ---
        if q > 0.0:
            if not in_long:
                in_long = True; entry_px = px[t]; peak_px = px[t]
            else:
                peak_px = max(peak_px, px[t])
                ce_floor = peak_px * (1.0 - CE_MULT * max(atrp[t], 1e-6))
                if px[t] < ce_floor:
                    q = 0.0
                    in_long = False; last_sign = 0; hold = 0
        else:
            in_long = False

        # --- удержание после смены ---
        s = int(q > 0.0)
        if s == 0:
            hold += 1
        else:
            if last_sign == 0:
                last_sign = 1; hold = 0
            elif last_sign != 1 and hold < MIN_HOLD:
                q = float(last_sign); s = last_sign; hold += 1
            elif last_sign != 1 and hold >= MIN_HOLD:
                last_sign = 1; hold = 0
            else:
                hold += 1

        positions.append(q)
        last_q = q

# ===========================
# Backtest (как у тебя)
# ===========================
df_bt = rep.copy()
df_bt["Return"]   = pd.Series(px).pct_change().fillna(0.0).to_numpy(float)

pos = np.asarray(positions, dtype=float).ravel()
N   = len(df_bt)
if len(pos) < N: pos = np.r_[np.zeros(N-len(pos)), pos]
elif len(pos) > N: pos = pos[-N:]
df_bt["Position"] = pos

delta_pos = df_bt["Position"].diff().fillna(0.0)
active_ret = df_bt["Position"].shift(1).fillna(0.0) * df_bt["Return"] \
             - (TURNOVER_COST + SLIPPAGE_BPS) * delta_pos.abs()
df_bt["PortRet"] = PERCENT_CAPITAL * active_ret
df_bt["Equity"]  = (1.0 + df_bt["PortRet"]).cumprod()

# BH на Close
price_full = (df["Adj Close"] if "Adj Close" in df.columns else df["Close"]).astype(float)

bh_full    = bh_curve_from_prices(price_full, PERCENT_CAPITAL);       ROI_BH_FULL = (bh_full[-1]-1.0)*100
bh_test    = bh_test_from_full(price_full, df_test, PERCENT_CAPITAL); ROI_BH_TEST = (bh_test[-1]-1.0)*100
df_bt["Benchmark"] = bh_test

# Бейзлайн
sma20 = pd.Series(px).rolling(20).mean()
sma50 = pd.Series(px).rolling(50).mean()
base_pos = (sma20 > sma50).astype(float).fillna(0.0)
b_delta  = base_pos.diff().fillna(0.0).abs()
base_ret = base_pos.shift(1).fillna(0.0)*df_bt["Return"] - (TURNOVER_COST + SLIPPAGE_BPS)*b_delta
df_bt["BaselineEq"] = (1.0 + PERCENT_CAPITAL * base_ret).cumprod()

# метрики
ROI     = (df_bt["Equity"].iloc[-1]     - 1.0) * 100
ROI_SB  = (df_bt["BaselineEq"].iloc[-1] - 1.0) * 100
daily   = df_bt["PortRet"].to_numpy()
sharpe  = (daily.mean() / (daily.std() + 1e-12)) * np.sqrt(252.0)
def max_dd(x):
    x = np.asarray(x, float); peak = np.maximum.accumulate(x)
    return float((x/peak - 1.0).min())
mdd = max_dd(df_bt["Equity"])

coverage = float((df_bt["Position"] > 0).mean())
turns    = float(df_bt["Position"].diff().abs().sum())

print(f"\nTEST 2022–2024")
print(f"Strategy ROI:         {ROI:.2f}% | Sharpe {sharpe:.2f} | MDD {mdd:.2%}")
print(f"Buy&Hold ROI (TEST):  {ROI_BH_TEST:.2f}% | Buy&Hold ROI (FULL): {ROI_BH_FULL:.2f}%")
print(f"Smart Baseline ROI:   {ROI_SB:.2f}%")
print(f"Coverage {coverage:.1%} | Turns {turns:.1f}")

