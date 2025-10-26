# ============================================
# Universal RL Trader + Multi-Ticker News + Markov Regimes
# (Boolean-safety + Stable Markov injection + News k floor + Diagnostics)
# ============================================
# pip install torch pandas numpy yfinance ta gym scikit-learn

import os, shutil, random, warnings, glob
warnings.filterwarnings("ignore")

from collections import deque
from typing import Optional, Dict, List, Tuple
import datetime as _dt

import numpy as np
import pandas as pd
import yfinance as yf

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from urllib.parse import urlparse
from pandas.tseries.offsets import BusinessDay as BDay

from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

# ------------- Repro / Device / AMP -------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); T.manual_seed(SEED)
if T.cuda.is_available(): T.cuda.manual_seed_all(SEED)
DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")
USE_AMP = bool(T.cuda.is_available())
AMP_DTYPE = T.float16
TRY_COMPILE = True

if hasattr(T, "set_float32_matmul_precision"):
    try:
        T.set_float32_matmul_precision("high")
    except Exception:
        pass

def maybe_compile_forward(module: nn.Module):
    if TRY_COMPILE and hasattr(T, "compile"):
        try:
            module.forward = T.compile(module.forward)
        except Exception as e:
            print(f"[compile] skip {module.__class__.__name__}: {e}")

# ============ RUN SETTINGS / DIAGNOSTICS ============
RUN_TS = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"boolean_safe_{RUN_TS}"
CHKPT_DIR = f"tmp/{RUN_ID}"
os.makedirs(CHKPT_DIR, exist_ok=True)

CLEAN_TMP = False
if CLEAN_TMP and os.path.isdir("tmp"):
    for p in glob.glob("tmp/*"):
        try:
            shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass
    os.makedirs(CHKPT_DIR, exist_ok=True)

# ============================================
# Гиперпараметры
# ============================================
FAST = True
INITIAL_ACCOUNT_BALANCE = 1_000.0
PERCENT_CAPITAL  = 0.30
TARGET_DAILY_VOL = 0.03
TURNOVER_COST    = 1e-5
SLIPPAGE_BPS     = 0.0
LEVER_CAP        = 1.50
KILL_THRESH      = 0.50
DD_COEF          = 0.50
TREND_BONUS_COEF = 0.05
TREND_STRENGTH_CLIP = (0.0, 1.0)
TREND_GATE = True
SMOOTH_COST = 5e-4
HOLD_COST   = 1e-4
WIN_REWARD, LOSS_PENALTY = 0.3, -0.3
WINLOSS_EPS, WINLOSS_Z = 2e-5, 0.012
ALPHA_PNL, REWARD_CLIP, BETA_EXCESS = 1.1, 3.0, 0.4
ADX_ENTER, ADX_FULL = 18.0, 28.0
APPLY_DEADZONE_TRAIN = False
SEQ_LEN = 48

# PPO / оптимизация
LR = 5e-4 if FAST else 3e-4
BATCH_SIZE = 384 if FAST else 512
N_EPOCHS = 8 if FAST else 12
ENTROPY_COEF = 0.02   # (понижено для стабильности)
VAL_EVERY = 60 if FAST else 120
STAGES = [
    ("stage1", 0,    900,   220 if FAST else 600),
    ("stage2", 300, 1200,   240 if FAST else 800),
    ("stage3", 0,   None,   300 if FAST else 1200),
]

# ============ NEWS / MARKOV TOGGLES ============
FORCE_NEWS_K: Optional[float] = 0.20   # Минимальный видимый эффект новостей; поставьте None, чтобы калибровать без пола
ENABLE_SURPRISE_SOFTGATE = True
LONG_ONLY = True
WEAKEN_RAW_GATES_FOR_TEST = False

# =========================================================
# ===================== NEWS IMPACT =======================
# =========================================================
def _to_trading_day(ts_local: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(ts_local):
        return pd.NaT
    d = pd.Timestamp(ts_local).normalize()
    if d.weekday() >= 5:
        d = d + BDay(1)
    if pd.Timestamp(ts_local).hour >= 16:
        d = d + BDay(1)
    return d

def _canon_title(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.lower().strip().split())

def _domain_from_url(u: str) -> str:
    try:
        net = urlparse(u).netloc.lower()
        return net[4:] if net.startswith("www.") else net
    except Exception:
        return ""

def _normalize_one_news_df(path: str, ticker_hint: Optional[str] = None) -> pd.DataFrame:
    if (not isinstance(path, str)) or (len(path) == 0) or (not os.path.exists(os.path.expanduser(path))):
        print(f"[news] file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(os.path.expanduser(path))
    if df is None or df.empty:
        return pd.DataFrame()

    date_col = next((c for c in ["date","datetime","pubDate","published_at","time","timestamp"] if c in df.columns), None)
    if date_col is None:
        raise RuntimeError(f"[news] '{os.path.basename(path)}' has no date column")
    ts = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    if ts.isna().all():
        ts = pd.to_datetime(df[date_col], errors="coerce")
    if getattr(ts, "dt", None) is not None and ts.dt.tz is None:
        try:
            ts = ts.dt.tz_localize("UTC")
        except Exception:
            ts = ts.dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT")
    df["date_utc"]   = ts
    df["date_local"] = df["date_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None)

    title_col = "title" if "title" in df.columns else ("headline" if "headline" in df.columns else None)
    url_col   = "url"   if "url"   in df.columns else ("link"     if "link"     in df.columns else None)
    if title_col is None:
        df["title"] = df.get("headline", "").fillna("").astype(str); title_col = "title"
    if url_col is None:
        df["url"] = df.get("link", "").fillna("").astype(str); url_col = "url"

    label_col = next((c for c in ["label","sentiment","class","polarity"] if c in df.columns), None)
    if label_col is None:
        score_col = next((c for c in ["sent_score","score"] if c in df.columns), None)
        if score_col is not None:
            thr_pos, thr_neg = 0.15, -0.15
            lab = np.where(df[score_col] >= thr_pos, "positive",
                           np.where(df[score_col] <= thr_neg, "negative", "neutral"))
            df["label"] = lab; label_col = "label"
    if label_col is None:
        raise RuntimeError(f"[news] '{os.path.basename(path)}' has no label/sentiment column")

    tcol = next((c for c in ["impact_tier","impact_strength","tier","impact"] if c in df.columns), None)
    wcol = next((c for c in ["impact_weight","weight","impact_score","score_weight"] if c in df.columns), None)
    tier_map = {"weak":"weak","слабый":"weak","low":"weak","низкий":"weak",
                "medium":"medium","средний":"medium","mid":"medium",
                "strong":"strong","сильный":"strong","high":"strong","высокий":"strong"}
    df["impact_tier"] = (df[tcol].astype(str).str.lower().map(tier_map).fillna("weak") if tcol else "weak")
    if wcol is None:
        tier2w = {"weak":0.25,"medium":0.55,"strong":0.85}
        df["impact_weight"] = df["impact_tier"].map(tier2w).astype(float)
    else:
        df["impact_weight"] = pd.to_numeric(df[wcol], errors="coerce").fillna(0.0).clip(0, 1)

    df["trading_day"] = df["date_local"].apply(_to_trading_day)
    df["title_canon"] = df[title_col].astype(str).map(_canon_title)
    df["domain"]      = df[url_col].astype(str).map(_domain_from_url)
    if ticker_hint is not None:
        df["ticker"] = str(ticker_hint)

    df = df.sort_values(["trading_day","impact_weight"])
    df = df.drop_duplicates(subset=["trading_day","title_canon","domain"], keep="last")
    return df

def build_daily_impact_from_file(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df is None or news_df.empty:
        return pd.DataFrame()
    df = news_df.copy()
    map_lbl = {
        "bullish":"bullish","bull":"bullish","long":"bullish","positive":"bullish","pos":"bullish","позитив":"bullish","бычий":"bullish",
        "bearish":"bearish","bear":"bearish","short":"bearish","negative":"bearish","neg":"bearish","негатив":"bearish","медвежий":"bearish",
        "neutral":"neutral","нейтральный":"neutral","нейтрал":"neutral"
    }
    lbl2sgn = {"bullish": +1.0, "bearish": -1.0, "neutral": 0.0}
    df["label_norm"] = df["label"].astype(str).str.lower().map(map_lbl).fillna("neutral")
    df["sent_sign"] = df["label_norm"].map(lbl2sgn).astype(float)
    df["signed_weight"] = df["sent_sign"] * df["impact_weight"]

    agg = (df.groupby("trading_day", as_index=False)
             .agg(NewsCount=("title_canon","count"),
                  BullCt=("label_norm", lambda s: (s=="bullish").sum()),
                  BearCt=("label_norm", lambda s: (s=="bearish").sum()),
                  NeuCt =("label_norm", lambda s: (s=="neutral").sum()),
                  ImpactWeighted=("signed_weight","sum"),
                  ImpactWeightedMean=("signed_weight","mean"),
                  ImpactWSumAbs=("impact_weight","sum")))
    tier_ct = (df.pivot_table(index="trading_day", columns="impact_tier",
                              values="title_canon", aggfunc="count")
                 .fillna(0).reset_index())
    tier_ct.columns = ["trading_day"] + [f"Tier_{c}_Ct" for c in tier_ct.columns[1:]]
    out = pd.merge(agg, tier_ct, on="trading_day", how="left").fillna(0)
    out["Date"] = pd.to_datetime(out["trading_day"])
    out = out.drop(columns=["trading_day"]).set_index("Date").sort_index()

    roll = out["NewsCount"].rolling(60, min_periods=10)
    out["NewsCtZ"] = (out["NewsCount"] - roll.mean()) / (roll.std() + 1e-12)
    out["ImpactWeighted_EMA10"] = out["ImpactWeighted"].ewm(span=10, adjust=False).mean()
    out["ImpactWeighted_EMA20"] = out["ImpactWeighted"].ewm(span=20, adjust=False).mean()
    out["ImpactSurprise"] = out["ImpactWeighted"] - out["ImpactWeighted_EMA20"]
    return out

def build_news_factor_from_daily(daily_impacts: pd.DataFrame, k=0.40, scale=1.20, cap=0.50) -> Dict[pd.Timestamp, float]:
    if daily_impacts is None or daily_impacts.empty:
        return {}
    x = daily_impacts["ImpactSurprise"].copy()
    f = 1.0 + float(k) * np.tanh(x / float(scale))
    f = f.clip(1.0 - cap, 1.0 + cap)
    s = pd.Series(f.values, index=pd.DatetimeIndex(daily_impacts.index))
    # ключи как даты (без времени), чтобы .normalize() идеально маппился
    return {pd.Timestamp(d).date(): float(v) for d, v in s.items()}

NEWS_BY_TICKER: Dict[str, pd.DataFrame] = {}
DAILY_IMPACTS_BY_TICKER: Dict[str, pd.DataFrame] = {}

def load_news_many(news_sources: dict, single_file: Optional[str] = None, tickers_whitelist=None):
    NEWS_BY_TICKER.clear(); DAILY_IMPACTS_BY_TICKER.clear()
    if isinstance(single_file, str) and len(single_file) > 0:
        hint = os.path.splitext(os.path.basename(single_file))[0].upper()
        df_one = _normalize_one_news_df(single_file, ticker_hint=hint)
        if df_one is not None and not df_one.empty:
            NEWS_BY_TICKER[hint] = df_one
            DAILY_IMPACTS_BY_TICKER[hint] = build_daily_impact_from_file(df_one)
        print(f"[news] loaded single file for {hint}: {len(df_one) if df_one is not None else 0}")
        return
    for tic, path in (news_sources or {}).items():
        if (tickers_whitelist is not None) and (tic not in tickers_whitelist):
            continue
        try:
            df = _normalize_one_news_df(path, ticker_hint=tic)
        except Exception as e:
            print(f"[news] {tic}: FAILED -> {e}")
            df = pd.DataFrame()
        NEWS_BY_TICKER[tic] = df
        DAILY_IMPACTS_BY_TICKER[tic] = build_daily_impact_from_file(df) if (df is not None and not df.empty) else pd.DataFrame()
        print(f"[news] {tic}: {len(df)} articles | {len(DAILY_IMPACTS_BY_TICKER[tic])} daily")

# ---------- Укажи свои пути к CSV ----------
NEWS_SOURCES = {
    "AAPL": "news_score/Apple_news_score_last.csv",
    "MSFT": "news_score/MSFT_score_last.csv",
    "GOOGL": "news_score/GOOGLE_score_last.csv",
    "AMZN": "news_score/AMZN_score_last.csv",
    "META": "news_score/META_score_last.csv",
}
NEWS_SINGLE_FILE = None
TICKERS = ["AAPL","MSFT","GOOGL","AMZN","META"]
load_news_many(NEWS_SOURCES, NEWS_SINGLE_FILE, tickers_whitelist=TICKERS)

# ============================================
# Markov regimes (fit on train; filter elsewhere)
# ============================================
from dataclasses import dataclass
@dataclass
class MarkovModel:
    states: int
    P: np.ndarray
    mu: np.ndarray
    var: np.ndarray
    feat_names: List[str]

def _label_regime_row(v_regime, adx_raw, trend_dir):
    # 0=RiskOff, 1=Range/Neutral, 2=TrendingUp
    if (trend_dir < 0) and (v_regime > 1.15) and (adx_raw < 22.0):
        return 0
    if (trend_dir > 0) and (adx_raw >= 22.0):
        return 2
    return 1

def _emit_vector(df):
    # жёсткая очистка исходников
    close = df["Close"].astype(float)
    adx   = df["ADX_RAW"].astype(float).ffill().bfill().fillna(0.0)
    ema20 = df["EMA20_RAW"].astype(float).ffill().bfill().fillna(close)
    ema50 = df["EMA50_RAW"].astype(float).ffill().bfill().fillna(close)

    ret_abs = close.pct_change().abs().fillna(0.0).to_numpy(float)
    adx_norm = (adx / 50.0).clip(0.0, 1.0).to_numpy(float)
    ema_gap  = (np.abs(ema20 - ema50) / np.maximum(close, 1e-12)).clip(0, 0.06).to_numpy(float)

    X = np.stack([ret_abs, adx_norm, ema_gap], axis=1).astype(float)
    X[~np.isfinite(X)] = 0.0
    return X, ["RET_ABS","ADX_NORM","EMA_GAP"]


def fit_markov_on_pack(pack) -> MarkovModel:
    df_tr = pack["train_raw"].copy()
    if "ADX_RAW" not in df_tr.columns:
        df_tr["ADX_RAW"] = ADXIndicator(df_tr["High"], df_tr["Low"], df_tr["Close"], window=14).adx()
    if "EMA20_RAW" not in df_tr.columns or "EMA50_RAW" not in df_tr.columns:
        df_tr["EMA20_RAW"] = EMAIndicator(df_tr["Close"], window=20).ema_indicator()
        df_tr["EMA50_RAW"] = EMAIndicator(df_tr["Close"], window=50).ema_indicator()
    df_tr["Trend_Dir"] = np.where(df_tr["EMA20_RAW"] > df_tr["EMA50_RAW"], 1.0, -1.0)

    ret1 = df_tr["Close"].pct_change()
    v20  = ret1.rolling(20).std()
    v100 = ret1.rolling(100).std()
    vol_ratio = (v20 / v100).replace([np.inf,-np.inf], np.nan).fillna(1.0).clip(0,5.0)
    df_tr["VOL_REGIME"] = vol_ratio

    reg = np.asarray([_label_regime_row(v,a,t) for v,a,t in zip(
        df_tr["VOL_REGIME"].to_numpy(float), df_tr["ADX_RAW"].to_numpy(float), df_tr["Trend_Dir"].to_numpy(float))], int)

    S = 3
    P = np.ones((S,S), float) * 1e-3
    for i in range(1, len(reg)):
        P[reg[i-1], reg[i]] += 1.0
    P = (P.T / P.sum(axis=1)).T

    X, names = _emit_vector(df_tr)
    mu = np.zeros((S, X.shape[1]), float); var = np.zeros((S, X.shape[1]), float)
    for s in range(S):
        xs = X[reg==s]
        if xs.shape[0] == 0:
            mu[s]=X.mean(axis=0); var[s]=X.var(axis=0)+1e-6
        else:
            mu[s]=xs.mean(axis=0); var[s]=xs.var(axis=0)+1e-6
    return MarkovModel(states=S, P=P, mu=mu, var=var, feat_names=names)

def hmm_filter_posterior(model: MarkovModel, df_like: pd.DataFrame):
    X, _ = _emit_vector(df_like)
    S, N = model.states, X.shape[0]
    post = np.zeros((N, S), float)

    # стационарное распределение
    w, v = np.linalg.eig(model.P.T)
    ix = np.argmin(np.abs(w-1))
    pi0 = np.real(v[:, ix]); pi0 = np.maximum(pi0, 0); pi0 = pi0 / (pi0.sum() + 1e-12)
    prev = pi0

    inv_var = 1.0 / np.maximum(model.var, 1e-8)
    logZ = -0.5*np.log(2*np.pi*np.maximum(model.var, 1e-8)).sum(axis=1)

    for t in range(N):
        prior = prev @ model.P
        xt = X[t]
        z = -0.5 * ((xt - model.mu)**2 * inv_var).sum(axis=1) + logZ
        like = np.exp(z - z.max())  # стабилизация
        if (not np.isfinite(like).all()) or (like.sum() <= 0):
            like = np.ones(S, float) / S

        p = prior * like
        if (not np.isfinite(p).all()) or (p.sum() <= 0):
            p = np.ones(S, float) / S

        p = p / p.sum()
        post[t] = p
        prev = p

    return post


# ============================================
# Фичи и сплит данных
# ============================================
def add_features(ddf: pd.DataFrame, daily_news_impacts: pd.DataFrame = None) -> pd.DataFrame:
    ddf = ddf.copy()

    ema20_raw = EMAIndicator(ddf["Close"], window=20).ema_indicator()
    ema50_raw = EMAIndicator(ddf["Close"], window=50).ema_indicator()
    adx_raw   = ADXIndicator(ddf["High"], ddf["Low"], ddf["Close"], window=14).adx()
    ddf["EMA20_RAW"], ddf["EMA50_RAW"], ddf["ADX_RAW"] = ema20_raw, ema50_raw, adx_raw

    vwap = VolumeWeightedAveragePrice(high=ddf["High"], low=ddf["Low"], close=ddf["Close"],
                                      volume=ddf["Volume"], window=14, fillna=False)
    ddf["VWAP"] = vwap.volume_weighted_average_price()

    ddf["RSI"] = 0.0
    try:
        from ta.momentum import RSIIndicator
        ddf["RSI"] = RSIIndicator(ddf["Close"], window=14).rsi()
    except Exception:
        pass

    ddf["EMA20"] = ema20_raw.copy(); ddf["EMA50"] = ema50_raw.copy()

    macd = MACD(ddf["Close"])
    ddf["MACD"] = macd.macd()
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

    # --- NEWS (лаг 1)
    news_cols = [
        "NewsCount","NewsCtZ","ImpactWeighted","ImpactWeightedMean","ImpactWSumAbs",
        "ImpactWeighted_EMA10","ImpactWeighted_EMA20","ImpactSurprise",
        "Tier_weak_Ct","Tier_medium_Ct","Tier_strong_Ct"
    ]
    if (daily_news_impacts is not None) and isinstance(ddf.index, pd.DatetimeIndex):
        idx_norm = ddf.index.normalize()
        aligned = daily_news_impacts.reindex(idx_norm).fillna(0.0)
        aligned = aligned.shift(1).fillna(0.0)
        for c in news_cols:
            if c not in aligned.columns:
                aligned[c] = 0.0
            ddf[c] = aligned[c].astype(float).values
    else:
        for c in news_cols:
            ddf[c] = 0.0

    # Марковские фичи — закинем нули, заполнится позже
    for c in ["REG_P0","REG_P1","REG_P2","REG_ID","REG_TREND","REG_CONF"]:
        if c not in ddf.columns:
            ddf[c] = 0.0

    feat_clean = [
        "VWAP","RSI","EMA20","EMA50","MACD","MACD_signal","MACD_hist",
        "BB_high","BB_low","BB_width","BB_pos",
        "RET1","RET5","RET20","ATR","ATR_PCT","VOL_REGIME",
        "EMA20_SLOPE","EMA50_SLOPE","Trend_Dir","VOL_Z","VOL_PCT"
    ] + news_cols + ["REG_P0","REG_P1","REG_P2","REG_ID","REG_TREND","REG_CONF"]
    for c in feat_clean:
        ddf[c] = ddf[c].replace([np.inf, -np.inf], np.nan)

    return ddf

BASE_FEAT_COLS = [
    "Open","High","Low","Close","Volume","VWAP",
    "RSI","EMA20","EMA50","EMA20_SLOPE","EMA50_SLOPE","Trend_Dir",
    "MACD","MACD_signal","MACD_hist",
    "BB_width","BB_pos",
    "RET1","RET5","RET20","ATR","ATR_PCT","VOL_REGIME",
    "VOL_Z","VOL_PCT",
    "NewsCount","NewsCtZ","ImpactWeighted","ImpactWeightedMean","ImpactWSumAbs",
    "ImpactWeighted_EMA10","ImpactWeighted_EMA20","ImpactSurprise",
    "Tier_weak_Ct","Tier_medium_Ct","Tier_strong_Ct",
    "REG_P0","REG_P1","REG_P2","REG_ID","REG_TREND","REG_CONF",
]
feat_cols = BASE_FEAT_COLS
CLIP_FEAT = 6.0

def prep_one_ticker(tic: str,
                    start="2017-01-01", end="2025-01-01",
                    train_until="2021-12-31",
                    val_year="2022",
                    test_span=("2023-01-01","2024-12-31"),
                    daily_news_impacts: pd.DataFrame = None):
    df = yf.download(tic, start=start, end=end, auto_adjust=False, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.sort_index()

    train_df = df.loc[:train_until].copy()
    val_df   = df.loc[f"{val_year}-01-01":f"{val_year}-12-31"].copy()
    test_df  = df.loc[test_span[0]:test_span[1]].copy()

    train_idx, val_idx, test_idx = train_df.index, val_df.index, test_df.index

    for d in (train_df, val_df, test_df):
        d["TICKER_HINT"] = tic

    train_df = add_features(train_df, daily_news_impacts)
    val_df   = add_features(val_df,   daily_news_impacts)
    test_df  = add_features(test_df,  daily_news_impacts)

    train_raw, val_raw, test_raw = train_df.copy(), val_df.copy(), test_df.copy()

    for ddf in (train_df, val_df, test_df):
        ddf[feat_cols] = (
            ddf[feat_cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .clip(-CLIP_FEAT, CLIP_FEAT)
        )

    return {
        "train": train_df.reset_index(drop=True),
        "val":   val_df.reset_index(drop=True),
        "test":  test_df.reset_index(drop=True),
        "train_raw": train_raw.reset_index(drop=True),
        "val_raw":   val_raw.reset_index(drop=True),
        "test_raw":  test_raw.reset_index(drop=True),
        "train_index": train_idx,
        "val_index":   val_idx,
        "test_index":  test_idx,
        "price_full": (df["Adj Close"] if "Adj Close" in df.columns else df["Close"]).astype(float)
    }

# ============================================
# Волатильность / RMS / Env
# ============================================
def _vol_from_returns(rets: pd.Series, lag: int, method: str="ewma", halflife: float=11.0) -> pd.Series:
    rets = rets.astype(float)
    if method.lower() == "ewma":
        v = rets.ewm(halflife=halflife, adjust=False).std()
    else:
        v = rets.rolling(lag).std()
    return v.fillna(rets.std())

class RunningMeanStd:
    def __init__(self, shape=(), clip=None, eps=1e-4):
        self.shape = tuple(shape) if isinstance(shape, (list, tuple)) else ( ) if shape == () else (shape,)
        self.mean  = np.zeros(self.shape, dtype=np.float32)
        self.var   = np.ones(self.shape,  dtype=np.float32)
        self.count = float(eps)
        self.clip  = clip
        self.eps   = 1e-8
    def update(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.shape == self.shape:
            x = x[np.newaxis, ...]
        elif self.shape != () and x.shape[-len(self.shape):] != self.shape:
            raise ValueError(f"RMS.update: shape mismatch. got {x.shape}, expected *{self.shape}")
        n = float(x.shape[0])
        if n == 0:
            return
        batch_mean = x.mean(axis=0)
        batch_var  = x.var(axis=0)
        delta = batch_mean - self.mean
        tot   = self.count + n
        new_mean = self.mean + delta * (n / tot)
        m_a = self.var * self.count
        m_b = batch_var * n
        M2  = m_a + m_b + (delta**2) * (self.count * n / tot)
        new_var = M2 / max(tot, 1.0)
        self.mean, self.var, self.count = new_mean.astype(np.float32), (new_var + self.eps).astype(np.float32), float(tot)

class ObsRewNormWrapper:
    def __init__(self, env, obs_rms: RunningMeanStd, rew_rms: RunningMeanStd, training: bool = True, obs_clip=8.0, rew_clip=8.0):
        self.env, self.obs_rms, self.rew_rms = env, obs_rms, rew_rms
        self.training, self.obs_clip, self.rew_clip = bool(training), obs_clip, rew_clip
        self.observation_space = env.observation_space
        self.action_space      = env.action_space
        self.unwrapped         = env
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs).astype(np.float32)
        if self.training:
            self.obs_rms.update(obs)
        obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        if self.obs_clip is not None:
            obs = np.clip(obs, -self.obs_clip, self.obs_clip)
        return obs.astype(np.float32)
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        if self.training:
            self.obs_rms.update(obs)
        obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        if self.obs_clip is not None:
            obs = np.clip(obs, -self.obs_clip, self.obs_clip)
        r = float(reward)
        if self.training:
            self.rew_rms.update(np.array([r], dtype=np.float32))
        r = r / float(np.sqrt(self.rew_rms.var + 1e-8))
        if self.rew_clip is not None:
            r = float(np.clip(r, -self.rew_clip, self.rew_clip))
        return obs.astype(np.float32), r, done, info

def wrap_env_with_rms(env, obs_rms, rew_rms, training=True):
    return ObsRewNormWrapper(env, obs_rms, rew_rms, training=training)

# --------------------------------------------
# OpenAI Gym-подобная среда
# --------------------------------------------
import gym

class StockTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, episode_len=256, randomize=True, lag=20):
        super().__init__()
        self.df = df
        self.lag = int(lag)
        self.max_steps = len(df)
        self.episode_len = int(min(episode_len or self.max_steps, max(1, self.max_steps - self.lag)))
        self.randomize = bool(randomize)

        sma20 = self.df["Close_Price"].rolling(20).mean()
        sma50 = self.df["Close_Price"].rolling(50).mean()
        self.baseline_pos_series = (sma20 > sma50).astype(float).fillna(0.0)

        self.available_balance = INITIAL_ACCOUNT_BALANCE
        self.net_profit = 0.0
        self.equity_peak = 1.0

        self.num_trades_long = 0
        self.num_trades_short = 0
        self.long_short_ratio = 0.0

        self.window_start = None
        self.window_end = None
        self.current_step = 0

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(len(feat_cols)+1,), dtype=np.float32)

        self.prev_position = 0.0

        self.ret_series = self.df["Close_Price"].pct_change().fillna(0.0).astype(float)
        self.vol_series = _vol_from_returns(self.ret_series, lag=self.lag, method="ewma", halflife=11.0)
        v0 = float(self.vol_series.iloc[self.lag]) if len(self.vol_series) > self.lag else 1e-6
        self.volatility = max(v0, 1e-6)

    def _next_observation(self):
        r = self.df.loc[self.current_step]
        core = np.array([float(r[c]) for c in feat_cols], dtype=np.float32)
        obs = np.concatenate([core, np.array([float(self.long_short_ratio)], dtype=np.float32)])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _take_action(self, action):
        position = float(action[0])
        if (self.current_step + 1) < self.max_steps:
            asset_ret = float(self.ret_series.iloc[self.current_step + 1])
        else:
            asset_ret = 0.0
        asset_ret = float(np.clip(asset_ret, -0.12, 0.12))

        vol = float(self.vol_series.iloc[self.current_step]) if self.current_step < len(self.vol_series) else 0.0
        self.volatility = vol if (np.isfinite(vol) and vol > 0) else 1e-6

        s_full = TARGET_DAILY_VOL / max(self.volatility, 1e-6)
        scale  = 0.9*s_full + 0.1*min(1.0, s_full)
        eff_position = float(np.clip(position * scale, -1.0, 1.0))

        if TREND_GATE:
            ts = float(self.df.loc[self.current_step, "ADX_RAW"]) / 50.0
            ts = float(np.clip(ts, 0.0, 1.0))
            eff_position = float(np.clip(eff_position * (0.5 + 0.5 * ts), -1.0, 1.0))

        raw_ret      = eff_position * asset_ret
        step_pnl     = max(self.available_balance, 0.0) * PERCENT_CAPITAL * float(raw_ret)

        self.net_profit        += step_pnl
        self.available_balance += step_pnl

        delta_pos    = eff_position - self.prev_position
        turnover_fee = abs(max(self.available_balance, 0.0) * PERCENT_CAPITAL) * TURNOVER_COST * abs(delta_pos)
        if turnover_fee != 0.0:
            self.available_balance -= turnover_fee
            self.net_profit        -= turnover_fee

        if eff_position > 0: self.num_trades_long  += 1
        if eff_position < 0: self.num_trades_short += 1
        den = self.num_trades_long + self.num_trades_short
        self.long_short_ratio = (self.num_trades_long / den) if den > 0 else 0.0

        return float(raw_ret), float(eff_position), float(delta_pos), float(raw_ret), float(asset_ret), float(turnover_fee)

    def step(self, action):
        realized_ret, eff_position, delta_pos, raw_ret, asset_ret, turnover_fee = self._take_action(action)

        smooth_pen = float(SMOOTH_COST * (delta_pos ** 2))
        hold_pen   = float(HOLD_COST   * (eff_position ** 2))

        trend_dir = 1.0 if float(self.df.loc[self.current_step, "Trend_Dir"]) >= 0.0 else -1.0
        trend_strength = float(self.df.loc[self.current_step, "ADX_RAW"]) / 50.0
        trend_strength = float(np.clip(trend_strength, *TREND_STRENGTH_CLIP))
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
        done = bool(hard_stop or (self.current_step >= self.window_end - 1))

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
        self.num_trades_long = 0
        self.num_trades_short = 0
        self.long_short_ratio = 0.0
        self.prev_position = 0.0
        self.equity_peak = 1.0

        if bool(self.randomize) and (self.max_steps > self.episode_len):
            start_idx = int(np.random.randint(self.lag, max(self.lag+1, self.max_steps - self.episode_len + 1)))
        else:
            start_idx = self.lag
        self.window_start = start_idx
        self.window_end   = min(self.max_steps, self.window_start + self.episode_len)
        self.current_step = self.window_start

        vol = float(self.vol_series.iloc[self.current_step]) if self.current_step < len(self.vol_series) else 1e-6
        self.volatility = vol if (np.isfinite(vol) and vol > 0) else 1e-6

        return self._next_observation()

# ============================================
# Сети (LSTM), память, агент PPO
# ============================================
class LSTMBackbone(nn.Module):
    def __init__(self, feat_dim, hidden=96, nlayers=2, dropout=0.15, bidirectional=False):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=nlayers,
            batch_first=True,
            dropout=dropout if nlayers > 1 else 0.0,
            bidirectional=self.bidirectional
        )
        self.out_dim = hidden * (2 if self.bidirectional else 1)
    def forward(self, x):
        y, (h, c) = self.lstm(x)
        if not self.bidirectional:
            return h[-1]
        return T.cat([h[-2], h[-1]], dim=-1)

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, lr, lstm_hidden=96, lstm_layers=2, dropout=0.1, chkpt_dir=CHKPT_DIR):
        super().__init__()
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_cont_multi.pt')
        feat_dim = int(input_dims[-1]) if isinstance(input_dims, (tuple, list, np.ndarray)) else int(input_dims)
        self.backbone = LSTMBackbone(feat_dim, hidden=lstm_hidden, nlayers=lstm_layers, dropout=dropout, bidirectional=False)
        out_dim = self.backbone.out_dim
        self.fc_mu   = nn.Linear(out_dim, 1)
        self.log_std = nn.Parameter(T.zeros(1, 1))
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        self.device = DEVICE
        self.to(self.device)
    def forward(self, state_seq):
        h = self.backbone(state_seq)
        mu = T.tanh(self.fc_mu(h))
        sigma = F.softplus(self.log_std) + 1e-4
        return T.distributions.Normal(mu, sigma)
    def save_checkpoint(self): T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self): self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, lr, lstm_hidden=96, lstm_layers=2, dropout=0.1, chkpt_dir=CHKPT_DIR):
        super().__init__()
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_cont_multi.pt')
        feat_dim = int(input_dims[-1]) if isinstance(input_dims, (tuple, list, np.ndarray)) else int(input_dims)
        self.backbone = LSTMBackbone(feat_dim, hidden=lstm_hidden, nlayers=lstm_layers, dropout=dropout, bidirectional=False)
        out_dim = self.backbone.out_dim
        self.fc_value = nn.Linear(out_dim, 1)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        self.device = DEVICE
        self.to(self.device)
    def forward(self, state_seq):
        h = self.backbone(state_seq)
        return self.fc_value(h)
    def save_checkpoint(self): T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self): self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

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

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, lr=3e-4,
                 gae_lambda=0.97, policy_clip=0.30, batch_size=512,
                 n_epochs=8, entropy_coef=ENTROPY_COEF, vf_clip=0.2,
                 target_kl=0.1, kl_coef=0.0, max_grad_norm=0.9):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef_base = entropy_coef
        self.entropy_floor = 0.01
        self.learn_calls = 0
        self.vf_clip = vf_clip
        self.target_kl = target_kl
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm

        self.actor  = ActorNetwork(input_dims, lr)
        self.critic = CriticNetwork(input_dims, lr)
        maybe_compile_forward(self.actor)
        maybe_compile_forward(self.critic)

        self.memory = PPOMemory(batch_size)
        self.seq_len = SEQ_LEN
        self.state_window = deque(maxlen=self.seq_len)

        self.actor_sched  = optim.lr_scheduler.CosineAnnealingLR(self.actor.optimizer,  T_max=200)
        self.critic_sched = optim.lr_scheduler.CosineAnnealingLR(self.critic.optimizer, T_max=200)

        self.TAU_ENTER, self.TAU_EXIT = 0.05, 0.02
        self.policy_pos_cache = 0.0

    def remember(self, state_seq, action, log_prob, value, reward, done):
        self.memory.store_memory(state_seq, action, log_prob, value, reward, done)

    def save_models(self):
        self.actor.save_checkpoint(); self.critic.save_checkpoint()

    def choose_action(self, observation):
        self.state_window.append(np.asarray(observation, dtype=np.float32))
        if len(self.state_window) < self.seq_len:
            return np.array([0.0], dtype=np.float32), 0.0, 0.0, None

        state_seq = np.stack(self.state_window, axis=0).astype(np.float32)
        state_tensor = T.tensor(state_seq, dtype=T.float32, device=self.actor.device).unsqueeze(0)
        with T.no_grad():
            dist  = self.actor(state_tensor)
            value = self.critic(state_tensor)
            action = T.clamp(dist.sample(), -1.0, 1.0)

        raw = float(action.squeeze().cpu().numpy())
        if APPLY_DEADZONE_TRAIN:
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
                        self.entropy_coef_base * (0.5 ** (self.learn_calls / 20)))
        (state_arr, action_arr, old_logp_arr,
         vals_arr, reward_arr, dones_arr, batches) = self.memory.generate_batches()

        rewards = np.asarray(reward_arr, dtype=np.float32)
        values  = np.asarray(vals_arr,    dtype=np.float32)
        dones   = np.asarray(dones_arr,   dtype=np.float32)

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

                with T.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
                    dist = self.actor(states)
                    new_logp = dist.log_prob(actions).sum(dim=-1)
                    ratio = (new_logp - old_logp).exp()
                    surr1 = ratio * adv
                    surr2 = T.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * adv
                    actor_loss = -T.min(surr1, surr2).mean()

                    approx_kl = (old_logp - new_logp).mean()

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

                # чуть выше порог отсечки по KL, чтобы не рано останавливаться
                if approx_kl.item() > 0.16:
                    early_stop = True
                    break

            self.actor_sched.step()
            self.critic_sched.step()
            if early_stop:
                break

        self.learn_calls += 1
        self.memory.clear_memory()

# ============================================
# Basket загрузка
# ============================================
print("Downloading & preparing:", TICKERS)
BASKET = {}
for tic in TICKERS:
    daily_imp = DAILY_IMPACTS_BY_TICKER.get(tic)
    BASKET[tic] = prep_one_ticker(tic, daily_news_impacts=daily_imp)
print("Prepared tickers:", list(BASKET.keys()))

# ============ Обучаем Markov по тикерам ============
MARKOV_BY_TICKER: Dict[str, Optional[MarkovModel]] = {}
for tic in TICKERS:
    try:
        MARKOV_BY_TICKER[tic] = fit_markov_on_pack(BASKET[tic])
        print(f"[markov] {tic}: P=\n{MARKOV_BY_TICKER[tic].P}")
    except Exception as e:
        print(f"[markov] {tic}: FAILED -> {e}")
        MARKOV_BY_TICKER[tic] = None

# ============ Инъекция марковских признаков (устойчивая) ============
def inject_markov_features(pack, model: Optional[MarkovModel]):
    if model is None:
        return
    rng = np.random.RandomState(42)
    for split in ["train","val","test"]:
        df_like = pack[f"{split}_raw"].copy()
        # гарантируем наличие столбцов
        for need in ["Close","High","Low","ADX_RAW","EMA20_RAW","EMA50_RAW"]:
            if need not in df_like.columns:
                src = pack[split] if (split in pack and need in pack[split].columns) else None
                base = pack[split]["Close"] if "Close" in pack[split].columns else pack[split]["Close_Price"]
                df_like[need] = (src[need].values if src is not None else base.values)

        post = hmm_filter_posterior(model, df_like).astype(np.float64)

        # лаг 1 день
        if post.shape[0] > 1:
            post = np.roll(post, 1, axis=0); post[0, :] = post[1, :]

        # очистка и лёгкий шум
        if (~np.isfinite(post)).any():
            post[~np.isfinite(post)] = 0.0
        eps = 1e-6
        post = np.clip(post + eps * rng.randn(*post.shape), 1e-6, 1.0)
        post = post / post.sum(axis=1, keepdims=True)

        out = pack[split]
        out["REG_P0"] = post[:, 0].astype(np.float32)
        out["REG_P1"] = post[:, 1].astype(np.float32)
        out["REG_P2"] = post[:, 2].astype(np.float32)
        out["REG_ID"] = np.argmax(post, axis=1).astype(np.float32)
        out["REG_TREND"] = (post[:, 2] - post[:, 0]).astype(np.float32)
        out["REG_CONF"]  = (np.max(post, axis=1) - np.min(post, axis=1)).astype(np.float32)


for tic in TICKERS:
    inject_markov_features(BASKET[tic], MARKOV_BY_TICKER.get(tic))

# ============================================
# Общие RMS и агент
# ============================================
obs_rms = RunningMeanStd(shape=(len(feat_cols) + 1,), clip=8.0)
rew_rms = RunningMeanStd(shape=(), clip=8.0)

_probe_env = wrap_env_with_rms(StockTradingEnv(BASKET[TICKERS[0]]["train"]), obs_rms, rew_rms, training=True)
agent = Agent(
    n_actions=1,
    input_dims=_probe_env.observation_space.shape,
    lr=LR, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
    entropy_coef=ENTROPY_COEF, policy_clip=0.30, target_kl=0.08,
    max_grad_norm=0.9, gae_lambda=0.97
)
del _probe_env

VAL_ENVS = {
    tic: wrap_env_with_rms(StockTradingEnv(BASKET[tic]["val"], randomize=False), obs_rms, rew_rms, training=False)
    for tic in TICKERS
}

def eval_agent_on_env(agent, env, episodes=1):
    sh_list, mdd_list = [], []
    for _ in range(episodes):
        obs = env.reset(); agent.state_window.clear()
        done = False
        daily_port = []
        while not done:
            action, _, _, _ = agent.choose_action(obs)
            obs, reward, done, info = env.step(action)
            realized_ret = float(info.get("realized_ret", 0.0))
            daily_port.append(realized_ret * PERCENT_CAPITAL)
        r = np.asarray(daily_port, float)
        sharpe = (r.mean() / (r.std() + 1e-12)) * np.sqrt(252.0) if r.size > 1 else 0.0
        eq = (1.0 + r).cumprod()
        peak = np.maximum.accumulate(eq) if eq.size > 0 else np.array([1.0])
        mdd = float((eq/peak - 1.0).min()) if eq.size > 0 else 0.0
        sh_list.append(sharpe); mdd_list.append(mdd)
    sharpe = float(np.mean(sh_list)); mdd = float(np.mean(mdd_list))
    val_score = sharpe - 0.5 * abs(mdd)
    return val_score, {"sharpe": sharpe, "mdd": mdd}

def eval_on_basket(agent, envs_dict):
    scores, shs, mdds = [], [], []
    for tic, env in envs_dict.items():
        s, st = eval_agent_on_env(agent, env, episodes=1)
        scores.append(s); shs.append(st["sharpe"]); mdds.append(st["mdd"])
    return float(np.mean(scores)), float(np.mean(shs)), float(np.mean(mdds))

# ============================================
# Обучение по корзине с best-чекпоинтом
# ============================================
best_val = -1e9; stale_val = 0; patience_val = 6
new_best_saved = False
print("... starting curriculum on basket:", TICKERS)
for name, s, e, n_games in STAGES:
    for ep in range(n_games):
        tic = random.choice(TICKERS)
        df_tr = BASKET[tic]["train"]
        end_idx = e if (e is not None) else len(df_tr)

        base_env = StockTradingEnv(df_tr.iloc[s:min(end_idx, len(df_tr))].reset_index(drop=True))
        env = wrap_env_with_rms(base_env, obs_rms, rew_rms, training=True)

        obs = env.reset(); agent.state_window.clear()
        done, score = False, 0.0
        while not done:
            action, logp, val, state_seq = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            if state_seq is not None:
                agent.remember(state_seq, float(action[0]), logp, val, reward, done)
            score += reward
            obs = obs_

        agent.learn()

        if ((ep + 1) % VAL_EVERY) == 0:
            val_score, mean_sh, mean_mdd = eval_on_basket(agent, VAL_ENVS)
            print(f"[{name}] ep {ep+1}/{n_games} | val_score {val_score:.3f} | Sharpe {mean_sh:.2f} | MDD {mean_mdd:.2%}")
            if val_score > best_val + 1e-3:
                best_val = val_score; stale_val = 0
                agent.save_models(); new_best_saved = True
                print(f"[VAL] ↑ new best checkpoint saved -> {CHKPT_DIR}")
            else:
                stale_val += 1
                if stale_val >= patience_val:
                    print(f"[VAL] early stop: no improve {patience_val} checks; best={best_val:.3f}")
                    break

if new_best_saved:
    try:
        agent.actor.load_checkpoint()
        agent.critic.load_checkpoint()
        print("Loaded best checkpoint (from this run).")
    except Exception as e:
        print("WARN: couldn't load best checkpoint ->", e)
else:
    print("NOTE: no new best saved in this run; using in-memory weights.")

# ============================================
# Helpers
# ============================================
def compute_vol_series(ret_series: pd.Series, mode="ewma", span=20, window=20, halflife=None) -> pd.Series:
    ret_series = pd.Series(ret_series, dtype=float)
    if mode == "ewma":
        vol = ret_series.ewm(span=span, adjust=False).std(bias=False)
    elif mode == "ewm_halflife":
        vol = ret_series.ewm(halflife=(10.0 if halflife is None else halflife), adjust=False).std(bias=False)
    else:
        vol = ret_series.rolling(window).std()
    return vol.fillna(ret_series.std())

def bh_curve_from_prices(price_series, pct_capital=1.0):
    px = np.asarray(price_series, dtype=float)
    if px.size == 0:
        return np.array([], float)
    ret = np.zeros_like(px, dtype=float)
    if px.size > 1:
        ret[1:] = (px[1:] - px[:-1]) / np.where(px[:-1]==0, 1e-12, px[:-1])
    port = (1.0 + pct_capital * ret)
    return port.cumprod()

def bh_test_from_full(price_full: pd.Series, test_like, pct_capital=1.0):
    idx = test_like.index if isinstance(test_like, (pd.DataFrame, pd.Series)) else test_like
    if isinstance(idx, pd.DatetimeIndex):
        px = price_full.loc[idx]
    else:
        px = price_full.iloc[-len(idx):]
    return bh_curve_from_prices(px, pct_capital)

def attach_top_news_at_change(df_bt: pd.DataFrame, news_df: pd.DataFrame, title_col="title", url_col="url", max_items=3) -> pd.DataFrame:
    out = df_bt.copy()
    out["Change"] = out["Position"].diff().fillna(0).ne(0)
    out["TopNewsAtChange"] = ""
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    idx_dates = out.index.normalize()
    if (news_df is None) or (news_df.empty):
        return out
    gcol = "trading_day" if "trading_day" in news_df.columns else "date"
    news_by_day = news_df.groupby(gcol)
    tier_rank = {"strong": 2, "medium": 1, "weak": 0}
    rows = []
    for i in range(len(out)):
        changed = bool(out["Change"].iloc[i])
        if not changed:
            rows.append("")
            continue
        d = idx_dates[i]
        day_key = d if gcol == "trading_day" else d.date()
        if day_key not in news_by_day.groups:
            rows.append("")
            continue
        g = news_by_day.get_group(day_key).copy()
        if "impact_tier" not in g.columns: g["impact_tier"] = "weak"
        if "impact_weight" not in g.columns: g["impact_weight"] = 0.5
        g["tier_rank"] = g["impact_tier"].map(tier_rank).fillna(0)
        g = g.sort_values(["tier_rank","impact_weight"], ascending=[False, False]).head(max_items)
        ts = g[title_col] if title_col in g.columns else g["title"]
        us = g[url_col]   if url_col   in g.columns else g["url"]
        links = [f'{t} [{u}] (tier={ti}, w={w:.2f})'
                 for t,u,ti,w in zip(ts, us, g["impact_tier"], g["impact_weight"])]
        rows.append(" | ".join(links))
    out["TopNewsAtChange"] = rows
    return out

def _factor_series_from_mapping(mapping: dict, index_norm: pd.DatetimeIndex) -> np.ndarray:
    if (mapping is None) or (len(mapping) == 0):
        return np.ones(len(index_norm), dtype=float)
    # ключи в mapping — date(), поэтому нормализуем к date()
    map_s = pd.Series(mapping)
    # реиндекс по датам:
    idx_dates = pd.Index([d.date() for d in index_norm])
    s = map_s.reindex(idx_dates).fillna(1.0)
    s = s.shift(1).fillna(1.0)
    return s.to_numpy(dtype=float)

# ============================================
# Инференс с NEWS_FACTOR + Markov адаптацией
# ============================================
def run_inference_on_ticker(tic: str, agent, pack, split="test",
                            news_factor_override: dict = None,
                            daily_impacts_override: pd.DataFrame = None):
    def _clamp(x, lo, hi): return float(np.minimum(np.maximum(x, lo), hi))
    def build_ticker_context(rep_hist: pd.DataFrame, raw_hist: pd.DataFrame) -> dict:
        px  = raw_hist["Close"].to_numpy(float)
        e20 = rep_hist["EMA20_RAW"].to_numpy(float)
        e50 = rep_hist["EMA50_RAW"].to_numpy(float)
        adx = rep_hist["ADX_RAW"].to_numpy(float)
        prev = np.r_[px[0], px[:-1]]
        tr   = np.maximum.reduce([raw_hist["High"]-raw_hist["Low"], np.abs(raw_hist["High"]-prev), np.abs(raw_hist["Low"] -prev)])
        atr  = pd.Series(tr).rolling(14).mean().bfill().to_numpy(float)
        atrp = atr/np.maximum(px,1e-12)
        gap  = np.abs(e20-e50)/np.maximum(px,1e-12)
        sma200 = pd.Series(px).rolling(200).mean().to_numpy()
        pct_bull  = float(np.mean(px >= np.nan_to_num(sma200, nan=np.inf))) if px.size>0 else 0.0
        pct_adx25 = float(np.mean(adx > 25)) if adx.size>0 else 0.0
        return dict(q_adx_55=float(np.nanquantile(adx, 0.55)),
                    q_adx_75=float(np.nanquantile(adx, 0.75)),
                    q_adx_85=float(np.nanquantile(adx, 0.85)),
                    q_gap_40=float(np.nanquantile(gap, 0.40)),
                    atrp_p60=float(np.nanpercentile(atrp, 60)),
                    pct_bull=pct_bull, pct_adx25=pct_adx25)

    def derive_hypers_from_ctx(ctx: dict) -> dict:
        ADX_MIN_INF  = _clamp(ctx["q_adx_55"], 12.0, 22.0)
        ADX_STRONG   = _clamp(ctx["q_adx_75"], 20.0, 35.0)
        ADX_VSTRONG  = _clamp(ctx["q_adx_85"], 24.0, 40.0)
        GAP_MIN_INF  = _clamp(ctx["q_gap_40"], 0.0020, 0.0080)
        L_break      = int(np.clip(ctx["atrp_p60"]*1e4, 35, 60))
        if ctx["pct_adx25"] < 0.30:
            TARGET_DAILY_VOL_INF = 0.065; ALPHA_SOFT = 0.75
        else:
            TARGET_DAILY_VOL_INF = 0.055; ALPHA_SOFT = 0.60
        if ctx["pct_adx25"] < 0.25:
            CE_MULT_WEAK, CE_MULT_STRONG, COOLDOWN_BARS = 2.6, 3.1, 5
        elif ctx["pct_adx25"] > 0.45:
            CE_MULT_WEAK, CE_MULT_STRONG, COOLDOWN_BARS = 2.2, 2.6, 2
        else:
            CE_MULT_WEAK, CE_MULT_STRONG, COOLDOWN_BARS = 2.4, 3.0, 3
        return dict(ADX_MIN_INF=ADX_MIN_INF, ADX_STRONG=ADX_STRONG, ADX_VSTRONG=ADX_VSTRONG,
                    GAP_MIN_INF=GAP_MIN_INF, L_break=L_break, L_trail=20,
                    TARGET_DAILY_VOL_INF=TARGET_DAILY_VOL_INF, ALPHA_SOFT=ALPHA_SOFT,
                    CE_MULT_WEAK=CE_MULT_WEAK, CE_MULT_STRONG=CE_MULT_STRONG, COOLDOWN_BARS=COOLDOWN_BARS,
                    PULLBACK_PCT20=0.004, NEAR_DON_ATR=0.5, MIN_HOLD=8, STEP_FRICTION=0.20, LONG_ONLY=LONG_ONLY,
                    Q0=0.14, Q1=0.28, Q2=0.42)

    rep = pack[split].copy()
    raw = pack[f"{split}_raw"].reset_index(drop=True)
    if len(raw) != len(rep):
        raw = raw.iloc[-len(rep):].reset_index(drop=True)

    U = derive_hypers_from_ctx(
        build_ticker_context(
            pd.concat([pack["train"], pack["val"]], ignore_index=True),
            pd.concat([pack["train_raw"], pack["val_raw"]], ignore_index=True)
        )
    )

    px = raw["Close"].to_numpy(float)
    hi = raw["High"].to_numpy(float)
    lo = raw["Low"].to_numpy(float)
    e20 = rep["EMA20_RAW"].to_numpy(float)
    e50 = rep["EMA50_RAW"].to_numpy(float)
    adx = rep["ADX_RAW"].to_numpy(float)

    prev_close = np.r_[px[0], px[:-1]]
    tr = np.maximum.reduce([hi-lo, np.abs(hi-prev_close), np.abs(lo-prev_close)])
    atr = pd.Series(tr).rolling(14).mean().bfill().to_numpy(float)
    atrp = (atr / np.maximum(px, 1e-12)).astype(float)

    L_break, L_trail = int(U["L_break"]), int(U["L_trail"])
    don_hi = pd.Series(px).rolling(L_break).max().shift(1).to_numpy()

    vol20 = compute_vol_series(pd.Series(px, dtype=float).pct_change().fillna(0.0), mode="ewma", span=20).to_numpy(float)
    sma200 = pd.Series(px).rolling(200).mean().to_numpy()
    bull = px >= np.nan_to_num(sma200, nan=np.inf)

    ADX_MIN_INF, GAP_MIN_INF = float(U["ADX_MIN_INF"]), float(U["GAP_MIN_INF"])
    ADX_STRONG, ADX_VSTRONG = float(U["ADX_STRONG"]), float(U["ADX_VSTRONG"])
    TAU_ENTER, TAU_EXIT  = 0.08, 0.035
    ALPHA_SOFT = float(U["ALPHA_SOFT"])
    Q0, Q1, Q2 = float(U["Q0"]), float(U["Q1"]), float(U["Q2"])
    CE_MULT_WEAK, CE_MULT_STRONG = float(U["CE_MULT_WEAK"]), float(U["CE_MULT_STRONG"])
    MIN_HOLD, STEP_FRICTION, COOLDOWN_BARS = int(U["MIN_HOLD"]), float(U["STEP_FRICTION"]), int(U["COOLDOWN_BARS"])
    PULLBACK_PCT20, NEAR_DON_ATR, LONG_ONLY_LOCAL = float(U["PULLBACK_PCT20"]), float(U["NEAR_DON_ATR"]), bool(U["LONG_ONLY"])
    base_target = float(U["TARGET_DAILY_VOL_INF"])

    # Markov
    if {"REG_P0","REG_P1","REG_P2"}.issubset(rep.columns):
        reg_p = rep[["REG_P0","REG_P1","REG_P2"]].to_numpy(float)
        rs = reg_p.sum(axis=1, keepdims=True)
        bad = (rs <= 1e-8).flatten()
        if bad.any():
            reg_p[bad] = np.array([1/3,1/3,1/3], float)
        else:
            reg_p = reg_p / rs
        reg_id = np.argmax(reg_p, axis=1).astype(int)
    else:
        reg_p = np.tile(np.array([1/3,1/3,1/3], float), (len(rep),1))
        reg_id = np.zeros(len(rep), int)

    # Новости: фактор и сюрприз
    idx_like = pack.get(f"{split}_index", None)
    if (idx_like is not None) and (len(idx_like) >= len(rep)):
        idx_like = pd.DatetimeIndex(idx_like)
        rep_index_norm = idx_like.normalize()
        if news_factor_override is not None:
            factor_vec = _factor_series_from_mapping(news_factor_override, rep_index_norm)
        else:
            daily_imp = daily_impacts_override if daily_impacts_override is not None else DAILY_IMPACTS_BY_TICKER.get(tic)
            default_map = build_news_factor_from_daily(daily_imp, k=0.40, scale=1.20, cap=0.50) if (daily_imp is not None and not daily_imp.empty) else {}
            factor_vec = _factor_series_from_mapping(default_map, rep_index_norm)

        daily_imp = daily_impacts_override if daily_impacts_override is not None else DAILY_IMPACTS_BY_TICKER.get(tic)
        if (daily_imp is not None) and (not daily_imp.empty):
            aligned = daily_imp.reindex(rep_index_norm).fillna(0.0).shift(1).fillna(0.0)
            surprise_vec = aligned["ImpactSurprise"].to_numpy(float)
        else:
            surprise_vec = np.zeros(len(rep), dtype=float)
    else:
        factor_vec = np.ones(len(rep), dtype=float)
        surprise_vec = np.zeros(len(rep), dtype=float)

    positions, state_window = [], deque(maxlen=SEQ_LEN)
    raw_smooth, pol_cache = 0.0, 0.0
    in_long, units, peak_px, last_units, last_sign, hold, cooldown, below20_cnt = False, 0, 0.0, 0, 0, 0, 0, 0

    with T.no_grad():
        for t in range(len(rep)):
            r = rep.iloc[t]
            core = r.reindex(feat_cols).astype(float).to_numpy(dtype=np.float32)
            obs  = np.concatenate([core, np.array([0.0], dtype=np.float32)])
            obs  = np.nan_to_num(obs, 0.0, 0.0, 0.0)

            state_window.append(obs)
            if len(state_window) < SEQ_LEN:
                positions.append(0.0)
                continue

            dist = agent.actor(T.tensor(np.stack(state_window, axis=0)[None, ...], dtype=T.float32, device=agent.actor.device))
            mu   = float(T.clamp(dist.mean, -1.0, 1.0).squeeze().cpu().numpy())
            raw_smooth = 0.85*raw_smooth + 0.15*mu
            x = raw_smooth

            if pol_cache == 0.0:
                if abs(x) < TAU_ENTER: x = 0.0
            else:
                if abs(x) < TAU_EXIT:  x = 0.0
            pol_cache = x

            # Мягкая Markov-адаптация
            target_today = base_target
            if reg_id[t] == 0:  # Risk-Off
                x *= 0.90
                target_today *= 0.85
            elif reg_id[t] == 2:  # TrendingUp
                x *= 1.05
                target_today *= 1.05

            # Масштаб по волатильности
            s_full = target_today / max(float(vol20[t]), 1e-6)
            scale  = ALPHA_SOFT*s_full + (1-ALPHA_SOFT)*min(1.0, s_full)
            eff    = float(np.clip(x * scale, -1.0, 1.0))

            # RAW-гейты (ослабление опционально)
            ema_gap   = abs(e20[t] - e50[t]) / max(abs(px[t]), 1e-6)
            trend_dir = 1.0 if e20[t] >= e50[t] else -1.0
            if not WEAKEN_RAW_GATES_FOR_TEST:
                if (adx[t] < ADX_MIN_INF) or (ema_gap < GAP_MIN_INF):
                    eff = 0.0
                gate = float(np.clip((adx[t] - ADX_ENTER) / max(ADX_FULL - ADX_ENTER, 1e-6), 0.0, 1.0))
                eff *= (0.3 + 0.7*gate)
                if eff * trend_dir < 0:
                    eff *= (1.0 - 0.5*gate)
            ts = float(np.clip(adx[t] / 50.0, 0.0, 1.0))
            eff = float(np.clip(eff * (0.5 + 0.5*ts), -1.0, 1.0))

            # NEWS factor & soft-gate
            eff *= float(factor_vec[t])
            if ENABLE_SURPRISE_SOFTGATE and (surprise_vec[t] < -0.6) and (adx[t] < 20.0):
                eff = min(eff, 0.5)

            if LONG_ONLY_LOCAL:
                if trend_dir < 0:
                    eff = 0.0
                elif (not bool(bull[t])) and (adx[t] < 25.0):
                    eff = 0.0
                eff = max(0.0, eff)

            # дискретизация позиций
            if eff <= Q0:   base_units = 0
            elif eff <= Q1: base_units = 1
            elif eff <= Q2: base_units = 2
            else:           base_units = 3

            adx_up     = (t > 0) and (adx[t] > adx[t-1])
            near_e20   = abs(px[t] - e20[t]) <= 0.004 * max(e20[t], 1e-6)
            breakoutUp = (px[t] > (don_hi[t] if not np.isnan(don_hi[t]) else -np.inf)) and (trend_dir > 0) and adx_up

            if cooldown > 0:
                target_units = 0
            else:
                target_units = base_units
                if (trend_dir > 0) and bool(bull[t]) and near_e20 and (adx[t] >= 20.0):
                    target_units = max(target_units, 1)
                if breakoutUp:
                    target_units = max(target_units, 2)
                if (adx[t] >= ADX_STRONG):
                    target_units = max(target_units, 3)
                if (adx[t] >= ADX_VSTRONG) and breakoutUp:
                    target_units = 3

            if abs(target_units - last_units) < (STEP_FRICTION * 3):
                target_units = last_units

            if target_units > 0:
                ce_mult = CE_MULT_WEAK if adx[t] < ADX_STRONG else CE_MULT_STRONG
                if not in_long:
                    in_long = True; peak_px = px[t]
                else:
                    peak_px = max(peak_px, px[t])
                    ce_floor = peak_px * (1.0 - ce_mult * max(atrp[t], 1e-6))
                    if px[t] < ce_floor:
                        target_units = 0; in_long = False; last_sign = 0; hold = 0; cooldown = COOLDOWN_BARS; below20_cnt = 0
            else:
                in_long = False

            if (target_units > 0) and (px[t] < e20[t]) and (adx[t] < 18.0):
                below20_cnt += 1
                if below20_cnt >= 3:
                    target_units = 0; in_long = False; last_sign = 0; hold = 0; cooldown = COOLDOWN_BARS
            else:
                if px[t] >= e20[t]:
                    below20_cnt = 0

            sgn = int(target_units > 0)
            if sgn == 0:
                hold += 1
            else:
                if last_sign == 0:
                    last_sign = 1; hold = 0
                elif last_sign != 1 and hold < MIN_HOLD:
                    target_units = max(1, last_units); sgn = 1; hold += 1
                elif last_sign != 1 and hold >= MIN_HOLD:
                    last_sign = 1; hold = 0
                else:
                    hold += 1

            units = int(np.clip(target_units, 0, 3))
            last_units = units
            positions.append([0.0, 0.5, 1.0, min(LEVER_CAP, 1.35)][units])
            if cooldown > 0:
                cooldown -= 1

    df_bt = rep.copy()
    df_bt["Return"] = rep["Close_Price"].pct_change().fillna(0.0)

    pos = np.asarray(positions, float)
    N = len(df_bt)
    if pos.size < N:
        pos = np.r_[np.zeros(N-pos.size), pos]
    elif pos.size > N:
        pos = pos[-N:]

    df_bt["Position"] = pos
    delta_pos = df_bt["Position"].diff().fillna(0.0)
    active_ret = df_bt["Position"].shift(1).fillna(0.0) * df_bt["Return"] - (TURNOVER_COST + SLIPPAGE_BPS) * delta_pos.abs()
    df_bt["PortRet"] = PERCENT_CAPITAL * active_ret
    df_bt["Equity"]  = (1.0 + df_bt["PortRet"]).cumprod()

    idx_like = pack.get(f"{split}_index", None)
    if idx_like is not None:
        df_bt.index = pd.DatetimeIndex(idx_like)

    price_full = pack["price_full"]
    bh_test = bh_test_from_full(price_full, df_bt.index, PERCENT_CAPITAL)
    df_bt["Benchmark"] = bh_test

    df_bt = attach_top_news_at_change(df_bt, NEWS_BY_TICKER.get(tic), "title", "url", max_items=3)

    ROI = (df_bt["Equity"].iloc[-1] - 1.0) * 100
    daily = df_bt["PortRet"].to_numpy()
    sharpe = (daily.mean() / (daily.std() + 1e-12)) * np.sqrt(252.0) if daily.size>1 else 0.0
    peak = np.maximum.accumulate(df_bt["Equity"].to_numpy())
    mdd = float((df_bt["Equity"].to_numpy()/peak - 1.0).min()) if df_bt["Equity"].size>0 else 0.0
    return float(ROI), float(sharpe), float(mdd), df_bt

# ============================================
# Калибровка k для NEWS_FACTOR (с нижним порогом)
# ============================================
def calibrate_news_k_for_ticker(tic: str, agent, pack, daily_impacts,
                                k_grid=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                                scale=1.20, cap=0.50, k_floor: float = 0.10):
    if daily_impacts is None or daily_impacts.empty:
        return 0.0
    best_k, best_sh = 0.0, -1e9
    for k in k_grid:
        factor_map = build_news_factor_from_daily(daily_impacts, k=k, scale=scale, cap=cap)
        _, sh, _, _ = run_inference_on_ticker(tic, agent, pack, split="val",
                                              news_factor_override=factor_map,
                                              daily_impacts_override=daily_impacts)
        if sh > best_sh + 1e-6:
            best_sh, best_k = sh, k
    best_k = max(k_floor, float(best_k))
    print(f"[calibrate] {tic}: best k = {best_k:.3f} (Sharpe={best_sh:.2f})")
    return float(best_k)

NEWS_FACTOR_BY_TICKER = {}
for tic in TICKERS:
    di = DAILY_IMPACTS_BY_TICKER.get(tic)
    if (di is not None) and (not di.empty):
        if FORCE_NEWS_K is not None:
            print(f"[calibrate] {tic}: FORCE k = {FORCE_NEWS_K:.3f}")
            NEWS_FACTOR_BY_TICKER[tic] = build_news_factor_from_daily(di, k=FORCE_NEWS_K, scale=1.20, cap=0.50)
        else:
            k_best = calibrate_news_k_for_ticker(tic, agent, BASKET[tic], di,
                                                 k_grid=(0.0,0.1,0.2,0.3,0.4,0.5),
                                                 k_floor=0.10)
            NEWS_FACTOR_BY_TICKER[tic] = build_news_factor_from_daily(di, k=k_best, scale=1.20, cap=0.50)

# ============================================
# Диагностика: файлы новостей, фактор, марковские фичи
# ============================================
print("\n[DEBUG] NEWS files loaded:")
for tic in TICKERS:
    df = NEWS_BY_TICKER.get(tic)
    di = DAILY_IMPACTS_BY_TICKER.get(tic)
    print(tic, "articles=", 0 if (df is None) else len(df),
          "daily_impacts=", 0 if (di is None) else len(di))

print("\n[DEBUG] NEWS factor variance:")
for tic in TICKERS:
    vals = list((NEWS_FACTOR_BY_TICKER.get(tic) or {}).values())
    v = (np.var(vals) if len(vals) > 0 else None)
    print(tic, "N=", len(vals), "var=", v)

print("\n[DEBUG] Markov features sanity (test split):")
for tic in TICKERS:
    df = BASKET[tic]["test"]
    cnt = df["REG_ID"].value_counts(dropna=False).to_dict()
    print(tic, "REG_ID counts:", cnt, "| REG_TREND std:", float(df["REG_TREND"].std()))

# ============================================
# Тест (инференс) на корзине
# ============================================
def print_inference_results(tag: str, tickers: List[str], agent):
    print(f"\n=== INFERENCE RESULTS {tag} ===")
    for tic in tickers:
        factor_override = NEWS_FACTOR_BY_TICKER.get(tic)
        daily_imp_override = DAILY_IMPACTS_BY_TICKER.get(tic)
        roi, sh, mdd, df_bt = run_inference_on_ticker(
            tic, agent, BASKET[tic], split="test",
            news_factor_override=factor_override,
            daily_impacts_override=daily_imp_override
        )
        print(f"{tic}: ROI {roi:.2f}% | Sharpe {sh:.2f} | MDD {mdd:.2%}")
        # при необходимости: df_bt.to_csv(f"{CHKPT_DIR}/bt_{tic}.csv", index=True)

print_inference_results("(basket, news + Markov SAFE)", TICKERS, agent)
