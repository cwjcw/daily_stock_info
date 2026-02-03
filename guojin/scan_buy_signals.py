# -*- coding: utf-8 -*-
"""
迅投 QMT 全市场选股扫描器 - 最终稳定版
功能：5项指标共振 + 倍量过滤 + 自动剔除涨停/ST/科创板
文档参考：https://dict.thinktrader.net/nativeApi/start_now.html
"""

import os
import sys
import math
import datetime
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from config import CONFIG

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# ================= 配置区域 =================
PORT = CONFIG['PORT']
SAVE_DIR = CONFIG['SAVE_DIR']
SCAN_SECTOR = CONFIG['SCAN_SECTOR']
TEST_SAMPLE = CONFIG['TEST_SAMPLE']
ENABLE_DOWNLOAD = CONFIG['ENABLE_DOWNLOAD']
MINUTE_COUNT = CONFIG['MINUTE_COUNT']
DAY_COUNT = CONFIG['DAY_COUNT']
CACHE_DIR = CONFIG['CACHE_DIR']
DIVIDEND_TYPE = CONFIG['DIVIDEND_TYPE']
RSI_COUNT = CONFIG['RSI_COUNT']
# 阈值/参数（统一从 CONFIG 读取）
SLOPE_THRESHOLD_BULL = CONFIG['SLOPE_THRESHOLD_BULL']
SLOPE_FLAT_MIN = CONFIG['SLOPE_FLAT_MIN']
SLOPE_FLAT_MAX = CONFIG['SLOPE_FLAT_MAX']
SLOPE_VETO = CONFIG['SLOPE_VETO']
SLOPE_SELL_FLAT_MIN = CONFIG['SLOPE_SELL_FLAT_MIN']
SLOPE_SELL_FLAT_MAX = CONFIG['SLOPE_SELL_FLAT_MAX']
MA20_NEAR_PCT = CONFIG['MA20_NEAR_PCT']
RSI_PERIODS = CONFIG['RSI_PERIODS']
RSI_BUY_MIN = CONFIG['RSI_BUY_MIN']
RSI_BUY_MAX = CONFIG['RSI_BUY_MAX']
RSI_SELL_HIGH = CONFIG['RSI_SELL_HIGH']
RSI_SELL_LOW = CONFIG['RSI_SELL_LOW']
RSI_DEAD_LINE = CONFIG['RSI_DEAD_LINE']
RSI_HIGH = CONFIG['RSI_HIGH']
RSI_MID = CONFIG['RSI_MID']
RSI_SELL_LOW_BUFFER = CONFIG['RSI_SELL_LOW_BUFFER']
VR_BUY_MIN = CONFIG['VR_BUY_MIN']
VR_BUY_MAX = CONFIG['VR_BUY_MAX']
VR_BUY_PART_MIN = CONFIG['VR_BUY_PART_MIN']
VR_BUY_PART_MAX = CONFIG['VR_BUY_PART_MAX']
VR_VETO_LOW = CONFIG['VR_VETO_LOW']
VR_VETO_HIGH = CONFIG['VR_VETO_HIGH']
VR_EARLY_SOFT_LOW = CONFIG['VR_EARLY_SOFT_LOW']
VR_EARLY_SOFT_HIGH = CONFIG['VR_EARLY_SOFT_HIGH']
VR_SOFT_CONFIRM_N = CONFIG['VR_SOFT_CONFIRM_N']
EARLY_START = CONFIG['EARLY_START']
EARLY_END = CONFIG['EARLY_END']
VR_SELL_FULL = CONFIG['VR_SELL_FULL']
VR_SELL_PART = CONFIG['VR_SELL_PART']
VR_SELL_VETO_DOWN_PCT = CONFIG['VR_SELL_VETO_DOWN_PCT']
VR_SELL_VETO_VOL = CONFIG['VR_SELL_VETO_VOL']
BOLL_PERIOD = CONFIG['BOLL_PERIOD']
BOLL_STD = CONFIG['BOLL_STD']
BOLL_TOUCH_SCORE = CONFIG['BOLL_TOUCH_SCORE']
SCORE_FULL = CONFIG['SCORE_FULL']
SCORE_BUY_TREND_FLAT = CONFIG['SCORE_BUY_TREND_FLAT']
SCORE_BUY_MA_PART = CONFIG['SCORE_BUY_MA_PART']
SCORE_BUY_MACD_PART = CONFIG['SCORE_BUY_MACD_PART']
SCORE_BUY_KDJ_PART = CONFIG['SCORE_BUY_KDJ_PART']
SCORE_BUY_RSI_PART1 = CONFIG['SCORE_BUY_RSI_PART1']
SCORE_BUY_RSI_PART2 = CONFIG['SCORE_BUY_RSI_PART2']
SCORE_BUY_BOLL_PART = CONFIG['SCORE_BUY_BOLL_PART']
SCORE_BUY_VOL_PART = CONFIG['SCORE_BUY_VOL_PART']
SCORE_SELL_TREND_PART = CONFIG['SCORE_SELL_TREND_PART']
SCORE_SELL_MA_PART = CONFIG['SCORE_SELL_MA_PART']
SCORE_SELL_MACD_PART = CONFIG['SCORE_SELL_MACD_PART']
SCORE_SELL_KDJ_PART = CONFIG['SCORE_SELL_KDJ_PART']
SCORE_SELL_RSI_PART = CONFIG['SCORE_SELL_RSI_PART']
SCORE_SELL_VOL_PART = CONFIG['SCORE_SELL_VOL_PART']
WEIGHT_TREND = CONFIG['WEIGHT_TREND']
WEIGHT_MACD = CONFIG['WEIGHT_MACD']
WEIGHT_RSI = CONFIG['WEIGHT_RSI']
WEIGHT_KDJ = CONFIG['WEIGHT_KDJ']
WEIGHT_BOLL = CONFIG['WEIGHT_BOLL']
WEIGHT_VR = CONFIG['WEIGHT_VR']
THRESH_STRONG_BUY = CONFIG['THRESH_STRONG_BUY']
THRESH_TRY_BUY_MIN = CONFIG['THRESH_TRY_BUY_MIN']
THRESH_TRY_BUY_MAX = CONFIG['THRESH_TRY_BUY_MAX']
THRESH_HOLD_MIN = CONFIG['THRESH_HOLD_MIN']
THRESH_HOLD_MAX = CONFIG['THRESH_HOLD_MAX']
KDJ_K_MAX_BUY = CONFIG['KDJ_K_MAX_BUY']
KDJ_K_NEAR_MIN = CONFIG['KDJ_K_NEAR_MIN']
KDJ_K_NEAR_MAX = CONFIG['KDJ_K_NEAR_MAX']
KDJ_K_VETO = CONFIG['KDJ_K_VETO']
KDJ_J_VETO = CONFIG['KDJ_J_VETO']
KDJ_J_FLAT_EPS = CONFIG['KDJ_J_FLAT_EPS']
MIN_K_BARS = CONFIG['MIN_K_BARS']
RET_ABS_MAX = CONFIG['RET_ABS_MAX']
MAX_MISSING_RATIO = CONFIG['MAX_MISSING_RATIO']
MAX_GAP_DAYS = CONFIG['MAX_GAP_DAYS']
# ===========================================

def calculate_indicators(df):
    """
    兼容旧调用，已不再使用
    """
    return False

def _parse_time_value(val) -> Optional[pd.Timestamp]:
    try:
        if isinstance(val, (np.integer, int)):
            ival = int(val)
            if ival > 10**12:
                return pd.to_datetime(ival, unit="ms")
            if ival > 10**10:
                return pd.to_datetime(ival, unit="s")
            s = str(ival)
        else:
            s = str(val)
        if len(s) == 14:
            return pd.to_datetime(s, format="%Y%m%d%H%M%S")
        if len(s) == 12:
            return pd.to_datetime(s, format="%Y%m%d%H%M")
        if len(s) == 8:
            return pd.to_datetime(s, format="%Y%m%d")
        return pd.to_datetime(s)
    except Exception:
        return None

def enrich_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        dt = df["time"].apply(_parse_time_value)
    else:
        dt = df.index.to_series().apply(_parse_time_value)
    if dt.isna().all():
        return df
    df = df.copy()
    df["dt"] = dt
    df = df.sort_values("dt")
    df["date"] = df["dt"].dt.date
    df["time_str"] = df["dt"].dt.strftime("%H%M")
    return df

def compute_volume_ratio(df_1m: pd.DataFrame) -> Optional[float]:
    """
    量比定义（TRADING_RULES.md）：
    VR(t) = (V_total_today / T_passed) / V_avg_5d
    V_avg_5d = 过去5日的日均每分钟成交量 = sum(DailyVolume)/ (5*240)
    """
    if "dt" not in df_1m.columns or "date" not in df_1m.columns:
        return None
    today = df_1m["date"].iloc[-1]
    today_df = df_1m[df_1m["date"] == today].sort_values("dt")
    if today_df.empty:
        return None
    minutes_passed = len(today_df)
    if minutes_passed <= 0:
        return None
    v_total_today = float(today_df["volume"].sum())
    past_dates = [d for d in df_1m["date"].unique() if d != today]
    if len(past_dates) < 1:
        return None
    past_dates = sorted(past_dates)[-5:]
    daily_vols = []
    for d in past_dates:
        ddf = df_1m[df_1m["date"] == d]
        if ddf.empty:
            continue
        daily_vols.append(float(ddf["volume"].sum()))
    if not daily_vols:
        return None
    v_avg_5d = float(np.mean(daily_vols)) / 240.0
    if v_avg_5d <= 0:
        return None
    return (v_total_today / minutes_passed) / v_avg_5d

def compute_volume_ratio_history(df_1m: pd.DataFrame, n: int) -> List[float]:
    if n <= 0:
        return []
    if "dt" not in df_1m.columns or "date" not in df_1m.columns:
        return []
    today = df_1m["date"].iloc[-1]
    today_df = df_1m[df_1m["date"] == today].sort_values("dt")
    if today_df.empty:
        return []
    past_dates = [d for d in df_1m["date"].unique() if d != today]
    if len(past_dates) < 1:
        return []
    past_dates = sorted(past_dates)[-5:]
    daily_vols = []
    for d in past_dates:
        ddf = df_1m[df_1m["date"] == d]
        if ddf.empty:
            continue
        daily_vols.append(float(ddf["volume"].sum()))
    if not daily_vols:
        return []
    v_avg_5d = float(np.mean(daily_vols)) / 240.0
    if v_avg_5d <= 0:
        return []

    ratios: List[float] = []
    total_minutes = len(today_df)
    for i in range(n):
        idx = total_minutes - i
        if idx <= 0:
            break
        slice_df = today_df.iloc[:idx]
        minutes_passed = len(slice_df)
        if minutes_passed <= 0:
            continue
        v_total_today = float(slice_df["volume"].sum())
        ratios.append((v_total_today / minutes_passed) / v_avg_5d)
    return list(reversed(ratios))

def _parse_hhmm(value: str) -> Optional[datetime.time]:
    try:
        return datetime.datetime.strptime(value, "%H:%M").time()
    except Exception:
        return None

def is_early_window(ts: Optional[pd.Timestamp]) -> bool:
    if ts is None:
        return False
    start = _parse_hhmm(EARLY_START)
    end = _parse_hhmm(EARLY_END)
    if not start or not end:
        return False
    t = ts.time()
    return start <= t < end

def _extract_dt_series(df: pd.DataFrame) -> pd.Series:
    if "time" in df.columns:
        dt = df["time"].apply(_parse_time_value)
    else:
        dt = df.index.to_series().apply(_parse_time_value)
    return dt.dropna().sort_values()

def _is_limit_move(code: str, ret_value: float) -> bool:
    if ret_value is None or np.isnan(ret_value):
        return False
    if code.startswith(("300", "301", "688")):
        limit_pct = 0.20
    else:
        limit_pct = 0.10
    return abs(ret_value) >= (limit_pct - 0.002)

def data_quality_hard_fail(code: str, df_1d: pd.DataFrame) -> Tuple[bool, str]:
    if df_1d is None or df_1d.empty:
        return True, "empty_or_none"
    if len(df_1d) < MIN_K_BARS:
        return True, "insufficient_bars"

    key_cols = ["open", "high", "low", "close", "volume"]
    for col in key_cols:
        if col not in df_1d.columns:
            return True, f"missing_col_{col}"

    df_tail = df_1d.tail(MIN_K_BARS).copy()
    missing_ratio = df_tail[key_cols].isna().mean().max()
    if missing_ratio > MAX_MISSING_RATIO:
        return True, "missing_ratio"

    o = df_tail["open"]
    h = df_tail["high"]
    l = df_tail["low"]
    c = df_tail["close"]
    v = df_tail["volume"]
    invalid_ohlc = (l > o) | (l > c) | (h < o) | (h < c) | (h < l)
    invalid_vol = v < 0
    if invalid_ohlc.any() or invalid_vol.any():
        return True, "ohlc_or_volume"

    dt = _extract_dt_series(df_tail)
    if len(dt) >= 2:
        gaps = dt.diff().dt.days
        if gaps.max() is not None and gaps.max() > MAX_GAP_DAYS:
            return True, "gap_days"

    close = df_tail["close"].astype(float)
    ret = close.pct_change()
    if len(ret) >= 2:
        jumps = ret.abs() > RET_ABS_MAX
        if jumps.any():
            for r in ret[jumps].dropna():
                if not _is_limit_move(code, float(r)):
                    return True, "ret_jump"

    return False, ""

def is_vr_soft_veto(metrics: Dict[str, Any]) -> bool:
    vr = metrics.get("量比_raw", None)
    if vr is None:
        return False
    if metrics.get("早盘窗口"):
        history = metrics.get("量比_history_raw", [])
        if len(history) < VR_SOFT_CONFIRM_N or VR_SOFT_CONFIRM_N <= 0:
            return False
        if all(v < VR_EARLY_SOFT_LOW for v in history):
            return True
        if all(v > VR_EARLY_SOFT_HIGH for v in history):
            return True
        return False
    return vr < VR_VETO_LOW or vr > VR_VETO_HIGH

def compute_price_metrics(df_1d: pd.DataFrame, df_1m: pd.DataFrame, float_volume: Optional[float]) -> Dict[str, Any]:
    df_1m = enrich_time_columns(df_1m)
    today = df_1m["date"].iloc[-1] if "date" in df_1m.columns else None
    if today is not None:
        today_df = df_1m[df_1m["date"] == today].sort_values("dt")
        open_price = float(today_df["open"].iloc[0])
        high_price = float(today_df["high"].max())
        low_price = float(today_df["low"].min())
        curr_price = float(today_df["close"].iloc[-1])
        vol_today = float(today_df["volume"].sum())
        last_dt = today_df["dt"].iloc[-1] if "dt" in today_df.columns else None
    else:
        open_price = float(df_1m["open"].iloc[-1])
        high_price = float(df_1m["high"].iloc[-1])
        low_price = float(df_1m["low"].iloc[-1])
        curr_price = float(df_1m["close"].iloc[-1])
        vol_today = float(df_1m["volume"].iloc[-1])
        last_dt = None

    prev_close = float(df_1d["close"].iloc[-2]) if len(df_1d) >= 2 else None
    change_pct = None
    if prev_close and prev_close > 0:
        change_pct = (curr_price / prev_close - 1) * 100

    turnover = None
    if float_volume and float_volume > 0:
        turnover = vol_today / float_volume * 100

    v_ratio = compute_volume_ratio(df_1m)
    v_ratio_hist = compute_volume_ratio_history(df_1m, VR_SOFT_CONFIRM_N)
    early_window = is_early_window(last_dt)

    return {
        "现价": curr_price,
        "开盘价": open_price,
        "最高价": high_price,
        "最低价": low_price,
        "涨幅%": round(change_pct, 2) if change_pct is not None else None,
        "换手率%": round(turnover, 2) if turnover is not None else None,
        "量比": round(v_ratio, 2) if v_ratio is not None else None,
        "量比_raw": v_ratio,
        "量比_history": [round(v, 2) for v in v_ratio_hist] if v_ratio_hist else [],
        "量比_history_raw": v_ratio_hist if v_ratio_hist else [],
        "早盘窗口": early_window,
        "今日成交量": vol_today,
    }

def compute_indicator_values(df_1d: pd.DataFrame, latest_price: float) -> Dict[str, Any]:
    """
    指标按日线计算，收盘价用盘中最新价替换当日收盘
    """
    df = df_1d.copy()
    df.loc[df.index[-1], "close"] = latest_price
    close = df["close"]
    low = df["low"]
    high = df["high"]
    vol = df["volume"]

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma5_prev = float(ma5.iloc[-2]) if len(ma5) >= 2 else float(ma5.iloc[-1])
    ma10_prev = float(ma10.iloc[-2]) if len(ma10) >= 2 else float(ma10.iloc[-1])
    ma20_prev = float(ma20.iloc[-2]) if len(ma20) >= 2 else float(ma20.iloc[-1])
    slope20 = (ma20.iloc[-1] - ma20.iloc[-2]) / ma20.iloc[-2] * 100 if len(ma20) >= 2 and ma20.iloc[-2] != 0 else 0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = (dif - dea) * 2
    hist_prev = float(hist.iloc[-2]) if len(hist) >= 2 else float(hist.iloc[-1])

    def rsi_wilder(series: pd.Series, n: int) -> float:
        # Wilder 原始算法：先用前n期均值做种子，再递归平滑
        if len(series) < n + 1:
            return float("nan")
        diffs = series.diff().dropna()
        gains = diffs.clip(lower=0.0)
        losses = (-diffs).clip(lower=0.0)
        avg_gain = gains.iloc[:n].mean()
        avg_loss = losses.iloc[:n].mean()
        for i in range(n, len(gains)):
            avg_gain = (avg_gain * (n - 1) + gains.iloc[i]) / n
            avg_loss = (avg_loss * (n - 1) + losses.iloc[i]) / n
        rs = avg_gain / avg_loss if avg_loss != 0 else math.inf
        return 100 - (100 / (1 + rs))

    rsi6 = rsi_wilder(close, RSI_PERIODS[0])
    rsi12 = rsi_wilder(close, RSI_PERIODS[1])
    rsi24 = rsi_wilder(close, RSI_PERIODS[2])

    low_min = low.rolling(9).min()
    high_max = high.rolling(9).max()
    rsv = (close - low_min) / (high_max - low_min + 1e-9) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    j_prev = float(j.iloc[-2]) if len(j) >= 2 else float(j.iloc[-1])

    mb = ma20
    sigma = close.rolling(BOLL_PERIOD).std()
    up_band = mb + BOLL_STD * sigma
    dn_band = mb - BOLL_STD * sigma

    v_ma5 = vol.rolling(5).mean()

    return {
        "MA5": round(float(ma5.iloc[-1]), 4),
        "MA10": round(float(ma10.iloc[-1]), 4),
        "MA20": round(float(ma20.iloc[-1]), 4),
        "MA5_prev": round(float(ma5_prev), 4),
        "MA10_prev": round(float(ma10_prev), 4),
        "MA20_prev": round(float(ma20_prev), 4),
        "Slope20": round(float(slope20), 4),
        "DIF": round(float(dif.iloc[-1]), 6),
        "DEA": round(float(dea.iloc[-1]), 6),
        "MACD_hist": round(float(hist.iloc[-1]), 6),
        "MACD_hist_prev": round(float(hist_prev), 6),
        "RSI6": round(float(rsi6), 4),
        "RSI12": round(float(rsi12), 4),
        "RSI24": round(float(rsi24), 4),
        "K": round(float(k.iloc[-1]), 4),
        "D": round(float(d.iloc[-1]), 4),
        "J": round(float(j.iloc[-1]), 4),
        "J_prev": round(float(j_prev), 4),
        "BOLL_mid": round(float(mb.iloc[-1]), 4),
        "BOLL_up": round(float(up_band.iloc[-1]), 4),
        "BOLL_dn": round(float(dn_band.iloc[-1]), 4),
        "VOL_MA5": round(float(v_ma5.iloc[-1]), 2),
    }

def _cache_key(name: str) -> str:
    day = datetime.datetime.now().strftime("%Y%m%d")
    return os.path.join(CACHE_DIR, f"{name}_{day}.pkl")

def load_cache(name: str) -> Optional[Any]:
    path = _cache_key(name)
    if os.path.exists(path):
        try:
            return pd.read_pickle(path)
        except Exception:
            return None
    return None

def save_cache(name: str, obj: Any) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_key(name)
    try:
        pd.to_pickle(obj, path)
    except Exception:
        pass

def calculate_sell_signals(df):
    """
    返回卖出信号布尔值字典
    """
    if df is None or len(df) < 35:
        return {}

    close = df['close']
    low = df['low']
    high = df['high']

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    cond_ma = (ma5.iloc[-1] < ma10.iloc[-1]) or (close.iloc[-1] < ma5.iloc[-1])

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = (dif - dea) * 2
    cond_macd = (dif.iloc[-1] < dea.iloc[-1]) or (hist.iloc[-1] < 0)

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    rs = gain / (loss + 1e-9)
    rsi6 = 100 - (100 / (1 + rs))
    gain12 = (delta.where(delta > 0, 0)).rolling(window=12).mean()
    loss12 = (-delta.where(delta < 0, 0)).rolling(window=12).mean()
    rs12 = gain12 / (loss12 + 1e-9)
    rsi12 = 100 - (100 / (1 + rs12))
    cond_rsi = (rsi6.iloc[-1] < 50) or (rsi6.iloc[-1] < rsi12.iloc[-1])

    low_min = low.rolling(9).min()
    high_max = high.rolling(9).max()
    rsv = (close - low_min) / (high_max - low_min + 1e-9) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    cond_kdj = (k.iloc[-1] < d.iloc[-1]) or (k.iloc[-1] > 80)

    cond_boll = close.iloc[-1] < ma20.iloc[-1]

    return {
        "MA": bool(cond_ma),
        "MACD": bool(cond_macd),
        "RSI": bool(cond_rsi),
        "KDJ": bool(cond_kdj),
        "BOLL": bool(cond_boll),
    }

def get_positions() -> List[Any]:
    if load_dotenv:
        load_dotenv()
    qmt_path = CONFIG.get("QMT_USERDATA_PATH", "")
    account_id = CONFIG.get("QMT_ACCOUNT_ID", "")
    account_type = CONFIG.get("QMT_ACCOUNT_TYPE", "STOCK")
    if not qmt_path or not account_id:
        print("⚠️ 未配置 QMT_USERDATA_PATH 或 QMT_ACCOUNT_ID，跳过持仓卖出信号。")
        return []

    session_id = int(datetime.datetime.now().timestamp())
    trader = XtQuantTrader(qmt_path, session_id)
    trader.start()
    trader.connect()
    account = StockAccount(account_id, account_type)
    try:
        positions = trader.query_stock_positions(account) or []
    except Exception:
        positions = []
    return positions

def run_scanner():
    start_ts = datetime.datetime.now()
    if load_dotenv:
        load_dotenv()
    print(f"📡 正在建立连接 (Port: {PORT})...")
    xtdata.connect(port=PORT)
    
    # 1. 获取初始名单
    stock_list = xtdata.get_stock_list_in_sector(SCAN_SECTOR)
    # 初步剔除科创板(688)和北交所(8, 4)
    filtered_list = [s for s in stock_list if not s.startswith(('688', '8', '4'))]
    if TEST_SAMPLE and len(filtered_list) > TEST_SAMPLE:
        import random
        filtered_list = random.sample(filtered_list, TEST_SAMPLE)
    
    print(f"📊 正在下载全市场历史数据 (共 {len(filtered_list)} 只)...")
    # 日线用于指标，1分钟用于当前价/量比/当日高低
    start_time_d = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime("%Y%m%d")
    start_time_m = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y%m%d")
    data_dict_1d = load_cache("data_1d")
    data_dict_1m = load_cache("data_1m")
    info_map = load_cache("info_map")
    if data_dict_1d is None or data_dict_1m is None or info_map is None:
        if ENABLE_DOWNLOAD:
            xtdata.download_history_data2(filtered_list, period='1d', start_time=start_time_d)
            xtdata.download_history_data2(filtered_list, period='1m', start_time=start_time_m)
    
    # 2. 批量读取数据
    print("🧠 正在进行多指标并行计算...")
    if data_dict_1d is None:
        data_dict_1d = xtdata.get_market_data_ex(
            [], filtered_list, period='1d', count=max(DAY_COUNT, RSI_COUNT), dividend_type=DIVIDEND_TYPE
        )
        save_cache("data_1d", data_dict_1d)
    if data_dict_1m is None:
        data_dict_1m = xtdata.get_market_data_ex(
            [], filtered_list, period='1m', count=MINUTE_COUNT, dividend_type=DIVIDEND_TYPE
        )
        save_cache("data_1m", data_dict_1m)
    if info_map is None:
        info_map = xtdata.get_instrument_detail_list(filtered_list, False) or {}
        save_cache("info_map", info_map)
    
    buy_rows = []
    total = len(filtered_list)
    last_print = -1
    hard_fail_counts: Dict[str, int] = {}
    hard_fail_rows: List[Dict[str, Any]] = []
    for idx, code in enumerate(filtered_list, start=1):
        progress = int(idx * 100 / max(total, 1))
        if progress != last_print and (progress % 5 == 0 or idx == total):
            print(f"⏳ 进度: {progress}% ({idx}/{total})")
            last_print = progress
        df_1d = data_dict_1d.get(code)
        df_1m = data_dict_1m.get(code)
        if df_1d is None or df_1d.empty or len(df_1d) < MIN_K_BARS:
            continue
        if df_1m is None or df_1m.empty or len(df_1m) < 100:
            continue
        hard_fail, hard_reason = data_quality_hard_fail(code, df_1d)
        if hard_fail:
            hard_fail_counts[hard_reason] = hard_fail_counts.get(hard_reason, 0) + 1
            hard_fail_rows.append({
                "代码": code,
                "阶段": "买入扫描",
                "原因": hard_reason
            })
            continue
            
        try:
            # 触发信号后才查询详细信息，极大地提升效率
            detail = info_map.get(code, {}) if isinstance(info_map, dict) else {}
            name = detail.get('InstrumentName', '未知') if isinstance(detail, dict) else '未知'
            float_volume = detail.get("FloatVolume", None) if isinstance(detail, dict) else None

            # 剔除 ST 股
            if 'ST' in name or '*' in name:
                continue

            metrics = compute_price_metrics(df_1d, df_1m, float_volume)
            vr_value = metrics.get("量比_raw", metrics["量比"])
            if metrics.get("早盘窗口"):
                vr_buy_max = VR_EARLY_MAX
            else:
                vr_buy_max = VR_BUY_MAX
            latest_price = metrics["现价"]
            ind_vals = compute_indicator_values(df_1d, latest_price)

            # 成交量（当日）
            curr_vol = metrics["今日成交量"]

            # 指标判断列（1/0）
            cond_trend = int(latest_price > ind_vals["MA20"] and ind_vals["Slope20"] > SLOPE_THRESHOLD_BULL)
            cond_ma = int(ind_vals["MA5"] > ind_vals["MA10"] > ind_vals["MA20"])
            cond_macd = int(ind_vals["DIF"] > ind_vals["DEA"] and ind_vals["MACD_hist"] > ind_vals["MACD_hist_prev"])
            cond_kdj = int(ind_vals["K"] > ind_vals["D"] and ind_vals["K"] < KDJ_K_MAX_BUY)
            cond_rsi = int(RSI_BUY_MIN < ind_vals["RSI6"] <= RSI_BUY_MAX)
            cond_boll = int(latest_price > ind_vals["BOLL_mid"] and latest_price < ind_vals["BOLL_up"])
            cond_vol_ratio = int(vr_value is not None and VR_BUY_MIN <= vr_value <= vr_buy_max)
            cond_vol = int(cond_vol_ratio == 1 and metrics["现价"] > metrics["开盘价"])

            # 评分规则（部分得分 + 一票否决）
            trend_veto = int(latest_price < ind_vals["MA20"] or ind_vals["Slope20"] < SLOPE_VETO)
            trend_score = SCORE_FULL if cond_trend else (SCORE_BUY_TREND_FLAT if (latest_price > ind_vals["MA20"] and SLOPE_FLAT_MIN <= ind_vals["Slope20"] <= SLOPE_FLAT_MAX) else 0.0)
            if trend_veto:
                trend_score = 0.0

            ma_veto = int(ind_vals["MA5"] < ind_vals["MA20"])
            ma_score = SCORE_FULL if cond_ma else (SCORE_BUY_MA_PART if (ind_vals["MA5"] > ind_vals["MA10"] and ind_vals["MA5"] < ind_vals["MA20"]) else 0.0)
            if ma_veto:
                ma_score = 0.0

            macd_veto = int(ind_vals["DIF"] < 0 and ind_vals["DIF"] < ind_vals["DEA"])
            macd_score = SCORE_FULL if (ind_vals["DIF"] > ind_vals["DEA"] and ind_vals["MACD_hist"] > ind_vals["MACD_hist_prev"]) else (
                SCORE_BUY_MACD_PART if (ind_vals["DIF"] > ind_vals["DEA"]) else 0.0
            )
            if macd_veto:
                macd_score = 0.0

            kdj_veto = int(ind_vals["J"] > KDJ_J_VETO or ind_vals["K"] > KDJ_K_VETO)
            kdj_score = SCORE_FULL if (ind_vals["K"] > ind_vals["D"] and ind_vals["K"] < KDJ_K_MAX_BUY and ind_vals["J"] > ind_vals["J_prev"]) else (
                SCORE_BUY_KDJ_PART if (ind_vals["K"] > ind_vals["D"] and KDJ_K_NEAR_MIN <= ind_vals["K"] < KDJ_K_NEAR_MAX) else 0.0
            )
            if kdj_veto:
                kdj_score = 0.0

            rsi_veto = int(ind_vals["RSI6"] < RSI_DEAD_LINE or (ind_vals["RSI6"] > RSI_BUY_MAX and ind_vals["RSI6"] < ind_vals["RSI12"]))
            rsi_score = SCORE_FULL if (ind_vals["RSI6"] > ind_vals["RSI12"] > ind_vals["RSI24"]) else (
                SCORE_BUY_RSI_PART1 if (ind_vals["RSI6"] > ind_vals["RSI12"] and ind_vals["RSI12"] > RSI_MID) else (
                    SCORE_BUY_RSI_PART2 if (ind_vals["RSI6"] > ind_vals["RSI12"]) else 0.0
                )
            )
            if rsi_veto:
                rsi_score = 0.0

            boll_veto = int(latest_price < ind_vals["BOLL_mid"])
            boll_score = SCORE_FULL if (latest_price > ind_vals["BOLL_mid"] and latest_price < ind_vals["BOLL_up"] and ind_vals["BOLL_up"] > ind_vals["BOLL_mid"]) else (
                SCORE_BUY_BOLL_PART if (latest_price > ind_vals["BOLL_mid"]) else 0.0
            )
            if boll_veto:
                boll_score = 0.0

            vol_veto = int(is_vr_soft_veto(metrics))
            vol_score = SCORE_FULL if (vr_value is not None and VR_BUY_MIN <= vr_value <= vr_buy_max and metrics["现价"] > metrics["开盘价"]) else (
                SCORE_BUY_VOL_PART if (vr_value is not None and VR_BUY_PART_MIN <= vr_value < VR_BUY_PART_MAX) else 0.0
            )
            if vol_veto:
                vol_score = 0.0

            veto_any = int(trend_veto or ma_veto or macd_veto or kdj_veto or rsi_veto or boll_veto or vol_veto)
            score_total = round(
                trend_score * WEIGHT_TREND
                + macd_score * WEIGHT_MACD
                + rsi_score * WEIGHT_RSI
                + kdj_score * WEIGHT_KDJ
                + boll_score * WEIGHT_BOLL
                + vol_score * WEIGHT_VR,
                4
            )

            hit_count = int(trend_score > 0) + int(ma_score > 0) + int(macd_score > 0) + int(rsi_score > 0) + int(kdj_score > 0) + int(boll_score > 0) + int(vol_score > 0)
            all_hit = int(hit_count == 7)

            buy_rows.append({
                '代码': code,
                '名称': name,
                '现价': metrics["现价"],
                '开盘价': metrics["开盘价"],
                '最高价': metrics["最高价"],
                '最低价': metrics["最低价"],
                '涨幅%': metrics["涨幅%"],
                '换手率%': metrics["换手率%"],
                '量比': metrics["量比"],
                'MA5': ind_vals["MA5"],
                'MA10': ind_vals["MA10"],
                'MA20': ind_vals["MA20"],
                'Slope20%': ind_vals["Slope20"],
                '趋势_满足': cond_trend,
                '趋势_评分': trend_score,
                '趋势_否决': trend_veto,
                'MA_满足': cond_ma,
                '排列_评分': ma_score,
                '排列_否决': ma_veto,
                'DIF': ind_vals["DIF"],
                'DEA': ind_vals["DEA"],
                'MACD_hist': ind_vals["MACD_hist"],
                'MACD_hist_prev': ind_vals["MACD_hist_prev"],
                'MACD_满足': cond_macd,
                '动能_评分': macd_score,
                '动能_否决': macd_veto,
                'RSI6': ind_vals["RSI6"],
                'RSI12': ind_vals["RSI12"],
                'RSI24': ind_vals["RSI24"],
                'RSI_满足': cond_rsi,
                '强度_评分': rsi_score,
                '强度_否决': rsi_veto,
                'K': ind_vals["K"],
                'D': ind_vals["D"],
                'J': ind_vals["J"],
                'J_prev': ind_vals["J_prev"],
                'KDJ_满足': cond_kdj,
                '超买_评分': kdj_score,
                '超买_否决': kdj_veto,
                'BOLL_mid': ind_vals["BOLL_mid"],
                'BOLL_up': ind_vals["BOLL_up"],
                'BOLL_dn': ind_vals["BOLL_dn"],
                'BOLL_满足': cond_boll,
                '通道_评分': boll_score,
                '通道_否决': boll_veto,
                'VOL_MA5': ind_vals["VOL_MA5"],
                '量比_满足': cond_vol_ratio,
                '量能_满足': cond_vol,
                '能量_评分': vol_score,
                '能量_否决': vol_veto,
                '全部买入信号': all_hit,
                '买入命中数': hit_count,
                '买入总分': score_total,
                '一票否决': veto_any,
            })
        except Exception as e:
            continue

    # 3. 持仓卖出信号
    results = [r for r in buy_rows if r.get("全部买入信号") == 1]
    positions = get_positions()
    sell_rows: List[Dict[str, Any]] = []
    if positions:
        for pos in positions:
            code = getattr(pos, "stock_code", "")
            if not code:
                continue
            df_1d = data_dict_1d.get(code)
            df_1m = data_dict_1m.get(code)
            if df_1d is None or df_1d.empty or len(df_1d) < MIN_K_BARS:
                continue
            if df_1m is None or df_1m.empty or len(df_1m) < 100:
                continue
            hard_fail, hard_reason = data_quality_hard_fail(code, df_1d)
            if hard_fail:
                hard_fail_counts[hard_reason] = hard_fail_counts.get(hard_reason, 0) + 1
                hard_fail_rows.append({
                    "代码": code,
                    "阶段": "持仓卖出扫描",
                    "原因": hard_reason
                })
                continue
            detail = info_map.get(code, {}) if isinstance(info_map, dict) else {}
            float_volume = detail.get("FloatVolume", None) if isinstance(detail, dict) else None
            metrics = compute_price_metrics(df_1d, df_1m, float_volume)
            vr_value = metrics.get("量比_raw", metrics["量比"])
            latest_price = metrics["现价"]
            ind_vals = compute_indicator_values(df_1d, latest_price)

            # 卖出规则（TRADING_RULES）
            cond_ma = int(latest_price < ind_vals["MA20"])
            cond_ma_combo = int(ind_vals["MA5"] < ind_vals["MA10"])
            cond_macd = int(ind_vals["DIF"] < ind_vals["DEA"])
            cond_kdj = int(ind_vals["K"] > KDJ_K_VETO or (ind_vals["J"] < ind_vals["J_prev"]))
            cond_rsi = int(ind_vals["RSI6"] < RSI_SELL_LOW or ind_vals["RSI6"] > RSI_SELL_HIGH)
            cond_boll = int(latest_price < ind_vals["BOLL_mid"])
            cond_vol = int(vr_value is not None and vr_value < VR_SELL_FULL and metrics["现价"] > metrics["开盘价"])

            # 卖出评分（部分得分 + 一票否决）
            # 趋势：满分=Slope20转负且Ct贴近MA20；部分=Slope20走平/显著减小；否决=Ct<MA20
            sell_trend_veto = int(latest_price < ind_vals["MA20"])
            near_ma20 = abs(latest_price - ind_vals["MA20"]) / ind_vals["MA20"] <= MA20_NEAR_PCT if ind_vals["MA20"] else False
            sell_trend_score = SCORE_FULL if (ind_vals["Slope20"] < 0 and near_ma20) else (
                SCORE_SELL_TREND_PART if (SLOPE_SELL_FLAT_MIN <= ind_vals["Slope20"] <= SLOPE_SELL_FLAT_MAX) else 0.0
            )
            if sell_trend_veto:
                sell_trend_score = 0.0

            # 排列：满分=MA5<MA10；部分=MA5拐头向下但未穿MA10；否决=MA5<MA20
            sell_ma_veto = int(ind_vals["MA5"] < ind_vals["MA20"])
            sell_ma_score = SCORE_FULL if (ind_vals["MA5"] < ind_vals["MA10"]) else (
                SCORE_SELL_MA_PART if (ind_vals["MA5"] < ind_vals["MA5_prev"] and ind_vals["MA5"] >= ind_vals["MA10"]) else 0.0
            )
            if sell_ma_veto:
                sell_ma_score = 0.0

            # 动能：满分=DIF<DEA（高位死叉）；部分=HIST红柱连续2日缩短；否决=DIF<0
            sell_macd_veto = int(ind_vals["DIF"] < 0)
            sell_macd_score = SCORE_FULL if (ind_vals["DIF"] < ind_vals["DEA"]) else (
                SCORE_SELL_MACD_PART if (ind_vals["MACD_hist"] > 0 and ind_vals["MACD_hist"] < ind_vals["MACD_hist_prev"]) else 0.0
            )
            if sell_macd_veto:
                sell_macd_score = 0.0

            # 超限：满分=J从高位掉头且K>80；部分=J走平或K>90；否决=J>100且大阴线
            sell_kdj_veto = int(ind_vals["J"] > KDJ_J_VETO and metrics["现价"] < metrics["开盘价"])
            sell_kdj_score = SCORE_FULL if (ind_vals["K"] > KDJ_K_MAX_BUY and ind_vals["J"] < ind_vals["J_prev"]) else (
                SCORE_SELL_KDJ_PART if (ind_vals["K"] > KDJ_K_VETO or abs(ind_vals["J"] - ind_vals["J_prev"]) < KDJ_J_FLAT_EPS) else 0.0
            )
            if sell_kdj_veto:
                sell_kdj_score = 0.0

            # 强度：满分/部分按细则；否决=RSI6<40
            sell_rsi_veto = int(ind_vals["RSI6"] < RSI_DEAD_LINE)
            sell_rsi_score = SCORE_FULL if (ind_vals["RSI6"] < RSI_SELL_LOW or ind_vals["RSI6"] > RSI_SELL_HIGH) else (
                SCORE_SELL_RSI_PART if (ind_vals["RSI6"] < RSI_SELL_LOW + RSI_SELL_LOW_BUFFER or ind_vals["RSI6"] > RSI_BUY_MAX) else 0.0
            )
            if sell_rsi_veto:
                sell_rsi_score = 0.0

            # 通道：满分=Ct跌破中轨；部分=触碰上轨受阻回落；否决=Ct跌破下轨
            sell_boll_veto = int(latest_price < ind_vals["BOLL_dn"])
            sell_boll_score = 1.0 if (latest_price < ind_vals["BOLL_mid"]) else (
                BOLL_TOUCH_SCORE if (metrics["最高价"] >= ind_vals["BOLL_up"] and latest_price < ind_vals["BOLL_up"]) else 0.0
            )
            if sell_boll_veto:
                sell_boll_score = 0.0

            # 能量：满分=缩量上涨（VR<0.8）；部分=异常巨量滞涨（VR>4.0）；否决=跌停放量
            sell_vol_veto = int(metrics["涨幅%"] is not None and metrics["涨幅%"] <= VR_SELL_VETO_DOWN_PCT and vr_value is not None and vr_value > VR_SELL_VETO_VOL)
            sell_vol_score = SCORE_FULL if (vr_value is not None and vr_value < VR_SELL_FULL) else (
                SCORE_SELL_VOL_PART if (vr_value is not None and vr_value > VR_SELL_PART) else 0.0
            )
            if sell_vol_veto:
                sell_vol_score = 0.0

            sell_veto_any = int(sell_trend_veto or sell_ma_veto or sell_macd_veto or sell_kdj_veto or sell_rsi_veto or sell_boll_veto or sell_vol_veto)
            sell_score_total = round(
                sell_trend_score * WEIGHT_TREND
                + sell_macd_score * WEIGHT_MACD
                + sell_rsi_score * WEIGHT_RSI
                + sell_kdj_score * WEIGHT_KDJ
                + sell_boll_score * WEIGHT_BOLL
                + sell_vol_score * WEIGHT_VR,
                4
            )

            hit_count = int(sell_trend_score > 0) + int(sell_ma_score > 0) + int(sell_macd_score > 0) + int(sell_kdj_score > 0) + int(sell_rsi_score > 0) + int(sell_boll_score > 0) + int(sell_vol_score > 0)
            sell_rows.append({
                "代码": code,
                "名称": getattr(pos, "instrument_name", ""),
                "持仓数量": getattr(pos, "volume", 0),
                "现价": metrics["现价"],
                "开盘价": metrics["开盘价"],
                "最高价": metrics["最高价"],
                "最低价": metrics["最低价"],
                "涨幅%": metrics["涨幅%"],
                "换手率%": metrics["换手率%"],
                "量比": metrics["量比"],
                "MA5": ind_vals["MA5"],
                "MA10": ind_vals["MA10"],
                "MA20": ind_vals["MA20"],
                "Slope20%": ind_vals["Slope20"],
                "趋势_满足": cond_ma,
                "趋势_评分": sell_trend_score,
                "趋势_否决": sell_trend_veto,
                "MA_满足": cond_ma_combo,
                "排列_评分": sell_ma_score,
                "排列_否决": sell_ma_veto,
                "DIF": ind_vals["DIF"],
                "DEA": ind_vals["DEA"],
                "MACD_hist": ind_vals["MACD_hist"],
                "MACD_hist_prev": ind_vals["MACD_hist_prev"],
                "MACD_满足": cond_macd,
                "动能_评分": sell_macd_score,
                "动能_否决": sell_macd_veto,
                "RSI6": ind_vals["RSI6"],
                "RSI12": ind_vals["RSI12"],
                "RSI24": ind_vals["RSI24"],
                "RSI_满足": cond_rsi,
                "强度_评分": sell_rsi_score,
                "强度_否决": sell_rsi_veto,
                "K": ind_vals["K"],
                "D": ind_vals["D"],
                "J": ind_vals["J"],
                "J_prev": ind_vals["J_prev"],
                "KDJ_满足": cond_kdj,
                "超买_评分": sell_kdj_score,
                "超买_否决": sell_kdj_veto,
                "BOLL_mid": ind_vals["BOLL_mid"],
                "BOLL_up": ind_vals["BOLL_up"],
                "BOLL_dn": ind_vals["BOLL_dn"],
                "BOLL_满足": cond_boll,
                "通道_评分": sell_boll_score,
                "通道_否决": sell_boll_veto,
                "量比_满足": cond_vol,
                "能量_评分": sell_vol_score,
                "能量_否决": sell_vol_veto,
                "任一卖出信号": int(hit_count > 0),
                "卖出命中数": hit_count,
                "卖出总分": sell_score_total,
                "一票否决": sell_veto_any,
            })

    # 4. 输出与保存
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    buy_df = pd.DataFrame(buy_rows) if buy_rows else pd.DataFrame(
        columns=[
            "代码","名称","现价","开盘价","最高价","最低价","涨幅%","换手率%","量比",
            "MA5","MA10","MA20","Slope20%","趋势_满足","趋势_评分","趋势_否决",
            "MA_满足","排列_评分","排列_否决",
            "DIF","DEA","MACD_hist","MACD_hist_prev","MACD_满足","动能_评分","动能_否决",
            "RSI6","RSI12","RSI24","RSI_满足","强度_评分","强度_否决",
            "K","D","J","J_prev","KDJ_满足","超买_评分","超买_否决",
            "BOLL_mid","BOLL_up","BOLL_dn","BOLL_满足","通道_评分","通道_否决",
            "VOL_MA5","量比_满足","量能_满足","能量_评分","能量_否决",
            "全部买入信号","买入命中数","买入总分","一票否决"
        ]
    )
    if not buy_df.empty and "买入命中数" in buy_df.columns:
        buy_df = buy_df.sort_values(by="买入命中数", ascending=False)

    sell_df = pd.DataFrame(sell_rows) if sell_rows else pd.DataFrame(
        columns=[
            "代码","名称","持仓数量","现价","开盘价","最高价","最低价","涨幅%","换手率%","量比",
            "MA5","MA10","MA20","Slope20%","趋势_满足","趋势_评分","趋势_否决",
            "MA_满足","排列_评分","排列_否决",
            "DIF","DEA","MACD_hist","MACD_hist_prev","MACD_满足","动能_评分","动能_否决",
            "RSI6","RSI12","RSI24","RSI_满足","强度_评分","强度_否决",
            "K","D","J","J_prev","KDJ_满足","超买_评分","超买_否决",
            "BOLL_mid","BOLL_up","BOLL_dn","BOLL_满足","通道_评分","通道_否决",
            "量比_满足","能量_评分","能量_否决",
            "任一卖出信号","卖出命中数","卖出总分","一票否决"
        ]
    )
    if not sell_df.empty and "卖出命中数" in sell_df.columns:
        sell_df = sell_df.sort_values(by="卖出命中数", ascending=False)

    hard_fail_df = pd.DataFrame(hard_fail_rows) if hard_fail_rows else pd.DataFrame(
        columns=["代码", "阶段", "原因"]
    )
    if not hard_fail_df.empty:
        hard_fail_df = hard_fail_df.sort_values(by=["原因", "代码"])

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"{SAVE_DIR}/信号扫描_{timestamp}.xlsx"
    try:
        with pd.ExcelWriter(file_name) as writer:
            buy_df.to_excel(writer, sheet_name="买入信号", index=False)
            sell_df.to_excel(writer, sheet_name="持仓卖出信号", index=False)
            hard_fail_df.to_excel(writer, sheet_name="Hard_Fail", index=False)
    except Exception:
        # 回退为CSV输出，避免缺少Excel依赖导致失败
        buy_csv = f"{SAVE_DIR}/买入信号_{timestamp}.csv"
        sell_csv = f"{SAVE_DIR}/持仓卖出信号_{timestamp}.csv"
        hard_csv = f"{SAVE_DIR}/Hard_Fail_{timestamp}.csv"
        buy_df.to_csv(buy_csv, index=False, encoding="utf-8-sig")
        sell_df.to_csv(sell_csv, index=False, encoding="utf-8-sig")
        hard_fail_df.to_csv(hard_csv, index=False, encoding="utf-8-sig")
        file_name = f"{buy_csv} / {sell_csv} / {hard_csv}"

    if results:
        print(f"\n🎉 扫描完成！找到 {len(results)} 个信号。")
    else:
        print("\n🍵 扫描结束，当前市场未发现符合共振条件的标的。")

    if hard_fail_counts:
        total_fail = sum(hard_fail_counts.values())
        reason_parts = [f"{k}:{v}" for k, v in sorted(hard_fail_counts.items())]
        print(f"🧹 Hard Fail 剔除: {total_fail} | " + ", ".join(reason_parts))

    print(f"💾 文件已存至: {file_name}")
    elapsed_min = (datetime.datetime.now() - start_ts).total_seconds() / 60.0
    print(f"⏱️ 全量扫描耗时: {elapsed_min:.2f} 分钟")

if __name__ == "__main__":
    run_scanner()
