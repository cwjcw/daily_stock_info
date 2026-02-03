import os
import time
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd
import tushare as ts
from dotenv import load_dotenv

import pathlib
import sys
import uuid
import urllib.parse
import requests

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kuai_proxy import enable_kuai_proxy

class Config:
    def __init__(self):
        self.env_path = ".env"
        load_dotenv(self.env_path)
        self.tushare_token = os.getenv("TUSHARE_TOKEN", "").strip()
        self.xq_token = os.getenv("XQ_TOKEN", "").strip()
        self.spot_source = os.getenv("SPOT_SOURCE", "sina").strip().lower()
        self.use_kuai_proxy = os.getenv("USE_KUAI_PROXY", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.kuai_user = os.getenv("KUAI_USER", "").strip()
        self.kuai_pass = os.getenv("KUAI_PASS", "").strip()
        self.kuai_host = os.getenv("KUAI_HOST", "tps.kdlapi.com").strip()
        self.kuai_port = int(os.getenv("KUAI_PORT", "15818"))
        self.kuai_mode = os.getenv("KUAI_MODE", "tps").strip().lower()
        self.kdl_secret_id = os.getenv("KDL_SECRET_ID", "").strip()
        self.kdl_secret_key = os.getenv("KDL_SECRET_KEY", "").strip()
        self.kdl_sign_type = os.getenv("KDL_SIGN_TYPE", "token").strip()
        self.kdl_api = os.getenv("KDL_API", "dps").strip().lower()
        self.kdl_api_num = int(os.getenv("KDL_API_NUM", "1"))
        self.disable_proxy = os.getenv("DISABLE_PROXY", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.max_symbols = int(os.getenv("MAX_SYMBOLS", "0"))
        self.sleep_seconds = float(os.getenv("TS_SLEEP", "0.2"))
        self.xq_sleep = float(os.getenv("XQ_SLEEP", "0.2"))
        if self.disable_proxy:
            self._disable_proxy_env()
        if self.use_kuai_proxy:
            if not self.kuai_user or not self.kuai_pass:
                raise ValueError("KUAI_USER/KUAI_PASS not set in .env")
            enable_kuai_proxy(
                self.kuai_user,
                self.kuai_pass,
                host=self.kuai_host,
                port=self.kuai_port,
                mode=self.kuai_mode,
                secret_id=self.kdl_secret_id,
                secret_key=self.kdl_secret_key,
                sign_type=self.kdl_sign_type,
                api=self.kdl_api,
                api_params={"num": self.kdl_api_num},
            )

    @staticmethod
    def _disable_proxy_env():
        for key in [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ]:
            os.environ.pop(key, None)
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"


def xq_symbol(code):
    return f"SH{code}" if code.startswith("6") else f"SZ{code}"


def build_spot_xq(config, pro):
    if not config.xq_token:
        print("[WARN] 未设置 XQ_TOKEN，跳过雪球实时行情")
        return pd.DataFrame()
    try:
        basic = pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,name",
        )
    except Exception as exc:
        print(f"[WARN] 获取股票列表失败: {exc}")
        return pd.DataFrame()
    if basic is None or basic.empty:
        return pd.DataFrame()

    basic = basic.copy()
    basic["code"] = basic["ts_code"].str.slice(0, 6)
    if config.max_symbols > 0:
        basic = basic.head(config.max_symbols)

    rows = []
    total = len(basic)
    print(f"[INFO] 雪球实时行情准备抓取 {total} 只")
    fail_count = 0
    for idx, row in basic.reset_index(drop=True).iterrows():
        code = row["code"]
        name = row["name"]
        try:
            df = ak.stock_individual_spot_xq(
                symbol=xq_symbol(code),
                token=config.xq_token,
            )
        except Exception as exc:
            print(f"[WARN] 雪球行情失败 {code}: {exc}")
            fail_count += 1
            time.sleep(config.xq_sleep)
            continue
        if df is None or df.empty:
            fail_count += 1
            time.sleep(config.xq_sleep)
            continue
        data_map = dict(zip(df["item"], df["value"]))
        rows.append(
            {
                "代码": code,
                "名称": data_map.get("名称", name),
                "最新价": data_map.get("现价"),
                "涨跌幅": data_map.get("涨幅"),
                "最高": data_map.get("最高"),
                "最低": data_map.get("最低"),
                "今开": data_map.get("今开"),
            }
        )
        if (idx + 1) % 200 == 0 or (idx + 1) == total:
            print(f"[INFO] 雪球进度 {idx + 1}/{total}，成功 {len(rows)} 只，失败 {fail_count} 只")
        time.sleep(config.xq_sleep)
    return pd.DataFrame(rows)


def get_spot_df(config, pro):
    source = config.spot_source
    if source == "xq":
        spot = build_spot_xq(config, pro)
        print(f"[INFO] 雪球实时行情返回 {len(spot)} 行")
        return spot
    if source == "sina":
        try:
            data = ak.stock_zh_a_spot()
            print(f"[INFO] 新浪实时行情返回 {len(data)} 行")
            return data
        except Exception as exc:
            print(f"[WARN] 新浪实时行情失败: {exc}")
            return pd.DataFrame()
    if source == "em":
        try:
            data = ak.stock_zh_a_spot_em()
            print(f"[INFO] 东财实时行情返回 {len(data)} 行")
            return data
        except Exception as exc:
            print(f"[WARN] 东财实时行情失败: {exc}")
            return pd.DataFrame()
    # auto: xq -> em -> sina
    spot = build_spot_xq(config, pro)
    if isinstance(spot, pd.DataFrame) and not spot.empty:
        print(f"[INFO] 雪球实时行情返回 {len(spot)} 行")
        return spot
    try:
        data = ak.stock_zh_a_spot_em()
        if isinstance(data, pd.DataFrame) and not data.empty:
            print(f"[INFO] 东财实时行情返回 {len(data)} 行")
            return data
    except Exception as exc:
        print(f"[WARN] 东财实时行情失败: {exc}")
    try:
        data = ak.stock_zh_a_spot()
        print(f"[INFO] 新浪实时行情返回 {len(data)} 行")
        return data
    except Exception as exc:
        print(f"[WARN] 新浪实时行情失败: {exc}")
        return pd.DataFrame()


def normalize_ts_code(code):
    if code.startswith("6"):
        return f"{code}.SH"
    return f"{code}.SZ"


def compute_indicators(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ma5 = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea

    diff = close.diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_gain6 = gain.rolling(6).mean()
    avg_loss6 = loss.rolling(6).mean()
    avg_gain12 = gain.rolling(12).mean()
    avg_loss12 = loss.rolling(12).mean()
    rsi6 = 100 - 100 / (1 + (avg_gain6 / avg_loss6.replace(0, pd.NA)))
    rsi12 = 100 - 100 / (1 + (avg_gain12 / avg_loss12.replace(0, pd.NA)))

    lowest = low.rolling(9).min()
    highest = high.rolling(9).max()
    rsv = (close - lowest) / (highest - lowest) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()

    return {
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "dif": dif.iloc[-1],
        "dea": dea.iloc[-1],
        "hist": hist.iloc[-1],
        "rsi6": rsi6.iloc[-1],
        "rsi12": rsi12.iloc[-1],
        "k": k.iloc[-1],
        "d": d.iloc[-1],
    }


def check_buy_rule(latest_close, indicators):
    ma_ok = (
        indicators["ma5"] > indicators["ma10"] > indicators["ma20"]
        and latest_close > indicators["ma20"]
    )
    macd_ok = indicators["dif"] > indicators["dea"] and indicators["hist"] > 0
    rsi_ok = indicators["rsi6"] > 50 and indicators["rsi6"] > indicators["rsi12"]
    kdj_ok = indicators["k"] > indicators["d"] and indicators["k"] < 80
    boll_ok = latest_close > indicators["ma20"]
    return ma_ok and macd_ok and rsi_ok and kdj_ok and boll_ok


def main():
    config = Config()
    if not config.tushare_token:
        raise ValueError("TUSHARE_TOKEN not set in .env")
    ts.set_token(config.tushare_token)
    pro = ts.pro_api()

    spot = get_spot_df(config, pro)
    if spot.empty:
        print("[ERROR] 实时行情为空，请切换 SPOT_SOURCE 或检查网络")
        return
    print(f"[INFO] 实时行情原始行数: {len(spot)}")
    spot = spot.copy()

    if "代码" in spot.columns:
        code_col = "代码"
    else:
        code_col = "code"

    name_col = "名称" if "名称" in spot.columns else "name"
    price_col = "最新价" if "最新价" in spot.columns else "trade"
    pct_col = "涨跌幅" if "涨跌幅" in spot.columns else "changepercent"
    high_col = "最高" if "最高" in spot.columns else "high"
    low_col = "最低" if "最低" in spot.columns else "low"
    open_col = "今开" if "今开" in spot.columns else "open"

    spot = spot[[code_col, name_col, price_col, pct_col, high_col, low_col, open_col]]
    spot = spot.dropna(subset=[code_col, price_col])
    spot[code_col] = spot[code_col].astype(str)

    # 过滤：A股、非科创、非ST、涨幅<=9%
    def is_valid(row):
        code = row[code_col]
        name = str(row[name_col])
        if len(code) != 6:
            return False
        if code.startswith("688"):
            return False
        if "ST" in name.upper():
            return False
        pct = row[pct_col]
        try:
            if float(pct) > 9:
                return False
        except Exception:
            return False
        return True

    candidates = spot[spot.apply(is_valid, axis=1)]
    print(f"[INFO] 过滤后候选数量: {len(candidates)}")

    if config.max_symbols > 0:
        candidates = candidates.head(config.max_symbols)

    results = []
    total = len(candidates)
    if total == 0:
        print("候选池为空，请检查实时行情过滤条件")
        return
    print(f"开始扫描：候选 {total} 只")
    today = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d")

    for idx, row in candidates.reset_index(drop=True).iterrows():
        code = row[code_col]
        name = row[name_col]
        latest_price = float(row[price_col])
        high_today = float(row[high_col]) if pd.notna(row[high_col]) else latest_price
        low_today = float(row[low_col]) if pd.notna(row[low_col]) else latest_price
        open_today = float(row[open_col]) if pd.notna(row[open_col]) else latest_price

        ts_code = normalize_ts_code(code)
        try:
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=today)
        except Exception:
            time.sleep(config.sleep_seconds)
            continue
        if df is None or df.empty:
            time.sleep(config.sleep_seconds)
            continue
        df = df.sort_values("trade_date")
        df = df.rename(
            columns={"open": "open", "close": "close", "high": "high", "low": "low"}
        )

        if pd.isna(row[pct_col]):
            prev_close = df.iloc[-1]["close"]
            if prev_close:
                row[pct_col] = (latest_price - float(prev_close)) / float(prev_close) * 100

        today_row = pd.DataFrame(
            [
                {
                    "trade_date": today,
                    "open": open_today,
                    "close": latest_price,
                    "high": high_today,
                    "low": low_today,
                }
            ]
        )
        df = pd.concat([df, today_row], ignore_index=True)
        df = df[["open", "close", "high", "low"]].astype(float)

        if len(df) < 26:
            time.sleep(config.sleep_seconds)
            continue

        indicators = compute_indicators(df)
        if check_buy_rule(latest_price, indicators):
            results.append(
                {
                    "code": code,
                    "name": name,
                    "latest_price": latest_price,
                    "pct": row[pct_col],
                }
            )

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"进度 {idx + 1}/{total}，已找到 {len(results)} 只")
        time.sleep(config.sleep_seconds)

    out = pd.DataFrame(results)
    if out.empty:
        print("无符合全部 5 个买入信号的股票")
    else:
        print(out)
        out.to_csv("output/buy_signals.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
