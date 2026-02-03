import os
import time
from datetime import date, datetime, timedelta

import akshare as ak
import pandas as pd
import requests
import tushare as ts
from dotenv import load_dotenv


GOLD_STOCKS = [
    "603013",
    "000700",
    "300394",
    "002475",
    "300751",
    "001330",
    "301301",
    "002714",
    "002258",
    "002738",
]


class Config:
    def __init__(self):
        self.env_path = ".env"
        self._ensure_env_file()
        load_dotenv(self.env_path)
        self.disable_proxy = self._get_bool_env("DISABLE_PROXY", False)
        if self.disable_proxy:
            self._disable_proxy_env()
        self.serverchan_key = os.getenv("SERVERCHAN_KEY", "").strip()
        self.spot_source = os.getenv("SPOT_SOURCE", "auto").strip().lower()
        self.max_abs_pct = self._get_float_env("MAX_ABS_PCT", 9.5)
        self.test_mode = self._get_bool_env("TEST_MODE", False)
        self.test_once = self._get_bool_env("TEST_ONCE", False)
        self.use_snapshot_only = self._get_bool_env("USE_SNAPSHOT_ONLY", False)
        self.tushare_token = os.getenv("TUSHARE_TOKEN", "").strip()
        if not self.serverchan_key:
            raise ValueError("SERVERCHAN_KEY not set in .env")

    def _ensure_env_file(self):
        if os.path.exists(self.env_path):
            return
        with open(self.env_path, "w", encoding="utf-8") as file:
            file.write("SERVERCHAN_KEY=\n")
            file.write("DISABLE_PROXY=false\n")
            file.write("SPOT_SOURCE=auto\n")
            file.write("TEST_ONCE=false\n")
            file.write("USE_SNAPSHOT_ONLY=false\n")
            file.write("TUSHARE_TOKEN=\n")

    @staticmethod
    def _get_float_env(key, default):
        raw = os.getenv(key)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    @staticmethod
    def _get_bool_env(key, default=False):
        raw = os.getenv(key)
        if raw is None or raw == "":
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

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


class DataFetcher:
    def __init__(self, pause_seconds=1.0, spot_source="auto"):
        self.pause_seconds = pause_seconds
        self.spot_source = spot_source

    def _sleep(self):
        time.sleep(self.pause_seconds)

    def _call_with_retry(self, func, desc, retries=3, backoff=2):
        for attempt in range(1, retries + 1):
            try:
                return func()
            except Exception as exc:
                print(f"[WARN] {desc} 失败(第{attempt}次): {exc}")
                if attempt < retries:
                    time.sleep(backoff * attempt)
        return None

    def get_spot(self):
        if self.spot_source == "off":
            return pd.DataFrame()
        if self.spot_source == "sina":
            data = self._call_with_retry(ak.stock_zh_a_spot, "获取实时行情(新浪)")
        elif self.spot_source == "em":
            data = self._call_with_retry(ak.stock_zh_a_spot_em, "获取实时行情(东财)")
        else:
            data = self._call_with_retry(ak.stock_zh_a_spot_em, "获取实时行情(东财)")
            if not isinstance(data, pd.DataFrame) or data.empty:
                data = self._call_with_retry(ak.stock_zh_a_spot, "获取实时行情(新浪)")
        self._sleep()
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame()

    def get_minute_k(self, symbol, period="5"):
        data = self._call_with_retry(
            lambda: ak.stock_zh_a_hist_min_em(symbol=symbol, period=period),
            f"获取分钟K线({symbol})",
        )
        self._sleep()
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame()

    def get_daily_last_tx(self, symbol_with_market, start_date, end_date):
        data = self._call_with_retry(
            lambda: ak.stock_zh_a_hist_tx(
                symbol=symbol_with_market, start_date=start_date, end_date=end_date
            ),
            f"获取日线数据({symbol_with_market})",
        )
        self._sleep()
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame()

    def get_rank_forecast(self, date_str):
        data = self._call_with_retry(
            lambda: ak.stock_rank_forecast_cninfo(date=date_str),
            f"获取评级预测({date_str})",
        )
        self._sleep()
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame()

    def get_latest_rank_forecast(self, lookback_days=30):
        today = datetime.now().date()
        for i in range(lookback_days + 1):
            d = today - pd.Timedelta(days=i)
            date_str = d.strftime("%Y%m%d")
            df = self.get_rank_forecast(date_str)
            if not df.empty:
                return df, date_str
        return pd.DataFrame(), None

    def get_fund_flow_latest(self, code):
        market = "sh" if code.startswith("6") else "sz"
        if code.startswith("8"):
            market = "bj"
        data = self._call_with_retry(
            lambda: ak.stock_individual_fund_flow(stock=code, market=market),
            f"获取主力资金流({code})",
        )
        self._sleep()
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame()


class Strategy:
    def __init__(self, fetcher):
        self.fetcher = fetcher
        self.daily_sent = set()
        self.last_reset_date = date.today()
        self.target_space = {}
        self.low_priority = set()

    @staticmethod
    def normalize_code(code):
        if code.startswith(("sh", "sz")):
            return code
        if code.startswith("6"):
            return f"sh{code}"
        return f"sz{code}"

    @staticmethod
    def pick_column(frame, candidates):
        for name in candidates:
            if name in frame.columns:
                return name
        return None

    def initialize_targets(self, spot_df):
        target_df, target_date = self.fetcher.get_latest_rank_forecast()
        if target_df.empty or spot_df.empty:
            return
        code_col = self.pick_column(target_df, ["证券代码", "股票代码", "代码"])
        low_col = self.pick_column(
            target_df,
            ["目标价-下限", "目标价下限", "目标价(下限)", "目标价下边界", "目标价"],
        )
        high_col = self.pick_column(
            target_df,
            ["目标价-上限", "目标价上限", "目标价(上限)", "目标价上边界"],
        )
        if not code_col or (not low_col and not high_col):
            print("[WARN] 目标价列或代码列未识别，跳过目标价空间计算")
            return
        spot_code_col = self.pick_column(spot_df, ["代码", "股票代码", "证券代码"])
        price_col = self.pick_column(spot_df, ["最新价", "现价", "收盘"])
        if not spot_code_col or not price_col:
            return
        spot_map = (
            spot_df[[spot_code_col, price_col]]
            .dropna()
            .set_index(spot_code_col)[price_col]
            .to_dict()
        )
        for code in GOLD_STOCKS:
            target_row = target_df[target_df[code_col] == code]
            if target_row.empty:
                continue
            current_price = spot_map.get(code)
            row = target_row.iloc[0]
            low_value = row[low_col] if low_col else None
            high_value = row[high_col] if high_col else None
            low_ok = low_value is not None and pd.notna(low_value)
            high_ok = high_value is not None and pd.notna(high_value)
            if low_ok and high_ok:
                target_price = (float(low_value) + float(high_value)) / 2
            else:
                target_price = low_value if low_ok else high_value if high_ok else None
            if not current_price or not target_price:
                continue
            try:
                space = (float(target_price) - float(current_price)) / float(current_price)
            except Exception:
                continue
            self.target_space[code] = space
            if space < 0.1:
                self.low_priority.add(code)
        if target_date:
            print(f"[INFO] 评级预测日期: {target_date}")

    def reset_daily(self):
        today = date.today()
        if today != self.last_reset_date:
            self.daily_sent.clear()
            self.last_reset_date = today

    @staticmethod
    def calc_rsi(close, window=6):
        diff = close.diff()
        gain = diff.clip(lower=0)
        loss = -diff.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calc_kdj(high, low, close, window=9):
        lowest = low.rolling(window).min()
        highest = high.rolling(window).max()
        rsv = (close - lowest) / (highest - lowest) * 100
        k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
        d = k.ewm(alpha=1 / 3, adjust=False).mean()
        j = 3 * k - 2 * d
        return k, d, j

    def check_signal(self, code, spot_df, fund_net_in):
        normalized = self.normalize_code(code)
        spot_code_col = self.pick_column(spot_df, ["代码", "股票代码", "证券代码"])
        name_col = self.pick_column(spot_df, ["名称", "股票简称", "证券简称"])
        price_col = self.pick_column(spot_df, ["最新价", "现价", "收盘"])
        pct_col = self.pick_column(spot_df, ["涨跌幅", "涨跌幅(%)", "涨跌幅%"])
        vol_ratio_col = self.pick_column(spot_df, ["量比"])
        if not spot_code_col or not price_col:
            return None
        row = spot_df[spot_df[spot_code_col] == code]
        if row.empty:
            return None
        row = row.iloc[0]
        name = row[name_col] if name_col else code
        current_price = row[price_col]
        pct_chg = row[pct_col] if pct_col else None
        volume_ratio = row[vol_ratio_col] if vol_ratio_col else None

        signals = []
        if volume_ratio is not None and pd.notna(volume_ratio) and float(volume_ratio) > 1.8:
            signals.append("量比>1.8")

        min_df = self.fetcher.get_minute_k(normalized, period="5")
        if not min_df.empty:
            high_col = self.pick_column(min_df, ["最高", "high"])
            low_col = self.pick_column(min_df, ["最低", "low"])
            close_col = self.pick_column(min_df, ["收盘", "close"])
            vol_col = self.pick_column(min_df, ["成交量", "volume"])
            amount_col = self.pick_column(min_df, ["成交额", "amount"])
            if close_col and vol_col:
                close_series = pd.to_numeric(min_df[close_col], errors="coerce")
                vol_series = pd.to_numeric(min_df[vol_col], errors="coerce")
                if amount_col:
                    amount_series = pd.to_numeric(min_df[amount_col], errors="coerce")
                else:
                    amount_series = close_series * vol_series
                cumulative_amount = amount_series.cumsum()
                cumulative_volume = vol_series.cumsum()
                vwap_series = cumulative_amount / cumulative_volume.replace(0, pd.NA)
                vwap_value = vwap_series.iloc[-1]
                if pd.notna(vwap_value) and current_price is not None:
                    if float(current_price) > float(vwap_value):
                        signals.append("VWAP支撑")
                rsi = self.calc_rsi(close_series, window=6)
                if pd.notna(rsi.iloc[-1]) and rsi.iloc[-1] < 30:
                    signals.append("RSI(6)<30")
                if high_col and low_col:
                    high_series = pd.to_numeric(min_df[high_col], errors="coerce")
                    low_series = pd.to_numeric(min_df[low_col], errors="coerce")
                    k, d, j = self.calc_kdj(high_series, low_series, close_series)
                    if len(j) >= 2 and pd.notna(j.iloc[-1]) and pd.notna(k.iloc[-1]):
                        if j.iloc[-1] > k.iloc[-1] and j.iloc[-2] <= k.iloc[-2]:
                            signals.append("KDJ金叉")

        if fund_net_in is not None and pd.notna(fund_net_in) and float(fund_net_in) > 0:
            signals.append("主力净流入>0")

        return {
            "code": code,
            "name": name,
            "price": current_price,
            "pct": pct_chg,
            "signals": signals,
            "triggered": len(signals) >= 3,
        }


class IndicatorEvaluator:
    def __init__(self, tushare_token):
        self.tushare_token = tushare_token
        self.pro = None
        self.cache = {}

    def init_client(self):
        if not self.tushare_token:
            return
        ts.set_token(self.tushare_token)
        self.pro = ts.pro_api()

    @staticmethod
    def to_ts_code(code):
        return f"{code}.SH" if code.startswith("6") else f"{code}.SZ"

    def get_daily_history(self, code):
        today = datetime.now().strftime("%Y%m%d")
        cache_key = (code, today)
        if cache_key in self.cache:
            return self.cache[cache_key]
        if not self.pro:
            return pd.DataFrame()
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
        df = self.pro.daily(
            ts_code=self.to_ts_code(code),
            start_date=start_date,
            end_date=today,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.sort_values("trade_date")
        self.cache[cache_key] = df
        return df

    @staticmethod
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

    @staticmethod
    def evaluate_rules(latest_close, indicators):
        buy = {
            "MA": indicators["ma5"] > indicators["ma10"] > indicators["ma20"]
            and latest_close > indicators["ma20"],
            "MACD": indicators["dif"] > indicators["dea"] and indicators["hist"] > 0,
            "RSI": indicators["rsi6"] > 50 and indicators["rsi6"] > indicators["rsi12"],
            "KDJ": indicators["k"] > indicators["d"] and indicators["k"] < 80,
            "BOLL": latest_close > indicators["ma20"],
        }
        sell = {
            "MA": indicators["ma5"] < indicators["ma10"] and latest_close < indicators["ma5"],
            "MACD": indicators["dif"] < indicators["dea"] and indicators["hist"] < 0,
            "RSI": indicators["rsi6"] < 50 and indicators["rsi6"] < indicators["rsi12"],
            "KDJ": indicators["k"] < indicators["d"] or indicators["k"] > 80,
            "BOLL": latest_close < indicators["ma20"],
        }
        return buy, sell

    def build_today_series(self, df, latest_price, open_today, high_today, low_today):
        today = datetime.now().strftime("%Y%m%d")
        df = df.copy()
        if "trade_date" in df.columns and not df.empty:
            df = df[df["trade_date"] != today]
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
        return df


class RiskControl:
    def __init__(self, max_abs_pct=9.5, min_price=None):
        self.max_abs_pct = max_abs_pct
        self.min_price = min_price

    @staticmethod
    def is_trading_time(now=None):
        now = now or datetime.now()
        if now.weekday() >= 5:
            return False
        t = now.strftime("%H:%M:%S")
        return ("09:30:00" <= t <= "11:30:00") or ("13:00:00" <= t <= "15:00:00")

    def validate(self, signal):
        reasons = []
        price = signal.get("price")
        pct = signal.get("pct")
        if price is None or pd.isna(price) or float(price) <= 0:
            reasons.append("价格无效")
        if self.min_price is not None and price is not None:
            if float(price) < self.min_price:
                reasons.append(f"价格低于{self.min_price}")
        if (
            self.max_abs_pct
            and pct is not None
            and pd.notna(pct)
            and abs(float(pct)) >= self.max_abs_pct
        ):
            reasons.append(f"涨跌幅超{self.max_abs_pct}%")
        return reasons


def send_serverchan(key, title, content):
    url = f"https://sctapi.ftqq.com/{key}.send"
    data = {"title": title, "desp": content}
    try:
        response = requests.post(url, data=data, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def format_space(space):
    if space is None:
        return "N/A"
    return f"{space * 100:.2f}%"


def main():
    config = Config()
    fetcher = DataFetcher(pause_seconds=1.2, spot_source=config.spot_source)
    strategy = Strategy(fetcher)
    risk = RiskControl(max_abs_pct=config.max_abs_pct, min_price=None)
    evaluator = IndicatorEvaluator(config.tushare_token)
    evaluator.init_client()

    print(f"[INFO] 启动成功，TEST_MODE={config.test_mode}")
    proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]
    active_proxies = {k: os.environ.get(k) for k in proxy_keys if os.environ.get(k)}
    print(f"[INFO] 代理状态: {'已禁用' if config.disable_proxy else '未禁用'}")
    if active_proxies and not config.disable_proxy:
        print(f"[WARN] 检测到系统代理环境变量: {', '.join(active_proxies.keys())}")

    spot_df = fetcher.get_spot()
    strategy.initialize_targets(spot_df)

    while True:
        start_time = time.time()
        strategy.reset_daily()

        if not config.test_mode and not risk.is_trading_time():
            print("[INFO] 非交易时间，等待中...")
            time.sleep(60)
            continue

        spot_df = pd.DataFrame() if config.use_snapshot_only else fetcher.get_spot()
        if spot_df.empty and config.test_mode:
            print("[WARN] 实时行情为空，尝试使用日线快照模拟...")
            snapshot_rows = []
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - pd.Timedelta(days=30)).strftime("%Y%m%d")
            for code in GOLD_STOCKS:
                market_code = f"sh{code}" if code.startswith("6") else f"sz{code}"
                daily = fetcher.get_daily_last_tx(market_code, start_date, end_date)
                if daily.empty:
                    continue
                last_row = daily.iloc[-1]
                snapshot_rows.append(
                    {
                        "代码": code,
                        "名称": code,
                        "最新价": last_row.get("close"),
                        "涨跌幅": pd.NA,
                        "成交量": last_row.get("amount"),
                        "成交额": pd.NA,
                        "最高": last_row.get("high"),
                        "最低": last_row.get("low"),
                        "今开": last_row.get("open"),
                        "昨收": pd.NA,
                        "量比": pd.NA,
                    }
                )
            spot_df = pd.DataFrame(snapshot_rows)
            if spot_df.empty:
                print("[WARN] 日线快照也为空，请检查网络或数据源")

        if spot_df.empty:
            print("[WARN] 实时行情为空，稍后重试...")
            time.sleep(120)
            continue

        for code in GOLD_STOCKS:
            if code in strategy.daily_sent:
                continue

            fund_df = fetcher.get_fund_flow_latest(code)
            fund_net_in = None
            if not fund_df.empty:
                net_col = strategy.pick_column(
                    fund_df,
                    ["主力净流入-净额", "主力净流入", "净流入", "今日主力净流入-净额"],
                )
                if net_col:
                    fund_net_in = pd.to_numeric(fund_df.iloc[-1][net_col], errors="coerce")

            signal = strategy.check_signal(code, spot_df, fund_net_in)
            if not signal:
                continue

            signals_text = "无" if not signal["signals"] else ", ".join(signal["signals"])
            pct_text = "N/A" if signal["pct"] is None else f"{float(signal['pct']):.2f}%"
            priority_text = "低优先级" if code in strategy.low_priority else "正常"
            risk_reasons = risk.validate(signal)
            risk_text = "通过" if not risk_reasons else f"过滤:{'、'.join(risk_reasons)}"
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{signal['name']}({signal['code']}) "
                f"现价 {float(signal['price']):.2f} "
                f"涨跌幅 {pct_text} "
                f"信号 {len(signal['signals'])}/4 [{signals_text}] "
                f"{'触发' if signal['triggered'] else '未触发'} "
                f"{priority_text} "
                f"{risk_text}"
            )

            if evaluator.pro:
                daily = evaluator.get_daily_history(code)
                if not daily.empty:
                    spot_code_col = strategy.pick_column(
                        spot_df, ["代码", "股票代码", "证券代码"]
                    )
                    high_col = strategy.pick_column(spot_df, ["最高", "high"])
                    low_col = strategy.pick_column(spot_df, ["最低", "low"])
                    open_col = strategy.pick_column(spot_df, ["今开", "open"])
                    if not spot_code_col:
                        continue
                    spot_row = spot_df[spot_df[spot_code_col] == code].iloc[0]
                    high_today = (
                        float(spot_row[high_col]) if high_col and pd.notna(spot_row[high_col]) else float(signal["price"])
                    )
                    low_today = (
                        float(spot_row[low_col]) if low_col and pd.notna(spot_row[low_col]) else float(signal["price"])
                    )
                    open_today = (
                        float(spot_row[open_col]) if open_col and pd.notna(spot_row[open_col]) else float(signal["price"])
                    )
                    series = evaluator.build_today_series(
                        daily, float(signal["price"]), open_today, high_today, low_today
                    )
                    indicators = evaluator.compute_indicators(series)
                    buy_rules, sell_rules = evaluator.evaluate_rules(
                        float(signal["price"]), indicators
                    )
                    buy_hit = [k for k, v in buy_rules.items() if v]
                    sell_hit = [k for k, v in sell_rules.items() if v]
                    print(
                        f"买入信号(5条)满足：{len(buy_hit)}条 -> {', '.join(buy_hit) if buy_hit else '无'}"
                    )
                    print(
                        f"卖出信号(5条)满足：{len(sell_hit)}条 -> {', '.join(sell_hit) if sell_hit else '无'}"
                    )

            if not signal["triggered"] or risk_reasons:
                continue
            space = strategy.target_space.get(code)
            title = f"【金股信号】{signal['name']}({signal['code']}) 触发买入提示"
            content = "\n".join(
                [
                    f"当前价格: {float(signal['price']):.2f}",
                    f"涨跌幅: {pct_text}",
                    f"触发因子: {', '.join(signal['signals'])}",
                    f"机构目标空间: {format_space(space)}",
                    f"优先级: {priority_text}",
                    f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ]
            )
            if send_serverchan(config.serverchan_key, title, content):
                strategy.daily_sent.add(code)

        elapsed = time.time() - start_time
        sleep_seconds = max(5, 120 - int(elapsed))
        if config.test_once:
            print("[INFO] TEST_ONCE=true，已完成一轮，退出")
            break
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
