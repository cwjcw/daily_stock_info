# -*- coding: utf-8 -*-
"""
Production-grade data maintainer for mystock (MySQL) using Tushare Pro + xtquant.
Python 3.12+
"""

from __future__ import annotations

import datetime
import gc
import os
import sys
import threading
import time
import sqlite3
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from xtquant import xtdata
import tushare as ts


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
try:
    from config import CONFIG
except Exception:
    CONFIG = {}


def load_env() -> None:
    load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))


def cfg_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is not None and raw != "":
        try:
            return int(raw)
        except Exception:
            return default
    try:
        return int(CONFIG.get(key, default))
    except Exception:
        return default


def build_engine() -> Engine:
    host = os.getenv("MYSQL_HOST")
    port = os.getenv("MYSQL_PORT")
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    db = os.getenv("MYSQL_DB")
    charset = os.getenv("MYSQL_CHARSET", "utf8mb4")
    if not all([host, port, user, password, db]):
        raise RuntimeError("Missing MySQL env vars. Please check .env.")
    uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db}?charset={charset}"
    return create_engine(
        uri,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )


def get_tushare_pro() -> ts.pro_api:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing TUSHARE_TOKEN in .env.")
    ts.set_token(token)
    return ts.pro_api()


class RateLimiter:
    def __init__(self, max_calls: int, period_seconds: int) -> None:
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.lock = threading.Lock()
        self.calls: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.time()
                while self.calls and now - self.calls[0] > self.period_seconds:
                    self.calls.popleft()
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return
                wait_for = self.period_seconds - (now - self.calls[0]) + 0.01
            time.sleep(max(wait_for, 0.05))


TS_RATE_LIMITER = RateLimiter(max_calls=500, period_seconds=60)
TS_SEMAPHORE = threading.Semaphore(8)
TABLE_LOCK = threading.Lock()
DISABLED_TUSHARE_APIS: set[str] = set()
TABLE_PREFIX = "S_"


def tn(name: str) -> str:
    return f"{TABLE_PREFIX}{name}"


def call_tushare(api_func, api_name: Optional[str] = None, **params) -> Optional[pd.DataFrame]:
    if api_name and api_name in DISABLED_TUSHARE_APIS:
        return None
    TS_RATE_LIMITER.acquire()
    with TS_SEMAPHORE:
        try:
            df = api_func(**params)
        except Exception as exc:
            msg = str(exc)
            if "请指定正确的接口名" in msg or "接口" in msg and "不存在" in msg:
                if api_name:
                    DISABLED_TUSHARE_APIS.add(api_name)
                    print(f"[Tushare] API not supported: {api_name} ({msg})")
                    return None
            print(f"[Tushare] error: {exc}")
            return None
        finally:
            time.sleep(0.15)
    if df is None or df.empty:
        return None
    return df


def _safe_api(pro: ts.pro_api, name: str):
    if name in DISABLED_TUSHARE_APIS:
        return None
    return getattr(pro, name, None)


def infer_sql_type(series: pd.Series, col_name: str, is_pk: bool = False, is_index: bool = False) -> Any:
    if col_name == "ts_code":
        return String(16)
    if col_name == "trade_date":
        return String(8)
    if col_name == "trade_time":
        return String(14)
    if pd.api.types.is_integer_dtype(series):
        return BigInteger()
    if pd.api.types.is_float_dtype(series):
        return Float()
    if pd.api.types.is_bool_dtype(series):
        return Integer()
    if pd.api.types.is_datetime64_any_dtype(series):
        return DateTime()
    if is_pk or is_index:
        return String(64)
    if pd.api.types.is_object_dtype(series):
        max_len = series.dropna().astype(str).str.len().max()
        if pd.notna(max_len) and max_len <= 128:
            return String(128)
    return Text()


def ensure_table(engine: Engine, table_name: str, df: pd.DataFrame, pk_cols: List[str], index_cols: List[str]) -> None:
    if df is None or df.empty:
        return
    inspector = inspect(engine)
    with TABLE_LOCK:
        if not inspector.has_table(table_name):
            metadata = MetaData()
            columns = []
            for col in df.columns:
                col_type = infer_sql_type(
                    df[col],
                    col,
                    is_pk=col in pk_cols,
                    is_index=col in index_cols,
                )
                columns.append(Column(col, col_type, primary_key=(col in pk_cols)))
            table = Table(table_name, metadata, *columns, mysql_charset="utf8mb4")
            for idx_col in index_cols:
                Index(f"idx_{table_name}_{idx_col}", table.c[idx_col])
            metadata.create_all(engine)
        else:
            existing = {col["name"] for col in inspector.get_columns(table_name)}
            missing = [c for c in df.columns if c not in existing]
            if missing:
                for col in missing:
                    col_type = infer_sql_type(
                        df[col],
                        col,
                        is_pk=col in pk_cols,
                        is_index=col in index_cols,
                    )
                    ddl = f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type.compile(engine.dialect)}"
                    with engine.begin() as conn:
                        conn.execute(text(ddl))


def upsert_dataframe(engine: Engine, table_name: str, df: pd.DataFrame, pk_cols: List[str], chunk_size: int = 1000) -> None:
    if df is None or df.empty:
        return
    df = df.replace({np.nan: None})
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)
    records = df.to_dict("records")
    with engine.begin() as conn:
        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            stmt = mysql_insert(table).values(chunk)
            update_cols = {
                c.name: stmt.inserted[c.name]
                for c in table.columns
                if c.name not in pk_cols
            }
            stmt = stmt.on_duplicate_key_update(**update_cols)
            conn.execute(stmt)


def fetch_trade_cal(pro: ts.pro_api, start_date: str, end_date: str) -> List[str]:
    df = call_tushare(pro.trade_cal, api_name="trade_cal", exchange="SSE", start_date=start_date, end_date=end_date, is_open=1)
    if df is None or df.empty:
        return []
    dates = df["cal_date"].tolist()
    dates = sorted(set(dates))
    return dates


def resolve_effective_today(pro: ts.pro_api, today: str) -> str:
    end_dt = datetime.datetime.strptime(today, "%Y%m%d")
    start_dt = end_dt - datetime.timedelta(days=365 * 5)
    dates = fetch_trade_cal(pro, start_dt.strftime("%Y%m%d"), today)
    if not dates:
        return today
    return dates[-1]


def get_trade_dates_ago(pro: ts.pro_api, end_date: str, n: int) -> List[str]:
    end_dt = datetime.datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - datetime.timedelta(days=int(n * 2.5))
    dates = fetch_trade_cal(pro, start_dt.strftime("%Y%m%d"), end_date)
    if len(dates) <= n:
        return dates
    return dates[-n:]


def get_max_value(engine: Engine, table: str, col: str) -> Optional[str]:
    inspector = inspect(engine)
    if not inspector.has_table(table):
        return None
    with engine.begin() as conn:
        result = conn.execute(text(f"SELECT MAX({col}) FROM {table}"))
        row = result.fetchone()
        return row[0] if row and row[0] is not None else None


def get_existing_dates(engine: Engine, table: str, col: str, start: str, end: str) -> set[str]:
    inspector = inspect(engine)
    if not inspector.has_table(table):
        return set()
    sql = text(f"SELECT DISTINCT {col} AS d FROM {table} WHERE {col} >= :start AND {col} <= :end")
    with engine.begin() as conn:
        rows = conn.execute(sql, {"start": start, "end": end}).fetchall()
    return {str(r[0]) for r in rows if r and r[0] is not None}


def summarize_missing(label: str, window_dates: List[str], missing_dates: List[str]) -> None:
    if not window_dates:
        print(f"[{label}] window empty")
        return
    if not missing_dates:
        print(f"[{label}] missing 0/{len(window_dates)} dates")
        return
    print(
        f"[{label}] missing {len(missing_dates)}/{len(window_dates)} dates, "
        f"range {missing_dates[0]} ~ {missing_dates[-1]}"
    )


def build_missing_ranges(window_dates: List[str], missing_dates: List[str]) -> List[Tuple[str, str]]:
    if not missing_dates:
        return []
    missing_set = set(missing_dates)
    ranges: List[Tuple[str, str]] = []
    start = None
    prev = None
    for d in window_dates:
        if d in missing_set:
            if start is None:
                start = d
            prev = d
        else:
            if start is not None and prev is not None:
                ranges.append((start, prev))
                start = None
                prev = None
    if start is not None and prev is not None:
        ranges.append((start, prev))
    return ranges


def fetch_tushare_paged(api_func, limit: int = 6000, api_name: Optional[str] = None, **params) -> Optional[pd.DataFrame]:
    offset = 0
    frames = []
    while True:
        df = call_tushare(api_func, api_name=api_name, **params, offset=offset, limit=limit)
        if df is None or df.empty:
            break
        frames.append(df)
        if len(df) < limit:
            break
        offset += limit
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def merge_daily_with_basic(daily: pd.DataFrame, basic: Optional[pd.DataFrame]) -> pd.DataFrame:
    if basic is None or basic.empty:
        return daily
    drop_cols = [c for c in basic.columns if c in daily.columns and c not in ("ts_code", "trade_date")]
    basic = basic.drop(columns=drop_cols)
    return pd.merge(daily, basic, on=["ts_code", "trade_date"], how="left")


def update_daily_and_adj(
    pro: ts.pro_api,
    engine: Engine,
    trade_dates: List[str],
) -> None:
    total = len(trade_dates)
    done = 0
    lock = threading.Lock()

    def _tick(date_str: str) -> None:
        nonlocal done
        with lock:
            done += 1
            if done == 1 or done % 10 == 0 or done == total:
                print(f"[daily_raw+adj_factor] {done}/{total} date={date_str}")

    def worker(date_str: str) -> None:
        try:
            daily = fetch_tushare_paged(pro.daily, api_name="daily", trade_date=date_str)
            if daily is None or daily.empty:
                _tick(date_str)
                return
            daily = normalize_daily_units(daily)
            basic = fetch_tushare_paged(pro.daily_basic, api_name="daily_basic", trade_date=date_str)
            merged = merge_daily_with_basic(daily, basic)
            ensure_table(engine, tn("daily_raw"), merged, ["ts_code", "trade_date"], ["trade_date"])
            upsert_dataframe(engine, tn("daily_raw"), merged, ["ts_code", "trade_date"])

            adj = fetch_tushare_paged(pro.adj_factor, api_name="adj_factor", trade_date=date_str)
            if adj is not None and not adj.empty:
                ensure_table(engine, tn("adj_factor"), adj, ["ts_code", "trade_date"], ["trade_date"])
                upsert_dataframe(engine, tn("adj_factor"), adj, ["ts_code", "trade_date"])
        except Exception as exc:
            print(f"[daily/adj] {date_str} error: {exc}")
        finally:
            _tick(date_str)

    with ThreadPoolExecutor(max_workers=6) as executor:
        executor.map(worker, trade_dates)


def update_moneyflow(
    pro: ts.pro_api,
    engine: Engine,
    trade_dates: List[str],
    missing_by_table: Optional[Dict[str, set[str]]] = None,
) -> None:
    if not trade_dates:
        return
    api_map = {
        "moneyflow_ind": "moneyflow_dc",
        "moneyflow_sector": "moneyflow_ind_dc",
        "moneyflow_mkt": "moneyflow_mkt_dc",
        "moneyflow_hsgt": "moneyflow_hsgt",
    }
    sector_types = ["行业", "概念", "地域"]

    total = len(trade_dates)
    done = 0
    lock = threading.Lock()

    def _tick(date_str: str) -> None:
        nonlocal done
        with lock:
            done += 1
            if done == 1 or done % 10 == 0 or done == total:
                print(f"[moneyflow] {done}/{total} date={date_str}")

    def worker(date_str: str) -> None:
        for table_name, api_name in api_map.items():
            api_func = _safe_api(pro, api_name)
            if api_func is None:
                print(f"[moneyflow] api not found: {api_name}")
                continue
            if missing_by_table and date_str not in missing_by_table.get(table_name, set()):
                continue
            try:
                if table_name == "moneyflow_sector":
                    for ct in sector_types:
                        df = fetch_tushare_paged(
                            api_func,
                            api_name=api_name,
                            trade_date=date_str,
                            content_type=ct,
                        )
                        if df is None or df.empty:
                            continue
                        pk = ["ts_code", "trade_date", "content_type"]
                        ensure_table(engine, tn(table_name), df, pk, ["trade_date"])
                        upsert_dataframe(engine, tn(table_name), df, pk)
                else:
                    df = fetch_tushare_paged(api_func, api_name=api_name, trade_date=date_str)
                    if df is None or df.empty:
                        continue
                    pk = ["trade_date"]
                    if "ts_code" in df.columns:
                        pk = ["ts_code", "trade_date"]
                    ensure_table(engine, tn(table_name), df, pk, ["trade_date"])
                    upsert_dataframe(engine, tn(table_name), df, pk)
            except Exception as exc:
                print(f"[moneyflow] {table_name} {date_str} error: {exc}")
        _tick(date_str)

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(worker, trade_dates)


def parse_time_value(val: Any) -> Optional[pd.Timestamp]:
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


def normalize_minute_df(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    df = df.copy()
    if "time" in df.columns:
        dt = df["time"].apply(parse_time_value)
    else:
        dt = df.index.to_series().apply(parse_time_value)
    df = df[dt.notna()].copy()
    dt = dt[dt.notna()]
    time_str = dt.dt.strftime("%Y%m%d%H%M%S")
    df["time"] = time_str
    df["trade_time"] = dt.dt.strftime("%Y%m%d%H%M%S")
    df["trade_date"] = dt.dt.strftime("%Y%m%d")
    df["ts_code"] = ts_code
    if "volume" in df.columns:
        df["vol"] = pd.to_numeric(df["volume"], errors="coerce")
    elif "vol" in df.columns:
        df["vol"] = pd.to_numeric(df["vol"], errors="coerce")
    return df


def normalize_daily_units(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "amount" in df.columns:
        amt = pd.to_numeric(df["amount"], errors="coerce")
        df["amount"] = amt * 1000.0
    return df


def init_realtime_sqlite(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS S_realtime_quote (
                ts_code TEXT NOT NULL,
                time TEXT NOT NULL,
                price REAL,
                open REAL,
                high REAL,
                low REAL,
                pre_close REAL,
                vol REAL,
                amount REAL,
                source TEXT,
                raw_json TEXT,
                PRIMARY KEY (ts_code, time)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_S_realtime_quote_time ON S_realtime_quote(time)")
        conn.commit()
    finally:
        conn.close()


def update_minute_raw(
    engine: Engine,
    date_ranges: List[Tuple[str, str]],
    chunk_size: int = 100,
    workers: int = 8,
) -> None:
    scan_sector = os.getenv("SCAN_SECTOR", "沪深A股")
    stock_list = xtdata.get_stock_list_in_sector(scan_sector)
    filtered_list = [s for s in stock_list if not s.startswith(("688", "8", "4"))]

    if not date_ranges:
        print("[minute_raw] missing 0 dates, skip")
        return

    chunks = [filtered_list[i : i + chunk_size] for i in range(0, len(filtered_list), chunk_size)]

    def _download_minute_history(codes: List[str], start_date: str, end_date: str) -> None:
        if hasattr(xtdata, "download_history_data2"):
            xtdata.download_history_data2(codes, period="1m", start_time=start_date, end_time=end_date)
        else:
            for code in codes:
                xtdata.download_history_data(code, period="1m", start_time=start_date, end_time=end_date)

    def worker(codes: List[str], start_date: str, end_date: str) -> None:
        try:
            _download_minute_history(codes, start_date, end_date)
            data = xtdata.get_market_data_ex(
                [], codes, period="1m", start_time=start_date, end_time=end_date, dividend_type="none"
            )
            for code, df in data.items():
                if df is None or df.empty:
                    continue
                df_norm = normalize_minute_df(df, code)
                ensure_table(engine, tn("minute_raw"), df_norm, ["ts_code", "trade_time"], ["trade_date"])
                upsert_dataframe(engine, tn("minute_raw"), df_norm, ["ts_code", "trade_time"])
                del df_norm
            del data
            gc.collect()
        except Exception as exc:
            print(f"[minute_raw] chunk error: {exc}")

    for idx, (start_date, end_date) in enumerate(date_ranges, start=1):
        print(f"[minute_raw] start range {idx}/{len(date_ranges)} {start_date}~{end_date}")
        total = len(chunks)
        done = 0
        lock = threading.Lock()

        def _tick() -> None:
            nonlocal done
            with lock:
                done += 1
                if done == 1 or done % 5 == 0 or done == total:
                    print(
                        f"[minute_raw] range {idx}/{len(date_ranges)} {start_date}~{end_date} "
                        f"chunk {done}/{total}"
                    )

        def _worker_wrapper(codes: List[str]) -> None:
            try:
                worker(codes, start_date, end_date)
            finally:
                _tick()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            executor.map(_worker_wrapper, chunks)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        executor.map(worker, chunks)


def get_qfq_data(
    engine: Engine,
    ts_code: str,
    start: str,
    end: str,
    freq: str = "1d",
) -> pd.DataFrame:
    if freq not in ("1d", "1m"):
        raise ValueError("freq must be '1d' or '1m'")

    if freq == "1d":
        sql = text(
            "SELECT * FROM S_daily_raw WHERE ts_code=:ts_code AND trade_date>=:start AND trade_date<=:end"
        )
    else:
        sql = text(
            "SELECT * FROM S_minute_raw WHERE ts_code=:ts_code AND trade_date>=:start AND trade_date<=:end"
        )

    with engine.begin() as conn:
        df = pd.read_sql(sql, conn, params={"ts_code": ts_code, "start": start, "end": end})
        adj = pd.read_sql(
            text("SELECT * FROM S_adj_factor WHERE ts_code=:ts_code AND trade_date>=:start AND trade_date<=:end"),
            conn,
            params={"ts_code": ts_code, "start": start, "end": end},
        )

    if df.empty or adj.empty:
        return df

    if freq == "1m":
        df["trade_date"] = df["trade_time"].astype(str).str[:8]

    merged = pd.merge(df, adj[["trade_date", "adj_factor"]], on="trade_date", how="left")
    merged = merged.sort_values("trade_date")
    latest_factor = merged["adj_factor"].dropna().iloc[-1]
    if latest_factor == 0:
        return merged

    for col in ("open", "high", "low", "close"):
        if col in merged.columns:
            merged[f"{col}_qfq"] = merged[col] * (merged["adj_factor"] / latest_factor)
    return merged


def main() -> None:
    load_env()
    engine = build_engine()
    pro = get_tushare_pro()

    now = datetime.datetime.now()
    today = now.strftime("%Y%m%d")
    effective_today = resolve_effective_today(pro, today)
    if effective_today != today:
        print(f"[calendar] system_date={today} -> effective_trade_date={effective_today}")
    effective_dt = datetime.datetime.strptime(effective_today, "%Y%m%d")
    init_realtime_sqlite(os.path.join(CONFIG.get("CACHE_DIR", "./temp_cache"), "realtime_cache.db"))

    # Daily & adj_factor update
    max_daily = get_max_value(engine, tn("daily_raw"), "trade_date")
    if max_daily:
        trade_dates = fetch_trade_cal(pro, max_daily, effective_today)
        trade_dates = [d for d in trade_dates if d > max_daily]
    else:
        backfill_days = cfg_int("DAILY_BACKFILL_TRADE_DAYS", 500)
        if backfill_days <= 0:
            trade_dates = [effective_today]
        else:
            trade_dates = get_trade_dates_ago(pro, effective_today, backfill_days)
    if trade_dates:
        print(f"[daily_raw+adj_factor] date range: {trade_dates[0]} ~ {trade_dates[-1]} (n={len(trade_dates)})")

        start_window = trade_dates[0]
        end_window = trade_dates[-1]
        existing_daily = get_existing_dates(engine, tn("daily_raw"), "trade_date", start_window, end_window)
        existing_adj = get_existing_dates(engine, tn("adj_factor"), "trade_date", start_window, end_window)
        missing_set = set(trade_dates) - existing_daily
        missing_set |= set(trade_dates) - existing_adj
        missing_dates = [d for d in trade_dates if d in missing_set]
        summarize_missing("daily_raw+adj_factor", trade_dates, missing_dates)
        if missing_dates:
            update_daily_and_adj(pro, engine, missing_dates)

    # Moneyflow update: max(Today-500 trade days, 20230911)
    mf_backfill_days = cfg_int("MONEYFLOW_BACKFILL_TRADE_DAYS", 500)
    trade_dates_500 = get_trade_dates_ago(pro, effective_today, mf_backfill_days)
    start_mf = max(trade_dates_500[0], "20230911") if trade_dates_500 else "20230911"
    mf_dates = fetch_trade_cal(pro, start_mf, effective_today)
    if mf_dates:
        mf_missing: Dict[str, set[str]] = {}
        for table in ["moneyflow_ind", "moneyflow_sector", "moneyflow_mkt", "moneyflow_hsgt"]:
            existing = get_existing_dates(engine, tn(table), "trade_date", mf_dates[0], mf_dates[-1])
            missing = {d for d in mf_dates if d not in existing}
            mf_missing[table] = missing
        total_missing = sum(len(v) for v in mf_missing.values())
        print(f"[moneyflow] missing total dates (by table): {total_missing}")
        update_moneyflow(pro, engine, mf_dates, missing_by_table=mf_missing)

    # Minute raw update: last 100 natural days
    minute_trade_days = cfg_int("MINUTE_BACKFILL_TRADE_DAYS", 5)
    minute_window = get_trade_dates_ago(pro, effective_today, minute_trade_days)
    if minute_window:
        existing_minute = get_existing_dates(engine, tn("minute_raw"), "trade_date", minute_window[0], minute_window[-1])
        missing_minute = [d for d in minute_window if d not in existing_minute]
        summarize_missing("minute_raw", minute_window, missing_minute)
        ranges = build_missing_ranges(minute_window, missing_minute)
    else:
        ranges = []
    xtdata.connect(port=int(os.getenv("QMT_PORT", "58610")))
    update_minute_raw(engine, ranges, chunk_size=100, workers=8)


if __name__ == "__main__":
    main()
